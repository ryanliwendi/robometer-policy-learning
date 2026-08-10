#!/usr/bin/env python3
"""The interactive-imitation DAgger loop.

`rdagger.gate_type` selects which gate decides when to intervene, and everything else is shared.
so a difference in the resulting policy is attributable to the gate alone:

    robometer   Robometer progress + the drop/plateau rule (robometer_gate.py)
    thrifty     ThriftyDAgger: ensemble novelty and Q-risk (thrifty_gate.py)
    hgdagger    a human decides, live or via a replay loop (human_gate.py)
    diffdagger  Diff-DAgger: the student's own diffusion loss + a quantile (baseline_gates.py)

The `rdagger.*` config namespace is shared by every arm.

Starting from a pretrained STUDENT (load_dir) and a frozen EXPERT (rdagger.expert_dir, or pi0),
each iteration:
  1. collects gated rollouts with GatedRolloutWorker until rollouts_per_iter episodes are KEPT.
     Every stored step is labelled intervention=1 (expert correction) or 0 (student). Takeover is
     one-way: once the gate fires the expert drives to the end of the episode;
  2. trains behavior cloning for rdagger.train_steps_per_iter steps on the online buffer,
     optionally mixed with the offline demos (MixedReplayBuffer);
  3. refreshes the gate if it needs it -- Diff-DAgger and ThriftyDAgger recalibrate against the
     retrained policy, Robometer and HG-DAgger have nothing to fit;
  4. evaluates autonomously and checkpoints;
then repeats.

Example usage:
    uv run python scripts/train_reward_dagger.py --config-name libero_rdagger_task1_config

HG-DAgger needs an operator for the whole run: `replay` prompts on a TTY (use `srun --pty`),
`live` serves a browser UI on rdagger.hg_port.
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# A pi0 expert is JAX; Robometer, DINOv2 and the student are torch. JAX preallocates ~75% of the
# GPU on first use, which starves torch and OOMs the reward model. Must precede any jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

from datetime import datetime

import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robometer.utils.logger import get_logger, setup_loguru_logging
from robometer_policy_learning.algorithms.bc import BCConfig
from robometer_policy_learning.algorithms.dp import DPConfig
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.mixed_replay_buffer import MixedReplayBuffer
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.utils.env_utils import make_env
from robometer_policy_learning.utils.hitl_utils_publish import compute_iwr_weights, compute_sirius_weights
from robometer_policy_learning.utils.reward_gate import RewardGate
from robometer_policy_learning.utils.training_utils import save_checkpoint
from robometer_policy_learning.loggers.wandb_logger import WandbLogger

from gated_rollout_worker import GatedRolloutWorker, Pi0Actor, load_actor

logger = get_logger()

LABEL_OFFLINE = 2

ALG_TO_CONFIG = {"bc": BCConfig, "dp": DPConfig}


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = HydraConfig.get().runtime.output_dir
    save_dir = os.path.join(output_dir, "checkpoints")

    setup_loguru_logging(log_level=OmegaConf.select(cfg, "logging.log_level", default="INFO"), output_dir=output_dir)

    string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_logger = WandbLogger(
        exp_name=f"{cfg.logging.wandb_name}_rdagger_{string_time}",
        offline=cfg.logging.wandb_offline,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        log_dir=f"{cfg.logging.wandb_log_dir_base}/{string_time}",
        prefix="offline",
    )

    # ---- Adopt env / training / model / policy from the student's pretraining run ----
    load_dir = OmegaConf.select(cfg, "load_dir", default=None)
    if not load_dir:
        raise ValueError("Set load_dir=<student pretraining run dir>.")
    pre_cfg = OmegaConf.load(os.path.join(load_dir, ".hydra", "config.yaml"))
    OmegaConf.set_struct(cfg, False)  # Disallow adding new fields
    for key in ("env", "training", "model", "policy"):
        if key in pre_cfg:
            cfg[key] = pre_cfg[key]
    OmegaConf.set_struct(cfg, True)
    OmegaConf.resolve(cfg)

    expert_type = str(OmegaConf.select(cfg, "rdagger.expert_type", default="dp"))
    pi0_checkpoint = os.path.expanduser(str(OmegaConf.select(
        cfg, "rdagger.pi0_checkpoint",
        default="~/.cache/openpi/openpi-assets/checkpoints/pi0_libero")))
    expert_dir = OmegaConf.select(cfg, "rdagger.expert_dir", default=None)  # unused for expert_type=pi0
    expert_checkpoint = OmegaConf.select(cfg, "rdagger.expert_checkpoint", default=None)
    reward_model_path = OmegaConf.select(cfg, "rdagger.reward_model", default="jesbu1/robometer-4b-fft-libero")
    expert_n_exec = int(OmegaConf.select(cfg, "rdagger.expert_n_action_steps", default=5))
    score_every = int(OmegaConf.select(cfg, "rdagger.score_every", default=1))
    num_iterations = int(OmegaConf.select(cfg, "rdagger.num_iterations", default=10))
    rollouts_per_iter = int(OmegaConf.select(cfg, "rdagger.rollouts_per_iter", default=5))
    train_steps_per_iter = int(OmegaConf.select(cfg, "rdagger.train_steps_per_iter", default=2000))
    use_offline = bool(OmegaConf.select(cfg, "rdagger.use_offline", default=True))
    offline_ratio = OmegaConf.select(cfg, "rdagger.offline_sample_ratio", default=None)
    store_only_expert = bool(OmegaConf.select(cfg, "rdagger.store_only_expert", default=False))
    require_intervention = bool(OmegaConf.select(cfg, "rdagger.require_intervention", default=False))
    require_success = bool(OmegaConf.select(cfg, "rdagger.require_success", default=False))
    reweighting = OmegaConf.select(cfg, "rdagger.reweighting", default=None)
    # Fraction of total gradient mass IWR assigns to expert corrections (0.5 = canonical IWR).
    # Higher means more gradient is given to optimize expert interventions
    iwr_target_intv = float(OmegaConf.select(cfg, "rdagger.iwr_target_intv", default=0.5))
    save_interval = int(OmegaConf.select(cfg, "rdagger.save_interval", default=1))
    # Baseline comparison arms: only the GATE changes -- same student, expert, loop and recipe.
    gate_type = str(OmegaConf.select(cfg, "rdagger.gate_type", default="robometer"))
    dd_alpha = float(OmegaConf.select(cfg, "rdagger.dd_alpha", default=0.99))
    dd_patience = int(OmegaConf.select(cfg, "rdagger.dd_patience", default=1))
    dd_patience_window = OmegaConf.select(cfg, "rdagger.dd_patience_window", default=None)
    dd_calib_samples = int(OmegaConf.select(cfg, "rdagger.dd_calib_samples", default=256))
    # ThriftyDAgger: alpha_h is the target intervention rate; thresholds are its (1-alpha_h) quantiles.
    thrifty_alpha_h = float(OmegaConf.select(cfg, "rdagger.thrifty_alpha_h", default=0.01))
    thrifty_num_nets = int(OmegaConf.select(cfg, "rdagger.thrifty_num_nets", default=5))
    thrifty_train_steps = int(OmegaConf.select(cfg, "rdagger.thrifty_train_steps", default=200))
    thrifty_lr = float(OmegaConf.select(cfg, "rdagger.thrifty_lr", default=1e-3))
    thrifty_gamma = float(OmegaConf.select(cfg, "rdagger.thrifty_gamma", default=0.9999))
    # HG-DAgger: 'live' needs a display; 'replay' asks the operator after each solo rollout.
    hg_backend = str(OmegaConf.select(cfg, "rdagger.hg_backend", default="live"))
    gate_kwargs = dict(
        method=str(OmegaConf.select(cfg, "rdagger.method", default="spearman")),
        short_window=int(OmegaConf.select(cfg, "rdagger.short_window", default=5)),
        drop_threshold=float(OmegaConf.select(cfg, "rdagger.drop_threshold", default=-0.5)),
        long_window=int(OmegaConf.select(cfg, "rdagger.long_window", default=30)),
        plateau_threshold=float(OmegaConf.select(cfg, "rdagger.plateau_threshold", default=0.3)),
        min_drop_magnitude=float(OmegaConf.select(cfg, "rdagger.min_drop_magnitude", default=0.0)),
        smoothing=float(OmegaConf.select(cfg, "rdagger.smoothing", default=0.0)),
    )

    student = load_actor(load_dir, device, OmegaConf.select(cfg, "checkpoint", default=None), trainable=True)

    if expert_type == "pi0":
        expert = Pi0Actor(pi0_checkpoint, device=device)
    else:
        if not expert_dir:
            raise ValueError("Set rdagger.expert_dir=<expert run dir> for rdagger.expert_type=dp.")
        expert = load_actor(expert_dir, device, expert_checkpoint)

    dino_image_keys = list(OmegaConf.select(cfg, "env.dino_image_keys", default=[]) or [])
    remove_obs_keys = list(getattr(student, "remove_obs_keys", None)
                           or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or [])

    dinov2_model = dinov2_processor = None
    if dino_image_keys:
        from transformers import AutoImageProcessor, AutoModel

        model_id = OmegaConf.select(cfg, "model.dinov2_model", default="facebook/dinov2-base")
        dinov2_model = AutoModel.from_pretrained(model_id).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(model_id)

    chunk_size = OmegaConf.select(cfg, "training.chunk_size", default=None)
    n_exec = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)
    normalize_lowdim = bool(OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False))
    env_name = f"{cfg.env.env_name}/{cfg.env.task_id}"

    # The collection env is unchunked; the worker chunks manually so control can
    # switch student to expert mid-episode and each side replans on takeover.
    collect_env, _ = make_env(
        env_name=env_name,
        num_envs=1,
        max_episode_steps=int(cfg.env.max_episode_steps),
        chunk_size=None,
        n_action_steps=1,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        dino_image_keys=dino_image_keys,
        seed=int(OmegaConf.select(cfg, "rdagger.seed", default=0)),
    )
    action_dim = int(collect_env.single_action_space.shape[0])

    # Action normalization bounds (buffers map stored env-space actions to [-1, 1] at sample time).
    asp = collect_env.single_action_space
    if np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
        action_min, action_max = np.asarray(asp.low, np.float32), np.asarray(asp.high, np.float32)
    else:
        action_min = action_max = None

    if chunk_size is None:
        sampler = RandomSampler()
    else:
        gamma = OmegaConf.select(cfg, "offline_algorithm.gamma", default=0.99)
        sampler = ChunkedSequentialSampler(chunk_size=int(chunk_size), obs_as_sequence=False, gamma=gamma)

    lowdim_stats = {}
    offline_buffer = None
    if use_offline or normalize_lowdim:
        offline_buffer = H5ReplayBuffer(
            h5_paths=[cfg.env.h5_dataset_path],
            sampler=sampler,
            remove_obs_keys=list(remove_obs_keys),
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            dino_embedding_keys=dino_image_keys,
            min_action=action_min,
            max_action=action_max,
            normalize_lowdim_obs=normalize_lowdim,
            default_intervention_label=LABEL_OFFLINE,
        )
        lowdim_stats = offline_buffer.lowdim_obs_stats
        if not use_offline:
            offline_buffer = None

    online_buffer = ReplayBuffer(
        capacity=int(OmegaConf.select(cfg, "rdagger.online_buffer_capacity", default=200000)),
        remove_obs_keys=list(remove_obs_keys),
        sampler=sampler,
        min_action=action_min,
        max_action=action_max,
    )
    if use_offline:
        train_buffer = MixedReplayBuffer(
            buffer_1=offline_buffer,
            buffer_2=online_buffer,
            sample_ratio=None if offline_ratio is None else float(offline_ratio),
            sampler=sampler,
        )
    else:
        train_buffer = online_buffer

    alg_name = str(OmegaConf.select(cfg, "alg.offline_alg_name", default="bc")).lower()
    if alg_name not in ALG_TO_CONFIG:
        raise ValueError(f"Unknown algorithm '{alg_name}' (choose from {sorted(ALG_TO_CONFIG)}).")
    algo_dict = OmegaConf.to_container(OmegaConf.select(cfg, "offline_algorithm"), resolve=True)
    algo_config = ALG_TO_CONFIG[alg_name](**algo_dict)
    algo_config.actor = student
    algo_config.buffer = train_buffer
    algo_config.logger = wandb_logger
    algo = algo_config.create()

    if reweighting in ("sirius", "iwr") and not OmegaConf.select(cfg, "offline_algorithm.use_weighted_bc", default=False):
        logger.warning(f"rdagger.reweighting={reweighting} needs offline_algorithm.use_weighted_bc=true to take effect.")

    # ---- Scorer + gate + gated worker ----
    dd_scorer = None
    refresh_gate = None
    if gate_type == "diffdagger":
        # Diff-DAgger baseline: the gating signal is the student's own diffusion loss. No
        # Robometer is loaded. The threshold is set below from the training-data loss
        # distribution, and recalibrated after every retrain (see the iteration loop).
        from robometer_policy_learning.utils.baseline_gates import (
            DiffDaggerScorer, QuantileGate, quantile_threshold)

        remove_keys = list(getattr(algo.actor, "remove_obs_keys", None)
                           or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or [])
        scorer = DiffDaggerScorer(algo.actor, remove_obs_keys=remove_keys, device=device)
        dd_scorer = scorer
        gate = QuantileGate(threshold=float("inf"), patience=dd_patience,
                            patience_window=dd_patience_window)

        def _refresh_gate(tag: str):
            """alpha-quantile of the diffusion loss over the current training distribution."""
            losses = dd_scorer.inner.calibrate_from_algo(algo, num_samples=dd_calib_samples)
            if len(losses) == 0:
                logger.warning(f"[diffdagger:{tag}] no calibration samples; threshold unchanged")
                return None
            thr = quantile_threshold(losses, dd_alpha)
            gate.threshold = thr
            logger.info(f"[diffdagger:{tag}] threshold={thr:.6f} (alpha={dd_alpha}, "
                        f"n={len(losses)}, loss mean={losses.mean():.6f} max={losses.max():.6f})")
            return dict(threshold=thr, loss_mean=float(losses.mean()))

        refresh_gate = _refresh_gate

    elif gate_type == "thrifty":
        import copy as _copy

        from robometer_policy_learning.utils.thrifty_gate import (
            ThriftyGate, ThriftyScorer, build_ensemble, collect_thrifty_scores,
            train_thrifty_models)

        remove_keys = list(getattr(algo.actor, "remove_obs_keys", None)
                           or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or [])
        feat_dim = int(algo.actor.global_cond_dim)
        # Ensemble: 5 MLP actors + twin Q critics, plus a frozen target copy.
        ac = build_ensemble(feat_dim, action_dim, device, num_nets=thrifty_num_nets)
        ac_targ = _copy.deepcopy(ac)
        for p in ac_targ.parameters():
            p.requires_grad = False
        q_opt = torch.optim.Adam(
            list(ac.q1.parameters()) + list(ac.q2.parameters()), lr=thrifty_lr)

        scorer = ThriftyScorer(algo, ac, remove_obs_keys=remove_keys, device=device)
        gate = ThriftyGate()

        _thrifty_iter = {"n": 0}

        def _refresh_thrifty(tag: str):
            it_n = _thrifty_iter["n"]
            steps = thrifty_train_steps * (1 + it_n)
            losses = train_thrifty_models(
                algo, ac, ac_targ,
                ens_opt_fn=lambda params: torch.optim.Adam(params, lr=thrifty_lr),
                q_opt=q_opt, grad_steps=steps, gamma=thrifty_gamma,
                num_nets=thrifty_num_nets, feat_dim=feat_dim, act_dim=action_dim, device=device,
                seed=collect_seed + 1000 * it_n)
            _thrifty_iter["n"] = it_n + 1
            nov, saf = collect_thrifty_scores(algo, ac, device=device)
            if len(nov) == 0:
                logger.warning(f"[thrifty:{tag}] no calibration samples; thresholds unchanged")
                return None
            gate.recalibrate(nov, saf, thrifty_alpha_h)
            if losses["n_positive"] == 0:
                logger.warning(f"[thrifty:{tag}] buffer holds NO goal-reaching transitions; "
                               f"the risk gate is inactive this iteration")
            logger.info(f"[thrifty:{tag}] steps={steps} n_trans={losses['n_transitions']} "
                        f"ens_loss={losses['ensemble_loss']:.5f} "
                        f"q_loss={losses['qrisk_loss']:.5f} n_pos={losses['n_positive']} | "
                        f"delta_h={gate.delta_h:.5f} beta_h={gate.beta_h:.5f} "
                        f"(novelty med={np.median(nov):.5f}, safety med={np.median(saf):.5f})")
            return dict(ensemble_loss=losses["ensemble_loss"], qrisk_loss=losses["qrisk_loss"],
                        n_positive=losses["n_positive"], novelty_median=float(np.median(nov)),
                        delta_h=gate.delta_h, beta_h=gate.beta_h)

        refresh_gate = _refresh_thrifty

    elif gate_type == "hgdagger":
        from robometer_policy_learning.utils.human_gate import (
            HumanGate, HumanScorer, collect_interactive_episode)

        scorer = HumanScorer(
            backend=hg_backend,
            port=int(OmegaConf.select(cfg, "rdagger.hg_port", default=8420)),
            pace_hz=float(OmegaConf.select(cfg, "rdagger.hg_pace_hz", default=20.0)))
        gate = HumanGate()
        logger.info(f"HG-DAgger gate: backend={hg_backend} (needs an operator at a terminal)")

    else:
        from robometer_policy_learning.utils.robometer_gate import RobometerScorer

        scorer = RobometerScorer(model_path=reward_model_path, device=device)
        gate = RewardGate(**gate_kwargs)

    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    # Videos are required for HG-DAgger 'replay' so the operator can pick the takeover step off a recording of each solo rollout.
    interactive = (gate_type == "hgdagger" and hg_backend == "replay")
    collect_seed = int(OmegaConf.select(cfg, "rdagger.seed", default=0))
    worker = GatedRolloutWorker(
        env=collect_env,
        student=algo.actor,
        expert=expert,
        scorer=scorer,
        gate=gate,
        online_buffer=online_buffer,
        device=device,
        action_dim=action_dim,
        lowdim_stats=lowdim_stats,
        remove_obs_keys=remove_obs_keys,
        student_n_action_steps=n_exec,
        expert_n_action_steps=expert_n_exec,
        score_every=score_every,
        store_only_expert=store_only_expert,
        video_dir=os.path.join(output_dir, "gated_videos") if (debug or interactive) else None,
        plot_progress=(gate_type == "robometer"),
    )

    # ---- Separate chunked env for evaluation ----
    _, eval_env = make_env(
        env_name=env_name,
        num_envs=1,
        max_episode_steps=int(cfg.env.max_episode_steps),
        chunk_size=chunk_size,
        n_action_steps=n_exec,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        dino_image_keys=dino_image_keys,
        seed=None,
    )
    eval_worker = EvaluationWorker(
        eval_env=eval_env,
        device=device,
        num_episodes=int(cfg.eval.eval_num_episodes),
        record_video=cfg.eval.eval_record_video,
        logger=wandb_logger,
        lowdim_obs_stats=lowdim_stats,
    )

    # ---- DAgger loop ----
    try:
        if bool(OmegaConf.select(cfg, "eval.eval_on_first_step", default=True)):
            logger.info("Evaluating the initial student before any correction...")
            wandb_logger.log(eval_worker.run(algo.actor), step=algo.step_counter, prefix="eval")

        if refresh_gate is not None:
            refresh_gate("init")

        for it in range(num_iterations):
            logger.info(f"===== DAgger iteration {it + 1}/{num_iterations} =====")

            ep_stats, num_kept, attempt, declined = [], 0, 0, 0
            if gate_type == "thrifty":
                gate.clear_online()   # fresh estimate pool for risk & novelty every iteration
            while num_kept < rollouts_per_iter:
                tag = f"it{it}_r{attempt}"
                if interactive:
                    # Two passes: solo rollout -> ask the operator -> re-run with the takeover.
                    # The seed must be unique per episode but stable, so a re-run reproduces it.
                    rec = collect_interactive_episode(
                        worker, scorer, tag, collect_seed + 1000 * it + attempt, store=True)
                    stats, kept = rec["stats"], rec["kept"]
                    declined += int(rec["declined"])
                else:
                    stats = worker.rollout_episode(
                        tag, store=True, 
                        require_success=require_success, require_intervention=require_intervention,
                    )
                    kept = stats["stored"] > 0
                attempt += 1
                num_kept += int(kept)
                if gate_type == "thrifty" and gate.recalibrate_online(thrifty_alpha_h):
                    logger.info(f"  [thrifty] refit on {len(gate.online_novelty)} online scores: "
                                f"delta_h={gate.delta_h:.5f} beta_h={gate.beta_h:.5f}")
                if stats is None:            # operator declined; there is no episode to log
                    logger.info(f"  kept {num_kept}/{rollouts_per_iter} (attempt {attempt}, declined)")
                    continue
                ep_stats.append(stats)
                logger.info(
                    f"  kept {num_kept}/{rollouts_per_iter} (attempt {attempt}, "
                    f"success={stats['success']}, interventions={stats['num_interventions']}, "
                    f"stored={stats['stored']})"
                )
            wandb_logger.log(
                {
                    "mean_episode_len": float(np.mean([s["steps"] for s in ep_stats])) if ep_stats else 0.0,
                    "success_rate": float(np.mean([s["success"] for s in ep_stats])) if ep_stats else 0.0,
                    "mean_interventions_per_ep": float(np.mean([s["num_interventions"] for s in ep_stats])) if ep_stats else 0.0,
                    "expert_step_fraction": float(np.mean([s["expert_steps"] / max(s["steps"], 1) for s in ep_stats])) if ep_stats else 0.0,
                    "episodes_attempted": attempt,
                    "episodes_declined": declined,
                    "online_buffer_size": len(online_buffer),
                    "iteration": it + 1,
                },
                step=algo.step_counter,
                prefix="collect",
            )

            if online_buffer.is_empty():
                logger.warning("Online buffer empty; skipping training this iteration.")
                continue

            # --- Class reweighting ---
            if reweighting == "sirius":
                stats = compute_sirius_weights(online_buffer, offline_buffer if use_offline else None)
                wandb_logger.log({f"weight_{k}": v for k, v in stats.get("weights", {}).items()},
                                 step=algo.step_counter, prefix="sirius")
            elif reweighting == "iwr":
                stats = compute_iwr_weights(online_buffer, target_intv=iwr_target_intv)
                wandb_logger.log({f"weight_{k}": v for k, v in stats.get("weights", {}).items()},
                                 step=algo.step_counter, prefix="iwr")

            # --- Train the student (weighted BC on mixed offline + gated online data) ---
            for i in tqdm(range(train_steps_per_iter), desc="Training", unit="step"):
                algo.train_step(logging_prefix="train")
                if cfg.eval.eval_freq and (i + 1) % cfg.eval.eval_freq == 0:
                    wandb_logger.log(eval_worker.run(algo.actor), step=algo.step_counter, prefix="eval")

            # --- Recalibrate the Diff-DAgger threshold against the retrained policy ---
            # The policy just changed, so its training-loss distribution shifted; a stale
            # threshold would drift toward firing on states it has already learned.
            if refresh_gate is not None:
                info = refresh_gate(f"it{it + 1}")
                if info:
                    wandb_logger.log(info, step=algo.step_counter, prefix=gate_type)

            # --- End-of-iteration eval + checkpoint ---
            wandb_logger.log(eval_worker.run(algo.actor), step=algo.step_counter, prefix="eval")
            if (it + 1) % save_interval == 0:
                save_checkpoint(algo, save_dir, it + 1)
                logger.info(f"Saved checkpoint to {os.path.join(save_dir, str(it + 1))}")

    except KeyboardInterrupt:
        logger.info("Interrupted; saving final checkpoint and exiting.")
        save_checkpoint(algo, save_dir, "interrupted")
    finally:
        collect_env.close()
        try:
            eval_env.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            wandb_logger.finish()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
