#!/usr/bin/env python3
"""The DAgger loop for Reward DAgger.

Starting from a pretrained STUDENT policy (load_dir) and a frozen EXPERT
(rdagger.expert_dir), each iteration:
  1. collects reward-gated rollouts with GatedRolloutWorker. Every stored step is labelled
     intervention=1 (expert correction) or 0 (student) — the same convention as HITL, so
     the SIRIUS/IWR reweighting utilities are reused unchanged;
  2. trains behavior cloning for rdagger.train_steps_per_iter steps on the online buffer,
     optionally mixed with the offline demos (MixedReplayBuffer) and optionally with
     per-sample reweighting (rdagger.reweighting in {sirius, iwr}) consumed by weighted BC
     (offline_algorithm.use_weighted_bc=true);
  3. evaluates autonomously (EvaluationWorker, no gate / no expert) and checkpoints;
then repeats. The only change vs human HG-DAgger is WHO supervises: a generalist reward
model decides WHEN to intervene, and a strong task policy decides WHAT the correction is.

Usage (note the '+' prefixes: load_dir and rdagger.* are NOT in libero_bc_config.yaml, and
hydra's struct mode rejects overrides of non-existent keys without '+'):
    uv run python scripts/train_reward_dagger.py --config-name libero_bc_config \
        +load_dir=/path/to/student_bc_run \
        +rdagger.expert_dir=/path/to/expert_dp_run \
        +offline_algorithm.use_weighted_bc=true +rdagger.reweighting=iwr
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

from gated_rollout_worker import GatedRolloutWorker, Pi0Actor, RobometerScorer, load_actor

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

    # ---- Adopt env / training / model / policy from the STUDENT's pretraining run ----
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
    expert_k = int(OmegaConf.select(cfg, "rdagger.expert_k", default=40))
    expert_n_exec = int(OmegaConf.select(cfg, "rdagger.expert_n_action_steps", default=5))
    expert_exit_mode = str(OmegaConf.select(cfg, "rdagger.expert_exit_mode", default="fixed"))
    recovery_delta = float(OmegaConf.select(cfg, "rdagger.recovery_delta", default=0.2))
    min_expert_steps = int(OmegaConf.select(cfg, "rdagger.min_expert_steps", default=10))
    max_expert_steps = int(OmegaConf.select(cfg, "rdagger.max_expert_steps", default=80))
    warmup_steps = int(OmegaConf.select(cfg, "rdagger.warmup_steps", default=0))
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
    save_interval = int(OmegaConf.select(cfg, "rdagger.save_interval", default=1))
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

    # ---- DINOv2 (both actors are DINO-mode; the eval env stack embeds frames online) ----
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

    # ---- Collection env: eval stack, UNchunked (the worker chunks manually so control can
    # switch student<->expert mid-episode and each side replans on takeover). ----
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

    # One sampler shared by all buffers so chunked sampling is consistent across them.
    if chunk_size is None:
        sampler = RandomSampler()
    else:
        gamma = OmegaConf.select(cfg, "offline_algorithm.gamma", default=0.99)
        sampler = ChunkedSequentialSampler(chunk_size=int(chunk_size), obs_as_sequence=False, gamma=gamma)

    # ---- Offline H5 buffer ----
    lowdim_stats = {}
    offline_buffer = None
    if use_offline or normalize_lowdim:
        offline_buffer = H5ReplayBuffer(
            h5_paths=[cfg.env.h5_dataset_path],
            sampler=sampler,
            remove_obs_keys=list(remove_obs_keys),
            # DINO embeddings must be attached so offline samples match the online rollout
            # samples (which carry dino_embedding from the env's DinoEmbeddingWrapper).
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            dino_embedding_keys=dino_image_keys,
            min_action=action_min,
            max_action=action_max,
            normalize_lowdim_obs=normalize_lowdim,
            default_intervention_label=LABEL_OFFLINE,  # offline demos -> SIRIUS 'demo' class
        )
        lowdim_stats = offline_buffer.lowdim_obs_stats
        if not use_offline:
            offline_buffer = None  # only needed for stats

    # ---- Online buffer + training buffer ----
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

    # ---- Student training algorithm ----
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

    # ---- Robometer scorer + gate + gated worker ----
    scorer = RobometerScorer(model_path=reward_model_path, device=device)
    gate = RewardGate(**gate_kwargs)
    debug = bool(OmegaConf.select(cfg, "debug", default=False))
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
        expert_k=expert_k,
        expert_exit_mode=expert_exit_mode,
        recovery_delta=recovery_delta,
        min_expert_steps=min_expert_steps,
        max_expert_steps=max_expert_steps,
        warmup_steps=warmup_steps,
        score_every=score_every,
        store_only_expert=store_only_expert,
        video_dir=os.path.join(output_dir, "gated_videos") if debug else None,
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

    # ---- Reward-DAgger loop ----
    try:
        if bool(OmegaConf.select(cfg, "eval.eval_on_first_step", default=True)):
            logger.info("Evaluating the initial student before any correction...")
            wandb_logger.log(eval_worker.run(algo.actor), step=algo.step_counter, prefix="eval")

        for it in range(num_iterations):
            logger.info(f"===== reward-DAgger iteration {it + 1}/{num_iterations} =====")

            # --- Collect gated rollouts until rollouts_per_iter episodes are kept ---
            ep_stats, num_kept, attempt = [], 0, 0
            while num_kept < rollouts_per_iter:
                stats = worker.rollout_episode(
                    f"it{it}_r{attempt}", store=True,
                    require_success=require_success, require_intervention=require_intervention,
                )
                attempt += 1
                ep_stats.append(stats)
                num_kept += int(stats["stored"] > 0)
                logger.info(
                    f"  kept {num_kept}/{rollouts_per_iter} (attempt {attempt}, "
                    f"success={stats['success']}, interventions={stats['num_interventions']}, "
                    f"stored={stats['stored']})"
                )
            wandb_logger.log(
                {
                    "mean_episode_len": float(np.mean([s["steps"] for s in ep_stats])),
                    "success_rate": float(np.mean([s["success"] for s in ep_stats])),
                    "mean_interventions_per_ep": float(np.mean([s["num_interventions"] for s in ep_stats])),
                    "expert_step_fraction": float(np.mean([s["expert_steps"] / max(s["steps"], 1) for s in ep_stats])),
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
                stats = compute_iwr_weights(online_buffer)
                wandb_logger.log({f"weight_{k}": v for k, v in stats.get("weights", {}).items()},
                                 step=algo.step_counter, prefix="iwr")

            # --- Train the student (weighted BC on mixed offline + gated online data) ---
            for i in tqdm(range(train_steps_per_iter), desc="Training", unit="step"):
                algo.train_step(logging_prefix="train")
                if cfg.eval.eval_freq and (i + 1) % cfg.eval.eval_freq == 0:
                    wandb_logger.log(eval_worker.run(algo.actor), step=algo.step_counter, prefix="eval")

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
