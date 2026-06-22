#!/usr/bin/env python3
"""Iterative human-in-the-loop training: HG-DAgger and its weighted (SIRIUS / IWR) variants.

Starting from a pretrained policy (and optionally the offline dataset used to train it), each
iteration:
  1. collects rollouts until ``hitl.rollouts_per_iter`` KEPT trajectories are stored. Every step is
     labelled intervention=1 (human correction) or 0 (policy). What gets stored to the online buffer
     is controlled by ``hitl.store_only_human``:
        * True  (HG-DAgger, default) -> store ONLY the human corrections (label 1),
        * False                      -> store ALL transitions (labels 0/1);
  2. trains behavior cloning (``bc``) for ``hitl.train_steps_per_iter`` steps on the online buffer,
     optionally mixed with the offline dataset (labelled intervention=2 via MixedReplayBuffer), and
     optionally with per-sample reweighting (``hitl.reweighting`` in {sirius, iwr}) consumed by
     weighted BC (``offline_algorithm.use_weighted_bc=true``);
then repeats.

Alternatively, set ``hitl.precollected_hitl_dataset`` to a HITL HDF5 written by
``scripts/collect_hitl_rollouts_publish.py`` to train on precollected interventions instead of
collecting online. In that mode online collection is skipped and ``hitl.num_iterations`` must be 1.

The human is a real teleoperator (keyboard or 3D SpaceMouse). ``load_dir`` points at a pretraining
run output directory; env / training / model / policy are read from ``load_dir/.hydra/config.yaml``
so they match the loaded actor, and weights are read from ``load_dir/checkpoints/<step>/actor.pt``.

Usage:
    uv run python scripts/train_hitl_publish.py --config-name robomimic_hgdagger \
        load_dir=/path/to/run
    # weighted HG-DAgger (IWR):
    uv run python scripts/train_hitl_publish.py --config-name robomimic_hgdagger \
        load_dir=/path/to/run hitl.store_only_human=false hitl.reweighting=iwr \
        offline_algorithm.use_weighted_bc=true
    # train on a precollected dataset:
    uv run python scripts/train_hitl_publish.py --config-name robomimic_hgdagger \
        load_dir=/path/to/run hitl.num_iterations=1 \
        hitl.precollected_hitl_dataset=/path/to/hitl_rollouts.hdf5

Controls (teleop):
  * Keyboard:   Tab = take/release control, wasd/rf + zx/tg/cv to move, space = gripper,
                q = abort episode, ESC = quit.
  * SpaceMouse: Tab = take/release control, move/twist the puck to move, left button = gripper,
                right button = abort episode, ESC = quit.
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2  # noqa: F401  (import order matters; keep above torch)

from datetime import datetime

import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robometer.utils.logger import get_logger, setup_loguru_logging
from robometer_policy_learning.algorithms.bc import BCConfig
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.mixed_replay_buffer import MixedReplayBuffer
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env
from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.utils.hitl_utils_publish import (
    HitlRolloutWorker,
    compute_iwr_weights,
    compute_sirius_weights,
    describe_control_mode,
    load_precollected_hitl,
)
from robometer_policy_learning.utils.training_utils import save_checkpoint

logger = get_logger()

# Only behavior cloning (HG-DAgger and its weighted variants) is supported in this public release.
ALG_TO_CONFIG = {"bc": BCConfig}

# Offline-demo intervention label (online labels 0=policy / 1=human are set by the rollout worker).
LABEL_OFFLINE = 2


def _load_pretrain_cfg(load_dir: str) -> DictConfig:
    """Load the Hydra config saved by the pretraining run at ``load_dir/.hydra/config.yaml``."""
    cfg_path = os.path.join(load_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Pretraining config not found at {cfg_path}. load_dir must be a pretraining run output "
            "directory (it should contain .hydra/config.yaml and checkpoints/<step>/actor.pt)."
        )
    return OmegaConf.load(cfg_path)


def _resolve_checkpoint_dir(load_dir: str, checkpoint=None) -> str:
    """Resolve the checkpoint directory (containing ``actor.pt``) inside a pretraining run dir.

    Expected layout is ``load_dir/checkpoints/<step>/actor.pt``. ``checkpoint`` picks a specific
    ``<step>`` (e.g. ``50000`` or ``latest``); when None, prefer ``latest`` else the largest numeric
    step. For backward compatibility, a ``load_dir`` that directly contains ``actor.pt`` is returned
    as-is.
    """
    if os.path.exists(os.path.join(load_dir, "actor.pt")):
        return load_dir
    ckpt_root = os.path.join(load_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"No 'checkpoints/' directory under {load_dir} (and no actor.pt directly in it).")
    if checkpoint is not None:
        cand = os.path.join(ckpt_root, str(checkpoint))
        if not os.path.exists(os.path.join(cand, "actor.pt")):
            raise FileNotFoundError(f"actor.pt not found in {cand} (checkpoint={checkpoint!r}).")
        return cand
    steps = [d for d in os.listdir(ckpt_root) if os.path.exists(os.path.join(ckpt_root, d, "actor.pt"))]
    if not steps:
        raise FileNotFoundError(f"No '<step>/actor.pt' checkpoints found under {ckpt_root}.")
    if "latest" in steps:
        chosen = "latest"
    else:
        numeric = [d for d in steps if d.isdigit()]
        chosen = max(numeric, key=int) if numeric else sorted(steps)[-1]
    return os.path.join(ckpt_root, chosen)


def _read_precollected_meta(path: str) -> dict:
    """Return the provenance metadata dict written under ``/meta`` by the collector (or {})."""
    import json

    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "meta" in f and "info" in f["meta"].attrs:
                return json.loads(f["meta"].attrs["info"])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read /meta from precollected dataset {path}: {e}")
    return {}


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = HydraConfig.get().runtime.output_dir
    save_dir = os.path.join(output_dir, "checkpoints")

    # Setup loguru logging (console + file sink under the Hydra output dir).
    log_level = OmegaConf.select(cfg, "logging.log_level", default="INFO")
    setup_loguru_logging(log_level=log_level, output_dir=output_dir)

    # Setup wandb logger.
    string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{cfg.logging.wandb_name}_{string_time}"
    wandb_logger = WandbLogger(
        exp_name=exp_name,
        offline=cfg.logging.wandb_offline,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        log_dir=f"{cfg.logging.wandb_log_dir_base}/{string_time}",
        prefix="offline",
    )

    # ---- Align with the loaded model: adopt env / training / model / policy from the pretraining
    # run's saved Hydra config so the env build, chunking, obs normalization and architecture match
    # exactly what the actor was trained with. ----
    load_dir = OmegaConf.select(cfg, "load_dir", default=None)
    if not load_dir:
        raise ValueError(
            "Set load_dir=<pretraining run dir> (contains .hydra/config.yaml and checkpoints/<step>/actor.pt)."
        )
    pre_cfg = _load_pretrain_cfg(load_dir)
    OmegaConf.set_struct(cfg, False)
    for key in ("env", "training", "model", "policy"):
        if key in pre_cfg:
            cfg[key] = pre_cfg[key]
    OmegaConf.set_struct(cfg, True)
    OmegaConf.resolve(cfg)
    logger.info(f"Adopted env/training/model/policy from {os.path.join(load_dir, '.hydra', 'config.yaml')}")

    alg_name = str(OmegaConf.select(cfg, "alg.offline_alg_name", default="bc")).lower()
    if alg_name not in ALG_TO_CONFIG:
        raise ValueError(
            f"Unknown HITL algorithm '{alg_name}'. This release only supports HG-DAgger via "
            f"'bc' (choose from {sorted(ALG_TO_CONFIG)})."
        )

    # ---- Precollected HITL dataset: when set, train on a saved HITL HDF5 instead of collecting
    # interventions online. There is no online loop to iterate, so require num_iterations == 1;
    # the online collection (env/worker) is skipped entirely. ----
    precollected_hitl_dataset = OmegaConf.select(cfg, "hitl.precollected_hitl_dataset", default=None)
    num_iterations = int(OmegaConf.select(cfg, "hitl.num_iterations", default=10))
    if precollected_hitl_dataset:
        if not os.path.exists(precollected_hitl_dataset):
            raise FileNotFoundError(f"hitl.precollected_hitl_dataset not found: {precollected_hitl_dataset}")
        assert num_iterations == 1, (
            "hitl.precollected_hitl_dataset is set, so there is no online collection to iterate over; "
            f"set hitl.num_iterations=1 (got {num_iterations})."
        )
        # Verify the dataset was collected with the SAME data-collection policy as this run's
        # load_dir / checkpoint, so the policy we train is the one that produced the interventions.
        _meta = _read_precollected_meta(precollected_hitl_dataset)
        _ds_load_dir = _meta.get("load_dir")
        if _ds_load_dir is None:
            logger.warning(
                f"Precollected dataset {precollected_hitl_dataset} has no 'load_dir' in its /meta; "
                "cannot verify it was collected with the same policy as load_dir."
            )
        elif os.path.normpath(str(_ds_load_dir)) != os.path.normpath(str(load_dir)):
            raise ValueError(
                "Precollected HITL dataset was collected with a different policy than the configured "
                f"load_dir:\n  dataset load_dir = {_ds_load_dir}\n  config  load_dir = {load_dir}\n"
                "These must match so the trained policy is the one that produced the interventions."
            )
        else:
            _ds_ckpt = _meta.get("checkpoint")
            _cfg_ckpt = OmegaConf.select(cfg, "checkpoint", default=None)
            if str(_ds_ckpt) != str(_cfg_ckpt):
                raise ValueError(
                    "Precollected HITL dataset was collected with a different checkpoint than the "
                    f"configured one:\n  dataset checkpoint = {_ds_ckpt}\n  config  checkpoint = {_cfg_ckpt}\n"
                    "These must match (or both be null/latest)."
                )
            logger.info(f"Precollected dataset matches config load_dir={load_dir}, checkpoint={_cfg_ckpt}.")
        logger.info(f"Precollected HITL dataset: {precollected_hitl_dataset} (online collection skipped).")

    # ---- Pretrained actor (a deployable BaseActor pickled by Algorithm.save) ----
    ckpt_dir = _resolve_checkpoint_dir(load_dir, OmegaConf.select(cfg, "checkpoint", default=None))
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor.pt not found in {actor_path}.")
    actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
    actor.train()
    logger.info(f"Loaded pretrained actor {type(actor).__name__} from {actor_path}")

    if OmegaConf.select(cfg, "env.dino_image_keys", default=None):
        raise NotImplementedError("DINO-embedding (Mode A) policies are not supported by the HITL collector yet.")

    # Keys to drop from stored/observed obs (match what the actor was trained with).
    remove_obs_keys = list(getattr(actor, "remove_obs_keys", None) or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or [])

    # ---- Single, non-chunked teleop/collection env (the HITL worker manages chunking manually).
    # Autonomous eval uses a separate EvaluationWorker + env (built below) so its videos are logged. ----
    env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path,
        n_envs=1,
        device=device,
        seed=OmegaConf.select(cfg, "hitl.seed", default=0),
        max_episode_steps=cfg.env.max_episode_steps,
        use_full_state=cfg.env.use_full_state,
        terminate_on_success=True,
        chunk_size=None,
        n_action_steps=1,
    )
    action_dim = int(env.single_action_space.shape[0])
    logger.info(f"Control mode: {describe_control_mode(env)}")

    # Action normalization bounds (buffers map stored env-space actions to [-1, 1] at sample time).
    asp = env.single_action_space
    if asp is not None and np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
        action_min = np.asarray(asp.low, dtype=np.float32)
        action_max = np.asarray(asp.high, dtype=np.float32)
    else:
        action_min = action_max = None

    chunk_size = OmegaConf.select(cfg, "training.chunk_size", default=None)
    n_exec = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)
    normalize_lowdim = bool(OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False))
    use_offline = bool(OmegaConf.select(cfg, "hitl.use_offline", default=False))
    store_only_human = bool(OmegaConf.select(cfg, "hitl.store_only_human", default=False))
    # When true, online collection keeps only successful rollouts that contain >= 1 human
    # intervention (discarding intervention-free successes); false keeps all successful rollouts.
    keep_only_hitl_rollouts = bool(OmegaConf.select(cfg, "hitl.keep_only_hitl_rollouts", default=False))

    # ---- The human is a real teleoperator (no simulated humans in this release). ----
    human_mode = str(OmegaConf.select(cfg, "hitl.human_mode", default="real")).lower()
    if human_mode != "real":
        raise ValueError(
            f"hitl.human_mode must be 'real' in this release (got {human_mode!r}); simulated-human "
            "collection is not included."
        )

    # One sampler instance shared by the online (and offline) buffers so chunked sampling is
    # consistent across them (the base ReplayBuffer.sample uses its own sampler).
    if chunk_size is None:
        sampler = RandomSampler()
    else:
        gamma = OmegaConf.select(cfg, "offline_algorithm.gamma", default=0.99)
        sampler = ChunkedSequentialSampler(chunk_size=int(chunk_size), obs_as_sequence=False, gamma=gamma)

    # ---- Offline H5 buffer (for low-dim obs stats and/or mixing). Built when mixing is on, or
    # whenever we need normalization stats that must match the pretrained policy. ----
    lowdim_stats = {}
    offline_buffer = None
    if use_offline or normalize_lowdim:
        logger.info("Building offline H5 buffer (low-dim cache; images loaded lazily)...")
        offline_buffer = H5ReplayBuffer(
            h5_paths=[cfg.env.h5_dataset_path],
            sampler=sampler,
            remove_obs_keys=list(remove_obs_keys),
            min_action=action_min,
            max_action=action_max,
            normalize_lowdim_obs=normalize_lowdim,
            default_intervention_label=LABEL_OFFLINE,  # offline demos -> SIRIUS 'demo' class
        )
        lowdim_stats = offline_buffer.lowdim_obs_stats
        if normalize_lowdim:
            logger.info(f"Low-dim obs normalization keys: {list(lowdim_stats.keys())}")
        if not use_offline:
            offline_buffer = None  # only needed for stats

    # ---- Online replay buffer (accumulates HITL rollouts, or the precollected dataset) ----
    online_buffer = ReplayBuffer(
        capacity=int(OmegaConf.select(cfg, "hitl.online_buffer_capacity", default=200000)),
        remove_obs_keys=list(remove_obs_keys),
        sampler=sampler,
        min_action=action_min,
        max_action=action_max,
    )
    if precollected_hitl_dataset:
        stats = load_precollected_hitl(precollected_hitl_dataset, online_buffer, lowdim_stats, remove_obs_keys)
        logger.info(
            f"Loaded precollected HITL dataset: {stats['transitions']} transitions across "
            f"{stats['episodes']} episodes ({stats['human']} human-correction steps)."
        )

    # ---- Training buffer (online, optionally mixed with offline) + algorithm ----
    if use_offline:
        # offline_sample_ratio=null -> mix by the buffers' live size ratio (offline grows-relative).
        offline_ratio = OmegaConf.select(cfg, "hitl.offline_sample_ratio", default=None)
        train_buffer = MixedReplayBuffer(
            buffer_1=offline_buffer,
            buffer_2=online_buffer,
            sample_ratio=None if offline_ratio is None else float(offline_ratio),
            sampler=sampler,
        )
    else:
        train_buffer = online_buffer

    algo_dict = OmegaConf.to_container(OmegaConf.select(cfg, "offline_algorithm"), resolve=True)
    algo_config = ALG_TO_CONFIG[alg_name](**algo_dict)
    algo_config.actor = actor
    algo_config.buffer = train_buffer
    algo_config.logger = wandb_logger
    algo = algo_config.create()
    logger.info(f"HITL algorithm: {alg_name} | store_only_human={store_only_human} | use_offline={use_offline}")

    reweighting = OmegaConf.select(cfg, "hitl.reweighting", default=None)
    if reweighting in ("sirius", "iwr") and not OmegaConf.select(cfg, "offline_algorithm.use_weighted_bc", default=False):
        logger.warning(
            f"hitl.reweighting={reweighting} computes per-sample weights, but they are only consumed "
            "by BC with use_weighted_bc=true. Set offline_algorithm.use_weighted_bc=true."
        )

    # ---- Rollout/teleop worker. Not built in precollected-dataset mode (no online collection). ----
    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    video_dir = os.path.join(output_dir, "debug_videos") if debug else None
    worker = None
    if not precollected_hitl_dataset:
        if debug:
            logger.info(f"debug=True: recording kept HITL collection trajectory videos to {video_dir}")
        worker = HitlRolloutWorker(
            env=env,
            actor=algo.actor,
            online_buffer=online_buffer,
            device=device,
            action_dim=action_dim,
            lowdim_stats=lowdim_stats,
            remove_obs_keys=remove_obs_keys,
            n_action_steps=n_exec,
            store_only_human=store_only_human,
            enable_render=bool(OmegaConf.select(cfg, "teleop.enable_render", default=True)),
            teleop_device=str(OmegaConf.select(cfg, "teleop.device", default="keyboard")),
            takeover_key=str(OmegaConf.select(cfg, "teleop.takeover_key", default="tab")),
            camera=OmegaConf.select(cfg, "teleop.camera", default="agentview"),
            wrist_camera=OmegaConf.select(cfg, "teleop.wrist_camera", default="robot0_eye_in_hand"),
            show_wrist=bool(OmegaConf.select(cfg, "teleop.show_wrist", default=True)),
            record_video=debug,
            video_dir=video_dir,
        )

    # ---- Autonomous eval via EvaluationWorker (separate env so chunked actors execute open-loop). ----
    eval_env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path,
        n_envs=1,
        device=device,
        seed=None,
        max_episode_steps=cfg.env.max_episode_steps,
        use_full_state=cfg.env.use_full_state,
        terminate_on_success=True,
        chunk_size=chunk_size,
        n_action_steps=n_exec,
    )
    eval_worker = EvaluationWorker(
        eval_env=eval_env,
        device=device,
        num_episodes=int(cfg.eval.eval_num_episodes),
        record_video=cfg.eval.eval_record_video,
        logger=wandb_logger,
        lowdim_obs_stats=lowdim_stats,
    )

    # =====================================================================================
    # Iterative loop (a single pass in precollected-dataset mode)
    # =====================================================================================
    rollouts_per_iter = int(OmegaConf.select(cfg, "hitl.rollouts_per_iter", default=5))
    train_steps_per_iter = int(OmegaConf.select(cfg, "hitl.train_steps_per_iter", default=2000))
    clear_each_iter = bool(OmegaConf.select(cfg, "hitl.clear_buffer_each_iter", default=False))
    save_interval = int(OmegaConf.select(cfg, "hitl.save_interval", default=1))

    try:
        if bool(OmegaConf.select(cfg, "eval.eval_on_first_step", default=True)):
            logger.info("Evaluating initial policy before training...")
            eval_metrics = eval_worker.run(algo.actor)
            wandb_logger.log(eval_metrics, step=algo.step_counter, prefix="eval")
        for it in range(num_iterations):
            logger.info(f"===== HiTL iteration {it + 1}/{num_iterations} ({alg_name}) =====")
            # In precollected-dataset mode the online buffer is already populated; do not clear it
            # and do not collect online.
            if clear_each_iter and not precollected_hitl_dataset:
                online_buffer.clear()

            # --- 1) Collect rollouts until rollouts_per_iter KEPT trajectories are stored. An
            # episode is kept only if it passes the require_success and keep_only_hitl_rollouts
            # (>=1 intervention) filters. No attempt cap (Ctrl+C aborts and saves a checkpoint).
            # Skipped entirely with a precollected dataset. ---
            if not precollected_hitl_dataset:
                succ, lengths, human_frac, stored_total = [], [], [], 0
                num_success, num_kept, attempt = 0, 0, 0
                while num_kept < rollouts_per_iter:
                    steps, human_steps, stored, success = worker.rollout_episode(
                        f"it{it}_r{attempt}", phase="COLLECT", allow_human=True, store=True,
                        require_success=True, require_intervention=keep_only_hitl_rollouts,
                    )
                    attempt += 1
                    succ.append(float(success))
                    lengths.append(steps)
                    human_frac.append(human_steps / max(steps, 1))
                    stored_total += stored
                    num_success += int(bool(success))
                    num_kept += int(stored > 0)
                    logger.info(
                        f"  kept {num_kept}/{rollouts_per_iter} rollouts "
                        f"(attempt {attempt}, success={bool(success)}, stored={stored})"
                    )
                wandb_logger.log(
                    {
                        "mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
                        "intervention_fraction": float(np.mean(human_frac)) if human_frac else 0.0,
                        "transitions_stored": stored_total,
                        "online_buffer_size": len(online_buffer),
                        "iteration": it + 1,
                        "successful_attempt_rate": num_success / max(attempt, 1),
                    },
                    step=algo.step_counter,
                    prefix="collect",
                )

            if online_buffer.is_empty():
                logger.warning("Online buffer is empty (no transitions stored yet); skipping training this iteration.")
                continue

            # --- 1b) Optional class reweighting (recomputed each round as the dataset grows) ---
            if reweighting == "sirius":
                stats = compute_sirius_weights(online_buffer, offline_buffer if use_offline else None)
                log_d = {f"count_{k}": v for k, v in stats["counts"].items()}
                log_d.update({f"weight_{k}": v for k, v in stats.get("weights", {}).items()})
                wandb_logger.log(log_d, step=algo.step_counter, prefix="sirius")
            elif reweighting == "iwr":
                stats = compute_iwr_weights(online_buffer)  # corrections (D_I) vs autonomy (D_R), online only
                log_d = {f"count_{k}": v for k, v in stats["counts"].items()}
                log_d.update({f"weight_{k}": v for k, v in stats.get("weights", {}).items()})
                wandb_logger.log(log_d, step=algo.step_counter, prefix="iwr")

            # --- 2) Train behavior cloning ---
            for i in tqdm(range(train_steps_per_iter), desc="Training", unit="step"):
                algo.train_step(logging_prefix="train")  # logs internally at algo.step_counter
                # --- 3) Autonomous (no-human) evaluation ---
                if (i + 1) % cfg.eval.eval_freq == 0 and cfg.eval.eval_freq is not None:
                    eval_metrics = eval_worker.run(algo.actor)
                    wandb_logger.log(eval_metrics, step=algo.step_counter, prefix="eval")
            if (it + 1) % save_interval == 0:
                save_checkpoint(algo, save_dir, it + 1)
                logger.info(f"Saved checkpoint to {os.path.join(save_dir, str(it + 1))}")

    except KeyboardInterrupt:
        logger.info("Interrupted; saving final checkpoint and exiting.")
        save_checkpoint(algo, save_dir, "interrupted")
    finally:
        if worker is not None:
            worker.close()
        env.close()
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
