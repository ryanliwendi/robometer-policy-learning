#!/usr/bin/env python3
"""Collect HG-DAgger human-in-the-loop rollouts and save them as a robomimic-style HDF5 dataset.

Starting from a pretrained policy (the data-collection / rollout policy), a real teleoperator
(keyboard or 3D SpaceMouse) takes over to correct the policy, and the collected trajectories are
written to an HDF5 file in the layout consumed by :class:`H5ReplayBuffer`:

    /data.attrs["env_args"]            (copied from the source dataset, so the env can be rebuilt)
    /data/demo_{i}/actions             [N, action_dim]   env-space actions actually executed
    /data/demo_{i}/rewards             [N]
    /data/demo_{i}/dones               [N]               1 at the episode's terminal/truncated step
    /data/demo_{i}/intervention        [N]               per-step HITL label (0=policy, 1=human)
    /data/demo_{i}/obs/{key}           [N, ...]           RAW observations (lowdim un-normalized)

Conventions (to stay compatible with H5ReplayBuffer):
  * Observations are stored RAW: the collection worker z-scores low-dim obs for the policy's act(),
    so we un-normalize them back with the same stats before writing.
  * Actions are stored in ENV space; H5ReplayBuffer maps them to [-1, 1] at sample time via
    ``min_action`` / ``max_action`` (also written under ``/meta`` here for convenience).
  * The per-step ``intervention`` dataset preserves the HG-DAgger label (0=policy / 1=human).

``load_dir`` points at a pretraining run output directory: env / training / model / policy are read
from ``load_dir/.hydra/config.yaml`` and the actor from ``load_dir/checkpoints/<step>/actor.pt``.

Usage:
    uv run python scripts/collect_hitl_rollouts_publish.py --config-name robomimic_collect_hgdagger \
        load_dir=/path/to/run hitl.collect_num_rollouts=50 \
        hitl.collect_output_path=/path/to/hitl_rollouts.hdf5
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2  # noqa: F401  (import order matters; keep above torch)

import json
from collections import OrderedDict
from datetime import datetime

import h5py
import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from robometer.utils.logger import get_logger, setup_loguru_logging
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.samplers import RandomSampler
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env
from robometer_policy_learning.utils.hitl_utils_publish import HitlRolloutWorker, describe_control_mode

logger = get_logger()

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
    """Resolve the checkpoint directory (containing ``actor.pt``) inside a pretraining run dir."""
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


def _to_np(x):
    """Convert a stored obs/action value (numpy or cpu torch tensor) to a numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _read_source_env_args(dataset_path: str):
    """Return the ``data.attrs['env_args']`` JSON string from the source dataset (or None)."""
    try:
        with h5py.File(dataset_path, "r") as f:
            if "data" in f and "env_args" in f["data"].attrs:
                return f["data"].attrs["env_args"]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read env_args from {dataset_path}: {e}")
    return None


def _write_h5(
    online_buffer,
    output_path,
    *,
    lowdim_stats,
    action_min,
    action_max,
    env_args,
    meta_extra,
):
    """Write the collected online buffer to a robomimic-style HDF5 file.

    Transitions are grouped into demos by ``episode_id`` (in first-seen order), sorted by
    ``step_in_episode``. Low-dim obs are un-normalized back to raw using ``lowdim_stats`` (the
    inverse of the collection worker's z-scoring) so the file follows the raw-obs convention.
    """
    transitions = [t for t in online_buffer.get_all_transitions() if t is not None]
    if not transitions:
        raise RuntimeError("Online buffer is empty; nothing to write.")

    episodes = OrderedDict()
    for t in transitions:
        episodes.setdefault(t.episode_id, []).append(t)
    for eid in episodes:
        episodes[eid].sort(key=lambda x: int(x.step_in_episode))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    total = 0
    intervention_total = 0
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for i, (eid, trs) in enumerate(episodes.items()):
            n = len(trs)
            g = data_grp.create_group(f"demo_{i}")

            actions = np.stack([_to_np(t.action).astype(np.float32).reshape(-1) for t in trs], axis=0)
            rewards = np.asarray([float(t.reward) for t in trs], dtype=np.float32)
            dones = np.asarray([float(bool(t.done) or bool(t.truncated)) for t in trs], dtype=np.float32)
            interv = np.asarray(
                [int(round(float((t.info or {}).get("intervention", 0)))) for t in trs], dtype=np.int64
            )

            g.create_dataset("actions", data=actions)
            g.create_dataset("rewards", data=rewards)
            g.create_dataset("dones", data=dones)
            g.create_dataset("intervention", data=interv)

            obs_grp = g.create_group("obs")
            for key in trs[0].obs.keys():
                seq = np.stack([_to_np(t.obs[key]) for t in trs], axis=0)
                if key in lowdim_stats:  # invert the worker's z-score -> raw
                    st = lowdim_stats[key]
                    seq = (seq.astype(np.float32) * st["std"] + st["mean"]).astype(np.float32)
                # gzip-compress image-like obs (per-step ndim >= 2) to keep files reasonable.
                kwargs = {"compression": "gzip"} if seq.ndim >= 3 else {}
                obs_grp.create_dataset(key, data=seq, **kwargs)

            g.attrs["num_samples"] = n
            total += n
            intervention_total += int((interv == 1).sum())

        data_grp.attrs["total"] = total
        if env_args is not None:
            data_grp.attrs["env_args"] = env_args  # robomimic env-rebuild metadata

        # ---- /meta: action bounds, source low-dim stats, and collection provenance ----
        meta_grp = f.create_group("meta")
        if action_min is not None and action_max is not None:
            meta_grp.create_dataset("min_action", data=np.asarray(action_min, dtype=np.float32))
            meta_grp.create_dataset("max_action", data=np.asarray(action_max, dtype=np.float32))
        if lowdim_stats:
            ls = meta_grp.create_group("lowdim_stats")
            for key, st in lowdim_stats.items():
                kg = ls.create_group(key)
                kg.create_dataset("mean", data=np.asarray(st["mean"], dtype=np.float32))
                kg.create_dataset("std", data=np.asarray(st["std"], dtype=np.float32))
        meta_grp.attrs["info"] = json.dumps(meta_extra)

    return {
        "num_demos": len(episodes),
        "total_transitions": total,
        "intervention_transitions": intervention_total,
        "intervention_fraction": (intervention_total / total) if total else 0.0,
    }


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = HydraConfig.get().runtime.output_dir
    log_level = OmegaConf.select(cfg, "logging.log_level", default="INFO")
    setup_loguru_logging(log_level=log_level, output_dir=output_dir)

    # ---- Adopt env / training / model / policy from the pretraining run's saved Hydra config. ----
    load_dir = OmegaConf.select(cfg, "load_dir", default=None)
    if not load_dir:
        raise ValueError("Set load_dir=<pretraining run dir> (contains .hydra/config.yaml and checkpoints/<step>/actor.pt).")
    pre_cfg = _load_pretrain_cfg(load_dir)
    OmegaConf.set_struct(cfg, False)
    for key in ("env", "training", "model", "policy"):
        if key in pre_cfg:
            cfg[key] = pre_cfg[key]
    OmegaConf.set_struct(cfg, True)
    OmegaConf.resolve(cfg)
    logger.info(f"Adopted env/training/model/policy from {os.path.join(load_dir, '.hydra', 'config.yaml')}")

    # ---- Data-collection (rollout) policy ----
    ckpt_dir = _resolve_checkpoint_dir(load_dir, OmegaConf.select(cfg, "checkpoint", default=None))
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor.pt not found in {actor_path}.")
    actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
    actor.eval()
    logger.info(f"Loaded rollout actor {type(actor).__name__} from {actor_path}")
    if OmegaConf.select(cfg, "env.dino_image_keys", default=None):
        raise NotImplementedError("DINO-embedding (Mode A) policies are not supported by the HITL collector yet.")

    remove_obs_keys = list(getattr(actor, "remove_obs_keys", None) or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or [])

    # ---- Single, non-chunked teleop/collection env (worker manages chunking manually). ----
    dataset_path = cfg.env.h5_dataset_path
    env, _ = setup_robomimic_env(
        dataset_path=dataset_path,
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
    asp = env.single_action_space
    if asp is not None and np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
        action_min = np.asarray(asp.low, dtype=np.float32)
        action_max = np.asarray(asp.high, dtype=np.float32)
    else:
        action_min = action_max = None

    n_exec = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)
    normalize_lowdim = bool(OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False))
    store_only_human = bool(OmegaConf.select(cfg, "hitl.store_only_human", default=False))

    # ---- The human is a real teleoperator (no simulated humans in this release). ----
    human_mode = str(OmegaConf.select(cfg, "hitl.human_mode", default="real")).lower()
    if human_mode != "real":
        raise ValueError(
            f"hitl.human_mode must be 'real' in this release (got {human_mode!r}); simulated-human "
            "collection is not included."
        )

    # ---- Low-dim obs normalization stats (built from the source dataset, matching the policy). The
    # worker z-scores obs for act(); we invert this when writing raw obs to the H5. ----
    lowdim_stats = {}
    if normalize_lowdim:
        logger.info("Building offline H5 buffer to recover low-dim obs normalization stats...")
        offline_buffer = H5ReplayBuffer(
            h5_paths=[dataset_path],
            sampler=RandomSampler(),
            remove_obs_keys=list(remove_obs_keys),
            min_action=action_min,
            max_action=action_max,
            normalize_lowdim_obs=True,
        )
        lowdim_stats = offline_buffer.lowdim_obs_stats
        logger.info(f"Low-dim obs normalization keys: {list(lowdim_stats.keys())}")

    # ---- Online buffer accumulating the collected rollouts (sampler unused during collection). ----
    online_buffer = ReplayBuffer(
        capacity=int(OmegaConf.select(cfg, "hitl.online_buffer_capacity", default=200000)),
        remove_obs_keys=list(remove_obs_keys),
        sampler=RandomSampler(),
        min_action=action_min,
        max_action=action_max,
    )

    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    video_dir = os.path.join(output_dir, "debug_videos") if debug else None
    worker = HitlRolloutWorker(
        env=env,
        actor=actor,
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

    # =====================================================================================
    # Collection loop
    # =====================================================================================
    num_target = int(OmegaConf.select(cfg, "hitl.collect_num_rollouts", default=OmegaConf.select(cfg, "hitl.rollouts_per_iter", default=10)))
    keep_only_hitl = bool(OmegaConf.select(cfg, "hitl.keep_only_hitl_rollouts", default=False))
    output_path = OmegaConf.select(cfg, "hitl.collect_output_path", default=None) or os.path.join(output_dir, "hitl_rollouts.hdf5")

    logger.info(
        f"Collecting {num_target} successful rollouts "
        f"(store_only_human={store_only_human}, keep_only_hitl_rollouts={keep_only_hitl}) -> {output_path}"
    )

    # Count KEPT rollouts (stored>0) toward the target: an episode is kept only if it passes the
    # require_success / keep_only_hitl_rollouts (>=1 intervention) filters in rollout_episode.
    collected, attempt, num_success = 0, 0, 0
    try:
        while collected < num_target:
            steps, human_steps, stored, success = worker.rollout_episode(
                f"ep{attempt}", phase="COLLECT", allow_human=True, store=True,
                require_success=True, require_intervention=keep_only_hitl,
            )
            attempt += 1
            num_success += int(bool(success))
            collected += int(stored > 0)
            logger.info(
                f"  rollout {attempt}: success={bool(success)} steps={steps} human_steps={human_steps} "
                f"stored={stored} | kept {collected}/{num_target}"
            )
    except KeyboardInterrupt:
        logger.info("Interrupted; writing what has been collected so far.")
    finally:
        worker.close()
        env.close()

    # =====================================================================================
    # Write the dataset
    # =====================================================================================
    if online_buffer.is_empty():
        logger.warning("Online buffer is empty; nothing written.")
        return

    env_args = _read_source_env_args(dataset_path)
    meta_extra = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "load_dir": load_dir,
        "checkpoint": OmegaConf.select(cfg, "checkpoint", default=None),
        "source_dataset": dataset_path,
        "human_mode": human_mode,
        "store_only_human": store_only_human,
        "keep_only_hitl_rollouts": keep_only_hitl,
        "n_action_steps": n_exec,
        "chunk_size": OmegaConf.select(cfg, "training.chunk_size", default=None),
        "normalize_lowdim_obs": normalize_lowdim,
        "obs_are_raw": True,  # low-dim obs were un-normalized back to raw before writing
        "remove_obs_keys": list(remove_obs_keys),
        "intervention_labels": {"policy": 0, "human": 1, "offline": LABEL_OFFLINE},
        "num_attempts": attempt,
        "num_successful": num_success,
    }
    stats = _write_h5(
        online_buffer,
        output_path,
        lowdim_stats=lowdim_stats,
        action_min=action_min,
        action_max=action_max,
        env_args=env_args,
        meta_extra=meta_extra,
    )
    logger.success(
        f"Wrote {stats['num_demos']} demos ({attempt} attempts) / {stats['total_transitions']} transitions "
        f"({stats['intervention_transitions']} human, {stats['intervention_fraction']:.1%}) to {output_path}"
    )


if __name__ == "__main__":
    main()
