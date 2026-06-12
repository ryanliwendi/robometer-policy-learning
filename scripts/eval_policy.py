#!/usr/bin/env python3
"""Evaluate a pretrained policy for a number of episodes and print the metrics.

Given a pretraining (or HITL) run output directory, this loads the saved Hydra config and actor
checkpoint, rebuilds the SAME evaluation environment used by ``scripts/train_hitl.py`` (chunked
open-loop execution + the matching low-dim obs normalization), runs autonomous (no-human) episodes
via :class:`EvaluationWorker`, and prints the aggregate metrics. Nothing is logged to wandb.

``load_dir`` is a run output directory containing ``.hydra/config.yaml`` (env/training/model/policy
are read from there so the env build and obs normalization match the loaded model) and
``checkpoints/<step>/actor.pt`` (``--checkpoint`` selects which <step>; default = latest).

Both image pipelines are supported: Mode A policies (precomputed/frozen DINO embeddings) get a
DINOv2 encoder rebuilt from ``model.dinov2_model`` and attached to the env so it emits the same
``dino_embedding`` key; Mode B policies (actor-side image featurizer) run on raw frames directly.

Usage:
    uv run python scripts/eval_policy.py --load-dir /path/to/outputs/2026-06-04/16-21-35
    uv run python scripts/eval_policy.py --load-dir /path/to/run --checkpoint 50000 --num-episodes 50
    uv run python scripts/eval_policy.py --load-dir /path/to/run --seed 0 --record-video
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2  # noqa: F401  (import order matters; keep above torch)

import argparse

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.samplers import RandomSampler
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker


def _load_pretrain_cfg(load_dir: str) -> DictConfig:
    """Load the Hydra config saved by the pretraining run at ``load_dir/.hydra/config.yaml``."""
    cfg_path = os.path.join(load_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Pretraining config not found at {cfg_path}. load_dir must be a run output directory "
            "(it should contain .hydra/config.yaml and checkpoints/<step>/actor.pt)."
        )
    return OmegaConf.load(cfg_path)


def _resolve_checkpoint_dir(load_dir: str, checkpoint=None) -> str:
    """Resolve the checkpoint directory (containing ``actor.pt``) inside a run dir.

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained policy and print metrics (no wandb).")
    parser.add_argument("--load-dir", required=True, help="Run output dir (contains .hydra/config.yaml and checkpoints/<step>/actor.pt).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint <step> to load (e.g. 50000 or latest). Default: latest.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Env seed (default: None = random).")
    parser.add_argument("--dataset-path", default=None, help="Override the H5 dataset path used to build the env.")
    parser.add_argument("--record-video", action="store_true", help="Record one episode to evaluation_videos/ (no wandb).")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Number of action steps to execute (default: None = use training.n_action_steps).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Adopt env / training / model / policy from the run's saved Hydra config so the env build,
    # chunking and obs normalization match exactly what the actor was trained with. ----
    pre_cfg = _load_pretrain_cfg(args.load_dir)
    OmegaConf.resolve(pre_cfg)

    # ---- Pretrained actor (a deployable BaseActor pickled by Algorithm.save) ----
    ckpt_dir = _resolve_checkpoint_dir(args.load_dir, args.checkpoint)
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor.pt not found in {actor_path}.")
    actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
    actor.eval()
    print(f"Loaded actor {type(actor).__name__} from {actor_path}")

    dataset_path = args.dataset_path or pre_cfg.env.h5_dataset_path
    chunk_size = OmegaConf.select(pre_cfg, "training.chunk_size", default=None)
    n_exec = int(OmegaConf.select(pre_cfg, "training.n_action_steps", default=1))
    if args.n_action_steps is not None:
        n_exec = int(args.n_action_steps)
    normalize_lowdim = bool(OmegaConf.select(pre_cfg, "training.normalize_lowdim_obs", default=False))

    dinov2_model_id = OmegaConf.select(pre_cfg, "model.dinov2_model", default=None)
    image_encoder_type = OmegaConf.select(pre_cfg, "model.image_encoder.type", default=None)
    featurizer_level_image_encoding = image_encoder_type in ("impala", "resnet", "dinov2")
    use_env_dino = bool(dinov2_model_id) and not featurizer_level_image_encoding

    dinov2_model = dinov2_processor = None
    dino_image_keys = list(OmegaConf.select(pre_cfg, "env.dino_image_keys", default=["image"]) or ["image"])
    if use_env_dino:
        from transformers import AutoImageProcessor, AutoModel

        print(f"Mode A: building DINOv2 encoder '{dinov2_model_id}' to embed {dino_image_keys} online...")
        dinov2_model = AutoModel.from_pretrained(dinov2_model_id).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_id)

    # Language embeddings: only when the run used a sentence model (adds a 'language' obs key).
    sentence_model_id = OmegaConf.select(pre_cfg, "model.sentence_model", default=None)
    sentence_model = None
    if sentence_model_id:
        from sentence_transformers import SentenceTransformer

        print(f"Building sentence model '{sentence_model_id}' for language embeddings...")
        sentence_model = SentenceTransformer(sentence_model_id)

    # Keys to drop from observed obs (match what the actor was trained with). For Mode A the actor's
    # own remove_obs_keys already includes the raw image keys, so they are dropped inside act().
    remove_obs_keys = list(
        getattr(actor, "remove_obs_keys", None)
        or OmegaConf.select(pre_cfg, "env.extra_keys_to_drop", default=[])
        or []
    )

    # ---- Evaluation env: chunked open-loop execution, matching train_hitl's eval_env. The DINO /
    # sentence encoders are attached inside setup_robomimic_env when provided (Mode A). ----
    eval_env, _ = setup_robomimic_env(
        dataset_path=dataset_path,
        n_envs=1,
        device=device,
        seed=args.seed,
        max_episode_steps=pre_cfg.env.max_episode_steps,
        use_full_state=pre_cfg.env.use_full_state,
        terminate_on_success=True,
        chunk_size=chunk_size,
        n_action_steps=n_exec,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        sentence_model=sentence_model,
        image_keys=dino_image_keys,
    )

    # ---- Low-dim obs normalization stats: the EvaluationWorker z-scores obs before act(), so when
    # the policy was trained with normalize_lowdim_obs we must reproduce the SAME stats from the
    # dataset (otherwise the policy sees mis-scaled observations). ----
    lowdim_stats = {}
    if normalize_lowdim:
        asp = eval_env.single_action_space
        if asp is not None and np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
            action_min = np.asarray(asp.low, dtype=np.float32)
            action_max = np.asarray(asp.high, dtype=np.float32)
        else:
            action_min = action_max = None
        print("Building offline H5 buffer to recover low-dim obs normalization stats...")
        offline_buffer = H5ReplayBuffer(
            h5_paths=[dataset_path],
            sampler=RandomSampler(),
            remove_obs_keys=list(remove_obs_keys),
            min_action=action_min,
            max_action=action_max,
            normalize_lowdim_obs=True,
        )
        lowdim_stats = offline_buffer.lowdim_obs_stats
        print(f"Low-dim obs normalization keys: {list(lowdim_stats.keys())}")

    # ---- Run evaluation (no wandb: logger=None). ----
    eval_worker = EvaluationWorker(
        eval_env=eval_env,
        device=device,
        num_episodes=args.num_episodes,
        record_video=args.record_video,
        logger=None,
        lowdim_obs_stats=lowdim_stats,
    )

    try:
        metrics = eval_worker.run(actor)
    finally:
        try:
            eval_env.close()
        except Exception:  # noqa: BLE001
            pass

    print(f"\n===== Evaluation metrics ({args.num_episodes} episodes) =====")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float, np.floating, np.integer)):
            print(f"  {key:>20}: {float(value):.4f}")
        else:
            print(f"  {key:>20}: {value}")


if __name__ == "__main__":
    main()
