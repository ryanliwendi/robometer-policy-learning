#!/usr/bin/env python3
"""Offline Robometer labelling of real-world DROID trajectories.

Layout expected (DROID):
    <root>/{Successes,Failures}/<episode>/recordings/SVO/<camera_serial>.mp4

Labels come from the Successes/Failures split; the h5 files are not read. One mp4 frame == one
control step (verified: frame count == h5 step count), so trace index == env step index.

Usage:
    uv run python scripts/label_real_world.py --camera 37998989 --score-every 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Dict, List

import numpy as np
import torch
from loguru import logger

logger.remove()
logger.add(lambda m: print(m, end=""), level="INFO")

from robometer.evals.eval_server import process_batch_helper
from robometer.evals.eval_utils import (
    extract_rewards_from_output,
    extract_success_probs_from_output,
    raw_dict_to_sample,
)
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

WRIST_CAMERA = "14064085"
SIDE_CAMERAS = ("33790348", "37998989")

TASK_INSTRUCTIONS = {
    "task_put": "Pick up the red ball and put it in the purple cup",
    "task_stack": "Stack the red block on top of the blue block",
}


class RobometerLabeler:
    """Causal prefix scorer. Mirrors _RewardModelInferenceMixin without the gym dependency."""

    def __init__(self, model_path: str, device: str, max_frames: int | None = None):
        cfg, tokenizer, processor, model = load_model_from_hf(model_path=model_path, device=device)
        model.eval()
        self.cfg = cfg
        self.model = model
        self.tokenizer = getattr(model, "tokenizer", None) or tokenizer
        self.processor = getattr(model, "processor", None) or processor

        data_cfg = getattr(cfg, "data", None)
        if data_cfg is not None and getattr(data_cfg, "use_multi_image", True) is False:
            data_cfg.use_multi_image = True  # Feed frames as a batch of images rather than a video clip
        self.max_frames = int(max_frames or getattr(data_cfg, "max_frames", 8))

        self.model_type = getattr(getattr(cfg, "model", None), "model_type", None)
        if self.model_type is None:
            raise ValueError("config.model.model_type missing; cannot run local inference")
        self.device = str(getattr(model, "device", None) or next(model.parameters()).device)

        loss_cfg = getattr(cfg, "loss", None)
        self.is_discrete = getattr(loss_cfg, "progress_loss_type", None) == "discrete"
        self.num_bins = getattr(loss_cfg, "progress_discrete_bins", None)
        self.batch_collator = setup_batch_collator(self.processor, self.tokenizer, cfg, is_eval=True)
        logger.info(
            f"Loaded {model_path} on {self.device} | max_frames={self.max_frames} "
            f"discrete={self.is_discrete} bins={self.num_bins}"
        )

    def score_prefixes(self, frames: np.ndarray, task: str, end_indices: List[int], batch_size: int):
        """Score frames[0:e+1] for each e in end_indices. Returns (progress, success_prob) arrays.

        Each prefix is pre-subsampled to max_frames here; raw_dict_to_sample's own linspace
        subsample is then a no-op, so this is equivalent to passing the full prefix but far
        cheaper in memory and collation time.
        """
        progress, success = [], []
        for start in range(0, len(end_indices), batch_size):
            chunk = end_indices[start : start + batch_size]
            raws = []
            for e in chunk:
                n = e + 1
                if n <= self.max_frames:
                    idx = np.arange(n)
                else:
                    idx = np.linspace(0, e, self.max_frames).astype(int)
                raws.append(
                    dict(
                        frames=frames[idx],
                        task=task,
                        id=str(e),
                        metadata=dict(subsequence_length=int(n)),
                        video_embeddings=None,
                        text_embedding=None,
                    )
                )
            samples = [
                raw_dict_to_sample(raw_data=r, max_frames=self.max_frames, sample_type="progress")
                for r in raws
            ]
            outputs = process_batch_helper(
                model_type=self.model_type,
                model=self.model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.device,
                batch_data=[s.model_dump() for s in samples],
                job_id=0,
                is_discrete_mode=bool(self.is_discrete),
                num_bins=self.num_bins,
            )
            progress.extend(extract_rewards_from_output(outputs).tolist())
            try:
                success.extend(extract_success_probs_from_output(outputs).tolist())
            except ValueError:
                success.extend([float("nan")] * len(chunk))
        return np.asarray(progress, dtype=np.float32), np.asarray(success, dtype=np.float32)


def discover_episodes(root: str) -> List[Dict]:
    """Enumerate episodes under a task directory; the successes/failures split is the label.

    Accepts <task_dir>/{successes,failures}/<episode>/... and also tolerates the older
    capitalised split names from the first data drop.
    """
    episodes = []
    for split, is_success in (("successes", True), ("failures", False)):
        pattern = os.path.join(root, split, "*")
        if not glob.glob(pattern):
            pattern = os.path.join(root, split.capitalize(), "*")
        for d in sorted(glob.glob(pattern)):
            if not os.path.isdir(d):
                continue
            name = os.path.basename(d)
            if not glob.glob(os.path.join(d, "recordings", "SVO", "*.mp4")):
                logger.warning(f"skipping {split}/{name}: no mp4s")
                continue
            episodes.append(
                dict(
                    episode_id=f"{split}/{name}",
                    name=name,
                    split=split,
                    success=is_success,
                    # "good7" -> "good": the qualitative bucket the operator assigned
                    category=re.sub(r"\d+$", "", name),
                    dir=d,
                )
            )
    return episodes


def load_frames(path: str, width: int, height: int) -> np.ndarray:
    import decord

    vr = decord.VideoReader(path, num_threads=2, width=width, height=height)
    return vr.get_batch(range(len(vr))).asnumpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/scr/liryan/real_world_data")
    ap.add_argument("--task-dir", default="task_put", choices=sorted(TASK_INSTRUCTIONS),
                    help="which task subdirectory to label")
    ap.add_argument("--task", default=None,
                    help="language instruction; defaults to the TASK_INSTRUCTIONS entry for --task-dir")
    ap.add_argument("--model-path", default="robometer/Robometer-4B")
    ap.add_argument("--camera", default=SIDE_CAMERAS[1], help=f"serial, or 'all'. wrist={WRIST_CAMERA}")
    ap.add_argument("--score-every", type=int, default=5,
                    help="score cadence in env steps; mirrors rdagger.score_every at deploy time")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=None, help="override model context frames")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="only label the first N episodes (smoke test)")
    ap.add_argument("--episodes", default=None,
                    help="comma-separated episode names to label, e.g. 'good1,mid1,far1' (smoke test)")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    # Deliberately outside --data-root: an earlier run lost its output when the data tree was
    # reorganised underneath it.
    ap.add_argument("--out-dir", default="/scr/liryan/robometer_policy_learning/outputs/robometer_labels")
    args = ap.parse_args()

    task_root = os.path.join(args.data_root, args.task_dir)
    instruction = args.task or TASK_INSTRUCTIONS[args.task_dir]
    cameras = list(SIDE_CAMERAS) + [WRIST_CAMERA] if args.camera == "all" else [args.camera]
    episodes = discover_episodes(task_root)
    if args.episodes:
        want = {n.strip() for n in args.episodes.split(",")}
        episodes = [e for e in episodes if e["name"] in want]
        missing = want - {e["name"] for e in episodes}
        if missing:
            logger.warning(f"requested episodes not found: {sorted(missing)}")
    if args.limit:
        episodes = episodes[: args.limit]
    n_s = sum(e["success"] for e in episodes)
    logger.info(f"[{args.task_dir}] {len(episodes)} episodes ({n_s} success, "
                f"{len(episodes) - n_s} failure); cameras={cameras}; task={instruction!r}")

    labeler = RobometerLabeler(args.model_path, args.device, args.max_frames)
    os.makedirs(args.out_dir, exist_ok=True)

    for cam in cameras:
        out_path = os.path.join(
            args.out_dir,
            f"episode_stats_{args.task_dir}_cam{cam}_every{args.score_every}{args.tag}.json",
        )
        records, t_start = [], time.time()

        for i, ep in enumerate(episodes):
            video = os.path.join(ep["dir"], "recordings", "SVO", f"{cam}.mp4")
            if not os.path.exists(video):
                logger.warning(f"  {ep['episode_id']}: missing {cam}.mp4, skipping")
                continue

            frames = load_frames(video, args.width, args.height)
            T = int(frames.shape[0])
            ticks = list(range(0, T, args.score_every))
            t0 = time.time()
            progress, success_prob = labeler.score_prefixes(
                frames, instruction, ticks, args.batch_size
            )
            del frames

            records.append(
                dict(
                    episode_id=ep["episode_id"],
                    name=ep["name"],
                    success=ep["success"],
                    category=ep["category"],
                    # absolute, so downstream plotting survives another data reorganisation
                    episode_dir=os.path.abspath(ep["dir"]),
                    num_steps=T,
                    score_every=args.score_every,
                    # env-step index of every scored tick; trace[i] was produced at step tick_steps[i]
                    tick_steps=ticks,
                    progress_trace=[float(p) for p in progress],
                    success_prob_trace=[float(s) for s in success_prob],
                )
            )
            dt = time.time() - t0
            logger.info(
                f"[{cam}] {i + 1}/{len(episodes)} {ep['episode_id']:<24s} T={T:4d} "
                f"ticks={len(ticks):3d} {dt:5.1f}s ({dt / max(len(ticks), 1):.2f}s/tick) "
                f"final_progress={progress[-1]:.3f} max={progress.max():.3f} success={ep['success']}"
            )

            with open(out_path, "w") as f:
                json.dump(
                    dict(
                        meta=dict(
                            reward_model=args.model_path,
                            task=instruction,
                            task_dir=args.task_dir,
                            camera=cam,
                            camera_role="wrist" if cam == WRIST_CAMERA else "side",
                            score_every=args.score_every,
                            max_frames=labeler.max_frames,
                            resize=[args.width, args.height],
                            data_root=args.data_root,
                            note="progress_trace is at scorer-tick resolution; "
                                 "env step of trace[i] == tick_steps[i]",
                        ),
                        episodes=records,
                    ),
                    f,
                )

        logger.success(
            f"[{cam}] {len(records)} episodes in {(time.time() - t_start) / 60:.1f} min -> {out_path}"
        )


if __name__ == "__main__":
    main()
