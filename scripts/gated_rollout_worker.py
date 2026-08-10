#!/usr/bin/env python3
"""Gated rollouts.

Standalone usage:
    srun --gres=shard:8 --mem=32G --time=2:00:00 \
      uv run python scripts/gated_rollout_worker.py \
        --student-dir outputs/<dp_run> --expert-dir outputs/<dp_run> \
        --episodes 3 --video-dir gated_videos
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# A pi0 expert is JAX; Robometer and DINOv2 are torch. JAX preallocates ~75% of the GPU on first
# use, which starves torch and OOMs the reward model. Must be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device  # noqa: E402
from robometer_policy_learning.modules.transformer.modeling_transformer_actor import TransformerActor  # noqa: E402
from robometer_policy_learning.algorithms.dp.modeling_dp import DiffusionActor  # noqa: E402

ROLLOUT_LABEL, INTERVENTION_LABEL = 0, 1

# ---- Helpers ----
def _extract0(batched):
    """Extract env 0 from a vectorized obs dict / array (n_envs=1)."""
    if isinstance(batched, dict):
        return {k: v[0] for k, v in batched.items()}
    return batched[0]


def _scalar(x):
    return np.asarray(x).reshape(-1)[0]


def _fmt_score(v):
    if isinstance(v, (tuple, list, np.ndarray)):
        return "(" + ", ".join(f"{float(x):.3f}" for x in np.asarray(v).reshape(-1)) + ")"
    return f"{float(v):.3f}"


def _success_from_info(info) -> bool:
    if isinstance(info, dict):
        for key in ("is_success", "success"):
            if key in info:
                return bool(np.asarray(info[key]).reshape(-1)[0])
    return False


def dump_episode_stats(episodes, save_path: str, meta: dict | None = None):
    """Write per-episode score traces + success labels to JSON.

    The on-disk key stays ``progress_trace`` regardless of gate: every stored corpus and every
    analysis script (gate_frontier, gate_transfer, analyze_gated_runs, the plotting scripts) reads
    that name. ``meta['gate_type']`` says which gate produced it.
    """
    import json

    payload = dict(
        meta=meta or {},
        episodes=[
            dict(
                episode=i,
                success=bool(s["success"]),
                steps=int(s["steps"]),
                progress_trace=[float(p) for p in s["score_trace"]],
                gate_fires=[int(f) for f in s["gate_fires"]],
                gate_reasons=list(s["gate_reasons"]),
                expert_steps=int(s["expert_steps"]),
                num_interventions=int(s["num_interventions"]),
            )
            for i, s in enumerate(episodes)
        ],
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f)
    n_succ = sum(e["success"] for e in payload["episodes"])
    logger.info(f"wrote {len(payload['episodes'])} episodes ({n_succ} success) -> {save_path}")


class Pi0Actor:
    """openpi pi0 policy behind the BaseActor .act() interface the worker expects.

    No preprocessing needed: LiberoPI0Wrapper already emits exactly pi0's input format (224x224
    uint8 images, 8-dim state, prompt) -- the DP actors just drop those keys via remove_obs_keys."""

    raw_obs = True  # take the unconverted numpy obs dict, not a device tensor
    obs_keys = ("observation/image", "observation/wrist_image", "observation/state", "prompt")

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        from robometer_policy_learning.utils.pi0_integration import load_pi0_policy

        self.policy = load_pi0_policy(os.path.expanduser(str(checkpoint_dir)))
        self.training = False
        self.remove_obs_keys = []

    # Compatibility if pi0 policy is instantiated as student
    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def act(self, obs, deterministic: bool = True):
        """obs: pi0-format numpy dict. Returns (actions (horizon, action_dim), actor_state)."""
        result = self.policy.infer(obs)
        return np.asarray(result["actions"], dtype=np.float32), None


def load_actor(run_dir: str, device: str, checkpoint=None, trainable: bool = False):
    """Load a BaseActor from a pretraining run dir (checkpoints/<step>/...)."""
    if os.path.exists(os.path.join(run_dir, "actor.pt")):
        ckpt_dir = run_dir
    else:
        root = os.path.join(run_dir, "checkpoints")
        steps = [d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "actor.pt"))]
        if checkpoint is not None:
            chosen = str(checkpoint)
        elif "latest" in steps:
            chosen = "latest"
        else:
            chosen = max((d for d in steps if d.isdigit()), key=int)
        ckpt_dir = os.path.join(root, chosen)
    names = ["online_actor.pt", "actor.pt"] if trainable else ["actor.pt"]  # online_actor.pt is DP's trainable network
    path = next(os.path.join(ckpt_dir, n) for n in names if os.path.exists(os.path.join(ckpt_dir, n)))
    actor = torch.load(path, map_location=device, weights_only=False).to(device)
    if trainable:
        for p in actor.parameters():
            p.requires_grad_(True)
        actor.train()
    else:
        actor.eval()
    logger.info(f"Loaded actor {type(actor).__name__} from {path} (trainable={trainable})")
    return actor


class GatedRolloutWorker:
    """DAgger-style rollouts where a reward-model gate replaces human interventions."""
    def __init__(
        self,
        env,
        student,
        expert,
        scorer,
        gate,
        online_buffer,
        device,
        action_dim: int,
        *,
        lowdim_stats: dict | None = None,
        remove_obs_keys = None,
        frame_key: str = "observation/image",
        student_n_action_steps: int = 10,
        expert_n_action_steps: int = 10,
        score_every: int = 1,
        store_only_expert: bool = False,
        video_dir: str | None = None,
        video_fps: int = 20,
        video_scale: int = 2,
        plot_progress: bool = True,
    ):
        self.env = env
        self.student = student
        self.expert = expert
        self.scorer = scorer
        self.gate = gate
        self.online_buffer = online_buffer
        self.device = device
        self.action_dim = int(action_dim)
        self.lowdim_stats = lowdim_stats or {}
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.frame_key = frame_key  # keys used by robometer scorer
        self.student_n_action_steps = int(student_n_action_steps)
        self.expert_n_action_steps = int(expert_n_action_steps)
        self.score_every = max(1, int(score_every))
        self.store_only_expert = bool(store_only_expert)
        self.video_dir = video_dir
        self.video_fps = int(video_fps)
        self.video_scale = int(video_scale)  # upscale so the time step index is readable
        self.plot_progress = bool(plot_progress)
 
        if self.plot_progress and type(scorer).__name__ != "RobometerScorer":
            raise ValueError(f"plot_progress=True is only valid with RobometerScorer")
        if self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

    # ---- Policy action with receding-horizon chunking ----
    def _actor_obs(self, actor, obs):
        """Each actor gets the observation it was trained on. The student (DP) runs on DINO
        embeddings + state as torch tensors; a pi0 expert runs on raw 224px images + prompt as
        numpy."""
        if getattr(actor, "raw_obs", False):
            return {k: obs[k] for k in actor.obs_keys if k in obs}
        return move_to_device(convert_to_tensor(self._prep_obs(obs)), self.device)

    def _policy_action(self, actor, obs, st, n_exec):
        "Take action from the current chunk if possible; replan otherwise."
        if st["chunk"] is None or st["pos"] >= len(st["chunk"]) or st["pos"] >= n_exec:
            with torch.inference_mode():
                pred, _ = actor.act(self._actor_obs(actor, obs), deterministic=True)
            if torch.is_tensor(pred):  # pi0 returns numpy; the torch actors return tensors
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred)
            st["chunk"] = pred.reshape(-1, self.action_dim) if pred.ndim == 3 else np.atleast_2d(pred)  # Batch_size = 1 since we're running with one env
            st["pos"] = 0
        a = st["chunk"][st["pos"]]
        st["pos"] += 1
        return a

    def _prep_obs(self, obs):
        out = {}
        for k, v in obs.items():
            if k in self.remove_obs_keys:
                continue
            if k in self.lowdim_stats:
                st = self.lowdim_stats[k]
                v = ((np.asarray(v, dtype=np.float32) - st["mean"]) / st["std"]).astype(np.float32)
            out[k] = v
        return out

    def _write_video(self, frames, labels, episode_id):
        """Write the episode video: STUDENT/EXPERT label on top, step index on the bottom."""
        if not self.video_dir or not frames:
            return
        import cv2

        from robometer_policy_learning.utils.human_gate import render_operator_view

        path = os.path.join(self.video_dir, f"gated_{episode_id}.mp4")
        h, w = render_operator_view(frames[0], 0, scale=self.video_scale).shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(self.video_fps), (w, h))
        for i, (f, lab) in enumerate(zip(frames, labels)):
            writer.write(render_operator_view(
                f, i, scale=self.video_scale,
                label="EXPERT" if lab == INTERVENTION_LABEL else "STUDENT"))
        writer.release()
        # cv2's mp4v fourcc is not decodable by browsers/VS Code; re-encode to H.264 in place
        try:
            import shutil, subprocess
            if shutil.which("ffmpeg"):
                tmp = path + ".h264.mp4"
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path,
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
                    check=True, timeout=120, stdin=subprocess.DEVNULL,
                )
                os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"H.264 re-encode failed for {path} ({e}); keeping mp4v file.")
        logger.info(f"Saved gated rollout video ({len(frames)} frames) to {path}")

    def rollout_episode(self, episode_id, store=True, require_success=False, require_intervention=False):
        was_training = self.student.training
        self.student.eval()
        obs, _ = self.env.reset()
        obs = _extract0(obs)  # removes batched dimension

        self.scorer.reset(task=str(obs.get("prompt", "")))
        self.gate.reset()
        self.scorer.observe(obs, self.frame_key)

        student_st = {"chunk": None, "pos": 0}
        expert_st = {"chunk": None, "pos": 0}
        expert_active = False
        seg_step = 0   # step index within the expert segment
        steps, expert_steps, num_interventions = 0, 0, 0
        success, done = False, False
        pending = []
        score_trace, gate_fires, gate_reasons = [], [], []
        video_frames, video_labels = [], []
        last_score = 0.0

        while not done:
            cur = self._prep_obs(obs)

            if expert_active:
                action = self._policy_action(self.expert, obs, expert_st, self.expert_n_action_steps)
                mode, label = "EXPERT", INTERVENTION_LABEL
                expert_steps += 1
            else:
                action = self._policy_action(self.student, obs, student_st, self.student_n_action_steps)
                mode, label = "STUDENT", ROLLOUT_LABEL

            next_b, rew, term, trunc, info = self.env.step(
                np.asarray(action, dtype=np.float32).reshape(1, self.action_dim)
            )
            next_obs = _extract0(next_b)
            terminated, truncated = bool(_scalar(term)), bool(_scalar(trunc))
            done = terminated or truncated
            if _success_from_info(info):
                success = True

            self.scorer.observe(next_obs, self.frame_key)
            scored_now = steps % self.score_every == 0
            if scored_now:
                last_score, _aux = self.scorer.score()
            score_trace.append(last_score)

            if not expert_active and scored_now and self.gate.update(last_score):
                gate_fires.append(steps)
                gate_reasons.append(self.gate.last_trigger)  # "drop" | "plateau"
                num_interventions += 1
                expert_active = True
                expert_st["chunk"] = None
                logger.info(f"  [gate] fired at step {steps} (score={_fmt_score(last_score)}, "
                            f"trigger={self.gate.last_trigger}) -> expert takeover for the rest "
                            f"of the episode")

            # Buffer episode
            if store and self.online_buffer is not None and (not self.store_only_expert or label == INTERVENTION_LABEL):
                if self.store_only_expert:
                    ep_store, step_store = f"{episode_id}_e", seg_step
                    seg_step += 1
                else:
                    ep_store, step_store = episode_id, steps
                pending.append(
                    dict(
                        obs=cur,
                        action=np.asarray(action, dtype=np.float32),
                        reward=float(_scalar(rew)),
                        next_obs=self._prep_obs(next_obs),
                        done=float(terminated),
                        truncated=float(truncated),
                        episode_id=ep_store,
                        step_in_episode=step_store,
                        info={"intervention": label},
                    )
                )

            if self.video_dir:
                video_frames.append(next_obs[self.frame_key])
                video_labels.append(label)

            obs = next_obs
            steps += 1

        # Flush the episode to the buffer if condition satisfied
        stored = 0
        if store and self.online_buffer is not None and pending:
            n_intv = sum(1 for t in pending if t["info"]["intervention"] == INTERVENTION_LABEL)
            if (success or not require_success) and (n_intv > 0 or not require_intervention):
                for t in pending:
                    t["info"]["episode_len"] = len(pending)
                    t["info"]["episode_num_interventions"] = n_intv
                    self.online_buffer.add(**t)
                stored = len(pending)

        if self.video_dir:
            self._write_video(video_frames, video_labels, episode_id)
            # Save progress curve for rdagger episodes that had an expert takeover
            if num_interventions > 0 and self.plot_progress:
                try:
                    from robometer_policy_learning.utils.robometer_gate import plot_progress_trace
                    plot_progress_trace(
                        dict(progress_trace=score_trace, gate_fires=gate_fires,
                             gate_reasons=gate_reasons, success=success,
                             num_interventions=num_interventions),
                        os.path.join(self.video_dir, f"gated_{episode_id}_progress.png"),
                        title=f"{episode_id}  success={success}  interventions={num_interventions}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"progress-plot failed for {episode_id}: {e}")

        if was_training:
            self.student.train()

        return dict(
            steps=steps,
            expert_steps=expert_steps,
            num_interventions=num_interventions,
            success=success,
            stored=stored,
            gate_fires=gate_fires,
            gate_reasons=gate_reasons,
            score_trace=score_trace,
        )


def run_interactive(args, worker, scorer):
    """HG-DAgger 'replay' mode. Rolls out the policy, asks for interventions (if any), 
    and rolls out again with the expert intervening at the given time."""
    import json
    from robometer_policy_learning.utils.human_gate import collect_interactive_episode

    records, kept, attempt = [], 0, 0
    while kept < args.episodes:
        rec = collect_interactive_episode(worker, scorer, f"hg{attempt}", args.seed + attempt,
                                          store=worker.online_buffer is not None)
        records.append(rec)
        kept += int(rec["kept"])
        attempt += 1
        logger.info(f"kept {kept}/{args.episodes} after {attempt} episodes")

    out_dir = args.video_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "collection_log.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "stats"} for r in records], f, indent=1)

    declined = sum(r["declined"] for r in records)
    logger.info(f"DONE: {kept} kept / {attempt} episodes ({declined} declined, "
                f"{attempt - kept - declined} rescues failed) -> {out_dir}/collection_log.json")


# ---- Testing ----
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Watch reward-gated rollouts (student + expert + gate).")
    parser.add_argument("--student-dir", required=True, help="student pretraining run dir (has .hydra/config.yaml)")
    parser.add_argument("--expert-dir", required=False, help="expert (DP) pretraining run dir; not needed for --expert-type pi0")
    parser.add_argument("--student-type", choices=["dp", "pi0"], default="dp")
    parser.add_argument("--expert-type", choices=["dp", "pi0"], default="pi0")
    parser.add_argument("--pi0-checkpoint", default=os.path.expanduser(
                        "~/.cache/openpi/openpi-assets/checkpoints/pi0_libero"),
                        help="'pi0' expert-type: openpi checkpoint dir (must contain 'libero' in the path)")
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--expert-checkpoint", default=None)
    parser.add_argument("--reward-model", default="jesbu1/robometer-4b-fft-libero")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--expert-n-action-steps", type=int, default=5, help="action-chunking steps the expert executes")
    parser.add_argument("--student-n-action-steps", type=int, default=None,
                        help="override the student's replan interval (default: from its training config).")
    parser.add_argument("--short-window", type=int, default=30)
    parser.add_argument("--long-window", type=int, default=120)
    parser.add_argument("--method", type=str, default="spearman", choices=["spearman", "pearson", "naive"])
    parser.add_argument("--drop-threshold", type=float, default=-0.7)
    parser.add_argument("--plateau-threshold", type=float, default=0.1)
    parser.add_argument("--min-drop-magnitude", type=float, default=0.1,
                        help="absolute drop for correlation methods. Correlation is scale-free so a tiny wiggle fires like a real collapse.")
    parser.add_argument("--smoothing", type=float, default=0.0, help="EMA weight on history in [0,1); 0 = off")
    parser.add_argument("--score-every", type=int, default=1)
    # ---- Baseline gates (comparison arms; only the gating rule changes, not student/expert) ----
    parser.add_argument("--gate-type", choices=["robometer", "diffdagger", "hgdagger"],
                        default="robometer",
                        help="'robometer' = ours (VLM progress + drop/plateau); "
                             "'diffdagger' = Diff-DAgger (student's own diffusion loss + quantile); "
                             "'hgdagger' = a human decides when to intervene")
    parser.add_argument("--hg-backend", choices=["live", "replay"], default="replay",
                        help="'live' renders frames as the episode runs and polls the keyboard for "
                             "SPACE (needs a display); 'replay' rolls the episode out solo, then "
                             "asks the operator to pick the takeover step off the recording and "
                             "re-runs it (headless, but needs a TTY for the prompts)")
    parser.add_argument("--hg-port", type=int, default=8420,
                        help="'live' only: port for the operator UI "
                             "(reach it with ssh -L <port>:localhost:<port> <host>)")
    parser.add_argument("--hg-pace-hz", type=float, default=20.0,
                        help="'live' only: hold the rollout to this wall-clock rate so the "
                             "operator sees real time, not a fast-forward. 0 = as fast as possible")
    parser.add_argument("--dd-threshold", type=float, default=None,
                        help="Diff-DAgger alpha-quantile threshold on the training-loss distribution")
    parser.add_argument("--dd-patience", type=int, default=1,
                        help="fire once this many of the last --dd-patience-window steps exceed the threshold")
    parser.add_argument("--dd-patience-window", type=int, default=None,
                        help="defaults to --dd-patience, which reproduces the paper's K-consecutive rule")
    parser.add_argument("--dd-batch-multiplier", type=int, default=1)
    parser.add_argument("--dd-num-per-batch", type=int, default=1)
    parser.add_argument("--video-dir", default="gated_videos")
    parser.add_argument("--stats-json-dir", default="gated_videos/episode_stats.json",
                        help="per-episode progress traces + success labels (input to the offline gate sweep)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    if args.expert_type == "dp" and not args.expert_dir:
        parser.error("--expert-dir is required for --expert-type dp")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Env/model params come from the STUDENT's saved training config, so the env build,
    # chunking and DINO keys match what the student was trained with
    pre_cfg = OmegaConf.load(os.path.join(args.student_dir, ".hydra", "config.yaml"))
    dino_image_keys = list(OmegaConf.select(pre_cfg, "env.dino_image_keys", default=[]) or [])
    n_exec = int(args.student_n_action_steps
                 or OmegaConf.select(pre_cfg, "training.n_action_steps", default=10))

    dinov2_model = dinov2_processor = None
    if dino_image_keys:
        from transformers import AutoImageProcessor, AutoModel

        model_id = OmegaConf.select(pre_cfg, "model.dinov2_model", default="facebook/dinov2-base")
        dinov2_model = AutoModel.from_pretrained(model_id).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(model_id)

    # Eval env stack, UNchunked (the worker chunks manually so control can switch mid-chunk).
    from robometer_policy_learning.utils.env_utils import make_env

    env, _ = make_env(
        env_name=f"{pre_cfg.env.env_name}/{pre_cfg.env.task_id}",
        num_envs=1,
        max_episode_steps=int(pre_cfg.env.max_episode_steps),
        chunk_size=None,
        n_action_steps=1,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        dino_image_keys=dino_image_keys,
        seed=args.seed,
    )
    action_dim = int(env.single_action_space.shape[0])

    # --student-type pi0 is for control to check success for PI0 policy
    if args.student_type == "pi0":
        student = Pi0Actor(args.pi0_checkpoint, device=device)
    else:
        student = load_actor(args.student_dir, device, args.student_checkpoint)
    if args.expert_type == "pi0":
        expert = Pi0Actor(args.pi0_checkpoint, device=device)
    else:
        expert = load_actor(args.expert_dir, device, args.expert_checkpoint)

    remove_obs_keys = list(getattr(student, "remove_obs_keys", None)
                           or OmegaConf.select(pre_cfg, "env.extra_keys_to_drop", default=[]) or [])

    if args.gate_type == "diffdagger":
        # Diff-DAgger baseline: the gating signal is the STUDENT's own diffusion loss, so no
        # Robometer is loaded at all. The threshold is the alpha-quantile of the loss over the
        # student's training data, recomputed per DAgger iteration (see baseline_gates.py).
        from robometer_policy_learning.utils.baseline_gates import DiffDaggerScorer, QuantileGate

        scorer = DiffDaggerScorer(
            student, remove_obs_keys=remove_obs_keys, device=device,
            batch_multiplier=args.dd_batch_multiplier, num_per_batch=args.dd_num_per_batch,
        )
        if args.dd_threshold is None:
            parser.error("--dd-threshold is required for --gate-type diffdagger "
                         "(calibrate it with scripts/calibrate_diffdagger.py)")
        gate = QuantileGate(threshold=args.dd_threshold, patience=args.dd_patience,
                            patience_window=args.dd_patience_window)
        logger.info(f"Diff-DAgger gate: {gate.describe()}")
    elif args.gate_type == "hgdagger":
        # A human decides when to intervene; the expert still performs the correction.
        from robometer_policy_learning.utils.human_gate import HumanGate, HumanScorer

        scorer = HumanScorer(backend=args.hg_backend, port=args.hg_port,
                             pace_hz=args.hg_pace_hz)
        gate = HumanGate()
        logger.info(f"HG-DAgger gate: backend={args.hg_backend}")
    else:
        # Imported here, not at module scope: a thrifty / hgdagger run must not pay for loading
        # the Robometer package at all.
        from robometer_policy_learning.utils.reward_gate import RewardGate
        from robometer_policy_learning.utils.robometer_gate import RobometerScorer

        scorer = RobometerScorer(model_path=args.reward_model, device=device)
        gate = RewardGate(
            short_window=args.short_window,
            drop_threshold=args.drop_threshold,
            long_window=args.long_window,
            plateau_threshold=args.plateau_threshold,
            method=args.method,
            smoothing=args.smoothing,
            min_drop_magnitude=args.min_drop_magnitude,
        )

    worker = GatedRolloutWorker(
        env=env,
        student=student,
        expert=expert,
        scorer=scorer,
        gate=gate,
        online_buffer=None,  # watch-only
        device=device,
        action_dim=action_dim,
        remove_obs_keys=remove_obs_keys,
        student_n_action_steps=n_exec,
        expert_n_action_steps=args.expert_n_action_steps,
        score_every=args.score_every,
        video_dir=args.video_dir,
        plot_progress=(args.gate_type == "robometer"),
    )

    # HG-DAgger 'replay' is a different shape of loop: each episode is rolled out solo, shown to an
    # operator, and only then re-run with the takeover. It collects until `--episodes` episodes are
    # KEPT (a rescue that fails is not kept), so it cannot be a fixed-length for-loop.
    if args.gate_type == "hgdagger" and args.hg_backend == "replay":
        return run_interactive(args, worker, scorer)

    episodes = []
    for ep in range(args.episodes):
        # Re-seed per episode
        _random.seed(args.seed + ep)
        np.random.seed(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        torch.cuda.manual_seed_all(args.seed + ep)

        stats = worker.rollout_episode(f"watch_{ep}", store=False)
        episodes.append(stats)
        logger.info(
            f"episode {ep}: steps={stats['steps']} success={stats['success']} "
            f"interventions={stats['num_interventions']} (at steps {stats['gate_fires']}) "
            f"expert_steps={stats['expert_steps']}"
        )
        if args.video_dir and worker.plot_progress:
            from robometer_policy_learning.utils.robometer_gate import plot_progress_trace

            plot_progress_trace(
                dict(stats, progress_trace=stats["score_trace"]),
                os.path.join(args.video_dir, f"progress_watch_{ep}.png"),
                title=f"episode {ep}: success={stats['success']}, interventions={stats['num_interventions']}",
            )
        # Re-dump every episode: a 50-episode job is long, don't lose it all to a late crash.
        dump_episode_stats(
            episodes,
            args.stats_json_dir,
            meta=dict(
                # `progress_trace` holds whatever signal the gate consumed; gate_type says which.
                gate_type=args.gate_type,
                dd_threshold=args.dd_threshold, dd_patience=args.dd_patience,
                dd_patience_window=args.dd_patience_window,
                reward_model=(args.reward_model if args.gate_type == "robometer" else None),
                student_dir=args.student_dir, student_checkpoint=args.student_checkpoint,
                expert_type=args.expert_type,
                expert_dir=args.expert_dir, expert_checkpoint=args.expert_checkpoint,
                pi0_checkpoint=(args.pi0_checkpoint if args.expert_type == "pi0" else None),
                method=args.method, short_window=args.short_window, long_window=args.long_window,
                drop_threshold=args.drop_threshold, plateau_threshold=args.plateau_threshold,
                min_drop_magnitude=args.min_drop_magnitude, smoothing=args.smoothing,
                score_every=args.score_every, seed=args.seed,
            ),
        )

    n_succ = sum(bool(s["success"]) for s in episodes)
    n_int = sum(s["num_interventions"] for s in episodes)
    logger.info(f"DONE: {n_succ}/{len(episodes)} success, {n_int} total interventions")
    env.close()


if __name__ == "__main__":
    main()
