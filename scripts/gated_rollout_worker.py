#!/usr/bin/env python3
"""Reward-gated rollouts (no training).

Rolls out the student policy on the DINO eval env stack; every executed step, Robometer
scores the causal frame history and the RewardGate watches the progress series. When the
gate fires (sharp drop or plateau) the DP expert takes over. The return mechanism decides
when the control is returned to the student policy.

Design notes:
  * One env, both obs streams: the eval stack (LiberoPI0Wrapper + DinoEmbeddingWrapper)
    yields dino_embedding + observation/state for the actors AND keeps the raw
    observation/image frames, which we feed to a standalone RobometerScorer. This avoids
    stacking LiberoRobometerRewardWrapper (which expects the raw LIBERO env) under the
    policy wrappers.
  * The env is built with chunk_size=None; the worker manages receding-horizon chunking
    manually, so student and expert can replan on control switches.

Standalone usage:
    srun --gres=shard:8 --mem=32G --time=2:00:00 \
      uv run python scripts/gated_rollout_worker.py \
        --student-dir outputs/<bc_run> --expert-dir outputs/<dp_run> \
        --episodes 3 --expert-k 40 --video-dir gated_videos
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# A pi0 expert is JAX; Robometer and DINOv2 are torch. JAX preallocates ~75% of the GPU on first
# use, which starves torch and OOMs the reward model. Must be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import sys

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from typing import List

sys.path.insert(0, "/scr/liryan/robometer_policy_learning/robometer/scripts")
from example_libero_robometer_wrapper import _RewardModelInferenceMixin  # noqa: E402

from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device  # noqa: E402
from robometer_policy_learning.utils.reward_gate import RewardGate  # noqa: E402
from robometer_policy_learning.modules.transformer.modeling_transformer_actor import TransformerActor  # noqa: E402
from robometer_policy_learning.algorithms.dp.modeling_dp import DiffusionActor  # noqa: E402

ROLLOUT_LABEL, INTERVENTION_LABEL = 0, 1


# ---- Robometer scoring for rewards and success_probs----
class RobometerScorer(_RewardModelInferenceMixin):
    """Standalone causal Robometer scorer: feed one frame per executed step, get
    (progress, success_prob) for the history so far.

    raw_dict_to_sample subsamples the history to the model's max_frames internally,
    so we keep the full episode frame list (matches the non-vector wrapper).
    """

    def __init__(self, model_path: str, device: str, max_frames=None):
        super().__init__(model_path=model_path, device=device, max_frames=max_frames)
        self.task = ""
        self.frames = []
        self.episode_id = 0

    def reset(self, task: str):
        self.task = str(task)
        self.frames = []
        self.episode_id += 1

    def append(self, frame: np.ndarray):
        self.frames.append(np.asarray(frame))

    def score(self):
        """Score the current causal history. Returns (progress, success_prob)."""
        raw = dict(
            frames=np.stack(self.frames, axis=0),
            task=self.task,
            id=self.episode_id,
            metadata=dict(subsequence_length=len(self.frames)),
            video_embeddings=None,
            text_embedding=None,
        )
        rewards, success_probs = self._compute_rewards_batch([raw])
        return float(rewards[0]), float(success_probs[0])


# ---- Helpers ----
def _extract0(batched):
    """Extract env 0 from a vectorized obs dict / array (n_envs=1)."""
    if isinstance(batched, dict):
        return {k: v[0] for k, v in batched.items()}
    return batched[0]


def _scalar(x):
    return np.asarray(x).reshape(-1)[0]


def _success_from_info(info) -> bool:
    if isinstance(info, dict):
        for key in ("is_success", "success"):
            if key in info:
                return bool(np.asarray(info[key]).reshape(-1)[0])
    return False


def plot_progress_trace(stats, save_path: str, title: str | None = None):
    """Plot the Robometer progress signal over one episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    progress = stats["progress_trace"]
    fires: List[int] = stats["gate_fires"]
    reasons = stats["gate_reasons"]  # "drop" | "plateau"
    handbacks = stats["handback_steps"]
    handback_reasons = stats["handback_reasons"]
    steps = len(progress)
    trigger_color = {"drop": "#d62728", "plateau": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(steps), progress, color="#1f77b4", lw=1.5, zorder=2)

    for i, (f, reason) in enumerate(zip(fires, reasons)):
        c = trigger_color.get(reason, "#7f7f7f")
        end = handbacks[i] if i < len(handbacks) else steps - 1  # episode ended mid-takeover
        hb = handback_reasons[i] if i < len(handback_reasons) else "episode_end"
        ax.axvspan(f, end, color=c, alpha=0.08, zorder=1)
        ax.axvline(f, color=c, lw=1.5, zorder=3)  # mark gate trigger
        hb_color = "#000000" if hb == "cap" else "green"   
        ax.axvline(end, color=hb_color, lw=1.2, ls="--", zorder=3)  # mark gate handback
        ax.annotate(f"{reason or '?'}\n@{f}", xy=(f, 1.0), xytext=(2, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=7, color=c)

    ax.set_xlabel("environment step")
    ax.set_ylabel("Robometer progress")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(title or f"success={stats['success']}  interventions={stats['num_interventions']}")
    ax.legend(handles=[
        Line2D([0], [0], color="#1f77b4", lw=1.5, label="progress"),
        Line2D([0], [0], color="#d62728", lw=1.5, label="drop takeover"),
        Line2D([0], [0], color="#ff7f0e", lw=1.5, label="plateau takeover"),
        Line2D([0], [0], color="green", lw=1.2, ls="--", label="return to student"),
        Line2D([0], [0], color="#000000", lw=1.2, ls="--", label="handback: capped"),
    ], loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def dump_episode_stats(episodes, save_path: str, meta: dict | None = None):
    """Write per-episode progress traces + success labels to JSON.

    The gate is a pure function of the progress trace, so these dumps let us grid-search
    gate hyperparameters OFFLINE on CPU (replaying RewardGate over the traces) instead of
    paying Robometer GPU inference once per candidate config.
    """
    import json

    payload = dict(
        meta=meta or {},
        episodes=[
            dict(
                episode=i,
                success=bool(s["success"]),
                steps=int(s["steps"]),
                progress_trace=[float(p) for p in s["progress_trace"]],
                gate_fires=[int(f) for f in s["gate_fires"]],
                gate_reasons=list(s["gate_reasons"]),
                handback_steps=[int(h) for h in s["handback_steps"]],
                handback_reasons=list(s["handback_reasons"]),
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

    # Whitelist, not blacklist: pi0 takes ONLY these. The DINO embeddings the DP student runs on
    # are meaningless to it, and their key names vary with env.dino_image_keys.
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
    """HG-DAgger-style rollouts where a reward-model gate replaces human interventions."""
    def __init__(
        self,
        env,
        student,
        expert,
        scorer: RobometerScorer,
        gate: RewardGate,
        online_buffer,
        device,
        action_dim: int,
        *,
        lowdim_stats: dict | None = None,
        remove_obs_keys = None,
        reward_frame_key: str = "observation/image",
        student_n_action_steps: int = 10,
        expert_n_action_steps: int = 10,
        expert_k: int = 40,
        expert_exit_mode: str = "fixed",   # "fixed" | "takeover" | "progress" | "gate"
        recovery_delta: float = 0.2,
        min_expert_steps: int = 5,
        max_expert_steps: int = 80,
        warmup_steps: int = 10,
        score_every: int = 1,
        store_only_expert: bool = False,
        video_dir: str | None = None,
        video_fps: int = 20,
    ):
        self.env = env
        self.student = student
        self.expert = expert
        self.scorer = scorer
        self.gate = gate
        self.online_buffer = online_buffer  # Store transitions for future training
        self.device = device
        self.action_dim = int(action_dim)
        self.lowdim_stats = lowdim_stats or {}
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.reward_frame_key = reward_frame_key  # keys used by robometer scorer
        self.student_n_action_steps = int(student_n_action_steps)
        self.expert_n_action_steps = int(expert_n_action_steps)
        self.expert_k = int(expert_k) # number of steps expert executes ('fixed' mode)
        self.expert_exit_mode = str(expert_exit_mode)
        self.recovery_delta = float(recovery_delta)
        self.min_expert_steps = int(min_expert_steps)
        self.max_expert_steps = int(max_expert_steps)
        assert self.expert_exit_mode in ("fixed", "takeover", "progress", "gate"), \
            f"unknown expert_exit_mode {self.expert_exit_mode!r}"
        assert self.min_expert_steps <= self.max_expert_steps, "min_expert_steps must be <= max_expert_steps"
        self.warmup_steps = int(warmup_steps)  # student steps before the gate may fire
        self.score_every = max(1, int(score_every))  # robometer cadence (need >1 for real world experiment due to latency)
        self.store_only_expert = bool(store_only_expert)  # Whether to store only expert transitions in the buffer
        self.video_dir = video_dir
        self.video_fps = int(video_fps)
        if self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

    # ---- Policy action with receding-horizon chunking ----
    def _actor_obs(self, actor, obs):
        """Each actor gets the observation it was trained on. The student (DP) runs on DINO
        embeddings + state as torch tensors; a pi0 expert runs on raw 224px images + prompt as
        numpy. A single shared obs dict cannot serve both -- the DP's remove_obs_keys drops
        exactly the keys pi0 needs."""
        if getattr(actor, "raw_obs", False):
            return {k: obs[k] for k in actor.obs_keys if k in obs}
        return move_to_device(convert_to_tensor(self._prep_obs(obs)), self.device)

    def _policy_action(self, actor, obs, st, n_exec):
        """st is a mutable {"chunk" (action_sequence), "pos" (scalar current_index)} dict; set st["chunk"]=None to force a replan."""
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
        """Normalize low-dim keys and drop unused keys -> dict used for BOTH act() and storage."""
        out = {}
        for k, v in obs.items():
            if k in self.remove_obs_keys:
                continue
            if k in self.lowdim_stats:
                st = self.lowdim_stats[k]
                v = ((np.asarray(v, dtype=np.float32) - st["mean"]) / st["std"]).astype(np.float32)
            out[k] = v
        return out

    def _handback_reason(self, takeover_steps, last_progress, takeover_progress, gate_fired_now):
        """Why the expert hands control back this step; None = keep control.
        Modes: 'fixed' (expert_k steps), 'takeover' (until episode end), 'progress' (progress
        recovered by recovery_delta above the takeover point), 'gate' (the gate would no longer
        fire on the expert-driven window). 'progress'/'gate' share the min/max-steps guardrails."""
        mode = self.expert_exit_mode
        if mode == "fixed":
            return "fixed" if takeover_steps >= self.expert_k else None
        if mode == "takeover":
            return None
        if takeover_steps >= self.max_expert_steps:
            return "cap"  # force handback if expert is also stuck
        if takeover_steps < self.min_expert_steps:
            return None
        if mode == "progress":
            return "progress recovered" if last_progress >= takeover_progress + self.recovery_delta else None
        if mode == "gate":
            # Hand back once the gate has a full short window AND would not fire
            if gate_fired_now is False and len(self.gate.history) >= self.gate.short_window:
                return "healthy"
        return None

    def _write_video(self, frames, labels, episode_id):
        """Write the episode video with a STUDENT/EXPERT banner per frame."""
        if not self.video_dir or not frames:
            return
        import cv2

        path = os.path.join(self.video_dir, f"gated_{episode_id}.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(self.video_fps), (w, h))
        for f, lab in zip(frames, labels):
            img = np.ascontiguousarray(np.asarray(f)[:, :, ::-1])  # RGB -> BGR
            color = (0, 0, 255) if lab == INTERVENTION_LABEL else (0, 180, 0)
            cv2.rectangle(img, (0, 0), (w, 18), color, -1)
            cv2.putText(img, "EXPERT" if lab == INTERVENTION_LABEL else "STUDENT",
                        (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
        writer.release()
        logger.info(f"Saved gated rollout video ({len(frames)} frames) to {path}")

    def rollout_episode(self, episode_id, store=True, require_success=False, require_intervention=False):
        """Run one gated episode. Returns a stats dict."""
        was_training = self.student.training
        self.student.eval()
        obs, _ = self.env.reset()
        obs = _extract0(obs)  # removes batched dimension

        self.scorer.reset(task=str(obs.get("prompt", "")))
        self.gate.reset()
        # Seed the reward-model history with the reset frame (like the wrapper's reset()).
        self.scorer.append(obs[self.reward_frame_key])

        student_st = {"chunk": None, "pos": 0}
        expert_st = {"chunk": None, "pos": 0}
        expert_active = False
        takeover_steps = 0
        takeover_progress = 0.0  # progress captured at takeover (for the 'progress' exit mode)
        expert_seg, seg_step = 0, 0  # expert-segment index + step within it
        steps, expert_steps, num_interventions = 0, 0, 0
        success, done = False, False
        pending = []
        progress_trace, gate_fires, gate_reasons = [], [], []
        handback_steps, handback_reasons = [], []
        video_frames, video_labels = [], []
        last_progress = 0.0

        while not done:
            cur = self._prep_obs(obs)

            if expert_active:
                action = self._policy_action(self.expert, obs, expert_st, self.expert_n_action_steps)
                mode, label = "EXPERT", INTERVENTION_LABEL
                expert_steps += 1
                takeover_steps += 1
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

            # ---- Robometer + gate (on the post-step frame) ----
            # Frames are appended every step, but the 4B-VLM is only queried every score_every steps
            self.scorer.append(next_obs[self.reward_frame_key])
            scored_now = steps % self.score_every == 0
            if scored_now:
                last_progress, _success_prob = self.scorer.score()  # scalar
            progress_trace.append(last_progress)

            # ---- Control transitions ----
            if not expert_active:
                if scored_now and steps >= self.warmup_steps and self.gate.update(last_progress):
                    gate_fires.append(steps)
                    gate_reasons.append(self.gate.last_trigger)  # "drop" | "plateau"
                    num_interventions += 1
                    expert_active = True
                    takeover_steps = 0
                    takeover_progress = last_progress
                    expert_st["chunk"] = None
                    self.gate.reset()  # fresh history for reward gate when switching control
                    expert_seg += 1
                    seg_step = 0
                    logger.info(f"  [gate] fired at step {steps} (progress={last_progress:.3f}, "
                                f"trigger={self.gate.last_trigger}) -> expert takeover (mode={self.expert_exit_mode})")
            else:
                gate_fired_now = self.gate.update(last_progress) if (scored_now and self.expert_exit_mode == "gate") else None
                reason = self._handback_reason(takeover_steps, last_progress, takeover_progress, gate_fired_now)
                if reason is not None:
                    handback_steps.append(steps)
                    handback_reasons.append(reason)
                    expert_active = False
                    student_st["chunk"] = None
                    self.gate.reset()
                    logger.info(f"  [gate] handback at step {steps} after {takeover_steps} expert steps "
                                f"(reason={reason})")

            # ---- Storage (episode buffered, flushed at the end) ----
            if store and self.online_buffer is not None and (not self.store_only_expert or label == INTERVENTION_LABEL):
                if self.store_only_expert:
                    ep_store, step_store = f"{episode_id}_e{expert_seg}", seg_step
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
                video_frames.append(next_obs[self.reward_frame_key])
                video_labels.append(label)

            obs = next_obs
            steps += 1

        # ---- Flush the whole episode ----
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
            handback_steps=handback_steps,
            handback_reasons=handback_reasons,
            progress_trace=progress_trace,
        )


# ---- Testing ----
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Watch reward-gated rollouts (student + expert + gate).")
    parser.add_argument("--student-dir", required=True, help="student pretraining run dir (has .hydra/config.yaml)")
    parser.add_argument("--expert-dir", required=False, help="expert (DP) pretraining run dir; not needed for --expert-type pi0")
    parser.add_argument("--student-type", choices=["dp", "pi0"], default="dp")
    parser.add_argument("--expert-type", choices=["dp", "pi0"], default="dp")
    parser.add_argument("--pi0-checkpoint", default=os.path.expanduser(
                        "~/.cache/openpi/openpi-assets/checkpoints/pi0_libero"),
                        help="'pi0' expert-type: openpi checkpoint dir (must contain 'libero' in the path)")
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--expert-checkpoint", default=None)
    parser.add_argument("--reward-model", default="jesbu1/robometer-4b-fft-libero")  # LIBERO-finetuned ckpt
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--expert-k", type=int, default=40, help="'fixed' exit mode: expert holds for this many steps")
    parser.add_argument("--expert-n-action-steps", type=int, default=5, help="action-chunking steps the expert executes")
    parser.add_argument("--student-n-action-steps", type=int, default=None,
                        help="override the student's replan interval (default: from its training config). "
                             "pi0 inference is expensive, so raise this when using --student-type pi0.")
    parser.add_argument("--expert-exit-mode", type=str, default="fixed",
                        choices=["fixed", "takeover", "progress", "gate"],
                        help="when the expert hands back: fixed k steps | until episode end | progress recovered | gate no longer fires")
    parser.add_argument("--recovery-delta", type=float, default=0.2, help="'progress' mode: rise above takeover progress to hand back")
    parser.add_argument("--min-expert-steps", type=int, default=10, help="'progress'/'gate': min hold before condition handback")
    parser.add_argument("--max-expert-steps", type=int, default=100, help="'progress'/'gate': hard cap (stuck-expert guardrail)")
    parser.add_argument("--short-window", type=int, default=30)
    parser.add_argument("--long-window", type=int, default=120)
    parser.add_argument("--method", type=str, default="spearman", choices=["spearman", "pearson", "naive"])
    parser.add_argument("--drop-threshold", type=float, default=-0.7)
    parser.add_argument("--plateau-threshold", type=float, default=0.1)
    parser.add_argument("--min-drop-magnitude", type=float, default=0.1,
                        help="absolute drop for correlation methods. Correlation is scale-free so a tiny wiggle fires like a real collapse.")
    parser.add_argument("--smoothing", type=float, default=0.0, help="EMA weight on history in [0,1); 0 = off")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--score-every", type=int, default=1)
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
        expert_k=args.expert_k,
        expert_exit_mode=args.expert_exit_mode,
        recovery_delta=args.recovery_delta,
        min_expert_steps=args.min_expert_steps,
        max_expert_steps=args.max_expert_steps,
        warmup_steps=args.warmup,
        score_every=args.score_every,
        video_dir=args.video_dir,
    )

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
            f"expert_steps={stats['expert_steps']} handbacks={stats['handback_reasons']}"
        )
        if args.video_dir:
            plot_progress_trace(
                stats,
                os.path.join(args.video_dir, f"progress_watch_{ep}.png"),
                title=f"episode {ep}: success={stats['success']}, interventions={stats['num_interventions']}",
            )
        # Re-dump every episode: a 50-episode job is long, don't lose it all to a late crash.
        dump_episode_stats(
            episodes,
            args.stats_json_dir,
            meta=dict(
                reward_model=args.reward_model,
                student_dir=args.student_dir, student_checkpoint=args.student_checkpoint,
                expert_type=args.expert_type,
                expert_dir=args.expert_dir, expert_checkpoint=args.expert_checkpoint,
                pi0_checkpoint=(args.pi0_checkpoint if args.expert_type == "pi0" else None),
                method=args.method, short_window=args.short_window, long_window=args.long_window,
                drop_threshold=args.drop_threshold, plateau_threshold=args.plateau_threshold,
                min_drop_magnitude=args.min_drop_magnitude, smoothing=args.smoothing,
                warmup=args.warmup, score_every=args.score_every, seed=args.seed,
                expert_exit_mode=args.expert_exit_mode,
            ),
        )

    n_succ = sum(bool(s["success"]) for s in episodes)
    n_int = sum(s["num_interventions"] for s in episodes)
    logger.info(f"DONE: {n_succ}/{len(episodes)} success, {n_int} total interventions")
    env.close()


if __name__ == "__main__":
    main()
