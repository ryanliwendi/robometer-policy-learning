#!/usr/bin/env python3
"""Reward-gated rollouts (no training).

Rolls out the student policy on the DINO eval env stack; every executed step, Robometer
scores the causal frame history and the RewardGate watches the progress series. When the
gate fires (sharp drop or plateau) the DP expert takes over for k steps, then control
returns to the student and the gate resets.

Design notes:
  * One env, both obs streams: the eval stack (LiberoPI0Wrapper + DinoEmbeddingWrapper)
    yields dino_embedding + observation/state for the actors AND keeps the raw
    observation/image frames, which we feed to a standalone RobometerScorer. This avoids
    stacking LiberoRobometerRewardWrapper (which expects the raw LIBERO env) under the
    policy wrappers.
  * The env is built with chunk_size=None; the worker manages receding-horizon chunking
    manually (like the HITL worker), so student and expert can replan on control switches.

Standalone usage:
    srun --gres=shard:8 --mem=32G --time=2:00:00 \
      uv run python scripts/gated_rollout_worker.py \
        --student-dir outputs/<bc_run> --expert-dir outputs/<dp_run> \
        --episodes 3 --expert-k 40 --video-dir gated_videos
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import sys

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, "/scr/liryan/robometer_policy_learning/robometer/scripts")
from example_libero_robometer_wrapper import _RewardModelInferenceMixin  # noqa: E402  # pyright: ignore[reportMissingImports]

from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device  # noqa: E402
from robometer_policy_learning.utils.reward_gate import RewardGate  # noqa: E402
from robometer_policy_learning.modules.transformer.modeling_transformer_actor import TransformerActor  # noqa: E402
from robometer_policy_learning.algorithms.dp.modeling_dp import DiffusionActor  # noqa: E402

# Intervention-label convention (matches the HITL buffers): 0=student rollout, 1=expert.
ROLLOUT_LABEL, INTERVENTION_LABEL = 0, 1


# ---------------------------------------------------------------------------------------
# Robometer scoring, decoupled from any env wrapper
# ---------------------------------------------------------------------------------------
class RobometerScorer(_RewardModelInferenceMixin):
    """Standalone causal Robometer scorer: feed one frame per executed step, get
    (progress, success_prob) for the history so far — the same numbers
    LiberoRobometerRewardWrapper puts into info, minus the env plumbing.

    raw_dict_to_sample subsamples the history to the model's max_frames internally,
    so we keep the full episode frame list (matches the non-vector wrapper).
    """

    def __init__(self, model_path: str, device: str, max_frames=None):
        super().__init__(model_path=model_path, device=device, max_frames=max_frames)
        self.task = ""
        self.frames = []
        self.episode_id = 0

    def reset(self, task: str):
        """Start a new episode: set the language instruction and clear the frame history."""
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


# ---------------------------------------------------------------------------------------
# Small helpers (mirrors hitl_utils_publish.py)
# ---------------------------------------------------------------------------------------
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


def load_actor(run_dir: str, device: str, checkpoint=None, trainable: bool = False):
    """Load a BaseActor from a pretraining run dir (checkpoints/<step>/...).

    trainable=False -> the deployable ``actor.pt`` (for DP that's the FROZEN EMA copy; fine
    for acting, not for training). trainable=True -> prefer ``online_actor.pt`` (DP's
    trainable network; falls back to actor.pt for BC, which saves only actor.pt) and
    re-enable grads, so an algorithm can continue training it.
    """
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
    names = ["online_actor.pt", "actor.pt"] if trainable else ["actor.pt"]
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


# ---------------------------------------------------------------------------------------
# The gated rollout worker
# ---------------------------------------------------------------------------------------
class GatedRolloutWorker:
    """HG-DAgger-style rollouts where a reward-model gate replaces the human.

    Per-step state machine:
      STUDENT: student acts (its receding-horizon chunk). The new frame is appended to the
               scorer; gate.update(progress) -> on fire, switch to EXPERT for expert_k steps.
      EXPERT : expert acts (its own chunk state; replans on takeover) for expert_k steps
               (label=1), then control returns to STUDENT (replans) and gate.reset().

    Robometer is still queried during expert control (for logging), but the gate is only
    updated while the student is in control; after handoff it restarts from empty history.
    """

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
        self.expert_k = int(expert_k) # number of steps expert executes
        self.warmup_steps = int(warmup_steps)  # student steps before the gate may fire
        self.score_every = max(1, int(score_every))  # robometer cadence (need >1 for real world experiment due to latency)
        self.store_only_expert = bool(store_only_expert)  # Whether to store only expert transitions in the buffer
        self.video_dir = video_dir
        self.video_fps = int(video_fps)
        if self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

    # -- policy action with receding-horizon chunking --
    def _policy_action(self, actor, obs_t, st, n_exec):
        """st is a mutable {"chunk" (action_sequence), "pos" (scalar current_index)} dict; set st["chunk"]=None to force a replan."""
        if st["chunk"] is None or st["pos"] >= len(st["chunk"]) or st["pos"] >= n_exec:
            with torch.inference_mode():
                pred, _ = actor.act(obs_t, deterministic=True)
            pred = pred.detach().cpu().numpy()
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
        """Run one gated episode. Returns a stats dict.

        Transitions are buffered and added to the online buffer ALL AT ONCE at episode end
        (so episode-level stats are known).
        require_success / require_intervention filter which episodes are kept.
        """
        was_training = self.student.training
        self.student.eval()  # no dropout/BN-updates while collecting; restored at episode end
        obs, _ = self.env.reset()
        obs = _extract0(obs)  # Removes the batch dimension from the obs

        self.scorer.reset(task=str(obs.get("prompt", "")))
        self.gate.reset()
        # Seed the reward-model history with the reset frame (like the wrapper's reset()).
        self.scorer.append(obs[self.reward_frame_key])

        student_st = {"chunk": None, "pos": 0}
        expert_st = {"chunk": None, "pos": 0}
        expert_left = 0          # >0 -> expert in control for this many more steps
        expert_seg, seg_step = 0, 0  # number of expert interventions and the step within the current expert takeover; used for storage
        steps, expert_steps, num_interventions = 0, 0, 0 
        success, done = False, False
        pending = []
        progress_trace, gate_fires = [], []
        video_frames, video_labels = [], []
        last_progress = 0.0

        while not done:
            cur = self._prep_obs(obs)
            obs_t = move_to_device(convert_to_tensor(cur), self.device)

            if expert_left > 0:
                action = self._policy_action(self.expert, obs_t, expert_st, self.expert_n_action_steps)
                mode, label = "EXPERT", INTERVENTION_LABEL
                expert_left -= 1
                expert_steps += 1
                if expert_left == 0:
                    # Handoff back to the student: replan from the corrected state, and give
                    # the gate a clean history (don't re-fire on pre-correction values).
                    student_st["chunk"] = None
                    self.gate.reset()
            else:
                action = self._policy_action(self.student, obs_t, student_st, self.student_n_action_steps)
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
            # Frames are appended EVERY step, but the 4B-VLM is only queried every score_every steps
            # The reward gate also updates every score_every steps
            self.scorer.append(next_obs[self.reward_frame_key])
            scored_now = steps % self.score_every == 0
            if scored_now:
                last_progress, _success_prob = self.scorer.score()  # scalar
            progress_trace.append(last_progress)
            if scored_now and label == ROLLOUT_LABEL and steps >= self.warmup_steps and expert_left == 0:
                if self.gate.update(last_progress):
                    gate_fires.append(steps)
                    num_interventions += 1
                    expert_left = self.expert_k
                    expert_st["chunk"] = None  # expert replans from the current state
                    expert_seg += 1
                    seg_step = 0
                    logger.info(f"  [gate] fired at step {steps} (progress={last_progress:.3f}) "
                                f"-> expert takes over for {self.expert_k} steps")

            # ---- storage (episode buffered, flushed at the end) ----
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

        # ---- flush the whole episode ----
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
            progress_trace=progress_trace,
        )


# --- Testing ---
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Watch reward-gated rollouts (student + expert + gate).")
    parser.add_argument("--student-dir", required=True, help="student pretraining run dir (has .hydra/config.yaml)")
    parser.add_argument("--expert-dir", required=True, help="expert (DP) pretraining run dir")
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--expert-checkpoint", default=None)
    parser.add_argument("--reward-model", default="robometer/Robometer-4B")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--expert-k", type=int, default=40)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--drop-threshold", type=float, default=0.15)
    parser.add_argument("--long-window", type=int, default=30)
    parser.add_argument("--plateau-threshold", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--score-every", type=int, default=1)
    parser.add_argument("--video-dir", default="gated_videos")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Env/model params come from the STUDENT's saved training config, so the env build,
    # chunking and DINO keys match what the student was trained with
    pre_cfg = OmegaConf.load(os.path.join(args.student_dir, ".hydra", "config.yaml"))
    dino_image_keys = list(OmegaConf.select(pre_cfg, "env.dino_image_keys", default=[]) or [])
    n_exec = int(OmegaConf.select(pre_cfg, "training.n_action_steps", default=10))

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

    student = load_actor(args.student_dir, device, args.student_checkpoint)
    expert = load_actor(args.expert_dir, device, args.expert_checkpoint)
    # Prefer the actor's own trained-with drop list (superset of the config's extra_keys_to_drop).
    remove_obs_keys = list(getattr(student, "remove_obs_keys", None)
                           or OmegaConf.select(pre_cfg, "env.extra_keys_to_drop", default=[]) or [])
    scorer = RobometerScorer(model_path=args.reward_model, device=device)
    gate = RewardGate(
        short_window=args.short_window,
        drop_threshold=args.drop_threshold,
        long_window=args.long_window,
        plateau_threshold=args.plateau_threshold,
    )

    worker = GatedRolloutWorker(
        env=env,
        student=student,
        expert=expert,
        scorer=scorer,
        gate=gate,
        online_buffer=None,  # watch-only: nothing stored
        device=device,
        action_dim=action_dim,
        remove_obs_keys=remove_obs_keys,
        student_n_action_steps=n_exec,
        expert_n_action_steps=5,
        expert_k=args.expert_k,
        warmup_steps=args.warmup,
        score_every=args.score_every,
        video_dir=args.video_dir,
    )

    for ep in range(args.episodes):
        stats = worker.rollout_episode(f"watch_{ep}", store=False)
        logger.info(
            f"episode {ep}: steps={stats['steps']} success={stats['success']} "
            f"interventions={stats['num_interventions']} (at steps {stats['gate_fires']}) "
            f"expert_steps={stats['expert_steps']}"
        )

    env.close()


if __name__ == "__main__":
    main()
