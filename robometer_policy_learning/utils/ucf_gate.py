"""The "Uncertainty Comes for Free" (UCF) intervention gate.

The reference repo only implements the uncertainty estimate and a threshold-calibration
script. The gate firing (patience, takeover) is our implementation.

Differences from the reference implementation:
  * The reference samples candidate *absolute* end-effector positions in a 5 cm ball around the
    current one. Our policy commands OSC *deltas*, and robosuite caps one delta step at exactly
    5 cm (``output_max`` 0.05 m at action 1.0), so the identical ball of reachable positions is
    the unit ball in normalized action space -- hence ``radius=1.0``.
  * The gripper/rotation fill-in values for the sampled actions comes from the action proposed 
    at the current state rather than the one executed at the previous step.
  * The GMM gets one initialization, instead of the reference's ten initializations for less latency.
    Also, on tested score fields the two pick the same number of modes 99% of the time.
    ``gmm_n_init`` restores the reference value.
"""

import os
import sys
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

from robometer_policy_learning.utils.diffdagger_gate import QuantileGate  # noqa: F401  (re-exported)

_VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "baselines")
if _VENDORED not in sys.path:
    sys.path.insert(0, _VENDORED)
from vendored.ucf_gmm import compute_uncertainty  # noqa: E402

N_SAMPLES = 100
RADIUS = 1.0  # `radius` is in normalized action units; see module docstring
GMM_ALPHA = 0.1  # weighting between inter-node and intra-node variance
MODE_CANDIDATES = (1, 2, 3, 4, 5)


def _limit_threads():
    """Hold BLAS/OpenMP to one thread while a GMM is fitted."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=1)


def sample_in_ball(n_samples: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Use rejection sampling to sample points inside a radius."""
    out = np.empty((n_samples, 3), dtype=np.float32)
    filled = 0
    while filled < n_samples:
        cand = rng.uniform(-radius, radius, size=(n_samples, 3))
        keep = cand[np.linalg.norm(cand, axis=1) <= radius]
        take = min(len(keep), n_samples - filled)
        out[filled:filled + take] = keep[:take]
        filled += take
    return out


def action_cloud(action_norm: Optional[np.ndarray], n_samples: int, radius: float,
                 action_dim: int, rng: np.random.Generator) -> np.ndarray:
    """(n_samples, action_dim) candidate actions in the policy's normalized action space.
    Used as samples for the diffusion policy to denoise. 
    """
    out = np.zeros((n_samples, action_dim), dtype=np.float32)
    out[:, :3] = sample_in_ball(n_samples, radius, rng)
    gripper = -1.0
    if action_norm is not None:
        gripper = float(np.asarray(action_norm).reshape(-1)[-1])
    out[:, -1] = gripper
    return np.clip(out, -1.0, 1.0)


class VectorFieldScorer:
    """UCF's uncertainty scorer: fits a GMM over the denoising vectors, eps_theta(o, a_i, k=0),
    for a cloud of actions a_i."""

    def __init__(
        self,
        actor,
        n_samples: int = N_SAMPLES,
        radius: float = RADIUS,
        alpha: float = GMM_ALPHA,
        mode_candidates: Sequence[int] = MODE_CANDIDATES,
        gmm_n_init: int = 1,
        random_state: int = 42,
        device: Optional[torch.device] = None,
    ):
        if not all(hasattr(actor, a) for a in ("predict_noise", "encode_obs", "horizon")):
            raise TypeError(f"The student must be a DiffusionActor; got {type(actor).__name__}.")
        self.actor = actor
        self.n_samples = int(n_samples)
        self.radius = float(radius)
        self.alpha = float(alpha)
        self.mode_candidates = tuple(int(k) for k in mode_candidates)
        self.gmm_n_init = int(gmm_n_init)
        self.random_state = int(random_state)
        self.device = device or next(actor.parameters()).device
        self.action_dim = int(actor.action_dim)
        self.horizon = int(actor.horizon)
        self.rng = np.random.default_rng(self.random_state)

    @torch.no_grad()
    def _noise_field(self, global_cond: torch.Tensor, actions_norm: torch.Tensor) -> np.ndarray:
        """(k, D) conditioning observation state and (k, N, A) action candidates -> 
        (k, N, 3) predicted-noise vectors for position. D is the policy's encoded representation of a state; 
        k is the number of states, and N is the number of action samples from each state."""
        k, n = actions_norm.shape[0], actions_norm.shape[1]
        gc = global_cond.repeat_interleave(n, dim=0)                       # (k*N, D)
        act = actions_norm.reshape(k * n, 1, self.action_dim)
        if self.horizon > 1:  # the reference repeats the single-step action across the action chunking horizon
            act = act.repeat(1, self.horizon, 1)
        timesteps = torch.zeros(k * n, dtype=torch.long, device=self.device)
        eps = self.actor.predict_noise(act.contiguous(), timesteps, gc)    # (k*N, H, A)
        return eps[:, 0, :3].reshape(k, n, 3).float().cpu().numpy()

    def _uncertainty(self, vectors: np.ndarray) -> float:
        with _limit_threads():
            value, _ = compute_uncertainty(vectors, alpha=self.alpha,
                                           mode_candidates=self.mode_candidates,
                                           random_state=self.random_state, n_init=self.gmm_n_init)
        return float(value)

    @torch.no_grad()
    def score(self, obs: Union[dict, torch.Tensor],
              action_norm: Optional[np.ndarray] = None) -> float:
        global_cond = self.actor.encode_obs(obs)  # (B, D), B == 1 at rollout
        if global_cond.shape[0] != 1:
            raise ValueError(f"VectorFieldScorer expects batch size 1, got {global_cond.shape[0]}")
        cloud = action_cloud(action_norm, self.n_samples, self.radius, self.action_dim, self.rng)
        actions = torch.from_numpy(cloud).to(self.device).unsqueeze(0)     # (1, N, A)
        return self._uncertainty(self._noise_field(global_cond, actions)[0])

    @torch.no_grad()
    def score_features(self, global_cond, actions_norm, max_rows: int = 8192) -> np.ndarray:
        """Uncertainty scores for batched observations and actions."""
        gc = torch.as_tensor(global_cond).to(self.device).float()
        acts = np.asarray(actions_norm, dtype=np.float32)
        chunk = max(1, int(max_rows) // self.n_samples)
        out = []
        for i in range(0, gc.shape[0], chunk):
            clouds = np.stack([action_cloud(a, self.n_samples, self.radius, self.action_dim,
                                            self.rng)
                               for a in acts[i:i + chunk]])                # (k, N, A)
            field = self._noise_field(gc[i:i + chunk], torch.from_numpy(clouds).to(self.device))
            out.extend(self._uncertainty(v) for v in field)
        return np.asarray(out, dtype=np.float64)

    @torch.no_grad()
    def calibrate_from_buffer(self, buffer, batch_size: int = 64, num_samples: int = 1024,
                              max_rows: int = 8192) -> np.ndarray:
        """The uncertainty at each of `num_samples` states drawn from `buffer`, as a flat array.
        Only returns the uncertainty scores, does not recalibrate thresholds. 
        Used for gate frontier analysis.
        """
        values = []
        # Calibration runs between training rounds, so the student is mid-training
        was_training = self.actor.training
        self.actor.eval()
        try:
            while len(values) < num_samples:
                batch = buffer.sample(int(batch_size), device=self.device)
                if not batch or len(batch.get("action", [])) == 0:
                    break
                actions = torch.as_tensor(batch["action"]).to(self.device).float()
                if actions.dim() == 3:  # (B, H, A) chunk -> the action taken at the stored state
                    actions = actions[:, 0, :]
                global_cond = self.actor.encode_obs(batch["obs"])
                values.extend(self.score_features(global_cond, actions.cpu().numpy(), max_rows))
        finally:
            if was_training:
                self.actor.train()
        return np.asarray(values[:num_samples], dtype=np.float64)

    def calibrate_from_algo(self, algo, num_samples: int = 1024, max_rows: int = 8192) -> np.ndarray:
        """``calibrate_from_buffer`` over the algo's live training buffer (demos + corrections)."""
        return self.calibrate_from_buffer(algo.buffer, algo.batch_size, num_samples, max_rows)


class UCFScorer:
    """Rollout-worker interface for the vector-field uncertainty."""

    def __init__(self, actor, remove_obs_keys=None, lowdim_stats=None, action_min=None,
                 action_max=None, device=None, **scorer_kwargs):
        self.actor = actor
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.lowdim_stats = lowdim_stats or {}
        self.device = device or next(actor.parameters()).device
        self.inner = VectorFieldScorer(actor, device=self.device, **scorer_kwargs)
        self.task = ""
        self.episode_id = 0
        self._last_obs = None
        self._last_action_norm = None
        self._record = None  # a list while recording a calibration rollout

    def reset(self, task: str = ""):
        # The worker already resets `_record` at the start of every episode
        self.task = str(task)
        self.episode_id += 1
        self._last_obs = None
        self._last_action_norm = None

    def start_recording(self):
        self._record = []

    def take_recording(self) -> np.ndarray:
        """The per-step uncertainties of the episode just recorded, and stop recording."""
        rows, self._record = self._record, None
        return np.asarray(rows or [], dtype=np.float64)

    def observe(self, obs, frame_key: Optional[str] = None):
        self._last_obs = obs

    def _actor_obs(self, obs):
        """Process the raw rollout obs into the form the buffer and actor expects."""
        from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

        prepped = {}
        for key, value in obs.items():
            if key in self.remove_obs_keys:
                continue
            if key in self.lowdim_stats:
                stats = self.lowdim_stats[key]
                value = ((np.asarray(value, dtype=np.float32) - stats["mean"])
                         / stats["std"]).astype(np.float32)
            prepped[key] = value
        return move_to_device(convert_to_tensor(prepped), self.device)

    @torch.no_grad()
    def score(self, action=None):
        """Uncertainty at the last observed state. ``action`` is in env space."""
        if self._last_obs is None:
            return 0.0, 0.0
        if action is not None:
            a = torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(-1),
                                device=self.device)
            self._last_action_norm = self.actor.normalize_action(a).cpu().numpy()
        value = float(self.inner.score(self._actor_obs(self._last_obs), self._last_action_norm))
        if self._record is not None:
            self._record.append(value)
        return value, 0.0

    def score_action(self, action):
        """Score the exact student action proposed for the current state (worker entry point)."""
        return self.score(action=action)


def build_gate(alpha: float = 0.95, patience: int = 1,
               patience_window: Optional[int] = None) -> QuantileGate:
    """UCF fires on a quantile of the training uncertainties"""
    return QuantileGate(patience=patience, patience_window=patience_window, alpha=alpha,
                        name="ucf", signal_label="uncertainty")


def set_threshold_from_buffer(gate: QuantileGate, scorer: UCFScorer, algo,
                            num_samples: int = 1024,
                            max_rows: int = 8192) -> Optional[Dict[str, Any]]:
    """Set the gate's threshold to the alpha-quantile of the training buffer's uncertainty.
    A cheap variant of ``set_threshold_from_rollouts`` without calibrating on extra rollout
    episodes, but now the data is no longer held-out.

    Note that the buffer does not contain failures, so the threshold is calibrated only on successes
    and behaviors might differ from ``set_threshold_from_rollouts``.
    """
    values = scorer.inner.calibrate_from_algo(algo, num_samples=num_samples, max_rows=max_rows)
    if len(values) == 0:
        return None
    return dict(gate.recalibrate(values), source="buffer")


def set_threshold_from_rollouts(gate: QuantileGate, scorer: UCFScorer, worker, *,
                              n_rollouts: int = 50, tag: str = "ucf",
                              successes_only: bool = False) -> Optional[Dict[str, Any]]:
    """Set the gate's threshold from fresh rollouts of the current policy.

    ``successes_only`` is off because the reference pools every episode regardless of outcome. That
    is harmless for the reference's policy, which solves its task essentially every time, and much
    less so for a student at 40-60%: its failures then help set the bar for what counts as normal,
    which raises the threshold exactly where the gate is supposed to fire. Turn it on to give the
    baseline SAFE's successful-rollouts-only convention instead.
    """
    from robometer_policy_learning.utils.reward_gate import collect_solo_rollouts

    episodes = collect_solo_rollouts(worker, scorer, n_rollouts, tag)
    kept = [e for e in episodes if len(e["trace"]) and (e["success"] or not successes_only)]
    if not kept:
        return None
    values = np.concatenate([e["trace"] for e in kept])
    n_succ = sum(e["success"] for e in episodes)
    return dict(gate.recalibrate(values), source="rollouts", n_rollouts=len(episodes),
                n_success=int(n_succ), n_calib_episodes=len(kept),
                solo_steps=int(sum(e["steps"] for e in episodes)))
