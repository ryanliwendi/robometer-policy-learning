"""The Diff-DAgger intervention gate.

Differences from the original implementation: 
    * The student is fine-tuned continuously; the reference re-instantiates the policy and
    trains from scratch each round.
    * No timeout-forced expert intervention; the reference hands control to the expert at the
    time limit even when the gate doesn't fire, and lets it run past the limit.
    * No image augmentation during calibration since the student takes in frozen DINOv2 embeddings.
"""

import os
import sys
from collections import deque
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

_VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "baselines")
if _VENDORED not in sys.path:
    sys.path.insert(0, _VENDORED)
from vendored.diffdagger_cdf import CDF  # noqa: E402 


def _as_action_chunk(actions: torch.Tensor) -> torch.Tensor:
    """(B, A) -> (B, 1, A)"""
    if actions.dim() == 2:
        return actions.unsqueeze(1)
    if actions.dim() != 3:
        raise ValueError(f"Unexpected action shape for the diffusion loss: {tuple(actions.shape)}")
    return actions


class DiffusionLossScorer:
    """Diff-DAgger's per-step uncertainty: E_{eps,t}[ ||target - f_theta(o, a_noised, t)||^2 ]."""

    def __init__(
        self,
        actor,
        batch_multiplier: int = 1, # Number of samples we draw from each noise level in a single batch
        num_per_batch: int = 1, # Number of independent noise draws to average the loss over
        device: Optional[torch.device] = None,
    ):
        if not all(hasattr(actor, a) for a in ("scheduler", "predict_noise", "encode_obs")):
            raise TypeError(f"The student must be a DiffusionActor; got {type(actor).__name__}.")
        self.actor = actor
        self.batch_multiplier = int(batch_multiplier)
        self.num_per_batch = int(num_per_batch)
        self.device = device or next(actor.parameters()).device
        self.num_train_timesteps = int(actor.scheduler.config.num_train_timesteps)  # number of noise levels diffusion model was trained on
        self.prediction_type = getattr(actor.scheduler.config, "prediction_type", "epsilon")

    @torch.no_grad()
    def _avg_loss(self, global_cond: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-item diffusion loss averaged over every timestep and ``num_per_batch`` noise draws.
        ``global_cond`` is (k, D) and ``actions`` (k, H, A)."""
        k = global_cond.shape[0]  # batch size
        reps = self.num_train_timesteps * self.batch_multiplier
        gc = global_cond.repeat_interleave(reps, dim=0)  # (k*reps, D)
        act = actions.to(self.device).repeat_interleave(reps, dim=0).contiguous()
        timesteps = (torch.arange(reps, device=self.device).repeat(k)
                     % self.num_train_timesteps).long()

        total = torch.zeros(k, device=self.device, dtype=torch.float32)
        for _ in range(self.num_per_batch):
            noise = torch.randn_like(act)
            noisy = self.actor.scheduler.add_noise(act, noise, timesteps)
            pred = self.actor.predict_noise(noisy, timesteps, gc)
            target = self._target(act, noise, timesteps)
            per = F.mse_loss(pred, target, reduction="none")  # (k*reps, H, A)
            total += per.reshape(k, reps, -1).mean(dim=(1, 2))
        return total / self.num_per_batch

    def _target(self, act: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "v_prediction":
            return self.actor.scheduler.get_velocity(act, noise, timesteps)
        return act  # "sample"

    @torch.no_grad()
    def score(
        self,
        obs: Union[dict, torch.Tensor],
        action_norm: Optional[torch.Tensor] = None,
    ) -> float:
        """Expected diffusion loss at ``obs``, evaluated at the policy's own action."""
        global_cond = self.actor.encode_obs(obs)  # (B, D), B == 1 at rollout
        if global_cond.shape[0] != 1:
            raise ValueError(f"DiffusionLossScorer expects batch size 1, got {global_cond.shape[0]}")

        if action_norm is None:
            action_norm = self.actor.sample_actions(obs)
        if action_norm.dim() == 2:  # (B, A) -> (B, 1, A)
            action_norm = action_norm.unsqueeze(1)
        return float(self._avg_loss(global_cond, action_norm)[0].item())

    @torch.no_grad()
    def calibrate_from_buffer(self, buffer, batch_size: int = 64, num_samples: int = 1024,
                              max_rows: int = 8192) -> np.ndarray:
        """Averaged diffusion losses over a training buffer, one value per datapoint.

        ``max_rows`` caps the number of rows in the expanded batch (each datapoint becomes ``T * batch_multiplier``
        rows) to avoid OOM errors when we batch multiple datapoints."""
        reps = self.num_train_timesteps * self.batch_multiplier
        chunk = max(1, int(max_rows) // reps)
        losses = []
        while len(losses) < num_samples:
            batch = buffer.sample(int(batch_size), device=self.device)
            if not batch or len(batch.get("action", [])) == 0:
                break
            actions = _as_action_chunk(torch.as_tensor(batch["action"]).to(self.device)).float()
            global_cond = self.actor.encode_obs(batch["obs"])

            for i in range(0, global_cond.shape[0], chunk):
                if len(losses) >= num_samples:
                    break
                losses.extend(self._avg_loss(global_cond[i:i + chunk],
                                             actions[i:i + chunk]).cpu().numpy().tolist())
        return np.asarray(losses[:num_samples], dtype=np.float64)

    def calibrate_from_algo(self, algo, num_samples: int = 1024, max_rows: int = 8192) -> np.ndarray:
        """``calibrate_from_buffer`` against the algo's own training buffer."""
        return self.calibrate_from_buffer(algo.buffer, algo.batch_size, num_samples, max_rows)


class DiffDaggerScorer:
    def __init__(self, actor, remove_obs_keys=None, device=None, **scorer_kwargs):
        self.actor = actor
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.device = device or next(actor.parameters()).device
        self.inner = DiffusionLossScorer(actor, device=self.device, **scorer_kwargs)
        self.task = ""
        self.episode_id = 0
        self._last_obs = None

    def reset(self, task: str = ""):
        self.task = str(task)
        self.episode_id += 1
        self._last_obs = None

    def observe(self, obs, frame_key: Optional[str] = None):
        self._last_obs = obs  # stores the raw obs

    def score(self):
        from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

        if self._last_obs is None:
            return 0.0, 0.0
        prepped = {k: v for k, v in self._last_obs.items() if k not in self.remove_obs_keys}
        actor_obs = move_to_device(convert_to_tensor(prepped), self.device)
        return float(self.inner.score(actor_obs)), 0.0


def _get_quantile(cdf: CDF, alpha: float) -> float:
    n = len(cdf.sorted_data)
    return float(cdf.get_quantile(min(float(alpha), (n - 1) / n)))


def quantile_threshold(losses: np.ndarray, alpha: float) -> float:
    """alpha-quantile of the training-loss distribution (Diff-DAgger's ``CDF.get_quantile``)."""
    if len(losses) == 0:
        raise ValueError("no calibration losses collected")
    return _get_quantile(CDF(np.asarray(losses, dtype=np.float64)), alpha)


class QuantileGate:
    def __init__(
        self,
        threshold: float = float("inf"),
        patience: int = 1,
        patience_window: Optional[int] = None,
        alpha: float = 0.99,
        name: str = "diffdagger",
        **_ignored: Any,
    ):
        # LogpZO fires on the same rule (score over the alpha-quantile of the training scores),
        # so it reuses this class.
        self.name = str(name)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.cdf: Optional[CDF] = None
        self.patience = int(patience)
        self.patience_window = int(patience_window if patience_window is not None else patience)
        if self.patience_window < self.patience:
            raise ValueError("patience_window must be >= patience")
        self.deque = deque([], maxlen=self.patience_window)
        self.history = deque(maxlen=self.patience_window)
        self.short_window = self.patience_window
        self.last_trigger = None
        self.last_cdf: Optional[float] = None

    def recalibrate(self, losses: np.ndarray, alpha: Optional[float] = None) -> Dict[str, Any]:
        """Refit the training-loss CDF and move the threshold to its alpha-quantile.
        Called once before collection starts and again after every retrain."""
        if alpha is not None:
            self.alpha = float(alpha)
        losses = np.asarray(losses, dtype=np.float64)
        if losses.size == 0:
            raise ValueError("no calibration losses collected")
        self.cdf = CDF(losses)
        self.threshold = _get_quantile(self.cdf, self.alpha)
        return dict(threshold=self.threshold, alpha=self.alpha, n=int(losses.size),
                    loss_mean=float(losses.mean()), loss_max=float(losses.max()))

    def update(self, value: float) -> bool:
        value = float(value)
        self.history.append(value)
        self.last_cdf = float(self.cdf(value)) if self.cdf is not None else None
        self.deque.append(value > self.threshold)
        fired = sum(self.deque) >= self.patience
        self.last_trigger = "diffusion_loss" if fired else None
        return fired

    def reset(self):
        self.deque.clear()
        self.history.clear()

    def describe(self) -> Dict[str, Any]:
        return dict(gate=self.name, threshold=self.threshold, alpha=self.alpha,
                    calibrated=self.cdf is not None,
                    patience=self.patience, patience_window=self.patience_window)

    def plot_trace(self, stats: Dict[str, Any], save_path: str, title: Optional[str] = None):
        """Plots the student's own diffusion loss against the calibrated threshold."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        losses = stats["progress_trace"]  # Progress_trace refers to the losses
        steps = len(losses)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(range(steps), losses, color="#1f77b4", lw=1.5, zorder=2, label="diffusion loss")
        ax.axhline(self.threshold, color="#2ca02c", lw=1.2, ls="--", zorder=3,
                   label=f"threshold (alpha={self.alpha:g})")
        for f in stats["gate_fires"]:
            ax.axvspan(f, steps - 1, color="#d62728", alpha=0.08, zorder=1)
            ax.axvline(f, color="#d62728", lw=1.5, zorder=4)
            ax.annotate(f"takeover\n@{f}", xy=(f, max(losses, default=1.0)), xytext=(2, -2),
                        textcoords="offset points", ha="left", va="top", fontsize=7,
                        color="#d62728")

        ax.set_xlabel("environment step")
        ax.set_ylabel("diffusion loss")
        ax.set_title(title or f"success={stats['success']}  "
                              f"interventions={stats['num_interventions']}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
