"""The Reward DAgger intervention gate."""

import math
import sys
from collections import deque
from typing import List, Optional

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr


def _compute_spearman(values) -> float:
    """Spearman correlation between step index and `values`."""
    n = len(values)
    if n < 2 or min(values) == max(values):
        return float("nan")
    res, _ = spearmanr(range(n), values)
    return float(res)


def _compute_pearson(values) -> float:
    """Pearson correlation between step index and `values`."""
    n = len(values)
    if n < 2 or min(values) == max(values):
        return float("nan")
    res, _ = pearsonr(range(n), values)
    return float(res)


class RewardGate:
    """Takes a scalar progress and decides when the expert should take over,"""

    def __init__(self,
        short_window=5,
        drop_threshold=-0.5,
        long_window=30,
        plateau_threshold=0.05,
        method="spearman",
        smoothing=0,
        min_drop_magnitude=0.0,
    ):
        self.short_window = short_window
        self.drop_threshold = drop_threshold
        self.long_window = long_window
        self.plateau_threshold = plateau_threshold
        # For the correlation methods; requires the absolute drop to be large enough to fire
        self.min_drop_magnitude = min_drop_magnitude
        self.history = deque(maxlen=long_window)
        self.method = method
        self.smoothing = smoothing
        self._ema = None
        self._corr = {"spearman": _compute_spearman, "pearson": _compute_pearson}
        self.last_trigger = None  # "drop" | "plateau" | None

        assert long_window >= short_window, "Long window should be greater than or equal to short window"
        assert 0 <= smoothing < 1, "Smoothing should be in range [0, 1)"
        if method == "naive":
            assert 0 <= drop_threshold <= 1, "Drop threshold has to be in range [0, 1] for naive gating"
            assert 0 <= plateau_threshold <= 1, "Plateau threshold has to be in range [0, 1] for naive gating"
        elif method in ["pearson", "spearman"]:
            assert -1 <= drop_threshold <= 1, "Drop threshold has to be in range [-1, 1] for pearson/spearman gating"
            assert -1 <= plateau_threshold <= 1, "Plateau threshold has to be in range [-1, 1] for pearson/spearman gating"
        else:
            raise ValueError("Unknown gating method; expected 'naive', 'pearson', or 'spearman'")

    def update(self, progress_value: float) -> bool:
        """Ingest one causal progress value. Return True if either trigger fires."""
        self._ema = progress_value if self._ema is None else self.smoothing * self._ema + (1 - self.smoothing) * progress_value
        self.history.append(self._ema)

        if len(self.history) < self.short_window:
            return False
        
        if self.method == "naive":
            should_drop = self._check_drop_naive()

            if len(self.history) >= self.long_window:
                should_plateau = self._check_plateau_naive()
            else:
                should_plateau = False
        else: 
            should_drop = self._check_drop()

            if len(self.history) >= self.long_window:
                should_plateau = self._check_plateau()
            else:
                should_plateau = False

        fired = should_drop or should_plateau
        self.last_trigger = ("drop" if should_drop else "plateau") if fired else None
        return fired

    def _check_drop(self) -> bool:
        """Correlation gate: progress is trending down over the short window,
        and the fall is large enough."""
        recent = list(self.history)[-self.short_window:]
        corr = self._corr[self.method](recent)
        if math.isnan(corr):
            return False
        if corr >= self.drop_threshold:
            return False
        if self.min_drop_magnitude > 0:
            if (max(recent) - recent[-1]) < self.min_drop_magnitude:
                return False
        return True

    def _check_plateau(self) -> bool:
        """Correlation gate: progress is not trending up over the long window."""
        recent = list(self.history)[-self.long_window:]
        corr = self._corr[self.method](recent)
        if math.isnan(corr):
            return True
        return corr < self.plateau_threshold

    def _check_drop_naive(self) -> bool:
        """Fire if progress dropped sharply from a recent peak."""
        recent = list(self.history)[-self.short_window:]
        peak = max(recent)
        current = recent[-1]
        drop = peak - current
        fired = drop >= self.drop_threshold
        return fired
    
    def _check_plateau_naive(self) -> bool:
        """Fire if progress has stalled over a long window."""
        if len(self.history) < self.long_window:
            return False
        old_val = list(self.history)[0]
        current = list(self.history)[-1]
        improvement = current - old_val
        fired = improvement <= self.plateau_threshold
        return fired
    
    def reset(self):
        """Call after an intervention to clear history."""
        self.history.clear()
        self._ema = None

    def plot_trace(self, stats, save_path: str, title: Optional[str] = None):
        plot_progress_trace(stats, save_path, title=title)


_ROBOMETER_SCRIPTS = "/scr/liryan/robometer_policy_learning/robometer/scripts"


class RobometerScorer:
    def __init__(self, model_path: str, device: str, max_frames=None):
        if _ROBOMETER_SCRIPTS not in sys.path:
            sys.path.insert(0, _ROBOMETER_SCRIPTS)
        from example_libero_robometer_wrapper import _RewardModelInferenceMixin

        self._model = _RewardModelInferenceMixin(
            model_path=model_path, device=device, max_frames=max_frames)
        self.task = ""
        self.frames: List[np.ndarray] = []
        self.episode_id = 0

    def reset(self, task: str = ""):
        self.task = str(task)
        self.frames = []
        self.episode_id += 1

    def append(self, frame: np.ndarray):
        self.frames.append(np.asarray(frame))

    def observe(self, obs, frame_key: str = "agentview_image"):
        self.append(obs[frame_key])

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
        rewards, success_probs = self._model._compute_rewards_batch([raw])
        return float(rewards[0]), float(success_probs[0])


def plot_progress_trace(stats, save_path: str, title: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    progress = stats["progress_trace"]
    fires: List[int] = stats["gate_fires"]
    reasons = stats["gate_reasons"]  # "drop" | "plateau"
    steps = len(progress)
    trigger_color = {"drop": "#d62728", "plateau": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(steps), progress, color="#1f77b4", lw=1.5, zorder=2)

    for f, reason in zip(fires, reasons):
        c = trigger_color.get(reason, "#7f7f7f")
        ax.axvspan(f, steps - 1, color=c, alpha=0.08, zorder=1)
        ax.axvline(f, color=c, lw=1.5, zorder=3)
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
    ], loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)