"""Robometer gate: the scorer and the plot that are specific to the Robometer progress signal.

Everything here is reward-DAgger only. The rollout worker stays gate-agnostic and imports this
lazily, so a ThriftyDAgger or HG-DAgger run never loads the 4B reward model's package at all.

The gate rule itself lives in ``reward_gate.RewardGate``; this module is the scorer that feeds it
and the diagnostic plot that reads its output.
"""

import sys
from typing import List, Optional

import numpy as np

_ROBOMETER_SCRIPTS = "/scr/liryan/robometer_policy_learning/robometer/scripts"
if _ROBOMETER_SCRIPTS not in sys.path:
    sys.path.insert(0, _ROBOMETER_SCRIPTS)
from example_libero_robometer_wrapper import _RewardModelInferenceMixin  # noqa: E402


class RobometerScorer(_RewardModelInferenceMixin):
    """Causal Robometer scorer: one frame per executed step -> (progress, success_prob).

    ``raw_dict_to_sample`` subsamples the history to the model's max_frames internally, so the
    full episode frame list is kept.
    """

    def __init__(self, model_path: str, device: str, max_frames=None):
        super().__init__(model_path=model_path, device=device, max_frames=max_frames)
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
        """Entry point shared with the baseline scorers, which may need other data from obs."""
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
        rewards, success_probs = self._compute_rewards_batch([raw])
        return float(rewards[0]), float(success_probs[0])


def plot_progress_trace(stats, save_path: str, title: Optional[str] = None):
    """Plot one episode's Robometer progress curve and the takeover span.

    Robometer-only: the y-axis is progress in [0, 1] and the legend names the drop/plateau
    triggers. The other gates store a different signal entirely (a 0/1 human decision, an
    unbounded diffusion loss, a novelty/safety pair), so drawing them here would be wrong --
    the worker guards this behind ``plot_progress``.
    """
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

    # At most one fire, and the expert holds control from there to the end of the episode.
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
