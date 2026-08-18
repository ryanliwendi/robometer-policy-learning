"""The LogpZO intervention gate.

Differences from the original implementation:
    * The features are the student's own observation encoding (``encode_obs``).
    * Adapts the original method with DAgger; The flow and thresholds is refit from scratch each round 
      from rollouts of the current policy.
"""

from typing import Any, Dict, List, Optional

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from robometer_policy_learning.modules.diffusion.unet import ConditionalUnet1D

# The reference's UNet width
DOWN_DIMS = (256, 512, 1024)
# Channel count the feature vector is folded into. 32 divides our 768-d encoding exactly.
IN_DIM = 32


def adjust_xshape(x: torch.Tensor, in_dim: int) -> torch.Tensor:
    """Fold a flat feature vector (N, D) into (N, D/in_dim, in_dim) so the 1D UNet can read it."""
    remain = x.shape[1] % in_dim
    if remain:
        x = torch.cat([x, torch.zeros(x.shape[0], in_dim - remain, device=x.device, dtype=x.dtype)],
                      dim=1)
    return x.reshape(x.shape[0], -1, in_dim)


def _build_unet(in_dim: int) -> ConditionalUnet1D:
    return ConditionalUnet1D(action_dim=in_dim, global_cond_dim=0, diffusion_step_embed_dim=128,
                             down_dims=DOWN_DIMS, kernel_size=5, n_groups=8)


class LogpZOModel(nn.Module):
    """Flow matching from the training-feature distribution to noise, used as a density score."""

    def __init__(self, feat_dim: int, in_dim: int = IN_DIM):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.in_dim = int(in_dim)
        self.net = _build_unet(self.in_dim)

    def _empty_cond(self, n: int, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(n, 0, device=ref.device, dtype=ref.dtype)

    def flow_loss(self, feat: torch.Tensor) -> torch.Tensor:
        """Train the net to carry a real feature to noise in a straight line.

        x0 is the real feature, x1 is noise, and the true velocity is the difference. 
        """
        x0 = adjust_xshape(feat, self.in_dim)
        x1 = torch.randn_like(x0)
        v_true = x1 - x0
        t = torch.rand(len(x0), device=x0.device, dtype=x0.dtype).view(-1, 1, 1)
        x_now = x0 + t * v_true
        # The UNet takes discrete diffusion steps, so the continuous time is scaled to 0..100.
        v_hat = self.net(x_now, (t.view(-1) * 100).long(), self._empty_cond(len(x0), x0))
        return (v_hat - v_true).pow(2).mean()

    @torch.no_grad()
    def score(self, feat: torch.Tensor) -> torch.Tensor:
        """Squared length of where the flow sends this feature, at time 0. One score per row."""
        x = adjust_xshape(feat, self.in_dim)
        t = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        v = self.net(x, t.long(), self._empty_cond(len(x), x))
        return (x + v).reshape(len(x), -1).pow(2).sum(dim=-1)


def fit_flow(model: LogpZOModel, opt: torch.optim.Optimizer, get_features, total: int,
             grad_steps: int, batch_size: int = 256, seed: int = 0,
             val_fraction: float = 0.1) -> Dict[str, float]:
    """Fit the flow on `total` rows, where `get_features(idx)` returns the rows at those indices.

    A slice of rows is held out and never trained on.
    """
    if total == 0:
        return dict(train_loss=float("nan"), val_loss=float("nan"), n_transitions=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    n_val = int(total * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    _features = get_features
    model.train()
    losses = []
    for _ in range(int(grad_steps)):
        idx = train_idx[rng.integers(0, len(train_idx), size=min(batch_size, len(train_idx)))]
        loss = model.flow_loss(_features(idx))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    val_loss = float("nan")
    if len(val_idx):
        with torch.no_grad():
            chunk = val_idx[:min(len(val_idx), 4 * batch_size)]
            val_loss = float(model.flow_loss(_features(chunk)).item())

    return dict(train_loss=float(np.mean(losses[-50:])) if losses else float("nan"),
                val_loss=val_loss, n_transitions=int(total), n_val=int(len(val_idx)))


def fit_flow_on_episodes(model: LogpZOModel, opt: torch.optim.Optimizer, episode_feats,
                         grad_steps: int, device, batch_size: int = 256, seed: int = 0,
                         val_fraction: float = 0.1) -> Dict[str, float]:
    """``fit_flow`` on every step of the given episodes, whose features are already encoded."""
    x = torch.from_numpy(np.concatenate(list(episode_feats))).to(device)
    try:
        return fit_flow(model, opt,
                        lambda idx: x[torch.from_numpy(np.asarray(idx)).to(device)],
                        len(x), grad_steps, batch_size=batch_size, seed=seed,
                        val_fraction=val_fraction)
    finally:
        del x


@torch.no_grad()
def score_features(model: LogpZOModel, feat: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """LogpZO score for every step of one episode."""
    model.eval()
    out = []
    for s in range(0, len(feat), batch_size):
        x = torch.from_numpy(feat[s:s + batch_size]).to(device)
        out.append(model.score(x).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def pad_to(trace, T: int) -> np.ndarray:
    """Cut or stretch one trace to T steps. Short episodes are stretched by repeating their last
    value, which is what SAFE does before building a band out of episodes of different lengths."""
    t = np.asarray(trace, dtype=np.float64)
    return t[:T] if len(t) >= T else np.pad(t, (0, T - len(t)), mode="edge")


def conformal_band(train_traces: np.ndarray, calib_traces: np.ndarray,
                   alphas: np.ndarray) -> np.ndarray:
    """Returns a conformal threshold for each time step."""
    eps = 1e-8
    train = np.asarray(train_traces, dtype=np.float64)
    calib = np.asarray(calib_traces, dtype=np.float64)
    if train.ndim != 2 or calib.ndim != 2 or train.shape[1] != calib.shape[1]:
        raise ValueError(f"need two (n, T) arrays of equal T; got {train.shape} and {calib.shape}")

    mean = train.mean(axis=0, keepdims=True)             # (1, T)
    dev = np.abs(train - mean)                           # (n_train, T)
    peak = dev.max(axis=1)                               # (n_train,)
    n_train = len(train)

    bands = []
    for a in alphas:

        k = int(np.ceil((n_train + 1) * (1.0 - float(a))))
        if k > n_train:
            mod = dev.max(axis=0, keepdims=True) + eps
        else:
            gamma = np.sort(peak)[k - 1]
            mod = dev[peak <= gamma].max(axis=0, keepdims=True) + eps
        width = np.quantile(((calib - mean) / mod).max(axis=1), 1.0 - float(a))
        bands.append((mean + width * mod)[0])
    return np.asarray(bands, dtype=np.float64)


class LogpZOScorer:
    """Per-step score for the rollout worker. Same interface as ``DiffDaggerScorer``.

    It can also just record the features it encodes and skip the flow entirely, which is what the
    rollouts that LogpZO is rebuilt from need: at that point the flow for this round does not exist
    yet, so there is no score to compute, only features to keep.
    """

    def __init__(self, algo, model: LogpZOModel, remove_obs_keys=None, device=None):
        self.algo = algo
        self.model = model
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.device = device or next(model.parameters()).device
        self.task = ""
        self.episode_id = 0
        self._last_obs = None
        self._record = None  # a list while recording, None otherwise

    def reset(self, task: str = ""):
        # Deliberately does not touch _record: the worker calls this at the start of every episode,
        # and recording is switched on from outside, one episode at a time.
        self.task = str(task)
        self.episode_id += 1
        self._last_obs = None

    def observe(self, obs, frame_key: Optional[str] = None):
        self._last_obs = obs

    def start_recording(self):
        self._record = []

    def take_recording(self) -> np.ndarray:
        """The (T, D) features of the episode just recorded, and stop recording."""
        rows, self._record = self._record, None
        return (np.concatenate(rows) if rows else np.empty((0, 0), dtype=np.float32))

    @torch.no_grad()
    def score(self):
        from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

        if self._last_obs is None:
            return 0.0, 0.0
        prepped = {k: v for k, v in self._last_obs.items() if k not in self.remove_obs_keys}
        actor_obs = move_to_device(convert_to_tensor(prepped), self.device)
        feat = self.algo.actor.encode_obs(actor_obs).float()
        if self._record is not None:
            self._record.append(feat.cpu().numpy())
            return 0.0, 0.0
        return float(self.model.score(feat).reshape(-1)[0]), 0.0


def collect_solo_rollouts(worker, scorer: LogpZOScorer, n_rollouts: int, tag: str) -> List[dict]:
    """Run `n_rollouts` ungated episodes of the current student, keeping each one's features.

    This is LogpZO's running cost. Nothing is stored in the training buffer -- the student is alone,
    so there is no expert action to learn from; the episodes buy the density model and nothing else.
    """
    from robometer_policy_learning.utils.reward_gate import NeverGate

    saved = (worker.gate, worker.video_dir, worker.dump_dir)
    worker.gate, worker.video_dir, worker.dump_dir = NeverGate(), None, None
    episodes = []
    try:
        for i in range(int(n_rollouts)):
            scorer.start_recording()
            stats = worker.rollout_episode(f"{tag}_solo{i}", store=False)
            episodes.append(dict(feat=scorer.take_recording(), success=bool(stats["success"]),
                                 steps=int(stats["steps"])))
    finally:
        scorer._record = None
        worker.gate, worker.video_dir, worker.dump_dir = saved
    return episodes


def refit_from_rollouts(worker, scorer: LogpZOScorer, feat_dim: int, horizon: int, *,
                        n_rollouts: int, alpha: float, grad_steps: int, lr: float, device,
                        tag: str = "", seed: int = 0, batch_size: int = 256):
    """One round of LogpZO: collect solo rollouts, fit the flow, build the band.

    The successful episodes are split in half -- one half fits the flow, the other is scored by it
    and becomes the band -- so the band is never built from episodes the flow trained on. Returns
    ``None`` if too few of the rollouts succeeded to do that, which leaves the caller's current
    gate alone.
    """
    episodes = collect_solo_rollouts(worker, scorer, n_rollouts, tag or "logpzo")
    succ = [e["feat"] for e in episodes if e["success"] and len(e["feat"])]
    n_succ = len(succ)
    if n_succ < 4:
        return None

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_succ)
    half = n_succ // 2
    flow_feats = [succ[i] for i in order[:half]]
    band_feats = [succ[i] for i in order[half:]]

    torch.manual_seed(seed)
    model = LogpZOModel(feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    fit = fit_flow_on_episodes(model, opt, flow_feats, grad_steps, device, batch_size=batch_size,
                               seed=seed)

    traces = np.asarray([pad_to(score_features(model, f, device, batch_size), horizon)
                         for f in band_feats])
    # SAFE's split of the band episodes: 30% sets the mean curve and the spread, 70% the width.
    idx = rng.permutation(len(traces))
    n_first = max(1, int(len(traces) * 0.3))
    band = conformal_band(traces[idx[:n_first]], traces[idx[n_first:]],
                          np.asarray([float(alpha)]))[0]

    # The 30% slice sets the band's whole shape, and it is tiny at realistic rollout budgets. With
    # one trace in it the band degenerates to that single episode plus a constant offset, because
    # the deviations it measures the spread from are all zero. Reported so the log says how thin the
    # band is rather than leaving it to be guessed from a jagged plot.
    info = dict(n_rollouts=len(episodes), n_success=n_succ, n_flow_episodes=len(flow_feats),
                n_band_episodes=len(band_feats), n_shape_episodes=n_first,
                n_width_episodes=len(traces) - n_first,
                solo_steps=int(sum(e["steps"] for e in episodes)),
                train_loss=float(fit["train_loss"]), val_loss=float(fit["val_loss"]),
                n_transitions=int(fit["n_transitions"]))
    return model, band, info


class BandGate:
    """Fire when the score exceeds LogpZO's conformal threshold."""

    def __init__(self, band=None, alpha: float = 0.1, patience: int = 1,
                 patience_window: Optional[int] = None, score_every: int = 1,
                 name: str = "logpzo", **_ignored: Any):
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError(f"alpha is a false-alarm budget; got {alpha}")
        if float(alpha) > 0.5:
            raise ValueError(f"alpha={alpha} would fire on most successful episodes; did you mean "
                             f"{1.0 - float(alpha):g}?")
        self.name = str(name)
        self.alpha = float(alpha)
        self.band = None if band is None else np.asarray(band, dtype=np.float64)
        self.score_every = max(1, int(score_every))
        self.patience = int(patience)
        self.patience_window = int(patience_window if patience_window is not None else patience)
        if self.patience_window < self.patience:
            raise ValueError("patience_window must be >= patience")
        self.deque = deque([], maxlen=self.patience_window)
        self.history = deque(maxlen=self.patience_window)
        self.n_updates = 0
        self.last_trigger = None
        self.last_threshold: Optional[float] = None

    def set_band(self, band, alpha: Optional[float] = None) -> Dict[str, Any]:
        if alpha is not None:
            self.alpha = float(alpha)
        self.band = np.asarray(band, dtype=np.float64)
        return dict(alpha=self.alpha, band_len=int(len(self.band)),
                    band_first=float(self.band[0]), band_last=float(self.band[-1]),
                    band_mean=float(self.band.mean()))

    def threshold_at(self, step: int) -> float:
        """The band at this step. Past the end of the band it holds at the last value, which is
        what an episode running longer than any of the ones the band was built from needs."""
        if self.band is None or len(self.band) == 0:
            return float("inf")
        return float(self.band[min(max(int(step), 0), len(self.band) - 1)])

    def update(self, value: float) -> bool:
        value = float(value)
        self.history.append(value)
        # The worker only scores every score_every-th step, so this counts env steps
        step = self.n_updates * self.score_every
        self.n_updates += 1
        self.last_threshold = self.threshold_at(step)
        self.deque.append(value > self.last_threshold)
        fired = sum(self.deque) >= self.patience
        self.last_trigger = "logpzo_band" if fired else None
        return fired

    def reset(self):
        self.deque.clear()
        self.history.clear()
        self.n_updates = 0
        self.last_trigger = None

    def describe(self) -> Dict[str, Any]:
        return dict(gate=self.name, alpha=self.alpha, calibrated=self.band is not None,
                    band_len=0 if self.band is None else int(len(self.band)),
                    patience=self.patience, patience_window=self.patience_window)

    def plot_trace(self, stats: Dict[str, Any], save_path: str, title: Optional[str] = None):
        """Plots the LogpZO score against the band it was judged by."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = list(stats["progress_trace"])
        steps = len(scores)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(range(steps), scores, color="#1f77b4", lw=1.5, zorder=2, label="logpZO score")
        if self.band is not None and len(self.band):
            ax.plot(range(steps), [self.threshold_at(t) for t in range(steps)], color="#2ca02c",
                    lw=1.2, ls="--", zorder=3, label=f"conformal band (alpha={self.alpha:g})")
        for f in stats["gate_fires"]:
            ax.axvspan(f, steps - 1, color="#d62728", alpha=0.08, zorder=1)
            ax.axvline(f, color="#d62728", lw=1.5, zorder=4)
            ax.annotate(f"takeover\n@{f}", xy=(f, max(scores, default=1.0)), xytext=(2, -2),
                        textcoords="offset points", ha="left", va="top", fontsize=7,
                        color="#d62728")

        ax.set_xlabel("environment step")
        ax.set_ylabel("logpZO score")
        ax.set_title(title or f"success={stats['success']}  "
                              f"interventions={stats['num_interventions']}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
