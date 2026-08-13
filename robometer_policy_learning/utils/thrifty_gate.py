"""ThriftyDAgger baseline.

Differences from the original implementation: 
    * Uses takeover mode instead of handing control back to the student. Retunes threshold 
      alpha (0.01 -> 0.001) to reduce frequency of expert interventions and adapt to the takeover mode.
    * Ensemble trained with MLP on top of a frozen backbone, since policies take raw images
      instead of low-dim states; DP drives actions while the ensemble is only used to detect novelty.
      The ensembles are retrained every iteration, and the DP policy is fine-tuned.
"""

import copy
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "baselines")
if _VENDORED not in sys.path:
    sys.path.insert(0, _VENDORED)
from vendored.thrifty_core import Ensemble  # noqa: E402

POLYAK = 0.995  # Between bahavior and target q-networks
POS_FRACTION = 0.1  # Fraction of samples for q-risk training with success labels


class _Space:
    """Replacement for gym Space."""
    def __init__(self, dim: int, high: float = 1.0):
        self.shape = (int(dim),)
        self.high = np.full((int(dim),), float(high), dtype=np.float32)
        self.low = -self.high


def build_ensemble(feat_dim: int, act_dim: int, device, num_nets: int = 5,
                   act_limit: float = 1.0) -> Ensemble:
    return Ensemble(_Space(feat_dim), _Space(act_dim, act_limit), device, num_nets=num_nets)


def compute_loss_pi(ac: Ensemble, obs, act, i: int):
    a_pred = ac.pis[i](obs)
    return torch.mean(torch.sum((act - a_pred) ** 2, dim=1))


def compute_loss_q(ac: Ensemble, ac_targ: Ensemble, obs, act, obs2, next_act, rew, done,
                   gamma: float):
    """Bellman loss for the twin Q critics.

    next_act should come from the diffusion policy, not from the ensemble,
    because the diffusion policy is what actually drives the robot.
    """
    q1, q2 = ac.q1(obs, act), ac.q2(obs, act)
    with torch.no_grad():
        backup = rew + gamma * (1 - done) * torch.min(
            ac_targ.q1(obs2, next_act), ac_targ.q2(obs2, next_act))
    return ((q1 - backup) ** 2).mean() + ((q2 - backup) ** 2).mean()


def polyak_update(ac: Ensemble, ac_targ: Ensemble):
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            p_targ.data.mul_(POLYAK)
            p_targ.data.add_((1 - POLYAK) * p.data)


@torch.no_grad()
def _encode_batch(algo, obs):
    return algo.actor.encode_obs(obs)


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n row numbers with replacement."""
    return rng.integers(0, n, size=n) if n else np.empty(0, dtype=np.int64)


def _buffer_parts(buf) -> List[Any]:
    if isinstance(buf, (list, tuple)):
        return [p for b in buf if b is not None for p in _buffer_parts(b)]
    if hasattr(buf, "buffer_1") and hasattr(buf, "buffer_2"):
        return _buffer_parts(buf.buffer_1) + _buffer_parts(buf.buffer_2)
    return [buf]


def _part_batch(part, idx: np.ndarray, device):
    """Fetches a batch by explicit indices"""
    if hasattr(part, "batch_from_indices"):
        return part.batch_from_indices(np.asarray(idx), device=device)
    return part.collate_transitions(part.transitions_from_indices(idx), device=device)


def _cat_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate one batch per buffer (offline demos, online rollouts) into a single batch."""
    batches = [b for b in batches if b]
    if not batches:
        return {}
    if len(batches) == 1:
        return batches[0]
    out: Dict[str, Any] = {}
    # Keep only the keys that both buffers have
    common = set(batches[0])
    for b in batches[1:]:
        common &= set(b)
    for k in batches[0]:
        if k not in common:
            continue
        vals = [b[k] for b in batches]
        if isinstance(vals[0], dict):
            sub = set(vals[0])
            for v in vals[1:]:
                sub &= set(v)
            out[k] = {kk: torch.cat([torch.as_tensor(v[kk]) for v in vals], dim=0) for kk in sub}
        elif torch.is_tensor(vals[0]):
            out[k] = torch.cat(vals, dim=0)
        else:
            out[k] = np.concatenate([np.asarray(v) for v in vals], axis=0)
    return out


class _IndexSpace:
    """Makes several separate buffers behave like one long list of transitions. Index 0 is 
    the first row of the first buffer, and the numbering continues straight into the next buffer, and so on."""

    def __init__(self, buf, device):
        self.parts = _buffer_parts(buf)
        self.sizes = [len(p) for p in self.parts]
        self.offsets = np.cumsum([0] + self.sizes)
        self.total = int(self.offsets[-1])
        self.device = device

    def batch(self, idx) -> Dict[str, Any]:
        idx = np.asarray(idx)
        out = []
        for k, part in enumerate(self.parts):
            lo, hi = int(self.offsets[k]), int(self.offsets[k + 1])
            local = idx[(idx >= lo) & (idx < hi)] - lo
            if local.size:
                out.append(_part_batch(part, local, self.device))
        return _cat_batches(out)

    def row_indices(self, idx) -> np.ndarray:
        """Returns the indices of each transition :meth:``batch`` sends back."""
        idx = np.asarray(idx)
        out = []
        for k in range(len(self.parts)):
            lo, hi = int(self.offsets[k]), int(self.offsets[k + 1])
            out.append(idx[(idx >= lo) & (idx < hi)])
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def success_values(self, idx) -> np.ndarray:
        """Success labels aligned with :meth:`batch`'s rows."""
        idx = np.asarray(idx)
        out = []
        for k, part in enumerate(self.parts):
            lo, hi = int(self.offsets[k]), int(self.offsets[k + 1])
            local = idx[(idx >= lo) & (idx < hi)] - lo
            if local.size:
                out.extend(_success_flag(t) for t in part.transitions_from_indices(local))
        return np.asarray(out, dtype=np.float32)

    def positive_indices(self) -> np.ndarray:
        """Find every transition that reached the goal.

        Only the last step of an episode can be a success, so we ask each buffer where its
        episodes end and check just those steps instead of scanning everything.
        """
        pos = []
        for k, part in enumerate(self.parts):
            lo = int(self.offsets[k])
            try:
                bounds = part.get_episode_boundaries() or {}  # returns a dict of {episode_id: (start_idx, end_idx)}
            except Exception:  # noqa: BLE001
                continue
            # get_episode_boundaries returns (start, end) with end inclusive
            ends = np.asarray(sorted({int(e) for (_s, e) in bounds.values()}), dtype=np.int64)
            ends = ends[(ends >= 0) & (ends < self.sizes[k])]
            if ends.size == 0:
                continue
            for t, gi in zip(part.transitions_from_indices(ends), ends + lo):
                if _success_flag(t) > 0:
                    pos.append(int(gi))
        return np.asarray(sorted(pos), dtype=np.int64)


def _success_flag(t) -> float:
    """The reference's reward, 1 only on goal-reaching steps."""
    if t is None:
        return 0.0
    info = getattr(t, "info", None)
    if isinstance(info, dict):
        for key in ("is_success", "success"):
            if key in info:
                return 1.0 if bool(np.asarray(info[key]).reshape(-1)[0]) else 0.0
    return 1.0 if float(np.asarray(t.done).reshape(-1)[0]) > 0 else 0.0  # fallback: use t.done

def q_batch_indices(n: int, batch_size: int, pos_fraction: float,
                    rng: np.random.Generator, pos_pool: np.ndarray,
                    neg_pool: Optional[np.ndarray] = None) -> np.ndarray:
    """LIBERO success is sparse, so uniform sampling shows the critic almost no positives and it
    learns to predict ~0 everywhere. Forcing ```POS_FRACTION``` positives per batch keeps the critic informative.
    """
    if n == 0:
        return np.empty(0, dtype=np.int64)
    pool = np.unique(np.asarray(pos_pool, dtype=np.int64))

    want_pos = min(batch_size, max(1, int(batch_size * pos_fraction))) if len(pool) else 0  # At least 1 positive sample
    if neg_pool is None:
        neg_pool = np.setdiff1d(np.arange(n, dtype=np.int64), pool, assume_unique=False)  # All indexes that are not positive
    neg_pool = np.asarray(neg_pool, dtype=np.int64)
    want_neg = batch_size - want_pos
    if want_neg and len(neg_pool) == 0:      # every transition is a success, nothing else to draw
        want_pos, want_neg = batch_size, 0
    neg = rng.choice(neg_pool, size=want_neg, replace=True) if want_neg else np.empty(0, np.int64)
    if want_pos == 0:
        return neg
    return np.concatenate([rng.choice(pool, size=want_pos, replace=True), neg])


class _NextActionCache:
    """Stores the DP policy's actions for the critic update's target calculation.
    
    Ensures that each transitions ``next_obs`` is passed through the DP policy
    at most once per refresh.
    """

    def __init__(self, algo, n: int, act_dim: int, device):
        self.algo = algo
        self.device = device
        self.have = np.zeros(int(n), dtype=bool)
        self.cache = torch.zeros(int(n), int(act_dim), device=device)
        self.n_sampled = 0

    def get(self, rows: np.ndarray, next_obs) -> torch.Tensor:
        miss = ~self.have[rows]
        if miss.any():
            sel = torch.as_tensor(np.flatnonzero(miss), device=self.device)
            sub = {k: torch.as_tensor(v).to(self.device)[sel]
                   for k, v in next_obs.items()
                   if torch.is_tensor(v) or isinstance(v, np.ndarray) and v.dtype != object}
            with torch.no_grad():
                a = self.algo.actor.sample_actions(sub)
            if a.dim() == 3:
                a = a[:, 0, :]
            self.cache[rows[miss]] = a.float()
            self.have[rows[miss]] = True
            self.n_sampled += int(miss.sum())
        return self.cache[rows]


def train_thrifty_models(algo, ac: Ensemble, ac_targ: Ensemble, ens_opt_fn, q_opt,
                         grad_steps: int, gamma: float = 0.9999,
                         retrain_from_scratch: bool = True, num_nets: int = 5,
                         feat_dim: int = 0, act_dim: int = 0, device=None, seed: int = 0,
                         q_buffer=None, q_steps_multiplier: int = 5):
    """Fit the ensemble policies and the Q critics from scratch."""
    ens_losses, q_losses = [], []
    rng = np.random.default_rng(seed)
    bs = algo.batch_size
    # Built once per refresh. Nothing is loaded here: the index space only records how many rows
    # each buffer has, and pulls the actual transitions when a batch asks for them.
    bc_space = _IndexSpace(algo.buffer, device)
    risk_space = _IndexSpace(q_buffer if q_buffer is not None else algo.buffer, device)
    n_bc, n_t = bc_space.total, risk_space.total

    if retrain_from_scratch:
        # rebuilds the entire actor_critic each iteration
        fresh = build_ensemble(feat_dim, act_dim, device, num_nets=num_nets)
        for i in range(num_nets):
            ac.pis[i].load_state_dict(fresh.pis[i].state_dict())
        ac.q1.load_state_dict(fresh.q1.state_dict())
        ac.q2.load_state_dict(fresh.q2.state_dict())
        ac_targ.q1.load_state_dict(fresh.q1.state_dict())
        ac_targ.q2.load_state_dict(fresh.q2.state_dict())
        q_opt.state.clear()
    pi_opts = [ens_opt_fn(ac.pis[i].parameters()) for i in range(num_nets)]

    # Train each policy separately with its own sampled data from the tmp buffer
    for i in range(num_nets):
        boot = bootstrap_indices(n_bc, rng)
        if boot.size == 0:
            break
        for _ in range(grad_steps):
            batch = bc_space.batch(rng.choice(boot, size=bs))
            if not batch or len(batch.get("action", [])) == 0:
                break
            feat = _encode_batch(algo, batch["obs"])
            act = algo._prepare_actions(batch["action"]).float()[:, 0, :]
            loss = compute_loss_pi(ac, feat, act, i)
            pi_opts[i].zero_grad()
            loss.backward()
            pi_opts[i].step()
            ens_losses.append(loss.item())

    # Train the risk estimators
    pos_pool = risk_space.positive_indices()
    n_pos = int(len(pos_pool))
    neg_pool = np.setdiff1d(np.arange(n_t, dtype=np.int64), pos_pool, assume_unique=True)
    # With no goal-reaching transition every Bellman target is 0 and the critic's scale carries no
    # information; the caller disables the risk gate in this case.
    q_steps = int(grad_steps * max(1, int(q_steps_multiplier))) if n_pos > 0 else 0
    next_actions = _NextActionCache(algo, n_t, act_dim, device)
    for step in range(q_steps):
        idx = q_batch_indices(
            n_t, max(2, bs // 2), POS_FRACTION, rng, pos_pool, neg_pool=neg_pool)
        batch = risk_space.batch(idx)
        if not batch or len(batch.get("action", [])) == 0:
            break
        rows = risk_space.row_indices(idx)
        feat = _encode_batch(algo, batch["obs"])
        feat2 = _encode_batch(algo, batch["next_obs"])
        act = algo._prepare_actions(batch["action"]).float()[:, 0, :]
        next_act = next_actions.get(rows, batch["next_obs"])

        term = torch.as_tensor(batch["done"], device=device).float().reshape(-1)
        trunc = torch.as_tensor(batch.get("truncated", batch["done"]),
                                device=device).float().reshape(-1)
        rew = torch.as_tensor(risk_space.success_values(idx), device=device).float().reshape(-1)
        done = torch.clamp(term + trunc, max=1.0)
        loss = compute_loss_q(ac, ac_targ, feat, act, feat2, next_act, rew, done, gamma)
        q_opt.zero_grad()
        loss.backward()
        q_opt.step()
        if step % 2 == 0:
            polyak_update(ac, ac_targ)
        q_losses.append(loss.item())

    return dict(
        ensemble_loss=float(np.mean(ens_losses)) if ens_losses else float("nan"),
        qrisk_loss=float(np.mean(q_losses)) if q_losses else float("nan"),
        n_positive=n_pos,
        n_transitions=n_t,
        n_bc_transitions=n_bc,
        n_dp_target_samples=next_actions.n_sampled,
    )


@torch.no_grad()
def collect_thrifty_scores(algo, ac: Ensemble, num_batches: int = 8, device=None):
    nov, saf = [], []
    for _ in range(num_batches):
        batch = algo.buffer.sample(algo.batch_size, device=device)
        if not batch or len(batch.get("action", [])) == 0:
            break
        feat = algo.actor.encode_obs(batch["obs"])
        act = algo.actor.sample_actions(batch["obs"])
        if act.dim() == 3:
            act = act[:, 0, :]
        for j in range(feat.shape[0]):
            nov.append(float(ac.variance(feat[j:j + 1].cpu().numpy())))
            saf.append(float(ac.safety(feat[j:j + 1].cpu().numpy(), act[j:j + 1].cpu().numpy())))
    return np.asarray(nov), np.asarray(saf)


class ThriftyGate:
    def __init__(self, delta_h: float = float("inf"), beta_h: float = -float("inf"),
                 **_ignored: Any):
        self.delta_h = float(delta_h)    # novelty threshold, robot -> human
        self.beta_h = float(beta_h)      # safety threshold, robot -> human
        self.history = deque(maxlen=1)
        self.short_window = 1
        self.last_trigger = None
        self.online_novelty: List[float] = []
        self.online_safety: List[float] = []
        self.risk_enabled = False

    def update(self, value) -> bool:
        novelty, safety = value
        self.history.append(float(novelty))
        self.online_novelty.append(float(novelty))
        self.online_safety.append(float(safety))
        if novelty > self.delta_h:
            self.last_trigger = "novelty"
            return True
        if self.risk_enabled and safety < self.beta_h:
            self.last_trigger = "risk"
            return True
        self.last_trigger = None
        return False

    def reset(self):
        self.history.clear()

    def recalibrate(self, novelties: np.ndarray, safeties: np.ndarray, target_rate: float,
                    risk_enabled: Optional[bool] = None):
        """Refit both thresholds to the (1 - target_rate) percent of data.
        ``risk_enabled=False`` when no successful episodes, since all rewards will be 0
        and training qrisk is useless.
        """
        n = len(novelties)
        if n == 0:
            return
        target_idx = min(int((1.0 - target_rate) * n), n - 1)
        self.delta_h = float(np.sort(novelties)[target_idx])
        if risk_enabled is not None:
            self.risk_enabled = bool(risk_enabled)
        if self.risk_enabled and len(safeties):
            m = len(safeties)
            ti = min(int((1.0 - target_rate) * m), m - 1)
            self.beta_h = float(np.sort(safeties)[::-1][ti])
        else:
            self.beta_h = -float("inf")

    def clear_online(self):
        self.online_novelty.clear()
        self.online_safety.clear()


    def recalibrate_online(self, target_rate: float, min_samples: int = 25) -> bool:
        # Refit after every episode from this iteration's scores
        if len(self.online_novelty) <= min_samples:
            return False
        self.recalibrate(np.asarray(self.online_novelty), np.asarray(self.online_safety),
                         target_rate)
        return True

    def describe(self) -> Dict[str, Any]:
        return dict(gate="thrifty", delta_h=self.delta_h, beta_h=self.beta_h,
                    risk_enabled=self.risk_enabled)

    def plot_trace(self, stats: Dict[str, Any], save_path: str, title: Optional[str] = None):
        """Plots novelty and qrisk for an episode."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trace = stats["progress_trace"]
        pairs = [t if isinstance(t, (tuple, list)) else (t, float("nan")) for t in trace]
        novelty = [float(p[0]) for p in pairs]
        safety = [float(p[1]) if len(p) > 1 else float("nan") for p in pairs]
        steps = len(pairs)

        show_safety = self.risk_enabled and np.isfinite(self.beta_h)
        n_panels = 2 if show_safety else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.2 * n_panels), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]

        axes[0].plot(range(steps), novelty, color="#1f77b4", lw=1.5, zorder=2, label="novelty")
        if np.isfinite(self.delta_h):
            axes[0].axhline(self.delta_h, color="#2ca02c", lw=1.2, ls="--", zorder=3,
                            label=f"delta_h={self.delta_h:.4g}")
        axes[0].set_ylabel("ensemble novelty")
        axes[0].legend(loc="upper left", fontsize=8)

        if show_safety:
            axes[1].plot(range(steps), safety, color="#9467bd", lw=1.5, zorder=2, label="safety")
            axes[1].axhline(self.beta_h, color="#2ca02c", lw=1.2, ls="--", zorder=3,
                            label=f"beta_h={self.beta_h:.4g}")
            axes[1].set_ylabel("Q-risk safety")
            axes[1].legend(loc="upper left", fontsize=8)

        trigger_color = {"novelty": "#1f77b4", "risk": "#9467bd"}
        for f, reason in zip(stats["gate_fires"], stats["gate_reasons"]):
            c = trigger_color.get(reason, "#d62728")
            for ax in axes:
                ax.axvspan(f, steps - 1, color=c, alpha=0.08, zorder=1)
                ax.axvline(f, color=c, lw=1.5, zorder=4)
            axes[0].annotate(f"{reason or '?'}\n@{f}", xy=(f, max(novelty, default=1.0)),
                             xytext=(2, -2), textcoords="offset points", ha="left", va="top",
                             fontsize=7, color=c)

        axes[-1].set_xlabel("environment step")
        axes[0].set_title(title or f"success={stats['success']}  "
                                   f"interventions={stats['num_interventions']}")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)


class ThriftyScorer:
    def __init__(self, algo, ac: Ensemble, remove_obs_keys=None, lowdim_stats=None,
                 action_min=None, action_max=None, device=None):
        self.algo = algo
        self.actor = algo.actor  # the diffusion actor
        self.ac = ac
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.lowdim_stats = lowdim_stats or {}
        self.action_min = None if action_min is None else np.asarray(action_min, dtype=np.float32)
        self.action_max = None if action_max is None else np.asarray(action_max, dtype=np.float32)
        self.device = device or next(self.actor.parameters()).device
        self.task, self.episode_id = "", 0
        self._last_obs = None

    def reset(self, task: str = ""):
        self.task = str(task)
        self.episode_id += 1
        self._last_obs = None

    def observe(self, obs, frame_key: Optional[str] = None):
        self._last_obs = obs

    def _actor_obs(self, obs):
        # Process the raw obs input in rollout time to fit the form of the current replay buffer
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

    def _normalize_action(self, action):
        """Map an env-space action into the [-1, 1] space the critic was trained on."""
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        if self.action_min is None or self.action_max is None:
            return action
        span = self.action_max - self.action_min
        return 2.0 * (action - self.action_min) / np.where(span == 0, 1.0, span) - 1.0

    @torch.no_grad()
    def score(self, action=None):
        """(novelty, safety) at the last observed state."""
        if self._last_obs is None:
            return (0.0, 1.0), 0.0
        aobs = self._actor_obs(self._last_obs)
        feat = self.actor.encode_obs(aobs)
        if action is None:
            a = self.actor.sample_actions(aobs)
            if a.dim() == 3:
                a = a[:, 0, :]
            action_np = a.cpu().numpy()
        else:
            action_np = self._normalize_action(action)
        f = feat.cpu().numpy()
        novelty = float(self.ac.variance(f))
        safety = float(self.ac.safety(f, action_np))
        return (novelty, safety), 0.0

    def score_action(self, action):
        """Score the exact student action proposed for the current state (worker entry point)."""
        return self.score(action=action)
