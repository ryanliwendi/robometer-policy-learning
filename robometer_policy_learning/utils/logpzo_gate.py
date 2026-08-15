"""The LogpZO intervention gate (SAFE's density baseline, https://github.com/vla-safe/SAFE).

LogpZO fits a flow-matching model to the features of states the policy was trained on, then scores
a new state by how far the model has to push it to look like training data. States unlike anything
in training get a high score.

Differences from the original implementation:
    * Only the success flow is fit, so no rollout labels are needed and the gate can run inside the
      DAgger loop. The reference also offers a second flow fit on failures, which turns the score
      into a log-odds between the two; that variant is supervised and belongs in the offline
      detector comparison, not here.
    * The features are the student's own observation encoding (``encode_obs``), the same vector
      ThriftyDAgger's novelty ensemble reads, rather than a VLA's hidden states. That keeps the two
      baselines a fair comparison: same input, different density model.
    * Refit every DAgger iteration on the growing buffer, like ThriftyDAgger's ensemble.

The firing rule is Diff-DAgger's ``QuantileGate``: fire when the score exceeds the alpha-quantile
of the scores over the training data, with the same M-of-N patience.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from robometer_policy_learning.modules.diffusion.unet import ConditionalUnet1D
from robometer_policy_learning.utils.thrifty_gate import _IndexSpace, _encode_batch

# The reference's UNet width. Kept as-is so the model has the capacity the paper reports.
DOWN_DIMS = (256, 512, 1024)
# Channel count the feature vector is folded into. 32 divides our 768-d encoding exactly.
IN_DIM = 32


def adjust_xshape(x: torch.Tensor, in_dim: int) -> torch.Tensor:
    """Fold a flat feature vector (N, D) into (N, D/in_dim, in_dim) so the 1D UNet can read it.

    The UNet treats the middle axis as time and `in_dim` as channels. D is right-padded with zeros
    when it does not divide by `in_dim`. The reference pads a second time to make the middle axis a
    multiple of 4; ours does that padding internally, so that step is left out.
    """
    remain = x.shape[1] % in_dim
    if remain:
        x = torch.cat([x, torch.zeros(x.shape[0], in_dim - remain, device=x.device, dtype=x.dtype)],
                      dim=1)
    return x.reshape(x.shape[0], -1, in_dim)


def _build_unet(in_dim: int) -> ConditionalUnet1D:
    """The reference passes no conditioning, so global_cond is a zero-width tensor at every call."""
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

        x0 is the real feature, x1 is noise, and the true velocity is the difference. The net sees
        a random point on the line between them and has to predict that velocity.
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
        """Squared length of where the flow sends this feature, at time 0. One score per row.

        A feature the model has seen a lot of needs little pushing, so the score is small.
        """
        x = adjust_xshape(feat, self.in_dim)
        t = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        v = self.net(x, t.long(), self._empty_cond(len(x), x))
        return (x + v).reshape(len(x), -1).pow(2).sum(dim=-1)


def train_logpzo(algo, model: LogpZOModel, opt: torch.optim.Optimizer, grad_steps: int,
                 batch_size: int = 256, device=None, seed: int = 0, buffers=None,
                 val_fraction: float = 0.1) -> Dict[str, float]:
    """Fit the flow on features drawn from the offline demos plus whatever has been collected.

    A slice of rows is held out and never trained on. This model is much larger than the feature
    set it is fit to, so the held-out loss is the only warning that it has started memorising.
    """
    space = _IndexSpace(buffers if buffers is not None else algo.buffer, device)
    if space.total == 0:
        return dict(train_loss=float("nan"), val_loss=float("nan"), n_transitions=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(space.total)
    n_val = int(space.total * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def _features(idx: np.ndarray) -> torch.Tensor:
        batch = space.batch(idx)
        return _encode_batch(algo, batch["obs"]).float()

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
                val_loss=val_loss, n_transitions=int(space.total), n_val=int(len(val_idx)))


@torch.no_grad()
def collect_logpzo_scores(algo, model: LogpZOModel, device=None, max_rows: int = 4096,
                          batch_size: int = 256, buffers=None, seed: int = 0) -> np.ndarray:
    """Score the training data, so the gate knows what a normal score looks like."""
    space = _IndexSpace(buffers if buffers is not None else algo.buffer, device)
    if space.total == 0:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(space.total, size=max_rows, replace=False)
           if space.total > max_rows else np.arange(space.total))
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(idx), batch_size):
        batch = space.batch(idx[start:start + batch_size])
        feat = _encode_batch(algo, batch["obs"]).float()
        out.append(model.score(feat).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


class LogpZOScorer:
    """Per-step score for the rollout worker. Same interface as ``DiffDaggerScorer``."""

    def __init__(self, algo, model: LogpZOModel, remove_obs_keys=None, device=None):
        self.algo = algo
        self.model = model
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.device = device or next(model.parameters()).device
        self.task = ""
        self.episode_id = 0
        self._last_obs = None

    def reset(self, task: str = ""):
        self.task = str(task)
        self.episode_id += 1
        self._last_obs = None

    def observe(self, obs, frame_key: Optional[str] = None):
        self._last_obs = obs

    @torch.no_grad()
    def score(self):
        from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

        if self._last_obs is None:
            return 0.0, 0.0
        prepped = {k: v for k, v in self._last_obs.items() if k not in self.remove_obs_keys}
        actor_obs = move_to_device(convert_to_tensor(prepped), self.device)
        feat = self.algo.actor.encode_obs(actor_obs).float()
        return float(self.model.score(feat).reshape(-1)[0]), 0.0
