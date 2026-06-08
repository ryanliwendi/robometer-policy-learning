"""Conditional MLP noise-prediction network (robust fallback for short horizons)."""

from typing import Tuple

import torch
import torch.nn as nn

from robometer_policy_learning.modules.diffusion.layers import SinusoidalPosEmb
from robometer_policy_learning.utils.featurizers import _build_mlp_layers


class ConditionalMLP(nn.Module):
    """Predicts noise over a flattened action chunk; robust to any (small) horizon."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        in_dim = action_dim * horizon + dsed + int(global_cond_dim)
        layers = _build_mlp_layers(in_dim, hidden_dims, activation="relu")
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(int(hidden_dims[-1]), action_dim * horizon)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        b = sample.shape[0]
        x_flat = sample.reshape(b, -1)
        t_emb = self.diffusion_step_encoder(timestep)
        x = torch.cat([x_flat, t_emb, global_cond], dim=-1)
        out = self.head(self.mlp(x))
        return out.reshape(b, self.horizon, self.action_dim)
