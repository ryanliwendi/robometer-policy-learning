"""Conditional Transformer noise-prediction network."""

import torch
import torch.nn as nn

from robometer_policy_learning.modules.diffusion.layers import SinusoidalPosEmb
from robometer_policy_learning.modules.transformer.transformer_utils import PositionalEncoding


class ConditionalTransformer(nn.Module):
    """Transformer that predicts noise over an action sequence.

    Input/output shape: ``(B, T, action_dim)``. Each action step is embedded into a token
    (with positional encoding over the horizon); the (timestep + obs) conditioning is encoded
    as one extra token prepended to the sequence that the action tokens attend to.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
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
        # Conditioning token = projection of [timestep embedding, global obs cond] -> d_model.
        self.cond_proj = nn.Linear(dsed + int(global_cond_dim), d_model)
        # Per-step action token embedding + positional encoding over the horizon.
        self.input_proj = nn.Linear(action_dim, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=horizon, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.head = nn.Linear(d_model, action_dim)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.diffusion_step_encoder(timestep)  # (B, dsed)
        cond_token = self.cond_proj(torch.cat([t_emb, global_cond], dim=-1)).unsqueeze(1)  # (B, 1, d_model)
        x = self.pos_emb(self.input_proj(sample))  # (B, T, d_model)
        x = torch.cat([cond_token, x], dim=1)  # (B, T+1, d_model)
        x = self.encoder(x)
        return self.head(x[:, 1:])  # drop conditioning token -> (B, T, action_dim)
