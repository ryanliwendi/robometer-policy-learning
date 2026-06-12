"""Conditional 1D U-Net noise-prediction network (Chi et al., 2023 style)."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from robometer_policy_learning.modules.diffusion.layers import SinusoidalPosEmb


def _safe_groups(n_groups: int, channels: int) -> int:
    """Largest valid GroupNorm group count not exceeding ``n_groups`` that divides channels."""
    g = min(int(n_groups), int(channels))
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(_safe_groups(n_groups, out_channels), out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning on the (timestep + obs) embedding."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        # FiLM: produce per-channel (scale, bias).
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0], embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    """Conditional 1D U-Net that predicts noise over an action sequence.

    Input/output shape: ``(B, T, action_dim)``. The temporal axis is internally padded to a
    multiple of the total downsampling factor so arbitrary horizons are supported.
    """

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        down_dims = tuple(int(d) for d in down_dims)
        all_dims = (action_dim, *down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + int(global_cond_dim)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        # Number of stride-2 downsampling ops (the deepest level keeps its resolution).
        self.downsample_factor = 2 ** (len(in_out) - 1) if len(in_out) > 1 else 1

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, action_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T)
        seq_len = sample.shape[1]
        x = sample.movedim(-1, -2)

        # Pad time axis so down/up sampling line up for arbitrary horizons.
        pad = (self.downsample_factor - seq_len % self.downsample_factor) % self.downsample_factor
        if pad:
            x = F.pad(x, (0, pad))

        t_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([t_emb, global_cond], dim=-1)

        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for res1, res2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.movedim(-2, -1)  # (B, C, T) -> (B, T, C)
        return x[:, :seq_len]
