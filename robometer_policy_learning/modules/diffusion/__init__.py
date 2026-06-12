"""Conditional noise-prediction networks for diffusion policies.

Each network maps a noisy action sequence ``(B, T, action_dim)`` + a diffusion timestep + a
global conditioning vector to a noise (or x0) prediction of the same shape. They are plain
``nn.Module`` building blocks (configured via kwargs, like ``modules.encoders``); diffusion
hyperparameters live in the consuming algorithm's config (e.g. ``DPConfig`` / ``dp.yaml``).
"""

from robometer_policy_learning.modules.diffusion.layers import SinusoidalPosEmb
from robometer_policy_learning.modules.diffusion.unet import ConditionalUnet1D
from robometer_policy_learning.modules.diffusion.mlp import ConditionalMLP
from robometer_policy_learning.modules.diffusion.transformer import ConditionalTransformer

__all__ = ["SinusoidalPosEmb", "ConditionalUnet1D", "ConditionalMLP", "ConditionalTransformer"]
