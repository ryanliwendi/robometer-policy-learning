import torch
import gymnasium as gym
from torch import nn
from dataclasses import dataclass
from typing import Callable

from robometer_policy_learning.modules.base import BaseCriticConfig


@dataclass
class MLPCriticConfig(BaseCriticConfig):
    """
    Configuration for MLP-based critic networks.

    featurizer: Optional[dict]
        A dictionary mapping observation keys to either:
        - a list/tuple of hidden dims (e.g., [512, 256]) for a simple MLP featurizer
        - an nn.Module for custom feature extraction
        If provided, each key in the observation dict will be processed by its corresponding featurizer,
        and the resulting features will be concatenated before passing to the main MLP.
    """

    observation_space: gym.Space = None
    action_space: gym.Space = None

    # MLP architecture parameters
    hidden_dims: tuple = (256, 256)  # Hidden layer dimensions
    activation: str = "relu"  # Activation function
    use_layer_norm: bool = False  # Whether to use layer normalization
    dropout_rate: float = 0.0  # Dropout rate (0.0 means no dropout)

    # Optional per-key featurizer for dict observations
    featurizer: dict = None

    # Optional preprocess_obs_transform for dict observations
    preprocess_obs_transform: Callable = None

    # Image encoder parameters (optional - passed to ObservationFeaturizer).
    # image_encoder_type in {impala, resnet, dinov2} enables featurizer-level image encoding.
    image_encoder_type: str = None
    finetune_image_encoder: bool = False
    image_feature_dim: int = 128
    # ResNet
    resnet_backbone: str = "ResNet18"
    resnet_pretrained: bool = True
    resnet_pool: str = "spatial_softmax"
    spatial_softmax_num_kp: int = 32
    # DINOv2 (model/processor injected at build time when image_encoder_type == "dinov2")
    dinov2_model: object = None
    dinov2_processor: object = None
    # IMPALA
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_output_dim: int = None

    @property
    def critic_class(self):
        from robometer_policy_learning.modules.mlp import MLPCritic

        return MLPCritic
