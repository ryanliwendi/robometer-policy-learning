from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig


@dataclass
class DPConfig(BaseAlgorithmConfig):
    """
    Configuration for the Diffusion Policy (DP) algorithm.

    Diffusion Policy (Chi et al., 2023) frames behavior cloning as conditional denoising:
    a network learns to predict the noise that was added to an (action chunk) given the
    current observation as conditioning. At inference time, actions are produced by running
    the learned reverse diffusion process starting from Gaussian noise.

    The policy operates entirely in the actor's normalized ([-1, 1]) action space. The
    replay buffer is responsible for mapping stored env-space actions into [-1, 1] (via
    ``min_action`` / ``max_action``); ``act()`` unnormalizes back to the env action space.
    """

    # Runtime fields (inherited from BaseAlgorithmConfig: env, actor, critic, buffer, logger)
    action_space: Optional[Any] = None

    # ----- Generic training parameters -----
    learning_starts: int = 0
    batch_size: int = 256
    num_updates_per_train_step: int = 1

    # ----- Optimizer -----
    actor_optimizer_lr: float = 1e-4
    actor_optimizer_eps: float = 1e-8
    actor_optimizer_weight_decay: float = 1e-6
    actor_optimizer_betas: Tuple[float, float] = (0.95, 0.999)
    clip_grad_norm: float = 1.0  # <= 0 disables gradient clipping

    # ----- Diffusion / noise scheduler -----
    num_train_timesteps: int = 100
    num_inference_steps: int = 100  # <= num_train_timesteps; fewer => faster sampling
    beta_schedule: str = "squaredcos_cap_v2"  # ["linear", "squaredcos_cap_v2"]
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_type: str = "epsilon"  # ["epsilon", "sample"]
    sampler: str = "ddpm"  # ["ddpm", "ddim"]
    clip_sample: bool = True  # clamp predicted x0 to [-clip_sample_range, clip_sample_range]
    clip_sample_range: float = 1.0

    # ----- Noise prediction network -----
    net_type: str = "unet"  # ["unet", "mlp", "transformer"]
    diffusion_step_embed_dim: int = 128
    # Conditional U-Net 1D (Chi et al. style). Downsampling factor is 2**(len(down_dims)-1);
    # action sequences are internally padded to a multiple of that factor.
    unet_down_dims: Tuple[int, ...] = (128, 256)
    unet_kernel_size: int = 5
    unet_n_groups: int = 8
    # Conditional MLP (robust fallback for arbitrary / very short horizons).
    mlp_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    # Conditional Transformer (attends over action-chunk tokens; conditioning is a prepended
    # [timestep + obs] token). Best suited to longer horizons (chunk_size > 1).
    transformer_d_model: int = 256
    transformer_nhead: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.0
    transformer_activation: str = "gelu"

    # ----- Observation conditioning encoder -----
    # MLP applied on top of the (featurized) observation to produce the global conditioning
    # vector. Empty tuple => use the raw featurized observation as conditioning.
    obs_encoder_hidden_dims: Tuple[int, ...] = (256, 256)

    # ----- Exponential moving average of weights (used for inference/eval) -----
    use_ema: bool = True
    ema_decay: float = 0.995

    # ----- Optional data-augmentation regularizers (parity with BC) -----
    obs_noise_std: float = 0.0
    action_noise_std: float = 0.0

    # ----- Diagnostics -----
    # If > 0, every N updates run the (expensive) reverse diffusion sampler on the training
    # batch and log the resulting action MSE against the expert actions. 0 disables it.
    log_sample_metrics_every: int = 0

    @property
    def algorithm_class(self):
        from robometer_policy_learning.algorithms.dp import DP

        return DP
