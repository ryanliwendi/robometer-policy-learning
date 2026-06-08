"""Diffusion Policy (DP) implementation.

This module contains everything needed to train and deploy a Diffusion Policy
(Chi et al., 2023, https://arxiv.org/abs/2303.04137) inside the robometer policy-learning
framework:

* the diffusers DDPM / DDIM noise schedulers (forward noising + reverse sampling),
* three conditional noise-prediction networks (a 1D conditional U-Net, a conditional MLP,
  and a conditional Transformer),
* :class:`DiffusionActor`, a :class:`BaseActor` whose ``act()`` runs the reverse diffusion
  process so it can be deployed/evaluated like any other actor,
* :class:`DP`, the :class:`BaseAlgorithm` that trains the actor by denoising-score matching.

The policy works in the actor's normalized ([-1, 1]) action space. The replay buffer maps
stored env-space actions into [-1, 1] (via ``min_action`` / ``max_action``); ``act()``
unnormalizes back to the env action space.
"""

import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler

from robometer_policy_learning.algorithms.dp.configuration_dp import DPConfig
from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.modules.base import BaseActorConfig
from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.modules.diffusion import ConditionalMLP, ConditionalTransformer, ConditionalUnet1D
from robometer_policy_learning.utils.featurizers import ObservationFeaturizer, _build_mlp_layers


# =====================================================================================
# Noise scheduler (diffusers)
# =====================================================================================
def make_noise_scheduler(
    sampler: str,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_schedule: str,
    prediction_type: str,
    clip_sample: bool,
    clip_sample_range: float,
) -> Union[DDPMScheduler, DDIMScheduler]:
    """Build a diffusers DDPM or DDIM scheduler.

    The same instance is used for both training (``add_noise`` forward process) and
    inference (``set_timesteps`` + ``step`` reverse process); the two scheduler types share
    the diffusion constants and differ only in the reverse update.
    """
    kwargs = dict(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        clip_sample=clip_sample,
        clip_sample_range=clip_sample_range,
    )
    if sampler == "ddpm":
        return DDPMScheduler(**kwargs)
    elif sampler == "ddim":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unknown sampler: {sampler!r} (expected 'ddpm' or 'ddim')")


# =====================================================================================
# Diffusion actor (deployable BaseActor)
# =====================================================================================
@dataclass
class DiffusionActorConfig(BaseActorConfig):
    """Config for :class:`DiffusionActor`.

    Carries the standard actor fields (obs/action space, normalization, featurizer/image
    encoder settings) plus the diffusion-specific hyperparameters from :class:`DPConfig`.
    """

    # Featurizer / image-encoder settings (copied from the source actor config)
    featurizer: Optional[dict] = None
    activation: str = "relu"
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    image_encoder_type: Optional[str] = None
    finetune_image_encoder: bool = False
    image_feature_dim: int = 128
    resnet_backbone: str = "ResNet18"
    resnet_pretrained: bool = True
    resnet_pool: str = "spatial_softmax"
    spatial_softmax_num_kp: int = 32
    dinov2_model: object = None
    dinov2_processor: object = None
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_output_dim: Optional[int] = None

    # Diffusion hyperparameters
    horizon: int = 1
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    sampler: str = "ddpm"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    net_type: str = "unet"
    diffusion_step_embed_dim: int = 128
    unet_down_dims: Tuple[int, ...] = (128, 256)
    unet_kernel_size: int = 5
    unet_n_groups: int = 8
    mlp_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    transformer_d_model: int = 256
    transformer_nhead: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.0
    transformer_activation: str = "gelu"
    obs_encoder_hidden_dims: Tuple[int, ...] = (256, 256)

    @property
    def actor_class(self):
        return DiffusionActor


class DiffusionActor(BaseActor):
    """Actor whose ``act()`` produces actions via reverse diffusion conditioned on the obs.

    Trainable parameters: an observation encoder (featurizer + MLP producing the global
    conditioning vector) and a noise-prediction network (U-Net, MLP, or Transformer). The DP algorithm
    drives training through :meth:`encode_obs` / :meth:`predict_noise`; deployment uses
    :meth:`_act` (called by :meth:`BaseActor.act`).
    """

    def __init__(self, config: DiffusionActorConfig):
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform

        if not self.is_continuous:
            raise ValueError("DiffusionActor only supports continuous (Box) action spaces")

        self.action_dim = int(np.prod(config.action_space.shape))
        self.horizon = int(config.horizon)

        # --- Observation encoder: featurizer -> (B, obs_dim) -> MLP -> (B, global_cond_dim) ---
        self.obs_featurizer = ObservationFeaturizer(
            observation_space=config.observation_space,
            featurizer_cfg=config.featurizer,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            dropout_rate=config.dropout_rate,
            image_encoder_type=config.image_encoder_type,
            finetune_image_encoder=config.finetune_image_encoder,
            image_feature_dim=config.image_feature_dim,
            resnet_backbone=config.resnet_backbone,
            resnet_pretrained=config.resnet_pretrained,
            resnet_pool=config.resnet_pool,
            spatial_softmax_num_kp=config.spatial_softmax_num_kp,
            dinov2_model=config.dinov2_model,
            dinov2_processor=config.dinov2_processor,
            impala_nn_scale=config.impala_nn_scale,
            impala_num_blocks_per_stack=config.impala_num_blocks_per_stack,
            impala_use_smaller=config.impala_use_smaller,
            impala_output_dim=config.impala_output_dim,
        )
        obs_dim = int(self.obs_featurizer.output_dim)
        if obs_dim <= 0:
            raise ValueError("ObservationFeaturizer produced invalid output dimension for DiffusionActor.")

        if config.obs_encoder_hidden_dims:
            self.obs_encoder = nn.Sequential(
                *_build_mlp_layers(
                    obs_dim,
                    config.obs_encoder_hidden_dims,
                    config.activation,
                    config.use_layer_norm,
                    config.dropout_rate,
                )
            )
            self.global_cond_dim = int(config.obs_encoder_hidden_dims[-1])
        else:
            self.obs_encoder = nn.Identity()
            self.global_cond_dim = obs_dim

        # --- Noise prediction network ---
        if config.net_type == "unet":
            self.net = ConditionalUnet1D(
                action_dim=self.action_dim,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                down_dims=config.unet_down_dims,
                kernel_size=config.unet_kernel_size,
                n_groups=config.unet_n_groups,
            )
        elif config.net_type == "mlp":
            self.net = ConditionalMLP(
                action_dim=self.action_dim,
                horizon=self.horizon,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                hidden_dims=config.mlp_hidden_dims,
            )
        elif config.net_type == "transformer":
            self.net = ConditionalTransformer(
                action_dim=self.action_dim,
                horizon=self.horizon,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
                activation=config.transformer_activation,
            )
        else:
            raise ValueError(f"Unknown net_type: {config.net_type!r} (expected 'unet', 'mlp', or 'transformer')")

        # --- Noise scheduler (diffusers) ---
        # Stored as a plain attribute (not a submodule); diffusers handles device placement
        # internally via ``add_noise`` and ``set_timesteps(device=...)``.
        self.scheduler = make_noise_scheduler(
            sampler=config.sampler,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )
        self.num_inference_steps = int(config.num_inference_steps)
        self.sampler = config.sampler

        # DiffusionActor has no Gaussian action distribution; act()/training use diffusion.
        self.action_dist = None

    # ------------------------------------------------------------------ helpers
    def encode_obs(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        """Featurize observations into the global conditioning vector ``(B, global_cond_dim)``."""
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        device = next(self.parameters()).device
        obs_flat = self.obs_featurizer.flatten_obs(obs, device=device)
        return self.obs_encoder(obs_flat.float())

    def predict_noise(
        self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """Network forward: predict the noise (or x0) added to ``noisy_actions``."""
        return self.net(noisy_actions, timesteps, global_cond)

    @torch.no_grad()
    def sample_actions(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        """Run the reverse diffusion process. Returns ``(B, horizon, action_dim)`` in [-1, 1]."""
        global_cond = self.encode_obs(obs)
        batch_size = global_cond.shape[0]
        device = global_cond.device

        x = torch.randn(batch_size, self.horizon, self.action_dim, device=device)
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        for t in self.scheduler.timesteps:
            # Per-sample timestep tensor for the network's sinusoidal embedding.
            t_batch = t.to(device=device, dtype=torch.long).expand(batch_size)
            model_output = self.net(x, t_batch, global_cond)
            x = self.scheduler.step(model_output, t, x).prev_sample
        return x.clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ BaseActor API
    def _act(
        self, obs: Union[dict, torch.Tensor], deterministic: bool = False, actor_state: Any = None
    ) -> Tuple[torch.Tensor, Any]:
        # Diffusion sampling is inherently stochastic; ``deterministic`` is accepted for API
        # compatibility (a fixed seed could be added here if exact reproducibility is needed).
        actions = self.sample_actions(obs)
        if self.horizon == 1:
            actions = actions.squeeze(1)  # (B, action_dim) for non-chunked deployment
        return actions, None

    def get_action_dist_params(self, obs, hidden=None):
        raise NotImplementedError("DiffusionActor does not expose an explicit action distribution.")

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))


# =====================================================================================
# Exponential moving average of weights
# =====================================================================================
class EMAModel:
    """Maintains an exponential moving average of another module's parameters/buffers."""

    def __init__(self, averaged_model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.averaged_model = averaged_model

    @torch.no_grad()
    def step(self, new_model: nn.Module):
        for avg_p, new_p in zip(self.averaged_model.parameters(), new_model.parameters()):
            if avg_p.dtype.is_floating_point:
                avg_p.mul_(self.decay).add_(new_p.detach(), alpha=1.0 - self.decay)
            else:
                avg_p.copy_(new_p.detach())
        for avg_b, new_b in zip(self.averaged_model.buffers(), new_model.buffers()):
            avg_b.copy_(new_b)


# =====================================================================================
# Diffusion Policy algorithm
# =====================================================================================
class DP(BaseAlgorithm):
    """Diffusion Policy: behavior cloning via conditional denoising-score matching.

    ``config.actor`` must be a :class:`DiffusionActor` (built by
    ``training_utils.build_actor_critic_models`` like every other actor); it holds all
    trainable parameters. The (EMA) diffusion actor is exposed as ``self.actor`` so it can be
    evaluated and checkpointed like any other policy.
    """

    def __init__(self, config: DPConfig):
        super().__init__(config)
        self.config = config

        online_actor = config.actor
        if online_actor is None:
            raise ValueError("A DiffusionActor is required for DP (built by build_actor_critic_models)")
        if not isinstance(online_actor, DiffusionActor):
            raise TypeError(
                f"DP requires a DiffusionActor, got {type(online_actor).__name__}. "
                "It is built in training_utils.build_actor_critic_models when offline_alg_name == 'dp'."
            )

        self.device = next(online_actor.parameters()).device
        self.horizon = online_actor.horizon
        self.online_actor = online_actor.to(self.device)

        # EMA weights are used for deployment/eval; otherwise the online net is deployed.
        self.use_ema = config.use_ema
        if self.use_ema:
            self.actor = copy.deepcopy(self.online_actor).to(self.device)
            for p in self.actor.parameters():
                p.requires_grad_(False)
            self.ema = EMAModel(self.actor, decay=config.ema_decay)
            self.component_names = ["actor", "online_actor", "actor_optimizer"]
        else:
            self.actor = self.online_actor
            self.ema = None
            self.component_names = ["actor", "actor_optimizer"]

        self.buffer = config.buffer
        self.batch_size = config.batch_size
        self.learning_starts = config.learning_starts
        self.prediction_type = config.prediction_type

        self.actor_optimizer = torch.optim.AdamW(
            self.online_actor.parameters(),
            lr=config.actor_optimizer_lr,
            betas=config.actor_optimizer_betas,
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        print(f"DP: net_type={config.net_type}, horizon={self.horizon}, action_dim={self.online_actor.action_dim}")
        print(
            f"DP: num_train_timesteps={config.num_train_timesteps}, "
            f"num_inference_steps={config.num_inference_steps}, prediction_type={self.prediction_type}, "
            f"sampler={config.sampler}, use_ema={self.use_ema}"
        )

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Coerce buffer actions into ``(B, horizon, action_dim)``."""
        if actions.dim() == 2:  # (B, action_dim) -> single-step chunk
            actions = actions.unsqueeze(1)
        elif actions.dim() != 3:
            raise ValueError(f"Unexpected action shape for DP: {tuple(actions.shape)}")
        return actions

    def train_step(self, logging_prefix: str = "dp", rollout_step: int = None) -> dict:
        if rollout_step is not None and rollout_step < self.learning_starts:
            return {}

        losses = []
        noise_pred_means = []
        expert_action_means = []
        sample_mse_errors = []
        unnormalized_sample_mse_errors = []

        gradient_steps = self.config.num_updates_per_train_step

        for _ in range(gradient_steps):
            batch = self.buffer.sample(self.batch_size, device=self.device)
            if not batch:
                print("Buffer is still empty. Skipping this training step")
                return {}

            obs = batch["obs"]
            expert_actions = batch["action"]
            if len(expert_actions) == 0:
                print("Buffer is still empty. Skipping this training step")
                return {}

            expert_actions = self._prepare_actions(expert_actions).float()

            # Optional data augmentation (parity with BC).
            if self.config.obs_noise_std > 0:
                if isinstance(obs, dict):
                    obs = {k: v + torch.randn_like(v) * self.config.obs_noise_std for k, v in obs.items()}
                else:
                    obs = obs + torch.randn_like(obs) * self.config.obs_noise_std
            if self.config.action_noise_std > 0:
                expert_actions = expert_actions + torch.randn_like(expert_actions) * self.config.action_noise_std

            batch_size = expert_actions.shape[0]

            # Forward diffusion: corrupt the expert action chunk.
            noise = torch.randn_like(expert_actions)
            timesteps = torch.randint(
                0, self.online_actor.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
            ).long()
            noisy_actions = self.online_actor.scheduler.add_noise(expert_actions, noise, timesteps)

            # Predict noise (or x0) and regress against the target.
            global_cond = self.online_actor.encode_obs(obs)
            model_pred = self.online_actor.predict_noise(noisy_actions, timesteps, global_cond)
            target = noise if self.prediction_type == "epsilon" else expert_actions

            loss = F.mse_loss(model_pred, target)

            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.config.clip_grad_norm and self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.online_actor.parameters(), max_norm=self.config.clip_grad_norm)
            self.actor_optimizer.step()

            if self.use_ema:
                self.ema.step(self.online_actor)

            losses.append(loss.item())
            noise_pred_means.append(model_pred.mean().item())
            expert_action_means.append(expert_actions.mean().item())

            # Optional (expensive) end-to-end sampling diagnostic.
            if (
                self.config.log_sample_metrics_every > 0
                and (self.step_counter + 1) % self.config.log_sample_metrics_every == 0
            ):
                with torch.no_grad():
                    sampled = self.actor.sample_actions(obs)
                    if sampled.shape != expert_actions.shape:
                        sampled = sampled.reshape(expert_actions.shape)
                    sample_mse_errors.append(F.mse_loss(sampled, expert_actions).item())
                    unnormalized_sample_mse_errors.append(
                        F.mse_loss(
                            self.actor.unnormalize_action(sampled),
                            self.actor.unnormalize_action(expert_actions),
                        ).item()
                    )

            self.step_counter += 1

        metrics_dict = {
            "actor_loss": float(np.mean(losses)),
            "noise_pred_mean": float(np.mean(noise_pred_means)),
            "expert_action_mean": float(np.mean(expert_action_means)),
        }
        if sample_mse_errors:
            metrics_dict["sample_mse_error"] = float(np.mean(sample_mse_errors))
            metrics_dict["unnormalized_sample_mse_error"] = float(np.mean(unnormalized_sample_mse_errors))

        if self.logger is not None:
            self.logger.log(metrics_dict, step=self.step_counter, prefix=logging_prefix)

        return metrics_dict

    @torch.no_grad()
    def evaluate_policy(self, eval_buffer, num_eval_batches: int = 10) -> dict:
        """Denoising-loss + one-shot sampling MSE on a held-out buffer (overfitting check)."""
        self.online_actor.eval()
        eval_losses = []
        eval_sample_mse = []
        for _ in range(num_eval_batches):
            batch = eval_buffer.sample(self.batch_size, device=self.device)
            if not batch:
                continue
            obs = batch["obs"]
            expert_actions = self._prepare_actions(batch["action"]).float()
            batch_size = expert_actions.shape[0]

            noise = torch.randn_like(expert_actions)
            timesteps = torch.randint(
                0, self.online_actor.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
            ).long()
            noisy_actions = self.online_actor.scheduler.add_noise(expert_actions, noise, timesteps)
            global_cond = self.online_actor.encode_obs(obs)
            model_pred = self.online_actor.predict_noise(noisy_actions, timesteps, global_cond)
            target = noise if self.prediction_type == "epsilon" else expert_actions
            eval_losses.append(F.mse_loss(model_pred, target).item())

            sampled = self.actor.sample_actions(obs)
            if sampled.shape != expert_actions.shape:
                sampled = sampled.reshape(expert_actions.shape)
            eval_sample_mse.append(F.mse_loss(sampled, expert_actions).item())

        self.online_actor.train()
        return {
            "eval_loss": float(np.mean(eval_losses)) if eval_losses else float("nan"),
            "eval_sample_mse_error": float(np.mean(eval_sample_mse)) if eval_sample_mse else float("nan"),
        }
