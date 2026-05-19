#!/usr/bin/env python3
"""
Main RL training script with optional async/distributed mode.

Modes:
  - serial (default): Standard serial training loop
  - learner: Async mode - trains policy, serves gRPC for experience ingestion
  - rollout: Async mode - collects experience and streams to learner
  - eval: Async mode - periodically evaluates latest policy from learner
"""

import os


# Prevent JAX from preallocating all GPU memory (needed when using PyTorch + JAX together)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Disable TensorFlow GPU to avoid conflicts with JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# Initialize JAX early before any other GPU libraries to ensure it gets a clean GPU context
import jax

jax.devices()  # Forces JAX initialization
del jax  # Clean up namespace, will be re-imported when needed

import time
import threading
import socket
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from datetime import datetime

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
from rich import print as rprint

try:
    import wandb
except Exception:
    wandb = None

from robometer_policy_learning.runners.serial_runner import SerialRunner

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.robometer_replay_buffer import RobometerReplayBuffer
from robometer_policy_learning.buffers.success_failure_replay_buffer import SuccessFailureReplayBuffer
from robometer_policy_learning.buffers.samplers import RandomSampler
from robometer_policy_learning.algorithms.sac import SAC, SACConfig
from robometer_policy_learning.rollouts.robometer_rollout_worker import DSRLwithRobometerRolloutWorker
from robometer_policy_learning.rollouts.dsrl_rollout_worker import DSRLRolloutWorker
from robometer_policy_learning.rollouts.dsrl_evaluation_worker import DSRLEvaluationWorker
from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer.utils.config_utils import display_config, convert_hydra_to_dataclass
from robometer.utils.logger import setup_loguru_logging
from robometer_policy_learning.utils.training_utils import build_actor_critic_models, load_checkpoint, save_checkpoint, create_buffer
from robometer_policy_learning.configs.register import register_configs
from robometer_policy_learning.configs.configs import DSRLConfig

# Register all configs with Hydra before main() is called
register_configs()
from robometer_policy_learning.utils.transitions_transforms import SuccessBonusTransform

from robometer.configs.experiment_configs import ExperimentConfig

# NOTE: Do NOT import from robometer.utils.setup_utils here - it loads Unsloth which breaks JAX
# from robometer.utils.setup_utils import setup_model_and_processor
from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env, DummyDSRLEnv, make_simpler_env, make_remote_robot_env
from robometer_policy_learning.utils.env_utils import make_env

from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper
from robometer_policy_learning.utils.rate_utils import RateMeter, RateLimiter

# Note: Distributed components (LearnerServer, PolicyClient, StreamingBufferAdapter)
# are imported lazily inside the async mode functions to avoid gRPC/JAX conflicts

GT_REW_SUCCESS_BONUS = 0
ROBOMETER_SUCCESS_BONUS = 64.0
RELATIVE_REW_SUCCESS_BONUS = 1.0


def log_buffer_statistics(buffer, logger, step):
    """
    Log buffer statistics, including success/failure buffer metrics if applicable.

    Args:
        buffer: Replay buffer instance
        logger: Logger instance (e.g., WandbLogger)
        step: Current training step for logging
    """
    if isinstance(buffer, SuccessFailureReplayBuffer):
        stats = buffer.get_statistics()

        # Log to console
        rprint(f"\n[Buffer Stats @ step {step}]")
        rprint(f"  Total episodes: {stats['total_episodes']}")
        rprint(f"  Success rate: {stats['success_rate']:.1%}")
        rprint(f"  Success buffer: {stats['success_buffer_size']} transitions")
        rprint(f"  Failure buffer: {stats['failure_buffer_size']} transitions")
        rprint(f"  Pending episodes: {stats['pending_episodes']}")

        # Log to wandb
        logger.log(
            {
                "buffer/total_episodes": stats["total_episodes"],
                "buffer/success_rate": stats["success_rate"],
                "buffer/successful_episodes": stats["successful_episodes"],
                "buffer/failed_episodes": stats["failed_episodes"],
                "buffer/success_buffer_size": stats["success_buffer_size"],
                "buffer/failure_buffer_size": stats["failure_buffer_size"],
                "buffer/pending_episodes": stats["pending_episodes"],
                "buffer/total_size": stats["total_size"],
            },
            step=step,
        )
    else:
        # Standard buffer - just log size
        logger.log(
            {
                "buffer/total_size": len(buffer),
            },
            step=step,
        )


def build_light_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Build a lightweight state dict containing only trainable + essential inference modules."""
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_prefixes = set(".".join(n.split(".")[:-1]) for n in trainable_param_names)
    sd = model.state_dict()
    include_prefixes = {"feature_extractor", "obs_projection", "position_embedding"}

    def keep_key(k: str) -> bool:
        for pref in trainable_prefixes:
            if pref and (k == pref or k.startswith(pref + ".")):
                return True
        for pref in include_prefixes:
            if k == pref or k.startswith(pref + "."):
                return True
        return False

    light = {k: v for k, v in sd.items() if keep_key(k)}
    return light if light else sd


@dataclass
class DSRLTrainingComponents:
    """Holds all components needed for DSRL training."""

    # Core
    cfg: DictConfig
    device: torch.device

    # Models
    dinov2_model: Any
    dinov2_processor: Any
    sentence_model: Any
    pi0_wrapper: Any

    # Environments
    env: Any
    eval_env: Any
    remove_obs_keys: list
    dummy_dsrl_env: Any
    reward_relabeling_keys: list

    # Actor/Critic
    actor: Any
    critic: Any
    v_net: Any

    # Algorithm
    algorithm: Any
    buffer: Any

    # Logging
    wandb_logger: Any
    save_dir: str
    save_interval: int

    # Reward model (optional)
    reward_model: Any = None
    reward_model_exp_cfg: Any = None
    use_gt_rewards: bool = True
    use_relative_rewards: bool = False

    # Computed values
    gamma: float = 0.99


def setup_dsrl_env(cfg: DictConfig, dinov2_model, dinov2_processor, sentence_model, device):
    """
    Setup DSRL environment based on config.
    Returns env, eval_env, remove_obs_keys.
    """
    extra_keys_to_drop = cfg.env.get("extra_keys_to_drop", [])

    if "libero" in cfg.env.env_name:
        # Build async reward relabel kwargs from reward_model config
        async_reward_relabel_kwargs = None
        if cfg.env.get("use_async_reward_relabel", False) and cfg.reward_model is not None:
            async_reward_relabel_kwargs = {
                "server_address": cfg.reward_model.async_reward_relabel_server_address,
                "batch_size": cfg.env.get("reward_relabel_batch_size", 32),
                "max_queue_size": 100,  # Default client queue size
                "timeout": 60.0,  # Default client timeout
                "flush_interval": 0.1,  # Default client flush interval
                "success_detection_duration": cfg.reward_model.success_detection_duration,
                "success_detection_threshold": cfg.reward_model.success_detection_threshold,
                "use_relative_rewards": cfg.reward_model.use_relative_rewards,
                "action_exec_len": cfg.dsrl.action_exec_len,  # For DSRL mode: detect chunking and relabel specific indices
            }

        env, remove_obs_keys = setup_libero_env(
            cfg.env.env_name,
            cfg.env.task_id,
            cfg.training.num_envs,
            dinov2_model,
            dinov2_processor,
            sentence_model,
            device,
            cfg.training.seed,
            max_episode_steps=cfg.env.max_episode_steps,
            image_keys=cfg.env.image_keys,
            extra_keys_to_drop=extra_keys_to_drop,
            async_reward_relabel_kwargs=async_reward_relabel_kwargs,
        )
        # Use synchronous reward relabeling for eval_env to prevent queue overflow
        # Evaluation doesn't need to be fast, and sync mode prevents queue buildup
        eval_async_reward_relabel_kwargs = None
        if async_reward_relabel_kwargs is not None:
            eval_async_reward_relabel_kwargs = async_reward_relabel_kwargs.copy()
            eval_async_reward_relabel_kwargs["sync_mode"] = True  # Use sync mode for evaluation
        
        eval_env, _ = setup_libero_env(
            cfg.env.env_name,
            cfg.env.task_id,
            1,
            dinov2_model,
            dinov2_processor,
            sentence_model,
            device,
            cfg.training.seed,
            max_episode_steps=cfg.env.max_episode_steps,
            image_keys=cfg.env.image_keys,
            extra_keys_to_drop=extra_keys_to_drop,
            async_reward_relabel_kwargs=eval_async_reward_relabel_kwargs,
        )
    elif "SIMPLER" in cfg.env.env_name:
        simpler_cfg = OmegaConf.select(cfg, "simpler", default=None)
        use_dense_reward = simpler_cfg.get("use_dense_reward", False) if simpler_cfg else False
        simpler_host = simpler_cfg.get("host", "0.0.0.0") if simpler_cfg else "0.0.0.0"
        simpler_port = simpler_cfg.get("port", 6000) if simpler_cfg else 6000

        env, remove_obs_keys = make_simpler_env(
            cfg.training.num_envs,
            extra_keys_to_drop=extra_keys_to_drop,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            sentence_model=sentence_model,
            device=device,
            use_dense_reward=use_dense_reward,
            host=simpler_host,
            port=simpler_port,
            num_stages=cfg.env.get("num_stages", 1),
        )
        eval_env = env
    elif cfg.env.env_name == "REMOTE_ROBOT":
        remote_cfg = OmegaConf.select(cfg, "remote_robot", default=None)
        if remote_cfg is None:
            remote_host = "localhost"
            remote_port = 6000
        else:
            # Handle both dict and dataclass formats
            if hasattr(remote_cfg, "host"):
                remote_host = remote_cfg.host
                remote_port = remote_cfg.port
            else:
                remote_host = remote_cfg.get("host", "localhost")
                remote_port = remote_cfg.get("port", 6000)

        logger.info(f"Connecting to remote robot at {remote_host}:{remote_port}")

        # Build async reward relabel kwargs from reward_model config
        async_reward_relabel_kwargs = None
        if cfg.env.get("use_async_reward_relabel", False) and cfg.reward_model is not None:
            async_reward_relabel_kwargs = {
                "server_address": cfg.reward_model.async_reward_relabel_server_address,
                "batch_size": cfg.env.get("reward_relabel_batch_size", 32),
                "max_queue_size": 100,  # Default client queue size
                "timeout": 60.0,  # Default client timeout
                "flush_interval": 0.1,  # Default client flush interval
                "success_detection_duration": cfg.reward_model.success_detection_duration,
                "success_detection_threshold": cfg.reward_model.success_detection_threshold,
                "use_relative_rewards": cfg.reward_model.use_relative_rewards,
                "action_exec_len": cfg.dsrl.action_exec_len,  # For DSRL mode: detect chunking and relabel specific indices
            }

        env, remove_obs_keys = make_remote_robot_env(
            n_envs=cfg.training.num_envs,
            host=remote_host,
            port=remote_port,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            sentence_model=sentence_model,
            device=device,
            obs_format=cfg.env.obs_format,
            image_keys=cfg.env.image_keys,
            extra_keys_to_drop=extra_keys_to_drop,
            num_stages=cfg.env.get("num_stages", 1),
            async_reward_relabel_kwargs=async_reward_relabel_kwargs,
        )

        eval_env = env
        # Use synchronous reward relabeling for eval_env to prevent queue overflow
        # Evaluation doesn't need to be fast, and sync mode prevents queue buildup
        # eval_async_reward_relabel_kwargs = None
        # if async_reward_relabel_kwargs is not None:
        #     eval_async_reward_relabel_kwargs = async_reward_relabel_kwargs.copy()
        #     eval_async_reward_relabel_kwargs["sync_mode"] = True  # Use sync mode for evaluation

        # eval_env, _ = make_remote_robot_env(
        #     n_envs=1,
        #     host=remote_host,
        #     port=remote_port,
        #     dinov2_model=dinov2_model,
        #     dinov2_processor=dinov2_processor,
        #     sentence_model=sentence_model,
        #     device=device,
        #     obs_format=cfg.env.obs_format,
        #     image_keys=cfg.env.image_keys,
        #     extra_keys_to_drop=extra_keys_to_drop,
        #     async_reward_relabel_kwargs=eval_async_reward_relabel_kwargs,
        # )
    else:
        env, eval_env = make_env(
            env_name=cfg.env.env_name,
            num_envs=cfg.training.num_envs,
            chunk_size=cfg.training.chunk_size,
            use_full_state=cfg.env.use_full_state,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            device=device,
            sentence_model=sentence_model if not cfg.env.use_full_state else None,
            render_mode="rgb_array",
            terminate_on_success=True,
        )
        remove_obs_keys = ["image"]
        
    return env, eval_env, remove_obs_keys


@dataclass
class DSRLWorkerComponents:
    """Holds minimal components needed for rollout/eval workers."""

    cfg: DictConfig
    device: torch.device

    # Models
    dinov2_model: Any
    dinov2_processor: Any
    sentence_model: Any
    pi0_wrapper: Any

    # Environments
    env: Any
    eval_env: Any
    remove_obs_keys: list
    dummy_dsrl_env: Any

    # Actor only (no critic for inference)
    actor: Any

    # Computed values
    gamma: float = 0.99


def setup_dsrl_worker(cfg: DictConfig, mode: str = "rollout") -> DSRLWorkerComponents:
    """
    Lightweight setup for rollout/eval workers.

    Unlike setup_dsrl_training, this doesn't create buffers, algorithms, or wandb loggers.
    Workers connect to the learner to get weights and optionally join its wandb run.

    Args:
        cfg: Hydra config
        mode: Worker mode (rollout, eval)

    Returns:
        DSRLWorkerComponents with minimal components for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{mode.upper()}] Using device: {device}")

    # Setup DINO and sentence models
    dinov2_model = AutoModel.from_pretrained(cfg.model.dinov2_model).to(device).eval()
    dinov2_processor = AutoImageProcessor.from_pretrained(cfg.model.dinov2_model)
    if cfg.model.sentence_model is not None:
        sentence_model = SentenceTransformer(cfg.model.sentence_model)
    else:
        sentence_model = None

    # Load Pi0
    logger.info(f"Loading Pi0 from {cfg.dsrl.pi0_checkpoint}")
    pi0_wrapper = Pi0Wrapper(cfg.dsrl.pi0_checkpoint, device=str(device))

    # Setup environments
    if cfg.training.num_envs > 1:
        raise ValueError("num_envs must be 1 for DSRL for now")

    env, eval_env, remove_obs_keys = setup_dsrl_env(cfg, dinov2_model, dinov2_processor, sentence_model, device)

    # Get observation and action spaces
    if hasattr(env, "single_observation_space"):
        observation_space = env.single_observation_space
    else:
        observation_space = env.observation_space

    if hasattr(env, "single_action_space"):
        action_space = env.single_action_space
    else:
        action_space = env.action_space

    # Create dummy DSRL env
    dummy_dsrl_env = DummyDSRLEnv(
        observation_space,
        action_space,
        pi0_wrapper,
        cfg.dsrl.noise_dim,
        chunk_size=cfg.training.chunk_size,
        action_bound=cfg.dsrl.noise_action_bound,
        use_vlm_features=cfg.dsrl.get("use_vlm_features", True),
    )
    dsrl_observation_space = dummy_dsrl_env.observation_space
    dsrl_action_space = dummy_dsrl_env.action_space

    # Build actor only (no critic needed for inference)
    actor, _, _ = build_actor_critic_models(dsrl_observation_space, dsrl_action_space, cfg, device, remove_obs_keys)
    logger.info(f"[{mode.upper()}] Actor: {actor.__class__.__name__}")

    # Compute gamma with action_exec_len
    gamma = cfg.online_algorithm.gamma**cfg.dsrl.action_exec_len
    logger.info(f"Overriding DSRL worker gamma based on gamma**cfg.action_exec_len: new gamma = {gamma}")

    return DSRLWorkerComponents(
        cfg=cfg,
        device=device,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        sentence_model=sentence_model,
        pi0_wrapper=pi0_wrapper,
        env=env,
        eval_env=eval_env,
        remove_obs_keys=remove_obs_keys,
        dummy_dsrl_env=dummy_dsrl_env,
        actor=actor,
        gamma=gamma,
    )


def setup_dsrl_training(
    cfg: DictConfig,
    mode: str = "serial",
    create_wandb_logger: bool = True,
    wandb_prefix: str = "offline",
) -> DSRLTrainingComponents:
    """
    Comprehensive setup for DSRL training.

    This function creates all components needed for training, shared across
    serial, learner, rollout, and eval modes.

    Args:
        cfg: Hydra config
        mode: Training mode (serial, learner, rollout, eval)
        create_wandb_logger: Whether to create a wandb logger
        wandb_prefix: Prefix for wandb logging

    Returns:
        DSRLTrainingComponents with all initialized components
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{mode.upper()}] Using device: {device}")

    # Load Pi0
    logger.info(f"Loading Pi0 from {cfg.dsrl.pi0_checkpoint}")
    pi0_wrapper = Pi0Wrapper(cfg.dsrl.pi0_checkpoint, device=str(device))

    # Get Hydra output directory
    hydra_cfg = HydraConfig.get()
    output_dir = os.path.abspath(hydra_cfg.runtime.output_dir)
    
    # Setup wandb logger
    wandb_logger = None
    if create_wandb_logger:
        string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{cfg.logging.wandb_name}_{string_time}"
        wandb_logger = WandbLogger(
            exp_name=exp_name,
            offline=cfg.logging.wandb_offline,
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            log_dir=f"{cfg.logging.wandb_log_dir_base}/{cfg.logging.wandb_name}",
            prefix=wandb_prefix,
        )
        # removes date and time from the output_dir (preserve absolute path)
        output_path = os.path.abspath(output_dir)
        output_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), exp_name)

    save_dir = f"{output_dir}/checkpoints"
    save_interval = cfg.training.save_interval
    os.makedirs(save_dir, exist_ok=True)

    # Setup loguru logging
    log_level = cfg.logging.log_level if hasattr(cfg, "logging") and hasattr(cfg.logging, "log_level") else "INFO"
    setup_loguru_logging(log_level=log_level, output_dir=output_dir)

    # Setup DINO and sentence models
    dinov2_model, dinov2_processor = None, None
    if cfg.model.dinov2_model:
        dinov2_model = AutoModel.from_pretrained(cfg.model.dinov2_model).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(cfg.model.dinov2_model)
    if cfg.model.sentence_model is not None:
        sentence_model = SentenceTransformer(cfg.model.sentence_model)
    else:
        sentence_model = None

    # Check if using async reward relabeling (reward model runs on separate server)
    reward_model_cfg = OmegaConf.select(cfg, "reward_model", default=None)
    use_async_reward_relabel = cfg.reward_model.use_async_reward_relabel if reward_model_cfg is not None else False

    # Reward model is loaded on the server, not here
    reward_model = None
    reward_model_exp_cfg = None
    use_gt_rewards = cfg.env.use_gt_rewards if not use_async_reward_relabel else True

    # use_relative_rewards can be set in config.yaml under reward_model, default to False
    use_relative_rewards = (
        reward_model_cfg.use_relative_rewards
        if reward_model_cfg is not None and hasattr(reward_model_cfg, "use_relative_rewards")
        else False
    )

    if use_async_reward_relabel:
        logger.info("Using async reward relabeling - reward model runs on separate server")
    elif reward_model_cfg is None:
        logger.info("Using ground truth rewards")
    else:
        logger.info("Reward model should be loaded locally (not using async relabeling)")

    # Setup environments
    if cfg.training.num_envs > 1:
        raise ValueError("num_envs must be 1 for DSRL for now")

    env, eval_env, remove_obs_keys = setup_dsrl_env(cfg, dinov2_model, dinov2_processor, sentence_model, device)

    # Get observation and action spaces
    if hasattr(env, "single_observation_space"):
        observation_space = env.single_observation_space
    else:
        observation_space = env.observation_space

    if hasattr(env, "single_action_space"):
        action_space = env.single_action_space
    else:
        action_space = env.action_space

    # Create dummy DSRL env
    dummy_dsrl_env = DummyDSRLEnv(
        observation_space,
        action_space,
        pi0_wrapper,
        noise_dim=cfg.dsrl.noise_dim,
        chunk_size=cfg.training.chunk_size,
        action_bound=cfg.dsrl.noise_action_bound,
        use_vlm_features=cfg.dsrl.get("use_vlm_features", True),
    )
    dsrl_observation_space = dummy_dsrl_env.observation_space
    dsrl_action_space = dummy_dsrl_env.action_space

    # Build models
    actor, critic, v_net = build_actor_critic_models(
        dsrl_observation_space, dsrl_action_space, cfg, device, remove_obs_keys
    )
    logger.info(f"Actor: {actor.__class__.__name__}")
    logger.info(f"Critic: {critic.__class__.__name__}")
    
    # import ipdb; ipdb.set_trace()
    # Define success bonus
    # if reward_model_cfg is None:
    success_bonus_fn = SuccessBonusTransform(GT_REW_SUCCESS_BONUS)
    # else:
    #     success_bonus_fn = (
    #         SuccessBonusTransform(RELATIVE_REW_SUCCESS_BONUS)
    #         if use_relative_rewards
    #         else SuccessBonusTransform(ROBOMETER_SUCCESS_BONUS)
    #     )

    # Create sampler based on whether we're using async reward relabeling
    # If using env-level async reward relabeling, use RelabeledOnlySampler to ensure
    # we only train on transitions that have been properly relabeled
    if cfg.env.get("use_async_reward_relabel", False):
        from robometer_policy_learning.buffers.samplers import RelabeledOnlySampler
        min_relabeled_ratio = cfg.buffer.get("min_relabeled_ratio", 0.1)
        sampler = RelabeledOnlySampler(min_relabeled_ratio=min_relabeled_ratio)
        logger.info(f"Using RelabeledOnlySampler (min_relabeled_ratio={min_relabeled_ratio})")
    else:
        sampler = RandomSampler()
        logger.info("Using RandomSampler")

    # Get success/failure buffer config
    use_success_fail_buffer = OmegaConf.select(cfg.buffer, "use_success_fail_buffer", default=False)
    success_fail_sample_ratio = OmegaConf.select(cfg.buffer, "success_fail_sample_ratio", default=0.5)

    # Create buffer
    reward_relabeling_keys = OmegaConf.select(cfg.env, "image_keys", default=["image"])

    # Get async reward relabeling server address (if using async relabeling)
    # Client parameters (queue_size, timeout, flush_interval) are handled by create_buffer with defaults
    reward_relabel_address = None
    if use_async_reward_relabel and reward_model_cfg is not None:
        reward_relabel_address = reward_model_cfg.async_reward_relabel_server_address
        logger.info(f"Connecting to async reward relabeling server at {reward_relabel_address}")

    buffer = create_buffer(
        sampler=sampler,
        use_gt_rewards=use_gt_rewards,
        use_relative_rewards=use_relative_rewards,
        capacity=cfg.buffer.capacity,
        remove_obs_keys=remove_obs_keys,
        post_transforms=[success_bonus_fn],
        use_eval_server=False,  # Not used when async relabeling
        eval_server_url=None,
        eval_server_timeout=120.0,
        use_async_reward_relabel=use_async_reward_relabel,
        use_success_fail_buffer=use_success_fail_buffer,
        success_fail_sample_ratio=success_fail_sample_ratio,
        reward_relabeling_keys=reward_relabeling_keys,
        reward_relabel_address=reward_relabel_address,
    )

    if use_success_fail_buffer:
        logger.info(f"✓ SuccessFailureReplayBuffer created (sample_ratio={success_fail_sample_ratio})")
    else:
        logger.info(f"✓ Standard buffer created (capacity={cfg.buffer.capacity})")

    # Set buffer reference on async reward relabeling env wrappers (if using env-level relabeling)
    # This enables retroactive reward updates in the buffer when async relabeling completes
    if cfg.env.get("use_async_reward_relabel", False):
        from robometer_policy_learning.envs.async_reward_relabel_wrapper import AsyncRewardRelabelEnvWrapper
        import gymnasium as gym

        def set_buffer_on_env_wrappers(env_instance, buffer_ref):
            """Recursively find and set buffer on AsyncRewardRelabelEnvWrapper instances."""
            # Check if this is the wrapper we're looking for
            if isinstance(env_instance, AsyncRewardRelabelEnvWrapper):
                env_instance.set_buffer(buffer_ref)
                return

            # For vectorized envs (SyncVectorEnv), access individual envs
            if isinstance(env_instance, gym.vector.VectorEnv):
                if hasattr(env_instance, "envs"):
                    for sub_env in env_instance.envs:
                        set_buffer_on_env_wrappers(sub_env, buffer_ref)

            # For wrappers, check .env attribute
            if hasattr(env_instance, "env"):
                set_buffer_on_env_wrappers(env_instance.env, buffer_ref)

        # Set buffer on training env only (not eval_env - evaluation shouldn't update the buffer)
        set_buffer_on_env_wrappers(env, buffer)
        # Note: We intentionally do NOT set buffer on eval_env to prevent buffer updates during evaluation
        logger.info("✓ Set buffer reference on async reward relabeling env wrappers for retroactive updates (training env only)")

    # Create algorithm
    if cfg.online_algorithm is None:
        raise ValueError(
            "online_algorithm config is required but is None. "
            "Make sure the config file has 'algorithm@online_algorithm: libero_dsrl_sac' (or similar) in defaults."
        )
    online_algo_dict = OmegaConf.to_container(cfg.online_algorithm, resolve=True)
    if cfg.alg.online_alg_name.lower() == "sac":
        online_algo_config = SACConfig(**online_algo_dict)
    else:
        raise ValueError(f"Unknown online algorithm: {cfg.alg.online_alg_name}")

    # Compute gamma with action_exec_len
    online_algo_config.env = dummy_dsrl_env
    online_algo_config.actor = actor
    online_algo_config.critic = critic
    online_algo_config.buffer = buffer
    online_algo_config.action_space = dsrl_action_space
    online_algo_config.logger = wandb_logger
    online_algo_config.compute_chunked_gamma = False
    logger.info("Disabling chunked gamma computation for DSRL on purpose as it uses 1-step rewards")
    online_algo_config.gamma = online_algo_config.gamma**cfg.dsrl.action_exec_len
    logger.info("Overriding DSRL gamma based on horizon of the task")
    logger.info(f"Online algorithm modified gamma^action_exec_len: {online_algo_config.gamma}")

    algorithm = SAC(online_algo_config)

    # Load checkpoint if specified
    if cfg.training.load_dir is not None:
        logger.info(f"Loading checkpoint from {cfg.training.load_dir}")
        load_checkpoint(algorithm, cfg.training.load_dir)
        logger.info("Resumed from checkpoint")

    logger.info(f"Algorithm: {algorithm.__class__.__name__}")

    # Log config
    if wandb_logger:
        rprint(OmegaConf.to_container(cfg))
        wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

    return DSRLTrainingComponents(
        cfg=cfg,
        device=device,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        sentence_model=sentence_model,
        pi0_wrapper=pi0_wrapper,
        env=env,
        eval_env=eval_env,
        remove_obs_keys=remove_obs_keys,
        reward_relabeling_keys=reward_relabeling_keys,
        dummy_dsrl_env=dummy_dsrl_env,
        actor=actor,
        critic=critic,
        v_net=v_net,
        algorithm=algorithm,
        buffer=buffer,
        wandb_logger=wandb_logger,
        save_dir=save_dir,
        save_interval=save_interval,
        reward_model=reward_model,
        reward_model_exp_cfg=reward_model_exp_cfg,
        use_gt_rewards=use_gt_rewards,
        use_relative_rewards=use_relative_rewards,
        gamma=online_algo_config.gamma,
    )


def run_learner_dsrl(cfg: DictConfig):
    """
    Learner mode for DSRL distributed training.
    Runs the training loop and serves gRPC for experience ingestion and policy distribution.
    """
    # Lazy import to avoid gRPC/JAX conflicts
    from robometer_policy_learning.distributed.servers.learner_server import LearnerServer

    # Use shared setup
    components = setup_dsrl_training(cfg, mode="learner", wandb_prefix="learner", create_wandb_logger=False)

    # Extract what we need
    env = components.env
    eval_env = components.eval_env
    actor = components.actor
    buffer = components.buffer
    algorithm = components.algorithm
    wandb_logger = components.wandb_logger
    save_dir = components.save_dir
    remove_obs_keys = components.remove_obs_keys
    reward_relabeling_keys = components.reward_relabeling_keys
    reward_model = components.reward_model

    # Policy info provider for gRPC
    def policy_info() -> Dict[str, Any]:
        return {
            "obs_keys": list(remove_obs_keys) if remove_obs_keys else [],
            "reward_relabeling_keys": list(reward_relabeling_keys) if reward_relabeling_keys else [],
            "chunk_size": cfg.training.chunk_size,
            "wandb_run_id": wandb_logger.logger.id if wandb_logger.logger else "",
            "wandb_project": wandb_logger.project or cfg.logging.wandb_project,
            "wandb_entity": cfg.logging.wandb_entity,
            "action_exec_len": cfg.dsrl.action_exec_len,
            "noise_dim": cfg.dsrl.noise_dim,
        }

    # Start gRPC server
    server_host = cfg.distributed.learner_server.host
    server_port = cfg.distributed.learner_server.port

    server = LearnerServer(
        buffer=buffer,
        reward_model=reward_model,
        host=server_host,
        port=server_port,
        max_msg_mb=cfg.distributed.learner_server.max_msg_mb,
        policy_info_provider=policy_info,
    )
    server.start()
    logger.success(f"[LEARNER] Started gRPC server at {server_host}:{server_port}")

    # Publish initial actor params
    initial_sd = build_light_state_dict(actor)
    server.set_actor_state_dict(initial_sd)

    # Enable ingestion
    server.set_ingest_enabled(True)
    logger.success("[LEARNER] Experience ingestion enabled")

    # Training loop
    train_meter = RateMeter(name="train", alpha=0.1)
    train_rl = RateLimiter(cfg.distributed.train_target_hz)

    def train_loop():
        last_log_t = time.time()
        last_step = getattr(algorithm, "_n_updates", 0)
        last_buffer_size = 0
        policy_step = 0

        logger.info("[LEARNER] Training loop started, waiting for data...")

        while True:
            current_buffer_size = len(buffer)

            # Only train if there's enough data
            if current_buffer_size < cfg.online_algorithm.batch_size:
                time.sleep(0.5)
                continue

            # Only train if there's new data
            if current_buffer_size > last_buffer_size or last_step == 0:
                try:
                    t0 = time.time()
                    metrics = algorithm.train_step(logging_prefix="online/policy")
                    dt_update = time.time() - t0
                    train_meter.record(dt_update)
                    last_buffer_size = current_buffer_size
                    policy_step += 1
                except Exception as e:
                    logger.exception(f"[LEARNER] Training step failed: {e}")
                    time.sleep(1)
                    continue
            else:
                time.sleep(0.1)
                continue

            step = getattr(algorithm, "_n_updates", 0)
            now = time.time()
            log_interval = cfg.distributed.log_interval

            if now - last_log_t >= log_interval:
                dt = max(now - last_log_t, 1e-6)
                dstep = step - last_step
                sps = dstep / dt
                avg_sec = train_meter.avg_seconds_per_update
                hz = train_meter.updates_per_second
                detail = " ".join([f"{k}={v:0.3f}" for k, v in (metrics or {}).items() if isinstance(v, (int, float))])
                logger.info(
                    f"[LEARNER] step={step} sps={sps:0.2f} avg_dt={avg_sec:0.4f}s hz={hz:0.2f} buffer={len(buffer)} {detail}"
                )
                last_log_t = now
                last_step = step

            # Update actor weights on server
            actor_update_freq = cfg.distributed.actor_update_freq
            if step % actor_update_freq == 0:
                sd = build_light_state_dict(algorithm.actor)
                server.set_actor_state_dict(sd)

            # Save checkpoint
            save_interval = (
                cfg.distributed.save_interval
                if cfg.distributed.save_interval is not None
                else cfg.training.save_interval
            )
            if step > 0 and step % save_interval == 0:
                save_checkpoint(algorithm, save_dir, step)

            train_rl.throttle()

    threading.Thread(target=train_loop, daemon=True).start()

    logger.info(f"[LEARNER] Async learner running at {server_host}:{server_port}")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("[LEARNER] Shutting down...")
    finally:
        env.close()
        if eval_env is not env:
            eval_env.close()
        try:
            wandb_logger.finish()
        except Exception:
            pass


def run_rollout_dsrl(cfg: DictConfig):
    """
    Rollout worker mode for DSRL distributed training.
    Collects experience and streams to the learner.
    """
    # Lazy imports to avoid gRPC/JAX conflicts
    from robometer_policy_learning.distributed.clients.policy_client import PolicyClient
    from robometer_policy_learning.distributed.clients.streaming_buffer_adapter import StreamingBufferAdapter

    rollout_log = logger.bind(process_name="ROLLOUT")

    # Connect to learner first
    learner_address = cfg.distributed.learner_server.address
    pc = PolicyClient(learner_address, ready_log_prefix="[ROLLOUT]", ready_timeout_per_attempt=5.0)
    info = pc.get_policy_info(block=True)

    rollout_log.info(f"[ROLLOUT] Connected to learner at {learner_address}")
    rollout_log.info(f"[ROLLOUT] Policy info: {info}")

    # Join wandb run if available
    wandb_run_id = info.get("wandb_run_id", "")
    wandb_project = info.get("wandb_project", cfg.logging.wandb_project)
    wandb_entity = info.get("wandb_entity", cfg.logging.wandb_entity)

    rollout_logger = None
    if wandb_run_id:
        try:
            label = f"rollout_{socket.gethostname()}_{os.getpid()}"
            rb_settings = None
            if wandb is not None:
                rb_settings = wandb.Settings(
                    mode="shared",
                    x_primary=False,
                    x_label=label,
                    x_update_finish_state=False,
                )
            rollout_logger = WandbLogger(
                exp_name="rollout_worker",
                id=wandb_run_id,
                project=wandb_project,
                entity=wandb_entity,
                prefix="rollout",
                job_type="rollout",
                tags=["rollout"],
                **({"settings": rb_settings} if rb_settings is not None else {}),
            )
            rollout_log.info(f"[ROLLOUT] Joined wandb run: {wandb_run_id}")
        except Exception as e:
            rollout_log.warning(f"[ROLLOUT] Failed to join wandb run: {e}")

    # Use shared worker setup
    components = setup_dsrl_worker(cfg, mode="rollout")
    env = components.env
    actor = components.actor
    pi0_wrapper = components.pi0_wrapper
    dummy_dsrl_env = components.dummy_dsrl_env
    gamma = components.gamma
    device = components.device

    # Wait for initial weights from learner
    rollout_log.info(f"[ROLLOUT] Waiting for initial weights from learner...")
    initial_sd = pc.fetch_latest(block=True)
    initial_fp = initial_sd.pop("__fingerprint__", None)
    actor.load_state_dict(initial_sd)
    if initial_fp:
        rollout_log.success(f"[ROLLOUT] Initial policy fingerprint: {initial_fp}")

    # Create streaming buffer adapter
    streaming_buffer = StreamingBufferAdapter(
        learner_address,
        flush_every=cfg.distributed.rollout.flush_every,
        max_message_mb=cfg.distributed.rollout.max_message_mb,
    )

    # Create inference actor (separate copy for safe inference)
    inference_actor = type(actor)(actor.config if hasattr(actor, "config") else None).to(device)
    inference_actor.load_state_dict(actor.state_dict())
    inference_actor.eval()
    for param in inference_actor.parameters():
        param.requires_grad_(False)

    # Create DSRL rollout worker
    worker = DSRLRolloutWorker(
        env=env,
        buffer=streaming_buffer,
        pi0_wrapper=pi0_wrapper,
        action_exec_len=cfg.dsrl.action_exec_len,
        gamma=gamma,
        actor=inference_actor,
        num_rollouts=cfg.distributed.rollout.num_rollouts,
        num_envs=cfg.training.num_envs,
        device=device,
        count_by="step",
        dummy_dsrl_env=dummy_dsrl_env,
    )

    # Track fingerprint
    last_fp = initial_fp or None

    def refresh_weights():
        """Background thread to refresh actor weights."""
        nonlocal last_fp
        while True:
            try:
                new_state_dict = pc.fetch_latest()
                fp = new_state_dict.pop("__fingerprint__", None)

                import copy

                inference_state_dict = copy.deepcopy(new_state_dict)
                inference_actor.load_state_dict(inference_state_dict)
                inference_actor.eval()
                for param in inference_actor.parameters():
                    param.requires_grad_(False)

                if fp is not None:
                    if last_fp is None:
                        rollout_log.info(f"[ROLLOUT] Policy fingerprint set: {fp}")
                    elif fp != last_fp:
                        rollout_log.success(f"[ROLLOUT] Policy updated: {last_fp} -> {fp}")
                    last_fp = fp
            except Exception as e:
                rollout_log.warning(f"[ROLLOUT] Weight update failed: {e}")

            time.sleep(cfg.distributed.rollout.refresh_secs)

    threading.Thread(target=refresh_weights, daemon=True).start()
    rollout_log.info(f"[ROLLOUT] Starting rollout streaming to {learner_address}...")

    rollout_failures = 0
    max_consecutive_failures = 5
    rollout_count = 0
    last_log_time = time.time()
    rollout_step = 0
    prev_total_steps = getattr(worker.episode_tracker, "total_steps", 0)
    rollout_meter = RateMeter(name="rollout", alpha=0.1)
    rollout_rl = RateLimiter(cfg.distributed.rollout.target_hz)

    while True:
        try:
            with torch.no_grad():
                t0 = time.time()
                rollout_metrics = worker.run()
                dt_cycle = time.time() - t0
                rollout_meter.record(dt_cycle)
            streaming_buffer.flush()
            rollout_failures = 0
            rollout_count += 1

            if rollout_logger:
                current_time = time.time()
                dt = max(current_time - last_log_time, 1e-6)
                total_steps_now = getattr(worker.episode_tracker, "total_steps", prev_total_steps)
                steps_collected = max(0, total_steps_now - prev_total_steps)
                steps_per_sec = steps_collected / dt
                prev_total_steps = total_steps_now
                last_log_time = current_time

                to_log = {
                    **(rollout_metrics or {}),
                    "steps_collected": float(steps_collected),
                    "steps_per_sec": float(steps_per_sec),
                    "rollout_cycles": float(rollout_count),
                    "avg_seconds_per_cycle": float(rollout_meter.avg_seconds_per_update),
                    "cycles_per_sec": float(rollout_meter.updates_per_second),
                }
                rollout_logger.log_dict(to_log, step=rollout_step, prefix="rollout")
                rollout_step += 1

            rollout_rl.throttle()

        except RuntimeError as e:
            rollout_failures += 1
            rollout_log.error(f"[ROLLOUT] Runtime error (attempt {rollout_failures}/{max_consecutive_failures}): {e}")

            if rollout_failures >= max_consecutive_failures:
                rollout_log.error(f"[ROLLOUT] Too many consecutive failures, exiting...")
                raise

            time.sleep(1)

        except Exception as e:
            rollout_failures += 1
            rollout_log.exception(
                f"[ROLLOUT] Unexpected error (attempt {rollout_failures}/{max_consecutive_failures}): {e}"
            )

            if rollout_failures >= max_consecutive_failures:
                rollout_log.error(f"[ROLLOUT] Too many consecutive failures, exiting...")
                raise

            time.sleep(2)


def run_eval_dsrl(cfg: DictConfig):
    """
    Evaluation worker mode for DSRL distributed training.
    Periodically evaluates the latest policy from the learner.
    """
    # Lazy import to avoid gRPC/JAX conflicts
    from robometer_policy_learning.distributed.clients.policy_client import PolicyClient

    eval_log = logger.bind(process_name="EVAL")

    # Connect to learner first
    learner_address = cfg.distributed.learner_server.address
    pc = PolicyClient(learner_address, ready_log_prefix="[EVAL]", ready_timeout_per_attempt=5.0)
    info = pc.get_policy_info(block=True)

    eval_log.info(f"[EVAL] Connected to learner at {learner_address}")

    # Join wandb run if available
    wandb_run_id = info.get("wandb_run_id", "")
    wandb_project = info.get("wandb_project", cfg.logging.wandb_project)
    wandb_entity = info.get("wandb_entity", cfg.logging.wandb_entity)

    eval_logger = None
    if wandb_run_id:
        try:
            label = f"eval_{socket.gethostname()}_{os.getpid()}"
            ev_settings = None
            if wandb is not None:
                ev_settings = wandb.Settings(
                    mode="shared",
                    x_primary=False,
                    x_label=label,
                    x_update_finish_state=False,
                )
            eval_logger = WandbLogger(
                exp_name="eval_worker",
                id=wandb_run_id,
                project=wandb_project,
                entity=wandb_entity,
                prefix="eval",
                job_type="eval",
                tags=["eval"],
                **({"settings": ev_settings} if ev_settings is not None else {}),
            )
            eval_log.info(f"[EVAL] Joined wandb run: {wandb_run_id}")
        except Exception as e:
            eval_log.warning(f"[EVAL] Failed to join wandb run: {e}")

    # Use shared worker setup
    components = setup_dsrl_worker(cfg, mode="eval")
    eval_env = components.eval_env
    actor = components.actor
    pi0_wrapper = components.pi0_wrapper
    gamma = components.gamma
    device = components.device

    # Wait for initial weights from learner
    eval_log.info(f"[EVAL] Waiting for initial weights from learner...")
    initial_sd = pc.fetch_latest(block=True)
    initial_fp = initial_sd.pop("__fingerprint__", None)
    actor.load_state_dict(initial_sd)
    if initial_fp:
        eval_log.success(f"[EVAL] Initial policy fingerprint: {initial_fp}")

    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)

    # Create DSRL evaluation worker
    worker = DSRLEvaluationWorker(
        eval_env=eval_env,
        device=device,
        pi0_wrapper=pi0_wrapper,
        action_exec_len=cfg.dsrl.action_exec_len,
        gamma=gamma,
        num_episodes=cfg.eval.eval_num_episodes,
        record_video=cfg.eval.eval_record_video,
        logger=eval_logger,
    )

    eval_log.info(f"[EVAL] Starting async evaluator against {learner_address}...")

    eval_failures = 0
    max_consecutive_failures = 3
    eval_step = 0
    last_eval_fp = initial_fp or None
    eval_meter = RateMeter(name="eval", alpha=0.1)

    try:
        while True:
            try:
                # Fetch latest weights
                new_state_dict = pc.fetch_latest()
                import copy

                new_state_dict = copy.deepcopy(new_state_dict)
                fp = new_state_dict.pop("__fingerprint__", None)
                actor.load_state_dict(new_state_dict)
                actor.eval()
                for param in actor.parameters():
                    param.requires_grad_(False)

                if fp is not None:
                    if last_eval_fp is None:
                        eval_log.info(f"[EVAL] Policy fingerprint set: {fp}")
                    elif fp != last_eval_fp:
                        eval_log.success(f"[EVAL] Policy updated: {last_eval_fp} -> {fp}")
                    else:
                        eval_log.info(f"[EVAL] Policy unchanged: {fp}")
                    last_eval_fp = fp

                # Run evaluation
                with torch.no_grad():
                    t0 = time.time()
                    eval_metrics = worker.run(actor)
                    dt_cycle = time.time() - t0
                    eval_meter.record(dt_cycle)

                if eval_logger and eval_metrics:
                    to_log = {
                        **eval_metrics,
                        "avg_seconds_per_cycle": float(eval_meter.avg_seconds_per_update),
                        "cycles_per_sec": float(eval_meter.updates_per_second),
                    }
                    eval_logger.log_dict(to_log, step=eval_step, prefix="eval")
                    eval_step += 1
                    eval_log.success(f"[EVAL] Completed evaluation: {eval_metrics}")

                eval_failures = 0
                eval_every = cfg.distributed.eval.eval_every
                time.sleep(eval_every)

            except RuntimeError as e:
                eval_failures += 1
                eval_log.error(f"[EVAL] Runtime error (attempt {eval_failures}/{max_consecutive_failures}): {e}")

                if eval_failures >= max_consecutive_failures:
                    eval_log.error(f"[EVAL] Too many consecutive failures, exiting...")
                    break

                time.sleep(2)

            except Exception as e:
                eval_failures += 1
                eval_log.exception(f"[EVAL] Unexpected error (attempt {eval_failures}/{max_consecutive_failures}): {e}")

                if eval_failures >= max_consecutive_failures:
                    eval_log.error(f"[EVAL] Too many consecutive failures, exiting...")
                    break

                time.sleep(2)

    except KeyboardInterrupt:
        eval_log.info("[EVAL] Shutting down...")


def run_serial_dsrl(cfg: DictConfig):
    """Serial training mode - uses shared setup for consistency."""
    # Use shared setup
    components = setup_dsrl_training(cfg, mode="serial", wandb_prefix="offline")

    # Extract what we need
    env = components.env
    eval_env = components.eval_env
    actor = components.actor
    buffer = components.buffer
    algorithm = components.algorithm
    wandb_logger = components.wandb_logger
    save_dir = components.save_dir
    save_interval = components.save_interval
    pi0_wrapper = components.pi0_wrapper
    dummy_dsrl_env = components.dummy_dsrl_env
    device = components.device
    reward_model = components.reward_model
    reward_relabeling_keys = OmegaConf.select(cfg.env, "image_keys", default=["image"])

    # Log additional details
    rprint(f"Actor: {actor}")
    rprint(f"Critic: {components.critic}")
    rprint(f"V-Net: {components.v_net}")

    # Create rollout worker - choose class based on whether we have a reward model
    reward_model_cfg = OmegaConf.select(cfg, "reward_model", default=None)
    # if reward_model_cfg is not None:
    #     rollout_worker_class = DSRLwithRobometerRolloutWorker
    # else:
    rollout_worker_class = DSRLRolloutWorker
    train_after_episode = OmegaConf.select(cfg.training, "train_after_episode", default=False)
    rollout_count_by = OmegaConf.select(cfg.training, "rollout_count_by", default="step")
    #if train_after_episode:
    #    rollout_count_by = "episode"

    rollout_worker = rollout_worker_class(
        env=env,
        buffer=buffer,
        num_rollouts=1,
        actor=actor,
        device=device,
        count_by=rollout_count_by,
        num_envs=cfg.training.num_envs,
        pi0_wrapper=pi0_wrapper,
        gamma=algorithm.gamma,
        action_exec_len=cfg.dsrl.action_exec_len,
        dummy_dsrl_env=dummy_dsrl_env,
        reward_relabeling_keys=reward_relabeling_keys if reward_model_cfg is not None else None,
    )
    logger.info(f"Rollout worker: {rollout_worker.num_envs} environments")

    # Create and run serial runner
    eval_freq = cfg.eval.eval_freq if cfg.eval.eval_freq is not None else cfg.training.num_rollouts // 100

    # For remote robots (or when eval_env is the same as env), only evaluate at episode boundaries
    # to avoid interrupting ongoing trajectories
    eval_at_episode_boundary = (eval_env is env) or OmegaConf.select(
        cfg.eval, "eval_at_episode_boundary", default=False
    )

    # Option to evaluate on the first step (before any training)
    eval_on_first_step = OmegaConf.select(cfg.eval, "eval_on_first_step", default=False)

    # Buffer saving options
    save_buffer_on_exit = OmegaConf.select(cfg.buffer, "save_buffer_on_exit", default=False)
    save_buffer_every = OmegaConf.select(cfg.buffer, "save_buffer_every", default=0)
    save_buffer_images = OmegaConf.select(cfg.buffer, "save_buffer_images", default=False)
    save_buffer_image_keys = OmegaConf.select(cfg.buffer, "save_buffer_image_keys", default=["image"])
    # Use output_dir (parent of save_dir) as prefix for replay buffer directory
    output_dir = os.path.dirname(save_dir)  # save_dir is {output_dir}/checkpoints
    save_buffer_subdir = OmegaConf.select(cfg.buffer, "save_buffer_dir", default="replay_buffers")
    save_buffer_dir = os.path.join(output_dir, save_buffer_subdir)

    runner = SerialRunner(
        env=env,
        eval_env=eval_env,
        algorithm=algorithm,
        buffer=buffer,
        actor=actor,
        rollout_worker=rollout_worker,
        num_rollouts=cfg.training.num_rollouts,
        eval_freq=eval_freq,
        eval_kwargs={
            "num_episodes": cfg.eval.eval_num_episodes,
            "record_video": cfg.eval.eval_record_video,
            "action_exec_len": cfg.dsrl.action_exec_len,
            "pi0_wrapper": pi0_wrapper,
            "gamma": algorithm.gamma,
        },
        logger=wandb_logger,
        evaluation_worker_class=DSRLEvaluationWorker,
        eval_at_episode_boundary=eval_at_episode_boundary,
        eval_on_first_step=eval_on_first_step,
        save_dir=save_dir,
        save_interval=save_interval,
        # Buffer saving options
        save_buffer_on_exit=save_buffer_on_exit,
        save_buffer_every=save_buffer_every,
        save_buffer_images=save_buffer_images,
        save_buffer_image_keys=save_buffer_image_keys,
        save_buffer_dir=save_buffer_dir,
        train_after_episode=train_after_episode,
    )
    
    # Register signal handler to save buffer on SIGINT/SIGTERM (Ctrl+C or kill)
    if save_buffer_on_exit:
        import signal
        
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received {sig_name}, saving buffer before exit...")
            runner.save_buffer_on_signal()
            logger.info("Buffer saved. Exiting...")
            # Re-raise the signal to allow normal cleanup
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Registered signal handlers for buffer saving on SIGINT/SIGTERM")

    rprint(f"Starting training for {cfg.training.num_rollouts} rollouts...")
    rprint("=" * 50)

    try:
        runner.run()
        rprint("\n" + "=" * 50)
        rprint("Training completed successfully!")
        rprint(f"Final buffer size: {len(buffer)}")

        # Save final checkpoint
        save_checkpoint(algorithm, save_dir, "final")

    except KeyboardInterrupt:
        rprint("\n" + "=" * 50)
        rprint("Training interrupted by user")
        rprint(f"Buffer size at interruption: {len(buffer)}")

        # Save checkpoint on interruption
        save_checkpoint(algorithm, save_dir, "interrupted")

    except Exception as e:
        rprint("\n" + "=" * 50)
        rprint(f"Training failed with error: {e}")
        # ensure wandb run is properly closed on failure
        try:
            wandb_logger.finish()
        except Exception:
            pass
        raise

    # clean up
    env.close()
    eval_env.close()
    # finalize logger/run
    try:
        wandb_logger.finish()
    except Exception:
        pass


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="dsrl_config")
def main(cfg: DictConfig):
    """
    Main entry point that dispatches to the appropriate mode based on config.

    Modes:
      - serial (default): Standard serial training loop
      - learner: Async mode - trains policy, serves gRPC for experience ingestion
      - rollout: Async mode - collects experience and streams to learner
      - eval: Async mode - periodically evaluates latest policy from learner
    """
    # Preserve online_algorithm before conversion (Hydra group config)
    # These are lost when converting to dataclass and back
    # Use OmegaConf.select() which handles missing keys gracefully
    original_online_algorithm = OmegaConf.select(cfg, "online_algorithm")
    
    logger.info(f"Loaded online_algorithm config: {original_online_algorithm is not None}")
    if original_online_algorithm is not None:
        logger.info(f"online_algorithm keys: {list(original_online_algorithm.keys())[:5]}")
    
    # Convert Hydra config to dataclass to validate and resolve interpolations
    # This matches the pattern in reward_fm/train.py
    # The conversion automatically resolves all interpolations via OmegaConf.to_container(cfg, resolve=True)
    cfg_dc = convert_hydra_to_dataclass(cfg, DSRLConfig)

    # Display configuration
    display_config(cfg_dc)

    from dataclasses import asdict

    cfg = OmegaConf.create(asdict(cfg_dc))
    # Restore online_algorithm (Hydra group config is lost in dataclass conversion)
    if original_online_algorithm is not None:
        cfg.online_algorithm = original_online_algorithm
    OmegaConf.set_struct(cfg, False)
    mode = cfg.mode

    logger.info(f"Starting DSRL training in '{mode}' mode")

    if mode == "serial":
        run_serial_dsrl(cfg)
    elif mode == "learner":
        run_learner_dsrl(cfg)
    elif mode == "rollout":
        run_rollout_dsrl(cfg)
    elif mode == "eval":
        run_eval_dsrl(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: serial, learner, rollout, eval")


if __name__ == "__main__":
    main()
