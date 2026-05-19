#!/usr/bin/env python3
"""
Evaluate a trained DSRL policy checkpoint.

Loads a trained DSRL policy and evaluates it in the specified environment.
Supports both SIMPLER environments and remote robot environments.

Usage:
    # Evaluate a trained policy (from repo root)
    uv run python scripts/eval_trained_dsrl.py policy_checkpoint=./checkpoints/policy.pt

    # Override connection and logging
    uv run python scripts/eval_trained_dsrl.py \
        policy_checkpoint="<PATH_TO_CHECKPOINT.pt>" \
        logging.wandb_name="<your-eval-run-name>" \
        server.host="<remote-env-host>" \
        server.port=<remote-env-port>

    # Use Hydra config name explicitly
    uv run python scripts/eval_trained_dsrl.py --config-name=eval_trained_dsrl \
        policy_checkpoint=./checkpoints/policy.pt

See README.md (Remote Reward Relabeling) for a longer example next to train_dsrl.
"""

import os

# Prevent JAX from preallocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Initialize JAX early before any other GPU libraries
import jax
jax.devices()
del jax

from datetime import datetime

import numpy as np
import torch
from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from transformers import AutoModel, AutoImageProcessor
from sentence_transformers import SentenceTransformer

from robometer_policy_learning.envs.dsrl_env_wrappers import make_simpler_env, make_remote_robot_env, DummyDSRLEnv, setup_libero_env
from robometer_policy_learning.rollouts.dsrl_evaluation_worker import DSRLEvaluationWorker
from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper
from robometer_policy_learning.utils.training_utils import build_actor_critic_models


def setup_environment(cfg: DictConfig, dinov2_model, dinov2_processor, sentence_model, device):
    """Set up the evaluation environment based on config."""
    env_type = cfg.env.env_type
    image_keys = list(cfg.env.image_keys) if cfg.env.image_keys else ["observation.images.image_0"]
    extra_keys_to_drop = list(cfg.env.extra_keys_to_drop) if cfg.env.extra_keys_to_drop else []
    
    if env_type == "simpler":
        env, remove_obs_keys = make_simpler_env(
            n_envs=1,
            extra_keys_to_drop=extra_keys_to_drop,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            sentence_model=sentence_model,
            device=device,
            use_dense_reward=False,
            host=cfg.server.host,
            port=cfg.server.port,
            num_stages=cfg.env.get("num_stages", 1),
        )
    elif env_type == "remote_robot":
        env, remove_obs_keys = make_remote_robot_env(
            n_envs=1,
            host=cfg.server.host,
            port=cfg.server.port,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            sentence_model=sentence_model,
            device=device,
            obs_format=cfg.env.obs_format,
            image_keys=image_keys,
            extra_keys_to_drop=extra_keys_to_drop,
            num_stages=cfg.env.get("num_stages", 1),
        )
    elif env_type == "libero":
        env, remove_obs_keys = setup_libero_env(
            cfg.env.env_name,
            cfg.env.task_id,
            n_envs=1,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            sentence_model=sentence_model,
            device=device,
            seed=cfg.seed,
            max_episode_steps=cfg.env.max_episode_steps,
            image_keys=image_keys,
            extra_keys_to_drop=extra_keys_to_drop,
            async_reward_relabel_kwargs=None,  # No reward relabeling during evaluation
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return env, remove_obs_keys


def load_policy(checkpoint_path: str, dummy_dsrl_env, device: torch.device, remove_obs_keys: list, cfg: DictConfig):
    """Load a trained DSRL policy from checkpoint."""
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # If checkpoint is a saved model object, return it directly
    if isinstance(checkpoint, torch.nn.Module):
        actor = checkpoint.to(device)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad_(False)
        logger.success(f"✓ Loaded policy module from {checkpoint_path}")
        return actor
    
    # Extract config from checkpoint if available
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        policy_cfg = OmegaConf.create(checkpoint["config"])
        logger.info("Using config from checkpoint")
    else:
        # Use provided config
        policy_cfg = cfg
        logger.warning("No config found in checkpoint, using provided config")
    
    # Build actor model
    dsrl_observation_space = dummy_dsrl_env.observation_space
    dsrl_action_space = dummy_dsrl_env.action_space
    
    actor, _, _ = build_actor_critic_models(
        dsrl_observation_space, dsrl_action_space, policy_cfg, device, remove_obs_keys
    )
    
    # Load actor weights
    if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["actor_state_dict"])
        logger.info("Loaded actor_state_dict from checkpoint")
    elif isinstance(checkpoint, dict) and "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
        logger.info("Loaded actor from checkpoint")
    else:
        # Assume checkpoint is the actor state dict directly
        actor.load_state_dict(checkpoint)
        logger.info("Loaded checkpoint as actor state dict")
    
    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)
    
    logger.success(f"✓ Loaded policy from {checkpoint_path}")
    
    # Log checkpoint info
    if "step" in checkpoint:
        logger.info(f"  Checkpoint step: {checkpoint['step']}")
    if "epoch" in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
    
    return actor


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="eval_trained_dsrl")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    # Print config
    rprint("[bold]DSRL Policy Evaluation Configuration:[/bold]")
    rprint(OmegaConf.to_yaml(cfg))
    rprint("=" * 60)
    
    # Validate arguments
    if cfg.policy_checkpoint is None:
        raise ValueError("policy_checkpoint must be specified")
    
    if not os.path.exists(cfg.policy_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.policy_checkpoint}")
    
    # Set seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Determine device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Optional wandb logging
    wandb_logger = None
    if OmegaConf.select(cfg, "logging.wandb_enable", default=False):
        try:
            string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            exp_name = f"{cfg.logging.wandb_name}_{string_time}"
            tags = list(OmegaConf.select(cfg, "logging.wandb_tags", default=[])) or None
            wandb_logger = WandbLogger(
                exp_name=exp_name,
                offline=OmegaConf.select(cfg, "logging.wandb_offline", default=False),
                project=OmegaConf.select(cfg, "logging.wandb_project", default=None),
                entity=OmegaConf.select(cfg, "logging.wandb_entity", default=None),
                log_dir=f"{cfg.logging.wandb_log_dir_base}/{cfg.logging.wandb_name}",
                prefix="eval",
                group=OmegaConf.select(cfg, "logging.wandb_group", default=None),
                job_type=OmegaConf.select(cfg, "logging.wandb_job_type", default=None),
                tags=tags,
            )
            wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
        except Exception as e:
            logger.warning(f"Failed to initialize wandb logger: {e}")
    
    # Setup optional models (DINO, sentence transformer)
    dinov2_model, dinov2_processor = None, None
    sentence_model = None
    
    if cfg.model.dinov2_model:
        logger.info(f"Loading DINOv2 model: {cfg.model.dinov2_model}")
        dinov2_model = AutoModel.from_pretrained(cfg.model.dinov2_model).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(cfg.model.dinov2_model)
    
    if cfg.model.sentence_model:
        logger.info(f"Loading sentence model: {cfg.model.sentence_model}")
        sentence_model = SentenceTransformer(cfg.model.sentence_model)
    
    # Load Pi0
    logger.info(f"Loading Pi0 from {cfg.pi0.checkpoint}")
    pi0_wrapper = Pi0Wrapper(cfg.pi0.checkpoint, device=str(device))
    
    # Setup environment
    logger.info(f"Setting up {cfg.env.env_type} environment at {cfg.server.host}:{cfg.server.port}")
    env, remove_obs_keys = setup_environment(cfg, dinov2_model, dinov2_processor, sentence_model, device)
    
    # Get observation and action spaces
    if hasattr(env, "single_observation_space"):
        observation_space = env.single_observation_space
    else:
        observation_space = env.observation_space
    
    if hasattr(env, "single_action_space"):
        action_space = env.single_action_space
    else:
        action_space = env.action_space
    
    # Create dummy DSRL env for action/observation space
    dummy_dsrl_env = DummyDSRLEnv(
        observation_space,
        action_space,
        pi0_wrapper,
        noise_dim=cfg.pi0.noise_dim,
        chunk_size=None,
        action_bound=cfg.pi0.noise_action_bound,
        use_vlm_features=cfg.pi0.get("use_vlm_features", True),
    )
    
    # Load policy
    actor = load_policy(cfg.policy_checkpoint, dummy_dsrl_env, device, remove_obs_keys, cfg)
    
    # Compute gamma with action_exec_len
    gamma = cfg.eval.gamma ** cfg.pi0.action_exec_len
    logger.info(f"Effective gamma (gamma^action_exec_len): {gamma:.4f}")
    
    # Get image keys for video recording
    image_keys = list(cfg.env.image_keys) if cfg.env.image_keys else ["observation.images.image_0"]
    
    # Create evaluation worker
    worker = DSRLEvaluationWorker(
        eval_env=env,
        device=device,
        pi0_wrapper=pi0_wrapper,
        action_exec_len=cfg.pi0.action_exec_len,
        gamma=gamma,
        num_episodes=cfg.eval.num_episodes,
        record_video=cfg.eval.record_video,
        logger=wandb_logger,
        image_keys=image_keys,
        use_random_noise=False,  # Always use trained policy
    )
    
    # Run evaluation
    logger.info(f"Starting evaluation with trained policy ({cfg.eval.num_episodes} episodes)")
    rprint("=" * 60)
    
    with torch.no_grad():
        eval_metrics = worker.run(actor)

    if wandb_logger is not None:
        wandb_logger.log_dict(eval_metrics, step=0, prefix="eval")
    
    # Print results
    rprint("=" * 60)
    rprint("[bold green]Evaluation Results:[/bold green]")
    rprint(f"  Policy Checkpoint: {cfg.policy_checkpoint}")
    rprint(f"  Episodes: {cfg.eval.num_episodes}")
    rprint(f"  Average Reward: {eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}")
    rprint(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
    rprint(f"  Average Steps: {eval_metrics['avg_steps']:.1f}")
    rprint(f"  Min/Max Reward: {eval_metrics['min_reward']:.3f} / {eval_metrics['max_reward']:.3f}")
    
    if cfg.eval.record_video and eval_metrics.get("video_saved", False):
        rprint(f"  Video Reward: {eval_metrics['video_reward']:.3f}")
        rprint(f"  Video Success: {eval_metrics['video_success']}")
    
    rprint("=" * 60)
    
    # Cleanup
    env.close()
    if wandb_logger is not None:
        try:
            wandb_logger.finish()
        except Exception:
            pass
    logger.success("Evaluation complete!")
    
    return eval_metrics


if __name__ == "__main__":
    main()
