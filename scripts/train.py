#!/usr/bin/env python3
"""
Main RL training script.
"""

# Configure headless rendering for MuJoCo/GLFW before importing gym/metaworld
import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from rich import print as rprint

from robometer_policy_learning.runners.serial_runner import SerialRunner

from robometer_policy_learning.buffers.mixed_replay_buffer import MixedReplayBuffer
from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.buffers.remote_reward_relabel_buffer import AsyncRewardRelabelBuffer
from robometer_policy_learning.algorithms.bc import BC, BCConfig
from robometer_policy_learning.algorithms.iql import IQL, IQLConfig
from robometer_policy_learning.algorithms.sac import SAC, SACConfig
from robometer_policy_learning.algorithms.dp import DP, DPConfig
from robometer_policy_learning.rollouts.robometer_rollout_worker import RobometerRolloutWorker
from robometer_policy_learning.rollouts.rollout_worker import RolloutWorker
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.utils.training_utils import load_checkpoint, save_checkpoint, setup_training, create_buffer

ALG_TO_CONFIG = {
    "iql": IQLConfig,
    "bc": BCConfig,
    "sac": SACConfig,
    "dp": DPConfig,
}


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    OmegaConf.resolve(cfg)

    # Setup all training components
    components = setup_training(cfg)

    # Extract components for easier access
    cfg = components.cfg
    device = components.device
    env = components.env
    eval_env = components.eval_env
    actor = components.actor
    critic = components.critic
    v_net = components.v_net
    remove_obs_keys = components.remove_obs_keys
    reward_model = components.reward_model
    reward_model_exp_cfg = components.reward_model_exp_cfg
    use_gt_rewards = components.use_gt_rewards
    use_relative_rewards = components.use_relative_rewards
    success_bonus_fn = components.success_bonus_fn
    save_dir = components.save_dir
    logger = components.logger
    wandb_logger = components.wandb_logger
    reward_model_cfg = OmegaConf.select(cfg, "reward_model", default=None)
    image_keys_to_be_used = components.dino_image_keys

    # Action bounds for buffer-side normalization: stored (env-space) actions are mapped to
    # the policy's [-1, 1] space for offline/online training, matching the actor's output.
    # Only set when the action space is finite (else the actor doesn't normalize either, so
    # the buffer shouldn't). The env still receives unnormalized actions (act() unnormalizes).
    _action_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
    if (
        _action_space is not None
        and hasattr(_action_space, "low")
        and np.all(np.isfinite(_action_space.low))
        and np.all(np.isfinite(_action_space.high))
    ):
        action_min = np.asarray(_action_space.low, dtype=np.float32)
        action_max = np.asarray(_action_space.high, dtype=np.float32)
    else:
        action_min = action_max = None
    logger.info(f"Action normalization bounds: min={action_min}, max={action_max}")

    # Override num_offline_steps in debug mode
    if cfg.debug:
        rprint("Debug mode enabled")
        cfg.training.num_offline_steps = 10
        cfg.logging.wandb_offline = True

    offline_algo = None
    offline_buffer = None

    logger.info(f"Success bonus function: {success_bonus_fn}")
    offline_algorithm_cfg = OmegaConf.select(cfg, "offline_algorithm", default=None)
    if offline_algorithm_cfg is not None:
        # Initialize offline buffer
        if cfg.env.use_full_state:
            assert use_gt_rewards, "use_gt_rewards must be True when use_full_state is True"
            print(
                "⚠️ Using only the ground truth state observations, no DINO embeddings or language embeddings will be computed. No reward relabeling will be performed."
            )
        if cfg.training.chunk_size is None:
            sampler = RandomSampler()
        else:
            gamma = offline_algorithm_cfg.gamma if hasattr(offline_algorithm_cfg, "gamma") else 0.99
            sampler = ChunkedSequentialSampler(
                chunk_size=cfg.training.chunk_size, obs_as_sequence=False, gamma=gamma
            )
        logger.info(f"Offline Sampler: {sampler.__class__.__name__}")
        offline_buffer = create_buffer(
            sampler=sampler,
            use_eval_server=components.use_eval_server,
            eval_server_url=components.eval_server_url,
            eval_server_timeout=components.eval_server_timeout,
            reward_model=reward_model,
            reward_model_exp_cfg=reward_model_exp_cfg,
            use_gt_rewards=use_gt_rewards,
            use_relative_rewards=use_relative_rewards,
            capacity=0,  # Not used for H5 buffers
            remove_obs_keys=remove_obs_keys,
            post_transforms=[success_bonus_fn] if success_bonus_fn is not None else [],
            h5_paths=[cfg.env.h5_dataset_path],
            use_full_state=cfg.env.use_full_state,
            sentence_model=components.sentence_model,
            dinov2_model=components.dinov2_model,
            dinov2_processor=components.dinov2_processor,
            image_keys_to_be_used=image_keys_to_be_used,
            min_action=action_min,
            max_action=action_max,
            use_success_detection=cfg.reward_model.use_success_detection if reward_model is not None else False,
            success_detection_duration=cfg.reward_model.success_detection_duration
            if reward_model is not None
            else 2,
            success_detection_threshold=cfg.reward_model.success_detection_threshold
            if reward_model is not None
            else 0.65,
            add_estimated_reward=cfg.reward_model.add_estimated_reward
            if reward_model is not None
            else False,
        )

        offline_algo_dict = OmegaConf.to_container(offline_algorithm_cfg)
        if cfg.alg.offline_alg_name.lower() not in ALG_TO_CONFIG:
            raise ValueError(f"Unknown offline algorithm: {cfg.alg.offline_alg_name}")
        offline_algo_config = ALG_TO_CONFIG[cfg.alg.offline_alg_name.lower()](**offline_algo_dict)

        # Set runtime fields
        offline_algo_config.env = env
        offline_algo_config.actor = actor
        offline_algo_config.critic = critic
        offline_algo_config.buffer = offline_buffer
        offline_algo_config.logger = wandb_logger

        # Add v_net for IQL
        if isinstance(offline_algo_config, IQLConfig):
            offline_algo_config.v_net = v_net

        offline_algo = offline_algo_config.create()

        # Load checkpoint if specified
        start_step = 0
        if cfg.training.load_dir is not None:
            logger.info(f"Loading checkpoint from {cfg.training.load_dir}")
            start_step = load_checkpoint(offline_algo, cfg.training.load_dir)
            logger.info(f"Resuming from step {start_step}")

            # Create save directory
            # Save the offline algorithm
            logger.info(f"Saving checkpoint to {save_dir}/latest")
            save_checkpoint(offline_algo, save_dir, "latest")

        # If we loaded a checkpoint, we skip offline training step
        if cfg.training.load_dir is None or cfg.training.continue_training:
            offline_evaluation_worker = EvaluationWorker(
                eval_env=eval_env,
                device=device,
                num_episodes=cfg.eval.eval_num_episodes,
                record_video=True,
                logger=wandb_logger,
            )
            offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
            wandb_logger.log(offline_eval_metrics, step=0, prefix="offline/eval")            
            # Training loop
            logger.info(f"Training offline algorithm for {cfg.training.num_offline_steps} steps")
            with tqdm(total=cfg.training.num_offline_steps, desc="Offline Training", unit="step") as pbar:
                for i in range(start_step, cfg.training.num_offline_steps):
                    metrics = offline_algo.train_step()
                    formatted_metrics = {k: f"{v:3.3f}" if isinstance(v, float) else v for k, v in metrics.items()}
                    pbar.update(1)
                    pbar.set_postfix(formatted_metrics)

                    # Save checkpoint periodically
                    if (i + 1) % cfg.training.save_interval == 0:
                        save_checkpoint(offline_algo, save_dir, i + 1)
                    if (i + 1) % cfg.eval.eval_freq == 0 and cfg.eval.eval_freq is not None:
                        offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
                        wandb_logger.log(offline_eval_metrics, step=i, prefix="offline/eval")

    if cfg.training.num_rollouts > 0:
        online_algorithm_cfg = OmegaConf.select(cfg, "online_algorithm", default=None)
        if online_algorithm_cfg is None:
            raise ValueError("online_algorithm config is required when num_rollouts > 0")
        online_algo_dict = OmegaConf.to_container(online_algorithm_cfg)

        if cfg.alg.online_alg_name.lower() == "sac":
            online_algo_config = SACConfig(**online_algo_dict)
        else:
            raise ValueError(f"Unknown online algorithm: {cfg.alg.online_alg_name}")

        # Check if distributed reward relabeling is enabled
        if cfg.env.get("use_async_reward_relabel", False): 
            use_async_reward_relabel = cfg.reward_model is not None
            reward_relabel_address = cfg.reward_model.async_reward_relabel_server_address
        else:
            use_async_reward_relabel = False
            reward_relabel_address = None
        
        # Create sampler based on whether we're using async reward relabeling
        # If using buffer-level async reward relabeling, use RelabeledOnlySampler to ensure
        # we only train on transitions that have been properly relabeled
        if cfg.training.chunk_size is None:
            if use_async_reward_relabel:
                from robometer_policy_learning.buffers.samplers import RelabeledOnlySampler
                min_relabeled_ratio = cfg.buffer.get("min_relabeled_ratio", 0.1)
                sampler = RelabeledOnlySampler(min_relabeled_ratio=min_relabeled_ratio)
                logger.info(f"Using RelabeledOnlySampler (min_relabeled_ratio={min_relabeled_ratio})")
            else:
                sampler = RandomSampler()
                logger.info("Using RandomSampler")
        else:
            sampler = ChunkedSequentialSampler(
                chunk_size=cfg.training.chunk_size, obs_as_sequence=False, gamma=online_algo_config.gamma
            )
            logger.info(f"Using ChunkedSequentialSampler (chunk_size={cfg.training.chunk_size})")
        
        logger.info(f"Online Sampler: {sampler.__class__.__name__}")

        # Create online replay buffer
        online_buffer = create_buffer(
            sampler=sampler,
            use_gt_rewards=use_gt_rewards if not use_async_reward_relabel else True,
            use_relative_rewards=use_relative_rewards,
            reward_model_exp_cfg=reward_model_exp_cfg,
            capacity=cfg.buffer.capacity,
            remove_obs_keys=remove_obs_keys,
            post_transforms=[success_bonus_fn] if success_bonus_fn is not None else [],
            use_eval_server=components.use_eval_server if not use_async_reward_relabel else False,
            eval_server_url=components.eval_server_url if not use_async_reward_relabel else None,
            eval_server_timeout=components.eval_server_timeout if not use_async_reward_relabel else 120.0,
            use_async_reward_relabel=use_async_reward_relabel and reward_model is not None,
            reward_model=reward_model,
            image_keys_to_be_used=image_keys_to_be_used,
            min_action=action_min,
            max_action=action_max,
            reward_relabel_address=reward_relabel_address
            if use_async_reward_relabel and reward_model is not None
            else None,
            use_success_detection=cfg.reward_model.use_success_detection
            if hasattr(cfg, "reward_model") and cfg.reward_model is not None
            else False,
            success_detection_duration=cfg.reward_model.success_detection_duration
            if hasattr(cfg, "reward_model") and cfg.reward_model is not None
            else 2,
            success_detection_threshold=cfg.reward_model.success_detection_threshold
            if hasattr(cfg, "reward_model") and cfg.reward_model is not None
            else 0.65,
            add_estimated_reward=cfg.reward_model.add_estimated_reward
            if hasattr(cfg, "reward_model") and cfg.reward_model is not None
            else False,
        )

        if use_async_reward_relabel and reward_model is not None:
            logger.info(f"Using async reward relabeling with server at {reward_relabel_address}")

        if offline_buffer is not None and cfg.buffer.sample_ratio > 0:
            buffer = MixedReplayBuffer(
                buffer_1=offline_buffer,
                buffer_2=online_buffer,
                sample_ratio=cfg.buffer.sample_ratio,
                buffer_to_add_to=2,
                remove_obs_keys=remove_obs_keys,
                sampler=sampler,
            )
        else:
            buffer = online_buffer

        # Set runtime fields (buffer is now created)
        online_algo_config.env = env
        online_algo_config.actor = actor
        online_algo_config.critic = critic
        online_algo_config.buffer = buffer
        online_algo_config.action_space = env.action_space
        online_algo_config.logger = wandb_logger

        # Create online algorithm
        if isinstance(online_algo_config, SACConfig):
            algorithm = SAC(online_algo_config)

        logger.info(f"Algorithm: {algorithm.__class__.__name__}")

        # Copy components from offline algorithm if it exists
        if offline_algo is not None:
            algorithm.copy_components(offline_algo)
        else:
            if cfg.training.load_dir is not None:
                logger.info(f"Loading checkpoint from {cfg.training.load_dir}")
                load_checkpoint(algorithm, cfg.training.load_dir)
                logger.info(f"Resuming from checkpoint")

        logger.info(f"Buffer capacity: {cfg.buffer.capacity}")
        # Create rollout worker
        if reward_model_cfg is not None:
            rollout_worker_class = RobometerRolloutWorker
        else:
            rollout_worker_class = RolloutWorker

        rollout_worker = rollout_worker_class(
            env=env,
            buffer=buffer,
            num_rollouts=1,
            actor=actor,
            device=device,
            count_by="step",
            num_envs=cfg.training.num_envs,
            reward_relabeling_keys=image_keys_to_be_used if reward_model_cfg is not None else None,
        )
        logger.info(f"Rollout worker: {rollout_worker.num_envs} environments")

        # Create and run serial runner
        runner = SerialRunner(
            env=env,
            eval_env=eval_env,
            algorithm=algorithm,
            buffer=buffer,
            actor=actor,
            rollout_worker=rollout_worker,
            num_rollouts=cfg.training.num_rollouts,
            eval_freq=cfg.eval.eval_freq if cfg.eval.eval_freq is not None else cfg.training.num_rollouts // 100,
            eval_kwargs={
                "num_episodes": cfg.eval.eval_num_episodes,
                "record_video": cfg.eval.eval_record_video,
            },
            logger=wandb_logger,
            eval_on_first_step=cfg.eval.eval_on_first_step,
        )

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
        finally:
            # Clean up remote reward relabeling client if used
            if use_async_reward_relabel and reward_model_cfg is not None:
                if isinstance(online_buffer, AsyncRewardRelabelBuffer):
                    logger.info("Stopping remote reward relabeling client...")
                    online_buffer.stop()

    else:
        offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
        wandb_logger.log(offline_eval_metrics, step=cfg.training.num_offline_steps, prefix="offline/eval")

    # clean up
    env.close()
    eval_env.close()
    # finalize logger/run
    try:
        wandb_logger.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
