#!/usr/bin/env python3
"""
Main RL training script.
"""

# Configure headless rendering for MuJoCo/GLFW before importing gym/metaworld
import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from rich import print as rprint

from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.algorithms.bc import BC, BCConfig
from robometer_policy_learning.algorithms.iql import IQL, IQLConfig
from robometer_policy_learning.algorithms.sac import SAC, SACConfig
from robometer_policy_learning.algorithms.dp import DP, DPConfig
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.utils.training_utils import load_checkpoint, save_checkpoint, setup_training, create_buffer
from robometer_policy_learning.utils.transitions_transforms import ImageAugmentationTransform

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
    remove_obs_keys = components.remove_obs_keys
    reward_model = components.reward_model
    reward_model_exp_cfg = components.reward_model_exp_cfg
    use_gt_rewards = components.use_gt_rewards
    use_relative_rewards = components.use_relative_rewards
    success_bonus_fn = components.success_bonus_fn
    save_dir = components.save_dir
    logger = components.logger
    wandb_logger = components.wandb_logger
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
    observation_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
    post_transforms = []
    if success_bonus_fn is not None:
        post_transforms.append(success_bonus_fn)
    use_image_transforms = OmegaConf.select(cfg, "training.use_image_transforms", default=False)
    if use_image_transforms:
        post_transforms.append(
            ImageAugmentationTransform(
                observation_space=observation_space,
                seed=cfg.training.seed,
            )
        )
    logger.info(f"Post transforms: {post_transforms}")

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
        post_transforms=post_transforms,
        h5_paths=[cfg.env.h5_dataset_path],
        use_full_state=cfg.env.use_full_state,
        sentence_model=components.sentence_model,
        dinov2_model=components.dinov2_model,
        dinov2_processor=components.dinov2_processor,
        image_keys_to_be_used=image_keys_to_be_used,
        min_action=action_min,
        max_action=action_max,
        normalize_lowdim_obs=OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False),
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
    offline_algo_config.buffer = offline_buffer
    offline_algo_config.logger = wandb_logger

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
            lowdim_obs_stats=getattr(offline_buffer, "lowdim_obs_stats", {}),
        )
        if cfg.eval.eval_on_first_step:
            offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
            wandb_logger.log(offline_eval_metrics, step=offline_algo.step_counter, prefix="eval")
        # Training loop
        logger.info(f"Training offline algorithm for {cfg.training.num_offline_steps} steps")
        with tqdm(total=cfg.training.num_offline_steps, desc="Offline Training", unit="step") as pbar:
            for i in range(start_step, cfg.training.num_offline_steps):
                metrics = offline_algo.train_step(logging_prefix="train")
                formatted_metrics = {k: f"{v:3.3f}" if isinstance(v, float) else v for k, v in metrics.items()}
                pbar.update(1)
                pbar.set_postfix(formatted_metrics)

                # Save checkpoint periodically
                if (i + 1) % cfg.training.save_interval == 0:
                    save_checkpoint(offline_algo, save_dir, i + 1)
                if (i + 1) % cfg.eval.eval_freq == 0 and cfg.eval.eval_freq is not None:
                    offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
                    wandb_logger.log(offline_eval_metrics, step=offline_algo.step_counter, prefix="eval")

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
