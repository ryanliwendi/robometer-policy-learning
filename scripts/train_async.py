#!/usr/bin/env python3
"""
Unified async runner with learner/rollout/eval modes.

This consolidates common flags and configuration for actor, env, buffers,
and networking. Uses Hydra config like train.py.

Modes:
  - learner: trains SAC online, serves gRPC Ingestion + Policy
  - rollout: collects experience and streams to learner, pulls weights
  - eval:    periodically evaluates latest policy from learner
"""

from __future__ import annotations

import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import socket
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except Exception:
    wandb = None

# Perf knobs for CNNs and matmul
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from robometer_policy_learning.distributed.servers.learner_server import LearnerServer
from robometer_policy_learning.distributed.clients.policy_client import PolicyClient
from robometer_policy_learning.distributed.clients.streaming_buffer_adapter import StreamingBufferAdapter

from robometer_policy_learning.rollouts.rollout_worker import RolloutWorker
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.rollouts.robometer_rollout_worker import RobometerRolloutWorker

from robometer_policy_learning.buffers.mixed_replay_buffer import MixedReplayBuffer
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.buffers.robometer_replay_buffer import RobometerH5ReplayBuffer, RobometerReplayBuffer
from robometer_policy_learning.buffers.remote_reward_relabel_buffer import AsyncRewardRelabelBuffer
from robometer_policy_learning.distributed.clients.reward_relabel_client import RewardRelabelClient

from robometer_policy_learning.algorithms.sac import SAC, SACConfig
from robometer_policy_learning.algorithms.bc import BC, BCConfig
from robometer_policy_learning.algorithms.iql import IQL, IQLConfig

from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer_policy_learning.utils.transitions_transforms import SuccessBonusTransform
from robometer_policy_learning.utils.training_utils import (
    build_actor_critic_models,
    load_checkpoint,
    save_checkpoint,
    load_reward_model_config_from_hf,
)
from robometer_policy_learning.utils.rate_utils import RateMeter, RateLimiter, EmaValue
from robometer_policy_learning.utils.env_utils import make_env
from robometer_policy_learning.envs.vector_wrappers import SingleEnvVectorWrapper
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor
import sys
# logger.remove()
# logger.add(sys.stderr, colorize=False)


GT_REW_SUCCESS_BONUS = 200
ROBOMETER_SUCCESS_BONUS = 64.0
RELATIVE_REW_SUCCESS_BONUS = 1.0


def build_light_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    # Keep trainable modules + essential frozen modules used at inference
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


def run_learner(cfg: DictConfig):
    """Main learner function using Hydra config (same as train.py)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get Hydra output directory (same as train.py)
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    save_dir = f"{output_dir}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = cfg.wandb_name
    wandb_logger = WandbLogger(
        exp_name=exp_name,
        offline=cfg.wandb_offline,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        log_dir=f"{cfg.wandb_log_dir_base}/{string_time}",
        prefix="async",
    )

    # Check if reward_model is in config and not None (same as train.py)
    reward_model_cfg = OmegaConf.select(cfg, "reward_model", default=None)

    # Check if async reward relabeling is enabled
    use_async_reward_relabel = OmegaConf.select(cfg, "distributed_reward_relabel.enabled", default=False)

    # Set device
    logger.info(f"Using device: {device}")

    # Setup DINO and sentence models (same as train.py)
    remove_obs_keys = ["image"]
    dinov2_model = AutoModel.from_pretrained(cfg.dinov2_model).to(device).eval()
    dinov2_processor = AutoImageProcessor.from_pretrained(cfg.dinov2_model)
    sentence_model = SentenceTransformer(cfg.sentence_model)

    use_gt_rewards = cfg.use_gt_rewards
    use_relative_rewards = reward_model_cfg.use_relative_rewards if reward_model_cfg is not None else False

    # Load reward model only if not using async relabeling
    if use_async_reward_relabel:
        logger.info("Using async reward relabeling - skipping local reward model loading")
        reward_model = None
        exp_config = None
        use_gt_rewards = True
    elif reward_model_cfg is not None:
        rollout_cfg = reward_model_cfg
        exp_config, tokenizer, processor, reward_model = load_reward_model_config_from_hf(
            model_path=rollout_cfg.model_path,
            device=device,
        )
        if exp_config is None:
            raise ImportError("Failed to load reward model. Ensure rfm modules are available.")
    else:
        logger.info("⚠️ No reward model provided, using ground truth rewards")
        reward_model = None
        exp_config = None
        use_gt_rewards = True

    # Create environment (same as train.py)
    env, eval_env = make_env(
        env_name=cfg.env_name,
        num_envs=cfg.num_envs,
        chunk_size=cfg.chunk_size,
        use_full_state=cfg.use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model if not cfg.use_full_state else None,
        render_mode="rgb_array",
        terminate_on_success=True,
    )

    # Get action and observation spaces (same as train.py)
    if hasattr(env, "single_action_space"):
        action_space = env.single_action_space
    else:
        action_space = env.action_space

    if hasattr(env, "single_observation_space"):
        observation_space = env.single_observation_space
    else:
        observation_space = env.observation_space

    # Build models (same as train.py)
    actor, critic, v_net = build_actor_critic_models(observation_space, action_space, cfg, device, remove_obs_keys)
    logger.info(f"Actor: {actor.__class__.__name__}")
    logger.info(f"Critic: {critic.__class__.__name__}")
    logger.info(f"V-Net: {v_net.__class__.__name__}")

    # Define success bonus based on the reward model (same as train.py)
    if reward_model_cfg is None:
        success_bonus_fn = SuccessBonusTransform(GT_REW_SUCCESS_BONUS)
    else:
        success_bonus_fn = (
            SuccessBonusTransform(RELATIVE_REW_SUCCESS_BONUS)
            if use_relative_rewards
            else SuccessBonusTransform(ROBOMETER_SUCCESS_BONUS)
        )

    # Offline buffer (same as train.py)
    offline_buffer = None
    offline_algo = None
    offline_algo_cfg = OmegaConf.select(cfg, "offline_algorithm", default=None)

    if offline_algo_cfg is not None:
        offline_algo_dict = OmegaConf.to_container(offline_algo_cfg, resolve=True)

        if cfg.offline_alg_name.lower() == "iql":
            offline_algo_config = IQLConfig(**offline_algo_dict)
        elif cfg.offline_alg_name.lower() == "bc":
            offline_algo_config = BCConfig(**offline_algo_dict)
        elif cfg.offline_alg_name.lower() == "sac":
            offline_algo_config = SACConfig(**offline_algo_dict)
        else:
            raise ValueError(f"Unknown offline algorithm: {cfg.offline_alg_name}")

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
        if cfg.load_dir is not None:
            logger.info(f"Loading checkpoint from {cfg.load_dir}")
            start_step = load_checkpoint(offline_algo, cfg.load_dir)
            logger.info(f"Resuming from step {start_step}")
        if cfg.chunk_size is None:
            sampler = RandomSampler()
        else:
            sampler = ChunkedSequentialSampler(
                chunk_size=cfg.chunk_size, obs_as_sequence=False, gamma=offline_algo_config.gamma
            )
        if not cfg.use_full_state:
            offline_buffer = RobometerH5ReplayBuffer(
                reward_model=reward_model,
                reward_model_config=exp_config,
                h5_paths=[cfg.h5_dataset_path],
                sampler=sampler,
                use_gt_rewards=use_gt_rewards,
                use_relative_rewards=use_relative_rewards,
                remove_obs_keys=remove_obs_keys,
                post_transforms=[success_bonus_fn],
                sentence_model=sentence_model,
                dinov2_model=dinov2_model,
                dinov2_processor=dinov2_processor,
            )
        else:
            assert use_gt_rewards, "use_gt_rewards must be True when use_full_state is True"
            offline_buffer = H5ReplayBuffer(
                h5_paths=[cfg.h5_dataset_path],
                sampler=sampler,
                remove_obs_keys=remove_obs_keys,
                post_transforms=[success_bonus_fn],
            )

        # Offline training loop
        if cfg.load_dir is None or cfg.continue_training:
            offline_evaluation_worker = EvaluationWorker(
                eval_env=eval_env,
                device=device,
                num_episodes=cfg.eval_num_episodes,
                record_video=False,
                logger=wandb_logger,
            )
            logger.info(f"Training offline algorithm for {cfg.num_offline_steps} steps")
            from tqdm import tqdm

            with tqdm(total=cfg.num_offline_steps, desc="Offline Training", unit="step") as pbar:
                for i in range(start_step, cfg.num_offline_steps):
                    metrics = offline_algo.train_step()
                    formatted_metrics = {k: f"{v:3.3f}" if isinstance(v, float) else v for k, v in metrics.items()}
                    pbar.update(1)
                    pbar.set_postfix(formatted_metrics)

                    # Save checkpoint periodically
                    if (i + 1) % cfg.save_interval == 0:
                        save_checkpoint(offline_algo, save_dir, i + 1)
                    if (i + 1) % cfg.eval_freq == 0 and cfg.eval_freq is not None:
                        offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
                        wandb_logger.log(offline_eval_metrics, step=i, prefix="offline/eval")

    if cfg.num_rollouts > 0:
        # Get reward relabel address (already checked use_async_reward_relabel above)
        reward_relabel_address = OmegaConf.select(
            cfg, "distributed_reward_relabel.server_address", default="localhost:50052"
        )
        online_algo_dict = OmegaConf.to_container(cfg.online_algorithm, resolve=True)
        if cfg.online_alg_name.lower() == "sac":
            online_algo_config = SACConfig(**online_algo_dict)
        else:
            raise ValueError(f"Unknown online algorithm: {cfg.online_alg_name}")

        # Set runtime fields
        online_algo_config.env = env
        online_algo_config.actor = actor
        online_algo_config.critic = critic
        online_algo_config.buffer = buffer
        online_algo_config.action_space = env.action_space
        online_algo_config.logger = wandb_logger

        # Create online algorithm
        algorithm = SAC(online_algo_config)

        # Copy components from offline algorithm if it exists
        if offline_algo is not None:
            algorithm.copy_components(offline_algo)
        else:
            if cfg.load_dir is not None:
                logger.info(f"Loading checkpoint from {cfg.load_dir}")
                load_checkpoint(algorithm, cfg.load_dir)
                logger.info(f"Resuming from checkpoint")

        logger.info(f"Algorithm: {algorithm.__class__.__name__}")

        if cfg.chunk_size is None:
            sampler = RandomSampler()
        else:
            sampler = ChunkedSequentialSampler(
                chunk_size=cfg.chunk_size, obs_as_sequence=False, gamma=online_algo_config.gamma
            )

        # Create Robometer replay buffer for online training (same as train.py)
        if reward_model_cfg is not None:
            if use_async_reward_relabel:
                # Use remote reward relabeling (pre mode)
                logger.info(f"Using async reward relabeling with server at {reward_relabel_address}")

                # Create underlying buffer (without reward relabeling)
                underlying_buffer = ReplayBuffer(
                    capacity=cfg.buffer_capacity,
                    remove_obs_keys=remove_obs_keys,
                    post_transforms=[success_bonus_fn],
                    sampler=sampler,
                )

                # Create reward relabeling client
                reward_relabel_client = RewardRelabelClient(
                    address=reward_relabel_address,
                    max_queue_size=OmegaConf.select(cfg, "distributed_reward_relabel.max_queue_size", default=100),
                    timeout=OmegaConf.select(cfg, "distributed_reward_relabel.timeout", default=60.0),
                    flush_interval=OmegaConf.select(cfg, "distributed_reward_relabel.flush_interval", default=0.1),
                )

                # Wrap with remote reward relabeling
                # Default batch size for remote reward relabeling (configurable via cfg.distributed_reward_relabel.batch_size)
                batch_size = OmegaConf.select(cfg, "distributed_reward_relabel.batch_size", default=32)
                online_buffer = AsyncRewardRelabelBuffer(
                    underlying_buffer=underlying_buffer,
                    reward_relabel_client=reward_relabel_client,
                    use_relative_rewards=use_relative_rewards,
                    batch_size=batch_size,
                    remove_obs_keys=remove_obs_keys,
                    post_transforms=[success_bonus_fn],
                    sampler=sampler,
                )
            else:
                # Use local Robometer replay buffer (synchronous reward relabeling)
                online_buffer = RobometerReplayBuffer(
                    reward_model=reward_model,
                    reward_model_config=exp_config,
                    sampler=sampler,
                    use_gt_rewards=use_gt_rewards,
                    use_relative_rewards=use_relative_rewards,
                    capacity=cfg.buffer_capacity,
                    remove_obs_keys=remove_obs_keys,
                    post_transforms=[success_bonus_fn],
                )
        else:
            online_buffer = ReplayBuffer(
                capacity=cfg.buffer_capacity,
                remove_obs_keys=remove_obs_keys,
                post_transforms=[success_bonus_fn],
            )

        if offline_buffer is not None and cfg.sample_ratio > 0:
            buffer = MixedReplayBuffer(
                buffer_1=offline_buffer,
                buffer_2=online_buffer,
                sample_ratio=cfg.sample_ratio,
                buffer_to_add_to=2,
                remove_obs_keys=remove_obs_keys,
                sampler=sampler,
            )
        else:
            buffer = online_buffer

        logger.info(f"Buffer capacity: {cfg.buffer_capacity}")
        # Option to evaluate on the first step (before any training)
        eval_on_first_step = OmegaConf.select(cfg, "eval_on_first_step", default=False)
        if eval_on_first_step:
            logger.info("Running initial evaluation before training...")
            initial_eval_worker = EvaluationWorker(
                eval_env=eval_env,
                device=device,
                num_episodes=cfg.eval_num_episodes,
                record_video=cfg.eval_record_video,
                logger=wandb_logger,
            )
            eval_metrics = initial_eval_worker.run(actor)
            wandb_logger.log(eval_metrics, step=0, prefix="eval")
            logger.info(f"Initial evaluation metrics: {eval_metrics}")

        # Start gRPC server (same pattern as train.py but async)
        def policy_info() -> Dict[str, Any]:
            return {
                "obs_keys": list(offline_buffer.obs_keys if offline_buffer else []),
                "chunk_size": cfg.chunk_size,
                "wandb_run_id": wandb_logger.logger.id if wandb_logger.logger else "",
                "wandb_project": wandb_logger.project or cfg.wandb_project,
                "wandb_entity": cfg.wandb_entity,
            }

        # For async relabeling, reward_model is None (handled by server)
        server_reward_model = reward_model if not use_async_reward_relabel else None

        # Get server host/port from config
        server_host = cfg.distributed.learner_server.host
        server_port = cfg.distributed.learner_server.port
        relabel_mode = cfg.distributed.learner_server.relabel_mode

        server = LearnerServer(
            buffer=online_buffer,
            reward_model=server_reward_model,
            host=server_host,
            port=server_port,
            relabel_mode=relabel_mode,
            max_msg_mb=cfg.distributed.learner_server.max_msg_mb,
            policy_info_provider=policy_info,
        )
        server.start()
        logger.success(f"Started learner server at {server_host}:{server_port}")

        # Publish initial actor params
        _sd0 = build_light_state_dict(actor)
        server.set_actor_state_dict(_sd0)

        # Handle ingestion based on offline algorithm presence
        if offline_algo is not None:
            # Disable ingestion during offline warm-start
            server.set_ingest_enabled(False)
            logger.info("Ingestion disabled during offline warm-start")

            # Run offline warm-start in a separate thread to allow server to accept connections
            def offline_warm_start():
                logger.info(f"Starting offline warm-start for {cfg.num_offline_steps} steps")
                for step in range(cfg.num_offline_steps):
                    offline_algo.train_step(logging_prefix="offline")
                    if step % 1000 == 0:
                        logger.info(f"Offline step {step}/{cfg.num_offline_steps}")
                logger.success("Offline warm-start completed")
                server.set_ingest_enabled(True)
                logger.success("Online ingestion enabled after offline warm-start")

            threading.Thread(target=offline_warm_start, daemon=True).start()
        else:
            # No offline algorithm - ingestion should be enabled from the start
            server.set_ingest_enabled(True)
            logger.success("Online ingestion enabled (no offline warm-start)")

        # Training loop
        train_meter = RateMeter(name="train", alpha=0.1)
        train_rl = RateLimiter(cfg.distributed.train_target_hz)

        def train_loop():
            last_log_t = time.time()
            last_step = getattr(algorithm, "_n_updates", 0)
            last_online_size = 0
            last_mixed_size = 0
            policy_step = 0
            logger.info("Starting training loop")
            logger.info(
                f"Initial buffer sizes: offline={len(offline_buffer) if offline_buffer else 0} online={len(online_buffer)} mixed={len(buffer)}"
            )

            while True:
                current_online_size = len(online_buffer)
                current_mixed_size = len(buffer)

                # Only train if there's new data or this is the first iteration
                if current_online_size > last_online_size or current_mixed_size > last_mixed_size or last_step == 0:
                    try:
                        t0 = time.time()
                        metrics = algorithm.train_step(logging_prefix="online/policy")
                        dt_update = time.time() - t0
                        train_meter.record(dt_update)
                        last_online_size = current_online_size
                        last_mixed_size = current_mixed_size
                        policy_step += 1
                        try:
                            wandb_logger.log_scalar("step", policy_step, prefix="online/policy")
                        except Exception:
                            pass
                    except Exception as e:
                        logger.exception(f"Training step failed: {e}")
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
                    detail = " ".join(
                        [f"{k}={v:0.3f}" for k, v in (metrics or {}).items() if isinstance(v, (int, float))]
                    )
                    logger.info(
                        f"step={step} sps={sps:0.2f} avg_dt={avg_sec:0.4f}s hz={hz:0.2f} buffer_online={len(online_buffer)} {detail}"
                    )
                    try:
                        wandb_logger.log_scalar("avg_seconds_per_update", float(avg_sec), prefix="online/policy")
                        wandb_logger.log_scalar("updates_per_sec", float(hz), prefix="online/policy")
                    except Exception:
                        pass
                    last_log_t = now
                    last_step = step

                actor_update_freq = cfg.distributed.actor_update_freq
                if step % actor_update_freq == 0:
                    _sd = build_light_state_dict(algorithm.actor)
                    server.set_actor_state_dict(_sd)

                save_interval = (
                    cfg.distributed.save_interval if cfg.distributed.save_interval is not None else cfg.save_interval
                )
                if step > 0 and step % save_interval == 0:
                    save_checkpoint(algorithm, save_dir, step)

                train_rl.throttle()

        threading.Thread(target=train_loop, daemon=True).start()

        logger.info(
            f"Async learner running at {server_host}:{server_port} (async_reward_relabel={use_async_reward_relabel})"
        )
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up remote reward relabeling client if used
            if use_async_reward_relabel and isinstance(online_buffer, AsyncRewardRelabelBuffer):
                logger.info("Stopping remote reward relabeling client...")
                online_buffer.stop()
            env.close()
            eval_env.close()
            try:
                wandb_logger.finish()
            except Exception:
                pass


def run_rollout(cfg: DictConfig):
    """Rollout worker function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Connect to learner for policy info
    rollout_log = logger.bind(process_name="ROLLOUT")
    learner_address = cfg.distributed.learner_server.address
    pc = PolicyClient(learner_address, ready_log_prefix="[ROLLOUT]", ready_timeout_per_attempt=5.0)
    info = pc.get_policy_info(block=True)
    obs_keys = info.get("obs_keys", None)
    policy_chunk = info.get("chunk_size") or cfg.chunk_size

    # Initialize wandb for rollout worker
    wandb_run_id = info.get("wandb_run_id", "")
    wandb_project = info.get("wandb_project", cfg.wandb_project)
    wandb_entity = info.get("wandb_entity", cfg.wandb_entity)

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
                exp_name=f"rollout_worker",
                id=wandb_run_id,
                project=wandb_project,
                entity=wandb_entity,
                prefix="rollout",
                job_type="rollout",
                tags=["rollout"],
                **({"settings": rb_settings} if rb_settings is not None else {}),
            )
            rollout_log.info(f"Joined wandb run: {wandb_run_id} (label={label})")
        except Exception as e:
            rollout_log.warning(f"Failed to join wandb run: {e}")

    # Create environment using rfm_rl's make_env (same as train.py)
    dinov2_model = AutoModel.from_pretrained(cfg.dinov2_model).to(device).eval()
    dinov2_processor = AutoImageProcessor.from_pretrained(cfg.dinov2_model)
    sentence_model = SentenceTransformer(cfg.sentence_model) if not cfg.use_full_state else None

    env, _ = make_env(
        env_name=cfg.env_name,
        num_envs=1,
        chunk_size=policy_chunk,
        use_full_state=cfg.use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model,
        render_mode="rgb_array",
        terminate_on_success=True,
    )
    env = SingleEnvVectorWrapper(env)
    action_space = getattr(env, "single_action_space", env.action_space)
    observation_space = getattr(env, "single_observation_space", env.observation_space)

    # Build actor using shared utility (same as train.py)
    actor, _, _ = build_actor_critic_models(observation_space, action_space, cfg, device, remove_obs_keys=["image"])

    # Wait indefinitely for initial weights from learner
    rollout_log.info(f"Waiting for initial weights from {learner_address} …")
    initial_sd = pc.fetch_latest(block=True)
    initial_fp = initial_sd.pop("__fingerprint__", None)
    actor.load_state_dict(initial_sd)
    if initial_fp:
        rollout_log.success(f"Initial policy fingerprint: {initial_fp}")

    streaming_buffer = StreamingBufferAdapter(
        learner_address,
        flush_every=cfg.distributed.rollout.flush_every,
        max_message_mb=cfg.distributed.rollout.max_message_mb,
    )

    # Create a separate inference actor to avoid tensor view conflicts
    inference_actor = type(actor)(actor.config if hasattr(actor, "config") else None).to(device)
    inference_actor.load_state_dict(actor.state_dict())
    inference_actor.eval()
    for param in inference_actor.parameters():
        param.requires_grad_(False)

    # Use the inference actor for rollouts
    worker = RolloutWorker(
        env=env,
        buffer=streaming_buffer,
        actor=inference_actor,
        num_rollouts=cfg.distributed.rollout.num_rollouts,
        device=device,
        count_by="step",
    )

    # Track last seen fingerprint
    last_fp = initial_fp or None

    def refresh():
        while True:
            try:
                new_state_dict = pc.fetch_latest()
                fp = new_state_dict.pop("__fingerprint__", None)
                actor.load_state_dict(new_state_dict)

                import copy

                inference_state_dict = copy.deepcopy(new_state_dict)
                inference_actor.load_state_dict(inference_state_dict)
                inference_actor.eval()
                for param in inference_actor.parameters():
                    param.requires_grad_(False)
                nonlocal last_fp
                if fp is not None:
                    if last_fp is None:
                        rollout_log.info(f"Policy fingerprint set: {fp}")
                    elif fp != last_fp:
                        rollout_log.success(f"Policy updated: {last_fp} -> {fp}")
                    last_fp = fp

            except Exception as e:
                rollout_log.warning(f"Weight update failed: {e}")
            refresh_secs = cfg.distributed.rollout.refresh_secs
            time.sleep(refresh_secs)

    threading.Thread(target=refresh, daemon=True).start()
    rollout_log.info(f"Starting rollout streaming to {learner_address} …")

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
                    "buffer_flushes_total": float(rollout_count),
                    "avg_seconds_per_cycle": float(rollout_meter.avg_seconds_per_update),
                    "cycles_per_sec": float(rollout_meter.updates_per_second),
                }
                rollout_logger.log_dict(to_log, step=rollout_step, prefix="rollout")
                rollout_step += 1
            rollout_rl.throttle()
        except RuntimeError as e:
            rollout_failures += 1
            if "view" in str(e) and "inplace" in str(e):
                rollout_log.warning(
                    f"PyTorch in-place operation error (attempt {rollout_failures}/{max_consecutive_failures}): {e}"
                )
            else:
                rollout_log.error(f"Runtime error (attempt {rollout_failures}/{max_consecutive_failures}): {e}")

            if rollout_failures >= max_consecutive_failures:
                rollout_log.error(f"Too many consecutive failures ({rollout_failures}), exiting…")
                raise

            time.sleep(1)
            try:
                rollout_log.info("Attempting to refresh actor weights…")
                new_state_dict = pc.fetch_latest()
                import copy

                fp = new_state_dict.pop("__fingerprint__", None)
                inference_state_dict = copy.deepcopy(new_state_dict)
                inference_actor.load_state_dict(inference_state_dict)
                inference_actor.eval()
                for param in inference_actor.parameters():
                    param.requires_grad_(False)
                if fp is not None:
                    if last_fp is None or fp != last_fp:
                        rollout_log.success(f"Actor weights refreshed successfully; new fp={fp}")
                    else:
                        rollout_log.info(f"Actor weights refresh complete; fp unchanged={fp}")
                    last_fp = fp
                else:
                    rollout_log.success("Actor weights refreshed successfully")
            except Exception as refresh_e:
                rollout_log.error(f"Failed to refresh actor weights: {refresh_e}")

        except Exception as e:
            rollout_failures += 1
            rollout_log.exception(f"Unexpected error (attempt {rollout_failures}/{max_consecutive_failures}): {e}")

            if rollout_failures >= max_consecutive_failures:
                rollout_log.error(f"Too many consecutive failures ({rollout_failures}), exiting…")
                raise

            time.sleep(2)


def run_eval(cfg: DictConfig):
    """Eval worker function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_log = logger.bind(process_name="EVAL")
    learner_address = cfg.distributed.learner_server.address
    pc = PolicyClient(learner_address, ready_log_prefix="[EVAL]", ready_timeout_per_attempt=5.0)
    info = pc.get_policy_info(block=True)
    obs_keys = info.get("obs_keys", None)
    policy_chunk = info.get("chunk_size") or cfg.chunk_size

    # Initialize wandb for eval worker
    wandb_run_id = info.get("wandb_run_id", "")
    wandb_project = info.get("wandb_project", cfg.wandb_project)
    wandb_entity = info.get("wandb_entity", cfg.wandb_entity)

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
                exp_name=f"eval_worker",
                id=wandb_run_id,
                project=wandb_project,
                entity=wandb_entity,
                prefix="eval",
                job_type="eval",
                tags=["eval"],
                **({"settings": ev_settings} if ev_settings is not None else {}),
            )
            eval_log.info(f"Joined wandb run: {wandb_run_id} (label={label})")
        except Exception as e:
            eval_log.warning(f"Failed to join wandb run: {e}")

    dinov2_model = AutoModel.from_pretrained(cfg.dinov2_model).to(device).eval()
    dinov2_processor = AutoImageProcessor.from_pretrained(cfg.dinov2_model)
    sentence_model = SentenceTransformer(cfg.sentence_model) if not cfg.use_full_state else None

    env, _ = make_env(
        env_name=cfg.env_name,
        num_envs=1,
        chunk_size=policy_chunk,
        use_full_state=cfg.use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model,
        render_mode="rgb_array",
        terminate_on_success=True,
    )
    env = SingleEnvVectorWrapper(env)
    action_space = getattr(env, "single_action_space", env.action_space)
    observation_space = getattr(env, "single_observation_space", env.observation_space)

    # Build actor using shared utility (same as train.py)
    actor, _, _ = build_actor_critic_models(observation_space, action_space, cfg, device, remove_obs_keys=["image"])

    # Wait indefinitely for initial weights from learner
    eval_log.info(f"Waiting for initial weights from {learner_address} …")
    initial_sd = pc.fetch_latest(block=True)
    initial_fp = initial_sd.pop("__fingerprint__", None)
    actor.load_state_dict(initial_sd)
    if initial_fp:
        eval_log.success(f"Initial policy fingerprint: {initial_fp}")

    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)

    # EvaluationWorker expects a vectorized env (even with num_envs=1)
    worker = EvaluationWorker(
        eval_env=env,
        device=device,
        num_episodes=cfg.eval_num_episodes,
        record_video=cfg.eval_record_video,
        logger=eval_logger,
    )
    eval_log.info(f"Starting async evaluator against {learner_address} …")

    eval_failures = 0
    max_consecutive_failures = 3
    eval_step = 0

    last_eval_fp = initial_fp or None
    eval_meter = RateMeter(name="eval", alpha=0.1)

    try:
        while True:
            try:
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
                        eval_log.info(f"Eval policy fingerprint set: {fp}")
                    elif fp != last_eval_fp:
                        eval_log.success(f"Eval policy updated: {last_eval_fp} -> {fp}")
                    else:
                        eval_log.info(f"Eval policy unchanged: {fp}")
                    last_eval_fp = fp

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
                    eval_log.success(f"Completed evaluation: {eval_metrics}")

                eval_failures = 0
                eval_every = cfg.distributed.eval.eval_every
                time.sleep(eval_every)

            except RuntimeError as e:
                eval_failures += 1
                if "view" in str(e) and "inplace" in str(e):
                    eval_log.warning(
                        f"PyTorch in-place operation error (attempt {eval_failures}/{max_consecutive_failures}): {e}"
                    )
                else:
                    eval_log.error(f"Runtime error (attempt {eval_failures}/{max_consecutive_failures}): {e}")

                if eval_failures >= max_consecutive_failures:
                    eval_log.error(f"Too many consecutive failures ({eval_failures}), exiting…")
                    break

                time.sleep(2)

            except Exception as e:
                eval_failures += 1
                eval_log.exception(f"Unexpected error (attempt {eval_failures}/{max_consecutive_failures}): {e}")

                if eval_failures >= max_consecutive_failures:
                    eval_log.error(f"Too many consecutive failures ({eval_failures}), exiting…")
                    break

                time.sleep(2)

    except KeyboardInterrupt:
        pass


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config_distributed")
def main(cfg: DictConfig):
    """Main entry point that dispatches to the appropriate mode based on config."""
    mode = cfg.mode

    if mode == "learner":
        run_learner(cfg)
    elif mode == "rollout":
        run_rollout(cfg)
    elif mode == "eval":
        run_eval(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: learner, rollout, eval")


if __name__ == "__main__":
    main()
