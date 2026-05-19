#!/usr/bin/env python3
"""
Simple script to start the reward relabeling server using Hydra config.

Supports reward backends exposed via Hydra `reward_model`:
- **Robometer** (default): Qwen-based progress and success (`reward_model=robometer`)
- **RoboReward**: discrete progress baseline (`reward_model=roboreward`)

Usage:
    # Default Robometer config (reads from configs/reward_relabel_server.yaml):
    uv run python scripts/start_reward_relabel_server.py

    # RoboReward baseline (4B):
    uv run python scripts/start_reward_relabel_server.py \
        reward_model=roboreward \
        reward_model.model_path=teetone/RoboReward-4B \
        server.port=50052 \
        server.host="0.0.0.0" \
        device=cuda \
        server.image_keys='["observation/exterior_image_1_left"]'

    # Robometer with a Hugging Face RFM / Qwen checkpoint (same stack as Robometer):
    uv run python scripts/start_reward_relabel_server.py \
        reward_model=robometer \
        reward_model.model_path="<HF_MODEL_ID>" \
        server.port=50052 \
        server.host="0.0.0.0" \
        device=cuda \
        server.image_keys='["observation/exterior_image_1_left"]'

See README.md (Remote Reward Relabeling) for full train_dsrl and eval examples.
"""

import time
import torch
from loguru import logger
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from robometer_policy_learning.distributed.servers.reward_relabel_server import RewardRelabelServer


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="reward_relabel_server")
def main(cfg: DictConfig):
    """Start reward relabeling server using Hydra config."""
    # Get reward model config
    if not hasattr(cfg, "reward_model"):
        raise ValueError(
            "reward_model config is required. Use --config-name=reward_relabel_server or specify reward_model=robometer"
        )

    reward_model_cfg = cfg.reward_model
    model_path = reward_model_cfg.model_path

    if not model_path:
        raise ValueError("reward_model.model_path must be specified in config")

    # Get reward model type (default: robometer)
    reward_model_type = OmegaConf.select(cfg, "reward_model.model_type", default="robometer")

    # Get server config
    server_host = cfg.server.host
    server_port = cfg.server.port
    max_workers = cfg.server.max_workers
    max_msg_mb = cfg.server.max_msg_mb
    image_keys = OmegaConf.select(cfg, "server.image_keys", default=None)
    language_key = OmegaConf.select(cfg, "server.language_key", default="language")

    # Convert image_keys from OmegaConf list to Python list if needed
    if image_keys is not None:
        image_keys = list(image_keys) if not isinstance(image_keys, list) else image_keys

    # Get device
    device_str = cfg.device if hasattr(cfg, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Get sentence transformer model name (optional, only for Robometer)
    sentence_model_name = OmegaConf.select(cfg, "sentence_model", default=None)

    # Get RoboReward-specific config
    roboreward_max_new_tokens = OmegaConf.select(cfg, "reward_model.roboreward_max_new_tokens", default=128)
    roboreward_use_unsloth = OmegaConf.select(cfg, "reward_model.roboreward_use_unsloth", default=False)

    logger.info(f"Starting reward relabel server with model_type={reward_model_type}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Server: {server_host}:{server_port}")
    if image_keys:
        logger.info(f"Image keys: {image_keys}")

    # Create and start server (model loading happens inside RewardRelabelServer)
    server = RewardRelabelServer(
        model_path=model_path,
        host=server_host,
        port=server_port,
        max_workers=max_workers,
        max_msg_mb=max_msg_mb,
        image_keys=image_keys,
        language_key=language_key,
        sentence_model_name=sentence_model_name,
        device=device,
        reward_model_type=reward_model_type,
        roboreward_max_new_tokens=roboreward_max_new_tokens,
        roboreward_use_unsloth=roboreward_use_unsloth,
    )

    server.start()

    # Server runs until interrupted
    try:
        logger.info(f"Reward relabeling server ({reward_model_type}) running. Press Ctrl-C to stop.")
        while True:
            time.sleep(5)
            stats = server.get_stats()
        #     logger.info(
        #         f"Stats: batches={stats['batches_processed']}, "
        #         f"transitions={stats['transitions_processed']}, "
        #         f"errors={stats['errors']}"
        #     )
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server.stop()
        logger.success("Server stopped")


if __name__ == "__main__":
    main()
