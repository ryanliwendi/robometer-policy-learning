# Robometer Policy Learning

A flexible reinforcement \ imitation learning framework supporting multiple algorithms (SAC, IQL, BC), reward models (Robometer, RoboReward), and distributed training. We will soon include detailed guides for DSRL (Diffusion-steering RL) with Pi0/0.5 on LIBERO and Real World tasks.

> **Note**
> This repository is under active development, so some modules and features may change over time. You may encounter issues when using features that is not yet documented in this README. Please feel free to open an issue — we will do our best to help.

## Table of Contents

- [Setup](#setup)
- [Training](#training)
  - [Basic Training](#basic-training-ground-truth-rewards)
  - [Training with Reward Model](#training-with-robometer-reward-model)
  - [Example: Online RL in LIBERO](#example-online-rl-in-libero)
  - [DSRL + Pi0 on LIBERO - Coming Soon]
- [Real-World Online RL with DSRL + Remote Reward Labeling - Coming Soon]
---

## Setup

### Prerequisites

- Git
- Python 3.10+
- NVIDIA Drivers (for GPU support)

### Installation

1. **Install `uv` (if not already installed):**
  ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
2. **Clone and setup submodules (required for DSRL/LIBERO):**
  ```bash
   git submodule init
   git submodule update --recursive
  ```
3. **Create and sync the virtual environment:**
  ```bash
   # Install dependencies from pyproject.toml
   uv sync

   # Optional: Install with development dependencies
   uv sync --extra dev
  ```

**Activate the environment:**

```bash
 source .venv/bin/activate
```



**Note that this repo assumes robometer is installed as a git submodule and located at `./robometer`. If you made any changes to robometer/have your own robometer fork, replace the submodule** 

---

## Training

### Basic Training (Ground Truth Rewards)

Train with ground truth rewards (default, no reward model - Online only):

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  env.use_gt_rewards=true
```

Offline pretraining with online fine-tuning:

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  algorithm@offline_algorithm=iql \
  algorithm@online_algorithm=sac \
  alg.offline_alg_name=iql \
  alg.online_alg_name=sac \
  env.use_gt_rewards=true
```

### Training with Robometer Reward Model

Train with Robometer reward model (Online only):

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  reward_model=robometer \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  env.use_gt_rewards=false \
  reward_model.model_path=robometer/Robometer-4B
```

Train with Robometer reward model (Offline-to-online):

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs   \
  --config-name=config   \
  reward_model=robometer   \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  algorithm@offline_algorithm=iql   \
  alg.offline_alg_name=iql   \
  env.use_gt_rewards=false   \
  reward_model.model_path=robometer/Robometer-4B
```

### Example: Online RL in LIBERO

Train a SAC policy in LIBERO using ground-truth rewards:

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs  \
  --config-name=libero_online_rl  \
  env.env_name=libero_90  \
  env.task_id=28  \
  env.use_gt_rewards=true  \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  training.num_rollouts=100000  \
  training.seed=100  \
  eval.eval_freq=5000  \
  eval.eval_num_episodes=20  \
  online_algorithm.num_critic_updates_per_actor_update=1  \
  online_algorithm.learning_starts=5000  \
  online_algorithm.critic_optimizer_lr=1e-5  \
  online_algorithm.actor_optimizer_lr=1e-5  \
  logging.wandb_name=libero_online_rl_gt_rewards  \
  logging.wandb_entity=YOUR_WANDB_ENTITY
```

Train a SAC policy in LIBERO using Robometer rewards:

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs  \
  --config-name=libero_online_rl  \
  reward_model=robometer  \
  reward_model.model_path=robometer/Robometer-4B  \
  reward_model.add_estimated_reward=true  \
  reward_model.use_success_detection=false  \
  env.env_name=libero_90  \
  env.task_id=28  \
  env.use_gt_rewards=false  \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  training.num_rollouts=100000  \
  training.seed=100  \
  eval.eval_freq=5000  \
  eval.eval_num_episodes=20  \
  online_algorithm.num_critic_updates_per_actor_update=1  \
  online_algorithm.learning_starts=5000  \
  online_algorithm.critic_optimizer_lr=1e-5  \
  online_algorithm.actor_optimizer_lr=1e-5  \
  logging.wandb_name=libero_online_rl_robometer_rewards  \
  logging.wandb_entity=YOUR_WANDB_ENTITY
```

You should see evaluation curves similar to the example below:

<p align="center">
  <img src="docs/libero_rl.png" alt="LIBERO RL Experiments" width="600"/>
</p>

---

# Real-World Online RL with DSRL + Remote Reward Labeling

Coming soon...
Files are in this repo but need to be cleaned up, should be done by mid June. 

Also coming soon: DSRL+Pi0 sanity check command before running real world online RL. 

---

# Additional Resources

- gRPC service definitions:
  `robometer_policy_learning/distributed/protos/`

- Configuration files:
  `robometer_policy_learning/configs/`

- Algorithm-specific configs:
  `robometer_policy_learning/configs/algorithm/`

