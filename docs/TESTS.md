## RL Training
### Using ground-truth rewards

Only online RL:

```bash
uv run python scripts/train.py \
  --config-path=../configs \
  --config-name=config \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  env.use_gt_rewards=true \
  env.env_name="Meta-World/MT1/open-drawer-v3"
```

Offline-to-online RL:
```bash
uv run python scripts/train.py   \
  --config-path=../configs   \
  --config-name=config   \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  algorithm@offline_algorithm=iql   \
  alg.offline_alg_name=iql   \
  env.use_gt_rewards=true   \
  env.h5_dataset_path: "/scr/shared/reward_fm/policy_training_datasets/metaworld_generation_converted.h5" \
  env.env_name="Meta-World/MT1/open-drawer-v3"

```

### Using Robometer reward model

Train with Robometer reward model (Online only):

```bash
uv run python scripts/train.py \
  --config-path=../configs \
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
  --config-path=../configs   \
  --config-name=config   \
  reward_model=robometer   \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  algorithm@offline_algorithm=iql   \
  alg.offline_alg_name=iql   \
  env.use_gt_rewards=false   \
  env.h5_dataset_path: "/scr/shared/reward_fm/policy_training_datasets/metaworld_generation_converted.h5" \
  reward_model.model_path=robometer/Robometer-4B

```

## DSRL Training
### DSRL on LIBERO with Pi0
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train_dsrl.py \
  --config-name=dsrl_config_new \
  logging.wandb_entity=ykorkmaz  \
  logging.wandb_name=train \
  dsrl.action_exec_len=20  \
  env.use_gt_rewards=true  \
  eval.eval_on_first_step=true 
```

### DSRL with Robometer relabeling
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train_dsrl.py \
  --config-name=dsrl_config_new \
  reward_model=robometer   \
  logging.wandb_entity=ykorkmaz  \
  logging.wandb_name=train \
  dsrl.action_exec_len=20  \
  env.use_gt_rewards=false  \
  eval.eval_on_first_step=true \
  reward_model.model_path=rewardfm/ant-rfm-qwen-4gpu-bs64-pref-prog-2frames-uniform-l1-20251219-182139
```