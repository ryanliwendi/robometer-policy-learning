"""
Integration tests for IQL (Implicit Q-Learning) on Pendulum-v1.

Covers the full offline-to-online pipeline:
  1. Buffer generation from random exploration
  2. Offline pre-training with IQL
  3. Online fine-tuning with IQL (collecting new transitions while training)

Both MLP and Transformer architectures are tested.

Marked ``integration`` so they can be selected / skipped independently:
    pytest -m integration          # run only integration tests
    pytest -m "not integration"    # skip integration tests
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from tqdm import tqdm

from robometer_policy_learning.envs.obs_wrappers import FlatToDictObsWrapper
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.samplers import RandomSampler, ChunkedSequentialSampler
from robometer_policy_learning.algorithms.iql import IQL, IQLConfig
from robometer_policy_learning.modules.mlp import MLPActor, MLPActorConfig, MLPCritic, MLPCriticConfig
from robometer_policy_learning.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
    TransformerCritic,
    TransformerCriticConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
FEATURIZER_DIM = 128
ENV_ID = "Pendulum-v1"


def _make_env(env_id=ENV_ID, num_envs=1, **env_kwargs):
    def _factory():
        return FlatToDictObsWrapper(gym.make(env_id, **env_kwargs))
    return gym.vector.SyncVectorEnv([_factory for _ in range(num_envs)])


def _obs_and_action_spaces(env):
    obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
    act_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
    return obs_space, act_space


def _featurizer_for(obs_space):
    return {key: [FEATURIZER_DIM] for key in obs_space.spaces}


# ---- MLP factories -------------------------------------------------------

def _make_mlp_actor_critic_vnet(obs_space, act_space, hidden_dims=(128, 128)):
    """Create MLP actor, Q-critic, and V-network."""
    featurizer = _featurizer_for(obs_space)

    actor_cfg = MLPActorConfig(
        observation_space=obs_space,
        action_space=act_space,
        featurizer=featurizer,
        hidden_dims=hidden_dims,
        activation="relu",
        use_tanh_output=False,
        deterministic=False,
        log_std_init=0,
        log_std_min=-20.0,
        log_std_max=2.0,
    )
    actor = MLPActor(actor_cfg).to(DEVICE)

    critic_cfg = MLPCriticConfig(
        observation_space=obs_space,
        action_space=act_space,
        featurizer=featurizer,
        hidden_dims=hidden_dims,
        activation="relu",
        use_action=True,
    )
    critic = MLPCritic(critic_cfg).to(DEVICE)

    v_net_cfg = MLPCriticConfig(
        observation_space=obs_space,
        action_space=act_space,
        featurizer=featurizer,
        hidden_dims=hidden_dims,
        activation="relu",
        use_action=False,
    )
    v_net = MLPCritic(v_net_cfg).to(DEVICE)

    return actor, critic, v_net


# ---- Transformer factories -----------------------------------------------

def _make_transformer_actor_critic_vnet(obs_space, act_space, hidden_dims=(128, 128), chunk_size=4):
    """Create Transformer actor, Q-critic, and V-network."""
    featurizer = _featurizer_for(obs_space)

    actor_cfg = TransformerActorConfig(
        observation_space=obs_space,
        action_space=act_space,
        featurizer=featurizer,
        chunk_size=chunk_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        transformer_dropout=0.0,
        transformer_activation="gelu",
        feature_hidden_dims=hidden_dims,
        dinov2_model=None,
        dinov2_processor=None,
        use_language_embeddings=False,
        use_tanh_output=False,
        log_std_init=0,
        log_std_min=-20.0,
        log_std_max=2.0,
    )
    actor = TransformerActor(actor_cfg).to(DEVICE)

    critic_cfg = TransformerCriticConfig(
        observation_space=obs_space,
        action_space=act_space,
        featurizer=featurizer,
        chunk_size=chunk_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        transformer_dropout=0.0,
        transformer_activation="gelu",
        feature_hidden_dims=hidden_dims,
        dinov2_model=None,
        dinov2_processor=None,
        use_language_embeddings=False,
        use_action=True,
    )
    critic = TransformerCritic(critic_cfg).to(DEVICE)

    v_net_cfg = TransformerCriticConfig(
        observation_space=obs_space,
        action_space=act_space,
        featurizer=featurizer,
        chunk_size=chunk_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        transformer_dropout=0.0,
        transformer_activation="gelu",
        feature_hidden_dims=hidden_dims,
        dinov2_model=None,
        dinov2_processor=None,
        use_language_embeddings=False,
        use_action=False,
    )
    v_net = TransformerCritic(v_net_cfg).to(DEVICE)

    return actor, critic, v_net


# ---- Algorithm factories -------------------------------------------------

def _make_iql(actor, critic, v_net, buffer, batch_size=128, lr=3e-4, gamma=0.99,
              advantage_temp=1.00):
    cfg = IQLConfig(
        actor=actor,
        critic=critic,
        v_net=v_net,
        buffer=buffer,
        logger=None,
        batch_size=batch_size,
        gamma=gamma,
        tau=0.005,
        num_critics=2,
        n_critics_to_sample=2,
        advantage_temp=advantage_temp,
        expectile=0.7,
        clip_score=100.0,
        policy_extraction="awr",
        actor_optimizer_lr=lr,
        critic_optimizer_lr=lr,
        v_net_optimizer_lr=lr,
        pooled_critic_features=True,
    )
    return cfg.create()


# ---- Buffer generation ----------------------------------------------------

def _fill_buffer_vectorized(env, buffer, n_transitions, desc="fill-buffer"):
    """Collect random transitions from a vectorized env into *buffer*."""
    num_envs = env.num_envs
    obs, _ = env.reset()
    ep_ids = list(range(num_envs))
    next_ep_id = num_envs
    steps = [0] * num_envs
    total = 0

    pbar = tqdm(total=n_transitions, desc=desc, unit="trans", leave=True)
    while total < n_transitions:
        actions = env.action_space.sample()
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for i in range(num_envs):
            act_i = actions[i] if actions.ndim > 1 else actions
            buffer.add(
                obs={k: v[i] for k, v in obs.items()},
                action=act_i,
                reward=float(rewards[i]),
                next_obs={k: v[i] for k, v in next_obs.items()},
                done=bool(terminateds[i]),
                truncated=bool(truncateds[i]),
                episode_id=ep_ids[i],
                step_in_episode=steps[i],
            )
            steps[i] += 1
            if terminateds[i] or truncateds[i]:
                ep_ids[i] = next_ep_id
                next_ep_id += 1
                steps[i] = 0

        total += num_envs
        pbar.update(num_envs)
        obs = next_obs
    pbar.close()


# ---- Pre-training ---------------------------------------------------------

def _pretrain_iql(iql, n_steps, desc="IQL-pretrain"):
    """Run offline IQL training for *n_steps* gradient updates."""
    metrics_history = []
    pbar = tqdm(range(n_steps), desc=desc, unit="step", leave=True)
    for _ in pbar:
        metrics = iql.train_step()
        if metrics:
            metrics_history.append(metrics)
            pbar.set_postfix(
                q_loss=f"{metrics['critic_loss']:.3f}",
                v_loss=f"{metrics['v_loss']:.3f}",
                actor_loss=f"{metrics['actor_loss']:.3f}",
            )
    return metrics_history


# ---- Online fine-tuning with IQL -----------------------------------------

def _finetune_iql_vectorized(
    env, actor, iql, buffer, n_env_steps, warmup_steps=0, desc="IQL-finetune",
):
    """Online IQL fine-tuning using vectorized envs.

    Collects new transitions with the current policy and interleaves
    IQL gradient updates.  For chunked (Transformer) actors the full
    action chunk is executed across all envs before re-querying.
    """
    num_envs = env.num_envs
    obs, _ = env.reset()
    ep_ids = list(range(num_envs))
    next_ep_id = num_envs
    steps = [0] * num_envs
    ep_rewards = [0.0] * num_envs
    last_ep_return = float("nan")
    completed_eps = 0
    total = 0
    action_chunk: list = []

    pbar = tqdm(total=n_env_steps, desc=desc, unit="trans", leave=True)
    while total < n_env_steps:
        if total < warmup_steps:
            actions = env.action_space.sample()
        else:
            if not action_chunk:
                obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in obs.items()}
                raw, _ = actor.act(obs_t)
                raw_np = raw.cpu().numpy()
                if raw_np.ndim == 3:
                    action_chunk = [raw_np[:, t, :] for t in range(raw_np.shape[1])]
                else:
                    action_chunk = [raw_np]
            actions = action_chunk.pop(0)

        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for i in range(num_envs):
            act_i = actions[i] if actions.ndim > 1 else actions
            buffer.add(
                obs={k: v[i] for k, v in obs.items()},
                action=act_i,
                reward=float(rewards[i]),
                next_obs={k: v[i] for k, v in next_obs.items()},
                done=bool(terminateds[i]),
                truncated=bool(truncateds[i]),
                episode_id=ep_ids[i],
                step_in_episode=steps[i],
            )
            ep_rewards[i] += float(rewards[i])
            steps[i] += 1
            if terminateds[i] or truncateds[i]:
                last_ep_return = ep_rewards[i]
                completed_eps += 1
                ep_ids[i] = next_ep_id
                next_ep_id += 1
                steps[i] = 0
                ep_rewards[i] = 0.0

        total += num_envs
        pbar.update(num_envs)
        pbar.set_postfix(ep=completed_eps, ret=f"{last_ep_return:.0f}", buf=len(buffer))
        obs = next_obs

        if total >= warmup_steps:
            iql.train_step()
    pbar.close()


# ---- Evaluation -----------------------------------------------------------

def _evaluate_vectorized(env, actor, n_episodes=10):
    """Run deterministic evaluation across parallel envs, return episode returns."""
    num_envs = env.num_envs
    obs, _ = env.reset()
    ep_rewards = [0.0] * num_envs
    returns: list[float] = []

    while len(returns) < n_episodes:
        obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in obs.items()}
        raw, _ = actor.act(obs_t, deterministic=True)
        raw_np = raw.cpu().numpy()
        actions = raw_np[:, 0, :] if raw_np.ndim == 3 else raw_np

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for i in range(num_envs):
            ep_rewards[i] += float(rewards[i])
            if terminateds[i] or truncateds[i]:
                returns.append(ep_rewards[i])
                ep_rewards[i] = 0.0
                if len(returns) >= n_episodes:
                    break

    return returns[:n_episodes]


# ---------------------------------------------------------------------------
# IQL — MLP on Pendulum-v1
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIQLMLP:
    """MLP IQL on Pendulum-v1: offline pretrain + online IQL finetune."""

    NUM_ENVS = 4

    def test_actor_forward(self):
        env = _make_env()
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _, _ = _make_mlp_actor_critic_vnet(obs_space, act_space)
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in obs.items()}
        action, _ = actor.act(obs_t)
        assert action.shape[-1] == act_space.shape[0]
        env.close()

    def test_v_net_forward(self):
        env = _make_env()
        obs_space, act_space = _obs_and_action_spaces(env)
        _, _, v_net = _make_mlp_actor_critic_vnet(obs_space, act_space)
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in obs.items()}
        with torch.no_grad():
            value = v_net(obs_t)
        assert value.shape == (1, 1)
        env.close()

    def test_iql_train_steps(self):
        """IQL runs train_step on a buffer of random transitions and returns metrics."""
        env = _make_env(num_envs=self.NUM_ENVS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic, v_net = _make_mlp_actor_critic_vnet(obs_space, act_space)

        buf = ReplayBuffer(capacity=5000, sampler=RandomSampler())
        _fill_buffer_vectorized(env, buf, n_transitions=2000, desc="fill-MLP")

        iql = _make_iql(actor, critic, v_net, buf, batch_size=64)

        for _ in range(5):
            metrics = iql.train_step()

        assert "critic_loss" in metrics
        assert "v_loss" in metrics
        assert "actor_loss" in metrics
        assert np.isfinite(metrics["critic_loss"])
        assert np.isfinite(metrics["v_loss"])
        assert np.isfinite(metrics["actor_loss"])
        env.close()

    def test_pretrain_and_finetune(self):
        """Full pipeline: generate buffer → IQL pretrain → IQL online finetune.

        Verifies that:
          1. IQL losses are finite after offline pre-training.
          2. IQL can fine-tune online (collect + train) without errors.
        """
        env = _make_env(num_envs=self.NUM_ENVS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic, v_net = _make_mlp_actor_critic_vnet(obs_space, act_space)

        buf = ReplayBuffer(capacity=10000, sampler=RandomSampler())
        _fill_buffer_vectorized(env, buf, n_transitions=5000, desc="offline-MLP")

        iql = _make_iql(actor, critic, v_net, buf, batch_size=64)
        history = _pretrain_iql(iql, n_steps=200, desc="IQL-MLP-pretrain")

        assert len(history) > 0
        last = history[-1]
        for key in ("critic_loss", "v_loss", "actor_loss"):
            assert np.isfinite(last[key]), f"{key} is not finite: {last[key]}"

        _finetune_iql_vectorized(
            env, actor, iql, buf,
            n_env_steps=2000,
            warmup_steps=0,
            desc="IQL-MLP-finetune",
        )

        eval_obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in eval_obs.items()}
        action, _ = actor.act(obs_t, deterministic=True)
        assert action.shape[-1] == act_space.shape[0], "Actor produces valid actions after fine-tuning"
        env.close()

    def test_converges(self):
        """IQL pretrain on random data → IQL online fine-tune → evaluate.

        Pendulum-v1: 3D obs, 1D continuous action, 200 steps/episode.
        Random policy ≈ -1200 to -1600.  Well-trained ≈ -200 to -300.
        Threshold: -800 (clearly above random).
        """
        env = _make_env(num_envs=self.NUM_ENVS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic, v_net = _make_mlp_actor_critic_vnet(obs_space, act_space)

        buf = ReplayBuffer(capacity=50000, sampler=RandomSampler())
        _fill_buffer_vectorized(env, buf, n_transitions=10000, desc="offline-MLP-converge")

        iql = _make_iql(actor, critic, v_net, buf, batch_size=128, advantage_temp=1.0)
        _pretrain_iql(iql, n_steps=2000, desc="IQL-MLP-converge-pretrain")

        _finetune_iql_vectorized(
            env, actor, iql, buf,
            n_env_steps=20000,
            warmup_steps=2000,
            desc="IQL-MLP-converge-finetune",
        )

        eval_env = _make_env(num_envs=self.NUM_ENVS)
        eval_returns = _evaluate_vectorized(eval_env, actor, n_episodes=10)
        avg_return = np.mean(eval_returns)
        env.close()
        eval_env.close()

        assert avg_return > -800, (
            f"IQL MLP on Pendulum-v1 did not converge: "
            f"avg return = {avg_return:.1f} (expected > -800, random ≈ -1400)"
        )


# ---------------------------------------------------------------------------
# IQL — Transformer on Pendulum-v1
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIQLTransformer:
    """Transformer IQL on Pendulum-v1: offline pretrain + online IQL finetune."""

    NUM_ENVS = 4
    CHUNK_SIZE = 3

    def test_actor_forward(self):
        env = _make_env()
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _, _ = _make_transformer_actor_critic_vnet(
            obs_space, act_space, chunk_size=self.CHUNK_SIZE,
        )
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in obs.items()}
        action, _ = actor.act(obs_t)
        assert action.shape[1] == self.CHUNK_SIZE
        assert action.shape[2] == act_space.shape[0]
        env.close()

    def test_v_net_forward(self):
        env = _make_env()
        obs_space, act_space = _obs_and_action_spaces(env)
        _, _, v_net = _make_transformer_actor_critic_vnet(
            obs_space, act_space, chunk_size=self.CHUNK_SIZE,
        )
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in obs.items()}
        with torch.no_grad():
            value = v_net(obs_t)
        assert value.shape == (1, 1)
        env.close()

    def test_iql_train_steps_with_chunked_sampler(self):
        """IQL with ChunkedSequentialSampler runs train_step and returns metrics."""
        env = _make_env(num_envs=self.NUM_ENVS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic, v_net = _make_transformer_actor_critic_vnet(
            obs_space, act_space, chunk_size=self.CHUNK_SIZE,
        )
        sampler = ChunkedSequentialSampler(chunk_size=self.CHUNK_SIZE, gamma=0.99)
        buf = ReplayBuffer(capacity=5000, sampler=sampler)
        _fill_buffer_vectorized(env, buf, n_transitions=2000, desc="fill-Transformer")

        iql = _make_iql(actor, critic, v_net, buf, batch_size=64)

        for _ in range(5):
            metrics = iql.train_step()

        assert "critic_loss" in metrics
        assert "v_loss" in metrics
        assert "actor_loss" in metrics
        assert np.isfinite(metrics["critic_loss"])
        assert np.isfinite(metrics["v_loss"])
        assert np.isfinite(metrics["actor_loss"])
        env.close()

    def test_pretrain_and_finetune(self):
        """Full pipeline: generate buffer → IQL pretrain → IQL online finetune.

        Same pipeline as MLP but with Transformer architecture and
        ChunkedSequentialSampler throughout.
        """
        env = _make_env(num_envs=self.NUM_ENVS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic, v_net = _make_transformer_actor_critic_vnet(
            obs_space, act_space, chunk_size=self.CHUNK_SIZE,
        )

        sampler = ChunkedSequentialSampler(chunk_size=self.CHUNK_SIZE, gamma=0.99)
        buf = ReplayBuffer(capacity=10000, sampler=sampler)
        _fill_buffer_vectorized(env, buf, n_transitions=5000, desc="offline-Transformer")

        iql = _make_iql(actor, critic, v_net, buf, batch_size=64)
        history = _pretrain_iql(iql, n_steps=200, desc="IQL-Transformer-pretrain")

        assert len(history) > 0
        last = history[-1]
        for key in ("critic_loss", "v_loss", "actor_loss"):
            assert np.isfinite(last[key]), f"{key} is not finite: {last[key]}"

        _finetune_iql_vectorized(
            env, actor, iql, buf,
            n_env_steps=2000,
            warmup_steps=500,
            desc="IQL-Transformer-finetune",
        )

        eval_obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) for k, v in eval_obs.items()}
        action, _ = actor.act(obs_t, deterministic=True)
        assert action.shape[1] == self.CHUNK_SIZE
        assert action.shape[2] == act_space.shape[0], "Actor produces valid chunked actions after fine-tuning"
        env.close()

    def test_converges(self):
        """IQL pretrain on random data → IQL online fine-tune → evaluate.

        Transformer variant with ChunkedSequentialSampler throughout.
        Pendulum-v1: random ≈ -1200 to -1600.  Threshold: -300.
        """
        env = _make_env(num_envs=self.NUM_ENVS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic, v_net = _make_transformer_actor_critic_vnet(
            obs_space, act_space, chunk_size=self.CHUNK_SIZE,
        )

        sampler = ChunkedSequentialSampler(chunk_size=self.CHUNK_SIZE, gamma=0.99)
        buf = ReplayBuffer(capacity=50000, sampler=sampler)
        _fill_buffer_vectorized(env, buf, n_transitions=10000, desc="offline-Transformer-converge")

        iql = _make_iql(actor, critic, v_net, buf, batch_size=128, advantage_temp=2.5, lr=1e-4)
        _pretrain_iql(iql, n_steps=4000, desc="IQL-Transformer-converge-pretrain")

        _finetune_iql_vectorized(
            env, actor, iql, buf,
            n_env_steps=20000,
            warmup_steps=2000,
            desc="IQL-Transformer-converge-finetune",
        )

        eval_env = _make_env(num_envs=self.NUM_ENVS)
        eval_returns = _evaluate_vectorized(eval_env, actor, n_episodes=10)
        avg_return = np.mean(eval_returns)
        env.close()
        eval_env.close()

        assert avg_return > -400, (
            f"IQL Transformer on Pendulum-v1 did not converge: "
            f"avg return = {avg_return:.1f} (expected > -400, random ≈ -1400)"
        )
