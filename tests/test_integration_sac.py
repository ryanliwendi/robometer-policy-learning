"""
Integration tests for SAC on continuous control environments.

Verifies both that the code runs correctly (shape tests, smoke tests) and that
SAC actually converges to good policies (convergence tests).

Environments:
  - Pendulum-v1: Dense reward, 1-D continuous action, 3-D obs
  - LunarLander-v3 (continuous): Dense reward, 2-D continuous action, 8-D obs

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
from robometer_policy_learning.algorithms.sac import SAC, SACConfig
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
FEATURIZER_DIM = 64


def _make_env(env_id="Pendulum-v1", num_envs=1, **env_kwargs):
    """Vectorised env with dict-obs wrapper (``{'state': ...}``)."""

    def _factory():
        return FlatToDictObsWrapper(gym.make(env_id, **env_kwargs))

    return gym.vector.SyncVectorEnv([_factory for _ in range(num_envs)])


def _obs_and_action_spaces(env):
    obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
    act_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
    return obs_space, act_space


def _featurizer_for(obs_space):
    return {key: [FEATURIZER_DIM] for key in obs_space.spaces}


def _make_mlp_actor_critic(obs_space, act_space, hidden_dims=(128, 128)):
    featurizer = _featurizer_for(obs_space)
    actor_cfg = MLPActorConfig(
        observation_space=obs_space,
        action_space=act_space,
        featurizer=featurizer,
        hidden_dims=hidden_dims,
        activation="relu",
        use_tanh_output=True,
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
    return actor, critic


def _make_transformer_actor_critic(obs_space, act_space, hidden_dims=(128, 128), chunk_size=4):
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
    return actor, critic


def _make_sac(
    actor,
    critic,
    buffer,
    env,
    batch_size=256,
    learning_starts=0,
    lr=3e-4,
    num_critic_updates_per_actor_update=1,
    gamma=0.99,
    ent_coef="auto",
    target_entropy="auto",
):
    cfg = SACConfig(
        env=env,
        actor=actor,
        critic=critic,
        buffer=buffer,
        logger=None,
        batch_size=batch_size,
        learning_starts=learning_starts,
        gamma=gamma,
        tau=0.01,
        num_critics=2,
        n_critics_to_sample=2,
        num_critic_updates_per_actor_update=num_critic_updates_per_actor_update,
        num_updates_per_train_step=1,
        actor_optimizer_lr=lr,
        critic_optimizer_lr=lr,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        train_actor_with_kl_divergence=False,
    )
    return cfg.create()


def _add_transition(buffer, obs, action, reward, next_obs, terminated, truncated, ep_id, step):
    """Add a single transition from a vectorised env (extracts env 0)."""
    buffer.add(
        obs={k: v[0] for k, v in obs.items()},
        action=action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action,
        reward=float(reward[0]) if isinstance(reward, np.ndarray) else float(reward),
        next_obs={k: v[0] for k, v in next_obs.items()},
        done=bool(terminated[0]) if isinstance(terminated, np.ndarray) else bool(terminated),
        truncated=bool(truncated[0]) if isinstance(truncated, np.ndarray) else bool(truncated),
        episode_id=ep_id,
        step_in_episode=step,
    )


def _fill_buffer_random(env, buffer, n_steps):
    """Fill buffer with random actions (for quick smoke tests)."""
    obs, _ = env.reset()
    ep_id, step = 0, 0
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated[0] if isinstance(terminated, np.ndarray) else terminated
        trunc = truncated[0] if isinstance(truncated, np.ndarray) else truncated

        _add_transition(buffer, obs, action, reward, next_obs, terminated, truncated, ep_id, step)

        obs = next_obs
        step += 1
        if done or trunc:
            obs, _ = env.reset()
            ep_id += 1
            step = 0


def _get_action_chunk(actor, obs, deterministic=False):
    """Query actor and return a list of per-step action arrays.

    MLP actors return a single action → list of length 1.
    Transformer actors return (batch, chunk_size, act_dim) → list of length chunk_size.
    """
    obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
    action, _ = actor.act(obs_t, deterministic=deterministic)
    action_np = action.cpu().numpy()
    if action_np.ndim == 3:
        return [action_np[:, t, :] for t in range(action_np.shape[1])]
    return [action_np]


def _train_sac(env, actor, sac, buffer, n_steps, warmup_steps=1000, desc="SAC"):
    """Train SAC: random warmup then interleaved collection + gradient steps.

    For chunked (Transformer) actors the full action chunk is executed in the
    environment before the actor is queried again.  The chunk is discarded
    early if the episode terminates mid-chunk.
    """
    obs, _ = env.reset()
    ep_id, step = 0, 0
    ep_reward = 0.0
    last_ep_return = float("nan")
    action_queue: list = []

    pbar = tqdm(range(n_steps), desc=desc, unit="step", leave=True)
    for global_step in pbar:
        if global_step < warmup_steps:
            action_np = env.action_space.sample()
        else:
            if not action_queue:
                action_queue = _get_action_chunk(actor, obs)
            action_np = action_queue.pop(0)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated[0] if isinstance(terminated, np.ndarray) else terminated
        trunc = truncated[0] if isinstance(truncated, np.ndarray) else truncated
        r = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
        ep_reward += r

        _add_transition(buffer, obs, action_np, reward, next_obs, terminated, truncated, ep_id, step)

        obs = next_obs
        step += 1
        if done or trunc:
            last_ep_return = ep_reward
            pbar.set_postfix(ep=ep_id, ret=f"{last_ep_return:.0f}", buf=len(buffer))
            obs, _ = env.reset()
            ep_id += 1
            step = 0
            ep_reward = 0.0
            action_queue.clear()

        if global_step >= warmup_steps:
            sac.train_step()


def _evaluate(env, actor, n_episodes=5):
    """Run deterministic rollouts and return list of episode returns.

    Executes the full action chunk for Transformer actors before re-querying.
    """
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        action_queue: list = []
        while not done:
            if not action_queue:
                action_queue = _get_action_chunk(actor, obs, deterministic=True)
            action_np = action_queue.pop(0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            d = terminated[0] if isinstance(terminated, np.ndarray) else terminated
            t = truncated[0] if isinstance(truncated, np.ndarray) else truncated
            done = bool(d) or bool(t)
            if done:
                action_queue.clear()
        returns.append(total_reward)
    return returns


# ---------------------------------------------------------------------------
# Multi-env helpers
# ---------------------------------------------------------------------------

def _train_sac_vectorized(
    env, actor, sac, buffer, n_env_steps, warmup_steps=2000, desc="SAC",
):
    """Train SAC using all parallel envs in ``env``.

    Each ``env.step()`` collects ``num_envs`` transitions.  ``n_env_steps`` is
    the *total* number of transitions to collect (across all envs).

    For chunked (Transformer) actors the full action chunk is executed across
    all envs before re-querying.  If an env auto-resets mid-chunk the
    remaining chunk actions still execute (harmless early-episode noise).
    """
    num_envs = env.num_envs
    obs, _ = env.reset()

    ep_ids = list(range(num_envs))
    next_ep_id = num_envs
    steps = [0] * num_envs
    ep_rewards = [0.0] * num_envs
    last_ep_return = float("nan")
    completed_eps = 0

    action_chunk: list = []
    total_transitions = 0

    pbar = tqdm(total=n_env_steps, desc=desc, unit="trans", leave=True)

    while total_transitions < n_env_steps:
        if total_transitions < warmup_steps:
            actions = env.action_space.sample()
        else:
            if not action_chunk:
                obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
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

        total_transitions += num_envs
        pbar.update(num_envs)
        pbar.set_postfix(ep=completed_eps, ret=f"{last_ep_return:.0f}", buf=len(buffer))

        obs = next_obs

        if total_transitions >= warmup_steps:
            sac.train_step()

    pbar.close()


def _evaluate_vectorized(env, actor, n_episodes=10):
    """Evaluate across all parallel envs, collecting ``n_episodes`` returns."""
    num_envs = env.num_envs
    obs, _ = env.reset()
    ep_rewards = [0.0] * num_envs
    returns: list[float] = []
    action_chunk: list = []

    while len(returns) < n_episodes:
        if not action_chunk:
            obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
            raw, _ = actor.act(obs_t, deterministic=True)
            raw_np = raw.cpu().numpy()
            if raw_np.ndim == 3:
                action_chunk = [raw_np[:, t, :] for t in range(raw_np.shape[1])]
            else:
                action_chunk = [raw_np]
        actions = action_chunk.pop(0)

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
# Pendulum — MLP
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPendulumMLP:
    """MLP actor/critic + SAC on Pendulum-v1."""

    def test_actor_forward(self):
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _ = _make_mlp_actor_critic(obs_space, act_space)
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        action, _ = actor.act(obs_t)
        assert action.shape[-1] == act_space.shape[0]
        env.close()

    def test_critic_forward(self):
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        _, critic = _make_mlp_actor_critic(obs_space, act_space)
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        action_t = torch.randn(1, act_space.shape[0])
        with torch.no_grad():
            q_vals = critic(obs_t, action_t)
        assert q_vals.shape[0] == 1
        env.close()

    def test_sac_train_steps(self):
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic = _make_mlp_actor_critic(obs_space, act_space)
        buf = ReplayBuffer(capacity=1000, sampler=RandomSampler())
        _fill_buffer_random(env, buf, n_steps=200)
        sac = _make_sac(actor, critic, buf, env, batch_size=32, learning_starts=0)
        for _ in range(5):
            metrics = sac.train_step()
        assert "critic_loss" in metrics
        env.close()

    def test_sample_produces_valid_batch(self):
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _ = _make_mlp_actor_critic(obs_space, act_space)
        buf = ReplayBuffer(capacity=500, sampler=RandomSampler())
        _fill_buffer_random(env, buf, n_steps=100)
        batch = buf.sample(batch_size=16, device=DEVICE)
        assert batch["obs"]["state"].shape == (16, obs_space["state"].shape[0])
        assert batch["action"].shape[0] == 16
        env.close()

    def test_sac_converges(self):
        """Train SAC on Pendulum-v1 and verify it learns a good policy.

        Random policy ≈ -1200 to -1600 per episode (200 steps).
        Well-trained  ≈ -200  to  -300 per episode.
        Threshold     : -500  (clearly above random).
        """
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic = _make_mlp_actor_critic(obs_space, act_space)
        buf = ReplayBuffer(capacity=50000, sampler=RandomSampler())
        sac = _make_sac(
            actor, critic, buf, env,
            batch_size=64,
            lr=3e-4,
            num_critic_updates_per_actor_update=2,
        )

        _train_sac(env, actor, sac, buf, n_steps=4000, warmup_steps=1000, desc="Pendulum-MLP")

        eval_returns = _evaluate(env, actor, n_episodes=10)
        avg_return = np.mean(eval_returns)
        env.close()

        assert avg_return > -300, (
            f"SAC on Pendulum-v1 did not converge: avg return = {avg_return:.1f} "
            f"(expected > -300, random ≈ -1400)"
        )


# ---------------------------------------------------------------------------
# Pendulum — Transformer
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPendulumTransformer:
    """Transformer actor/critic + SAC on Pendulum-v1."""

    CHUNK_SIZE = 4

    def test_actor_forward(self):
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _ = _make_transformer_actor_critic(obs_space, act_space, chunk_size=self.CHUNK_SIZE)
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        action, _ = actor.act(obs_t)
        # Transformer actor outputs (batch, chunk_size, action_dim)
        assert action.shape[1] == self.CHUNK_SIZE
        assert action.shape[2] == act_space.shape[0]
        env.close()

    def test_sac_train_steps_with_chunked_sampler(self):
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic = _make_transformer_actor_critic(obs_space, act_space, chunk_size=self.CHUNK_SIZE)
        sampler = ChunkedSequentialSampler(chunk_size=self.CHUNK_SIZE, gamma=0.99)
        buf = ReplayBuffer(capacity=2000, sampler=sampler)
        _fill_buffer_random(env, buf, n_steps=500)
        sac = _make_sac(actor, critic, buf, env, batch_size=16, learning_starts=0)
        for _ in range(3):
            metrics = sac.train_step()
        assert "critic_loss" in metrics
        env.close()

    def test_chunked_sample_produces_valid_batch(self):
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _ = _make_transformer_actor_critic(obs_space, act_space, chunk_size=self.CHUNK_SIZE)
        sampler = ChunkedSequentialSampler(chunk_size=self.CHUNK_SIZE, gamma=0.99)
        buf = ReplayBuffer(capacity=2000, sampler=sampler)
        _fill_buffer_random(env, buf, n_steps=500)
        batch = buf.sample(batch_size=8, device=DEVICE)
        assert batch["action"].shape[0] == 8
        assert batch["action"].shape[1] == self.CHUNK_SIZE
        env.close()

    def test_sac_converges(self):
        """Train Transformer SAC on Pendulum-v1 and verify it learns.

        Uses ChunkedSequentialSampler for training; data collection takes the
        last action from the chunk.  Threshold is more lenient than MLP since
        transformers are slower to converge on small-scale RL.
        """
        env = _make_env("Pendulum-v1")
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic = _make_transformer_actor_critic(
            obs_space, act_space, chunk_size=self.CHUNK_SIZE,
        )
        sampler = ChunkedSequentialSampler(chunk_size=self.CHUNK_SIZE, gamma=0.99)
        buf = ReplayBuffer(capacity=50000, sampler=sampler)
        sac = _make_sac(
            actor, critic, buf, env,
            batch_size=64,
            lr=3e-4,
            num_critic_updates_per_actor_update=2,
        )

        _train_sac(env, actor, sac, buf, n_steps=4000, warmup_steps=1000,
                    desc="Pendulum-Transformer")

        eval_returns = _evaluate(env, actor, n_episodes=10)
        avg_return = np.mean(eval_returns)
        env.close()

        assert avg_return > -300, (
            f"Transformer SAC on Pendulum-v1 did not converge: avg return = {avg_return:.1f} "
            f"(expected > -300, random ≈ -1400)"
        )


# ---------------------------------------------------------------------------
# LunarLander (continuous) — MLP  (parallel envs)
# ---------------------------------------------------------------------------

LUNAR_ENV_KWARGS = {"continuous": True}


@pytest.mark.integration
class TestLunarLanderMLP:
    """MLP actor/critic + SAC on LunarLander-v3 (continuous) with parallel envs."""

    NUM_ENVS = 4


    def test_actor_forward(self):
        env = _make_env("LunarLander-v3", **LUNAR_ENV_KWARGS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _ = _make_mlp_actor_critic(obs_space, act_space)
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        action, _ = actor.act(obs_t)
        assert action.shape[-1] == act_space.shape[0]
        env.close()

    def test_sac_converges(self):
        """Train SAC on LunarLander-v3 (continuous) using parallel envs.

        8D obs, 2D continuous action, dense reward, ~100 steps/episode.
        Random policy ≈ -200.  Solved ≈ 200+.
        Threshold     : 0   (clearly above random, learning is happening).
        """
        env = _make_env("LunarLander-v3", num_envs=self.NUM_ENVS, **LUNAR_ENV_KWARGS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic = _make_mlp_actor_critic(obs_space, act_space, hidden_dims=(300, 200))
        buf = ReplayBuffer(capacity=40000, sampler=RandomSampler())
        sac = _make_sac(
            actor, critic, buf, env,
            batch_size=128,
            lr=3e-4,
            num_critic_updates_per_actor_update=5,
            gamma=0.99,
        )

        _train_sac_vectorized(
            env, actor, sac, buf,
            n_env_steps=40000, warmup_steps=10000,
            desc="LunarLander-MLP",
        )

        eval_env = _make_env("LunarLander-v3", num_envs=self.NUM_ENVS, **LUNAR_ENV_KWARGS)
        eval_returns = _evaluate_vectorized(eval_env, actor, n_episodes=10)
        avg_return = np.mean(eval_returns)
        env.close()
        eval_env.close()

        assert avg_return > 0, (
            f"SAC on LunarLander-v3 did not converge: avg return = {avg_return:.1f} "
            f"(expected > 0, random ≈ -200)"
        )


# ---------------------------------------------------------------------------
# LunarLander (continuous) — Transformer  (parallel envs)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLunarLanderTransformer:
    """Transformer actor/critic + SAC on LunarLander-v3 (continuous) with parallel envs."""

    NUM_ENVS = 4
    CHUNK_SIZE = 2

    def test_actor_forward(self):
        env = _make_env("LunarLander-v3", **LUNAR_ENV_KWARGS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, _ = _make_transformer_actor_critic(obs_space, act_space, chunk_size=self.CHUNK_SIZE)
        obs, _ = env.reset()
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        action, _ = actor.act(obs_t)
        assert action.shape[1] == self.CHUNK_SIZE
        assert action.shape[2] == act_space.shape[0]
        env.close()

    def test_sac_converges(self):
        """Train Transformer SAC on LunarLander-v3 (continuous) using parallel envs.

        Uses ChunkedSequentialSampler; full action chunks are executed in all
        envs before re-querying the actor.
        """
        env = _make_env("LunarLander-v3", num_envs=self.NUM_ENVS, **LUNAR_ENV_KWARGS)
        obs_space, act_space = _obs_and_action_spaces(env)
        actor, critic = _make_transformer_actor_critic(
            obs_space, act_space, chunk_size=self.CHUNK_SIZE,
            hidden_dims=(300, 200),
        )
        sampler = ChunkedSequentialSampler(chunk_size=self.CHUNK_SIZE, gamma=0.99)
        buf = ReplayBuffer(capacity=40000, sampler=sampler)
        sac = _make_sac(
            actor, critic, buf, env,
            batch_size=128,
            lr=3e-4,
            num_critic_updates_per_actor_update=5,
            gamma=0.99,
        )

        _train_sac_vectorized(
            env, actor, sac, buf,
            n_env_steps=40000, warmup_steps=10000,
            desc="LunarLander-Transformer",
        )

        eval_env = _make_env("LunarLander-v3", num_envs=self.NUM_ENVS, **LUNAR_ENV_KWARGS)
        eval_returns = _evaluate_vectorized(eval_env, actor, n_episodes=10)
        avg_return = np.mean(eval_returns)
        env.close()
        eval_env.close()

        assert avg_return > 0, (
            f"Transformer SAC on LunarLander-v3 did not converge: "
            f"avg return = {avg_return:.1f} (expected > 0, random ≈ -200)"
        )
