"""Shared fixtures for the rfm_rl test suite."""

import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dict_obs(dim: int = 4, batch: bool = False):
    """Return a dict observation suitable for buffers that expect dict obs."""
    obs = {"state": np.random.randn(dim).astype(np.float32)}
    return obs


def make_transition_kwargs(
    obs_dim: int = 4, act_dim: int = 2, episode_id=0, step=0, done=False, truncated=False,
):
    """Return keyword arguments for ``buffer.add(...)``."""
    return dict(
        obs=make_dict_obs(obs_dim),
        action=np.random.randn(act_dim).astype(np.float32),
        reward=float(np.random.randn()),
        next_obs=make_dict_obs(obs_dim),
        done=done,
        truncated=truncated,
        episode_id=episode_id,
        step_in_episode=step,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obs_dim():
    return 4


@pytest.fixture
def act_dim():
    return 2


@pytest.fixture
def small_capacity():
    return 64


@pytest.fixture
def replay_buffer(small_capacity):
    from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
    return ReplayBuffer(capacity=small_capacity)


@pytest.fixture
def filled_replay_buffer(replay_buffer, small_capacity):
    """A ReplayBuffer already filled to capacity with sequential episodes."""
    steps_per_episode = 10
    ep = 0
    for i in range(small_capacity):
        step = i % steps_per_episode
        if step == 0 and i > 0:
            ep += 1
        done = step == steps_per_episode - 1
        replay_buffer.add(**make_transition_kwargs(episode_id=ep, step=step, done=done))
    return replay_buffer
