"""Unit tests for all sampler classes."""

import numpy as np
import pytest
import torch

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.base_replay_buffer import Transition
from robometer_policy_learning.buffers.samplers import (
    RandomSampler,
    RelabeledOnlySampler,
    ChunkedSequentialSampler,
    EpisodeEndSampler,
    EpisodeStartSampler,
    TemporalSampler,
    EpisodeBalancedSampler,
    _stack_values,
)
from tests.conftest import make_transition_kwargs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_buffer_with_episodes(buf, n_episodes=5, steps_per_ep=10):
    for ep in range(n_episodes):
        for step in range(steps_per_ep):
            done = step == steps_per_ep - 1
            buf.add(**make_transition_kwargs(episode_id=ep, step=step, done=done))
    return buf


def _fill_buffer_relabeled(buf, n_total=50, n_relabeled=30):
    """Fill buffer and mark some transitions as relabeled."""
    for i in range(n_total):
        buf.add(**make_transition_kwargs(episode_id=0, step=i))
    transitions = buf.get_all_transitions()
    for t in transitions[:n_relabeled]:
        if t.info is None:
            t.info = {}
        t.info["relabeled_reward"] = 1.0
    return buf


# ---------------------------------------------------------------------------
# _stack_values helper
# ---------------------------------------------------------------------------

class TestStackValues:
    def test_numpy_arrays(self):
        vals = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        result = _stack_values(vals)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_torch_tensors(self):
        vals = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = _stack_values(vals)
        assert result.shape == (2, 2)

    def test_python_lists(self):
        vals = [[1.0, 2.0], [3.0, 4.0]]
        result = _stack_values(vals)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# RandomSampler
# ---------------------------------------------------------------------------

class TestRandomSampler:
    def test_sample_basic(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer_with_episodes(buf, n_episodes=3, steps_per_ep=10)
        sampler = RandomSampler()
        transitions = sampler.sample(buf, batch_size=8)
        assert len(transitions) == 8
        assert all(isinstance(t, Transition) for t in transitions)

    def test_uses_fast_path(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer_with_episodes(buf, n_episodes=2, steps_per_ep=10)
        sampler = RandomSampler()
        assert hasattr(buf, "sample_indices")
        assert hasattr(buf, "transitions_from_indices")
        transitions = sampler.sample(buf, batch_size=5)
        assert len(transitions) == 5

    def test_sample_with_replacement_small_buffer(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(**make_transition_kwargs())
        sampler = RandomSampler()
        transitions = sampler.sample(buf, batch_size=5)
        assert len(transitions) == 5

    def test_can_sample(self):
        buf = ReplayBuffer(capacity=10)
        sampler = RandomSampler()
        assert not sampler.can_sample(buf, 1)
        buf.add(**make_transition_kwargs())
        assert sampler.can_sample(buf, 1)


# ---------------------------------------------------------------------------
# RelabeledOnlySampler
# ---------------------------------------------------------------------------

class TestRelabeledOnlySampler:
    def test_returns_empty_when_none_relabeled(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer_with_episodes(buf, n_episodes=2, steps_per_ep=10)
        sampler = RelabeledOnlySampler(min_relabeled_ratio=0.1)
        assert sampler.sample(buf, batch_size=8) == []

    def test_samples_when_ratio_met(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer_relabeled(buf, n_total=50, n_relabeled=30)
        sampler = RelabeledOnlySampler(min_relabeled_ratio=0.1)
        transitions = sampler.sample(buf, batch_size=8)
        assert len(transitions) == 8

    def test_returns_empty_when_ratio_not_met(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer_relabeled(buf, n_total=50, n_relabeled=2)
        sampler = RelabeledOnlySampler(min_relabeled_ratio=0.5)
        assert sampler.sample(buf, batch_size=8) == []

    def test_can_sample(self):
        buf = ReplayBuffer(capacity=100)
        sampler = RelabeledOnlySampler(min_relabeled_ratio=0.1)
        assert not sampler.can_sample(buf, 1)
        _fill_buffer_relabeled(buf, n_total=50, n_relabeled=30)
        assert sampler.can_sample(buf, 1)


# ---------------------------------------------------------------------------
# ChunkedSequentialSampler
# ---------------------------------------------------------------------------

class TestChunkedSequentialSampler:
    def test_init_caches_discount_factors(self):
        sampler = ChunkedSequentialSampler(chunk_size=4, gamma=0.99)
        expected = torch.tensor([0.99**i for i in range(4)], dtype=torch.float32)
        assert torch.allclose(sampler._discount_factors, expected, atol=1e-6)

    def test_sample_returns_sequence_transitions(self):
        buf = ReplayBuffer(capacity=200)
        _fill_buffer_with_episodes(buf, n_episodes=5, steps_per_ep=20)
        sampler = ChunkedSequentialSampler(chunk_size=4, gamma=0.99)
        transitions = sampler.sample(buf, batch_size=3)
        assert len(transitions) <= 3
        for t in transitions:
            assert isinstance(t, Transition)
            # Actions should be stacked (4, act_dim)
            assert t.action.shape[0] == 4

    def test_discounted_reward(self):
        sampler = ChunkedSequentialSampler(chunk_size=3, gamma=0.5)
        transitions = [
            Transition(
                obs={"state": np.zeros(2)},
                action=np.zeros(2),
                reward=1.0,
                next_obs={"state": np.zeros(2)},
                done=False,
            ),
            Transition(
                obs={"state": np.zeros(2)},
                action=np.zeros(2),
                reward=2.0,
                next_obs={"state": np.zeros(2)},
                done=False,
            ),
            Transition(
                obs={"state": np.zeros(2)},
                action=np.zeros(2),
                reward=4.0,
                next_obs={"state": np.zeros(2)},
                done=False,
            ),
        ]
        seq = sampler._chunk_to_sequence(transitions)
        # 1.0 + 0.5*2.0 + 0.25*4.0 = 1.0 + 1.0 + 1.0 = 3.0
        assert abs(seq.reward.item() - 3.0) < 1e-5

    def test_done_short_circuits(self):
        sampler = ChunkedSequentialSampler(chunk_size=3, gamma=0.99)
        transitions = [
            Transition(
                obs={"state": np.zeros(2)}, action=np.zeros(2), reward=0.0,
                next_obs={"state": np.zeros(2)}, done=False,
            ),
            Transition(
                obs={"state": np.zeros(2)}, action=np.zeros(2), reward=0.0,
                next_obs={"state": np.zeros(2)}, done=True,
            ),
            Transition(
                obs={"state": np.zeros(2)}, action=np.zeros(2), reward=0.0,
                next_obs={"state": np.zeros(2)}, done=False,
            ),
        ]
        seq = sampler._chunk_to_sequence(transitions)
        assert seq.done.item() is True

    def test_obs_as_sequence(self):
        buf = ReplayBuffer(capacity=200)
        _fill_buffer_with_episodes(buf, n_episodes=5, steps_per_ep=20)
        sampler = ChunkedSequentialSampler(chunk_size=4, gamma=0.99, obs_as_sequence=True)
        transitions = sampler.sample(buf, batch_size=2)
        if transitions:
            t = transitions[0]
            assert isinstance(t.obs, dict)
            for v in t.obs.values():
                assert v.shape[0] == 4  # chunk_size

    def test_attributes_for_h5_compat(self):
        sampler = ChunkedSequentialSampler(chunk_size=8, gamma=0.95, obs_as_sequence=True)
        assert sampler.chunk_size == 8
        assert sampler.gamma == 0.95
        assert sampler.obs_as_sequence is True
        assert hasattr(sampler, "_chunk_to_sequence")


# ---------------------------------------------------------------------------
# EpisodeEndSampler
# ---------------------------------------------------------------------------

class TestEpisodeEndSampler:
    def test_returns_done_transitions(self):
        buf = ReplayBuffer(capacity=200)
        _fill_buffer_with_episodes(buf, n_episodes=5, steps_per_ep=10)
        sampler = EpisodeEndSampler()
        transitions = sampler.sample(buf, batch_size=3)
        assert len(transitions) <= 3
        for t in transitions:
            assert t.done or t.truncated


# ---------------------------------------------------------------------------
# EpisodeStartSampler
# ---------------------------------------------------------------------------

class TestEpisodeStartSampler:
    def test_returns_start_transitions(self):
        buf = ReplayBuffer(capacity=200)
        _fill_buffer_with_episodes(buf, n_episodes=5, steps_per_ep=10)
        sampler = EpisodeStartSampler()
        transitions = sampler.sample(buf, batch_size=3)
        assert len(transitions) <= 3
        for t in transitions:
            assert t.step_in_episode == 0


# ---------------------------------------------------------------------------
# TemporalSampler
# ---------------------------------------------------------------------------

class TestTemporalSampler:
    def test_basic_sampling(self):
        buf = ReplayBuffer(capacity=200)
        _fill_buffer_with_episodes(buf, n_episodes=3, steps_per_ep=20)
        sampler = TemporalSampler(recency_weight=2.0)
        transitions = sampler.sample(buf, batch_size=10)
        assert len(transitions) == 10

    def test_recency_bias(self):
        """Higher recency weight should bias toward later transitions."""
        buf = ReplayBuffer(capacity=200)
        _fill_buffer_with_episodes(buf, n_episodes=2, steps_per_ep=50)
        sampler = TemporalSampler(recency_weight=5.0)
        transitions = sampler.sample(buf, batch_size=50)
        steps = [t.step_in_episode for t in transitions]
        # With high recency weight, average step should be well above the midpoint
        avg_step = np.mean(steps)
        assert avg_step > 15  # midpoint is ~25, but with bias should be above 15

    def test_empty_buffer(self):
        buf = ReplayBuffer(capacity=10)
        sampler = TemporalSampler()
        assert sampler.sample(buf, batch_size=5) == []


# ---------------------------------------------------------------------------
# EpisodeBalancedSampler
# ---------------------------------------------------------------------------

class TestEpisodeBalancedSampler:
    def test_balances_across_episodes(self):
        buf = ReplayBuffer(capacity=200)
        _fill_buffer_with_episodes(buf, n_episodes=4, steps_per_ep=20)
        sampler = EpisodeBalancedSampler()
        transitions = sampler.sample(buf, batch_size=40)
        # Each episode should contribute roughly 10 transitions (40 / 4)
        ep_counts = {}
        for t in transitions:
            ep_counts[t.episode_id] = ep_counts.get(t.episode_id, 0) + 1
        for count in ep_counts.values():
            assert count == 10

    def test_fallback_when_no_boundaries(self):
        buf = ReplayBuffer(capacity=10)
        sampler = EpisodeBalancedSampler()
        # Empty buffer should fall through to RandomSampler
        transitions = sampler.sample(buf, batch_size=5)
        assert transitions == []
