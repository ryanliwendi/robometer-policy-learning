"""Unit tests for the ring-buffer-based ReplayBuffer."""

import numpy as np
import pytest

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.base_replay_buffer import Transition
from tests.conftest import make_transition_kwargs, make_dict_obs


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestReplayBufferInit:
    def test_empty_buffer(self):
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0
        assert buf.is_empty()
        assert buf.size() == 0

    def test_capacity_stored(self):
        buf = ReplayBuffer(capacity=42)
        assert buf.capacity == 42


# ---------------------------------------------------------------------------
# Adding transitions
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_single(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(**make_transition_kwargs())
        assert len(buf) == 1
        assert not buf.is_empty()

    def test_add_fills_to_capacity(self, small_capacity):
        buf = ReplayBuffer(capacity=small_capacity)
        for i in range(small_capacity):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        assert len(buf) == small_capacity

    def test_add_past_capacity_stays_at_capacity(self, small_capacity):
        buf = ReplayBuffer(capacity=small_capacity)
        for i in range(small_capacity + 20):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        assert len(buf) == small_capacity

    def test_ring_buffer_overwrites_oldest(self):
        cap = 5
        buf = ReplayBuffer(capacity=cap)
        for i in range(cap + 3):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))

        transitions = buf.get_all_transitions()
        assert len(transitions) == cap
        # The oldest surviving step should be 3 (steps 0,1,2 evicted)
        assert transitions[0].step_in_episode == 3

    def test_ring_buffer_write_pos_wraps(self):
        cap = 4
        buf = ReplayBuffer(capacity=cap)
        for i in range(cap + 2):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        assert buf._write_pos == 2  # (cap + 2) % cap


# ---------------------------------------------------------------------------
# get_all_transitions ordering
# ---------------------------------------------------------------------------

class TestGetAllTransitions:
    def test_insertion_order_before_wrap(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        steps = [t.step_in_episode for t in buf.get_all_transitions()]
        assert steps == [0, 1, 2, 3, 4]

    def test_insertion_order_after_wrap(self):
        cap = 4
        buf = ReplayBuffer(capacity=cap)
        for i in range(7):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        steps = [t.step_in_episode for t in buf.get_all_transitions()]
        assert steps == [3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSample:
    def test_sample_returns_batch(self, filled_replay_buffer):
        batch = filled_replay_buffer.sample(batch_size=8, device="cpu")
        assert "obs" in batch
        assert "action" in batch
        assert batch["reward"].shape[0] == 8

    def test_sample_with_replacement_when_small(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(**make_transition_kwargs())
        batch = buf.sample(batch_size=4, device="cpu")
        assert batch["reward"].shape[0] == 4

    def test_sample_indices_range(self, small_capacity):
        buf = ReplayBuffer(capacity=small_capacity)
        for i in range(small_capacity):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        idxs = buf.sample_indices(batch_size=16)
        assert all(0 <= idx < small_capacity for idx in idxs)

    def test_transitions_from_indices(self, small_capacity):
        buf = ReplayBuffer(capacity=small_capacity)
        for i in range(small_capacity):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        transitions = buf.transitions_from_indices([0, 1, 2])
        assert len(transitions) == 3
        assert all(isinstance(t, Transition) for t in transitions)


# ---------------------------------------------------------------------------
# Transition index (update_reward / update_info)
# ---------------------------------------------------------------------------

class TestTransitionIndex:
    def test_update_reward(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(**make_transition_kwargs(episode_id="ep1", step=0))
        assert buf.update_reward("ep1", 0, 99.0)
        t = buf.get_all_transitions()[0]
        assert t.reward == 99.0

    def test_update_reward_after_eviction_returns_false(self):
        cap = 3
        buf = ReplayBuffer(capacity=cap)
        buf.add(**make_transition_kwargs(episode_id="ep0", step=0))
        for i in range(cap):
            buf.add(**make_transition_kwargs(episode_id="ep1", step=i))
        # ep0/step0 has been evicted
        assert not buf.update_reward("ep0", 0, 99.0)

    def test_update_info(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(**make_transition_kwargs(episode_id="ep1", step=0))
        assert buf.update_info("ep1", 0, {"relabeled_reward": 1.0})
        t = buf.get_all_transitions()[0]
        assert t.info["relabeled_reward"] == 1.0

    def test_index_cleaned_on_eviction(self):
        cap = 3
        buf = ReplayBuffer(capacity=cap)
        buf.add(**make_transition_kwargs(episode_id="evict_me", step=0))
        for i in range(cap):
            buf.add(**make_transition_kwargs(episode_id="keeper", step=i))
        assert ("evict_me", 0) not in buf._transition_index


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_buffer(self, filled_replay_buffer):
        filled_replay_buffer.clear()
        assert len(filled_replay_buffer) == 0
        assert filled_replay_buffer.is_empty()
        assert filled_replay_buffer._write_pos == 0
        assert len(filled_replay_buffer._transition_index) == 0
