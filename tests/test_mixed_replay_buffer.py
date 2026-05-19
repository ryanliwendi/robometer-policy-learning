"""Unit tests for MixedReplayBuffer."""

import numpy as np
import pytest

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.mixed_replay_buffer import MixedReplayBuffer
from tests.conftest import make_transition_kwargs


class TestMixedReplayBuffer:
    def _make_filled_buffers(self, cap=100, n_each=30):
        buf1 = ReplayBuffer(capacity=cap)
        buf2 = ReplayBuffer(capacity=cap)
        for i in range(n_each):
            buf1.add(**make_transition_kwargs(episode_id="b1", step=i))
            buf2.add(**make_transition_kwargs(episode_id="b2", step=i))
        return buf1, buf2

    def test_init(self):
        buf1, buf2 = self._make_filled_buffers()
        mixed = MixedReplayBuffer(buffer_1=buf1, buffer_2=buf2, sample_ratio=0.5)
        assert len(mixed) == 60

    def test_get_all_transitions(self):
        buf1, buf2 = self._make_filled_buffers(n_each=10)
        mixed = MixedReplayBuffer(buffer_1=buf1, buffer_2=buf2, sample_ratio=0.5)
        all_t = mixed.get_all_transitions()
        assert len(all_t) == 20

    def test_add_routes_to_correct_buffer(self):
        buf1, buf2 = self._make_filled_buffers(n_each=5)
        mixed = MixedReplayBuffer(buffer_1=buf1, buffer_2=buf2, buffer_to_add_to=2)
        initial_b2_len = len(buf2)
        mixed.add(**make_transition_kwargs(episode_id="new", step=0))
        assert len(buf2) == initial_b2_len + 1

    def test_sample(self):
        buf1, buf2 = self._make_filled_buffers(n_each=30)
        mixed = MixedReplayBuffer(buffer_1=buf1, buffer_2=buf2, sample_ratio=0.5)
        batch = mixed.sample(batch_size=16, device="cpu")
        assert batch["reward"].shape[0] == 16

    def test_clear(self):
        buf1, buf2 = self._make_filled_buffers(n_each=10)
        mixed = MixedReplayBuffer(buffer_1=buf1, buffer_2=buf2)
        mixed.clear()
        assert len(mixed) == 0

    def test_sample_ratio(self):
        buf1 = ReplayBuffer(capacity=200)
        buf2 = ReplayBuffer(capacity=200)
        for i in range(100):
            buf1.add(**make_transition_kwargs(episode_id="b1", step=i))
            buf2.add(**make_transition_kwargs(episode_id="b2", step=i))
        # ratio=0.8 means 80% from buf1, 20% from buf2
        mixed = MixedReplayBuffer(buffer_1=buf1, buffer_2=buf2, sample_ratio=0.8)
        batch = mixed.sample(batch_size=100, device="cpu")
        assert batch["reward"].shape[0] == 100
