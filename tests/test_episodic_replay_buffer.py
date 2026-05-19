"""Unit tests for EpisodicReplayBuffer."""

import numpy as np
import pytest

from robometer_policy_learning.buffers.episodic_replay_buffer import EpisodicReplayBuffer
from tests.conftest import make_transition_kwargs


class TestEpisodicReplayBuffer:
    def test_init(self):
        buf = EpisodicReplayBuffer(capacity=100)
        assert len(buf) == 0
        assert buf.is_empty()

    def test_add_transitions(self):
        buf = EpisodicReplayBuffer(capacity=100)
        for i in range(10):
            done = i == 9
            buf.add(**make_transition_kwargs(episode_id=0, step=i, done=done))
        assert len(buf) == 10

    def test_capacity_enforced(self):
        cap = 20
        buf = EpisodicReplayBuffer(capacity=cap)
        for i in range(cap + 10):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        assert len(buf) <= cap

    def test_get_all_transitions_returns_copies(self):
        buf = EpisodicReplayBuffer(capacity=100)
        for i in range(5):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        t1 = buf.get_all_transitions()
        t2 = buf.get_all_transitions()
        # Should be separate list objects (defensive copy)
        assert t1 is not t2

    def test_clear(self):
        buf = EpisodicReplayBuffer(capacity=100)
        for i in range(10):
            buf.add(**make_transition_kwargs(episode_id=0, step=i))
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty()

    def test_sample(self):
        buf = EpisodicReplayBuffer(capacity=100)
        for ep in range(3):
            for step in range(10):
                done = step == 9
                buf.add(**make_transition_kwargs(episode_id=ep, step=step, done=done))
        batch = buf.sample(batch_size=8, device="cpu")
        assert batch["reward"].shape[0] == 8

    def test_episode_boundaries(self):
        buf = EpisodicReplayBuffer(capacity=200)
        for ep in range(3):
            for step in range(10):
                done = step == 9
                buf.add(**make_transition_kwargs(episode_id=ep, step=step, done=done))
        boundaries = buf.get_episode_boundaries()
        assert len(boundaries) == 3
        for ep_id, (start, end) in boundaries.items():
            assert end - start + 1 == 10
