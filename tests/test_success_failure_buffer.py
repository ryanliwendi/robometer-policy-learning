"""Unit tests for SuccessFailureReplayBuffer."""

import numpy as np
import pytest

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.success_failure_replay_buffer import SuccessFailureReplayBuffer
from tests.conftest import make_transition_kwargs


def _make_buffer(cap=100, sample_ratio=0.5):
    success = ReplayBuffer(capacity=cap)
    failure = ReplayBuffer(capacity=cap)
    return SuccessFailureReplayBuffer(
        success_buffer=success,
        failure_buffer=failure,
        sample_ratio=sample_ratio,
    )


def _add_episode(buf, episode_id, n_steps=10, done=True, truncated=False, **terminal_kwargs):
    """Add a full episode through the buffer's _add routing (via buf.add)."""
    for step in range(n_steps):
        is_last = step == n_steps - 1
        buf.add(
            **make_transition_kwargs(
                episode_id=episode_id,
                step=step,
                done=done and is_last,
                truncated=truncated and is_last,
            ),
            **(terminal_kwargs if is_last else {}),
        )


# ---------------------------------------------------------------------------
# Basic init / len / empty / clear
# ---------------------------------------------------------------------------

class TestBasics:
    def test_init(self):
        buf = _make_buffer()
        assert len(buf) == 0
        assert buf.is_empty()

    def test_invalid_sample_ratio(self):
        with pytest.raises(ValueError, match="sample_ratio"):
            _make_buffer(sample_ratio=1.5)
        with pytest.raises(ValueError, match="sample_ratio"):
            _make_buffer(sample_ratio=-0.1)

    def test_clear(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        _add_episode(buf, "f0", is_success=False)
        assert len(buf) > 0
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty()
        assert buf.stats["total_episodes"] == 0

    def test_clear_individual_buffers(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        _add_episode(buf, "f0", is_success=False)
        assert len(buf.success_buffer) == 10
        assert len(buf.failure_buffer) == 10

        buf.clear_buffer("success")
        assert len(buf.success_buffer) == 0
        assert len(buf.failure_buffer) == 10

        buf.clear_buffer("failure")
        assert len(buf.failure_buffer) == 0

    def test_clear_pending(self):
        buf = _make_buffer()
        # Add a partial episode (not done yet)
        buf.add(**make_transition_kwargs(episode_id="pending", step=0))
        assert len(buf._pending_episodes) == 1
        buf.clear_buffer("pending")
        assert len(buf._pending_episodes) == 0

    def test_clear_buffer_invalid_name(self):
        buf = _make_buffer()
        with pytest.raises(ValueError):
            buf.clear_buffer("invalid")


# ---------------------------------------------------------------------------
# Episode routing via _add
# ---------------------------------------------------------------------------

class TestEpisodeRouting:
    def test_routes_success_via_is_success_kwarg(self):
        buf = _make_buffer()
        _add_episode(buf, "ep0", is_success=True)
        assert len(buf.success_buffer) == 10
        assert len(buf.failure_buffer) == 0

    def test_routes_failure_via_is_success_kwarg(self):
        buf = _make_buffer()
        _add_episode(buf, "ep0", is_success=False)
        assert len(buf.success_buffer) == 0
        assert len(buf.failure_buffer) == 10

    def test_routes_success_via_success_kwarg(self):
        buf = _make_buffer()
        _add_episode(buf, "ep0", success=True)
        assert len(buf.success_buffer) == 10

    def test_defaults_to_failure_when_no_success_info(self):
        buf = _make_buffer()
        _add_episode(buf, "ep0")
        assert len(buf.success_buffer) == 0
        assert len(buf.failure_buffer) == 10

    def test_routes_truncated_episode(self):
        buf = _make_buffer()
        _add_episode(buf, "ep0", done=False, truncated=True, is_success=True)
        assert len(buf.success_buffer) == 10

    def test_pending_until_done(self):
        buf = _make_buffer()
        for step in range(5):
            buf.add(**make_transition_kwargs(episode_id="ep0", step=step, done=False))
        assert len(buf) == 0
        assert len(buf._pending_episodes) == 1

        buf.add(**make_transition_kwargs(episode_id="ep0", step=5, done=True))
        assert len(buf) == 6
        assert len(buf._pending_episodes) == 0

    def test_episode_id_required(self):
        buf = _make_buffer()
        with pytest.raises(ValueError, match="episode_id"):
            buf._add(
                obs={"state": np.zeros(4, dtype=np.float32)},
                action=np.zeros(2, dtype=np.float32),
                reward=0.0,
                next_obs={"state": np.zeros(4, dtype=np.float32)},
                done=True,
                truncated=False,
            )

    def test_multiple_episodes_routed_correctly(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        _add_episode(buf, "f0", is_success=False)
        _add_episode(buf, "s1", is_success=True)
        _add_episode(buf, "f1", is_success=False)
        _add_episode(buf, "s2", is_success=True)

        assert len(buf.success_buffer) == 30
        assert len(buf.failure_buffer) == 20

    def test_concurrent_pending_episodes(self):
        """Multiple episodes in flight at once (e.g. parallel envs)."""
        buf = _make_buffer()
        for step in range(5):
            buf.add(**make_transition_kwargs(episode_id="env0", step=step, done=False))
            buf.add(**make_transition_kwargs(episode_id="env1", step=step, done=False))

        assert len(buf._pending_episodes) == 2
        assert len(buf) == 0

        buf.add(**make_transition_kwargs(episode_id="env0", step=5, done=True), is_success=True)
        assert len(buf.success_buffer) == 6
        assert len(buf._pending_episodes) == 1

        buf.add(**make_transition_kwargs(episode_id="env1", step=5, done=True), is_success=False)
        assert len(buf.failure_buffer) == 6
        assert len(buf._pending_episodes) == 0


# ---------------------------------------------------------------------------
# add_episode_info
# ---------------------------------------------------------------------------

class TestAddEpisodeInfo:
    def test_info_determines_success(self):
        buf = _make_buffer()
        buf.add_episode_info("ep0", {"is_success": True})
        _add_episode(buf, "ep0")
        assert len(buf.success_buffer) == 10

    def test_info_determines_failure(self):
        buf = _make_buffer()
        buf.add_episode_info("ep0", {"is_success": False})
        _add_episode(buf, "ep0")
        assert len(buf.failure_buffer) == 10

    def test_terminal_kwargs_override_stored_info(self):
        """is_success in the terminal transition kwargs takes priority."""
        buf = _make_buffer()
        buf.add_episode_info("ep0", {"is_success": False})
        _add_episode(buf, "ep0", is_success=True)
        assert len(buf.success_buffer) == 10
        assert len(buf.failure_buffer) == 0

    def test_success_key_also_works(self):
        buf = _make_buffer()
        buf.add_episode_info("ep0", {"success": True})
        _add_episode(buf, "ep0")
        assert len(buf.success_buffer) == 10


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_stats_update_on_finalize(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        _add_episode(buf, "f0", is_success=False)
        _add_episode(buf, "s1", is_success=True)

        assert buf.stats["total_episodes"] == 3
        assert buf.stats["successful_episodes"] == 2
        assert buf.stats["failed_episodes"] == 1
        assert buf.stats["pending_episodes"] == 0

    def test_get_statistics(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        _add_episode(buf, "f0", is_success=False)
        stats = buf.get_statistics()

        assert stats["total_episodes"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["success_buffer_size"] == 10
        assert stats["failure_buffer_size"] == 10
        assert stats["total_size"] == 20

    def test_get_buffer_sizes(self):
        buf = _make_buffer()
        buf.add(**make_transition_kwargs(episode_id="pending", step=0))
        _add_episode(buf, "s0", is_success=True)

        sizes = buf.get_buffer_sizes()
        assert sizes["success_buffer_size"] == 10
        assert sizes["failure_buffer_size"] == 0
        assert sizes["pending_episodes"] == 1
        assert sizes["pending_transitions"] == 1

    def test_stats_empty(self):
        buf = _make_buffer()
        stats = buf.get_statistics()
        assert stats["success_rate"] == 0.0
        assert stats["total_episodes"] == 0


# ---------------------------------------------------------------------------
# Episode integrity verification
# ---------------------------------------------------------------------------

class TestVerifyIntegrity:
    def test_valid_no_overlap(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        _add_episode(buf, "f0", is_success=False)
        verification = buf.verify_episode_integrity()
        assert verification["is_valid"]
        assert verification["num_success_episodes"] == 1
        assert verification["num_failure_episodes"] == 1

    def test_counts_transitions_per_episode(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", n_steps=7, is_success=True)
        _add_episode(buf, "f0", n_steps=12, is_success=False)
        verification = buf.verify_episode_integrity()
        assert verification["success_episodes"]["s0"] == 7
        assert verification["failure_episodes"]["f0"] == 12
        assert verification["total_success_transitions"] == 7
        assert verification["total_failure_transitions"] == 12


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def _populated_buffer(self, n_success_eps=3, n_failure_eps=3, steps=10):
        buf = _make_buffer(cap=500, sample_ratio=0.5)
        for i in range(n_success_eps):
            _add_episode(buf, f"s{i}", n_steps=steps, is_success=True)
        for i in range(n_failure_eps):
            _add_episode(buf, f"f{i}", n_steps=steps, is_success=False)
        return buf

    def test_sample_with_both_populated(self):
        buf = self._populated_buffer()
        batch = buf.sample(batch_size=16, device="cpu")
        assert batch["reward"].shape[0] == 16

    def test_sample_only_success(self):
        buf = _make_buffer()
        for i in range(3):
            _add_episode(buf, f"s{i}", is_success=True)
        batch = buf.sample(batch_size=8, device="cpu")
        assert batch["reward"].shape[0] == 8

    def test_sample_only_failure(self):
        buf = _make_buffer()
        for i in range(3):
            _add_episode(buf, f"f{i}", is_success=False)
        batch = buf.sample(batch_size=8, device="cpu")
        assert batch["reward"].shape[0] == 8

    def test_sample_empty_returns_empty_dict(self):
        buf = _make_buffer()
        assert buf.sample(batch_size=8, device="cpu") == {}

    def test_sample_ratio_respected(self):
        buf = _make_buffer(cap=500, sample_ratio=0.75)
        for i in range(5):
            _add_episode(buf, f"s{i}", is_success=True)
        for i in range(5):
            _add_episode(buf, f"f{i}", is_success=False)
        batch = buf.sample(batch_size=100, device="cpu")
        assert batch["reward"].shape[0] == 100

    def test_set_sample_ratio(self):
        buf = _make_buffer()
        buf.set_sample_ratio(0.8)
        assert buf.sample_ratio == 0.8

    def test_set_sample_ratio_invalid(self):
        buf = _make_buffer()
        with pytest.raises(ValueError):
            buf.set_sample_ratio(1.5)
        with pytest.raises(ValueError):
            buf.set_sample_ratio(-0.1)


# ---------------------------------------------------------------------------
# get_all_transitions / get_episode_boundaries
# ---------------------------------------------------------------------------

class TestTransitionsAndBoundaries:
    def test_get_all_transitions_merges(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", n_steps=3, is_success=True)
        _add_episode(buf, "f0", n_steps=4, is_success=False)
        all_t = buf.get_all_transitions()
        assert len(all_t) == 7

    def test_get_episode_boundaries(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", n_steps=5, is_success=True)
        _add_episode(buf, "f0", n_steps=7, is_success=False)
        boundaries = buf.get_episode_boundaries()
        assert "success_s0" in boundaries
        assert "failure_f0" in boundaries
        s_start, s_end = boundaries["success_s0"]
        assert s_end - s_start + 1 == 5

    def test_len_excludes_pending(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        buf.add(**make_transition_kwargs(episode_id="pending", step=0))
        assert len(buf) == 10
        assert len(buf._pending_episodes) == 1

    def test_size_equals_len(self):
        buf = _make_buffer()
        _add_episode(buf, "s0", is_success=True)
        assert buf.size() == len(buf)
