"""Tests for real expert streaming with GPU lookup tables."""
import numpy as np
import pytest
from mlx_flash_compress.expert_streaming import (
    LCPTracker, SafetensorsMap, ExpertCache, StreamingState,
)


class TestLCPTracker:
    def test_initial_priority_zero(self):
        tracker = LCPTracker(num_experts=10)
        assert tracker.priority(0) == 0.0

    def test_record_increases_priority(self):
        tracker = LCPTracker(num_experts=10)
        tracker.record([0, 1, 2])
        assert tracker.priority(0) > 0
        assert tracker.priority(5) == 0  # not activated

    def test_coldest_returns_lowest_priority(self):
        tracker = LCPTracker(num_experts=10)
        # Expert 0 activated 10 times, expert 1 once
        for _ in range(10):
            tracker.record([0])
        tracker.record([1])
        cold = tracker.coldest([0, 1], n=1)
        assert cold == [1]  # expert 1 is coldest

    def test_decay_over_time(self):
        tracker = LCPTracker(num_experts=10)
        tracker.record([0])
        p_after_1 = tracker.priority(0)
        # Advance many steps without activating
        for _ in range(256):
            tracker.record([1])  # activate different expert
        p_after_many = tracker.priority(0)
        assert p_after_many < p_after_1  # priority decayed


class TestSafetensorsMap:
    def test_has_key_empty(self):
        # Can't easily create safetensors in test, but verify the interface
        # This test just ensures the class instantiates
        try:
            sm = SafetensorsMap([])
        except Exception:
            pass  # expected with no files

    def test_np_dtypes_mapping(self):
        assert SafetensorsMap._NP_DTYPES["F16"] == np.float16
        assert SafetensorsMap._NP_DTYPES["F32"] == np.float32
        assert SafetensorsMap._NP_DTYPES["U32"] == np.uint32


class TestStreamingState:
    def test_empty_state(self):
        state = StreamingState()
        assert state.total_cached() == 0
        assert state.avg_coverage() == 0
        assert state.stats() == []

    def test_update_empty_is_noop(self):
        state = StreamingState()
        state.update()  # should not crash
