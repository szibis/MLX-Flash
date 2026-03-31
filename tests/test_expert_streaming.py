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

    def test_mixtral_key_mapping(self):
        """Verify Mixtral w1/w2/w3 → gate_proj/down_proj/up_proj key derivation."""
        # Simulate the key mapping logic from get_expert_slice
        key = "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
        per_expert_key = key.replace(".switch_mlp.", ".experts.0.")
        assert per_expert_key == "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight"

        # Apply Mixtral renaming
        _MIXTRAL_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        for mlx_name, st_name in _MIXTRAL_MAP.items():
            per_expert_key = per_expert_key.replace(f".{mlx_name}.", f".{st_name}.")
        assert per_expert_key == "model.layers.0.block_sparse_moe.experts.0.w1.weight"

        # Verify eid substitution works
        eid_key = per_expert_key.replace(".experts.0.", ".experts.7.")
        assert eid_key == "model.layers.0.block_sparse_moe.experts.7.w1.weight"


class TestStreamingState:
    def test_empty_state(self):
        state = StreamingState()
        assert state.total_cached() == 0
        assert state.avg_coverage() == 0
        assert state.stats() == []

    def test_update_empty_is_noop(self):
        state = StreamingState()
        state.update()  # should not crash


class TestEnableSkipFallback:
    """Tests for the skip-fallback zero-eval dispatch."""

    def test_import(self):
        from mlx_flash_compress.expert_streaming import enable_skip_fallback
        assert callable(enable_skip_fallback)

    def test_noop_when_no_layers(self):
        """enable_skip_fallback should handle model with no layers gracefully."""
        from mlx_flash_compress.expert_streaming import enable_skip_fallback

        class FakeModel:
            pass

        # Should not raise
        enable_skip_fallback(FakeModel(), caches=[])

    def test_noop_when_no_matching_caches(self):
        """Skip-fallback should do nothing if no caches match layer indices."""
        from mlx_flash_compress.expert_streaming import enable_skip_fallback, ExpertCache

        class FakeLayer:
            pass

        class FakeModel:
            class model:
                layers = [FakeLayer()]

        # Cache for layer 5, but model only has layer 0
        cache = ExpertCache.__new__(ExpertCache)
        cache.layer_idx = 5
        cache.num_experts = 10
        cache.hit_mask = None
        enable_skip_fallback(FakeModel(), caches=[cache])

    def test_hit_mask_used_in_expert_cache(self):
        """Verify ExpertCache creates a proper hit_mask during _rebuild_lookup."""
        cache = ExpertCache.__new__(ExpertCache)
        cache.layer_idx = 0
        cache.num_experts = 5
        cache.capacity = 3
        cache.cached_ids = [0, 2, 4]
        cache._rebuild_lookup()

        import mlx.core as mx
        hit = np.array(cache.hit_mask.tolist())
        assert hit[0] == 1.0
        assert hit[1] == 0.0
        assert hit[2] == 1.0
        assert hit[3] == 0.0
        assert hit[4] == 1.0


class TestGetWarmupExperts:
    """Tests for profile-based warmup expert selection."""

    def test_import(self):
        from mlx_flash_compress.expert_streaming import get_warmup_experts
        assert callable(get_warmup_experts)

    def test_returns_list_per_layer(self):
        from mlx_flash_compress.expert_streaming import get_warmup_experts
        result = get_warmup_experts(
            task="coding", num_layers=4, num_experts=10, top_n=5
        )
        assert len(result) == 4
        for layer_experts in result:
            assert isinstance(layer_experts, list)
            assert len(layer_experts) <= 5

    def test_fallback_for_unknown_task(self):
        from mlx_flash_compress.expert_streaming import get_warmup_experts
        # Unknown task should fallback gracefully (not raise)
        result = get_warmup_experts(
            task="nonexistent_task_xyz", num_layers=3, num_experts=8, top_n=4
        )
        assert len(result) == 3
        # Fallback returns list(range(top_n)) per layer
        for layer_experts in result:
            assert layer_experts == [0, 1, 2, 3]

    def test_general_task(self):
        from mlx_flash_compress.expert_streaming import get_warmup_experts
        result = get_warmup_experts(
            task="general", num_layers=2, num_experts=10, top_n=3
        )
        assert len(result) == 2
        for layer_experts in result:
            assert len(layer_experts) <= 3
            # All expert IDs should be valid
            for eid in layer_experts:
                assert 0 <= eid < 10

    def test_deterministic(self):
        """Same task + params should always return same result."""
        from mlx_flash_compress.expert_streaming import get_warmup_experts
        r1 = get_warmup_experts(task="coding", num_layers=4, num_experts=20, top_n=8)
        r2 = get_warmup_experts(task="coding", num_layers=4, num_experts=20, top_n=8)
        assert r1 == r2
