"""Tests for real expert streaming with GPU lookup tables."""

import numpy as np
import pytest

try:
    from mlx_flash_compress.expert_streaming import (
        ExpertCache,
        LCPTracker,
        SafetensorsMap,
        StreamingState,
    )

    HAS_MODULE = True
except (ImportError, ModuleNotFoundError):
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="expert_streaming requires mlx")


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
        from mlx_flash_compress.expert_streaming import ExpertCache, enable_skip_fallback

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

        result = get_warmup_experts(task="coding", num_layers=4, num_experts=10, top_n=5)
        assert len(result) == 4
        for layer_experts in result:
            assert isinstance(layer_experts, list)
            assert len(layer_experts) <= 5

    def test_fallback_for_unknown_task(self):
        from mlx_flash_compress.expert_streaming import get_warmup_experts

        # Unknown task should fallback gracefully (not raise)
        result = get_warmup_experts(task="nonexistent_task_xyz", num_layers=3, num_experts=8, top_n=4)
        assert len(result) == 3
        # Fallback returns list(range(top_n)) per layer
        for layer_experts in result:
            assert layer_experts == [0, 1, 2, 3]

    def test_general_task(self):
        from mlx_flash_compress.expert_streaming import get_warmup_experts

        result = get_warmup_experts(task="general", num_layers=2, num_experts=10, top_n=3)
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


class TestLCPTrackerDepthBias:
    """Test layer-depth bias in LCPTracker (FATE paper feature)."""

    def test_no_bias_default(self):
        tracker = LCPTracker(num_experts=10, layer_depth_bias=0.0)
        assert tracker._depth_multiplier == 1.0

    def test_shallow_layer_higher_priority(self):
        """Early layers (low layer_frac) should get higher depth multiplier."""
        shallow = LCPTracker(num_experts=10, layer_depth_bias=0.5, layer_frac=0.0)
        deep = LCPTracker(num_experts=10, layer_depth_bias=0.5, layer_frac=1.0)
        assert shallow._depth_multiplier > deep._depth_multiplier

    def test_depth_bias_affects_priority(self):
        shallow = LCPTracker(num_experts=10, layer_depth_bias=1.0, layer_frac=0.0)
        deep = LCPTracker(num_experts=10, layer_depth_bias=1.0, layer_frac=1.0)
        shallow.record([0])
        deep.record([0])
        assert shallow.priority(0) > deep.priority(0)

    def test_record_bounds_checking(self):
        """Out-of-bounds expert IDs should be handled safely."""
        tracker = LCPTracker(num_experts=5)
        tracker.record([0, 4])  # valid
        tracker.record([-1, 5, 100])  # out of bounds
        assert tracker.frequency[0] == 1
        assert tracker.frequency[4] == 1


class TestExpertCacheStats:
    """Test ExpertCache.stats() without needing actual safetensors files."""

    def test_new_cache_stats(self):
        cache = ExpertCache.__new__(ExpertCache)
        cache.layer_idx = 3
        cache.num_experts = 60
        cache.capacity = 20
        cache.cached_ids = list(range(20))
        cache.total_tokens = 0
        cache.cache_updates = 0
        stats = cache.stats()
        assert stats["layer"] == 3
        assert stats["cached"] == 20
        assert stats["capacity"] == 20
        assert abs(stats["coverage"] - 20 / 60) < 0.001
        assert stats["tokens"] == 0
        assert stats["updates"] == 0

    def test_empty_cache_stats(self):
        cache = ExpertCache.__new__(ExpertCache)
        cache.layer_idx = 0
        cache.num_experts = 10
        cache.capacity = 5
        cache.cached_ids = []
        cache.total_tokens = 100
        cache.cache_updates = 5
        stats = cache.stats()
        assert stats["cached"] == 0
        assert stats["coverage"] == 0.0


class TestStreamingStateAggregation:
    """Test StreamingState methods that aggregate across multiple caches."""

    def test_total_cached_multiple_caches(self):
        state = StreamingState()
        c1 = ExpertCache.__new__(ExpertCache)
        c1.layer_idx = 0
        c1.num_experts = 10
        c1.capacity = 5
        c1.cached_ids = [0, 1, 2]
        c1.total_tokens = 0
        c1.cache_updates = 0

        c2 = ExpertCache.__new__(ExpertCache)
        c2.layer_idx = 1
        c2.num_experts = 10
        c2.capacity = 5
        c2.cached_ids = [0, 1, 2, 3, 4]
        c2.total_tokens = 0
        c2.cache_updates = 0

        state.caches = [c1, c2]
        assert state.total_cached() == 8

    def test_avg_coverage(self):
        state = StreamingState()
        c1 = ExpertCache.__new__(ExpertCache)
        c1.layer_idx = 0
        c1.num_experts = 10
        c1.capacity = 5
        c1.cached_ids = [0, 1, 2, 3, 4]  # 50% coverage
        c1.total_tokens = 0
        c1.cache_updates = 0

        c2 = ExpertCache.__new__(ExpertCache)
        c2.layer_idx = 1
        c2.num_experts = 10
        c2.capacity = 10
        c2.cached_ids = list(range(10))  # 100% coverage
        c2.total_tokens = 0
        c2.cache_updates = 0

        state.caches = [c1, c2]
        avg = state.avg_coverage()
        assert abs(avg - 0.75) < 0.001  # (0.5 + 1.0) / 2
