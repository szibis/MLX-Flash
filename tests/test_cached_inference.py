"""Tests for cached inference router hooks and cache simulation."""
import pytest
from collections import defaultdict
from mlx_flash_compress.cached_inference import (
    ExpertRouter, RoutingEvent, CacheSimState, RustCacheState,
)


class TestCacheSimState:
    def test_empty_cache_all_misses(self):
        cache = CacheSimState(capacity_experts=10)
        events = [RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0)]
        hits, misses = cache.process_token(events)
        assert hits == 0
        assert misses == 2

    def test_second_access_is_hit(self):
        cache = CacheSimState(capacity_experts=10)
        events = [RoutingEvent(layer_idx=0, expert_indices=[1], token_idx=0, timestamp=0)]
        cache.process_token(events)
        hits, misses = cache.process_token(events)
        assert hits == 1
        assert misses == 0

    def test_capacity_enforcement(self):
        cache = CacheSimState(capacity_experts=2)
        e1 = [RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0)]
        cache.process_token(e1)
        assert len(cache.cached) == 2
        e2 = [RoutingEvent(layer_idx=0, expert_indices=[3], token_idx=1, timestamp=0)]
        cache.process_token(e2)
        assert len(cache.cached) <= 2

    def test_hit_rate_calculation(self):
        cache = CacheSimState(capacity_experts=100)
        events = [RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0)]
        cache.process_token(events)
        cache.process_token(events)
        assert cache.hit_rate == 0.5

    def test_lcp_eviction_prefers_cold(self):
        cache = CacheSimState(capacity_experts=2)
        for i in range(5):
            cache.process_token([
                RoutingEvent(layer_idx=0, expert_indices=[1], token_idx=i, timestamp=0)
            ])
        cache.process_token([
            RoutingEvent(layer_idx=0, expert_indices=[2], token_idx=5, timestamp=0)
        ])
        cache.process_token([
            RoutingEvent(layer_idx=0, expert_indices=[3], token_idx=6, timestamp=0)
        ])
        assert (0, 1) in cache.cached
        assert (0, 2) not in cache.cached


class TestExpertRouter:
    def test_router_initializes(self):
        router = ExpertRouter()
        assert router.token_counter == 0
        assert len(router.events) == 0

    def test_get_expert_frequencies(self):
        router = ExpertRouter()
        router.events = [
            RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0),
            RoutingEvent(layer_idx=0, expert_indices=[1, 3], token_idx=1, timestamp=0),
        ]
        freqs = router.get_expert_frequencies()
        assert freqs[(0, 1)] == 2
        assert freqs[(0, 2)] == 1
        assert freqs[(0, 3)] == 1
