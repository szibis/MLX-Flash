"""Tests for vertical expert splitting."""
import numpy as np
import pytest
from mlx_flash_compress.vertical_split import (
    SplitConfig,
    VerticalSplitCache,
    estimate_split_benefit,
)


class TestSplitConfig:
    def test_default_config(self):
        cfg = SplitConfig()
        assert cfg.split_factor == 2
        assert cfg.cached_fraction == 0.5

    def test_custom_factor(self):
        cfg = SplitConfig(split_factor=4)
        assert cfg.cached_fraction == 0.25


class TestVerticalSplitCache:
    def test_create(self):
        cache = VerticalSplitCache(
            num_experts=60, rows=256, cols=128, capacity=10
        )
        assert cache.split_rows == 128  # 256 / 2
        assert cache.split_capacity == 20  # 10 * 2
        assert cache.full_capacity == 10

    def test_coverage_empty(self):
        cache = VerticalSplitCache(num_experts=60, rows=256, cols=128, capacity=10)
        cov = cache.coverage()
        assert cov["partial_cached"] == 0
        assert cov["full_cached"] == 0
        assert cov["effective_coverage"] == 0

    def test_plan_allocation_all_hot(self):
        cache = VerticalSplitCache(num_experts=60, rows=256, cols=128, capacity=5)
        plan = cache.plan_allocation(
            hot_expert_ids=[0, 1, 2, 3, 4],
            warm_expert_ids=[10, 11, 12, 13, 14]
        )
        assert plan["full_count"] == 5
        assert plan["partial_count"] == 0  # no room left
        assert plan["total_experts_cached"] == 5

    def test_plan_allocation_mixed(self):
        cache = VerticalSplitCache(num_experts=60, rows=256, cols=128, capacity=5)
        plan = cache.plan_allocation(
            hot_expert_ids=[0, 1],  # 2 full = 4 split units
            warm_expert_ids=[10, 11, 12, 13, 14, 15]
        )
        assert plan["full_count"] == 2
        assert plan["partial_count"] == 6  # 10 - 4 = 6 remaining units
        assert plan["total_experts_cached"] == 8

    def test_plan_allocation_all_warm(self):
        cache = VerticalSplitCache(num_experts=60, rows=256, cols=128, capacity=5)
        plan = cache.plan_allocation(
            hot_expert_ids=[],
            warm_expert_ids=list(range(20))
        )
        assert plan["full_count"] == 0
        assert plan["partial_count"] == 10  # 5*2 = 10 split units
        assert plan["total_experts_cached"] == 10

    def test_simulate_hit_rate(self):
        cache = VerticalSplitCache(num_experts=10, rows=64, cols=32, capacity=3)
        cache.full_ids = [0, 1, 2]
        cache.partial_ids = [3, 4, 5, 6, 7, 8]

        trace = [[0, 1], [2, 3], [4, 9], [0, 5]]
        result = cache.simulate_hit_rate(trace)

        assert result["total_lookups"] == 8
        assert result["full_hits"] == 4  # 0,1,2,0
        assert result["partial_hits"] == 3  # 3,4,5
        assert result["misses"] == 1  # 9
        assert result["total_hit_rate"] == 7 / 8

    def test_simulate_all_hits(self):
        cache = VerticalSplitCache(num_experts=4, rows=64, cols=32, capacity=2)
        cache.full_ids = [0, 1]
        cache.partial_ids = [2, 3]

        trace = [[0, 1], [2, 3]]
        result = cache.simulate_hit_rate(trace)
        assert result["total_hit_rate"] == 1.0
        assert result["misses"] == 0

    def test_split_factor_3(self):
        cache = VerticalSplitCache(
            num_experts=60, rows=300, cols=128, capacity=5, split_factor=3
        )
        assert cache.split_rows == 100  # 300 / 3
        assert cache.split_capacity == 15  # 5 * 3


class TestEstimateSplitBenefit:
    def test_basic_estimate(self):
        result = estimate_split_benefit(num_experts=60, capacity=10)
        assert result["full_cache_hit_rate"] > 0
        assert result["split_cache_hit_rate"] >= result["full_cache_hit_rate"]
        assert result["improvement"] >= 0

    def test_split_improves_hit_rate(self):
        result = estimate_split_benefit(num_experts=60, capacity=10, split_factor=2)
        assert result["split_cache_hit_rate"] > result["full_cache_hit_rate"]
        assert result["experts_cached_split"] == 20
        assert result["experts_cached_full"] == 10

    def test_full_capacity_no_improvement(self):
        """When capacity >= num_experts, splitting adds nothing."""
        result = estimate_split_benefit(num_experts=8, capacity=8, split_factor=2)
        # Both should be ~1.0
        assert result["full_cache_hit_rate"] > 0.99
        assert result["split_cache_hit_rate"] > 0.99

    def test_larger_split_factor(self):
        r2 = estimate_split_benefit(num_experts=60, capacity=10, split_factor=2)
        r4 = estimate_split_benefit(num_experts=60, capacity=10, split_factor=4)
        # 4x split should cover more experts than 2x
        assert r4["experts_cached_split"] >= r2["experts_cached_split"]
