"""Tests for tier optimizer: optimal SSD/RAM split."""

import numpy as np
import pytest

from mlx_flash_compress.tier_optimizer import (
    HardwareProfile,
    ModelProfile,
    TierConfig,
    compute_hit_rate,
    optimize_tiers,
)


class TestHardwareProfile:
    def test_defaults(self):
        hw = HardwareProfile()
        assert hw.total_ram_gb == 48.0
        assert hw.os_overhead_gb == 6.0
        assert hw.kv_cache_gb == 0.5
        assert hw.ssd_bandwidth_gbs == 17.5

    def test_available_ram(self):
        hw = HardwareProfile(total_ram_gb=48.0, os_overhead_gb=6.0, kv_cache_gb=0.5)
        assert hw.available_ram_gb == 48.0 - 6.0 - 0.5

    def test_total_ssd_bandwidth_no_external(self):
        hw = HardwareProfile(ssd_bandwidth_gbs=17.5, num_tb5_drives=0)
        assert hw.total_ssd_bandwidth == 17.5

    def test_total_ssd_bandwidth_with_external(self):
        hw = HardwareProfile(ssd_bandwidth_gbs=17.5, num_tb5_drives=2, tb5_bandwidth_gbs=8.0)
        assert hw.total_ssd_bandwidth == 17.5 + 2 * 8.0

    def test_custom_values(self):
        hw = HardwareProfile(total_ram_gb=192.0, ssd_bandwidth_gbs=20.0)
        assert hw.total_ram_gb == 192.0
        assert hw.available_ram_gb == 192.0 - 6.0 - 0.5


class TestModelProfile:
    def test_defaults(self):
        model = ModelProfile()
        assert model.total_expert_gb == 209.0
        assert model.num_layers == 60
        assert model.num_experts == 512
        assert model.k == 4

    def test_custom_values(self):
        model = ModelProfile(num_layers=32, num_experts=8, k=2)
        assert model.num_layers == 32
        assert model.num_experts == 8
        assert model.k == 2


class TestComputeHitRate:
    def test_all_cached(self):
        rate = compute_hit_rate(ram_experts=100, total_experts=100, k=4, zipf_alpha=0.8)
        assert rate == 1.0

    def test_none_cached(self):
        rate = compute_hit_rate(ram_experts=0, total_experts=100, k=4, zipf_alpha=0.8)
        assert rate == 0.0

    def test_partial_cached(self):
        rate = compute_hit_rate(ram_experts=50, total_experts=100, k=4, zipf_alpha=0.8)
        assert 0.0 < rate < 1.0

    def test_higher_alpha_concentrates_mass(self):
        # Higher alpha = more skewed distribution = higher hit rate for same cache size
        rate_low = compute_hit_rate(ram_experts=10, total_experts=100, k=4, zipf_alpha=0.5)
        rate_high = compute_hit_rate(ram_experts=10, total_experts=100, k=4, zipf_alpha=1.5)
        assert rate_high > rate_low

    def test_more_cached_higher_rate(self):
        rate_small = compute_hit_rate(ram_experts=10, total_experts=100, k=4, zipf_alpha=0.8)
        rate_large = compute_hit_rate(ram_experts=50, total_experts=100, k=4, zipf_alpha=0.8)
        assert rate_large > rate_small

    def test_negative_ram_experts(self):
        rate = compute_hit_rate(ram_experts=-1, total_experts=100, k=4, zipf_alpha=0.8)
        assert rate == 0.0

    def test_overcached(self):
        rate = compute_hit_rate(ram_experts=200, total_experts=100, k=4, zipf_alpha=0.8)
        assert rate == 1.0


class TestOptimizeTiers:
    def test_returns_sorted_results(self):
        hw = HardwareProfile(total_ram_gb=48.0)
        model = ModelProfile(total_expert_gb=100.0, num_layers=30, num_experts=64)
        results = optimize_tiers(hw, model, compression_ratios=[1.0], granularity=5)
        assert len(results) > 0
        # Check sorted by tok/s descending
        for i in range(len(results) - 1):
            assert results[i].tok_per_s >= results[i + 1].tok_per_s

    def test_returns_tier_configs(self):
        hw = HardwareProfile(total_ram_gb=48.0)
        model = ModelProfile(total_expert_gb=100.0, num_layers=30, num_experts=64)
        results = optimize_tiers(hw, model, granularity=3)
        assert all(isinstance(r, TierConfig) for r in results)

    def test_compression_improves_results(self):
        hw = HardwareProfile(total_ram_gb=16.0)
        model = ModelProfile(total_expert_gb=200.0, num_layers=60, num_experts=512)
        results_no_comp = optimize_tiers(hw, model, compression_ratios=[1.0], granularity=5)
        results_comp = optimize_tiers(hw, model, compression_ratios=[2.0], granularity=5)
        best_no_comp = results_no_comp[0].tok_per_s
        best_comp = results_comp[0].tok_per_s
        assert best_comp >= best_no_comp

    def test_tier_config_fields(self):
        hw = HardwareProfile(total_ram_gb=48.0)
        model = ModelProfile(total_expert_gb=100.0)
        results = optimize_tiers(hw, model, granularity=2)
        cfg = results[0]
        assert cfg.ram_fraction >= 0
        assert cfg.compression_ratio > 0
        assert cfg.hit_rate >= 0
        assert cfg.tok_per_s > 0
        assert cfg.layer_ms > 0

    def test_more_ram_better_throughput(self):
        model = ModelProfile(total_expert_gb=200.0)
        hw_small = HardwareProfile(total_ram_gb=16.0)
        hw_large = HardwareProfile(total_ram_gb=192.0)
        best_small = optimize_tiers(hw_small, model, compression_ratios=[1.0], granularity=5)[0].tok_per_s
        best_large = optimize_tiers(hw_large, model, compression_ratios=[1.0], granularity=5)[0].tok_per_s
        assert best_large >= best_small
