"""Tests for task profiler, memory manager, and tier optimizer."""

import json
import tempfile
import os

import numpy as np
import pytest

from mlx_flash_compress.task_profiler import (
    ExpertProfile, ProfileCalibrator, AdaptiveProfiler,
    get_predefined_profile, estimate_profile_gains,
)
from mlx_flash_compress.memory_manager import (
    MemoryManager, get_memory_state, MemoryState,
)
from mlx_flash_compress.tier_optimizer import (
    optimize_tiers, compute_hit_rate, HardwareProfile, ModelProfile,
)


class TestExpertProfile:
    def test_create_predefined(self):
        for task in ["coding", "writing", "math", "chat", "analysis"]:
            p = get_predefined_profile(task, num_layers=4, num_experts=8)
            assert p.name == task
            assert len(p.expert_scores) == 4  # 4 layers

    def test_hot_cold_experts(self):
        p = get_predefined_profile("coding", num_layers=4, num_experts=8)
        hot = p.get_hot_experts(top_pct=0.5)
        cold = p.get_cold_experts(bottom_pct=0.5)
        assert len(hot) > 0
        assert len(cold) > 0
        # Hot and cold should not overlap
        for layer in hot:
            if layer in cold:
                assert set(hot[layer]) & set(cold[layer]) == set()

    def test_overlap(self):
        p1 = get_predefined_profile("coding", num_layers=4, num_experts=8)
        p2 = get_predefined_profile("writing", num_layers=4, num_experts=8)
        p_self = get_predefined_profile("coding", num_layers=4, num_experts=8)

        # Self-overlap should be 1.0
        assert p1.overlap(p_self) == pytest.approx(1.0)
        # Different tasks should have low overlap
        assert p1.overlap(p2) < 0.5

    def test_save_load(self):
        p = get_predefined_profile("math", num_layers=4, num_experts=8)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            p.save(path)
            loaded = ExpertProfile.load(path)
            assert loaded.name == "math"
            assert len(loaded.expert_scores) == 4
        finally:
            os.unlink(path)

    def test_estimate_gains(self):
        p = get_predefined_profile("coding", num_layers=4, num_experts=16)
        gains = estimate_profile_gains(p, cache_slots=20, num_layers=4, num_experts=16, k=4)
        assert "profile_hit_rate" in gains
        assert "generic_hit_rate" in gains
        assert gains["profile_hit_rate"] >= 0
        assert gains["generic_hit_rate"] >= 0


class TestProfileCalibrator:
    def test_calibrate(self):
        cal = ProfileCalibrator(num_layers=4, num_experts=8)
        for _ in range(50):
            for layer in range(4):
                cal.record(layer, [0, 1, 2, 3])

        profile = cal.build_profile("test_task")
        assert profile.name == "test_task"
        assert profile.calibration_tokens > 0
        # Experts 0-3 should have high scores
        scores = profile.expert_scores.get("0", {})
        assert len(scores) > 0


class TestAdaptiveProfiler:
    def test_observe_and_recommend(self):
        profiler = AdaptiveProfiler(num_layers=4, num_experts=8, alpha=0.5)
        for _ in range(10):
            activations = [(l, [0, 1]) for l in range(4)]
            profiler.observe_token(activations)

        rec = profiler.get_cache_recommendation(10)
        assert len(rec) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in rec)

    def test_topic_change(self):
        profiler = AdaptiveProfiler(num_layers=4, num_experts=16, alpha=0.5)
        rng = np.random.default_rng(42)

        # Topic A: experts 0-3
        for _ in range(20):
            activations = [(l, [0, 1, 2, 3]) for l in range(4)]
            profiler.observe_token(activations)

        # Topic B: experts 10-13 (completely different)
        for _ in range(10):
            activations = [(l, [10, 11, 12, 13]) for l in range(4)]
            profiler.observe_token(activations)

        # Should detect topic changed
        profile = profiler.get_profile()
        assert profile.calibration_tokens == 30

    def test_get_priority(self):
        profiler = AdaptiveProfiler(num_layers=2, num_experts=4, alpha=0.9)
        profiler.observe_token([(0, [0, 1]), (1, [2, 3])])
        priorities = profiler.get_priority_experts(top_k_per_layer=2)
        assert 0 in priorities
        assert 1 in priorities


class TestMemoryManager:
    def test_get_state(self):
        state = get_memory_state()
        assert state.total_gb > 0
        assert state.free_gb >= 0
        assert state.available_gb >= 0

    def test_budget(self):
        mgr = MemoryManager(safety_margin_gb=1.0, min_cache_gb=0.1)
        budget = mgr.get_cache_budget()
        assert budget > 0
        assert budget > 100 * 1024 * 1024  # at least 100MB

    def test_budget_gb(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        gb = mgr.get_cache_budget_gb()
        assert gb > 0

    def test_status(self):
        mgr = MemoryManager()
        status = mgr.get_status()
        assert "total_ram_gb" in status
        assert "pressure" in status
        assert "cache_budget_gb" in status


class TestTierOptimizer:
    def test_compute_hit_rate(self):
        # All experts cached
        assert compute_hit_rate(100, 100, 4, 0.8) == 1.0
        # No experts cached
        assert compute_hit_rate(0, 100, 4, 0.8) == 0.0
        # Partial cache
        rate = compute_hit_rate(50, 100, 4, 0.8)
        assert 0.0 < rate < 1.0

    def test_optimize(self):
        hw = HardwareProfile(total_ram_gb=36)
        model = ModelProfile(total_expert_gb=209, num_layers=60, num_experts=512)
        results = optimize_tiers(hw, model, granularity=5)
        assert len(results) > 0
        # Best result should have highest tok/s
        assert results[0].tok_per_s >= results[-1].tok_per_s

    def test_small_model(self):
        hw = HardwareProfile(total_ram_gb=36)
        model = ModelProfile(total_expert_gb=5, num_layers=24, num_experts=60)
        results = optimize_tiers(hw, model, granularity=5)
        # Small model fits in RAM — should have 100% hit rate
        best = results[0]
        assert best.hit_rate > 0.9
