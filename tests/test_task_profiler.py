"""Tests for task-aware expert profiler: profiles, calibration, adaptive profiling."""

import json
import tempfile
import time

import numpy as np
import pytest

from mlx_flash_compress.task_profiler import (
    PREDEFINED_TASKS,
    AdaptiveProfiler,
    ExpertProfile,
    ProfileCalibrator,
    estimate_profile_gains,
    get_predefined_profile,
)


class TestExpertProfile:
    def test_create_empty(self):
        p = ExpertProfile(name="test")
        assert p.name == "test"
        assert p.expert_scores == {}
        assert p.calibration_tokens == 0

    def test_get_hot_experts_empty(self):
        p = ExpertProfile(name="empty")
        hot = p.get_hot_experts(top_pct=0.3)
        assert hot == {}

    def test_get_hot_experts(self):
        p = ExpertProfile(
            name="test",
            expert_scores={
                "0": {"0": 0.9, "1": 0.1, "2": 0.5, "3": 0.3},
            },
        )
        hot = p.get_hot_experts(top_pct=0.5)
        # top 50% of 4 experts = top 2
        assert 0 in hot
        assert len(hot[0]) == 2
        assert hot[0][0] == 0  # expert 0 is hottest
        assert hot[0][1] == 2  # expert 2 is second

    def test_get_cold_experts(self):
        p = ExpertProfile(
            name="test",
            expert_scores={
                "0": {"0": 0.9, "1": 0.1, "2": 0.5, "3": 0.3},
            },
        )
        cold = p.get_cold_experts(bottom_pct=0.5)
        assert 0 in cold
        assert len(cold[0]) == 2
        assert cold[0][0] == 1  # expert 1 is coldest
        assert cold[0][1] == 3  # expert 3 is second coldest

    def test_get_hot_experts_minimum_one(self):
        """Even with a single expert, top_pct should return at least 1."""
        p = ExpertProfile(
            name="test",
            expert_scores={"0": {"0": 1.0}},
        )
        hot = p.get_hot_experts(top_pct=0.01)
        assert len(hot[0]) == 1

    def test_overlap_identical(self):
        scores = {"0": {"0": 0.9, "1": 0.1, "2": 0.8}}
        p1 = ExpertProfile(name="a", expert_scores=scores)
        p2 = ExpertProfile(name="b", expert_scores=scores)
        overlap = p1.overlap(p2)
        assert overlap == 1.0

    def test_overlap_empty(self):
        p1 = ExpertProfile(name="a")
        p2 = ExpertProfile(name="b")
        assert p1.overlap(p2) == 0.0

    def test_overlap_one_empty(self):
        p1 = ExpertProfile(name="a", expert_scores={"0": {"0": 0.9}})
        p2 = ExpertProfile(name="b")
        assert p1.overlap(p2) == 0.0

    def test_overlap_different(self):
        """Different profiles should have overlap < 1.0."""
        p1 = ExpertProfile(
            name="a",
            expert_scores={"0": {str(i): 0.9 if i < 3 else 0.01 for i in range(10)}},
        )
        p2 = ExpertProfile(
            name="b",
            expert_scores={"0": {str(i): 0.01 if i < 3 else 0.9 for i in range(10)}},
        )
        overlap = p1.overlap(p2)
        assert 0.0 <= overlap < 1.0

    def test_save_load_roundtrip(self):
        p = ExpertProfile(
            name="roundtrip",
            description="Test save/load",
            expert_scores={"0": {"0": 0.5, "1": 0.3}},
            calibration_tokens=100,
            created_at=1000.0,
            updated_at=2000.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        p.save(path)
        loaded = ExpertProfile.load(path)
        assert loaded.name == "roundtrip"
        assert loaded.description == "Test save/load"
        assert loaded.calibration_tokens == 100
        assert loaded.expert_scores["0"]["0"] == 0.5

    def test_multiple_layers(self):
        p = ExpertProfile(
            name="multi",
            expert_scores={
                "0": {"0": 0.9, "1": 0.1},
                "1": {"0": 0.2, "1": 0.8},
            },
        )
        hot = p.get_hot_experts(top_pct=0.5)
        assert hot[0] == [0]  # layer 0 hot expert is 0
        assert hot[1] == [1]  # layer 1 hot expert is 1


class TestPredefinedTasks:
    def test_all_tasks_defined(self):
        expected = {"coding", "writing", "math", "chat", "analysis", "translation"}
        assert set(PREDEFINED_TASKS.keys()) == expected

    def test_each_task_has_required_fields(self):
        for task, info in PREDEFINED_TASKS.items():
            assert "seed" in info
            assert "hot_fraction" in info
            assert "description" in info

    def test_get_predefined_profile_coding(self):
        p = get_predefined_profile("coding", num_layers=4, num_experts=10)
        assert p.name == "coding"
        assert len(p.expert_scores) == 4
        for layer_str in p.expert_scores:
            assert len(p.expert_scores[layer_str]) == 10

    def test_get_predefined_profile_all_tasks(self):
        for task in PREDEFINED_TASKS:
            p = get_predefined_profile(task, num_layers=4, num_experts=8)
            assert p.name == task
            assert p.calibration_tokens == 10000

    def test_get_predefined_profile_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_predefined_profile("nonexistent_task")

    def test_different_tasks_produce_different_profiles(self):
        p1 = get_predefined_profile("coding", num_layers=4, num_experts=10)
        p2 = get_predefined_profile("writing", num_layers=4, num_experts=10)
        # Different seeds should produce different scores
        assert p1.expert_scores != p2.expert_scores

    def test_deterministic_profiles(self):
        p1 = get_predefined_profile("math", num_layers=4, num_experts=10)
        p2 = get_predefined_profile("math", num_layers=4, num_experts=10)
        assert p1.expert_scores == p2.expert_scores


class TestProfileCalibrator:
    def test_empty_build(self):
        cal = ProfileCalibrator(num_layers=2, num_experts=4)
        profile = cal.build_profile("empty")
        assert profile.name == "empty"
        assert profile.calibration_tokens == 0
        # All scores should be 0
        for layer_str in profile.expert_scores:
            for eid_str, score in profile.expert_scores[layer_str].items():
                assert score == 0.0

    def test_record_and_build(self):
        cal = ProfileCalibrator(num_layers=2, num_experts=4)
        # Layer 0: experts 0, 1 activated equally
        cal.record(0, [0, 1])
        cal.record(0, [0, 1])
        # Layer 1: expert 2 activated heavily
        cal.record(1, [2, 2, 2])

        profile = cal.build_profile("test_task", description="test")
        assert profile.description == "test"
        # Layer 0: experts 0 and 1 should have equal non-zero scores
        s0 = profile.expert_scores["0"]
        assert float(s0["0"]) > 0
        assert float(s0["0"]) == float(s0["1"])
        # Expert 2 and 3 should have zero in layer 0
        assert float(s0["2"]) == 0.0
        assert float(s0["3"]) == 0.0

    def test_normalization(self):
        cal = ProfileCalibrator(num_layers=1, num_experts=3)
        cal.record(0, [0])
        cal.record(0, [0])
        cal.record(0, [1])
        profile = cal.build_profile("norm")
        s = profile.expert_scores["0"]
        # Total activations in layer 0: 3. Expert 0: 2/3, Expert 1: 1/3
        assert abs(float(s["0"]) - 2 / 3) < 0.01
        assert abs(float(s["1"]) - 1 / 3) < 0.01


class TestAdaptiveProfiler:
    def test_initial_state(self):
        ap = AdaptiveProfiler(num_layers=2, num_experts=4)
        assert ap._token_count == 0
        priority = ap.get_priority_experts(top_k_per_layer=2)
        # All scores are zero, so no priority experts
        for layer in range(2):
            assert len(priority[layer]) == 0

    def test_observe_token(self):
        ap = AdaptiveProfiler(num_layers=2, num_experts=4, alpha=0.5)
        ap.observe_token([(0, [0, 1]), (1, [2])])
        assert ap._token_count == 1
        priority = ap.get_priority_experts(top_k_per_layer=4)
        assert 0 in priority[0]
        assert 1 in priority[0]
        assert 2 in priority[1]

    def test_ema_decay(self):
        ap = AdaptiveProfiler(num_layers=1, num_experts=4, alpha=0.5)
        # First: expert 0 active
        ap.observe_token([(0, [0])])
        score_after_one = ap._scores[0, 0]
        # Second: expert 1 active (expert 0 not active)
        ap.observe_token([(0, [1])])
        score_after_two = ap._scores[0, 0]
        # Expert 0's score should have decayed
        assert score_after_two < score_after_one

    def test_get_cache_recommendation(self):
        ap = AdaptiveProfiler(num_layers=2, num_experts=4, alpha=0.5)
        ap.observe_token([(0, [0, 1]), (1, [2, 3])])
        recs = ap.get_cache_recommendation(cache_slots=3)
        assert len(recs) <= 3
        for layer, eid in recs:
            assert 0 <= layer < 2
            assert 0 <= eid < 4

    def test_get_cache_recommendation_empty(self):
        ap = AdaptiveProfiler(num_layers=2, num_experts=4)
        recs = ap.get_cache_recommendation(cache_slots=10)
        assert recs == []

    def test_get_profile(self):
        ap = AdaptiveProfiler(num_layers=2, num_experts=4, alpha=0.5)
        ap.observe_token([(0, [0]), (1, [3])])
        profile = ap.get_profile()
        assert profile.name == "adaptive_live"
        assert profile.calibration_tokens == 1

    def test_detect_topic_change_not_enough_data(self):
        ap = AdaptiveProfiler(num_layers=2, num_experts=4)
        assert not ap.detect_topic_change()

    def test_detect_topic_change_stable(self):
        ap = AdaptiveProfiler(num_layers=1, num_experts=4, alpha=0.2)
        # Feed consistent data
        for _ in range(20):
            ap.observe_token([(0, [0, 1])])
        assert not ap.detect_topic_change()

    def test_detect_topic_change_shift(self):
        ap = AdaptiveProfiler(num_layers=1, num_experts=10, alpha=0.05)
        # Build up EMA for experts 0-2
        for _ in range(50):
            ap.observe_token([(0, [0, 1, 2])])
        # Sudden shift to completely different experts
        for _ in range(6):
            ap.observe_token([(0, [7, 8, 9])])
        # Should detect the shift
        changed = ap.detect_topic_change(threshold=0.3)
        assert changed

    def test_window_size_limit(self):
        ap = AdaptiveProfiler(num_layers=1, num_experts=4, window_size=5)
        for _ in range(20):
            ap.observe_token([(0, [0])])
        assert len(ap._window) == 5

    def test_out_of_bounds_ignored(self):
        ap = AdaptiveProfiler(num_layers=2, num_experts=4)
        # Out-of-bounds layer and expert should be silently ignored
        ap.observe_token([(5, [0]), (0, [10])])
        assert ap._token_count == 1


class TestEstimateProfileGains:
    def test_basic_gains(self):
        profile = get_predefined_profile("coding", num_layers=4, num_experts=10)
        gains = estimate_profile_gains(
            profile=profile,
            cache_slots=20,
            num_layers=4,
            num_experts=10,
            k=4,
        )
        assert "profile_hit_rate" in gains
        assert "generic_hit_rate" in gains
        assert "improvement" in gains
        assert "improvement_pct" in gains
        assert 0.0 <= gains["profile_hit_rate"] <= 1.0
        assert 0.0 <= gains["generic_hit_rate"] <= 1.0

    def test_more_cache_slots_higher_hit_rate(self):
        profile = get_predefined_profile("coding", num_layers=4, num_experts=10)
        gains_low = estimate_profile_gains(
            profile=profile, cache_slots=5, num_layers=4, num_experts=10
        )
        gains_high = estimate_profile_gains(
            profile=profile, cache_slots=30, num_layers=4, num_experts=10
        )
        assert gains_high["generic_hit_rate"] >= gains_low["generic_hit_rate"]
