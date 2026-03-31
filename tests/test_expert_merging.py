"""Tests for offline expert merging."""
import numpy as np
import pytest
from mlx_flash_compress.expert_merging import (
    cosine_similarity_matrix,
    plan_expert_merges,
    apply_merges,
    estimate_merge_savings,
    MergePlan,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        w = [np.ones((4, 4), dtype=np.float32)] * 3
        sim = cosine_similarity_matrix(w)
        np.testing.assert_allclose(sim, 1.0, atol=1e-5)

    def test_orthogonal_vectors(self):
        w1 = np.array([1, 0, 0, 0], dtype=np.float32)
        w2 = np.array([0, 1, 0, 0], dtype=np.float32)
        sim = cosine_similarity_matrix([w1, w2])
        assert abs(sim[0, 1]) < 1e-5

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        w = [rng.standard_normal((8, 8)).astype(np.float32) for _ in range(5)]
        sim = cosine_similarity_matrix(w)
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)


class TestPlanExpertMerges:
    def test_no_merges_when_dissimilar(self):
        rng = np.random.default_rng(42)
        # Orthogonal-ish random vectors
        weights = [rng.standard_normal((32, 32)).astype(np.float32) for _ in range(5)]
        plan = plan_expert_merges(weights, threshold=0.99)
        assert plan.merged_count == 5  # all unique

    def test_merge_identical(self):
        base = np.ones((16, 16), dtype=np.float32)
        weights = [base, base.copy(), base.copy(), np.zeros((16, 16), dtype=np.float32)]
        plan = plan_expert_merges(weights, threshold=0.99)
        assert plan.merged_count == 2  # 3 identical + 1 different
        assert plan.reduction > 0

    def test_merge_near_duplicates(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal((16, 16)).astype(np.float32)
        w1 = base.copy()
        w2 = base + rng.standard_normal(base.shape).astype(np.float32) * 0.01
        w3 = rng.standard_normal(base.shape).astype(np.float32)  # different
        plan = plan_expert_merges([w1, w2, w3], threshold=0.95)
        assert plan.merged_count <= 2  # w1 and w2 should merge

    def test_redirect_mapping(self):
        base = np.ones((8, 8), dtype=np.float32)
        weights = [base, base.copy()]
        plan = plan_expert_merges(weights, threshold=0.99)
        # Both should redirect to same cluster
        assert plan.redirect[0] == plan.redirect[1]

    def test_empty(self):
        plan = plan_expert_merges([], threshold=0.95)
        assert plan.merged_count == 0
        assert plan.reduction == 0.0


class TestApplyMerges:
    def test_apply_averaging(self):
        w1 = np.full((4, 4), 2.0, dtype=np.float32)
        w2 = np.full((4, 4), 4.0, dtype=np.float32)
        plan = MergePlan(
            clusters={0: [0, 1]},
            redirect={0: 0, 1: 0},
            original_count=2,
            merged_count=1,
        )
        merged = apply_merges([w1, w2], plan)
        assert len(merged) == 1
        np.testing.assert_allclose(merged[0], 3.0)  # average of 2 and 4

    def test_apply_no_merge(self):
        w1 = np.ones((4, 4), dtype=np.float32)
        w2 = np.zeros((4, 4), dtype=np.float32)
        plan = MergePlan(
            clusters={0: [0], 1: [1]},
            redirect={0: 0, 1: 1},
            original_count=2,
            merged_count=2,
        )
        merged = apply_merges([w1, w2], plan)
        assert len(merged) == 2
        np.testing.assert_array_equal(merged[0], w1)
        np.testing.assert_array_equal(merged[1], w2)


class TestEstimateMergeSavings:
    def test_basic_estimate(self):
        result = estimate_merge_savings(num_experts=20, threshold=0.95)
        assert result["original_experts"] == 20
        assert result["merged_experts"] <= 20
        assert result["reduction_pct"] >= 0

    def test_high_threshold_fewer_merges(self):
        r_high = estimate_merge_savings(num_experts=20, threshold=0.999)
        r_low = estimate_merge_savings(num_experts=20, threshold=0.90)
        assert r_low["merged_experts"] <= r_high["merged_experts"]

    def test_deterministic(self):
        r1 = estimate_merge_savings(num_experts=10, seed=42)
        r2 = estimate_merge_savings(num_experts=10, seed=42)
        assert r1 == r2
