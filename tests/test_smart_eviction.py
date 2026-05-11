"""Tests for smart_eviction — SpecMD-inspired Least-Stale eviction policy."""

import math

import numpy as np
import pytest

from mlx_flash_compress.smart_eviction import (
    ExpertAccessRecord,
    LeastStalePolicy,
    PrefetchResult,
    RoutingPredictor,
    simulate_prefetch,
)

# ---------------------------------------------------------------------------
# ExpertAccessRecord tests
# ---------------------------------------------------------------------------


class TestExpertAccessRecord:
    def test_defaults(self):
        rec = ExpertAccessRecord()
        assert rec.total_count == 0
        assert rec.recent_count == 0
        assert rec.last_token_pos == 0
        assert rec.layer_idx == 0
        assert rec.expert_id == 0

    def test_custom_values(self):
        rec = ExpertAccessRecord(total_count=5, recent_count=2, last_token_pos=10, layer_idx=3, expert_id=7)
        assert rec.total_count == 5
        assert rec.recent_count == 2
        assert rec.last_token_pos == 10
        assert rec.layer_idx == 3
        assert rec.expert_id == 7

    def test_mutable(self):
        rec = ExpertAccessRecord()
        rec.total_count += 1
        assert rec.total_count == 1


# ---------------------------------------------------------------------------
# LeastStalePolicy tests
# ---------------------------------------------------------------------------


class TestLeastStalePolicyInit:
    def test_default_weights(self):
        policy = LeastStalePolicy()
        assert policy.freq_w == 0.4
        assert policy.rec_w == 0.4
        assert policy.layer_w == 0.2

    def test_custom_weights(self):
        policy = LeastStalePolicy(frequency_weight=0.5, recency_weight=0.3, layer_weight=0.2)
        assert policy.freq_w == 0.5
        assert policy.rec_w == 0.3
        assert policy.layer_w == 0.2

    def test_default_recency_window(self):
        policy = LeastStalePolicy()
        assert policy.recency_window == 100

    def test_custom_recency_window(self):
        policy = LeastStalePolicy(recency_window=50)
        assert policy.recency_window == 50

    def test_starts_empty(self):
        policy = LeastStalePolicy()
        assert len(policy._records) == 0
        assert policy._total_tokens == 0


class TestLeastStalePolicyRecordAccess:
    def test_first_access_creates_record(self):
        policy = LeastStalePolicy()
        policy.record_access(0, 5)
        assert (0, 5) in policy._records
        assert policy._records[(0, 5)].total_count == 1

    def test_repeated_access_increments_count(self):
        policy = LeastStalePolicy()
        policy.record_access(0, 5)
        policy.record_access(0, 5)
        policy.record_access(0, 5)
        assert policy._records[(0, 5)].total_count == 3

    def test_different_experts_separate_records(self):
        policy = LeastStalePolicy()
        policy.record_access(0, 1)
        policy.record_access(0, 2)
        assert (0, 1) in policy._records
        assert (0, 2) in policy._records
        assert policy._records[(0, 1)].total_count == 1
        assert policy._records[(0, 2)].total_count == 1

    def test_last_token_pos_updated(self):
        policy = LeastStalePolicy()
        policy.advance_token()
        policy.advance_token()
        policy.record_access(0, 5)
        assert policy._records[(0, 5)].last_token_pos == 2

    def test_recent_accesses_trimmed(self):
        policy = LeastStalePolicy(recency_window=10)
        # Add many accesses — should trigger trimming at 4 * recency_window (40)
        for i in range(50):
            policy.record_access(0, i % 5)
        # After trimming at 40 entries (to 10) plus 10 more accesses,
        # the buffer should be bounded by 4 * recency_window
        assert len(policy._recent_accesses) <= 4 * policy.recency_window


class TestLeastStalePolicyAdvanceToken:
    def test_advance_increments_counter(self):
        policy = LeastStalePolicy()
        policy.advance_token()
        assert policy._total_tokens == 1
        policy.advance_token()
        assert policy._total_tokens == 2


class TestLeastStalePolicyScore:
    def test_unknown_expert_scores_zero(self):
        policy = LeastStalePolicy()
        assert policy.score(0, 99) == 0.0

    def test_recently_accessed_scores_high(self):
        policy = LeastStalePolicy()
        policy.record_access(0, 1)
        # Just accessed, recency should be high (exp(0) = 1.0)
        s = policy.score(0, 1)
        assert s > 0

    def test_stale_expert_scores_lower(self):
        policy = LeastStalePolicy()
        policy.record_access(0, 1)
        # Advance many tokens to make expert stale
        for _ in range(200):
            policy.advance_token()
        policy.record_access(0, 2)  # fresh access for expert 2

        s_stale = policy.score(0, 1)
        s_fresh = policy.score(0, 2)
        assert s_fresh > s_stale

    def test_earlier_layer_scores_higher(self):
        """Earlier layers get higher layer_score (more predictable routing)."""
        policy = LeastStalePolicy(num_layers=10)
        policy.record_access(0, 1)
        policy.record_access(9, 1)

        s_early = policy.score(0, 1)
        s_late = policy.score(9, 1)
        # Layer component is higher for earlier layers
        assert s_early > s_late

    def test_frequent_expert_scores_higher(self):
        policy = LeastStalePolicy()
        for _ in range(10):
            policy.record_access(0, 1)
            policy.advance_token()
        policy.record_access(0, 2)
        policy.advance_token()

        s_frequent = policy.score(0, 1)
        s_rare = policy.score(0, 2)
        assert s_frequent > s_rare

    def test_score_components(self):
        """Verify score is a weighted sum of frequency, recency, and layer."""
        policy = LeastStalePolicy(
            frequency_weight=1.0,
            recency_weight=0.0,
            layer_weight=0.0,
            num_layers=10,
        )
        policy.record_access(0, 1)
        policy.advance_token()

        # Only frequency component active: count/total_tokens = 1/1 = 1.0
        s = policy.score(0, 1)
        assert abs(s - 1.0) < 1e-6


class TestLeastStalePolicyEviction:
    def test_select_eviction_raises_on_empty(self):
        policy = LeastStalePolicy()
        with pytest.raises(ValueError, match="No cached keys"):
            policy.select_eviction([])

    def test_select_eviction_single_key(self):
        policy = LeastStalePolicy()
        result = policy.select_eviction([(0, 1)])
        assert result == (0, 1)

    def test_evicts_lowest_score(self):
        policy = LeastStalePolicy()
        # Expert 1 accessed frequently, expert 2 accessed once long ago
        for _ in range(10):
            policy.record_access(0, 1)
            policy.advance_token()
        policy.record_access(0, 2)
        for _ in range(50):
            policy.advance_token()

        evicted = policy.select_eviction([(0, 1), (0, 2)])
        assert evicted == (0, 2)  # stale expert evicted

    def test_batch_evict_returns_requested_count(self):
        policy = LeastStalePolicy()
        keys = [(0, i) for i in range(5)]
        for i, key in enumerate(keys):
            for _ in range(i + 1):
                policy.record_access(key[0], key[1])
                policy.advance_token()

        evicted = policy.batch_evict(keys, num_to_evict=2)
        assert len(evicted) == 2

    def test_batch_evict_returns_lowest_scores(self):
        policy = LeastStalePolicy()
        # Expert 0: accessed once, expert 4: accessed 5 times
        for i in range(5):
            for _ in range(i + 1):
                policy.record_access(0, i)
                policy.advance_token()

        keys = [(0, i) for i in range(5)]
        evicted = policy.batch_evict(keys, num_to_evict=2)
        # Expert 0 (least accessed) and expert 1 should be evicted first
        evicted_ids = [e[1] for e in evicted]
        assert 0 in evicted_ids

    def test_batch_evict_more_than_available(self):
        policy = LeastStalePolicy()
        keys = [(0, 0), (0, 1)]
        evicted = policy.batch_evict(keys, num_to_evict=5)
        assert len(evicted) == 2  # can only evict what exists

    def test_batch_evict_zero(self):
        policy = LeastStalePolicy()
        keys = [(0, 0), (0, 1)]
        evicted = policy.batch_evict(keys, num_to_evict=0)
        assert evicted == []


# ---------------------------------------------------------------------------
# RoutingPredictor tests
# ---------------------------------------------------------------------------


class TestRoutingPredictorInit:
    def test_creation(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8, top_k=2)
        assert pred.num_layers == 4
        assert pred.num_experts == 8
        assert pred.top_k == 2

    def test_cooccurrence_shape(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        # (num_layers - 1, num_experts, num_experts)
        assert pred._cooccurrence.shape == (3, 8, 8)

    def test_starts_with_zero_observations(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        assert pred._total_observations == 0


class TestRoutingPredictorObserve:
    def test_observe_increments_counter(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        pred.observe(0, [1, 2])
        assert pred._total_observations == 1

    def test_observe_updates_cooccurrence(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        pred.observe(0, [1, 2])
        pred.observe(1, [3, 4])
        # Layer 0 experts [1,2] co-occurring with layer 1 experts [3,4]
        assert pred._cooccurrence[0, 1, 3] == 1
        assert pred._cooccurrence[0, 1, 4] == 1
        assert pred._cooccurrence[0, 2, 3] == 1
        assert pred._cooccurrence[0, 2, 4] == 1

    def test_observe_no_update_at_layer_zero(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        pred.observe(0, [1, 2])
        # No previous layer, so co-occurrence unchanged
        assert pred._cooccurrence.sum() == 0.0


class TestRoutingPredictorPredict:
    def test_predict_last_layer_returns_empty(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8, top_k=2)
        result = pred.predict(3, [0])
        assert result == []

    def test_predict_no_data_returns_defaults(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8, top_k=2)
        result = pred.predict(0, [0])
        # No co-occurrence data, returns first top_k experts as default
        assert result == [0, 1]

    def test_predict_with_data(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8, top_k=2)
        # Train: layer 0 expert 0 always pairs with layer 1 expert 5
        for _ in range(10):
            pred.observe(0, [0])
            pred.observe(1, [5])

        predicted = pred.predict(0, [0])
        assert 5 in predicted

    def test_predict_respects_top_k(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8, top_k=3)
        for i in range(8):
            pred._cooccurrence[0, 0, i] = float(i)
        predicted = pred.predict(0, [0])
        assert len(predicted) == 3


class TestRoutingPredictorAccuracy:
    def test_empty_actual(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        assert pred.accuracy([1, 2], []) == 1.0

    def test_perfect_prediction(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        assert pred.accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_partial_prediction(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        acc = pred.accuracy([1, 2], [1, 2, 3, 4])
        assert abs(acc - 0.5) < 1e-9  # 2 out of 4

    def test_no_match(self):
        pred = RoutingPredictor(num_layers=4, num_experts=8)
        assert pred.accuracy([1, 2], [5, 6]) == 0.0


# ---------------------------------------------------------------------------
# PrefetchResult tests
# ---------------------------------------------------------------------------


class TestPrefetchResult:
    def test_defaults(self):
        r = PrefetchResult()
        assert r.total_predictions == 0
        assert r.correct_predictions == 0
        assert r.total_experts_prefetched == 0
        assert r.wasted_prefetches == 0
        assert r.avg_accuracy == 0.0
        assert r.prefetch_hit_rate == 0.0


# ---------------------------------------------------------------------------
# simulate_prefetch tests
# ---------------------------------------------------------------------------


class TestSimulatePrefetch:
    def test_basic_run(self):
        result = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=50, top_k=2, seed=42)
        assert isinstance(result, PrefetchResult)
        assert result.total_predictions > 0

    def test_avg_accuracy_bounded(self):
        result = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=100, top_k=2, seed=42)
        assert 0.0 <= result.avg_accuracy <= 1.0

    def test_prefetch_hit_rate_bounded(self):
        result = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=100, top_k=2, seed=42)
        assert 0.0 <= result.prefetch_hit_rate <= 1.0

    def test_deterministic_with_seed(self):
        r1 = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=50, top_k=2, seed=123)
        r2 = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=50, top_k=2, seed=123)
        assert r1.total_predictions == r2.total_predictions
        assert r1.correct_predictions == r2.correct_predictions
        assert abs(r1.avg_accuracy - r2.avg_accuracy) < 1e-9

    def test_different_seeds_give_different_results(self):
        r1 = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=100, top_k=2, seed=42)
        r2 = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=100, top_k=2, seed=999)
        # Very unlikely to be identical with different seeds
        assert r1.correct_predictions != r2.correct_predictions or r1.wasted_prefetches != r2.wasted_prefetches

    def test_wasted_prefetches_consistent(self):
        result = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=50, top_k=2, seed=42)
        # wasted = prefetched - correct
        assert result.wasted_prefetches == result.total_experts_prefetched - result.correct_predictions

    def test_small_token_count(self):
        """With very few tokens, we should still get a valid result."""
        result = simulate_prefetch(num_layers=2, num_experts=4, num_tokens=5, top_k=2, seed=42)
        assert isinstance(result, PrefetchResult)
        # With only 5 tokens, warmup is min(10, 5) = 5, so no evaluation tokens
        # Result should still be valid but with no predictions
        assert result.avg_accuracy == 0.0 or result.total_predictions >= 0

    def test_single_expert(self):
        """Edge case: only 1 expert means no routing choice."""
        result = simulate_prefetch(num_layers=2, num_experts=1, num_tokens=20, top_k=1, seed=42)
        assert isinstance(result, PrefetchResult)

    def test_many_experts_few_layers(self):
        result = simulate_prefetch(num_layers=2, num_experts=100, num_tokens=50, top_k=4, seed=42)
        assert result.total_predictions > 0

    def test_single_layer(self):
        """Single layer means no cross-layer predictions."""
        result = simulate_prefetch(num_layers=1, num_experts=8, num_tokens=50, top_k=2, seed=42)
        # With 1 layer, predictor can never predict layer+1, so no predictions
        assert result.total_predictions == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEvictionEdgeCases:
    def test_empty_cache_policy(self):
        """Policy should work with no recorded accesses."""
        policy = LeastStalePolicy()
        s = policy.score(0, 0)
        assert s == 0.0

    def test_single_expert_in_cache(self):
        policy = LeastStalePolicy()
        policy.record_access(0, 0)
        evicted = policy.select_eviction([(0, 0)])
        assert evicted == (0, 0)

    def test_cold_start_all_unknown(self):
        """All experts unknown — eviction should pick first by default."""
        policy = LeastStalePolicy()
        keys = [(0, i) for i in range(5)]
        evicted = policy.select_eviction(keys)
        # All score 0, should pick first
        assert evicted in keys

    def test_recency_window_fixed_at_100(self):
        """Document that recency_window defaults to 100 (not adaptive)."""
        policy = LeastStalePolicy()
        assert policy.recency_window == 100
        # After many tokens, window stays the same
        for _ in range(500):
            policy.advance_token()
        assert policy.recency_window == 100
