"""Tests for speculative expert techniques."""
import numpy as np
import pytest
from mlx_flash_compress.speculative_experts import (
    ResidualPredictor,
    ForwardLookingEvictor,
    SpeculativeExecutor,
    SpeculativeResult,
    simulate_speculative_pipeline,
)


class TestResidualPredictor:
    def test_create(self):
        pred = ResidualPredictor(num_layers=8, num_experts=16, hidden_dim=32)
        assert pred.hidden_dim == 32

    def test_predict_without_hidden(self):
        pred = ResidualPredictor(num_layers=4, num_experts=8, top_k=2)
        result = pred.predict(0, hidden_state=None)
        assert len(result) == 2

    def test_predict_with_hidden(self):
        pred = ResidualPredictor(num_layers=4, num_experts=8, top_k=2, hidden_dim=16)
        h = np.random.randn(16).astype(np.float32)
        result = pred.predict(0, hidden_state=h)
        assert len(result) == 2
        assert all(0 <= e < 8 for e in result)

    def test_learn_pattern(self):
        pred = ResidualPredictor(num_layers=4, num_experts=8, top_k=4,
                                  hidden_dim=8, lr=0.1, seed=42)
        # Train: specific hidden state pattern -> specific experts
        h0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        h1 = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
        for _ in range(500):
            pred.observe(0, [5, 6], hidden_state=h0)
            pred.observe(1, [5, 6], hidden_state=h1)

        predicted = pred.predict(0, hidden_state=h0)
        assert 5 in predicted or 6 in predicted

    def test_predict_at_last_layer(self):
        pred = ResidualPredictor(num_layers=4, num_experts=8)
        assert pred.predict(3) == []

    def test_stats(self):
        pred = ResidualPredictor(num_layers=4, num_experts=8, hidden_dim=16)
        h = np.random.randn(16).astype(np.float32)
        pred.observe(0, [1], hidden_state=h)
        pred.observe(1, [2], hidden_state=h)
        s = pred.stats()
        assert s["training_steps"] == 1
        assert s["params_per_layer"] == 16 * 8


class TestForwardLookingEvictor:
    def test_create(self):
        ev = ForwardLookingEvictor(num_experts=10)
        assert ev.num_experts == 10

    def test_no_predictions_equals_lcp(self):
        ev = ForwardLookingEvictor(num_experts=10)
        lcp = {0: 5.0, 1: 3.0, 2: 1.0}
        evicted = ev.select_eviction([0, 1, 2], lcp, n=1)
        assert evicted == [2]  # lowest LCP

    def test_predicted_expert_protected(self):
        ev = ForwardLookingEvictor(num_experts=10, protection_weight=100.0)
        ev.update_predictions([2], steps_ahead=1)
        lcp = {0: 5.0, 1: 3.0, 2: 1.0}  # expert 2 has lowest LCP
        evicted = ev.select_eviction([0, 1, 2], lcp, n=1)
        # Expert 2 should be protected despite low LCP
        assert 2 not in evicted

    def test_clear_predictions(self):
        ev = ForwardLookingEvictor(num_experts=10)
        ev.update_predictions([5], steps_ahead=2)
        ev.clear_predictions()
        assert 5 in ev._predicted_needs  # still alive (distance was 2, now 1)
        ev.clear_predictions()
        assert 5 not in ev._predicted_needs  # expired

    def test_eviction_score_with_bonus(self):
        ev = ForwardLookingEvictor(num_experts=10, protection_weight=10.0)
        ev.update_predictions([3], steps_ahead=1)
        score_protected = ev.eviction_score(3, lcp_priority=1.0)
        score_unprotected = ev.eviction_score(4, lcp_priority=1.0)
        assert score_protected > score_unprotected


class TestSpeculativeExecutor:
    def test_perfect_prediction(self):
        ex = SpeculativeExecutor()
        result = ex.evaluate_speculation([1, 2, 3], [1, 2, 3])
        assert result.hits == 3
        assert result.misses == 0
        assert result.accuracy == 1.0

    def test_partial_prediction(self):
        ex = SpeculativeExecutor()
        result = ex.evaluate_speculation([1, 2, 3], [1, 2, 4])
        assert result.hits == 2
        assert result.misses == 1

    def test_no_overlap(self):
        ex = SpeculativeExecutor()
        result = ex.evaluate_speculation([1, 2], [3, 4])
        assert result.hits == 0
        assert result.misses == 2

    def test_stats_accumulate(self):
        ex = SpeculativeExecutor()
        ex.evaluate_speculation([1, 2], [1, 3])
        ex.evaluate_speculation([4, 5], [4, 5])
        s = ex.stats()
        assert s["hits"] == 3
        assert s["misses"] == 1
        assert s["accuracy"] == 0.75

    def test_time_savings(self):
        ex = SpeculativeExecutor(hit_cost_ms=0.0, miss_cost_ms=1.0)
        result = ex.evaluate_speculation([1, 2, 3, 4], [1, 2, 3, 4])
        assert result.speculative_saves_ms == 4.0


class TestSimulateSpeculativePipeline:
    def test_runs(self):
        result = simulate_speculative_pipeline(
            num_layers=4, num_experts=10, num_tokens=50, top_k=2
        )
        assert "predictor" in result
        assert "executor" in result
        assert result["executor"]["total_speculations"] > 0

    def test_deterministic(self):
        r1 = simulate_speculative_pipeline(num_layers=4, num_experts=10, seed=42)
        r2 = simulate_speculative_pipeline(num_layers=4, num_experts=10, seed=42)
        assert r1["executor"]["accuracy"] == r2["executor"]["accuracy"]

    def test_beats_random(self):
        result = simulate_speculative_pipeline(
            num_layers=8, num_experts=20, num_tokens=200, top_k=4
        )
        random_accuracy = 4 / 20
        assert result["executor"]["accuracy"] >= random_accuracy
