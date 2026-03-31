"""Tests for advanced prefetching: cross-layer and shadow model predictors."""
import numpy as np
import pytest
from mlx_flash_compress.advanced_prefetch import (
    CrossLayerPredictor,
    ShadowPredictor,
    PrefetchBenchResult,
    benchmark_predictors,
)


class TestCrossLayerPredictor:
    def test_create(self):
        pred = CrossLayerPredictor(num_layers=8, num_experts=16, top_k=4)
        assert pred.lookahead == 3
        assert pred.num_experts == 16

    def test_predict_empty_before_training(self):
        pred = CrossLayerPredictor(num_layers=4, num_experts=8, top_k=2)
        result = pred.predict(0, [0, 1])
        # With no training data, returns empty or default
        assert isinstance(result, list)

    def test_observe_and_predict(self):
        pred = CrossLayerPredictor(num_layers=4, num_experts=8, top_k=2, lookahead=2)
        # Train with consistent pattern: layer 0 expert 0 → layer 1 expert 3
        for _ in range(50):
            pred.observe(0, [0])
            pred.observe(1, [3])
            pred.observe(2, [5])

        # Predict layer 1 from layer 0
        predicted = pred.predict(0, [0])
        assert 3 in predicted  # should learn the association

    def test_predict_multi_returns_multiple_layers(self):
        pred = CrossLayerPredictor(num_layers=6, num_experts=10, top_k=2, lookahead=3)
        # Train
        for _ in range(100):
            pred.observe(0, [0])
            pred.observe(1, [3])
            pred.observe(2, [5])
            pred.observe(3, [7])

        preds = pred.predict_multi(0, [0])
        # Should have predictions for layers 1, 2, and possibly 3
        assert len(preds) >= 1
        assert 1 in preds  # at least layer 1

    def test_lookahead_capped_by_num_layers(self):
        pred = CrossLayerPredictor(num_layers=3, num_experts=8, top_k=2, lookahead=10)
        assert pred.lookahead == 2  # capped to num_layers - 1

    def test_accuracy_perfect(self):
        pred = CrossLayerPredictor(num_layers=4, num_experts=8)
        assert pred.accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_accuracy_partial(self):
        pred = CrossLayerPredictor(num_layers=4, num_experts=8)
        assert pred.accuracy([1, 2], [1, 3]) == 0.5

    def test_stats(self):
        pred = CrossLayerPredictor(num_layers=4, num_experts=8, lookahead=2)
        pred.observe(0, [1])
        s = pred.stats()
        assert s["observations"] == 1
        assert s["lookahead"] == 2

    def test_predict_at_last_layer_returns_empty(self):
        pred = CrossLayerPredictor(num_layers=4, num_experts=8)
        assert pred.predict(3, [0, 1]) == []


class TestShadowPredictor:
    def test_create(self):
        pred = ShadowPredictor(num_layers=8, num_experts=16, top_k=4, hidden_dim=32)
        assert pred.hidden_dim == 32
        assert pred.num_experts == 16

    def test_predict_before_training(self):
        pred = ShadowPredictor(num_layers=4, num_experts=8, top_k=2)
        result = pred.predict(0, [0, 1])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_learns_simple_pattern(self):
        pred = ShadowPredictor(num_layers=4, num_experts=8, top_k=2, lr=0.05)
        # Train: layer 0 expert 0 → layer 1 expert 5
        for _ in range(200):
            pred.observe(0, [0])
            pred.observe(1, [5])

        predicted = pred.predict(0, [0])
        assert 5 in predicted  # should learn the direct mapping

    def test_loss_decreases(self):
        pred = ShadowPredictor(num_layers=4, num_experts=8, top_k=2, lr=0.01)
        # Record loss after 10 steps vs after 100 steps
        for _ in range(10):
            pred.observe(0, [0])
            pred.observe(1, [3])
        early_loss = pred._total_loss / max(pred._training_steps, 1)

        pred2 = ShadowPredictor(num_layers=4, num_experts=8, top_k=2, lr=0.01)
        for _ in range(200):
            pred2.observe(0, [0])
            pred2.observe(1, [3])
        late_loss = pred2._total_loss / max(pred2._training_steps, 1)

        assert late_loss < early_loss  # loss should decrease with training

    def test_stats(self):
        pred = ShadowPredictor(num_layers=4, num_experts=8, hidden_dim=32)
        pred.observe(0, [1])
        pred.observe(1, [2])
        s = pred.stats()
        assert s["training_steps"] == 1
        assert s["hidden_dim"] == 32
        assert s["params_per_layer"] == 8 * 32 * 2  # W1 + W2

    def test_predict_at_last_layer_returns_empty(self):
        pred = ShadowPredictor(num_layers=4, num_experts=8)
        assert pred.predict(3, [0, 1]) == []

    def test_accuracy_method(self):
        pred = ShadowPredictor(num_layers=4, num_experts=8)
        assert pred.accuracy([1, 2, 3], [1, 2, 3]) == 1.0
        assert pred.accuracy([1, 2], [3, 4]) == 0.0
        assert pred.accuracy([1], [1, 2]) == 0.5


class TestBenchmarkPredictors:
    def test_benchmark_runs(self):
        results = benchmark_predictors(
            num_layers=4, num_experts=10, num_tokens=50, top_k=2, seed=42
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, PrefetchBenchResult)
            assert r.total_predictions > 0
            assert 0 <= r.avg_accuracy <= 1.0

    def test_benchmark_deterministic(self):
        r1 = benchmark_predictors(num_layers=4, num_experts=10, num_tokens=50, seed=123)
        r2 = benchmark_predictors(num_layers=4, num_experts=10, num_tokens=50, seed=123)
        for a, b in zip(r1, r2):
            assert a.avg_accuracy == b.avg_accuracy
            assert a.correct_predictions == b.correct_predictions

    def test_shadow_beats_random(self):
        """Shadow predictor should beat random guessing (1/num_experts)."""
        results = benchmark_predictors(
            num_layers=8, num_experts=20, num_tokens=200, top_k=4, seed=42
        )
        shadow = [r for r in results if "shadow" in r.predictor_name][0]
        random_accuracy = 4 / 20  # top_k / num_experts
        assert shadow.avg_accuracy > random_accuracy
