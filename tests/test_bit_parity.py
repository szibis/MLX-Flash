"""Tests for bit-parity verification."""

import pytest

from mlx_flash_compress.bit_parity import (
    ParityResult,
    fp32_linear,
    compare_logits,
    verify_parity,
)


class TestParityResult:
    def test_bit_perfect(self):
        r = ParityResult(
            model_name="test", prompt="hi", tokens_compared=5,
            max_delta=0.0, mean_delta=0.0, bit_perfect=True,
        )
        assert r.parity_grade == "BIT-PERFECT"

    def test_near_perfect(self):
        r = ParityResult(
            model_name="test", prompt="hi", tokens_compared=5,
            max_delta=1e-7, mean_delta=1e-8, bit_perfect=False,
        )
        assert r.parity_grade == "NEAR-PERFECT"

    def test_acceptable(self):
        r = ParityResult(
            model_name="test", prompt="hi", tokens_compared=5,
            max_delta=1e-4, mean_delta=1e-5, bit_perfect=False,
        )
        assert r.parity_grade == "ACCEPTABLE"

    def test_divergent(self):
        r = ParityResult(
            model_name="test", prompt="hi", tokens_compared=5,
            max_delta=0.1, mean_delta=0.05, bit_perfect=False,
        )
        assert r.parity_grade == "DIVERGENT"


class TestFP32Linear:
    @pytest.mark.skipif(
        not pytest.importorskip("mlx", reason="MLX not available"),
        reason="MLX required"
    )
    def test_basic_matmul(self):
        import mlx.core as mx
        w = mx.ones((4, 3))
        x = mx.ones((1, 3))
        result = fp32_linear(w, x)
        assert result.shape == (1, 4)
        # Each row of w is [1,1,1], x is [1,1,1], dot = 3.0
        expected = mx.full((1, 4), 3.0)
        assert float(mx.max(mx.abs(result - expected))) < 1e-6

    @pytest.mark.skipif(
        not pytest.importorskip("mlx", reason="MLX not available"),
        reason="MLX required"
    )
    def test_with_bias(self):
        import mlx.core as mx
        w = mx.ones((4, 3))
        x = mx.ones((1, 3))
        bias = mx.full((4,), 1.0)
        result = fp32_linear(w, x, bias=bias)
        expected = mx.full((1, 4), 4.0)  # 3.0 + 1.0
        assert float(mx.max(mx.abs(result - expected))) < 1e-6


class TestCompareLogits:
    @pytest.mark.skipif(
        not pytest.importorskip("mlx", reason="MLX not available"),
        reason="MLX required"
    )
    def test_identical(self):
        import mlx.core as mx
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([1.0, 2.0, 3.0])
        result = compare_logits(a, b)
        assert result["max_delta"] == 0.0
        assert result["nonzero_count"] == 0

    @pytest.mark.skipif(
        not pytest.importorskip("mlx", reason="MLX not available"),
        reason="MLX required"
    )
    def test_different(self):
        import mlx.core as mx
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([1.1, 2.0, 3.0])
        result = compare_logits(a, b)
        assert result["max_delta"] > 0.09
        assert result["nonzero_count"] == 1


class TestVerifyParity:
    def test_result_structure(self):
        """ParityResult has correct fields and grade logic."""
        result = ParityResult(
            model_name="test",
            prompt="hello",
            tokens_compared=0,
            max_delta=float("inf"),
            mean_delta=float("inf"),
            bit_perfect=False,
        )
        assert isinstance(result, ParityResult)
        assert result.parity_grade == "DIVERGENT"
        assert result.fp32_accumulation is True
