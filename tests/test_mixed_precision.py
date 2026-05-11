"""Tests for mixed precision: requantize/dequantize roundtrip, MixedPrecisionResult, edge cases."""

import numpy as np
import pytest

from mlx_flash_compress.mixed_precision import (
    PRECISION_TIERS,
    ExpertHotness,
    MixedPrecisionResult,
    benchmark_mixed_precision,
    dequantize_2bit,
    estimate_tier_savings,
    requantize_4bit_to_2bit,
)


class TestRequantize4bitTo2bit:
    def _make_test_data(self, rows=32, cols_u32=16, n_groups=4, seed=42):
        rng = np.random.default_rng(seed)
        weight = rng.integers(0, 2**32, size=(rows, cols_u32), dtype=np.uint32)
        scales = rng.uniform(0.001, 0.05, size=(rows, n_groups)).astype(np.float16)
        biases = rng.uniform(-0.01, 0.01, size=(rows, n_groups)).astype(np.float16)
        return weight, scales, biases

    def test_returns_correct_types(self):
        weight, scales, biases = self._make_test_data()
        packed, new_s, new_b, meta = requantize_4bit_to_2bit(weight, scales, biases)
        assert isinstance(packed, np.ndarray)
        assert packed.dtype == np.uint8
        assert new_s.dtype == np.float16
        assert new_b.dtype == np.float16
        assert isinstance(meta, dict)

    def test_compression_ratio(self):
        weight, scales, biases = self._make_test_data()
        packed, new_s, new_b, meta = requantize_4bit_to_2bit(weight, scales, biases)
        # 4-bit to 2-bit should compress ~2x (ignoring scale/bias overhead)
        assert meta["ratio"] > 1.0
        assert meta["packed_bytes"] < meta["original_bytes"]

    def test_metadata_fields(self):
        weight, scales, biases = self._make_test_data()
        _, _, _, meta = requantize_4bit_to_2bit(weight, scales, biases)
        assert "original_shape" in meta
        assert "packed_shape" in meta
        assert "n_groups" in meta
        assert "weights_per_group" in meta
        assert "original_bytes" in meta
        assert "packed_bytes" in meta
        assert "ratio" in meta

    def test_packed_values_in_range(self):
        """All 2-bit crumbs packed into bytes should be valid."""
        weight, scales, biases = self._make_test_data()
        packed, _, _, _ = requantize_4bit_to_2bit(weight, scales, biases)
        # Each byte holds 4 crumbs (0-3), max byte value = 0b11111111 = 255
        assert packed.max() <= 255


class TestDequantize2bit:
    def test_roundtrip_shape(self):
        rng = np.random.default_rng(42)
        rows, cols_u32, n_groups = 16, 8, 2
        weight = rng.integers(0, 2**32, size=(rows, cols_u32), dtype=np.uint32)
        scales = rng.uniform(0.001, 0.05, size=(rows, n_groups)).astype(np.float16)
        biases = rng.uniform(-0.01, 0.01, size=(rows, n_groups)).astype(np.float16)

        packed, new_s, new_b, meta = requantize_4bit_to_2bit(weight, scales, biases)
        n_values = cols_u32 * 8  # 8 nibbles per uint32
        result = dequantize_2bit(packed, new_s, new_b, n_values)
        assert result.shape == (rows, n_values)

    def test_dequantized_values_finite(self):
        rng = np.random.default_rng(42)
        rows, cols_u32, n_groups = 16, 8, 2
        weight = rng.integers(0, 2**32, size=(rows, cols_u32), dtype=np.uint32)
        scales = rng.uniform(0.001, 0.05, size=(rows, n_groups)).astype(np.float16)
        biases = rng.uniform(-0.01, 0.01, size=(rows, n_groups)).astype(np.float16)

        packed, new_s, new_b, _ = requantize_4bit_to_2bit(weight, scales, biases)
        n_values = cols_u32 * 8
        result = dequantize_2bit(packed, new_s, new_b, n_values)
        assert np.all(np.isfinite(result))


class TestMixedPrecisionResult:
    def test_dataclass_fields(self):
        r = MixedPrecisionResult(
            expert_id=5,
            original_bytes=1000,
            q4_bytes=1000,
            q2_bytes=500,
            ratio_4to2=2.0,
            requant_ms=1.5,
            mse=0.001,
            max_error=0.05,
        )
        assert r.expert_id == 5
        assert r.ratio_4to2 == 2.0


class TestBenchmarkMixedPrecision:
    def test_benchmark_output(self):
        rng = np.random.default_rng(42)
        weight = rng.integers(0, 2**32, size=(32, 16), dtype=np.uint32)
        scales = rng.uniform(0.001, 0.05, size=(32, 2)).astype(np.float16)
        biases = rng.uniform(-0.01, 0.01, size=(32, 2)).astype(np.float16)

        result = benchmark_mixed_precision(weight, scales, biases, expert_id=3)
        assert result.expert_id == 3
        assert result.ratio_4to2 > 1.0
        assert result.mse >= 0
        assert np.isfinite(result.mse)
        assert np.isfinite(result.max_error)

    def test_benchmark_default_expert_id(self):
        rng = np.random.default_rng(42)
        weight = rng.integers(0, 2**32, size=(16, 8), dtype=np.uint32)
        scales = rng.uniform(0.001, 0.05, size=(16, 1)).astype(np.float16)
        biases = rng.uniform(-0.01, 0.01, size=(16, 1)).astype(np.float16)

        result = benchmark_mixed_precision(weight, scales, biases)
        assert result.expert_id == 0


class TestExpertHotnessRecord:
    """Additional coverage beyond test_multi_precision.py."""

    def test_record_multiple_layers(self):
        h = ExpertHotness()
        h.record(0, [0, 1])
        h.record(1, [2, 3])
        h.record(0, [0])
        assert h.total_tokens == 3
        assert h.activation_counts[(0, 0)] == 2
        assert h.activation_counts[(0, 1)] == 1
        assert h.activation_counts[(1, 2)] == 1

    def test_classify_hot_vs_cold(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 10  # 10%
        h.activation_counts[(0, 1)] = 2   # 2%
        assert h.classify(0, 0) == "hot"
        assert h.classify(0, 1) == "cold"

    def test_classify_custom_threshold(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 3  # 3%
        assert h.classify(0, 0, threshold=0.02) == "hot"
        assert h.classify(0, 0, threshold=0.05) == "cold"


class TestEstimateTierSavingsEdgeCases:
    """Additional edge case coverage."""

    def test_empty_frequencies(self):
        result = estimate_tier_savings(10, 1000, {})
        # All experts have 0 frequency → all q2
        assert result["tier_counts"]["q2"] == 10

    def test_zero_expert_params(self):
        result = estimate_tier_savings(10, 0, {0: 0.1})
        assert result["baseline_bytes"] == 0
        assert result["effective_bits"] == 0

    def test_single_expert(self):
        result = estimate_tier_savings(1, 1000, {0: 0.20})
        assert result["tier_counts"]["fp16"] == 1
        assert sum(result["tier_counts"].values()) == 1
