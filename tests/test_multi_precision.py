"""Tests for multi-precision expert quantization tiers."""

import pytest
import numpy as np

from mlx_flash_compress.mixed_precision import (
    ExpertHotness,
    PRECISION_TIERS,
    estimate_tier_savings,
)


class TestPrecisionTiers:
    def test_all_tiers_defined(self):
        expected = {"fp16", "q8", "q6", "q5", "q4", "q3", "q2"}
        assert set(PRECISION_TIERS.keys()) == expected

    def test_tier_bits_descending(self):
        bits = [PRECISION_TIERS[t]["bits"] for t in ["fp16", "q8", "q6", "q5", "q4", "q3", "q2"]]
        assert bits == sorted(bits, reverse=True)

    def test_tier_bytes_descending(self):
        bpp = [PRECISION_TIERS[t]["bytes_per_param"] for t in ["fp16", "q8", "q6", "q5", "q4", "q3", "q2"]]
        assert bpp == sorted(bpp, reverse=True)

    def test_fp16_is_2_bytes(self):
        assert PRECISION_TIERS["fp16"]["bytes_per_param"] == 2.0

    def test_q4_is_half_byte(self):
        assert PRECISION_TIERS["q4"]["bytes_per_param"] == 0.5

    def test_q2_is_quarter_byte(self):
        assert PRECISION_TIERS["q2"]["bytes_per_param"] == 0.25


class TestExpertHotnessClassifyPrecision:
    def test_very_hot_gets_fp16(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 20  # 20%
        assert h.classify_precision(0, 0) == "fp16"

    def test_hot_gets_q8(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 10  # 10%
        assert h.classify_precision(0, 0) == "q8"

    def test_warm_gets_q4(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 6  # 6%
        assert h.classify_precision(0, 0) == "q4"

    def test_cool_gets_q3(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 3  # 3%
        assert h.classify_precision(0, 0) == "q3"

    def test_cold_gets_q2(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 1  # 1%
        assert h.classify_precision(0, 0) == "q2"

    def test_never_activated_gets_q2(self):
        h = ExpertHotness()
        h.total_tokens = 100
        assert h.classify_precision(0, 99) == "q2"

    def test_boundary_15_percent(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 15  # exactly 15%
        assert h.classify_precision(0, 0) == "fp16"

    def test_boundary_8_percent(self):
        h = ExpertHotness()
        h.total_tokens = 100
        h.activation_counts[(0, 0)] = 8  # exactly 8%
        assert h.classify_precision(0, 0) == "q8"


class TestEstimateTierSavings:
    def test_uniform_frequency(self):
        """All experts equally active → mostly Q4."""
        freqs = {i: 0.06 for i in range(100)}
        result = estimate_tier_savings(100, 1000, freqs)
        assert result["tier_counts"]["q4"] == 100

    def test_power_law_distribution(self):
        """Realistic: few hot, many cold."""
        freqs = {}
        for i in range(128):
            if i < 5:
                freqs[i] = 0.20  # top 5 very hot
            elif i < 20:
                freqs[i] = 0.10  # next 15 hot
            elif i < 50:
                freqs[i] = 0.06  # 30 warm
            elif i < 80:
                freqs[i] = 0.03  # 30 cool
            else:
                freqs[i] = 0.01  # 48 cold

        result = estimate_tier_savings(128, 10000, freqs)
        assert result["tier_counts"]["fp16"] == 5
        assert result["tier_counts"]["q8"] == 15
        assert result["tier_counts"]["q4"] == 30
        assert result["tier_counts"]["q3"] == 30
        assert result["tier_counts"]["q2"] == 48

    def test_savings_with_mixed_tiers(self):
        """Mixed tiers should save vs all-Q4 baseline."""
        freqs = {i: 0.01 for i in range(100)}  # all cold
        result = estimate_tier_savings(100, 1000, freqs)
        assert result["savings_ratio"] > 0.4  # Q2 is 50% of Q4

    def test_all_hot_costs_more(self):
        """All FP16 costs more than Q4 baseline."""
        freqs = {i: 0.20 for i in range(10)}
        result = estimate_tier_savings(10, 1000, freqs)
        assert result["tiered_bytes"] > result["baseline_bytes"]

    def test_effective_bits(self):
        freqs = {i: 0.06 for i in range(100)}  # all Q4
        result = estimate_tier_savings(100, 1000, freqs)
        assert abs(result["effective_bits"] - 4.0) < 0.1
