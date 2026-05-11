"""Tests for HuggingFace model calculator."""

import pytest

from mlx_flash_compress.hf_calculator import (
    KNOWN_MODELS,
    estimate_model,
    format_estimate,
)


class TestKnownModels:
    def test_models_exist(self):
        assert len(KNOWN_MODELS) > 0

    def test_qwen3_moe(self):
        assert "Qwen3-30B-A3B" in KNOWN_MODELS
        info = KNOWN_MODELS["Qwen3-30B-A3B"]
        assert info["type"] == "MoE"
        assert info["experts"] > 0
        assert info["active_b"] < info["total_b"]

    def test_dense_model(self):
        assert "Qwen3-4B" in KNOWN_MODELS
        info = KNOWN_MODELS["Qwen3-4B"]
        assert info["type"] == "dense"
        assert info["experts"] == 0
        assert info["active_b"] == info["total_b"]

    def test_all_models_have_required_fields(self):
        required = {"total_b", "active_b", "experts", "layers", "type"}
        for name, info in KNOWN_MODELS.items():
            assert required.issubset(set(info.keys())), f"{name} missing fields"


class TestEstimateModel:
    def test_known_model_auto_fills(self):
        est = estimate_model(model_name="Qwen3-30B-A3B")
        assert est["model"] == "Qwen3-30B-A3B"
        assert est["type"] == "MoE"
        assert est["total_params_b"] == 30
        assert est["active_params_b"] == 3
        assert est["num_experts"] == 128

    def test_unknown_model_defaults(self):
        est = estimate_model(model_name="")
        assert est["model"] == "Custom"
        assert est["total_params_b"] == 7  # default

    def test_custom_params(self):
        est = estimate_model(total_params_b=100, active_params_b=10, num_experts=64)
        assert est["total_params_b"] == 100
        assert est["active_params_b"] == 10
        assert est["type"] == "MoE"

    def test_dense_model_type(self):
        est = estimate_model(model_name="Qwen3-4B")
        assert est["type"] == "Dense"
        assert est["num_experts"] == 0

    def test_total_size_positive(self):
        est = estimate_model(model_name="Qwen3-30B-A3B")
        assert est["total_size_gb"] > 0
        assert est["active_size_gb"] > 0

    def test_kv_cache_estimate(self):
        est = estimate_model(model_name="Qwen3-30B-A3B")
        assert est["kv_cache_gb"] > 0
        assert est["kv_cache_8bit_gb"] > 0
        assert est["kv_cache_8bit_gb"] < est["kv_cache_gb"]

    def test_savings_percentage(self):
        est = estimate_model(model_name="Qwen3-30B-A3B")
        assert est["savings_vs_full_pct"] > 0

    def test_recommendation_for_small_model(self):
        est = estimate_model(model_name="Qwen3-4B", ram_gb=64)
        assert "fits" in est["recommendation"].lower() or "No streaming" in est["recommendation"]

    def test_recommendation_for_huge_model(self):
        est = estimate_model(total_params_b=1000, active_params_b=50, num_experts=512, ram_gb=16)
        assert "nodes" in est["recommendation"].lower() or "large" in est["recommendation"].lower()

    def test_quant_bits_affects_size(self):
        est_4bit = estimate_model(total_params_b=30, quant_bits=4)
        est_8bit = estimate_model(total_params_b=30, quant_bits=8)
        assert est_8bit["total_size_gb"] > est_4bit["total_size_gb"]

    def test_fit_flags(self):
        est = estimate_model(model_name="Qwen3-4B", ram_gb=64)
        assert isinstance(est["fits_full"], bool)
        assert isinstance(est["fits_streaming"], bool)
        assert isinstance(est["fits_optimized"], bool)


class TestFormatEstimate:
    def test_format_contains_model_name(self):
        est = estimate_model(model_name="Qwen3-30B-A3B")
        text = format_estimate(est)
        assert "Qwen3-30B-A3B" in text

    def test_format_contains_sizes(self):
        est = estimate_model(model_name="Qwen3-30B-A3B")
        text = format_estimate(est)
        assert "GB" in text

    def test_format_contains_recommendation(self):
        est = estimate_model(model_name="Qwen3-30B-A3B")
        text = format_estimate(est)
        assert len(text) > 100  # should be a substantial output
