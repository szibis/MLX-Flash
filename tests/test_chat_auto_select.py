"""Tests for chat model auto-selection and Gemma 4 catalog."""

import pytest

try:
    from mlx_flash_compress.chat import auto_select_model, MODELS
    HAS_CHAT = True
except ImportError:
    HAS_CHAT = False

pytestmark = pytest.mark.skipif(not HAS_CHAT, reason="mlx not installed")


class TestAutoSelectModel:
    def test_8gb_gets_gemma4_e4b(self):
        model = auto_select_model(8.0)
        assert "gemma-4-E4B" in model or "gemma-4-E2B" in model

    def test_16gb_gets_gemma4(self):
        model = auto_select_model(16.0)
        # 16GB * 0.70 = 11.2GB budget → E4B (2.8GB) fits, 26B MoE (15GB) doesn't
        assert "gemma-4" in model

    def test_24gb_gets_gemma4_31b_or_26b(self):
        model = auto_select_model(24.0)
        # 24GB * 0.70 = 16.8GB budget → 26B MoE (15GB) fits, 31B (20GB) doesn't
        assert "gemma-4-26b" in model or "gemma-4-31b" in model

    def test_32gb_gets_gemma4_31b(self):
        model = auto_select_model(32.0)
        # 32GB * 0.70 = 22.4GB budget → 31B (20GB) fits
        assert "gemma-4-31b" in model

    def test_48gb_gets_gemma4_31b(self):
        model = auto_select_model(48.0)
        assert "gemma-4-31b" in model

    def test_4gb_gets_smallest(self):
        model = auto_select_model(4.0)
        # 4GB * 0.70 = 2.8GB → E4B just barely fits, E2B definitely fits
        assert "gemma-4" in model

    def test_1gb_gets_e2b_fallback(self):
        model = auto_select_model(1.0)
        assert "gemma-4-E2B" in model


class TestModelCatalog:
    def test_gemma4_models_in_catalog(self):
        names = [m[0] for m in MODELS]
        gemma4_models = [n for n in names if "gemma-4" in n]
        assert len(gemma4_models) >= 3  # E2B, E4B, 26B, 31B

    def test_catalog_has_e2b(self):
        names = [m[0] for m in MODELS]
        assert any("gemma-4-E2B" in n for n in names)

    def test_catalog_has_e4b(self):
        names = [m[0] for m in MODELS]
        assert any("gemma-4-E4B" in n for n in names)

    def test_catalog_has_26b_moe(self):
        names = [m[0] for m in MODELS]
        assert any("gemma-4-26b" in n for n in names)

    def test_catalog_has_31b(self):
        names = [m[0] for m in MODELS]
        assert any("gemma-4-31b" in n for n in names)

    def test_all_models_have_six_fields(self):
        for model in MODELS:
            assert len(model) == 6, f"Model {model[0]} has {len(model)} fields, expected 6"

    def test_model_sizes_positive(self):
        for name, _, _, size, _, _ in MODELS:
            assert size > 0, f"Model {name} has non-positive size"

    def test_model_types_valid(self):
        valid_types = {"dense", "MoE"}
        for name, _, _, _, mtype, _ in MODELS:
            assert mtype in valid_types, f"Model {name} has invalid type {mtype}"
