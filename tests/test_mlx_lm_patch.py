"""Tests for mlx-lm monkey patch integration."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from mlx_flash_compress.mlx_lm_patch import (
    apply_flash_patch,
    remove_flash_patch,
    is_patched,
    _is_moe_model,
)


class TestMoEDetection:
    def test_detect_mixtral_by_name(self):
        model = MagicMock()
        model.__class__.__name__ = "MixtralForCausalLM"
        assert _is_moe_model(model) is True

    def test_detect_dense_model(self):
        model = MagicMock()
        model.__class__.__name__ = "LlamaForCausalLM"
        # No named_modules with expert
        model.named_modules = MagicMock(return_value=[])
        assert _is_moe_model(model) is False

    def test_detect_by_expert_layer(self):
        model = MagicMock()
        model.__class__.__name__ = "SomeModel"
        model.named_modules = MagicMock(return_value=[
            ("layer.0.expert_gate", MagicMock()),
        ])
        assert _is_moe_model(model) is True

    def test_detect_by_gate_layer(self):
        model = MagicMock()
        model.__class__.__name__ = "CustomModel"
        model.named_modules = MagicMock(return_value=[
            ("layer.0.gate", MagicMock()),
        ])
        assert _is_moe_model(model) is True

    def test_deepseek_detection(self):
        model = MagicMock()
        model.__class__.__name__ = "DeepSeekV3Model"
        assert _is_moe_model(model) is True


class TestPatchLifecycle:
    def test_is_patched_initially_false(self):
        assert is_patched() is False

    def test_remove_when_not_patched(self):
        remove_flash_patch()  # Should not raise
        assert is_patched() is False

    @pytest.mark.skipif(
        not pytest.importorskip("mlx_lm", reason="mlx-lm not installed"),
        reason="mlx-lm required"
    )
    def test_apply_and_remove(self):
        apply_flash_patch(ram_budget_gb=8.0)
        assert is_patched() is True
        remove_flash_patch()
        assert is_patched() is False

    def test_apply_without_mlx_lm(self):
        """Should raise ImportError if mlx-lm not available."""
        # Only test if mlx-lm is NOT installed
        try:
            import mlx_lm
            pytest.skip("mlx-lm is installed")
        except ImportError:
            with pytest.raises(ImportError):
                apply_flash_patch()
