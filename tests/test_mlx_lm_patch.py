"""Tests for mlx-lm monkey patch integration."""

import sys
import pytest

from mlx_flash_compress.mlx_lm_patch import (
    apply_flash_patch,
    remove_flash_patch,
    is_patched,
    _is_moe_model,
)

try:
    import mlx_lm
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


def _make_model(class_name, named_modules_result=None):
    """Create a mock model with a real class name for type() detection."""
    cls = type(class_name, (), {})
    instance = cls()
    if named_modules_result is not None:
        instance.named_modules = lambda: named_modules_result
    return instance


class TestMoEDetection:
    def test_detect_mixtral_by_name(self):
        model = _make_model("MixtralForCausalLM")
        assert _is_moe_model(model) is True

    def test_detect_dense_model(self):
        model = _make_model("LlamaForCausalLM", named_modules_result=[])
        assert _is_moe_model(model) is False

    def test_detect_by_expert_layer(self):
        model = _make_model("SomeModel", named_modules_result=[
            ("layer.0.expert_gate", None),
        ])
        assert _is_moe_model(model) is True

    def test_detect_by_gate_layer(self):
        model = _make_model("CustomModel", named_modules_result=[
            ("layer.0.gate", None),
        ])
        assert _is_moe_model(model) is True

    def test_deepseek_detection(self):
        model = _make_model("DeepSeekV3Model")
        assert _is_moe_model(model) is True

    def test_no_named_modules(self):
        """Model without named_modules should not crash."""
        model = _make_model("PlainModel")
        assert _is_moe_model(model) is False

    def test_moe_in_class_name(self):
        model = _make_model("SomeMoEModel")
        assert _is_moe_model(model) is True

    def test_switch_transformer(self):
        model = _make_model("SwitchTransformer")
        assert _is_moe_model(model) is True


class TestPatchLifecycle:
    def test_is_patched_initially_false(self):
        assert is_patched() is False

    def test_remove_when_not_patched(self):
        remove_flash_patch()  # Should not raise
        assert is_patched() is False

    @pytest.mark.skipif(not HAS_MLX_LM, reason="mlx-lm not installed")
    def test_apply_and_remove(self):
        apply_flash_patch(ram_budget_gb=8.0)
        assert is_patched() is True
        remove_flash_patch()
        assert is_patched() is False

    @pytest.mark.skipif(HAS_MLX_LM, reason="mlx-lm is installed")
    def test_apply_without_mlx_lm(self):
        """Should raise ImportError if mlx-lm not available."""
        with pytest.raises(ImportError):
            apply_flash_patch()
