"""Unit tests for mlx-lm patch detection and version checking logic."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from mlx_flash_compress.mlx_lm_patch import (
    _is_moe_model,
    _patch_active,
    is_patched,
    remove_flash_patch,
)


class TestIsMoEModel:
    def test_mixtral_by_class_name(self):
        model = type("MixtralForCausalLM", (), {})()
        assert _is_moe_model(model) is True

    def test_deepseek_by_class_name(self):
        model = type("DeepSeekV3ForCausalLM", (), {})()
        assert _is_moe_model(model) is True

    def test_switch_by_class_name(self):
        model = type("SwitchTransformerModel", (), {})()
        assert _is_moe_model(model) is True

    def test_moe_by_class_name(self):
        model = type("CustomMoEModel", (), {})()
        assert _is_moe_model(model) is True

    def test_dense_model(self):
        model = type("LlamaForCausalLM", (), {"named_modules": lambda self: []})()
        assert _is_moe_model(model) is False

    def test_expert_in_named_modules(self):
        model = type("Model", (), {"named_modules": lambda self: [("layer.0.expert", None)]})()
        assert _is_moe_model(model) is True

    def test_gate_in_named_modules(self):
        model = type("Model", (), {"named_modules": lambda self: [("layer.0.gate", None)]})()
        assert _is_moe_model(model) is True

    def test_no_named_modules(self):
        model = type("PlainModel", (), {})()
        assert _is_moe_model(model) is False

    def test_named_modules_not_callable(self):
        model = type("Model", (), {"named_modules": "not callable"})()
        assert _is_moe_model(model) is False

    def test_empty_named_modules(self):
        model = type("Model", (), {"named_modules": lambda self: []})()
        assert _is_moe_model(model) is False


class TestPatchLifecycle:
    def test_is_patched_returns_bool(self):
        assert isinstance(is_patched(), bool)

    def test_remove_when_not_patched(self):
        remove_flash_patch()
        assert is_patched() is False

    def test_remove_idempotent(self):
        remove_flash_patch()
        remove_flash_patch()
        assert is_patched() is False
