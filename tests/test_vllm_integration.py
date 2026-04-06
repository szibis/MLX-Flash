"""Tests for vllm-mlx integration adapter."""
import pytest

try:
    from mlx_flash_compress.vllm_integration import (
        get_model_info,
        auto_configure,
    )
    HAS_MODULE = True
except (ImportError, ModuleNotFoundError):
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="vllm_integration requires mlx")


class TestGetModelInfo:
    def test_no_model(self):
        class FakeModel:
            pass
        info = get_model_info(FakeModel())
        assert not info["is_moe"]

    def test_model_with_layers_no_moe(self):
        class FakeLayer:
            pass
        class FakeModel:
            class model:
                layers = [FakeLayer(), FakeLayer()]
        info = get_model_info(FakeModel())
        assert not info["is_moe"]
        assert info["num_layers"] == 2


class TestAutoConfigure:
    def test_dense_model(self):
        class FakeLayer:
            pass
        class FakeModel:
            class model:
                layers = [FakeLayer()]
        cfg = auto_configure(FakeModel(), ram_gb=36)
        assert cfg["capacity_per_layer"] == 0
        assert "dense" in cfg["note"].lower() or "Dense" in cfg["note"]

    def test_returns_dict(self):
        class FakeModel:
            pass
        cfg = auto_configure(FakeModel(), ram_gb=36)
        assert isinstance(cfg, dict)
        assert "capacity_per_layer" in cfg
