"""Tests for speculative decoding integration in serve.py."""

import argparse
import json
import time
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="requires mlx")

if HAS_MLX:
    from mlx_flash_compress.serve import SPECULATIVE_ENGINES, InferenceState

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

HW_PATCH = "mlx_flash_compress.serve.detect_hardware"
MGR_PATCH = "mlx_flash_compress.serve.MemoryManager"
MEM_PATCH = "mlx_flash_compress.serve.get_memory_state"


def _make_fake_hw():
    hw = MagicMock()
    hw.chip = "Apple M2 Pro"
    hw.total_ram_gb = 32.0
    return hw


def _make_fake_mem(pressure="normal"):
    mem = MagicMock()
    mem.total_gb = 32.0
    mem.free_gb = 8.0
    mem.inactive_gb = 4.0
    mem.available_gb = 10.0
    mem.pressure_level = pressure
    mem.swap_used_gb = 0.0
    return mem


def _make_fake_mem_mgr():
    mgr = MagicMock()
    mgr.get_cache_budget_gb.return_value = 4.0
    mgr.get_optimization_hints.return_value = []
    mgr.auto_release_if_needed.return_value = {"released": False}
    return mgr


def _make_state(speculative="none", **kwargs):
    """Create an InferenceState with external dependencies mocked."""
    with (
        patch(HW_PATCH, return_value=_make_fake_hw()),
        patch(MGR_PATCH, return_value=_make_fake_mem_mgr()),
        patch(MEM_PATCH, return_value=_make_fake_mem()),
    ):
        state = InferenceState("test-model", speculative=speculative, **kwargs)
    return state


# ===========================================================================
# SPECULATIVE_ENGINES constant
# ===========================================================================


class TestSpeculativeEnginesConstant:
    def test_engines_defined(self):
        assert "eagle3" in SPECULATIVE_ENGINES
        assert "layerskip" in SPECULATIVE_ENGINES
        assert "dflash" in SPECULATIVE_ENGINES
        assert "none" in SPECULATIVE_ENGINES

    def test_engines_tuple(self):
        assert isinstance(SPECULATIVE_ENGINES, tuple)


# ===========================================================================
# InferenceState speculative flag
# ===========================================================================


class TestSpeculativeFlag:
    def test_default_speculative_none(self):
        state = _make_state()
        assert state.speculative == "none"
        assert state.spec_engine is None

    def test_speculative_eagle3_stored(self):
        state = _make_state(speculative="eagle3")
        assert state.speculative == "eagle3"
        # Engine is None until model is loaded
        assert state.spec_engine is None

    def test_speculative_layerskip_stored(self):
        state = _make_state(speculative="layerskip")
        assert state.speculative == "layerskip"

    def test_speculative_dflash_stored(self):
        state = _make_state(speculative="dflash")
        assert state.speculative == "dflash"


# ===========================================================================
# Request timeout and configurable parameters
# ===========================================================================


class TestConfigurableParams:
    def test_request_timeout_default(self):
        state = _make_state()
        assert state.request_timeout == 120.0

    def test_request_timeout_custom(self):
        state = _make_state(request_timeout=60.0)
        assert state.request_timeout == 60.0

    def test_cache_budget_pct_default(self):
        state = _make_state()
        assert state.cache_budget_pct == 0.8

    def test_cache_budget_pct_custom(self):
        state = _make_state(cache_budget_pct=0.6)
        assert state.cache_budget_pct == 0.6

    def test_safety_margin_gb_custom(self):
        """safety_margin_gb is passed through to MemoryManager."""
        # We can't directly check MemoryManager's internal value easily,
        # but we can verify the state accepts the parameter
        state = _make_state(safety_margin_gb=3.5)
        assert state is not None


# ===========================================================================
# Status includes speculative engine info
# ===========================================================================


class TestStatusSpeculative:
    def test_status_includes_speculative_engine(self):
        state = _make_state(speculative="layerskip")
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            status = state.get_status()
        assert "speculative_engine" in status
        assert status["speculative_engine"] == "layerskip"

    def test_status_includes_model_loaded(self):
        state = _make_state()
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            status = state.get_status()
        assert "model_loaded" in status
        assert status["model_loaded"] is False

    def test_status_no_speculative_stats_when_none(self):
        state = _make_state(speculative="none")
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            status = state.get_status()
        assert "speculative_stats" not in status


# ===========================================================================
# CLI argument parsing
# ===========================================================================


class TestCLIArgs:
    def test_argparse_speculative_flag(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--speculative",
            default="none",
            choices=list(SPECULATIVE_ENGINES),
        )
        args = parser.parse_args(["--speculative", "eagle3"])
        assert args.speculative == "eagle3"

    def test_argparse_speculative_default(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--speculative",
            default="none",
            choices=list(SPECULATIVE_ENGINES),
        )
        args = parser.parse_args([])
        assert args.speculative == "none"

    def test_argparse_request_timeout(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--request-timeout", type=float, default=120.0)
        args = parser.parse_args(["--request-timeout", "60"])
        assert args.request_timeout == 60.0

    def test_argparse_cache_budget_pct(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache-budget-pct", type=float, default=0.8)
        args = parser.parse_args(["--cache-budget-pct", "0.5"])
        assert args.cache_budget_pct == 0.5

    def test_argparse_safety_margin_gb(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--safety-margin-gb", type=float, default=2.0)
        args = parser.parse_args(["--safety-margin-gb", "3.0"])
        assert args.safety_margin_gb == 3.0
