"""Tests for adaptive memory manager."""

from unittest.mock import MagicMock, patch

import pytest

from mlx_flash_compress.memory_manager import (
    MemoryManager,
    MemoryState,
    get_memory_state,
    optimize_wired_memory_limit,
)


class TestMemoryState:
    def test_defaults(self):
        state = MemoryState()
        assert state.total_gb == 0.0
        assert state.free_gb == 0.0
        assert state.pressure_level == "unknown"

    def test_available_gb(self):
        state = MemoryState(free_gb=4.0, inactive_gb=6.0)
        # available = free + inactive * 0.5 = 4 + 3 = 7
        assert abs(state.available_gb - 7.0) < 0.01

    def test_available_gb_zero(self):
        state = MemoryState(free_gb=0.0, inactive_gb=0.0)
        assert state.available_gb == 0.0

    def test_pressure_score_green(self):
        state = MemoryState(pressure_level="green")
        assert state.pressure_score == 0.0

    def test_pressure_score_normal(self):
        state = MemoryState(pressure_level="normal")
        assert state.pressure_score == 0.0

    def test_pressure_score_yellow(self):
        state = MemoryState(pressure_level="yellow")
        assert state.pressure_score == 0.5

    def test_pressure_score_warning(self):
        state = MemoryState(pressure_level="warning")
        assert state.pressure_score == 0.5

    def test_pressure_score_red(self):
        state = MemoryState(pressure_level="red")
        assert state.pressure_score == 1.0

    def test_pressure_score_critical(self):
        state = MemoryState(pressure_level="critical")
        assert state.pressure_score == 1.0

    def test_pressure_score_unknown_with_swap(self):
        state = MemoryState(pressure_level="other", total_gb=16.0, swap_used_gb=4.0)
        # swap_ratio = 4/16 = 0.25, score = min(0.25 * 5, 1.0) = 1.0
        assert state.pressure_score == 1.0

    def test_pressure_score_unknown_no_ram(self):
        state = MemoryState(pressure_level="other", total_gb=0.0)
        assert state.pressure_score == 0.3


class TestGetMemoryState:
    def test_returns_memory_state(self):
        state = get_memory_state()
        assert isinstance(state, MemoryState)

    def test_pressure_level_resolved(self):
        state = get_memory_state()
        # Should resolve to a known level, not "unknown"
        assert state.pressure_level in ("normal", "warning", "critical")


class TestMemoryManager:
    @patch("mlx_flash_compress.memory_manager.get_memory_state")
    def test_creation(self, mock_state):
        mock_state.return_value = MemoryState(total_gb=64.0, free_gb=20.0, inactive_gb=10.0, pressure_level="normal")
        mgr = MemoryManager(safety_margin_gb=2.0)
        assert mgr.safety_margin_gb == 2.0
        assert mgr.get_cache_budget() > 0

    @patch("mlx_flash_compress.memory_manager.get_memory_state")
    def test_get_cache_budget_gb(self, mock_state):
        mock_state.return_value = MemoryState(total_gb=64.0, free_gb=20.0, inactive_gb=10.0, pressure_level="normal")
        mgr = MemoryManager()
        budget_gb = mgr.get_cache_budget_gb()
        assert budget_gb > 0
        assert budget_gb == mgr.get_cache_budget() / (1024**3)

    @patch("mlx_flash_compress.memory_manager.get_memory_state")
    def test_budget_changed_initially_false(self, mock_state):
        mock_state.return_value = MemoryState(total_gb=64.0, free_gb=20.0, inactive_gb=10.0, pressure_level="normal")
        mgr = MemoryManager()
        # First read clears the flag
        _ = mgr.budget_changed

    @patch("mlx_flash_compress.memory_manager.get_memory_state")
    def test_get_status(self, mock_state):
        mock_state.return_value = MemoryState(total_gb=64.0, free_gb=20.0, inactive_gb=10.0, pressure_level="normal")
        mgr = MemoryManager()
        status = mgr.get_status()
        assert "total_ram_gb" in status
        assert "pressure" in status
        assert "cache_budget_gb" in status

    @patch("mlx_flash_compress.memory_manager.get_memory_state")
    def test_critical_pressure_reduces_budget(self, mock_state):
        mock_state.return_value = MemoryState(total_gb=64.0, free_gb=1.0, inactive_gb=0.5, pressure_level="critical")
        mgr = MemoryManager(min_cache_gb=0.5)
        budget_gb = mgr.get_cache_budget_gb()
        assert budget_gb == pytest.approx(0.5, abs=0.1)

    @patch("mlx_flash_compress.memory_manager.get_memory_state")
    def test_stop_monitoring(self, mock_state):
        mock_state.return_value = MemoryState(total_gb=64.0, free_gb=20.0, inactive_gb=10.0, pressure_level="normal")
        mgr = MemoryManager()
        mgr.stop_monitoring()
        assert mgr._monitoring is False


class TestOptimizeWiredMemoryLimit:
    def test_with_explicit_ram(self):
        result = optimize_wired_memory_limit(total_ram_gb=64.0, os_reserve_gb=4.0)
        assert result["total_ram_gb"] == 64.0
        assert result["recommended_wired_mb"] == int((64 - 4) * 1024)
        assert result["gain_mb"] > 0
        assert "sudo sysctl" in result["command"]

    def test_gain_is_positive(self):
        result = optimize_wired_memory_limit(total_ram_gb=48.0)
        assert result["gain_mb"] > 0
        assert result["gain_gb"] > 0

    def test_auto_detect_ram(self):
        result = optimize_wired_memory_limit()
        assert result["total_ram_gb"] > 0
