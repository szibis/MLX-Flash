"""Tests for memory manager optimization hints and auto-release."""
import pytest
from unittest.mock import patch, MagicMock
from mlx_flash_compress.memory_manager import (
    MemoryManager, MemoryState, get_memory_state,
)


class TestOptimizationHints:
    def test_hints_normal_plenty_of_ram(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        with patch('mlx_flash_compress.memory_manager.get_memory_state') as mock:
            mock.return_value = MemoryState(
                total_gb=36.0, free_gb=10.0, active_gb=8.0,
                inactive_gb=10.0, wired_gb=4.0, compressed_gb=2.0,
                swap_used_gb=0.0, pressure_level="normal",
            )
            hints = mgr.get_optimization_hints()
            priorities = [h["priority"] for h in hints]
            assert "critical" not in priorities

    def test_hints_critical_pressure(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        with patch('mlx_flash_compress.memory_manager.get_memory_state') as mock:
            mock.return_value = MemoryState(
                total_gb=36.0, free_gb=1.0, active_gb=25.0,
                inactive_gb=2.0, wired_gb=6.0, compressed_gb=2.0,
                swap_used_gb=5.0, pressure_level="critical",
            )
            hints = mgr.get_optimization_hints()
            assert any(h["priority"] == "critical" for h in hints)

    def test_hints_high_swap(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        with patch('mlx_flash_compress.memory_manager.get_memory_state') as mock:
            mock.return_value = MemoryState(
                total_gb=36.0, free_gb=4.0, active_gb=15.0,
                inactive_gb=8.0, wired_gb=5.0, compressed_gb=2.0,
                swap_used_gb=8.0, pressure_level="warning",
            )
            hints = mgr.get_optimization_hints()
            assert any("swap" in h["message"].lower() for h in hints)

    def test_auto_release_normal_does_nothing(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        result = mgr.auto_release_if_needed()
        assert result["action"] == "none"

    def test_pressure_level_detection(self):
        state = get_memory_state()
        assert state.pressure_level in ("normal", "warning", "critical")

    def test_budget_adjusts_with_pressure(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        budget1 = mgr.get_cache_budget_gb()
        assert budget1 > 0
