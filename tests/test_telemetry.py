"""Tests for hardware telemetry module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from mlx_flash_compress.telemetry import (
    _MAX_HISTORY,
    HardwareTelemetry,
    TelemetrySample,
)


class TestTelemetrySample:
    def test_defaults(self):
        s = TelemetrySample()
        assert s.timestamp == 0.0
        assert s.gpu_util_pct == 0.0
        assert s.gpu_renderer_pct == 0.0
        assert s.gpu_tiler_pct == 0.0
        assert s.gpu_memory_used_gb == 0.0
        assert s.gpu_memory_total_gb == 0.0
        assert s.power_watts == 0.0
        assert s.cpu_temp_c == 0.0
        assert s.gpu_temp_c == 0.0
        assert s.memory_pressure == "normal"
        assert s.memory_used_gb == 0.0
        assert s.memory_total_gb == 0.0
        assert s.ane_util_pct == 0.0

    def test_custom_values(self):
        s = TelemetrySample(
            timestamp=1000.0,
            gpu_util_pct=75.0,
            power_watts=12.5,
            cpu_temp_c=65.0,
            gpu_temp_c=72.0,
            memory_pressure="warn",
        )
        assert s.gpu_util_pct == 75.0
        assert s.power_watts == 12.5
        assert s.cpu_temp_c == 65.0
        assert s.memory_pressure == "warn"

    def test_to_dict(self):
        s = TelemetrySample(timestamp=1.0, gpu_util_pct=50.0)
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d["timestamp"] == 1.0
        assert d["gpu_util_pct"] == 50.0
        assert d["memory_pressure"] == "normal"
        assert len(d) == 13  # all fields


# Reusable mock ioreg output for GPU metrics
MOCK_IOREG_GPU = """
+-o IOAccelerator  <class IOAccelerator>
  |   "Device Utilization %" = 42
  |   "Renderer Utilization %" = 38
  |   "Tiler Utilization %" = 15
  |   "In use system memory"= 2147483648
  |   "Alloc system memory"= 4294967296
"""

MOCK_IOREG_BATTERY = """
+-o AppleSmartBattery  <class AppleSmartBattery>
  |   "InstantAmperage"= 1500
  |   "Voltage"= 12800
"""

MOCK_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               50000.
Pages active:                            200000.
Pages inactive:                          100000.
Pages speculative:                         5000.
Pages throttled:                              0.
Pages wired down:                         80000.
Pages purgeable:                          10000.
"Translation faults":                 500000000.
Pages copy-on-write:                   30000000.
Pages zero filled:                    200000000.
Pages reactivated:                      1000000.
Pages purged:                           2000000.
File-backed pages:                       120000.
Anonymous pages:                          80000.
Pages stored in compressor:               30000.
Pages occupied by compressor:             15000.
"""


class TestHardwareTelemetry:
    def _make_telemetry(self):
        """Create a HardwareTelemetry with mocked total memory detection."""
        with patch.object(HardwareTelemetry, "_detect_total_memory", return_value=32.0):
            return HardwareTelemetry()

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_sample_with_gpu_metrics(self, mock_cmd):
        """Test that GPU metrics are parsed from ioreg output."""

        def side_effect(args, timeout=3.0):
            if "IOAccelerator" in args:
                return MOCK_IOREG_GPU
            return ""

        mock_cmd.side_effect = side_effect
        tel = self._make_telemetry()
        s = tel.sample()

        assert s.gpu_util_pct == 42.0
        assert s.gpu_renderer_pct == 38.0
        assert s.gpu_tiler_pct == 15.0
        assert s.gpu_memory_used_gb == pytest.approx(2.0, abs=0.1)
        assert s.gpu_memory_total_gb == pytest.approx(4.0, abs=0.1)

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_sample_with_battery_power(self, mock_cmd):
        """Test power extraction from battery info."""

        def side_effect(args, timeout=3.0):
            if "AppleSmartBattery" in args:
                return MOCK_IOREG_BATTERY
            return ""

        mock_cmd.side_effect = side_effect
        tel = self._make_telemetry()
        s = tel.sample()

        # 1500 mA * 12800 mV = 19,200,000 / 1,000,000 = 19.2 W
        assert s.power_watts == pytest.approx(19.2, abs=0.1)

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_sample_with_vm_stat(self, mock_cmd):
        """Test memory parsing from vm_stat output."""

        def side_effect(args, timeout=3.0):
            if args == ["vm_stat"]:
                return MOCK_VM_STAT
            if args == ["memory_pressure"]:
                return "System-wide memory free percentage: 50%"
            return ""

        mock_cmd.side_effect = side_effect

        # Also mock psutil as not available
        with patch.dict("sys.modules", {"psutil": None}):
            tel = self._make_telemetry()
            s = tel.sample()

        assert s.memory_total_gb == 32.0
        # used = (active + wired + compressed) * page_size / 1GB
        # = (200000 + 80000 + 15000) * 16384 / 1073741824
        expected_used = (200000 + 80000 + 15000) * 16384 / (1024**3)
        assert s.memory_used_gb == pytest.approx(expected_used, abs=0.5)

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_graceful_failure_all_commands(self, mock_cmd):
        """Test that sample returns zeros when all commands fail."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()
        s = tel.sample()

        assert s.gpu_util_pct == 0.0
        assert s.power_watts == 0.0
        assert s.cpu_temp_c == 0.0
        assert s.gpu_temp_c == 0.0
        assert s.memory_pressure == "normal"
        assert s.timestamp > 0  # timestamp always set

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_circular_buffer_history(self, mock_cmd):
        """Test that history is bounded by _MAX_HISTORY."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()

        # Insert more samples than buffer size
        for i in range(_MAX_HISTORY + 20):
            tel.sample()

        history = tel.get_history(seconds=9999)
        assert len(history) == _MAX_HISTORY

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_get_history_time_filter(self, mock_cmd):
        """Test that get_history respects the seconds parameter."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()

        # Manually insert samples with old timestamps
        old_sample = TelemetrySample(timestamp=time.time() - 300)  # 5 min ago
        recent_sample = TelemetrySample(timestamp=time.time() - 10)  # 10s ago
        tel._history.append(old_sample)
        tel._history.append(recent_sample)

        history = tel.get_history(seconds=60)
        assert len(history) == 1  # only the recent one

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_get_stats_empty(self, mock_cmd):
        """Test get_stats with no samples."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()
        stats = tel.get_stats()

        assert stats["samples_count"] == 0
        assert "current" in stats
        assert stats["current"]["gpu_util_pct"] == 0.0

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_get_stats_aggregation(self, mock_cmd):
        """Test that get_stats computes correct min/max/avg."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()

        # Insert samples with known values
        for gpu_pct in [10.0, 20.0, 30.0, 40.0, 50.0]:
            s = TelemetrySample(timestamp=time.time(), gpu_util_pct=gpu_pct)
            tel._history.append(s)

        stats = tel.get_stats()
        assert stats["samples_count"] == 5
        assert stats["avg"]["gpu_util_pct"] == 30.0
        assert stats["max"]["gpu_util_pct"] == 50.0
        assert stats["min"]["gpu_util_pct"] == 10.0

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_start_stop_sampling(self, mock_cmd):
        """Test background sampling thread lifecycle."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()

        assert not tel._sampling
        tel.start_sampling(interval_ms=50)
        assert tel._sampling
        assert tel._thread is not None
        assert tel._thread.is_alive()

        # Let it run a few cycles
        time.sleep(0.2)
        tel.stop_sampling()

        assert not tel._sampling
        # Should have collected some samples
        assert len(tel._history) > 0

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_start_sampling_idempotent(self, mock_cmd):
        """Test that calling start_sampling twice doesn't create two threads."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()

        tel.start_sampling(interval_ms=100)
        thread1 = tel._thread
        tel.start_sampling(interval_ms=100)
        thread2 = tel._thread

        assert thread1 is thread2
        tel.stop_sampling()

    @patch("mlx_flash_compress.telemetry.HardwareTelemetry._run_cmd")
    def test_sample_thread_safety(self, mock_cmd):
        """Test concurrent access to sample history."""
        mock_cmd.return_value = ""
        tel = self._make_telemetry()

        import threading

        errors = []

        def reader():
            try:
                for _ in range(50):
                    tel.get_history(seconds=120)
                    tel.get_stats()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(50):
                    tel.sample()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_extract_ioreg_int(self):
        """Test the ioreg integer extraction helper."""
        tel = self._make_telemetry()

        text = '  "Device Utilization %" = 42\n  "Renderer Utilization %" = 38'
        assert tel._extract_ioreg_int(text, "Device Utilization %") == 42
        assert tel._extract_ioreg_int(text, "Renderer Utilization %") == 38
        assert tel._extract_ioreg_int(text, "NonExistent") == 0

    def test_extract_ioreg_int_large_value(self):
        """Test extraction of large byte values."""
        tel = self._make_telemetry()
        text = '  "In use system memory"= 2147483648'
        assert tel._extract_ioreg_int(text, '"In use system memory"=') == 2147483648

    def test_detect_total_memory(self):
        """Test total memory detection via sysctl."""
        with patch.object(
            HardwareTelemetry,
            "_run_cmd",
            return_value="34359738368\n",
        ):
            tel = HardwareTelemetry()
            assert tel._total_memory_gb == pytest.approx(32.0, abs=0.1)

    def test_detect_total_memory_failure(self):
        """Test graceful failure when sysctl is unavailable."""
        with patch.object(HardwareTelemetry, "_run_cmd", return_value=""):
            tel = HardwareTelemetry()
            assert tel._total_memory_gb == 0.0
