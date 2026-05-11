"""Tests for hardware detection module."""

import platform
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlx_flash_compress.hardware import (
    CHIP_SPECS,
    MacHardware,
    PerformanceEstimate,
    detect_hardware,
    estimate_performance,
)


class TestMacHardware:
    def test_defaults(self):
        hw = MacHardware()
        assert hw.chip == "Unknown"
        assert hw.chip_family == "Unknown"
        assert hw.chip_tier == "Unknown"
        assert hw.cpu_cores == 0
        assert hw.gpu_cores == 0
        assert hw.total_ram_gb == 0.0

    def test_available_ram_with_zero(self):
        hw = MacHardware(total_ram_gb=0.0)
        assert hw.available_ram_gb == 1.0  # max(0 - 0, 1.0)

    def test_available_ram_small(self):
        hw = MacHardware(total_ram_gb=8.0)
        # os_overhead = min(8 * 0.15, 8) = 1.2
        expected = max(8.0 - 1.2, 1.0)
        assert abs(hw.available_ram_gb - expected) < 0.01

    def test_available_ram_large(self):
        hw = MacHardware(total_ram_gb=192.0)
        # os_overhead = min(192 * 0.15, 8) = 8.0 (capped)
        expected = max(192.0 - 8.0, 1.0)
        assert abs(hw.available_ram_gb - expected) < 0.01

    def test_ssd_latency_with_bandwidth(self):
        hw = MacHardware(estimated_ssd_read_gbs=17.5)
        latency = hw.ssd_latency_ms_per_mb
        assert latency > 0
        assert latency < 1.0  # reasonable for fast SSD

    def test_ssd_latency_zero_bandwidth(self):
        hw = MacHardware(estimated_ssd_read_gbs=0.0)
        assert hw.ssd_latency_ms_per_mb == 0.1  # fallback

    def test_ssd_latency_negative_bandwidth(self):
        hw = MacHardware(estimated_ssd_read_gbs=-1.0)
        assert hw.ssd_latency_ms_per_mb == 0.1  # fallback


class TestChipSpecs:
    def test_all_chips_have_7_fields(self):
        for chip, specs in CHIP_SPECS.items():
            assert len(specs) == 7, f"{chip} has {len(specs)} fields, expected 7"

    def test_known_chips_exist(self):
        assert "M1" in CHIP_SPECS
        assert "M4 Max" in CHIP_SPECS
        assert "M5" in CHIP_SPECS

    def test_bandwidth_positive(self):
        for chip, specs in CHIP_SPECS.items():
            assert specs[0] > 0, f"{chip} memory bandwidth must be > 0"

    def test_gpu_cores_positive(self):
        for chip, specs in CHIP_SPECS.items():
            assert specs[1] > 0, f"{chip} GPU cores must be > 0"


class TestDetectHardware:
    def test_returns_mac_hardware(self):
        hw = detect_hardware()
        assert isinstance(hw, MacHardware)

    def test_has_macos_version(self):
        hw = detect_hardware()
        assert isinstance(hw.macos_version, str)

    @patch("mlx_flash_compress.hardware.subprocess.run")
    def test_handles_system_profiler_timeout(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="system_profiler", timeout=10)
        hw = detect_hardware()
        assert isinstance(hw, MacHardware)
        assert hw.chip == "Unknown"

    @patch("mlx_flash_compress.hardware.subprocess.run")
    def test_handles_invalid_json(self, mock_run):
        import json
        import subprocess

        mock_result = MagicMock()
        mock_result.stdout = "not json"
        mock_run.return_value = mock_result
        # Should not raise
        hw = detect_hardware()
        assert isinstance(hw, MacHardware)


class TestEstimatePerformance:
    def test_model_fits_in_ram(self):
        hw = MacHardware(total_ram_gb=64.0, estimated_ssd_read_gbs=17.5)
        est = estimate_performance(hw, model_gb=5.0, model_name="small")
        assert est.fits_in_ram is True
        assert est.estimated_hit_rate == 1.0
        assert est.bottleneck == "compute"
        assert est.estimated_tok_per_s > 0

    def test_model_exceeds_ram(self):
        hw = MacHardware(total_ram_gb=16.0, estimated_ssd_read_gbs=5.0)
        est = estimate_performance(hw, model_gb=200.0, model_name="huge")
        assert est.fits_in_ram is False
        assert est.estimated_hit_rate < 1.0
        assert est.estimated_tok_per_s > 0

    def test_with_compression(self):
        hw = MacHardware(total_ram_gb=16.0, estimated_ssd_read_gbs=5.0)
        est_no_comp = estimate_performance(hw, model_gb=200.0, compression=1.0)
        est_comp = estimate_performance(hw, model_gb=200.0, compression=1.8)
        # Compression should improve hit rate
        assert est_comp.estimated_hit_rate >= est_no_comp.estimated_hit_rate

    def test_performance_estimate_fields(self):
        hw = MacHardware(total_ram_gb=64.0, estimated_ssd_read_gbs=17.5)
        est = estimate_performance(hw, model_gb=5.0, model_name="test")
        assert est.model_name == "test"
        assert est.model_gb == 5.0
        assert isinstance(est.estimated_layer_ms, float)
