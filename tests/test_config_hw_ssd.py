"""Tests for config, hardware detection, and SSD protection."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from mlx_flash_compress.config import FlashConfig, get_config, CacheConfig
from mlx_flash_compress.hardware import detect_hardware, MacHardware, estimate_performance
from mlx_flash_compress.ssd_protection import (
    SSDProtectedReader, ReadPolicy, estimate_ssd_impact, check_ssd_health,
)


class TestFlashConfig:
    def test_auto_detect(self):
        cfg = FlashConfig.auto_detect()
        assert cfg.cache.enable is True
        assert cfg.cache.ram_mb > 0
        assert cfg.engine.backend in ("python", "c_gcd")
        assert cfg.detected_ram_gb > 0

    def test_defaults(self):
        cfg = FlashConfig()
        assert cfg.cache.eviction == "lcp"
        assert cfg.cache.hot_algo == "lz4"
        assert cfg.prefetch.workers == 2
        assert cfg.mixed_precision.cold_bits == 2
        assert cfg.skip_fallback.enable is False
        assert cfg.ssd_protection.enable is True

    def test_from_json(self):
        data = {
            "cache": {"ram_mb": 1024, "eviction": "lfu"},
            "prefetch": {"enable": False},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = FlashConfig.from_file(f.name)
        os.unlink(f.name)

        assert cfg.cache.ram_mb == 1024
        assert cfg.cache.eviction == "lfu"
        assert cfg.prefetch.enable is False

    def test_from_env(self):
        os.environ["FLASH_CACHE_RAM_MB"] = "2048"
        os.environ["FLASH_ENABLE_PREFETCH"] = "0"
        try:
            cfg = FlashConfig.from_env()
            assert cfg.cache.ram_mb == 2048
            assert cfg.prefetch.enable is False
        finally:
            del os.environ["FLASH_CACHE_RAM_MB"]
            del os.environ["FLASH_ENABLE_PREFETCH"]

    def test_to_json_roundtrip(self):
        cfg = FlashConfig.auto_detect()
        json_str = cfg.to_json()
        parsed = json.loads(json_str)
        assert parsed["cache"]["eviction"] == "lcp"
        assert parsed["ssd_protection"]["enable"] is True

    def test_summary(self):
        cfg = FlashConfig.auto_detect()
        s = cfg.summary()
        assert "Cache" in s
        assert "ON" in s or "OFF" in s

    def test_save_load(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            cfg = FlashConfig.auto_detect()
            cfg.cache.ram_mb = 999
            cfg.save(path)

            loaded = FlashConfig.from_file(path)
            assert loaded.cache.ram_mb == 999
        finally:
            os.unlink(path)


class TestHardwareDetection:
    def test_detect(self):
        hw = detect_hardware()
        assert hw.chip != "Unknown"
        assert hw.total_ram_gb > 0
        assert hw.cpu_cores > 0

    def test_available_ram(self):
        hw = detect_hardware()
        assert hw.available_ram_gb > 0
        assert hw.available_ram_gb < hw.total_ram_gb

    def test_ssd_latency(self):
        hw = detect_hardware()
        assert hw.ssd_latency_ms_per_mb > 0

    def test_estimate_small_model(self):
        hw = detect_hardware()
        est = estimate_performance(hw, model_gb=5.0, model_name="small")
        assert est.fits_in_ram is True
        assert est.estimated_tok_per_s > 0
        assert est.estimated_hit_rate == 1.0

    def test_estimate_large_model(self):
        hw = detect_hardware()
        est = estimate_performance(hw, model_gb=500.0, model_name="huge")
        assert est.fits_in_ram is False
        assert est.estimated_hit_rate < 1.0
        assert est.estimated_tok_per_s > 0
        assert est.bottleneck in ("compute", "memory_bandwidth", "ssd_bandwidth")


class TestSSDProtection:
    def test_read_policy_defaults(self):
        p = ReadPolicy()
        assert p.max_read_rate_gbs > 0
        assert p.thermal_throttle_temp_c == 70.0
        assert p.write_during_inference is False

    def test_protected_reader(self):
        reader = SSDProtectedReader()
        assert reader._throttled is False

        # Write a temp file and read it
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data" * 100)
            path = f.name

        try:
            data = reader.read_expert(path)
            assert len(data) == 900
        finally:
            os.unlink(path)

    def test_ssd_impact_estimate(self):
        impact = estimate_ssd_impact(
            model_gb=209, tokens_per_day=10000,
            cache_hit_rate=0.7, k=4, num_layers=60,
        )
        assert impact["daily_read_gb"] > 0
        assert "NONE" in impact["ssd_write_impact"]
        assert impact["thermal_risk"] in ("LOW", "MODERATE", "HIGH")

    def test_health_check(self):
        health = check_ssd_health()
        # May or may not work depending on permissions
        assert isinstance(health.available, bool)
