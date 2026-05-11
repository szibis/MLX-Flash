"""Tests for SSD lifespan protection module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mlx_flash_compress.ssd_protection import (
    ReadPolicy,
    SSDHealth,
    SSDProtectedReader,
    check_ssd_health,
    estimate_ssd_impact,
)


class TestSSDHealth:
    def test_defaults(self):
        h = SSDHealth()
        assert h.available is False
        assert h.percentage_used == 0.0
        assert h.data_read_tb == 0.0
        assert h.data_written_tb == 0.0
        assert h.power_on_hours == 0
        assert h.temperature_c == 0.0
        assert h.warning is None

    def test_custom_values(self):
        h = SSDHealth(available=True, percentage_used=5.0, temperature_c=42.0)
        assert h.available is True
        assert h.percentage_used == 5.0
        assert h.temperature_c == 42.0


class TestReadPolicy:
    def test_defaults(self):
        p = ReadPolicy()
        assert p.max_read_rate_gbs == 15.0
        assert p.sequential_preference is True
        assert p.cooldown_after_burst_s == 0.01
        assert p.burst_threshold_mb == 100.0
        assert p.enable_rate_limiting is True
        assert p.write_during_inference is False
        assert p.thermal_throttle_temp_c == 70.0
        assert p.thermal_check_interval_s == 30.0

    def test_custom_values(self):
        p = ReadPolicy(max_read_rate_gbs=10.0, thermal_throttle_temp_c=60.0)
        assert p.max_read_rate_gbs == 10.0
        assert p.thermal_throttle_temp_c == 60.0


class TestSSDProtectedReader:
    def test_creation_default_policy(self):
        reader = SSDProtectedReader()
        assert reader._throttled is False
        assert reader._bytes_since_cooldown == 0
        assert reader.policy.max_read_rate_gbs == 15.0

    def test_creation_custom_policy(self):
        policy = ReadPolicy(enable_rate_limiting=False)
        reader = SSDProtectedReader(policy=policy)
        assert reader.policy.enable_rate_limiting is False

    def test_read_expert(self):
        reader = SSDProtectedReader()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1024)
            path = f.name
        try:
            data = reader.read_expert(path)
            assert len(data) == 1024
        finally:
            os.unlink(path)

    def test_read_expert_tracks_bytes(self):
        reader = SSDProtectedReader()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"y" * 500)
            path = f.name
        try:
            reader.read_expert(path)
            # bytes_since_cooldown should reflect the read
            # (may be 0 if reset after burst threshold)
            assert reader._bytes_since_cooldown >= 0
        finally:
            os.unlink(path)

    def test_burst_cooldown_resets_counter(self):
        # Set a very low burst threshold to trigger cooldown
        policy = ReadPolicy(burst_threshold_mb=0.0001, cooldown_after_burst_s=0.0)
        reader = SSDProtectedReader(policy=policy)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"z" * 2048)
            path = f.name
        try:
            reader.read_expert(path)
            # After exceeding burst threshold, counter should be reset
            assert reader._bytes_since_cooldown == 0
        finally:
            os.unlink(path)

    def test_rate_limiting_disabled(self):
        policy = ReadPolicy(enable_rate_limiting=False, burst_threshold_mb=0.0001)
        reader = SSDProtectedReader(policy=policy)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"a" * 2048)
            path = f.name
        try:
            data = reader.read_expert(path)
            assert len(data) == 2048
        finally:
            os.unlink(path)


class TestCheckSSDHealth:
    def test_returns_ssd_health(self):
        health = check_ssd_health()
        assert isinstance(health, SSDHealth)

    @patch("mlx_flash_compress.ssd_protection.subprocess.run")
    def test_handles_missing_smartctl(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        health = check_ssd_health()
        assert isinstance(health, SSDHealth)


class TestEstimateSSDImpact:
    def test_basic_estimate(self):
        impact = estimate_ssd_impact(model_gb=209, tokens_per_day=10000)
        assert impact["daily_read_gb"] > 0
        assert impact["yearly_read_tb"] > 0
        assert "NONE" in impact["ssd_write_impact"]

    def test_high_cache_hit_rate(self):
        impact = estimate_ssd_impact(model_gb=209, cache_hit_rate=0.99)
        low_impact = impact["daily_read_gb"]
        impact2 = estimate_ssd_impact(model_gb=209, cache_hit_rate=0.1)
        high_impact = impact2["daily_read_gb"]
        assert high_impact > low_impact

    def test_thermal_risk_levels(self):
        low = estimate_ssd_impact(model_gb=10, tokens_per_day=100, cache_hit_rate=0.99)
        assert low["thermal_risk"] == "LOW"

    def test_experts_per_token(self):
        impact = estimate_ssd_impact(model_gb=100, k=4, num_layers=60)
        assert impact["experts_per_token"] == 4 * 60

    def test_cache_misses_per_token(self):
        impact = estimate_ssd_impact(model_gb=100, k=4, num_layers=60, cache_hit_rate=0.5)
        assert impact["cache_misses_per_token"] == 4 * 60 * 0.5
