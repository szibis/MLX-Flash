"""Unit tests for CacheTier enum, CacheStats fields, and ExpertCacheManager config validation."""

import pytest

from mlx_flash_compress.cache import CacheStats, CacheTier, ExpertCacheManager


class TestCacheTier:
    def test_hot_exists(self):
        assert CacheTier.HOT is not None

    def test_warm_exists(self):
        assert CacheTier.WARM is not None

    def test_cold_exists(self):
        assert CacheTier.COLD is not None

    def test_enum_values_distinct(self):
        assert CacheTier.HOT != CacheTier.WARM
        assert CacheTier.WARM != CacheTier.COLD
        assert CacheTier.HOT != CacheTier.COLD

    def test_tier_count(self):
        members = list(CacheTier)
        assert len(members) == 3


class TestCacheStats:
    def test_defaults(self):
        stats = CacheStats()
        assert stats.hot_hits == 0
        assert stats.warm_hits == 0
        assert stats.cold_hits == 0
        assert stats.hot_bytes == 0
        assert stats.warm_bytes == 0
        assert stats.total_decompress_ms == 0.0
        assert stats.total_ssd_read_ms == 0.0
        assert stats.evictions == 0

    def test_total_hits(self):
        stats = CacheStats(hot_hits=10, warm_hits=5, cold_hits=3)
        assert stats.total_hits == 18

    def test_hit_rate_all_hot(self):
        stats = CacheStats(hot_hits=10, warm_hits=0, cold_hits=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_cold(self):
        stats = CacheStats(hot_hits=0, warm_hits=0, cold_hits=10)
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        stats = CacheStats(hot_hits=5, warm_hits=3, cold_hits=2)
        assert stats.hit_rate == (5 + 3) / 10

    def test_hit_rate_zero_total(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_summary_keys(self):
        stats = CacheStats(hot_hits=5, warm_hits=3, cold_hits=2)
        summary = stats.summary()
        expected_keys = {
            "total_requests",
            "hot_hits",
            "warm_hits",
            "cold_hits",
            "cache_hit_rate",
            "hot_bytes_mb",
            "warm_bytes_mb",
            "total_decompress_ms",
            "total_ssd_read_ms",
            "evictions",
        }
        assert set(summary.keys()) == expected_keys

    def test_summary_values(self):
        stats = CacheStats(hot_hits=5, warm_hits=3, cold_hits=2, evictions=1)
        summary = stats.summary()
        assert summary["total_requests"] == 10
        assert summary["hot_hits"] == 5
        assert summary["evictions"] == 1


class TestExpertCacheManagerConfig:
    def test_default_creation(self, tmp_path):
        cache = ExpertCacheManager(
            expert_dir=str(tmp_path),
            hot_limit_bytes=1024,
            warm_limit_bytes=512,
        )
        assert cache.hot_limit == 1024
        assert cache.warm_limit == 512
        assert cache.enable_hot is True
        assert cache.enable_warm is True
        cache.shutdown()

    def test_disable_hot(self, tmp_path):
        cache = ExpertCacheManager(
            expert_dir=str(tmp_path),
            enable_hot=False,
        )
        assert cache.enable_hot is False
        cache.shutdown()

    def test_disable_warm(self, tmp_path):
        cache = ExpertCacheManager(
            expert_dir=str(tmp_path),
            enable_warm=False,
        )
        assert cache.enable_warm is False
        cache.shutdown()

    def test_custom_workers(self, tmp_path):
        cache = ExpertCacheManager(
            expert_dir=str(tmp_path),
            num_workers=8,
        )
        assert cache.num_workers == 8
        cache.shutdown()

    def test_promotion_threshold(self, tmp_path):
        cache = ExpertCacheManager(
            expert_dir=str(tmp_path),
            promotion_threshold=5,
        )
        assert cache.promotion_threshold == 5
        cache.shutdown()

    def test_clear_resets_stats(self, tmp_path):
        cache = ExpertCacheManager(expert_dir=str(tmp_path))
        cache.stats.hot_hits = 10
        cache.stats.evictions = 5
        cache.clear()
        assert cache.stats.hot_hits == 0
        assert cache.stats.evictions == 0
        cache.shutdown()

    def test_reset_stats(self, tmp_path):
        cache = ExpertCacheManager(expert_dir=str(tmp_path))
        cache.stats.cold_hits = 100
        cache.reset_stats()
        assert cache.stats.cold_hits == 0
        cache.shutdown()
