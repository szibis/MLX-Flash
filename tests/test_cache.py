"""Tests for the ExpertCacheManager."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlx_flash_compress.cache import ExpertCacheManager, CacheTier


@pytest.fixture
def expert_dir():
    """Create a temporary directory with synthetic expert files."""
    tmpdir = tempfile.mkdtemp()
    expert_path = Path(tmpdir) / "experts"
    rng = np.random.default_rng(42)

    for layer_idx in range(4):
        layer_dir = expert_path / f"layer_{layer_idx:03d}"
        layer_dir.mkdir(parents=True)
        for expert_id in range(8):
            data = rng.normal(0, 0.02, 512).astype(np.float16)
            path = layer_dir / f"expert_{expert_id:04d}.bin"
            data.tofile(path)

    yield str(expert_path)
    shutil.rmtree(tmpdir)


class TestCacheManager:
    def test_cold_fetch(self, expert_dir):
        """First fetch should be a cold (SSD) hit."""
        cache = ExpertCacheManager(
            expert_dir=expert_dir,
            hot_limit_bytes=1024 * 1024,
            warm_limit_bytes=0,
            num_workers=2,
            enable_hot=False,
            enable_warm=False,
        )
        results = cache.fetch_experts(0, [0, 1], np.float16)
        assert len(results) == 2
        assert results[0][1] == CacheTier.COLD
        assert cache.stats.cold_hits == 2
        cache.shutdown()

    def test_hot_cache_hit(self, expert_dir):
        """After promotion, should get hot cache hits."""
        cache = ExpertCacheManager(
            expert_dir=expert_dir,
            hot_limit_bytes=10 * 1024 * 1024,
            num_workers=2,
            enable_hot=True,
            enable_warm=False,
            promotion_threshold=1,  # immediate promotion
        )

        # First fetch: cold
        cache.fetch_experts(0, [0], np.float16)
        assert cache.stats.cold_hits == 1

        # Wait for async cache population to complete
        cache.flush_pending()

        # Second fetch: should be hot
        cache.fetch_experts(0, [0], np.float16)
        assert cache.stats.hot_hits == 1

        cache.shutdown()

    def test_warm_cache_hit(self, expert_dir):
        """ZSTD warm tier should work."""
        cache = ExpertCacheManager(
            expert_dir=expert_dir,
            hot_limit_bytes=0,
            warm_limit_bytes=10 * 1024 * 1024,
            num_workers=2,
            enable_hot=False,
            enable_warm=True,
            promotion_threshold=1,
        )

        # First fetch: cold
        cache.fetch_experts(0, [0], np.float16)
        assert cache.stats.cold_hits == 1

        # Wait for async cache population
        cache.flush_pending()

        # Second fetch: warm
        cache.fetch_experts(0, [0], np.float16)
        assert cache.stats.warm_hits == 1

        cache.shutdown()

    def test_eviction(self, expert_dir):
        """Cache should evict when limit is reached."""
        # Very small cache: 2KB (only fits ~1 expert)
        cache = ExpertCacheManager(
            expert_dir=expert_dir,
            hot_limit_bytes=2048,
            num_workers=1,
            enable_hot=True,
            enable_warm=False,
            promotion_threshold=1,
        )

        # Fetch multiple experts to trigger eviction
        for eid in range(8):
            cache.fetch_experts(0, [eid], np.float16)
            cache.fetch_experts(0, [eid], np.float16)  # trigger promotion

        assert cache.stats.evictions > 0
        cache.shutdown()

    def test_parallel_fetch(self, expert_dir):
        """Parallel fetch of 4 experts should work correctly."""
        cache = ExpertCacheManager(
            expert_dir=expert_dir,
            hot_limit_bytes=10 * 1024 * 1024,
            num_workers=4,
            enable_hot=True,
            promotion_threshold=1,
        )

        results = cache.fetch_experts(0, [0, 1, 2, 3], np.float16)
        assert len(results) == 4
        for arr, tier in results:
            assert arr.dtype == np.float16
            assert len(arr) > 0

        cache.shutdown()

    def test_stats_tracking(self, expert_dir):
        """Stats should accurately track cache behavior."""
        cache = ExpertCacheManager(
            expert_dir=expert_dir,
            hot_limit_bytes=10 * 1024 * 1024,
            num_workers=2,
            enable_hot=True,
            promotion_threshold=1,
        )

        cache.fetch_experts(0, [0, 1], np.float16)  # 2 cold
        cache.flush_pending()  # wait for async cache inserts
        cache.fetch_experts(0, [0, 1], np.float16)  # 2 hot

        stats = cache.get_stats()
        assert stats.cold_hits == 2
        assert stats.hot_hits == 2
        assert stats.total_hits == 4
        assert stats.hit_rate == 0.5  # 2 hot out of 4 total

        cache.shutdown()
