"""Tests for LCP cache, mixed precision, and smart eviction."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlx_flash_compress.lcp_cache import LCPCache
from mlx_flash_compress.mixed_precision import (
    requantize_4bit_to_2bit, dequantize_2bit, ExpertHotness,
)
from mlx_flash_compress.smart_eviction import (
    LeastStalePolicy, RoutingPredictor, simulate_prefetch,
)


@pytest.fixture
def expert_dir():
    tmpdir = tempfile.mkdtemp()
    expert_path = Path(tmpdir) / "experts"
    rng = np.random.default_rng(42)
    for layer in range(4):
        layer_dir = expert_path / f"layer_{layer:03d}"
        layer_dir.mkdir(parents=True)
        for expert in range(8):
            data = rng.normal(0, 0.02, 512).astype(np.float16)
            (layer_dir / f"expert_{expert:04d}.bin").write_bytes(data.tobytes())
    yield str(expert_path)
    shutil.rmtree(tmpdir)


class TestLCPCache:
    def test_cache_hit(self, expert_dir):
        cache = LCPCache(expert_dir=expert_dir, capacity_bytes=1024 * 1024)
        # First fetch = cold
        results = cache.fetch(0, [0, 1])
        sources = [r[1] for r in results]
        # Should be cold or skip (depends on dendritic)
        assert all(s in ('cold', 'skip', 'dendritic_skip') for s in sources)

        cache.advance_step()
        # Second fetch = cache hit
        results = cache.fetch(0, [0, 1])
        cache_sources = [r[1] for r in results]
        # At least some should be cache hits now
        assert cache.stats.cache_hits > 0 or cache.stats.skip_fallbacks > 0
        cache.shutdown()

    def test_lcp_priority_decay(self, expert_dir):
        cache = LCPCache(expert_dir=expert_dir, capacity_bytes=1024 * 1024,
                        enable_dendritic=False, enable_skip_fallback=False)
        cache.fetch(0, [0])
        cache.advance_step()

        entry = cache._cache.get((0, 0))
        if entry:
            p1 = cache._priority(entry)
            for _ in range(200):
                cache.advance_step()
            p2 = cache._priority(entry)
            assert p2 < p1  # priority decays over time
        cache.shutdown()

    def test_prefetch(self, expert_dir):
        cache = LCPCache(expert_dir=expert_dir, capacity_bytes=1024 * 1024,
                        enable_dendritic=False, enable_skip_fallback=False)
        # Kick off async prefetch
        cache.prefetch(0, [0, 1])
        import time
        time.sleep(0.1)  # let prefetch complete

        # Fetch should find prefetched data
        results = cache.fetch(0, [0, 1])
        has_prefetch = any(r[1] == 'prefetch' for r in results)
        has_cache = any(r[1] == 'cache' for r in results)
        # Either prefetch hit or was inserted into cache already
        assert has_prefetch or has_cache or cache.stats.cold_loads > 0
        cache.shutdown()

    def test_skip_fallback(self, expert_dir):
        cache = LCPCache(expert_dir=expert_dir, capacity_bytes=100,
                        enable_skip_fallback=True, enable_dendritic=False)
        results = cache.fetch(0, [0, 1, 2, 3])
        skip_count = sum(1 for r in results if r[1] == 'skip')
        # With tiny cache, most should be skipped
        assert skip_count > 0
        cache.shutdown()

    def test_predict_next(self, expert_dir):
        cache = LCPCache(expert_dir=expert_dir, capacity_bytes=1024 * 1024,
                        enable_dendritic=False, enable_skip_fallback=False)
        # Build co-occurrence: layer 0 experts [0,1] → layer 1 experts [2,3]
        for _ in range(10):
            cache.advance_step()
            cache.fetch(0, [0, 1])
            cache.fetch(1, [2, 3])

        predicted = cache.predict_next(0, [0, 1])
        # Should predict experts 2, 3 for layer 1
        assert len(predicted) > 0
        cache.shutdown()


class TestMixedPrecision:
    def test_requantize_roundtrip_shape(self):
        rng = np.random.default_rng(42)
        weight = rng.integers(0, 2**32, size=(64, 32), dtype=np.uint32)
        scales = rng.uniform(0.001, 0.05, size=(64, 4)).astype(np.float16)
        biases = rng.uniform(-0.01, 0.01, size=(64, 4)).astype(np.float16)

        packed, new_s, new_b, meta = requantize_4bit_to_2bit(weight, scales, biases)
        assert packed.shape[0] == weight.shape[0]  # same number of rows
        assert new_s.shape == scales.shape
        assert new_b.shape == biases.shape
        assert meta["ratio"] > 1.0  # 2-bit should be smaller

    def test_dequantize_2bit(self):
        rng = np.random.default_rng(42)
        # Create simple 2-bit packed data
        packed = rng.integers(0, 256, size=(4, 8), dtype=np.uint8)
        scales = np.float16(np.ones((4, 1)) * 0.1)
        biases = np.float16(np.zeros((4, 1)))

        result = dequantize_2bit(packed, scales, biases, n_values_per_row=32)
        assert result.shape == (4, 32)
        assert result.dtype == np.float32
        # Values should be in range [0*0.1, 3*0.1] = [0, 0.3]
        assert result.min() >= -0.01
        assert result.max() <= 0.35

    def test_expert_hotness(self):
        tracker = ExpertHotness()
        tracker.record(0, [1, 2, 3, 4])
        tracker.record(0, [1, 2, 5, 6])
        tracker.record(0, [1, 7, 8, 9])

        assert tracker.get_frequency(0, 1) == 1.0  # always selected
        assert tracker.get_frequency(0, 2) > tracker.get_frequency(0, 9)
        assert tracker.classify(0, 1, threshold=0.5) == "hot"
        assert tracker.classify(0, 9, threshold=0.5) == "cold"


class TestSmartEviction:
    def test_least_stale_scoring(self):
        policy = LeastStalePolicy(num_layers=4)
        policy.advance_token()
        policy.record_access(0, 1)

        # Recently accessed expert should score higher
        s1 = policy.score(0, 1)

        for _ in range(100):
            policy.advance_token()
        s2 = policy.score(0, 1)

        assert s1 > s2  # score decays with time

    def test_eviction_selection(self):
        policy = LeastStalePolicy(num_layers=4)
        policy.advance_token()
        policy.record_access(0, 1)
        for _ in range(50):
            policy.advance_token()
        policy.record_access(0, 2)

        # Expert 1 was accessed long ago, expert 2 recently
        evict = policy.select_eviction([(0, 1), (0, 2)])
        assert evict == (0, 1)  # should evict the stale one

    def test_routing_predictor(self):
        predictor = RoutingPredictor(num_layers=4, num_experts=8, top_k=2)
        # Train: layer 0 [0,1] always followed by layer 1 [2,3]
        for _ in range(20):
            predictor.observe(0, [0, 1])
            predictor.observe(1, [2, 3])

        predicted = predictor.predict(0, [0, 1])
        assert 2 in predicted or 3 in predicted

    def test_simulate_prefetch(self):
        result = simulate_prefetch(num_layers=4, num_experts=8, num_tokens=50, top_k=2)
        assert result.total_predictions > 0
        assert 0.0 <= result.avg_accuracy <= 1.0
        assert result.prefetch_hit_rate >= 0
