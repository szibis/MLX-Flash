"""Tests for StreamingLLM KV cache eviction."""

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.streaming_llm import (
    StreamingLLMConfig,
    StreamingLLMCache,
)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX required")

# Test dimensions
NUM_LAYERS = 4
NUM_HEADS = 8
HEAD_DIM = 64


@pytest.fixture
def config():
    return StreamingLLMConfig(num_sink_tokens=4, window_size=16, eviction_batch=8)


@pytest.fixture
def cache(config):
    return StreamingLLMCache(config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)


def _make_kv(seq_len, num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=0):
    """Create synthetic KV tensors [num_heads, seq_len, head_dim]."""
    mx.random.seed(seed)
    keys = mx.random.normal((num_heads, seq_len, head_dim))
    values = mx.random.normal((num_heads, seq_len, head_dim))
    return keys, values


class TestStreamingLLMConfig:
    def test_defaults(self):
        config = StreamingLLMConfig()
        assert config.num_sink_tokens == 4
        assert config.window_size == 1024
        assert config.eviction_batch == 256

    def test_custom(self):
        config = StreamingLLMConfig(num_sink_tokens=8, window_size=512, eviction_batch=64)
        assert config.num_sink_tokens == 8
        assert config.window_size == 512
        assert config.eviction_batch == 64


class TestStreamingLLMCacheInit:
    def test_empty_cache(self, cache):
        assert cache.current_length == 0
        assert cache.max_length == 20  # 4 sink + 16 window
        assert not cache.is_full

    def test_get_kv_empty(self, cache):
        keys, values = cache.get_kv(0)
        assert keys.shape == (NUM_HEADS, 0, HEAD_DIM)
        assert values.shape == (NUM_HEADS, 0, HEAD_DIM)

    def test_stats_empty(self, cache):
        stats = cache.get_stats()
        assert stats["current_length"] == 0
        assert stats["eviction_count"] == 0
        assert stats["total_evicted_tokens"] == 0


class TestStreamingLLMUpdate:
    def test_single_token(self, cache):
        k, v = _make_kv(1)
        out_k, out_v = cache.update(0, k, v)
        assert out_k.shape == (NUM_HEADS, 1, HEAD_DIM)
        assert out_v.shape == (NUM_HEADS, 1, HEAD_DIM)
        assert cache.current_length == 1

    def test_multiple_tokens(self, cache):
        k, v = _make_kv(5)
        out_k, out_v = cache.update(0, k, v)
        assert out_k.shape == (NUM_HEADS, 5, HEAD_DIM)
        assert cache.current_length == 5

    def test_incremental_update(self, cache):
        k1, v1 = _make_kv(3, seed=0)
        k2, v2 = _make_kv(4, seed=1)
        cache.update(0, k1, v1)
        out_k, out_v = cache.update(0, k2, v2)
        assert out_k.shape == (NUM_HEADS, 7, HEAD_DIM)
        assert cache.current_length == 7

    def test_fill_to_capacity(self, cache):
        """Fill exactly to max_length (sink + window), no eviction."""
        k, v = _make_kv(20)  # 4 sink + 16 window = 20
        out_k, out_v = cache.update(0, k, v)
        assert out_k.shape == (NUM_HEADS, 20, HEAD_DIM)
        assert cache.is_full
        stats = cache.get_stats()
        assert stats["eviction_count"] == 0


class TestStreamingLLMEviction:
    def test_eviction_triggers(self, cache):
        """Adding beyond capacity triggers eviction."""
        # Fill beyond capacity
        k, v = _make_kv(25)
        out_k, out_v = cache.update(0, k, v)
        # After eviction: should be sink + window = 20
        assert out_k.shape == (NUM_HEADS, 20, HEAD_DIM)
        stats = cache.get_stats()
        assert stats["eviction_count"] == 1
        assert stats["total_evicted_tokens"] == 5

    def test_sink_preservation(self, cache):
        """Sink tokens must be preserved after eviction."""
        # Create identifiable sink tokens
        mx.random.seed(42)
        sink_k = mx.ones((NUM_HEADS, 4, HEAD_DIM)) * 100.0
        sink_v = mx.ones((NUM_HEADS, 4, HEAD_DIM)) * 200.0

        # Fill with sink + middle + window
        middle_k, middle_v = _make_kv(10, seed=1)
        window_k, window_v = _make_kv(16, seed=2)

        # Build initial cache: 4 sink + 10 middle = 14 (under capacity)
        cache.update(0, sink_k, sink_v)
        cache.update(0, middle_k, middle_v)
        # Now add window that pushes over: 14 + 16 = 30 > 20
        out_k, out_v = cache.update(0, window_k, window_v)

        # Verify sink tokens preserved (first 4 should be our identifiable ones)
        mx.eval(out_k, out_v)
        sink_region_k = out_k[:, :4, :]
        sink_region_v = out_v[:, :4, :]
        assert float(mx.mean(sink_region_k).item()) == pytest.approx(100.0, abs=0.01)
        assert float(mx.mean(sink_region_v).item()) == pytest.approx(200.0, abs=0.01)

    def test_window_preservation(self, cache):
        """Recent window tokens must be the last W tokens added."""
        # Add tokens in two batches to exceed capacity
        mx.random.seed(42)
        k1, v1 = _make_kv(10, seed=0)
        cache.update(0, k1, v1)

        # Add identifiable window tokens
        window_k = mx.ones((NUM_HEADS, 16, HEAD_DIM)) * 50.0
        window_v = mx.ones((NUM_HEADS, 16, HEAD_DIM)) * 75.0
        out_k, out_v = cache.update(0, window_k, window_v)

        # After eviction: last 16 tokens should be our window
        mx.eval(out_k, out_v)
        window_region_k = out_k[:, -16:, :]
        window_region_v = out_v[:, -16:, :]
        assert float(mx.mean(window_region_k).item()) == pytest.approx(50.0, abs=0.01)
        assert float(mx.mean(window_region_v).item()) == pytest.approx(75.0, abs=0.01)

    def test_multiple_evictions(self, cache):
        """Multiple sequential evictions should work correctly."""
        for i in range(10):
            k, v = _make_kv(5, seed=i)
            cache.update(0, k, v)

        # 10 * 5 = 50 tokens total, capacity = 20
        # Should have had multiple evictions
        assert cache.current_length == 20
        stats = cache.get_stats()
        assert stats["eviction_count"] > 0
        assert stats["total_tokens_seen"] == 50

    def test_large_batch_eviction(self, cache):
        """Adding many tokens at once triggers eviction."""
        k, v = _make_kv(100)
        out_k, out_v = cache.update(0, k, v)
        assert out_k.shape == (NUM_HEADS, 20, HEAD_DIM)


class TestStreamingLLMPositions:
    def test_initial_positions(self, cache):
        k, v = _make_kv(5)
        cache.update(0, k, v)
        positions = cache.get_positions(0)
        mx.eval(positions)
        assert positions.shape == (5,)
        # Positions should be 0, 1, 2, 3, 4
        for i in range(5):
            assert int(positions[i].item()) == i

    def test_positions_after_eviction(self, cache):
        """Sink tokens should keep original positions after eviction."""
        k, v = _make_kv(25)
        cache.update(0, k, v)
        positions = cache.get_positions(0)
        mx.eval(positions)

        # First 4 positions (sink) should be 0, 1, 2, 3
        for i in range(4):
            assert int(positions[i].item()) == i

        # Window positions should be the last 16 original positions
        # Original: 0..24, window = positions 9..24
        for i in range(16):
            assert int(positions[4 + i].item()) == 9 + i

    def test_empty_positions(self, cache):
        positions = cache.get_positions(0)
        mx.eval(positions)
        assert positions.shape == (0,)


class TestStreamingLLMMultiLayer:
    def test_independent_layers(self, cache):
        """Each layer maintains its own KV cache."""
        k0, v0 = _make_kv(5, seed=0)
        k1, v1 = _make_kv(3, seed=1)
        cache.update(0, k0, v0)
        cache.update(1, k1, v1)

        out_k0, _ = cache.get_kv(0)
        out_k1, _ = cache.get_kv(1)
        assert out_k0.shape[1] == 5
        assert out_k1.shape[1] == 3

    def test_all_layers_evict(self, cache):
        """Eviction in one layer doesn't affect others."""
        k_big, v_big = _make_kv(25, seed=0)
        k_small, v_small = _make_kv(5, seed=1)

        cache.update(0, k_big, v_big)
        cache.update(1, k_small, v_small)

        out_k0, _ = cache.get_kv(0)
        out_k1, _ = cache.get_kv(1)
        assert out_k0.shape[1] == 20  # evicted
        assert out_k1.shape[1] == 5   # untouched


class TestStreamingLLMReset:
    def test_reset(self, cache):
        k, v = _make_kv(10)
        cache.update(0, k, v)
        assert cache.current_length == 10

        cache.reset()
        assert cache.current_length == 0
        assert not cache.is_full
        stats = cache.get_stats()
        assert stats["eviction_count"] == 0
        assert stats["total_tokens_seen"] == 0


class TestStreamingLLMEdgeCases:
    def test_sink_larger_than_input(self):
        """When num_sink_tokens > input length, no crash."""
        config = StreamingLLMConfig(num_sink_tokens=10, window_size=5)
        cache = StreamingLLMCache(config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(3)
        out_k, out_v = cache.update(0, k, v)
        assert out_k.shape[1] == 3

    def test_window_size_one(self):
        """Minimal window size."""
        config = StreamingLLMConfig(num_sink_tokens=2, window_size=1)
        cache = StreamingLLMCache(config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        for i in range(10):
            k, v = _make_kv(1, seed=i)
            cache.update(0, k, v)
        assert cache.current_length == 3  # 2 sink + 1 window

    def test_sink_zero(self):
        """Zero sink tokens should work (pure sliding window)."""
        config = StreamingLLMConfig(num_sink_tokens=0, window_size=5)
        cache = StreamingLLMCache(config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(10)
        out_k, out_v = cache.update(0, k, v)
        assert out_k.shape[1] == 5  # just window

    def test_exact_capacity(self, cache):
        """Exactly at capacity should not trigger eviction."""
        k, v = _make_kv(20)
        cache.update(0, k, v)
        stats = cache.get_stats()
        assert stats["eviction_count"] == 0
        assert cache.current_length == 20

    def test_one_over_capacity(self, cache):
        """One token over capacity should trigger eviction."""
        k, v = _make_kv(21)
        cache.update(0, k, v)
        assert cache.current_length == 20
        stats = cache.get_stats()
        assert stats["eviction_count"] == 1
        assert stats["total_evicted_tokens"] == 1
