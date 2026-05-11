"""Tests for unified KV cache backend abstraction."""

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.kv_cache_backend import (
    HybridKVCache,
    KVCacheBackend,
    PlainKVCache,
    QuantizedKVCache,
    StreamingKVCache,
    _AttentionWrapper,
    _KVCacheHandle,
    create_kv_cache,
    install_kv_cache,
)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX required")

# Common test dimensions
NUM_LAYERS = 4
NUM_HEADS = 2
HEAD_DIM = 64


def _make_kv(seq_len, num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=0):
    """Create synthetic KV tensors [num_heads, seq_len, head_dim]."""
    mx.random.seed(seed)
    keys = mx.random.normal((num_heads, seq_len, head_dim))
    values = mx.random.normal((num_heads, seq_len, head_dim))
    return keys, values


# ---------------------------------------------------------------------------
# PlainKVCache
# ---------------------------------------------------------------------------


class TestPlainKVCache:
    def test_update_and_get(self):
        cache = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(5)
        all_k, all_v = cache.update(0, k, v)
        assert all_k.shape == (NUM_HEADS, 5, HEAD_DIM)
        assert all_v.shape == (NUM_HEADS, 5, HEAD_DIM)

    def test_append(self):
        cache = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k1, v1 = _make_kv(3, seed=1)
        k2, v2 = _make_kv(4, seed=2)
        cache.update(0, k1, v1)
        all_k, all_v = cache.update(0, k2, v2)
        assert all_k.shape == (NUM_HEADS, 7, HEAD_DIM)

    def test_get_kv_empty(self):
        cache = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = cache.get_kv(0)
        assert k.shape == (NUM_HEADS, 0, HEAD_DIM)
        assert v.shape == (NUM_HEADS, 0, HEAD_DIM)

    def test_reset(self):
        cache = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(5)
        cache.update(0, k, v)
        cache.reset()
        rk, rv = cache.get_kv(0)
        assert rk.shape == (NUM_HEADS, 0, HEAD_DIM)

    def test_stats(self):
        cache = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(10)
        cache.update(0, k, v)
        cache.update(1, k, v)
        stats = cache.get_stats()
        assert stats["strategy"] == "plain"
        assert stats["num_layers"] == NUM_LAYERS
        assert stats["total_tokens"] == 20  # 10 per layer, 2 layers

    def test_multiple_layers(self):
        cache = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        for i in range(NUM_LAYERS):
            k, v = _make_kv(3 + i, seed=i)
            cache.update(i, k, v)
        for i in range(NUM_LAYERS):
            k, v = cache.get_kv(i)
            assert k.shape[1] == 3 + i


# ---------------------------------------------------------------------------
# StreamingKVCache
# ---------------------------------------------------------------------------


class TestStreamingKVCache:
    def test_update_and_get(self):
        cache = StreamingKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM, num_sink_tokens=2, window_size=8)
        k, v = _make_kv(5)
        all_k, all_v = cache.update(0, k, v)
        assert all_k.shape == (NUM_HEADS, 5, HEAD_DIM)

    def test_eviction(self):
        sink = 2
        window = 4
        cache = StreamingKVCache(
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            num_sink_tokens=sink,
            window_size=window,
        )
        # Fill past capacity (sink + window = 6)
        k1, v1 = _make_kv(10)
        all_k, all_v = cache.update(0, k1, v1)
        # After eviction: sink + window = 6
        assert all_k.shape[1] == sink + window

    def test_stats_include_strategy(self):
        cache = StreamingKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        stats = cache.get_stats()
        assert stats["strategy"] == "streaming"

    def test_reset(self):
        cache = StreamingKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM, window_size=8)
        k, v = _make_kv(5)
        cache.update(0, k, v)
        cache.reset()
        rk, rv = cache.get_kv(0)
        assert rk.shape == (NUM_HEADS, 0, HEAD_DIM)


# ---------------------------------------------------------------------------
# QuantizedKVCache
# ---------------------------------------------------------------------------


class TestQuantizedKVCache:
    def test_update_and_get(self):
        cache = QuantizedKVCache(
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            key_bits=4,
            value_bits=4,
            calibration_tokens=4,
        )
        k, v = _make_kv(8)
        cache.update(0, k, v)
        rk, rv = cache.get_kv(0)
        # Output shape should match [heads, seq, dim]
        assert rk.shape == (NUM_HEADS, 8, HEAD_DIM)
        assert rv.shape == (NUM_HEADS, 8, HEAD_DIM)

    def test_get_kv_empty(self):
        cache = QuantizedKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = cache.get_kv(0)
        assert k.shape == (NUM_HEADS, 0, HEAD_DIM)

    def test_quantization_lossy(self):
        """Quantized values should differ slightly from originals (lossy)."""
        cache = QuantizedKVCache(
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            key_bits=4,
            value_bits=4,
            calibration_tokens=0,
        )
        k, v = _make_kv(16)
        cache.update(0, k, v)
        rk, rv = cache.get_kv(0)
        # Should be close but not identical
        diff = mx.mean(mx.abs(rk - k)).item()
        assert diff > 0, "Quantized output should differ from input"
        assert diff < 1.0, "Quantization error should be small"

    def test_stats_include_strategy(self):
        cache = QuantizedKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(8)
        cache.update(0, k, v)
        stats = cache.get_stats()
        assert stats["strategy"] == "quantized"

    def test_reset(self):
        cache = QuantizedKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(8)
        cache.update(0, k, v)
        cache.reset()
        rk, rv = cache.get_kv(0)
        assert rk.shape == (NUM_HEADS, 0, HEAD_DIM)


# ---------------------------------------------------------------------------
# HybridKVCache
# ---------------------------------------------------------------------------


class TestHybridKVCache:
    def test_update_and_get(self):
        cache = HybridKVCache(
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            num_sink_tokens=2,
            window_size=8,
            key_bits=4,
            value_bits=4,
            calibration_tokens=0,
        )
        k, v = _make_kv(5)
        all_k, all_v = cache.update(0, k, v)
        assert all_k.shape == (NUM_HEADS, 5, HEAD_DIM)

    def test_combines_quantization_and_eviction(self):
        """Hybrid should quantize (lossy) AND evict (bounded length)."""
        sink = 2
        window = 4
        cache = HybridKVCache(
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            num_sink_tokens=sink,
            window_size=window,
            key_bits=4,
            value_bits=4,
            calibration_tokens=0,
        )
        # Insert more tokens than sink + window
        k, v = _make_kv(12)
        all_k, all_v = cache.update(0, k, v)
        # Eviction should cap at sink + window = 6
        assert all_k.shape[1] == sink + window

        # Values should be lossy (quantization)
        diff = mx.mean(mx.abs(all_k - k[:, :6, :])).item()
        # The sink tokens (first 2) are from the beginning, window tokens
        # (last 4) are from the end. The quantization introduces error.
        assert diff > 0, "Hybrid should introduce quantization error"

    def test_stats(self):
        cache = HybridKVCache(
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            num_sink_tokens=2,
            window_size=8,
        )
        k, v = _make_kv(5)
        cache.update(0, k, v)
        stats = cache.get_stats()
        assert stats["strategy"] == "hybrid"
        assert "streaming" in stats
        assert "quantization" in stats

    def test_reset(self):
        cache = HybridKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(5)
        cache.update(0, k, v)
        cache.reset()
        rk, rv = cache.get_kv(0)
        assert rk.shape == (NUM_HEADS, 0, HEAD_DIM)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    @pytest.mark.parametrize(
        "strategy,cls",
        [
            ("plain", PlainKVCache),
            ("streaming", StreamingKVCache),
            ("quantized", QuantizedKVCache),
            ("hybrid", HybridKVCache),
        ],
    )
    def test_create_by_name(self, strategy, cls):
        backend = create_kv_cache(strategy, num_layers=4, num_heads=2, head_dim=64)
        assert isinstance(backend, cls)
        assert isinstance(backend, KVCacheBackend)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown KV cache strategy"):
            create_kv_cache("nonexistent")

    def test_kwargs_forwarded(self):
        backend = create_kv_cache(
            "streaming",
            num_layers=4,
            num_heads=2,
            head_dim=64,
            window_size=32,
            num_sink_tokens=8,
        )
        assert isinstance(backend, StreamingKVCache)
        stats = backend.get_stats()
        assert stats["window_size"] == 32
        assert stats["num_sink_tokens"] == 8


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


class _MockAttention:
    """Minimal mock attention that returns (output, (keys, values))."""

    def __init__(self, num_heads, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, x):
        seq_len = x.shape[0] if isinstance(x, mx.array) else 1
        k = mx.ones((self.num_heads, seq_len, self.head_dim))
        v = mx.ones((self.num_heads, seq_len, self.head_dim)) * 2
        output = mx.zeros((seq_len, self.num_heads * self.head_dim))
        return output, (k, v)


class _MockLayer:
    def __init__(self, num_heads, head_dim):
        self.self_attn = _MockAttention(num_heads, head_dim)


class _MockModel:
    def __init__(self, num_layers, num_heads, head_dim):
        self.layers = [_MockLayer(num_heads, head_dim) for _ in range(num_layers)]


class TestInstallKVCache:
    def test_install_captures_kv(self):
        model = _MockModel(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        backend = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        handle = install_kv_cache(model, backend)

        # Call each attention layer
        for i, layer in enumerate(model.layers):
            layer.self_attn(mx.ones((3,)))

        # Backend should have captured KV for each layer
        for i in range(NUM_LAYERS):
            k, v = backend.get_kv(i)
            assert k.shape[1] > 0, f"Layer {i} should have cached KV"

        handle.uninstall()

    def test_uninstall_restores_original(self):
        model = _MockModel(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        backend = PlainKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        originals = [layer.self_attn for layer in model.layers]

        handle = install_kv_cache(model, backend)
        # After install, self_attn should be replaced with a wrapper
        for i, layer in enumerate(model.layers):
            assert isinstance(layer.self_attn, _AttentionWrapper)
            assert layer.self_attn is not originals[i]

        handle.uninstall()
        # After uninstall, originals should be restored
        for i, layer in enumerate(model.layers):
            assert layer.self_attn is originals[i]
            assert isinstance(layer.self_attn, _MockAttention)

    def test_install_on_bad_model_raises(self):
        with pytest.raises(ValueError, match="Cannot find model layers"):
            install_kv_cache(object(), PlainKVCache(1, 1, 1))

    def test_observation_does_not_alter_output(self):
        model = _MockModel(2, NUM_HEADS, HEAD_DIM)
        backend = PlainKVCache(2, NUM_HEADS, HEAD_DIM)

        # Get output without hooks
        result_before = model.layers[0].self_attn(mx.ones((3,)))

        handle = install_kv_cache(model, backend)
        result_after = model.layers[0].self_attn(mx.ones((3,)))
        handle.uninstall()

        # Output tensors should be identical
        assert mx.array_equal(result_before[0], result_after[0])

    def test_install_with_model_model_layers(self):
        """Test the model.model.layers nesting pattern."""

        class _NestedModel:
            def __init__(self):
                self.model = _MockModel(2, NUM_HEADS, HEAD_DIM)

        model = _NestedModel()
        backend = PlainKVCache(2, NUM_HEADS, HEAD_DIM)
        handle = install_kv_cache(model, backend)

        model.model.layers[0].self_attn(mx.ones((3,)))
        k, v = backend.get_kv(0)
        assert k.shape[1] > 0

        handle.uninstall()


# ---------------------------------------------------------------------------
# Abstract base class contract
# ---------------------------------------------------------------------------


class TestKVCacheBackendContract:
    """Verify all backends satisfy the abstract interface."""

    @pytest.fixture(params=["plain", "streaming", "quantized", "hybrid"])
    def backend(self, request):
        return create_kv_cache(
            request.param,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            window_size=16,
            num_sink_tokens=2,
            calibration_tokens=0,
            key_bits=4,
            value_bits=4,
        )

    def test_is_kv_cache_backend(self, backend):
        assert isinstance(backend, KVCacheBackend)

    def test_update_returns_tuple(self, backend):
        k, v = _make_kv(5)
        result = backend.update(0, k, v)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_kv_returns_tuple(self, backend):
        k, v = _make_kv(5)
        backend.update(0, k, v)
        result = backend.get_kv(0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_clears_data(self, backend):
        k, v = _make_kv(5)
        backend.update(0, k, v)
        backend.reset()
        rk, rv = backend.get_kv(0)
        assert rk.shape[1] == 0

    def test_get_stats_returns_dict(self, backend):
        stats = backend.get_stats()
        assert isinstance(stats, dict)
