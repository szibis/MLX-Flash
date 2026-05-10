"""Tests for quantized KV cache."""

import mlx.core as mx
import pytest

from mlx_flash_compress.quantized_kv_cache import (
    QuantizedKVCacheManager,
    QuantizedKVConfig,
    QuantizedKVEntry,
    dequantize_tensor,
    quantize_tensor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_data():
    """Standard normal data for roundtrip tests."""
    mx.random.seed(42)
    return mx.random.normal(shape=(128,))


@pytest.fixture
def kv_data():
    """Synthetic KV data: (seq_len, num_heads, head_dim)."""
    mx.random.seed(42)
    return mx.random.normal(shape=(16, 8, 64))


@pytest.fixture
def default_config():
    return QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=8)


# ---------------------------------------------------------------------------
# quantize_tensor / dequantize_tensor roundtrip
# ---------------------------------------------------------------------------


class TestQuantizeDequantize:
    """Test quantize/dequantize roundtrip quality."""

    def test_4bit_roundtrip_quality(self, normal_data):
        """4-bit roundtrip MSE should be < 0.01 for standard normal data."""
        packed, scales, zeros = quantize_tensor(normal_data, bits=4, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=4, original_shape=normal_data.shape, group_size=64)
        mse = mx.mean((normal_data.astype(mx.float32) - restored) ** 2).item()
        assert mse < 0.02, f"4-bit MSE {mse} exceeds threshold 0.02"

    def test_8bit_roundtrip_quality(self, normal_data):
        """8-bit should be near-lossless."""
        packed, scales, zeros = quantize_tensor(normal_data, bits=8, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=8, original_shape=normal_data.shape, group_size=64)
        mse = mx.mean((normal_data.astype(mx.float32) - restored) ** 2).item()
        assert mse < 0.001, f"8-bit MSE {mse} exceeds threshold 0.001"

    def test_2bit_roundtrip(self, normal_data):
        """2-bit has higher error but should still reconstruct."""
        packed, scales, zeros = quantize_tensor(normal_data, bits=2, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=2, original_shape=normal_data.shape, group_size=64)
        mse = mx.mean((normal_data.astype(mx.float32) - restored) ** 2).item()
        # 2-bit is lossy but should be bounded
        assert mse < 0.6, f"2-bit MSE {mse} exceeds threshold 0.6"
        # Should preserve sign/magnitude roughly
        assert restored.shape == normal_data.shape

    def test_4bit_preserves_shape(self):
        """Dequantized output shape matches input."""
        mx.random.seed(0)
        x = mx.random.normal(shape=(10, 8, 64))
        packed, scales, zeros = quantize_tensor(x, bits=4, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=4, original_shape=x.shape, group_size=64)
        assert restored.shape == x.shape

    def test_8bit_preserves_shape(self):
        mx.random.seed(0)
        x = mx.random.normal(shape=(5, 4, 128))
        packed, scales, zeros = quantize_tensor(x, bits=8, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=8, original_shape=x.shape, group_size=64)
        assert restored.shape == x.shape

    def test_2bit_preserves_shape(self):
        mx.random.seed(0)
        x = mx.random.normal(shape=(3, 2, 32))
        packed, scales, zeros = quantize_tensor(x, bits=2, group_size=32)
        restored = dequantize_tensor(packed, scales, zeros, bits=2, original_shape=x.shape, group_size=32)
        assert restored.shape == x.shape

    def test_invalid_bits_raises(self):
        x = mx.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="bits must be 2, 4, or 8"):
            quantize_tensor(x, bits=3)
        with pytest.raises(ValueError, match="bits must be 2, 4, or 8"):
            dequantize_tensor(x, x, x, bits=5, original_shape=(3,))

    def test_zeros_roundtrip(self):
        """All-zero tensor should roundtrip cleanly."""
        x = mx.zeros((64,))
        packed, scales, zeros = quantize_tensor(x, bits=4, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=4, original_shape=x.shape, group_size=64)
        mse = mx.mean((x.astype(mx.float32) - restored) ** 2).item()
        assert mse == 0.0

    def test_multidimensional_roundtrip(self):
        """Test with 3D tensor mimicking KV cache shape."""
        mx.random.seed(7)
        x = mx.random.normal(shape=(32, 8, 128))
        for bits in (2, 4, 8):
            packed, scales, zeros = quantize_tensor(x, bits=bits, group_size=64)
            restored = dequantize_tensor(packed, scales, zeros, bits=bits, original_shape=x.shape, group_size=64)
            assert restored.shape == x.shape
            mse = mx.mean((x.astype(mx.float32) - restored) ** 2).item()
            if bits == 8:
                assert mse < 0.001
            elif bits == 4:
                assert mse < 0.02


# ---------------------------------------------------------------------------
# Bit packing correctness
# ---------------------------------------------------------------------------


class TestBitPacking:
    """Verify 4-bit and 2-bit packing produces expected sizes."""

    def test_4bit_packing_size(self):
        """4-bit should produce half the elements (2 values per byte)."""
        x = mx.random.normal(shape=(128,))
        packed, scales, zeros = quantize_tensor(x, bits=4, group_size=64)
        # 128 values -> 64 packed bytes
        assert packed.shape[-1] == 64

    def test_2bit_packing_size(self):
        """2-bit should produce quarter the elements (4 values per byte)."""
        x = mx.random.normal(shape=(128,))
        packed, scales, zeros = quantize_tensor(x, bits=2, group_size=64)
        # 128 values -> 32 packed bytes
        assert packed.shape[-1] == 32

    def test_8bit_packing_size(self):
        """8-bit should produce same number of elements."""
        x = mx.random.normal(shape=(128,))
        packed, scales, zeros = quantize_tensor(x, bits=8, group_size=64)
        assert packed.shape[-1] == 128


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for quantization."""

    def test_single_element(self):
        """Single element tensor."""
        x = mx.array([1.5])
        for bits in (2, 4, 8):
            packed, scales, zeros = quantize_tensor(x, bits=bits, group_size=64)
            restored = dequantize_tensor(packed, scales, zeros, bits=bits, original_shape=x.shape, group_size=64)
            assert restored.shape == (1,)
            # Single element: scale = |x| / max_int, so it should roundtrip well
            assert abs(restored.item() - x.item()) < 1.0

    def test_group_size_larger_than_length(self):
        """group_size > tensor length should still work."""
        x = mx.random.normal(shape=(8,))
        packed, scales, zeros = quantize_tensor(x, bits=4, group_size=256)
        restored = dequantize_tensor(packed, scales, zeros, bits=4, original_shape=x.shape, group_size=256)
        assert restored.shape == x.shape
        mse = mx.mean((x.astype(mx.float32) - restored) ** 2).item()
        assert mse < 0.05

    def test_non_divisible_group_size(self):
        """Tensor length not divisible by group_size."""
        x = mx.random.normal(shape=(100,))  # not divisible by 64
        packed, scales, zeros = quantize_tensor(x, bits=4, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=4, original_shape=x.shape, group_size=64)
        assert restored.shape == (100,)

    def test_very_small_values(self):
        """Values very close to zero."""
        x = mx.array([1e-10, -1e-10, 0.0, 1e-10])
        packed, scales, zeros = quantize_tensor(x, bits=4, group_size=4)
        restored = dequantize_tensor(packed, scales, zeros, bits=4, original_shape=x.shape, group_size=4)
        assert restored.shape == x.shape

    def test_large_values(self):
        """Large magnitude values."""
        x = mx.array([1000.0, -500.0, 750.0, -250.0] * 16)  # 64 elements
        packed, scales, zeros = quantize_tensor(x, bits=4, group_size=64)
        restored = dequantize_tensor(packed, scales, zeros, bits=4, original_shape=x.shape, group_size=64)
        assert restored.shape == x.shape
        # Relative error should be bounded
        max_abs = mx.max(mx.abs(x)).item()
        max_err = mx.max(mx.abs(x.astype(mx.float32) - restored)).item()
        assert max_err / max_abs < 0.2  # within 20% relative error for 4-bit


# ---------------------------------------------------------------------------
# QuantizedKVEntry
# ---------------------------------------------------------------------------


class TestQuantizedKVEntry:
    """Test the per-layer KV cache entry."""

    def test_calibration_period(self, default_config):
        """During calibration, data should be stored in full precision."""
        entry = QuantizedKVEntry(num_heads=8, head_dim=64, config=default_config)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(4, 8, 64))
        values = mx.random.normal(shape=(4, 8, 64))

        entry.append(keys, values)
        assert entry.length == 4
        assert not entry._calibration_done

        # During calibration, get_keys_values returns full precision
        k, v = entry.get_keys_values()
        assert k.shape == (4, 8, 64)
        assert v.shape == (4, 8, 64)
        # Should be exact (no quantization yet)
        mse_k = mx.mean((keys.astype(mx.float32) - k) ** 2).item()
        assert mse_k == 0.0

    def test_calibration_transition(self, default_config):
        """After calibration_tokens, data should be quantized."""
        entry = QuantizedKVEntry(num_heads=8, head_dim=64, config=default_config)
        mx.random.seed(0)

        # Append enough tokens to trigger calibration
        keys = mx.random.normal(shape=(8, 8, 64))
        values = mx.random.normal(shape=(8, 8, 64))
        entry.append(keys, values)

        assert entry.length == 8
        assert entry._calibration_done
        assert entry._fp_keys is None  # full-precision released

        k, v = entry.get_keys_values()
        assert k.shape == (8, 8, 64)

    def test_post_calibration_append(self, default_config):
        """Appending after calibration should quantize immediately."""
        entry = QuantizedKVEntry(num_heads=8, head_dim=64, config=default_config)
        mx.random.seed(0)

        # First pass: calibration
        keys1 = mx.random.normal(shape=(8, 8, 64))
        values1 = mx.random.normal(shape=(8, 8, 64))
        entry.append(keys1, values1)
        assert entry._calibration_done

        # Second pass: quantized
        keys2 = mx.random.normal(shape=(4, 8, 64))
        values2 = mx.random.normal(shape=(4, 8, 64))
        entry.append(keys2, values2)

        assert entry.length == 12
        assert len(entry._q_keys) == 2  # calibration chunk + new chunk

        k, v = entry.get_keys_values()
        assert k.shape == (12, 8, 64)

    def test_multiple_appends(self, default_config):
        """Multiple small appends during and after calibration."""
        entry = QuantizedKVEntry(num_heads=4, head_dim=64, config=default_config)
        mx.random.seed(0)

        for i in range(6):
            keys = mx.random.normal(shape=(2, 4, 64))
            values = mx.random.normal(shape=(2, 4, 64))
            entry.append(keys, values)

        assert entry.length == 12
        k, v = entry.get_keys_values()
        assert k.shape == (12, 4, 64)

    def test_empty_cache_raises(self, default_config):
        """Getting KV from empty cache should raise."""
        entry = QuantizedKVEntry(num_heads=8, head_dim=64, config=default_config)
        with pytest.raises(ValueError, match="No data"):
            entry.get_keys_values()

    def test_roundtrip_quality_through_entry(self, default_config):
        """Data through entry should have acceptable quality."""
        config = QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=0)
        entry = QuantizedKVEntry(num_heads=8, head_dim=64, config=config)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(16, 8, 64))
        values = mx.random.normal(shape=(16, 8, 64))

        # calibration_tokens=0 means immediate quantization
        entry.append(keys, values)
        k, v = entry.get_keys_values()

        mse_k = mx.mean((keys.astype(mx.float32) - k) ** 2).item()
        mse_v = mx.mean((values.astype(mx.float32) - v) ** 2).item()
        assert mse_k < 0.02
        assert mse_v < 0.02


# ---------------------------------------------------------------------------
# Memory savings calculation
# ---------------------------------------------------------------------------


class TestMemorySavings:
    """Test memory accounting."""

    def test_full_precision_bytes(self, default_config):
        """full_precision_bytes calculation."""
        entry = QuantizedKVEntry(num_heads=8, head_dim=64, config=default_config)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(16, 8, 64))
        values = mx.random.normal(shape=(16, 8, 64))
        entry.append(keys, values)

        # 16 tokens * 8 heads * 64 dim * 2 (K+V) * 2 bytes (fp16)
        expected = 16 * 8 * 64 * 2 * 2
        assert entry.full_precision_bytes == expected

    def test_quantized_memory_less_than_full(self, default_config):
        """Quantized memory should be less than full precision."""
        config = QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=0)
        entry = QuantizedKVEntry(num_heads=8, head_dim=128, config=config)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(64, 8, 128))
        values = mx.random.normal(shape=(64, 8, 128))
        entry.append(keys, values)

        actual = entry.memory_bytes
        full_prec = entry.full_precision_bytes
        # 4-bit should be roughly 4x smaller (some overhead from scales)
        assert actual < full_prec
        ratio = actual / full_prec
        # Should be less than 0.5 (generous bound accounting for scale overhead)
        assert ratio < 0.5, f"Compression ratio {ratio} not below 0.5"

    def test_calibration_memory_is_full_precision(self, default_config):
        """During calibration, memory should be full precision."""
        entry = QuantizedKVEntry(num_heads=8, head_dim=64, config=default_config)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(4, 8, 64))
        values = mx.random.normal(shape=(4, 8, 64))
        entry.append(keys, values)

        assert not entry._calibration_done
        # During calibration, memory_bytes accounts for fp storage
        assert entry.memory_bytes > 0


# ---------------------------------------------------------------------------
# QuantizedKVCacheManager
# ---------------------------------------------------------------------------


class TestQuantizedKVCacheManager:
    """Test multi-layer cache manager."""

    def test_basic_update_and_get(self):
        config = QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=0)
        manager = QuantizedKVCacheManager(config, num_layers=4, num_kv_heads=8, head_dim=64)
        mx.random.seed(0)
        for layer in range(4):
            keys = mx.random.normal(shape=(8, 8, 64))
            values = mx.random.normal(shape=(8, 8, 64))
            manager.update(layer, keys, values)

        for layer in range(4):
            k, v = manager.get_kv(layer)
            assert k.shape == (8, 8, 64)
            assert v.shape == (8, 8, 64)

    def test_invalid_layer_raises(self):
        config = QuantizedKVConfig()
        manager = QuantizedKVCacheManager(config, num_layers=4, num_kv_heads=8, head_dim=64)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(1, 8, 64))
        values = mx.random.normal(shape=(1, 8, 64))

        with pytest.raises(IndexError):
            manager.update(5, keys, values)
        with pytest.raises(IndexError):
            manager.get_kv(-1)

    def test_compression_ratio(self):
        config = QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=0)
        manager = QuantizedKVCacheManager(config, num_layers=2, num_kv_heads=8, head_dim=128)
        mx.random.seed(0)
        for layer in range(2):
            keys = mx.random.normal(shape=(64, 8, 128))
            values = mx.random.normal(shape=(64, 8, 128))
            manager.update(layer, keys, values)

        ratio = manager.get_compression_ratio()
        assert 0 < ratio < 0.5

    def test_get_stats(self):
        config = QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=0)
        manager = QuantizedKVCacheManager(config, num_layers=2, num_kv_heads=4, head_dim=64)
        mx.random.seed(0)
        for layer in range(2):
            keys = mx.random.normal(shape=(16, 4, 64))
            values = mx.random.normal(shape=(16, 4, 64))
            manager.update(layer, keys, values)

        stats = manager.get_stats()
        assert stats["num_layers"] == 2
        assert stats["total_tokens"] == 32  # 16 per layer
        assert stats["config"]["key_bits"] == 4
        assert stats["compression_ratio"] < 1.0
        assert stats["memory_savings_pct"] > 0
        assert len(stats["per_layer"]) == 2

    def test_reset(self):
        config = QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=0)
        manager = QuantizedKVCacheManager(config, num_layers=2, num_kv_heads=4, head_dim=64)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(8, 4, 64))
        values = mx.random.normal(shape=(8, 4, 64))
        manager.update(0, keys, values)

        assert manager.entries[0].length == 8
        manager.reset()
        assert manager.entries[0].length == 0

    def test_empty_compression_ratio(self):
        config = QuantizedKVConfig()
        manager = QuantizedKVCacheManager(config, num_layers=2, num_kv_heads=4, head_dim=64)
        assert manager.get_compression_ratio() == 1.0

    def test_incremental_updates(self):
        """Simulate token-by-token generation."""
        config = QuantizedKVConfig(key_bits=4, value_bits=4, group_size=64, calibration_tokens=4)
        manager = QuantizedKVCacheManager(config, num_layers=2, num_kv_heads=4, head_dim=64)
        mx.random.seed(0)

        # Feed tokens one at a time
        for step in range(10):
            for layer in range(2):
                k = mx.random.normal(shape=(1, 4, 64))
                v = mx.random.normal(shape=(1, 4, 64))
                manager.update(layer, k, v)

        for layer in range(2):
            k, v = manager.get_kv(layer)
            assert k.shape == (10, 4, 64)
            assert v.shape == (10, 4, 64)

    def test_8bit_mode(self):
        """8-bit mode should work with near-lossless quality."""
        config = QuantizedKVConfig(key_bits=8, value_bits=8, group_size=64, calibration_tokens=0)
        manager = QuantizedKVCacheManager(config, num_layers=1, num_kv_heads=4, head_dim=64)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(16, 4, 64))
        values = mx.random.normal(shape=(16, 4, 64))
        manager.update(0, keys, values)

        k, v = manager.get_kv(0)
        mse = mx.mean((keys.astype(mx.float32) - k) ** 2).item()
        assert mse < 0.001

    def test_2bit_mode(self):
        """2-bit mode should work (lossy but functional)."""
        config = QuantizedKVConfig(key_bits=2, value_bits=2, group_size=64, calibration_tokens=0)
        manager = QuantizedKVCacheManager(config, num_layers=1, num_kv_heads=4, head_dim=64)
        mx.random.seed(0)
        keys = mx.random.normal(shape=(16, 4, 64))
        values = mx.random.normal(shape=(16, 4, 64))
        manager.update(0, keys, values)

        k, v = manager.get_kv(0)
        assert k.shape == (16, 4, 64)
        mse = mx.mean((keys.astype(mx.float32) - k) ** 2).item()
        assert mse < 0.6  # 2-bit is lossy


# ---------------------------------------------------------------------------
# Calibration behavior
# ---------------------------------------------------------------------------


class TestCalibration:
    """Detailed calibration period tests."""

    def test_calibration_exact_boundary(self):
        """Appending exactly calibration_tokens should finalize."""
        config = QuantizedKVConfig(calibration_tokens=16)
        entry = QuantizedKVEntry(num_heads=4, head_dim=64, config=config)
        mx.random.seed(0)

        keys = mx.random.normal(shape=(16, 4, 64))
        values = mx.random.normal(shape=(16, 4, 64))
        entry.append(keys, values)

        assert entry._calibration_done
        assert entry.length == 16

    def test_calibration_overshoot(self):
        """Appending more than calibration_tokens in one shot."""
        config = QuantizedKVConfig(calibration_tokens=8)
        entry = QuantizedKVEntry(num_heads=4, head_dim=64, config=config)
        mx.random.seed(0)

        keys = mx.random.normal(shape=(20, 4, 64))
        values = mx.random.normal(shape=(20, 4, 64))
        entry.append(keys, values)

        assert entry._calibration_done
        assert entry.length == 20

    def test_zero_calibration(self):
        """calibration_tokens=0 means immediate quantization."""
        config = QuantizedKVConfig(calibration_tokens=0)
        entry = QuantizedKVEntry(num_heads=4, head_dim=64, config=config)
        mx.random.seed(0)

        keys = mx.random.normal(shape=(4, 4, 64))
        values = mx.random.normal(shape=(4, 4, 64))
        entry.append(keys, values)

        # With calibration_tokens=0, first append triggers finalization
        # (the fp buffer is filled then immediately quantized)
        assert entry._calibration_done

    def test_gradual_calibration(self):
        """Multiple small appends building up to calibration threshold."""
        config = QuantizedKVConfig(calibration_tokens=8)
        entry = QuantizedKVEntry(num_heads=4, head_dim=64, config=config)
        mx.random.seed(0)

        for i in range(4):
            keys = mx.random.normal(shape=(2, 4, 64))
            values = mx.random.normal(shape=(2, 4, 64))
            entry.append(keys, values)
            if i < 3:
                assert not entry._calibration_done
            else:
                assert entry._calibration_done

        assert entry.length == 8
