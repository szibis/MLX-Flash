"""Tests for compression backends."""

import numpy as np
import pytest

from mlx_flash_compress.compression import (
    LZ4Compressor,
    ZSTDCompressor,
    CompressionAlgo,
)
from mlx_flash_compress.compression_native import (
    is_available,
    NativeCompressor,
    Algorithm,
)

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


@pytest.fixture
def sample_data():
    """Generate sample data mimicking quantized weights."""
    rng = np.random.default_rng(42)
    # Structured data: packed nibbles + scales
    data = rng.binomial(15, 0.5, size=4096).astype(np.uint8)
    return data.tobytes()


@pytest.fixture
def random_data():
    """Generate random incompressible data."""
    rng = np.random.default_rng(42)
    return rng.bytes(4096)


@pytest.mark.skipif(not HAS_LZ4, reason="lz4 not installed")
class TestLZ4:
    def test_roundtrip(self, sample_data):
        comp = LZ4Compressor()
        buf = comp.compress(sample_data)
        result = comp.decompress(buf)
        assert result == sample_data

    def test_metadata(self, sample_data):
        comp = LZ4Compressor()
        buf = comp.compress(sample_data)
        assert buf.original_size == len(sample_data)
        assert buf.compressed_size > 0
        assert buf.algo == CompressionAlgo.LZ4
        assert buf.compress_time_ms >= 0

    def test_ratio_on_structured_data(self, sample_data):
        comp = LZ4Compressor()
        buf = comp.compress(sample_data)
        # LZ4 on small structured data may not compress well
        assert buf.ratio >= 0.9  # at least not much expansion


@pytest.mark.skipif(not HAS_ZSTD, reason="zstandard not installed")
class TestZSTD:
    def test_roundtrip(self, sample_data):
        comp = ZSTDCompressor(level=3)
        buf = comp.compress(sample_data)
        result = comp.decompress(buf)
        assert result == sample_data

    def test_different_levels(self, sample_data):
        for level in [1, 3, 6]:
            comp = ZSTDCompressor(level=level)
            buf = comp.compress(sample_data)
            result = comp.decompress(buf)
            assert result == sample_data

    def test_ratio_better_than_lz4(self, sample_data):
        lz4 = LZ4Compressor()
        zstd = ZSTDCompressor(level=3)
        lz4_buf = lz4.compress(sample_data)
        zstd_buf = zstd.compress(sample_data)
        # ZSTD should achieve at least as good ratio as LZ4
        assert zstd_buf.ratio >= lz4_buf.ratio * 0.95


@pytest.mark.skipif(not is_available(), reason="libcompression not available")
class TestNativeCompression:
    def test_lz4_roundtrip(self, sample_data):
        comp = NativeCompressor(Algorithm.LZ4)
        buf = comp.compress(sample_data)
        result = comp.decompress(buf)
        assert result == sample_data

    def test_lzfse_roundtrip(self, sample_data):
        comp = NativeCompressor(Algorithm.LZFSE)
        buf = comp.compress(sample_data)
        result = comp.decompress(buf)
        assert result == sample_data

    def test_lz4_raw_roundtrip(self, sample_data):
        comp = NativeCompressor(Algorithm.LZ4_RAW)
        buf = comp.compress(sample_data)
        result = comp.decompress(buf)
        assert result == sample_data
