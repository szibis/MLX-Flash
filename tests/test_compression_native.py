"""Tests for native Apple compression via libcompression."""

import pytest

from mlx_flash_compress.compression_native import (
    Algorithm,
    NativeCompressedBuffer,
    NativeCompressor,
    is_available,
)

HAS_NATIVE = is_available()


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


class TestAlgorithm:
    def test_lz4_value(self):
        assert Algorithm.LZ4 == 0x100

    def test_lz4_raw_value(self):
        assert Algorithm.LZ4_RAW == 0x101

    def test_lzfse_value(self):
        assert Algorithm.LZFSE == 0x801

    def test_zlib_value(self):
        assert Algorithm.ZLIB == 0x205

    def test_lzma_value(self):
        assert Algorithm.LZMA == 0x306

    def test_enum_members(self):
        members = list(Algorithm)
        assert len(members) >= 5


class TestNativeCompressedBuffer:
    def test_creation(self):
        buf = NativeCompressedBuffer(
            data=b"compressed",
            original_size=100,
            compressed_size=50,
            algo="LZ4",
            compress_time_ms=1.5,
        )
        assert buf.original_size == 100
        assert buf.compressed_size == 50
        assert buf.algo == "LZ4"

    def test_ratio(self):
        buf = NativeCompressedBuffer(
            data=b"x",
            original_size=200,
            compressed_size=100,
            algo="LZ4",
            compress_time_ms=0,
        )
        assert buf.ratio == 2.0

    def test_ratio_zero_compressed(self):
        buf = NativeCompressedBuffer(
            data=b"",
            original_size=100,
            compressed_size=0,
            algo="LZ4",
            compress_time_ms=0,
        )
        assert buf.ratio == 0.0


class TestNativeCompressor:
    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_creation_lz4(self):
        comp = NativeCompressor(Algorithm.LZ4)
        assert comp._algo == Algorithm.LZ4

    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_creation_lzfse(self):
        comp = NativeCompressor(Algorithm.LZFSE)
        assert comp._algo == Algorithm.LZFSE

    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_compress_returns_buffer(self):
        comp = NativeCompressor(Algorithm.LZ4)
        data = b"hello world " * 1000
        buf = comp.compress(data)
        assert isinstance(buf, NativeCompressedBuffer)
        assert buf.original_size == len(data)
        assert buf.compressed_size > 0

    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_roundtrip_lz4(self):
        comp = NativeCompressor(Algorithm.LZ4)
        original = b"test data for native compression " * 100
        buf = comp.compress(original)
        restored = comp.decompress(buf)
        assert restored == original

    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_roundtrip_lzfse(self):
        comp = NativeCompressor(Algorithm.LZFSE)
        original = b"LZFSE test data " * 100
        buf = comp.compress(original)
        restored = comp.decompress(buf)
        assert restored == original

    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_roundtrip_lz4_raw(self):
        comp = NativeCompressor(Algorithm.LZ4_RAW)
        original = b"LZ4 raw test " * 100
        buf = comp.compress(original)
        restored = comp.decompress(buf)
        assert restored == original

    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_compression_ratio_positive(self):
        comp = NativeCompressor(Algorithm.LZ4)
        # Highly compressible data
        data = b"\x00" * 10000
        buf = comp.compress(data)
        assert buf.ratio > 1.0

    def test_not_available_raises(self):
        if not HAS_NATIVE:
            with pytest.raises(RuntimeError, match="not available"):
                NativeCompressor(Algorithm.LZ4)

    @pytest.mark.skipif(not HAS_NATIVE, reason="libcompression not available")
    def test_compress_time_nonnegative(self):
        comp = NativeCompressor(Algorithm.LZ4)
        buf = comp.compress(b"data" * 100)
        assert buf.compress_time_ms >= 0
