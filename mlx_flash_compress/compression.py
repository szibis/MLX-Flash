"""Compression backends: LZ4 (speed) and ZSTD (ratio) for expert weight caching.

Design rationale:
- LZ4: ~5 GB/s decompress on M-series, ~1.6:1 on 4-bit quantized weights
- ZSTD: ~1.5 GB/s decompress, ~2.2:1 ratio — better density, slower access
- Both operate on raw bytes (numpy .tobytes() of expert weight tensors)
- Thread-safe: each call is independent, safe for concurrent GCD-style dispatch
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import lz4.frame
import zstandard as zstd


class CompressionAlgo(Enum):
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class CompressedBuffer:
    """A compressed expert weight buffer with metadata."""
    data: bytes
    original_size: int
    compressed_size: int
    algo: CompressionAlgo
    compress_time_ms: float
    layer_idx: int
    expert_id: int

    @property
    def ratio(self) -> float:
        if self.compressed_size == 0:
            return 0.0
        return self.original_size / self.compressed_size


class LZ4Compressor:
    """LZ4 frame compression — optimized for decompression speed."""

    def __init__(self, acceleration: int = 1):
        # acceleration=1 is default (fast), higher = faster but worse ratio
        self._acceleration = acceleration

    def compress(self, data: bytes, layer_idx: int = 0, expert_id: int = 0) -> CompressedBuffer:
        t0 = time.monotonic()
        compressed = lz4.frame.compress(
            data,
            compression_level=0,  # 0 = use acceleration parameter
            store_size=True,
        )
        elapsed = (time.monotonic() - t0) * 1000
        return CompressedBuffer(
            data=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            algo=CompressionAlgo.LZ4,
            compress_time_ms=elapsed,
            layer_idx=layer_idx,
            expert_id=expert_id,
        )

    def decompress(self, buf: CompressedBuffer) -> bytes:
        return lz4.frame.decompress(buf.data)


class ZSTDCompressor:
    """ZSTD compression — optimized for compression ratio."""

    def __init__(self, level: int = 3):
        self._cctx = zstd.ZstdCompressor(level=level, write_content_size=True)
        self._dctx = zstd.ZstdDecompressor()

    def compress(self, data: bytes, layer_idx: int = 0, expert_id: int = 0) -> CompressedBuffer:
        t0 = time.monotonic()
        compressed = self._cctx.compress(data)
        elapsed = (time.monotonic() - t0) * 1000
        return CompressedBuffer(
            data=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            algo=CompressionAlgo.ZSTD,
            compress_time_ms=elapsed,
            layer_idx=layer_idx,
            expert_id=expert_id,
        )

    def decompress(self, buf: CompressedBuffer) -> bytes:
        # Use content size from frame header when available, fall back to generous limit
        try:
            return self._dctx.decompress(buf.data)
        except zstd.ZstdError:
            return self._dctx.decompress(buf.data, max_output_size=buf.original_size * 2)


def get_compressor(algo: CompressionAlgo, **kwargs):
    """Factory for compression backends."""
    if algo == CompressionAlgo.LZ4:
        return LZ4Compressor(**kwargs)
    elif algo == CompressionAlgo.ZSTD:
        return ZSTDCompressor(**kwargs)
    else:
        return None
