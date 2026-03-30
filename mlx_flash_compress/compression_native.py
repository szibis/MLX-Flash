"""Native Apple compression via libcompression (ctypes).

Apple's libcompression provides hardware-accelerated LZ4 and ZSTD on macOS.
It's typically faster than the Python C extension wrappers because:
1. It uses Apple's Accelerate framework optimizations for ARM NEON
2. Zero Python overhead — direct C library call via ctypes
3. Can use LZFSE (Apple-proprietary, fastest on Apple Silicon)

Available algorithms:
  COMPRESSION_LZ4:     Standard LZ4 (fastest decompress)
  COMPRESSION_LZ4_RAW: LZ4 without frame header (even less overhead)
  COMPRESSION_ZLIB:    Standard zlib
  COMPRESSION_LZMA:    LZMA (best ratio, slow)
  COMPRESSION_LZFSE:   Apple LZFSE (excellent speed+ratio on Apple Silicon)
  COMPRESSION_ZSTD:    ZSTD (available macOS 15+)
"""

import ctypes
import ctypes.util
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

# Load Apple's libcompression
_lib_path = ctypes.util.find_library("compression")
if _lib_path:
    _libcompression = ctypes.cdll.LoadLibrary(_lib_path)
else:
    _libcompression = None


class Algorithm(IntEnum):
    """Apple compression algorithm constants."""
    LZ4 = 0x100       # COMPRESSION_LZ4
    ZLIB = 0x205       # COMPRESSION_ZLIB
    LZMA = 0x306       # COMPRESSION_LZMA
    LZ4_RAW = 0x101    # COMPRESSION_LZ4_RAW
    LZFSE = 0x801      # COMPRESSION_LZFSE (Apple-proprietary, fast on Apple Silicon)


# Check if ZSTD is available (macOS 15+)
try:
    ZSTD = 0xB01  # COMPRESSION_ZSTD
    Algorithm.ZSTD = ZSTD
except Exception:
    ZSTD = None


@dataclass
class NativeCompressedBuffer:
    """Result of native compression."""
    data: bytes
    original_size: int
    compressed_size: int
    algo: str
    compress_time_ms: float

    @property
    def ratio(self) -> float:
        if self.compressed_size == 0:
            return 0.0
        return self.original_size / self.compressed_size


def is_available() -> bool:
    """Check if Apple's libcompression is available."""
    return _libcompression is not None


class NativeCompressor:
    """Apple libcompression wrapper — zero-overhead compression via ctypes.

    Usage:
        comp = NativeCompressor(Algorithm.LZFSE)
        buf = comp.compress(data)
        original = comp.decompress(buf)
    """

    def __init__(self, algo: Algorithm = Algorithm.LZ4):
        if not is_available():
            raise RuntimeError("Apple libcompression not available")
        self._algo = algo
        self._algo_name = algo.name if hasattr(algo, 'name') else str(algo)

        # Set up function signatures
        # size_t compression_encode_buffer(
        #     uint8_t *dst, size_t dst_size,
        #     const uint8_t *src, size_t src_size,
        #     void *scratch_buffer,
        #     compression_algorithm algorithm)
        self._encode = _libcompression.compression_encode_buffer
        self._encode.restype = ctypes.c_size_t
        self._encode.argtypes = [
            ctypes.c_void_p,  # dst
            ctypes.c_size_t,  # dst_size
            ctypes.c_void_p,  # src
            ctypes.c_size_t,  # src_size
            ctypes.c_void_p,  # scratch (NULL)
            ctypes.c_uint32,  # algorithm
        ]

        self._decode = _libcompression.compression_decode_buffer
        self._decode.restype = ctypes.c_size_t
        self._decode.argtypes = [
            ctypes.c_void_p,  # dst
            ctypes.c_size_t,  # dst_size
            ctypes.c_void_p,  # src
            ctypes.c_size_t,  # src_size
            ctypes.c_void_p,  # scratch (NULL)
            ctypes.c_uint32,  # algorithm
        ]

    def compress(self, data: bytes) -> NativeCompressedBuffer:
        """Compress data using Apple's libcompression."""
        src_size = len(data)
        # Worst case: output could be larger than input
        dst_size = src_size + (src_size // 100) + 1024
        dst_buf = ctypes.create_string_buffer(dst_size)

        t0 = time.monotonic()
        result_size = self._encode(
            dst_buf,
            dst_size,
            data,
            src_size,
            None,  # scratch buffer (NULL = library allocates)
            int(self._algo),
        )
        elapsed = (time.monotonic() - t0) * 1000

        if result_size == 0:
            raise RuntimeError(f"Compression failed with {self._algo_name}")

        return NativeCompressedBuffer(
            data=dst_buf.raw[:result_size],
            original_size=src_size,
            compressed_size=result_size,
            algo=self._algo_name,
            compress_time_ms=elapsed,
        )

    def decompress(self, buf: NativeCompressedBuffer) -> bytes:
        """Decompress data using Apple's libcompression."""
        dst_buf = ctypes.create_string_buffer(buf.original_size)

        result_size = self._decode(
            dst_buf,
            buf.original_size,
            buf.data,
            buf.compressed_size,
            None,
            int(self._algo),
        )

        if result_size == 0:
            raise RuntimeError(f"Decompression failed with {self._algo_name}")

        return dst_buf.raw[:result_size]


def benchmark_native_algorithms(data: bytes, iterations: int = 5) -> list[dict]:
    """Benchmark all available native compression algorithms."""
    if not is_available():
        return []

    algorithms = [
        ("LZ4", Algorithm.LZ4),
        ("LZ4_RAW", Algorithm.LZ4_RAW),
        ("LZFSE", Algorithm.LZFSE),
    ]

    # Try ZSTD (macOS 15+)
    try:
        test_comp = NativeCompressor(0xB01)
        test_buf = test_comp.compress(data[:1024])
        test_comp.decompress(test_buf)
        algorithms.append(("ZSTD_native", 0xB01))
    except (RuntimeError, OSError):
        pass

    results = []
    for name, algo in algorithms:
        try:
            comp = NativeCompressor(algo)

            # Warm up
            buf = comp.compress(data)
            _ = comp.decompress(buf)

            # Benchmark compress
            compress_times = []
            for _ in range(iterations):
                t0 = time.monotonic()
                buf = comp.compress(data)
                compress_times.append(time.monotonic() - t0)

            # Benchmark decompress
            decompress_times = []
            for _ in range(iterations):
                t0 = time.monotonic()
                _ = comp.decompress(buf)
                decompress_times.append(time.monotonic() - t0)

            avg_compress = sum(compress_times) / len(compress_times)
            avg_decompress = sum(decompress_times) / len(decompress_times)
            data_mb = len(data) / 1e6

            results.append({
                "algo": name,
                "ratio": f"{buf.ratio:.2f}x",
                "compress_mbs": f"{data_mb / avg_compress:.0f}" if avg_compress > 0 else "N/A",
                "decompress_mbs": f"{data_mb / avg_decompress:.0f}" if avg_decompress > 0 else "N/A",
                "compress_ms": f"{avg_compress * 1000:.2f}",
                "decompress_ms": f"{avg_decompress * 1000:.2f}",
                "compressed_bytes": buf.compressed_size,
                "original_bytes": buf.original_size,
            })
        except (RuntimeError, OSError) as e:
            results.append({
                "algo": name,
                "error": str(e),
            })

    return results
