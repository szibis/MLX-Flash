"""Python ctypes bindings for the C GCD-accelerated cache (libfastcache.dylib).

Falls back to Python LCPCache if the dylib is not available.
"""

import ctypes
import ctypes.util
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CStats:
    cache_hits: int
    prefetch_hits: int
    cold_loads: int
    skip_fallbacks: int
    total_requests: int
    evictions: int
    total_lookup_us: float
    total_read_us: float
    total_decomp_us: float


class _CacheStats(ctypes.Structure):
    _fields_ = [
        ("cache_hits", ctypes.c_uint64),
        ("prefetch_hits", ctypes.c_uint64),
        ("cold_loads", ctypes.c_uint64),
        ("skip_fallbacks", ctypes.c_uint64),
        ("total_requests", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
        ("total_lookup_us", ctypes.c_double),
        ("total_read_us", ctypes.c_double),
        ("total_decomp_us", ctypes.c_double),
    ]


def _find_dylib() -> Optional[str]:
    """Find libfastcache.dylib."""
    # Check next to this Python file
    pkg_dir = Path(__file__).parent
    local = pkg_dir / "libfastcache.dylib"
    if local.exists():
        return str(local)
    # Check csrc
    csrc = pkg_dir.parent / "csrc" / "libfastcache.dylib"
    if csrc.exists():
        return str(csrc)
    return None


_lib = None
_lib_path = _find_dylib()

if _lib_path:
    try:
        _lib = ctypes.CDLL(_lib_path)

        # fc_create
        _lib.fc_create.restype = ctypes.c_void_p
        _lib.fc_create.argtypes = [
            ctypes.c_char_p, ctypes.c_uint64, ctypes.c_double,
            ctypes.c_int32, ctypes.c_int32,
        ]

        # fc_destroy
        _lib.fc_destroy.restype = None
        _lib.fc_destroy.argtypes = [ctypes.c_void_p]

        # fc_advance_step
        _lib.fc_advance_step.restype = None
        _lib.fc_advance_step.argtypes = [ctypes.c_void_p]

        # fc_fetch_one
        _lib.fc_fetch_one.restype = ctypes.POINTER(ctypes.c_uint8)
        _lib.fc_fetch_one.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32),
        ]

        # fc_fetch_parallel
        _lib.fc_fetch_parallel.restype = None
        _lib.fc_fetch_parallel.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ]

        # fc_prefetch
        _lib.fc_prefetch.restype = None
        _lib.fc_prefetch.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ]

        # fc_get_stats
        _lib.fc_get_stats.restype = _CacheStats
        _lib.fc_get_stats.argtypes = [ctypes.c_void_p]

        # fc_reset_stats
        _lib.fc_reset_stats.restype = None
        _lib.fc_reset_stats.argtypes = [ctypes.c_void_p]

        # fc_benchmark_dispatch
        _lib.fc_benchmark_dispatch.restype = ctypes.c_double
        _lib.fc_benchmark_dispatch.argtypes = [ctypes.c_void_p, ctypes.c_int32]

    except OSError:
        _lib = None


def is_available() -> bool:
    return _lib is not None


class FastCacheC:
    """Python wrapper around the C GCD-accelerated cache."""

    def __init__(self, expert_dir: str, capacity_bytes: int = 2 * 1024**3,
                 lcp_base: float = 0.25, lcp_decay: int = 128, num_workers: int = 4):
        if not is_available():
            raise RuntimeError("libfastcache.dylib not found. Run: make -C csrc install")

        self._handle = _lib.fc_create(
            expert_dir.encode('utf-8'),
            ctypes.c_uint64(capacity_bytes),
            ctypes.c_double(lcp_base),
            ctypes.c_int32(lcp_decay),
            ctypes.c_int32(num_workers),
        )
        if not self._handle:
            raise RuntimeError("fc_create returned NULL")

    def advance_step(self):
        _lib.fc_advance_step(self._handle)

    def fetch_one(self, layer_idx: int, expert_id: int) -> tuple[Optional[bytes], str]:
        out_size = ctypes.c_uint64(0)
        out_source = ctypes.c_int32(0)
        ptr = _lib.fc_fetch_one(
            self._handle, layer_idx, expert_id,
            ctypes.byref(out_size), ctypes.byref(out_source),
        )
        sources = {0: 'cache', 1: 'prefetch', 2: 'cold', 3: 'skip'}
        source = sources.get(out_source.value, 'unknown')
        if ptr and out_size.value > 0:
            data = ctypes.string_at(ptr, out_size.value)
            return data, source
        return None, source

    def fetch_parallel(self, layer_idx: int, expert_ids: list[int]) -> list[tuple[Optional[bytes], str]]:
        k = len(expert_ids)
        c_ids = (ctypes.c_int32 * k)(*expert_ids)
        c_sizes = (ctypes.c_uint64 * k)()
        c_sources = (ctypes.c_int32 * k)()
        c_ptrs = (ctypes.POINTER(ctypes.c_uint8) * k)()

        _lib.fc_fetch_parallel(
            self._handle, layer_idx, c_ids, k,
            c_sizes, c_sources, c_ptrs,
        )

        sources_map = {0: 'cache', 1: 'prefetch', 2: 'cold', 3: 'skip'}
        results = []
        for i in range(k):
            source = sources_map.get(c_sources[i], 'unknown')
            if c_ptrs[i] and c_sizes[i] > 0:
                data = ctypes.string_at(c_ptrs[i], c_sizes[i])
                results.append((data, source))
            else:
                results.append((None, source))
        return results

    def prefetch(self, layer_idx: int, expert_ids: list[int]):
        k = len(expert_ids)
        c_ids = (ctypes.c_int32 * k)(*expert_ids)
        _lib.fc_prefetch(self._handle, layer_idx, c_ids, k)

    def get_stats(self) -> CStats:
        s = _lib.fc_get_stats(self._handle)
        return CStats(
            cache_hits=s.cache_hits, prefetch_hits=s.prefetch_hits,
            cold_loads=s.cold_loads, skip_fallbacks=s.skip_fallbacks,
            total_requests=s.total_requests, evictions=s.evictions,
            total_lookup_us=s.total_lookup_us, total_read_us=s.total_read_us,
            total_decomp_us=s.total_decomp_us,
        )

    def benchmark_dispatch(self, iterations: int = 10000) -> float:
        """Returns average dispatch overhead in microseconds."""
        return _lib.fc_benchmark_dispatch(self._handle, iterations)

    def destroy(self):
        if self._handle:
            _lib.fc_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.destroy()
