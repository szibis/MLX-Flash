"""Tiered Expert Cache Manager — the core innovation.

Architecture:
  HOT tier  (LZ4):  Fast decompress (~5 GB/s), moderate ratio (~1.6:1)
  WARM tier (ZSTD): Slower decompress (~1.5 GB/s), better ratio (~2.2:1)
  COLD tier (SSD):  Direct file read via mmap/pread, no compression

Eviction: Frequency-aware (LFU) — MoE routing follows a power law,
          so frequently-routed experts stay cached.

Thread model: Decompression dispatched to concurrent.futures ThreadPool
              (simulating GCD parallel dispatch on idle CPU P-cores).
"""

import os
import time
import mmap
import struct
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

from mlx_flash_compress.compression import (
    CompressedBuffer,
    CompressionAlgo,
    LZ4Compressor,
    ZSTDCompressor,
)
from mlx_flash_compress.compression_native import (
    is_available as native_available,
    NativeCompressor,
    NativeCompressedBuffer,
    Algorithm as NativeAlgorithm,
)


class CacheTier(Enum):
    HOT = auto()   # LZ4 compressed in RAM
    WARM = auto()  # ZSTD compressed in RAM
    COLD = auto()  # On SSD (mmap or pread)


@dataclass
class CacheStats:
    """Runtime statistics for the expert cache."""
    hot_hits: int = 0
    warm_hits: int = 0
    cold_hits: int = 0
    hot_bytes: int = 0
    warm_bytes: int = 0
    total_decompress_ms: float = 0.0
    total_ssd_read_ms: float = 0.0
    evictions: int = 0

    @property
    def total_hits(self) -> int:
        return self.hot_hits + self.warm_hits + self.cold_hits

    @property
    def hit_rate(self) -> float:
        total = self.total_hits
        if total == 0:
            return 0.0
        return (self.hot_hits + self.warm_hits) / total

    def summary(self) -> dict:
        return {
            "total_requests": self.total_hits,
            "hot_hits": self.hot_hits,
            "warm_hits": self.warm_hits,
            "cold_hits": self.cold_hits,
            "cache_hit_rate": f"{self.hit_rate:.1%}",
            "hot_bytes_mb": f"{self.hot_bytes / 1e6:.1f}",
            "warm_bytes_mb": f"{self.warm_bytes / 1e6:.1f}",
            "total_decompress_ms": f"{self.total_decompress_ms:.1f}",
            "total_ssd_read_ms": f"{self.total_ssd_read_ms:.1f}",
            "evictions": self.evictions,
        }


@dataclass
class _CacheEntry:
    """Internal cache entry."""
    buf: CompressedBuffer
    tier: CacheTier
    access_count: int = 0
    last_access_token: int = 0


class ExpertCacheManager:
    """Tiered compressed expert cache with frequency-aware eviction.

    Usage:
        cache = ExpertCacheManager(
            expert_dir="path/to/expert_weights/",
            hot_limit_bytes=20 * 1024**3,   # 20GB for LZ4 tier
            warm_limit_bytes=10 * 1024**3,   # 10GB for ZSTD tier
            num_workers=4,                    # parallel decompression threads
        )
        weights = cache.fetch_experts(layer_idx=5, expert_ids=[12, 45, 200, 387])
    """

    def __init__(
        self,
        expert_dir: str,
        hot_limit_bytes: int = 4 * 1024**3,
        warm_limit_bytes: int = 2 * 1024**3,
        num_workers: int = 4,
        enable_hot: bool = True,
        enable_warm: bool = True,
        hot_algo: str = "lz4",  # "lz4", "lzfse", or "lz4_native"
        promotion_threshold: int = 3,  # accesses before promoting warm→hot
        bypass_os_cache: bool = False,  # F_NOCACHE for fair SSD benchmarks
        simulated_ssd_latency_ms: float = 0.0,  # Add artificial SSD latency per read
    ):
        self.expert_dir = Path(expert_dir)
        self._bypass_os_cache = bypass_os_cache
        self._ssd_latency_ms = simulated_ssd_latency_ms
        self.hot_limit = hot_limit_bytes
        self.warm_limit = warm_limit_bytes
        self.num_workers = num_workers
        self.enable_hot = enable_hot
        self.enable_warm = enable_warm
        self.promotion_threshold = promotion_threshold
        self._hot_algo = hot_algo

        # Compression backends
        # ZSTD contexts are not thread-safe, so we create per-use instances
        self._lz4 = LZ4Compressor()
        self._zstd_level = 3

        # Native Apple compression (LZFSE, LZ4_RAW)
        self._use_native_hot = hot_algo in ("lzfse", "lz4_native") and native_available()
        if self._use_native_hot:
            algo = NativeAlgorithm.LZFSE if hot_algo == "lzfse" else NativeAlgorithm.LZ4_RAW
            self._native_algo = algo

        # Cache stores: key = (layer_idx, expert_id)
        self._hot: dict[tuple[int, int], _CacheEntry] = {}
        self._warm: dict[tuple[int, int], _CacheEntry] = {}

        # Access frequency tracking
        self._access_count: dict[tuple[int, int], int] = defaultdict(int)
        self._token_counter: int = 0

        # Thread pool for parallel SSD reads
        self._pool = ThreadPoolExecutor(max_workers=num_workers)
        # Background pool for async cache population (non-blocking compress)
        self._bg_pool = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()

        # Stats
        self.stats = CacheStats()

        # SSD file handles (lazy-opened)
        self._file_handles: dict[str, mmap.mmap] = {}

    def _expert_path(self, layer_idx: int, expert_id: int) -> Path:
        return self.expert_dir / f"layer_{layer_idx:03d}" / f"expert_{expert_id:04d}.bin"

    def _read_from_ssd(self, layer_idx: int, expert_id: int) -> bytes:
        """Cold tier: read expert weights directly from SSD.

        Uses F_NOCACHE on macOS to bypass OS page cache, simulating
        the cold-read path of a model too large for RAM.
        """
        path = self._expert_path(layer_idx, expert_id)
        t0 = time.monotonic()

        if self._bypass_os_cache:
            import fcntl
            fd = os.open(str(path), os.O_RDONLY)
            try:
                fcntl.fcntl(fd, 48, 1)  # F_NOCACHE = 48 on macOS
                data = os.read(fd, path.stat().st_size)
            finally:
                os.close(fd)
        else:
            data = path.read_bytes()

        # Simulate real SSD latency for models that exceed RAM
        # Real NVMe: ~0.4ms per 6.75MB expert at 17.5 GB/s
        # But with queue depth contention: ~0.6ms per expert
        if self._ssd_latency_ms > 0:
            file_size = len(data)
            # Scale latency by file size (base latency is for 2MB)
            scale = file_size / (2 * 1024 * 1024)
            time.sleep(self._ssd_latency_ms * scale / 1000)

        elapsed = (time.monotonic() - t0) * 1000
        self.stats.total_ssd_read_ms += elapsed
        return data

    def _current_hot_bytes(self) -> int:
        return sum(e.buf.compressed_size for e in self._hot.values())

    def _current_warm_bytes(self) -> int:
        return sum(e.buf.compressed_size for e in self._warm.values())

    def _evict_lfu(self, cache: dict, target_free: int, tier_name: str) -> int:
        """Evict least-frequently-used entries until target_free bytes are available."""
        freed = 0
        # Sort by access count ascending (evict least used first)
        by_freq = sorted(cache.items(), key=lambda kv: kv[1].access_count)
        for key, entry in by_freq:
            if freed >= target_free:
                break
            freed += entry.buf.compressed_size
            del cache[key]
            self.stats.evictions += 1
        return freed

    def _compress_hot(self, raw_data: bytes, layer_idx: int, expert_id: int):
        """Compress for hot tier using configured algorithm."""
        if self._use_native_hot:
            comp = NativeCompressor(self._native_algo)
            nbuf = comp.compress(raw_data)
            # Wrap as CompressedBuffer for unified interface
            return CompressedBuffer(
                data=nbuf.data,
                original_size=nbuf.original_size,
                compressed_size=nbuf.compressed_size,
                algo=CompressionAlgo.LZ4,  # reuse enum, native flag is separate
                compress_time_ms=nbuf.compress_time_ms,
                layer_idx=layer_idx,
                expert_id=expert_id,
            )
        return self._lz4.compress(raw_data, layer_idx=layer_idx, expert_id=expert_id)

    def _decompress_hot(self, buf: CompressedBuffer) -> bytes:
        """Decompress from hot tier using configured algorithm."""
        if self._use_native_hot:
            comp = NativeCompressor(self._native_algo)
            nbuf = NativeCompressedBuffer(
                data=buf.data,
                original_size=buf.original_size,
                compressed_size=buf.compressed_size,
                algo=self._hot_algo,
                compress_time_ms=0,
            )
            return comp.decompress(nbuf)
        return self._lz4.decompress(buf)

    def _insert_hot(self, key: tuple[int, int], raw_data: bytes) -> _CacheEntry:
        """Compress with configured hot algorithm and insert into hot tier."""
        buf = self._compress_hot(raw_data, layer_idx=key[0], expert_id=key[1])
        needed = buf.compressed_size
        current = self._current_hot_bytes()

        if current + needed > self.hot_limit:
            self._evict_lfu(self._hot, needed - (self.hot_limit - current), "hot")

        entry = _CacheEntry(buf=buf, tier=CacheTier.HOT)
        self._hot[key] = entry
        self.stats.hot_bytes = self._current_hot_bytes()
        return entry

    def _make_zstd(self) -> ZSTDCompressor:
        """Create a fresh ZSTD compressor (thread-safe)."""
        return ZSTDCompressor(level=self._zstd_level)

    def _insert_warm(self, key: tuple[int, int], raw_data: bytes) -> _CacheEntry:
        """Compress with ZSTD and insert into warm tier."""
        buf = self._make_zstd().compress(raw_data, layer_idx=key[0], expert_id=key[1])
        needed = buf.compressed_size
        current = self._current_warm_bytes()

        if current + needed > self.warm_limit:
            self._evict_lfu(self._warm, needed - (self.warm_limit - current), "warm")

        entry = _CacheEntry(buf=buf, tier=CacheTier.WARM)
        self._warm[key] = entry
        self.stats.warm_bytes = self._current_warm_bytes()
        return entry

    def _fetch_single(self, layer_idx: int, expert_id: int) -> tuple[bytes, CacheTier]:
        """Fetch a single expert through the tier hierarchy."""
        key = (layer_idx, expert_id)
        self._access_count[key] += 1

        # Check HOT tier (LZ4)
        if self.enable_hot and key in self._hot:
            entry = self._hot[key]
            entry.access_count += 1
            entry.last_access_token = self._token_counter
            t0 = time.monotonic()
            raw = self._decompress_hot(entry.buf)
            self.stats.total_decompress_ms += (time.monotonic() - t0) * 1000
            self.stats.hot_hits += 1
            return raw, CacheTier.HOT

        # Check WARM tier (ZSTD)
        if self.enable_warm and key in self._warm:
            entry = self._warm[key]
            entry.access_count += 1
            entry.last_access_token = self._token_counter
            t0 = time.monotonic()
            raw = self._make_zstd().decompress(entry.buf)
            self.stats.total_decompress_ms += (time.monotonic() - t0) * 1000
            self.stats.warm_hits += 1

            # Promote to HOT if accessed enough times
            if entry.access_count >= self.promotion_threshold and self.enable_hot:
                del self._warm[key]
                self._insert_hot(key, raw)

            return raw, CacheTier.WARM

        # COLD tier: read from SSD
        raw = self._read_from_ssd(layer_idx, expert_id)
        self.stats.cold_hits += 1

        # Insert into cache
        with self._lock:
            if self.enable_hot and self._access_count[key] >= self.promotion_threshold:
                self._insert_hot(key, raw)
            elif self.enable_warm:
                self._insert_warm(key, raw)
            elif self.enable_hot:
                self._insert_hot(key, raw)

        return raw, CacheTier.COLD

    def fetch_experts(
        self,
        layer_idx: int,
        expert_ids: list[int],
        expert_dtype: np.dtype = np.float16,
    ) -> list[tuple[np.ndarray, CacheTier]]:
        """Fetch K experts in parallel, returning decompressed weight arrays.

        Fast-path: cache hits are resolved synchronously (no thread pool overhead).
        Only cold SSD reads are dispatched to the thread pool for parallel I/O.
        This avoids Python's ThreadPoolExecutor overhead (~0.05ms per future)
        on the hot path where LZ4 decompress takes only ~0.02ms.

        In a C implementation (flash-moe style), GCD dispatch_group_async
        would have <1us overhead, making parallel decompress always profitable.
        """
        self._token_counter += 1

        results = {}
        cold_experts = []

        # Phase 1: Resolve cache hits synchronously (fast-path)
        for eid in expert_ids:
            key = (layer_idx, eid)
            self._access_count[key] += 1

            # Check HOT tier
            if self.enable_hot and key in self._hot:
                entry = self._hot[key]
                entry.access_count += 1
                entry.last_access_token = self._token_counter
                t0 = time.monotonic()
                raw = self._decompress_hot(entry.buf)
                self.stats.total_decompress_ms += (time.monotonic() - t0) * 1000
                self.stats.hot_hits += 1
                results[eid] = (np.frombuffer(raw, dtype=expert_dtype), CacheTier.HOT)
                continue

            # Check WARM tier
            if self.enable_warm and key in self._warm:
                entry = self._warm[key]
                entry.access_count += 1
                entry.last_access_token = self._token_counter
                t0 = time.monotonic()
                raw = self._make_zstd().decompress(entry.buf)
                self.stats.total_decompress_ms += (time.monotonic() - t0) * 1000
                self.stats.warm_hits += 1
                if entry.access_count >= self.promotion_threshold and self.enable_hot:
                    del self._warm[key]
                    self._insert_hot(key, raw)
                results[eid] = (np.frombuffer(raw, dtype=expert_dtype), CacheTier.WARM)
                continue

            # Not in cache — queue for parallel SSD read
            cold_experts.append(eid)

        # Phase 2: Dispatch cold SSD reads in parallel (the expensive path)
        if cold_experts:
            futures = {}
            for eid in cold_experts:
                fut = self._pool.submit(self._fetch_cold, layer_idx, eid)
                futures[fut] = eid

            for fut in as_completed(futures):
                eid = futures[fut]
                raw_bytes, tier = fut.result()
                results[eid] = (np.frombuffer(raw_bytes, dtype=expert_dtype), tier)

        return [results[eid] for eid in expert_ids]

    def _fetch_cold(self, layer_idx: int, expert_id: int) -> tuple[bytes, CacheTier]:
        """Fetch a single expert from SSD and cache asynchronously.

        Key optimization: read SSD (blocking), then dispatch compression
        to a background thread — don't block the caller on compression.
        The caller gets raw data immediately; the cache is populated later.
        """
        key = (layer_idx, expert_id)
        raw = self._read_from_ssd(layer_idx, expert_id)
        self.stats.cold_hits += 1

        # Async cache population — don't block on compression
        if self.enable_hot or self.enable_warm:
            raw_copy = bytes(raw)  # copy for background thread safety
            self._bg_pool.submit(self._async_cache_insert, key, raw_copy)

        return raw, CacheTier.COLD

    def _async_cache_insert(self, key: tuple[int, int], raw: bytes):
        """Background thread: compress and insert into cache."""
        with self._lock:
            # Skip if already cached (another thread may have inserted)
            if key in self._hot or key in self._warm:
                return
            if self.enable_hot:
                self._insert_hot(key, raw)
            elif self.enable_warm:
                self._insert_warm(key, raw)

    def prewarm(self, num_layers: int, num_experts: int):
        """Pre-warm the cache by compressing all experts from disk.

        Call this once before inference. In a production system, you'd
        pre-compress the model file and load compressed data directly.
        This simulates that by reading + compressing all experts upfront.
        """
        import sys
        total = num_layers * num_experts
        done = 0
        for layer_idx in range(num_layers):
            for expert_id in range(num_experts):
                path = self._expert_path(layer_idx, expert_id)
                if not path.exists():
                    continue
                raw = path.read_bytes()
                key = (layer_idx, expert_id)
                with self._lock:
                    if key not in self._hot and key not in self._warm:
                        if self.enable_hot:
                            self._insert_hot(key, raw)
                        elif self.enable_warm:
                            self._insert_warm(key, raw)
                done += 1
                if done % 100 == 0:
                    print(f"    Pre-warming: {done}/{total} experts cached...",
                          file=sys.stderr, flush=True)
        return done

    def get_stats(self) -> CacheStats:
        return self.stats

    def reset_stats(self):
        self.stats = CacheStats()

    def clear(self):
        """Clear all cached data."""
        self._hot.clear()
        self._warm.clear()
        self._access_count.clear()
        self.stats = CacheStats()

    def flush_pending(self):
        """Wait for all background cache insertions to complete."""
        self._bg_pool.shutdown(wait=True)
        self._bg_pool = ThreadPoolExecutor(max_workers=2)

    def shutdown(self):
        self._bg_pool.shutdown(wait=True)
        self._pool.shutdown(wait=False)
