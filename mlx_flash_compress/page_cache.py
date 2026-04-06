"""macOS page cache control via madvise for expert weight eviction.

Uses ctypes to call madvise() directly on mmap'd safetensors byte ranges,
giving the kernel explicit hints about which expert weights can be released
from the page cache.

This is the key mechanism that keeps memory pressure low: after an expert
is evicted from the LCP cache, we call MADV_FREE on its byte range so the
kernel can reclaim those pages without writing them to swap.

Strategies:
  MADV_FREE:      Best on macOS 10.9+. Pages become "lazy free" — reusable
                  by the kernel but still valid if not reclaimed yet.
  MADV_DONTNEED:  Advisory hint, less aggressive. Falls back on Linux.
  NONE:           Trust OS LRU entirely (for comparison benchmarks).

Usage:
  advisor = PageCacheAdvisor()
  advisor.advise_will_need(mmap_obj, offset, length)   # prefetch
  advisor.advise_free(mmap_obj, offset, length)         # evict
"""

import ctypes
import ctypes.util
import mmap
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvictionStrategy(Enum):
    MADV_FREE = "madv_free"           # macOS: lazy free (best)
    MADV_DONTNEED = "madv_dontneed"   # POSIX: advisory discard
    NONE = "none"                      # no hints, trust OS


# macOS madvise constants
_MADV_NORMAL = 0
_MADV_SEQUENTIAL = 2
_MADV_WILLNEED = 3
_MADV_DONTNEED = 4
_MADV_FREE = 5           # macOS specific
_MADV_FREE_REUSABLE = 7  # macOS specific: mark as reusable


def _get_libc():
    """Load libc for madvise syscall."""
    if sys.platform != "darwin":
        return None
    path = ctypes.util.find_library("c")
    if not path:
        return None
    try:
        lib = ctypes.CDLL(path, use_errno=True)
        lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.madvise.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None


_libc = _get_libc()


def _mmap_base_address(mm: mmap.mmap) -> Optional[int]:
    """Get the virtual address of a Python mmap object via ctypes."""
    try:
        buf = (ctypes.c_char * len(mm)).from_buffer(mm)
        return ctypes.addressof(buf)
    except (TypeError, ValueError):
        return None


@dataclass
class PageCacheStats:
    """Track page cache advisory operations."""
    will_need_calls: int = 0
    will_need_bytes: int = 0
    free_calls: int = 0
    free_bytes: int = 0
    sequential_calls: int = 0
    errors: int = 0


class PageCacheAdvisor:
    """Control macOS page cache behavior for mmap'd model weights.

    Wraps madvise() to give the kernel hints about memory access patterns:
    - WILLNEED: tell kernel to prefetch pages (readahead)
    - FREE: tell kernel pages can be reclaimed (eviction)
    - SEQUENTIAL: optimize for sequential reads (layer scanning)
    """

    def __init__(self, strategy: EvictionStrategy = EvictionStrategy.MADV_FREE):
        self.strategy = strategy
        self.stats = PageCacheStats()
        self._available = _libc is not None and sys.platform == "darwin"
        self._page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

    @property
    def available(self) -> bool:
        return self._available

    def _align_range(self, offset: int, length: int):
        """Page-align offset and length for madvise."""
        aligned_offset = (offset // self._page_size) * self._page_size
        end = offset + length
        aligned_end = ((end + self._page_size - 1) // self._page_size) * self._page_size
        return aligned_offset, aligned_end - aligned_offset

    def _call_madvise(self, mm: mmap.mmap, offset: int, length: int, advice: int) -> bool:
        """Call madvise on a page-aligned region of an mmap."""
        if not self._available or self.strategy == EvictionStrategy.NONE:
            return False

        base = _mmap_base_address(mm)
        if base is None:
            self.stats.errors += 1
            return False

        aligned_offset, aligned_length = self._align_range(offset, length)
        addr = base + aligned_offset

        ret = _libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(aligned_length), advice)
        if ret != 0:
            self.stats.errors += 1
            return False
        return True

    def advise_will_need(self, mm: mmap.mmap, offset: int, length: int) -> bool:
        """Hint that pages will be needed soon (prefetch into page cache)."""
        ok = self._call_madvise(mm, offset, length, _MADV_WILLNEED)
        if ok:
            self.stats.will_need_calls += 1
            self.stats.will_need_bytes += length
        return ok

    def advise_sequential(self, mm: mmap.mmap, offset: int, length: int) -> bool:
        """Hint that access will be sequential (optimize readahead)."""
        ok = self._call_madvise(mm, offset, length, _MADV_SEQUENTIAL)
        if ok:
            self.stats.sequential_calls += 1
        return ok

    def advise_free(self, mm: mmap.mmap, offset: int, length: int) -> bool:
        """Mark pages as reclaimable (evict from page cache).

        Uses MADV_FREE on macOS (lazy free — pages remain valid until
        kernel needs them) or MADV_DONTNEED as fallback.
        """
        if self.strategy == EvictionStrategy.MADV_FREE:
            advice = _MADV_FREE
        elif self.strategy == EvictionStrategy.MADV_DONTNEED:
            advice = _MADV_DONTNEED
        else:
            return False

        ok = self._call_madvise(mm, offset, length, advice)
        if ok:
            self.stats.free_calls += 1
            self.stats.free_bytes += length
        return ok

    def advise_free_reusable(self, mm: mmap.mmap, offset: int, length: int) -> bool:
        """macOS-specific: mark as reusable (stronger than MADV_FREE)."""
        ok = self._call_madvise(mm, offset, length, _MADV_FREE_REUSABLE)
        if ok:
            self.stats.free_calls += 1
            self.stats.free_bytes += length
        return ok

    def prefetch_expert(self, mm: mmap.mmap, offset: int, length: int) -> bool:
        """Prefetch an expert's weight range with sequential + willneed hints."""
        self.advise_sequential(mm, offset, length)
        return self.advise_will_need(mm, offset, length)

    def evict_expert(self, mm: mmap.mmap, offset: int, length: int) -> bool:
        """Evict an expert's weight range from page cache."""
        return self.advise_free(mm, offset, length)
