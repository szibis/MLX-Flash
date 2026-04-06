"""Tests for macOS page cache control via madvise."""

import mmap
import os
import sys
import tempfile

import pytest

from mlx_flash_compress.page_cache import (
    PageCacheAdvisor,
    PageCacheStats,
    EvictionStrategy,
    _MADV_FREE,
    _MADV_WILLNEED,
    _MADV_DONTNEED,
    _MADV_SEQUENTIAL,
    _mmap_base_address,
)


@pytest.fixture
def temp_mmap():
    """Create a temporary file with mmap for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Write 64KB of data (multiple pages)
        data = os.urandom(65536)
        f.write(data)
        f.flush()
        path = f.name

    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        yield mm
        mm.close()

    os.unlink(path)


@pytest.fixture
def advisor():
    return PageCacheAdvisor(strategy=EvictionStrategy.MADV_FREE)


class TestPageCacheAdvisor:
    def test_init_default_strategy(self):
        adv = PageCacheAdvisor()
        assert adv.strategy == EvictionStrategy.MADV_FREE

    def test_init_custom_strategy(self):
        adv = PageCacheAdvisor(strategy=EvictionStrategy.MADV_DONTNEED)
        assert adv.strategy == EvictionStrategy.MADV_DONTNEED

    def test_init_none_strategy(self):
        adv = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        assert adv.strategy == EvictionStrategy.NONE

    def test_available_on_macos(self, advisor):
        if sys.platform == "darwin":
            assert advisor.available is True
        else:
            assert advisor.available is False

    def test_stats_initial(self, advisor):
        assert advisor.stats.will_need_calls == 0
        assert advisor.stats.free_calls == 0
        assert advisor.stats.errors == 0

    def test_page_alignment(self, advisor):
        offset, length = advisor._align_range(100, 200)
        assert offset % advisor._page_size == 0
        assert (offset + length) % advisor._page_size == 0
        assert offset <= 100
        assert offset + length >= 300

    def test_page_alignment_already_aligned(self, advisor):
        ps = advisor._page_size
        offset, length = advisor._align_range(ps, ps * 2)
        assert offset == ps
        assert length == ps * 2

    def test_page_alignment_zero(self, advisor):
        offset, length = advisor._align_range(0, 4096)
        assert offset == 0

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_advise_will_need(self, advisor, temp_mmap):
        ok = advisor.advise_will_need(temp_mmap, 0, 4096)
        assert ok is True
        assert advisor.stats.will_need_calls == 1
        assert advisor.stats.will_need_bytes == 4096

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_advise_sequential(self, advisor, temp_mmap):
        ok = advisor.advise_sequential(temp_mmap, 0, 8192)
        assert ok is True
        assert advisor.stats.sequential_calls == 1

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_advise_free(self, advisor, temp_mmap):
        ok = advisor.advise_free(temp_mmap, 0, 4096)
        assert ok is True
        assert advisor.stats.free_calls == 1
        assert advisor.stats.free_bytes == 4096

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_advise_free_dontneed_strategy(self, temp_mmap):
        adv = PageCacheAdvisor(strategy=EvictionStrategy.MADV_DONTNEED)
        ok = adv.advise_free(temp_mmap, 0, 4096)
        assert ok is True

    def test_advise_free_none_strategy(self, advisor, temp_mmap):
        adv = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        ok = adv.advise_free(temp_mmap, 0, 4096)
        assert ok is False
        assert adv.stats.free_calls == 0

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_prefetch_expert(self, advisor, temp_mmap):
        ok = advisor.prefetch_expert(temp_mmap, 0, 8192)
        assert ok is True
        assert advisor.stats.will_need_calls == 1
        assert advisor.stats.sequential_calls == 1

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_evict_expert(self, advisor, temp_mmap):
        ok = advisor.evict_expert(temp_mmap, 0, 4096)
        assert ok is True
        assert advisor.stats.free_calls == 1

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_multiple_operations(self, advisor, temp_mmap):
        advisor.prefetch_expert(temp_mmap, 0, 4096)
        advisor.prefetch_expert(temp_mmap, 4096, 4096)
        advisor.evict_expert(temp_mmap, 0, 4096)
        assert advisor.stats.will_need_calls == 2
        assert advisor.stats.free_calls == 1

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_large_range(self, advisor, temp_mmap):
        ok = advisor.advise_will_need(temp_mmap, 0, 65536)
        assert ok is True


class TestMmapBaseAddress:
    def test_valid_mmap(self, temp_mmap):
        addr = _mmap_base_address(temp_mmap)
        if sys.platform == "darwin":
            assert addr is not None
            assert addr > 0
        # On non-darwin may return None


class TestEvictionStrategy:
    def test_enum_values(self):
        assert EvictionStrategy.MADV_FREE.value == "madv_free"
        assert EvictionStrategy.MADV_DONTNEED.value == "madv_dontneed"
        assert EvictionStrategy.NONE.value == "none"


class TestPageCacheStats:
    def test_defaults(self):
        stats = PageCacheStats()
        assert stats.will_need_calls == 0
        assert stats.will_need_bytes == 0
        assert stats.free_calls == 0
        assert stats.free_bytes == 0
        assert stats.sequential_calls == 0
        assert stats.errors == 0
