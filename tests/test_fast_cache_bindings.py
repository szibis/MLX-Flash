"""Tests for Metal kernel bindings: fallback paths, availability check, dataclass."""

import pytest

from mlx_flash_compress.fast_cache_bindings import (
    CStats,
    FastCacheC,
    _find_dylib,
    is_available,
)


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


class TestFindDylib:
    def test_returns_str_or_none(self):
        result = _find_dylib()
        assert result is None or isinstance(result, str)


class TestCStats:
    def test_create(self):
        s = CStats(
            cache_hits=10,
            prefetch_hits=5,
            cold_loads=2,
            skip_fallbacks=1,
            total_requests=18,
            evictions=3,
            total_lookup_us=100.0,
            total_read_us=200.0,
            total_decomp_us=50.0,
        )
        assert s.cache_hits == 10
        assert s.prefetch_hits == 5
        assert s.cold_loads == 2
        assert s.skip_fallbacks == 1
        assert s.total_requests == 18
        assert s.evictions == 3
        assert s.total_lookup_us == 100.0
        assert s.total_read_us == 200.0
        assert s.total_decomp_us == 50.0

    def test_all_zeros(self):
        s = CStats(
            cache_hits=0,
            prefetch_hits=0,
            cold_loads=0,
            skip_fallbacks=0,
            total_requests=0,
            evictions=0,
            total_lookup_us=0.0,
            total_read_us=0.0,
            total_decomp_us=0.0,
        )
        assert s.total_requests == 0


class TestFastCacheC:
    def test_raises_without_dylib(self):
        """FastCacheC should raise RuntimeError when dylib is not found."""
        if is_available():
            pytest.skip("dylib is available, cannot test fallback path")
        with pytest.raises(RuntimeError, match="libfastcache.dylib not found"):
            FastCacheC("/nonexistent/dir")

    def test_raises_with_custom_params(self):
        """Verify all constructor parameters are accepted before availability check."""
        if is_available():
            pytest.skip("dylib is available, cannot test fallback path")
        with pytest.raises(RuntimeError):
            FastCacheC(
                expert_dir="/tmp/experts",
                capacity_bytes=1024**3,
                lcp_base=0.5,
                lcp_decay=64,
                num_workers=2,
            )


class TestCacheStructure:
    """Test the ctypes _CacheStats structure definition."""

    def test_structure_fields(self):
        from mlx_flash_compress.fast_cache_bindings import _CacheStats

        field_names = [f[0] for f in _CacheStats._fields_]
        expected = [
            "cache_hits",
            "prefetch_hits",
            "cold_loads",
            "skip_fallbacks",
            "total_requests",
            "evictions",
            "total_lookup_us",
            "total_read_us",
            "total_decomp_us",
        ]
        assert field_names == expected
