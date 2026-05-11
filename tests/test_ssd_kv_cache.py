"""Tests for SSD KV cache persistence (hot/cold 2-tier cache)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.kv_cache_backend import (
    KVCacheBackend,
    PlainKVCache,
    create_kv_cache,
)
from mlx_flash_compress.ssd_kv_cache import SSDKVCache

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX required")

NUM_LAYERS = 4
NUM_HEADS = 2
HEAD_DIM = 64


def _make_kv(seq_len, num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=0):
    mx.random.seed(seed)
    keys = mx.random.normal((num_heads, seq_len, head_dim))
    values = mx.random.normal((num_heads, seq_len, head_dim))
    return keys, values


class TestSSDKVCacheBasic:
    def test_is_kv_cache_backend(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM, cache_dir=td)
            assert isinstance(cache, KVCacheBackend)

    def test_update_and_get_within_ram(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=100,
                cache_dir=td,
            )
            k, v = _make_kv(10)
            all_k, all_v = cache.update(0, k, v)
            assert all_k.shape == (NUM_HEADS, 10, HEAD_DIM)
            assert all_v.shape == (NUM_HEADS, 10, HEAD_DIM)

    def test_get_kv_empty(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM, cache_dir=td)
            k, v = cache.get_kv(0)
            assert k.shape == (NUM_HEADS, 0, HEAD_DIM)

    def test_stats_initial(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM, cache_dir=td)
            stats = cache.get_stats()
            assert stats["strategy"] == "ssd"
            assert stats["ssd_writes"] == 0
            assert stats["ssd_reads"] == 0


class TestSSDEviction:
    def test_eviction_to_ssd(self):
        with tempfile.TemporaryDirectory() as td:
            max_ram = 8
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=max_ram,
                cache_dir=td,
            )
            k, v = _make_kv(12, seed=1)
            cache.update(0, k, v)

            hot_k, hot_v = cache._hot.get_kv(0)
            assert hot_k.shape[1] <= max_ram

            stats = cache.get_stats()
            assert stats["ssd_writes"] >= 1
            assert stats["bytes_on_ssd"] > 0

    def test_get_kv_merges_hot_and_cold(self):
        with tempfile.TemporaryDirectory() as td:
            max_ram = 5
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=max_ram,
                cache_dir=td,
            )
            k, v = _make_kv(10, seed=2)
            cache.update(0, k, v)

            all_k, all_v = cache.get_kv(0)
            assert all_k.shape[1] == 10

    def test_multiple_evictions_accumulate(self):
        with tempfile.TemporaryDirectory() as td:
            max_ram = 4
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=max_ram,
                cache_dir=td,
            )
            for i in range(3):
                k, v = _make_kv(5, seed=i)
                cache.update(0, k, v)

            all_k, _ = cache.get_kv(0)
            assert all_k.shape[1] == 15

            stats = cache.get_stats()
            assert stats["ssd_writes"] >= 2


class TestSSDRoundTrip:
    def test_write_read_data_integrity(self):
        """Values written to SSD and read back must match the originals."""
        with tempfile.TemporaryDirectory() as td:
            max_ram = 4
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=max_ram,
                cache_dir=td,
            )
            k, v = _make_kv(8, seed=42)
            cache.update(0, k, v)

            all_k, all_v = cache.get_kv(0)
            assert mx.allclose(all_k, k, atol=1e-5)
            assert mx.allclose(all_v, v, atol=1e-5)

    def test_safetensors_files_created(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=4,
                cache_dir=td,
            )
            k, v = _make_kv(10, seed=3)
            cache.update(0, k, v)

            sf_files = list(Path(td).rglob("*.safetensors"))
            assert len(sf_files) >= 1


class TestSSDSessionIsolation:
    def test_different_prefixes_get_different_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            c1 = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                cache_dir=td,
                prompt_prefix="session_alpha",
            )
            c2 = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                cache_dir=td,
                prompt_prefix="session_beta",
            )
            assert c1._session_dir != c2._session_dir

    def test_sessions_do_not_share_data(self):
        with tempfile.TemporaryDirectory() as td:
            c1 = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=4,
                cache_dir=td,
                prompt_prefix="a",
            )
            c2 = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=4,
                cache_dir=td,
                prompt_prefix="b",
            )
            k, v = _make_kv(8, seed=10)
            c1.update(0, k, v)

            k2, v2 = c2.get_kv(0)
            assert k2.shape[1] == 0


class TestSSDReset:
    def test_reset_clears_hot_and_cold(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=4,
                cache_dir=td,
            )
            k, v = _make_kv(10, seed=5)
            cache.update(0, k, v)

            cache.reset()

            rk, rv = cache.get_kv(0)
            assert rk.shape[1] == 0

            stats = cache.get_stats()
            assert stats["ssd_writes"] == 0
            assert stats["bytes_on_ssd"] == 0

    def test_reset_removes_ssd_files(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=4,
                cache_dir=td,
            )
            k, v = _make_kv(10, seed=6)
            cache.update(0, k, v)
            session_dir = cache._session_dir

            cache.reset()

            sf_files = list(session_dir.rglob("*.safetensors"))
            assert len(sf_files) == 0


class TestSSDMultiLayer:
    def test_independent_layer_eviction(self):
        with tempfile.TemporaryDirectory() as td:
            cache = SSDKVCache(
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                max_ram_tokens=4,
                cache_dir=td,
            )
            for layer in range(NUM_LAYERS):
                k, v = _make_kv(6 + layer, seed=layer)
                cache.update(layer, k, v)

            for layer in range(NUM_LAYERS):
                all_k, _ = cache.get_kv(layer)
                assert all_k.shape[1] == 6 + layer


class TestSSDFactory:
    def test_create_ssd(self):
        with tempfile.TemporaryDirectory() as td:
            backend = create_kv_cache(
                "ssd",
                num_layers=4,
                num_heads=2,
                head_dim=64,
                max_ram_tokens=16,
                cache_dir=td,
            )
            assert isinstance(backend, SSDKVCache)

    def test_create_ssd_hybrid(self):
        with tempfile.TemporaryDirectory() as td:
            backend = create_kv_cache(
                "ssd_hybrid",
                num_layers=4,
                num_heads=2,
                head_dim=64,
                max_ram_tokens=16,
                cache_dir=td,
                key_bits=4,
                value_bits=4,
                calibration_tokens=0,
            )
            assert isinstance(backend, SSDKVCache)

    def test_ssd_hybrid_inner_is_quantized(self):
        from mlx_flash_compress.kv_cache_backend import QuantizedKVCache

        with tempfile.TemporaryDirectory() as td:
            backend = create_kv_cache(
                "ssd_hybrid",
                num_layers=4,
                num_heads=2,
                head_dim=64,
                max_ram_tokens=16,
                cache_dir=td,
                key_bits=4,
                value_bits=4,
                calibration_tokens=0,
            )
            assert isinstance(backend._hot, QuantizedKVCache)
