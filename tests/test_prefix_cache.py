"""Tests for RadixAttention-style prefix caching trie."""

from __future__ import annotations

import threading
import time

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.prefix_cache import (
    PrefixCacheTrie,
    TrieNode,
    get_or_compute_prefix,
)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX required")

NUM_LAYERS = 4
NUM_HEADS = 2
HEAD_DIM = 64


def _make_kv_list(seq_len, num_layers=NUM_LAYERS, seed=0):
    mx.random.seed(seed)
    result = []
    for _ in range(num_layers):
        k = mx.random.normal((NUM_HEADS, seq_len, HEAD_DIM))
        v = mx.random.normal((NUM_HEADS, seq_len, HEAD_DIM))
        result.append((k, v))
    return result


class TestInsertAndLookup:
    def test_insert_and_exact_lookup(self):
        trie = PrefixCacheTrie()
        tokens = [1, 2, 3, 4, 5]
        kv = _make_kv_list(5)
        trie.insert(tokens, kv)

        length, result = trie.lookup(tokens)
        assert length == 5
        assert result is not None
        assert len(result) == NUM_LAYERS

    def test_lookup_no_match(self):
        trie = PrefixCacheTrie()
        trie.insert([1, 2, 3], _make_kv_list(3))

        length, result = trie.lookup([9, 8, 7])
        assert length == 0
        assert result is None

    def test_lookup_empty_trie(self):
        trie = PrefixCacheTrie()
        length, result = trie.lookup([1, 2, 3])
        assert length == 0
        assert result is None

    def test_insert_empty_tokens(self):
        trie = PrefixCacheTrie()
        kv = _make_kv_list(0)
        trie.insert([], kv)
        length, result = trie.lookup([])
        assert length == 0


class TestPrefixMatching:
    def test_longest_prefix_match(self):
        trie = PrefixCacheTrie()
        kv3 = _make_kv_list(3, seed=1)
        kv5 = _make_kv_list(5, seed=2)
        trie.insert([1, 2, 3], kv3)
        trie.insert([1, 2, 3, 4, 5], kv5)

        length, result = trie.lookup([1, 2, 3, 4, 5, 6, 7])
        assert length == 5
        assert result is kv5

    def test_partial_prefix_match(self):
        trie = PrefixCacheTrie()
        kv = _make_kv_list(3, seed=1)
        trie.insert([1, 2, 3], kv)

        length, result = trie.lookup([1, 2, 3, 4, 5])
        assert length == 3
        assert result is kv

    def test_shorter_query_than_entry(self):
        """Query shorter than any stored prefix should not match."""
        trie = PrefixCacheTrie()
        trie.insert([1, 2, 3, 4, 5], _make_kv_list(5))

        length, result = trie.lookup([1, 2])
        assert length == 0
        assert result is None

    def test_divergent_prefixes(self):
        trie = PrefixCacheTrie()
        kv_a = _make_kv_list(3, seed=10)
        kv_b = _make_kv_list(3, seed=20)
        trie.insert([1, 2, 3], kv_a)
        trie.insert([1, 2, 4], kv_b)

        len_a, res_a = trie.lookup([1, 2, 3])
        assert len_a == 3
        assert res_a is kv_a

        len_b, res_b = trie.lookup([1, 2, 4])
        assert len_b == 3
        assert res_b is kv_b

    def test_multiple_prefixes_share_common_root(self):
        trie = PrefixCacheTrie()
        kv1 = _make_kv_list(2, seed=1)
        kv2 = _make_kv_list(4, seed=2)
        kv3 = _make_kv_list(6, seed=3)
        trie.insert([10, 20], kv1)
        trie.insert([10, 20, 30, 40], kv2)
        trie.insert([10, 20, 30, 40, 50, 60], kv3)

        length, result = trie.lookup([10, 20, 30, 40, 50])
        assert length == 4
        assert result is kv2


class TestLRUEviction:
    def test_evict_reduces_entries(self):
        trie = PrefixCacheTrie(max_entries=2)
        trie.insert([1], _make_kv_list(1, seed=1))
        time.sleep(0.01)
        trie.insert([2], _make_kv_list(1, seed=2))
        time.sleep(0.01)
        trie.insert([3], _make_kv_list(1, seed=3))

        stats = trie.get_stats()
        assert stats["entries"] <= 2

    def test_evict_removes_oldest(self):
        trie = PrefixCacheTrie(max_entries=100)
        kv_old = _make_kv_list(1, seed=1)
        kv_new = _make_kv_list(1, seed=2)
        trie.insert([1], kv_old)
        time.sleep(0.01)
        trie.insert([2], kv_new)

        trie.evict_lru(1)

        _, res_old = trie.lookup([1])
        _, res_new = trie.lookup([2])
        assert res_old is None
        assert res_new is not None

    def test_auto_eviction_on_insert(self):
        trie = PrefixCacheTrie(max_entries=3)
        for i in range(5):
            trie.insert([i * 10], _make_kv_list(1, seed=i))
            time.sleep(0.005)

        stats = trie.get_stats()
        assert stats["entries"] <= 3


class TestRadixCompression:
    def test_single_child_chains_collapsed(self):
        trie = PrefixCacheTrie()
        trie.insert([1, 2, 3, 4, 5], _make_kv_list(5, seed=1))

        # After insert the path should be a single compressed node
        assert 1 in trie._root.children
        child = trie._root.children[1]
        assert child.tokens == [1, 2, 3, 4, 5]

    def test_split_preserves_data(self):
        trie = PrefixCacheTrie()
        kv_long = _make_kv_list(5, seed=1)
        kv_short = _make_kv_list(3, seed=2)

        trie.insert([1, 2, 3, 4, 5], kv_long)
        trie.insert([1, 2, 3], kv_short)

        len_short, res_short = trie.lookup([1, 2, 3])
        assert len_short == 3
        assert res_short is kv_short

        len_long, res_long = trie.lookup([1, 2, 3, 4, 5])
        assert len_long == 5
        assert res_long is kv_long

    def test_compression_after_eviction(self):
        trie = PrefixCacheTrie(max_entries=100)
        trie.insert([1, 2, 3], _make_kv_list(3, seed=1))
        time.sleep(0.01)
        trie.insert([1, 2, 3, 4, 5], _make_kv_list(5, seed=2))

        # Evict the shorter one, then the chain [1,2,3] -> [4,5] should compress
        trie.evict_lru(1)

        len_long, res_long = trie.lookup([1, 2, 3, 4, 5])
        assert len_long == 5
        assert res_long is not None


class TestStats:
    def test_initial_stats(self):
        trie = PrefixCacheTrie()
        stats = trie.get_stats()
        assert stats["entries"] == 0
        assert stats["total_lookups"] == 0
        assert stats["total_hits"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_rate(self):
        trie = PrefixCacheTrie()
        trie.insert([1, 2, 3], _make_kv_list(3))

        trie.lookup([1, 2, 3])
        trie.lookup([9, 9, 9])

        stats = trie.get_stats()
        assert stats["total_lookups"] == 2
        assert stats["total_hits"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 1e-6

    def test_memory_bytes(self):
        trie = PrefixCacheTrie()
        kv = _make_kv_list(10)
        trie.insert([1, 2, 3], kv)

        stats = trie.get_stats()
        assert stats["memory_bytes"] > 0


class TestThreadSafety:
    def test_concurrent_inserts(self):
        trie = PrefixCacheTrie(max_entries=200)
        errors: list[Exception] = []

        def inserter(start):
            try:
                for i in range(20):
                    tokens = [start * 1000 + i]
                    trie.insert(tokens, _make_kv_list(1, seed=start * 100 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inserter, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = trie.get_stats()
        assert stats["entries"] <= 200
        assert stats["entries"] > 0

    def test_concurrent_lookups(self):
        trie = PrefixCacheTrie()
        trie.insert([1, 2, 3], _make_kv_list(3))
        errors: list[Exception] = []

        def reader():
            try:
                for _ in range(50):
                    length, _ = trie.lookup([1, 2, 3])
                    assert length == 3
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestGetOrComputePrefix:
    def test_cache_miss_calls_compute(self):
        trie = PrefixCacheTrie()
        called = [False]

        def compute(tokens):
            called[0] = True
            return _make_kv_list(len(tokens))

        hit_len, kv = get_or_compute_prefix(trie, [1, 2, 3], compute, NUM_LAYERS)
        assert hit_len == 0
        assert called[0]
        assert kv is not None

    def test_cache_hit_skips_compute(self):
        trie = PrefixCacheTrie()
        kv_original = _make_kv_list(3, seed=42)
        trie.insert([1, 2, 3], kv_original)
        called = [False]

        def compute(tokens):
            called[0] = True
            return _make_kv_list(len(tokens))

        hit_len, kv = get_or_compute_prefix(trie, [1, 2, 3], compute, NUM_LAYERS)
        assert hit_len == 3
        assert not called[0]
        assert kv is kv_original

    def test_stores_result_for_later(self):
        trie = PrefixCacheTrie()

        def compute(tokens):
            return _make_kv_list(len(tokens), seed=99)

        get_or_compute_prefix(trie, [5, 6, 7], compute, NUM_LAYERS)

        length, result = trie.lookup([5, 6, 7])
        assert length == 3
        assert result is not None
