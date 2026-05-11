"""Tests for DraftExpertPrefetcher — expert prefetch from draft predictions."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="requires mlx")


# -- Mock objects --


class MockExpertCache:
    """Mock cache that records prefetch calls."""

    def __init__(self):
        self.prefetch_calls: list[tuple[int, list[int]]] = []

    def prefetch(self, layer_idx: int, expert_ids: list[int]):
        self.prefetch_calls.append((layer_idx, expert_ids))


class MockEmbed(nn.Module):
    """Minimal embedding function for testing."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def __call__(self, x):
        return self.embed(x)


# -- Tests --


class TestDraftExpertPrefetcher:
    def _make_prefetcher(self, num_layers=3, num_experts=8, hidden_size=32, top_k=2):
        from mlx_flash_compress.draft_expert_prefetch import DraftExpertPrefetcher

        cache = MockExpertCache()
        # Create router weights: layer_idx -> [num_experts, hidden_size]
        router_weights = {}
        for layer_idx in range(num_layers):
            w = mx.random.normal((num_experts, hidden_size))
            mx.eval(w)
            router_weights[layer_idx] = w

        embed = MockEmbed(vocab_size=100, hidden_size=hidden_size)
        mx.eval(embed.parameters())

        prefetcher = DraftExpertPrefetcher(
            expert_cache=cache,
            router_weights=router_weights,
            top_k=top_k,
        )
        return prefetcher, cache, embed

    def test_prefetch_returns_layer_expert_pairs(self):
        prefetcher, cache, embed = self._make_prefetcher(
            num_layers=3,
            num_experts=8,
            top_k=2,
        )
        draft_ids = [10, 20, 30]
        pairs = prefetcher.prefetch_from_drafts(draft_ids, embed)

        # Should return top_k experts per layer
        assert len(pairs) == 3 * 2  # 3 layers x 2 top_k
        for layer_idx, expert_idx in pairs:
            assert 0 <= layer_idx < 3
            assert 0 <= expert_idx < 8

    def test_prefetch_triggers_cache_calls(self):
        prefetcher, cache, embed = self._make_prefetcher(
            num_layers=2,
            num_experts=4,
            top_k=2,
        )
        draft_ids = [5, 15]
        prefetcher.prefetch_from_drafts(draft_ids, embed)

        # Should have called prefetch on the cache for each layer
        assert len(cache.prefetch_calls) == 2
        for layer_idx, expert_ids in cache.prefetch_calls:
            assert 0 <= layer_idx < 2
            assert len(expert_ids) == 2

    def test_empty_draft_ids(self):
        prefetcher, cache, embed = self._make_prefetcher()
        pairs = prefetcher.prefetch_from_drafts([], embed)
        assert pairs == []
        assert len(cache.prefetch_calls) == 0

    def test_empty_router_weights(self):
        from mlx_flash_compress.draft_expert_prefetch import DraftExpertPrefetcher

        cache = MockExpertCache()
        embed = MockEmbed()
        mx.eval(embed.parameters())

        prefetcher = DraftExpertPrefetcher(
            expert_cache=cache,
            router_weights={},
            top_k=2,
        )
        pairs = prefetcher.prefetch_from_drafts([1, 2, 3], embed)
        assert pairs == []

    def test_stats_tracking(self):
        prefetcher, cache, embed = self._make_prefetcher(
            num_layers=2,
            num_experts=4,
            top_k=2,
        )
        stats = prefetcher.get_stats()
        assert stats["prefetch_requests"] == 0
        assert stats["total_predictions"] == 0
        assert stats["accuracy"] == 0.0

        prefetcher.prefetch_from_drafts([5, 10], embed)

        stats = prefetcher.get_stats()
        assert stats["prefetch_requests"] == 2  # one per layer
        assert stats["total_predictions"] == 4  # 2 layers x 2 top_k

    def test_record_cache_hit(self):
        prefetcher, cache, embed = self._make_prefetcher(
            num_layers=1,
            num_experts=4,
            top_k=2,
        )
        prefetcher.prefetch_from_drafts([5], embed)
        prefetcher.record_cache_hit(count=2)

        stats = prefetcher.get_stats()
        assert stats["cache_hits_from_prefetch"] == 2
        assert stats["accuracy"] == 2 / 2  # 2 hits / 2 predictions

    def test_none_cache_graceful(self):
        """Prefetcher with None cache should work (no-op prefetch)."""
        from mlx_flash_compress.draft_expert_prefetch import DraftExpertPrefetcher

        embed = MockEmbed()
        mx.eval(embed.parameters())

        router_weights = {0: mx.random.normal((4, 32))}
        mx.eval(router_weights[0])

        prefetcher = DraftExpertPrefetcher(
            expert_cache=None,
            router_weights=router_weights,
            top_k=2,
        )
        pairs = prefetcher.prefetch_from_drafts([1, 2, 3], embed)

        # Should still return predictions even without a cache
        assert len(pairs) == 2  # 1 layer x 2 top_k
        stats = prefetcher.get_stats()
        # No prefetch_requests because cache is None (no prefetch call made)
        assert stats["prefetch_requests"] == 0
        assert stats["total_predictions"] == 2

    def test_router_weight_transposed_layout(self):
        """Router weights in [hidden_size, num_experts] layout should also work."""
        from mlx_flash_compress.draft_expert_prefetch import DraftExpertPrefetcher

        cache = MockExpertCache()
        embed = MockEmbed(hidden_size=32)
        mx.eval(embed.parameters())

        # [hidden_size, num_experts] layout (transposed from typical)
        router_weights = {0: mx.random.normal((32, 8))}
        mx.eval(router_weights[0])

        prefetcher = DraftExpertPrefetcher(
            expert_cache=cache,
            router_weights=router_weights,
            top_k=3,
        )
        pairs = prefetcher.prefetch_from_drafts([1, 2], embed)
        assert len(pairs) == 3  # 1 layer x 3 top_k

    def test_top_k_clamped_to_num_experts(self):
        """top_k larger than num_experts should be clamped."""
        prefetcher, cache, embed = self._make_prefetcher(
            num_layers=1,
            num_experts=3,
            top_k=10,
        )
        pairs = prefetcher.prefetch_from_drafts([1, 2], embed)
        # Should get at most num_experts (3) results
        assert len(pairs) == 3
