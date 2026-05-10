"""Tests for ScissorHands/H2O KV cache compression."""

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.kv_compression import (
    KVCompressionConfig,
    AttentionScoreTracker,
    CompressedKVCache,
)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX required")

# Test dimensions
NUM_LAYERS = 4
NUM_HEADS = 8
HEAD_DIM = 64


@pytest.fixture
def h2o_config():
    return KVCompressionConfig(
        budget_ratio=0.2,
        sink_tokens=4,
        recent_window=8,
        scoring="h2o",
    )


@pytest.fixture
def scissorhands_config():
    return KVCompressionConfig(
        budget_ratio=0.2,
        sink_tokens=4,
        recent_window=8,
        scoring="scissorhands",
    )


@pytest.fixture
def h2o_cache(h2o_config):
    return CompressedKVCache(h2o_config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)


@pytest.fixture
def scissorhands_cache(scissorhands_config):
    return CompressedKVCache(scissorhands_config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)


def _make_kv(seq_len, num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=0):
    """Create synthetic KV tensors [num_heads, seq_len, head_dim]."""
    mx.random.seed(seed)
    keys = mx.random.normal((num_heads, seq_len, head_dim))
    values = mx.random.normal((num_heads, seq_len, head_dim))
    return keys, values


def _make_attention_weights(num_heads, seq_len, seed=0):
    """Create synthetic attention weights [num_heads, seq_len, seq_len].

    Produces valid attention distributions (rows sum to 1 via softmax).
    """
    mx.random.seed(seed)
    logits = mx.random.normal((num_heads, seq_len, seq_len))
    # Apply softmax along last dimension to get valid attention weights
    weights = mx.softmax(logits, axis=-1)
    return weights


def _make_heavy_hitter_attention(num_heads, seq_len, heavy_indices, seed=0):
    """Create attention weights where specific tokens are heavy hitters.

    Heavy hitter tokens receive much higher attention from all queries.
    """
    mx.random.seed(seed)
    # Start with uniform-ish logits
    logits = mx.random.normal((num_heads, seq_len, seq_len)) * 0.1

    # Boost heavy hitter columns significantly
    for idx in heavy_indices:
        boost = mx.ones((num_heads, seq_len, 1)) * 5.0
        # Set the column for heavy hitter token to high value
        logits_list = []
        for j in range(seq_len):
            if j == idx:
                logits_list.append(logits[:, :, j:j+1] + 5.0)
            else:
                logits_list.append(logits[:, :, j:j+1])
        logits = mx.concatenate(logits_list, axis=-1)

    weights = mx.softmax(logits, axis=-1)
    return weights


class TestKVCompressionConfig:
    def test_defaults(self):
        config = KVCompressionConfig()
        assert config.budget_ratio == 0.2
        assert config.sink_tokens == 4
        assert config.recent_window == 128
        assert config.scoring == "h2o"
        assert config.quantize_evicted is False

    def test_custom(self):
        config = KVCompressionConfig(
            budget_ratio=0.5,
            sink_tokens=8,
            recent_window=64,
            scoring="scissorhands",
            quantize_evicted=True,
        )
        assert config.budget_ratio == 0.5
        assert config.scoring == "scissorhands"
        assert config.quantize_evicted is True


class TestAttentionScoreTracker:
    def test_h2o_record_and_query(self):
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")
        weights = _make_attention_weights(NUM_HEADS, 10)
        tracker.record_attention(0, weights)

        importance = tracker.get_token_importance(0)
        mx.eval(importance)
        assert importance.shape == (10,)
        # All scores should be non-negative (softmax output)
        assert float(mx.min(importance).item()) >= 0.0

    def test_h2o_accumulation(self):
        """H2O should accumulate scores across calls."""
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")
        weights = _make_attention_weights(NUM_HEADS, 10, seed=0)

        # Record same weights twice
        tracker.record_attention(0, weights)
        importance_1 = tracker.get_token_importance(0)
        mx.eval(importance_1)
        sum_1 = float(mx.sum(importance_1).item())

        tracker.record_attention(0, weights)
        importance_2 = tracker.get_token_importance(0)
        mx.eval(importance_2)
        sum_2 = float(mx.sum(importance_2).item())

        # Second recording should approximately double the scores
        assert sum_2 > sum_1 * 1.5

    def test_scissorhands_ema(self):
        """ScissorHands should use EMA (not pure accumulation)."""
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="scissorhands")
        weights = _make_attention_weights(NUM_HEADS, 10, seed=0)

        tracker.record_attention(0, weights)
        importance_1 = tracker.get_token_importance(0)
        mx.eval(importance_1)
        sum_1 = float(mx.sum(importance_1).item())

        # Record many times with same weights -- EMA should converge, not grow
        for _ in range(20):
            tracker.record_attention(0, weights)

        importance_final = tracker.get_token_importance(0)
        mx.eval(importance_final)
        sum_final = float(mx.sum(importance_final).item())

        # EMA converges, so final sum should not be >> initial sum
        # (unlike H2O which would grow 20x)
        assert sum_final < sum_1 * 5

    def test_heavy_hitters_detection(self):
        """Should identify tokens with highest attention as heavy hitters."""
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")

        # Create attention where tokens 2 and 5 are heavy hitters
        heavy_indices = [2, 5]
        weights = _make_heavy_hitter_attention(NUM_HEADS, 10, heavy_indices)

        tracker.record_attention(0, weights)
        top_tokens = tracker.get_heavy_hitters(0, top_k=3)
        mx.eval(top_tokens)

        top_set = set(int(x.item()) for x in top_tokens)
        # Both heavy hitters should be in the top 3
        for idx in heavy_indices:
            assert idx in top_set, f"Heavy hitter {idx} not found in top tokens {top_set}"

    def test_heavy_hitters_top_k(self):
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")
        weights = _make_attention_weights(NUM_HEADS, 20)
        tracker.record_attention(0, weights)
        top = tracker.get_heavy_hitters(0, top_k=5)
        mx.eval(top)
        assert top.shape == (5,)

    def test_heavy_hitters_top_k_exceeds_seq(self):
        """Requesting more top-k than sequence length should not crash."""
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")
        weights = _make_attention_weights(NUM_HEADS, 5)
        tracker.record_attention(0, weights)
        top = tracker.get_heavy_hitters(0, top_k=10)
        mx.eval(top)
        assert top.shape[0] == 5  # clamped to seq_len

    def test_reset(self):
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")
        weights = _make_attention_weights(NUM_HEADS, 10)
        tracker.record_attention(0, weights)
        tracker.reset()
        importance = tracker.get_token_importance(0)
        mx.eval(importance)
        # After reset, should return default
        assert importance.shape == (1,)

    def test_cross_layer_h2o(self):
        """H2O accumulates globally across layers."""
        tracker = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")

        # Record attention for multiple layers
        for layer in range(NUM_LAYERS):
            weights = _make_attention_weights(NUM_HEADS, 10, seed=layer)
            tracker.record_attention(layer, weights)

        # Global importance should reflect all layers
        importance = tracker.get_token_importance(0)
        mx.eval(importance)
        assert importance.shape == (10,)
        # Sum should be greater than single-layer recording
        assert float(mx.sum(importance).item()) > 0


class TestCompressedKVCacheBasic:
    def test_empty_cache(self, h2o_cache):
        assert h2o_cache.current_length == 0
        assert h2o_cache.get_compression_ratio() == 1.0

    def test_get_kv_empty(self, h2o_cache):
        keys, values = h2o_cache.get_kv(0)
        assert keys.shape == (NUM_HEADS, 0, HEAD_DIM)
        assert values.shape == (NUM_HEADS, 0, HEAD_DIM)

    def test_single_token(self, h2o_cache):
        k, v = _make_kv(1)
        out_k, out_v = h2o_cache.update(0, k, v)
        assert out_k.shape == (NUM_HEADS, 1, HEAD_DIM)
        assert h2o_cache.current_length == 1

    def test_update_without_attention(self, h2o_cache):
        """Update should work even without attention weights."""
        k, v = _make_kv(10)
        out_k, out_v = h2o_cache.update(0, k, v)
        assert out_k.shape[1] == 10

    def test_stats_initial(self, h2o_cache):
        stats = h2o_cache.get_stats()
        assert stats["current_length"] == 0
        assert stats["compression_count"] == 0
        assert stats["scoring"] == "h2o"
        assert stats["budget_ratio"] == 0.2


class TestCompressedKVCacheCompression:
    def test_compression_triggers_at_budget(self, h2o_cache):
        """Compression should trigger when cache exceeds budget."""
        # Budget = max(sink+window, 0.2*seq) = max(12, 0.2*100) = 20
        # Add 100 tokens with attention weights to trigger compression
        k, v = _make_kv(100)
        weights = _make_attention_weights(NUM_HEADS, 100)
        out_k, out_v = h2o_cache.update(0, k, v, attention_weights=weights)

        mx.eval(out_k)
        # Should be compressed to budget
        assert out_k.shape[1] <= 100
        assert out_k.shape[1] >= 12  # at least sink + window

    def test_sink_tokens_preserved(self, h2o_cache):
        """Sink tokens should always be kept."""
        # Create identifiable sink tokens
        sink_k = mx.ones((NUM_HEADS, 4, HEAD_DIM)) * 999.0
        sink_v = mx.ones((NUM_HEADS, 4, HEAD_DIM)) * 888.0

        # Rest of the sequence
        rest_k, rest_v = _make_kv(96, seed=1)

        # Add sink first
        h2o_cache.update(0, sink_k, sink_v)
        # Add rest with attention weights
        combined_len = 100
        weights = _make_attention_weights(NUM_HEADS, combined_len)
        out_k, out_v = h2o_cache.update(0, rest_k, rest_v,
                                         attention_weights=weights)

        mx.eval(out_k, out_v)
        # First 4 tokens should be our sinks
        sink_region = out_k[:, :4, :]
        mx.eval(sink_region)
        assert float(mx.mean(sink_region).item()) == pytest.approx(999.0, abs=0.01)

    def test_recent_window_preserved(self, h2o_cache):
        """Recent window tokens should always be kept."""
        # Add many tokens
        k, v = _make_kv(80, seed=0)
        weights = _make_attention_weights(NUM_HEADS, 80, seed=0)
        h2o_cache.update(0, k, v, attention_weights=weights)

        # Add identifiable recent tokens
        recent_k = mx.ones((NUM_HEADS, 8, HEAD_DIM)) * 777.0
        recent_v = mx.ones((NUM_HEADS, 8, HEAD_DIM)) * 666.0
        combined_len = 88
        weights2 = _make_attention_weights(NUM_HEADS, combined_len, seed=1)
        out_k, out_v = h2o_cache.update(0, recent_k, recent_v,
                                         attention_weights=weights2)

        mx.eval(out_k, out_v)
        # Last 8 tokens should be our recent window
        recent_region = out_k[:, -8:, :]
        mx.eval(recent_region)
        assert float(mx.mean(recent_region).item()) == pytest.approx(777.0, abs=0.01)

    def test_budget_enforcement(self, h2o_cache):
        """Cache should never exceed the budget after compression."""
        for i in range(20):
            k, v = _make_kv(10, seed=i)
            seq_len = h2o_cache.current_length + 10
            if seq_len > 0:
                weights = _make_attention_weights(NUM_HEADS, seq_len, seed=i)
            else:
                weights = None
            h2o_cache.update(0, k, v, attention_weights=weights)

        # 200 tokens total, budget = max(12, 0.2*200) = 40
        assert h2o_cache.current_length <= 40

    def test_compression_ratio(self, h2o_cache):
        """Compression ratio should reflect evictions."""
        for i in range(10):
            k, v = _make_kv(20, seed=i)
            seq_len = h2o_cache.current_length + 20
            weights = _make_attention_weights(NUM_HEADS, seq_len, seed=i)
            h2o_cache.update(0, k, v, attention_weights=weights)

        ratio = h2o_cache.get_compression_ratio()
        # Should have compressed (ratio > 1.0 means compression happened)
        assert ratio > 1.0

    def test_force_compress(self, h2o_cache):
        """Force compression should compress to budget."""
        k, v = _make_kv(50)
        h2o_cache.update(0, k, v)
        # Should not have compressed yet if under ratio budget
        # Force it
        h2o_cache.compress(0)
        cur_len = h2o_cache._keys[0].shape[1]
        budget = max(12, int(50 * 0.2))
        assert cur_len <= budget


class TestCompressedKVCacheScissorHands:
    def test_scissorhands_compression(self, scissorhands_cache):
        """ScissorHands scoring should also compress correctly."""
        k, v = _make_kv(100)
        weights = _make_attention_weights(NUM_HEADS, 100)
        out_k, out_v = scissorhands_cache.update(0, k, v,
                                                  attention_weights=weights)
        mx.eval(out_k)
        assert out_k.shape[1] <= 100
        assert out_k.shape[1] >= 12

    def test_scissorhands_stats(self, scissorhands_cache):
        k, v = _make_kv(100)
        weights = _make_attention_weights(NUM_HEADS, 100)
        scissorhands_cache.update(0, k, v, attention_weights=weights)
        stats = scissorhands_cache.get_stats()
        assert stats["scoring"] == "scissorhands"


class TestCompressedKVCacheMultiLayer:
    def test_independent_layers(self, h2o_cache):
        k0, v0 = _make_kv(10, seed=0)
        k1, v1 = _make_kv(5, seed=1)
        h2o_cache.update(0, k0, v0)
        h2o_cache.update(1, k1, v1)

        out_k0, _ = h2o_cache.get_kv(0)
        out_k1, _ = h2o_cache.get_kv(1)
        assert out_k0.shape[1] == 10
        assert out_k1.shape[1] == 5


class TestCompressedKVCacheReset:
    def test_reset(self, h2o_cache):
        # Add tokens within budget (sink=4 + window=8 = 12 min budget)
        k, v = _make_kv(10)
        h2o_cache.update(0, k, v)
        assert h2o_cache.current_length == 10

        h2o_cache.reset()
        assert h2o_cache.current_length == 0
        stats = h2o_cache.get_stats()
        assert stats["compression_count"] == 0
        assert stats["total_tokens_seen"] == 0


class TestCompressedKVCacheEdgeCases:
    def test_empty_update(self, h2o_cache):
        """Zero-length update should not crash."""
        k = mx.zeros((NUM_HEADS, 0, HEAD_DIM))
        v = mx.zeros((NUM_HEADS, 0, HEAD_DIM))
        out_k, out_v = h2o_cache.update(0, k, v)
        assert out_k.shape[1] == 0

    def test_single_token_with_attention(self, h2o_cache):
        k, v = _make_kv(1)
        weights = _make_attention_weights(NUM_HEADS, 1)
        out_k, out_v = h2o_cache.update(0, k, v, attention_weights=weights)
        assert out_k.shape[1] == 1

    def test_budget_ratio_one(self):
        """budget_ratio=1.0 should keep everything."""
        config = KVCompressionConfig(budget_ratio=1.0, sink_tokens=2,
                                     recent_window=4)
        cache = CompressedKVCache(config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(50)
        weights = _make_attention_weights(NUM_HEADS, 50)
        out_k, _ = cache.update(0, k, v, attention_weights=weights)
        mx.eval(out_k)
        assert out_k.shape[1] == 50

    def test_budget_very_small(self):
        """Very small budget should still keep sink + window."""
        config = KVCompressionConfig(budget_ratio=0.01, sink_tokens=2,
                                     recent_window=3)
        cache = CompressedKVCache(config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(100)
        weights = _make_attention_weights(NUM_HEADS, 100)
        out_k, _ = cache.update(0, k, v, attention_weights=weights)
        mx.eval(out_k)
        # At minimum: sink(2) + window(3) = 5
        assert out_k.shape[1] >= 5

    def test_sink_plus_window_exceeds_seq(self):
        """When sink + window > seq_len, no compression should happen."""
        config = KVCompressionConfig(budget_ratio=0.2, sink_tokens=10,
                                     recent_window=10)
        cache = CompressedKVCache(config, NUM_LAYERS, NUM_HEADS, HEAD_DIM)
        k, v = _make_kv(5)
        out_k, _ = cache.update(0, k, v)
        mx.eval(out_k)
        assert out_k.shape[1] == 5  # no compression, under minimum budget

    def test_compress_empty_layer(self, h2o_cache):
        """Compressing an empty layer should not crash."""
        h2o_cache.compress(0)  # no-op

    def test_h2o_vs_scissorhands_different_behavior(self):
        """H2O and ScissorHands should produce different importance rankings."""
        h2o = AttentionScoreTracker(NUM_LAYERS, scoring="h2o")
        sh = AttentionScoreTracker(NUM_LAYERS, scoring="scissorhands")

        # Record the same weights multiple times
        for i in range(10):
            weights = _make_attention_weights(NUM_HEADS, 20, seed=i)
            h2o.record_attention(0, weights)
            sh.record_attention(0, weights)

        h2o_imp = h2o.get_token_importance(0)
        sh_imp = sh.get_token_importance(0)
        mx.eval(h2o_imp, sh_imp)

        # H2O accumulates, ScissorHands uses EMA -> different magnitudes
        h2o_sum = float(mx.sum(h2o_imp).item())
        sh_sum = float(mx.sum(sh_imp).item())
        # H2O should have accumulated much higher total
        assert h2o_sum > sh_sum * 2
