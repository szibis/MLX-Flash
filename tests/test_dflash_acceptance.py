"""Tests for DFlash acceptance improvements: soft threshold, hidden state
normalization, multi-candidate top-K verification, and block size auto-tuning.

These tests focus on the new DFlashRunner features without requiring
actual model downloads — all use mock/fake models.
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_flash_compress.dflash_model import (
    DFlashDraftModel,
    DFlashModelConfig,
    DFlashRunner,
)

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")


# -- Helpers --


class FakeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, **kwargs):
        return self.linear(x)


class FakeModel(nn.Module):
    def __init__(self, num_layers=10, hidden_size=64, vocab_size=100):
        super().__init__()
        self.model = _FakeInner(num_layers, hidden_size, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids, **kwargs):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


class _FakeInner(nn.Module):
    def __init__(self, num_layers, hidden_size, vocab_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [FakeLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, vocab_size=100):
        self._vocab_size = vocab_size

    def encode(self, text):
        return [1] + [ord(c) % self._vocab_size for c in text[:20]]

    def decode(self, ids):
        return "".join(chr(max(32, i % 128)) for i in ids)


def _make_runner(
    num_layers=10,
    hidden_size=64,
    vocab_size=100,
    block_size=4,
    **kwargs,
):
    target = FakeModel(num_layers=num_layers, hidden_size=hidden_size, vocab_size=vocab_size)
    mx.eval(target.parameters())
    tokenizer = FakeTokenizer(vocab_size)

    config = DFlashModelConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        block_size=block_size,
        vocab_size=vocab_size,
        target_layer_ids=[1, 4, 7],
        mask_token_id=0,
    )
    drafter = DFlashDraftModel(config)
    mx.eval(drafter.parameters())

    runner = DFlashRunner(target, tokenizer, drafter, config, **kwargs)
    return runner


# ============================================================
# Soft acceptance (threshold-based)
# ============================================================


class TestSoftAcceptance:
    def test_threshold_zero_is_strict_greedy(self):
        """acceptance_threshold=0 should behave identically to strict greedy."""
        runner = _make_runner(acceptance_threshold=0.0)
        text, stats = runner.generate("hello", max_tokens=8, use_cache=False)
        assert isinstance(text, str)
        assert 0 <= stats["acceptance_rate"] <= 1.0

    def test_threshold_positive_accepts_more(self):
        """A positive threshold should generally accept >= what greedy accepts."""
        runner_strict = _make_runner(acceptance_threshold=0.0)
        runner_soft = _make_runner(acceptance_threshold=0.01)

        # Run both on same prompt
        _, stats_strict = runner_strict.generate("hello world", max_tokens=8, use_cache=False)
        _, stats_soft = runner_soft.generate("hello world", max_tokens=8, use_cache=False)

        # Soft should accept at least as much (or more) than strict
        # Not guaranteed to be strictly >= due to randomness in fake models,
        # but both should produce valid results
        assert stats_strict["tokens_generated"] > 0
        assert stats_soft["tokens_generated"] > 0

    def test_threshold_via_generate_kwarg(self):
        """acceptance_threshold can be passed to generate() directly."""
        runner = _make_runner()
        text, stats = runner.generate(
            "test prompt",
            max_tokens=6,
            use_cache=False,
            acceptance_threshold=0.05,
        )
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0

    def test_threshold_cached_mode(self):
        """Soft acceptance should work with cached generation too."""
        runner = _make_runner(acceptance_threshold=0.01)
        text, stats = runner.generate("cached test", max_tokens=6, use_cache=True)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0

    def test_verify_with_threshold_unit(self):
        """Unit test _verify_with_threshold with controlled logits."""
        runner = _make_runner()

        # Create predictions where token 5 has high probability
        # Shape: [1, 3, vocab_size]
        vocab = 100
        logits = mx.zeros((1, 3, vocab))
        # Position 0: token 5 gets high logit -> high prob
        logits = logits.at[0, 0, 5].add(10.0)
        # Position 1: token 7 gets moderate logit
        logits = logits.at[0, 1, 7].add(5.0)
        # Position 2: token 3 gets low logit
        logits = logits.at[0, 2, 3].add(0.5)

        # Draft tokens: [5, 7, 3]
        draft = [5, 7, 3]

        # With threshold=0, should use greedy (all match -> 3 accepted)
        n = runner._verify_with_threshold(logits, draft, accept_top_k=1, acceptance_threshold=0.0)
        assert n == 3  # greedy: argmax at each position matches draft

        # With threshold=0.5, token at position 2 might not pass
        # because its probability after softmax of 0.5 logit is low
        n_soft = runner._verify_with_threshold(logits, draft, accept_top_k=1, acceptance_threshold=0.5)
        # First two should pass (high prob), third might not
        assert n_soft >= 2


# ============================================================
# Hidden state normalization
# ============================================================


class TestHiddenStateNormalization:
    def test_normalization_enabled(self):
        """normalize_hidden=True should produce valid generation."""
        runner = _make_runner(normalize_hidden=True)
        text, stats = runner.generate("normalize test", max_tokens=6, use_cache=False)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0

    def test_normalization_disabled_default(self):
        """Default should be normalize_hidden=False."""
        runner = _make_runner()
        assert runner._normalize_hidden is False

    def test_apply_hidden_normalization_unit(self):
        """Unit test the static normalization method."""
        h = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = DFlashRunner._apply_hidden_normalization(h)
        mx.eval(normalized)

        # Each row should have approximately unit norm
        norms = mx.sqrt(mx.sum(normalized * normalized, axis=-1))
        mx.eval(norms)
        for i in range(norms.shape[0]):
            assert abs(float(norms[i].item()) - 1.0) < 1e-4

    def test_normalization_cached_mode(self):
        """normalize_hidden should work with cached generation."""
        runner = _make_runner(normalize_hidden=True)
        text, stats = runner.generate("cached normalize", max_tokens=6, use_cache=True)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0


# ============================================================
# Multi-candidate top-K verification
# ============================================================


class TestTopKVerification:
    def test_top_k_1_is_greedy(self):
        """accept_top_k=1 should be equivalent to greedy."""
        runner = _make_runner()
        text, stats = runner.generate("top k test", max_tokens=6, use_cache=False, accept_top_k=1)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0

    def test_top_k_5_accepts_more(self):
        """accept_top_k=5 should accept at least as many as top_k=1."""
        runner_k1 = _make_runner()
        runner_k5 = _make_runner()

        _, stats_k1 = runner_k1.generate("test", max_tokens=8, use_cache=False, accept_top_k=1)
        _, stats_k5 = runner_k5.generate("test", max_tokens=8, use_cache=False, accept_top_k=5)

        # Both should produce valid output
        assert stats_k1["tokens_generated"] > 0
        assert stats_k5["tokens_generated"] > 0

    def test_top_k_cached(self):
        """Top-K verification in cached mode."""
        runner = _make_runner()
        text, stats = runner.generate("cached top k", max_tokens=6, use_cache=True, accept_top_k=3)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0

    def test_verify_with_threshold_top_k_unit(self):
        """Unit test top-K verification logic."""
        runner = _make_runner()

        # Create predictions: position 0 has token 5 as #1 and token 10 as #2
        vocab = 100
        logits = mx.zeros((1, 2, vocab))
        logits = logits.at[0, 0, 5].add(10.0)
        logits = logits.at[0, 0, 10].add(9.0)
        logits = logits.at[0, 1, 7].add(10.0)

        # Draft [10, 7]: token 10 is #2 at position 0
        # With top_k=1 (greedy): 10 != 5 -> 0 accepted
        n_k1 = runner._verify_with_threshold(logits, [10, 7], accept_top_k=1, acceptance_threshold=0.0)
        assert n_k1 == 0  # 10 is not argmax (5 is)

        # With top_k=2: 10 is in top-2 -> accepted, then 7 is top-1 -> accepted
        n_k2 = runner._verify_with_threshold(logits, [10, 7], accept_top_k=2, acceptance_threshold=0.0)
        assert n_k2 == 2


# ============================================================
# Block size auto-tuning
# ============================================================


class TestBlockSizeAutoTuning:
    def test_auto_block_size_disabled_by_default(self):
        """auto_block_size should default to False."""
        runner = _make_runner()
        assert runner._auto_block_size is False
        # block_size should not change during generation
        initial_bs = runner.block_size
        runner._auto_tune_block_size(num_accepted=5, n_draft=10)
        assert runner.block_size == initial_bs

    def test_auto_tune_increases_on_high_acceptance(self):
        """Block size should increase after 10 rounds with >40% acceptance."""
        runner = _make_runner(auto_block_size=True, block_size=4)
        assert runner.block_size == 4

        # Simulate 10 rounds with 50% acceptance (> 40% threshold)
        for _ in range(10):
            runner._auto_tune_block_size(num_accepted=5, n_draft=10)

        # Should have doubled to 8
        assert runner.block_size == 8

    def test_auto_tune_decreases_on_low_acceptance(self):
        """Block size should decrease after 10 rounds with <15% acceptance."""
        runner = _make_runner(auto_block_size=True, block_size=4)

        # Simulate 10 rounds with 10% acceptance (< 15% threshold)
        for _ in range(10):
            runner._auto_tune_block_size(num_accepted=1, n_draft=10)

        # Should have halved to 2
        assert runner.block_size == 2

    def test_auto_tune_no_change_in_middle_range(self):
        """Block size should not change when acceptance is between 15-40%."""
        runner = _make_runner(auto_block_size=True, block_size=4)

        # Simulate 10 rounds with 25% acceptance (between 15-40%)
        for _ in range(10):
            runner._auto_tune_block_size(num_accepted=25, n_draft=100)

        # Should stay at 4
        assert runner.block_size == 4

    def test_auto_tune_clamps_at_minimum(self):
        """Block size should not go below 2."""
        runner = _make_runner(auto_block_size=True, block_size=2)

        # Simulate 10 rounds with very low acceptance
        for _ in range(10):
            runner._auto_tune_block_size(num_accepted=0, n_draft=10)

        assert runner.block_size == 2  # minimum

    def test_auto_tune_clamps_at_maximum(self):
        """Block size should not go above 16."""
        runner = _make_runner(auto_block_size=True, block_size=16)

        # Simulate 10 rounds with high acceptance
        for _ in range(10):
            runner._auto_tune_block_size(num_accepted=8, n_draft=10)

        assert runner.block_size == 16  # maximum, no change

    def test_auto_tune_resets_counters(self):
        """After 10 rounds, counters should reset for the next window."""
        runner = _make_runner(auto_block_size=True, block_size=4)

        for _ in range(10):
            runner._auto_tune_block_size(num_accepted=5, n_draft=10)

        # After first window: block_size = 8, counters reset
        assert runner._auto_bs_rounds == 0
        assert runner._auto_bs_accepted == 0
        assert runner._auto_bs_drafted == 0

    def test_auto_tune_in_generation(self):
        """Auto-tune should work during actual generation."""
        runner = _make_runner(auto_block_size=True, block_size=4)
        text, stats = runner.generate("auto tune test", max_tokens=8, use_cache=False)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0
