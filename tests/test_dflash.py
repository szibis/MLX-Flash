"""Comprehensive tests for DFlash speculative decoding engine (dflash.py).

Tests DFlashConfig, DFlashStats, NGramDrafter, BlockDiffusionDrafter,
DrafterBlock, and DFlashEngine including the fallback generation path.

NOTE: Tests for DFlashDraftModel/DFlashRunner live in test_dflash_model.py.
      Tests for DDTree integration live in test_dflash_ddtree.py.
      This file focuses on the core dflash.py module.
"""

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_flash_compress.dflash import (
        BlockDiffusionDrafter,
        DFlashConfig,
        DFlashEngine,
        DFlashStats,
        DrafterBlock,
        NGramDrafter,
    )

    HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="dflash requires mlx")


# -- Helper mock models --


class MockTargetModel(nn.Module):
    """Minimal target model for testing DFlashEngine."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32, num_layers: int = 4):
        super().__init__()
        self.model = MockInnerModel(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        if len(x.shape) == 1:
            x = mx.expand_dims(x, axis=0)
        h = self.model(x)
        return self.lm_head(h)


class MockInnerModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [MockLayer(hidden_dim) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(self, x):
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class MockLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, **kwargs):
        return x + self.linear(x)


class MockTokenizer:
    eos_token_id = 2

    def encode(self, text):
        return [1, 3, 4, 5]

    def decode(self, ids):
        return "".join(chr(max(32, i % 128)) for i in ids)


# ============================================================
# DFlashConfig tests
# ============================================================


class TestDFlashConfig:
    def test_defaults(self):
        cfg = DFlashConfig()
        assert cfg.num_spec_tokens == 15
        assert cfg.num_denoise_steps == 2
        assert cfg.checkpoint_layers == []
        assert cfg.temperature == 0.0
        assert cfg.mask_token_id == -1
        assert cfg.tree_width == 1

    def test_custom_values(self):
        cfg = DFlashConfig(
            num_spec_tokens=10,
            num_denoise_steps=4,
            checkpoint_layers=[0, 5, 10],
            temperature=0.7,
            mask_token_id=99,
            tree_width=3,
        )
        assert cfg.num_spec_tokens == 10
        assert cfg.num_denoise_steps == 4
        assert cfg.checkpoint_layers == [0, 5, 10]
        assert cfg.temperature == 0.7
        assert cfg.mask_token_id == 99
        assert cfg.tree_width == 3

    def test_checkpoint_layers_mutable(self):
        """Checkpoint layers list should be independent per instance."""
        cfg1 = DFlashConfig()
        cfg2 = DFlashConfig()
        cfg1.checkpoint_layers.append(42)
        assert 42 not in cfg2.checkpoint_layers

    def test_zero_denoise_steps(self):
        cfg = DFlashConfig(num_denoise_steps=0)
        assert cfg.num_denoise_steps == 0

    def test_single_spec_token(self):
        cfg = DFlashConfig(num_spec_tokens=1)
        assert cfg.num_spec_tokens == 1


# ============================================================
# DFlashStats tests
# ============================================================


class TestDFlashStats:
    def test_empty_stats(self):
        stats = DFlashStats()
        assert stats.acceptance_rate == 0.0
        assert stats.tokens_per_draft == 0.0
        assert stats.speedup_factor == 1.0

    def test_acceptance_rate(self):
        stats = DFlashStats(total_drafts=200, total_accepted=150)
        assert stats.acceptance_rate == 0.75

    def test_tokens_per_draft(self):
        stats = DFlashStats(total_accepted=50, total_target_calls=10)
        assert stats.tokens_per_draft == 5.0

    def test_speedup_factor(self):
        stats = DFlashStats(total_accepted=100, total_target_calls=20)
        assert stats.speedup_factor == 5.0

    def test_stats_with_zero_drafts(self):
        stats = DFlashStats(total_drafts=0, total_accepted=0, total_target_calls=0)
        assert stats.acceptance_rate == 0.0
        assert stats.tokens_per_draft == 0.0
        assert stats.speedup_factor == 1.0

    def test_draft_times_empty(self):
        stats = DFlashStats()
        assert stats.draft_times_ms == []
        assert stats.verify_times_ms == []

    def test_draft_times_accumulation(self):
        stats = DFlashStats()
        stats.draft_times_ms.append(10.0)
        stats.draft_times_ms.append(20.0)
        assert len(stats.draft_times_ms) == 2
        assert np.mean(stats.draft_times_ms) == 15.0

    def test_perfect_acceptance(self):
        stats = DFlashStats(total_drafts=100, total_accepted=100, total_target_calls=10)
        assert stats.acceptance_rate == 1.0
        assert stats.tokens_per_draft == 10.0

    def test_stats_independence(self):
        """Stats instances should not share mutable state."""
        s1 = DFlashStats()
        s2 = DFlashStats()
        s1.draft_times_ms.append(5.0)
        assert len(s2.draft_times_ms) == 0


# ============================================================
# NGramDrafter tests
# ============================================================


class TestNGramDrafter:
    def test_defaults(self):
        drafter = NGramDrafter()
        assert drafter.n == 4
        assert drafter.num_draft == 15
        assert drafter._ngram_table == {}

    def test_custom_params(self):
        drafter = NGramDrafter(n=2, num_draft=5)
        assert drafter.n == 2
        assert drafter.num_draft == 5

    def test_observe_builds_table(self):
        drafter = NGramDrafter(n=3, num_draft=5)
        tokens = [1, 2, 3, 4, 5, 6]
        drafter.observe(tokens)
        assert (1, 2, 3) in drafter._ngram_table
        assert drafter._ngram_table[(1, 2, 3)] == [4]

    def test_observe_and_draft_basic(self):
        drafter = NGramDrafter(n=3, num_draft=5)
        tokens = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]
        drafter.observe(tokens)
        drafts = drafter.draft([1, 2, 3])
        assert len(drafts) > 0
        assert drafts[0] == 4

    def test_draft_empty_table(self):
        drafter = NGramDrafter(n=3, num_draft=5)
        drafts = drafter.draft([1, 2, 3])
        assert drafts == []

    def test_draft_no_match(self):
        drafter = NGramDrafter(n=3, num_draft=5)
        drafter.observe([10, 20, 30, 40])
        drafts = drafter.draft([1, 2, 3])
        assert drafts == []

    def test_draft_chain(self):
        """Should predict chains when the pattern repeats."""
        drafter = NGramDrafter(n=2, num_draft=10)
        tokens = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        drafter.observe(tokens)
        drafts = drafter.draft([3, 4])
        assert len(drafts) > 0
        assert drafts[0] == 1

    def test_draft_most_common(self):
        """Should pick most common continuation when multiple exist."""
        drafter = NGramDrafter(n=2, num_draft=3)
        # After [1,2], token 3 appears twice, token 9 appears once
        drafter.observe([1, 2, 3, 1, 2, 3, 1, 2, 9])
        drafts = drafter.draft([1, 2])
        assert drafts[0] == 3

    def test_draft_respects_num_draft(self):
        drafter = NGramDrafter(n=2, num_draft=3)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
        drafter.observe(tokens)
        drafts = drafter.draft([1, 2])
        assert len(drafts) <= 3

    def test_single_token_context(self):
        """When context is shorter than n, key is the last n tokens."""
        drafter = NGramDrafter(n=3, num_draft=5)
        drafter.observe([1, 2, 3, 4])
        # Context [3] -> key would be (3,) but n=3 expects 3-gram
        # Since context is [3], key = tuple([3][-3:]) = (3,) which won't match (1,2,3)
        drafts = drafter.draft([3])
        assert drafts == []

    def test_observe_multiple_times(self):
        """Multiple observe calls should accumulate the table."""
        drafter = NGramDrafter(n=2, num_draft=5)
        drafter.observe([1, 2, 3])
        drafter.observe([1, 2, 4])
        assert (1, 2) in drafter._ngram_table
        assert 3 in drafter._ngram_table[(1, 2)]
        assert 4 in drafter._ngram_table[(1, 2)]


# ============================================================
# DrafterBlock tests
# ============================================================


class TestDrafterBlock:
    def test_forward_preserves_shape(self):
        block = DrafterBlock(hidden_dim=64, num_heads=4)
        x = mx.random.normal((1, 8, 64))
        out = block(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input due to attention + MLP."""
        block = DrafterBlock(hidden_dim=32, num_heads=2)
        mx.eval(block.parameters())
        x = mx.random.normal((1, 4, 32))
        out = block(x)
        mx.eval(out)
        assert not mx.array_equal(x, out)

    def test_batch_dimension(self):
        block = DrafterBlock(hidden_dim=32, num_heads=2)
        x = mx.random.normal((3, 6, 32))
        out = block(x)
        mx.eval(out)
        assert out.shape == (3, 6, 32)


# ============================================================
# BlockDiffusionDrafter tests
# ============================================================


class TestBlockDiffusionDrafter:
    def test_forward_shape(self):
        vocab_size = 100
        hidden_dim = 64
        drafter = BlockDiffusionDrafter(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_draft_positions=8,
            num_checkpoint_layers=3,
        )

        hidden_states = [mx.random.normal((1, 5, hidden_dim)) for _ in range(3)]
        noisy_tokens = mx.random.normal((1, 8, hidden_dim))
        logits = drafter(hidden_states, noisy_tokens)
        mx.eval(logits)
        assert logits.shape == (1, 8, vocab_size)

    def test_different_vocab_sizes(self):
        for vocab in [32, 256, 1000]:
            drafter = BlockDiffusionDrafter(
                vocab_size=vocab,
                hidden_dim=32,
                num_layers=1,
                num_heads=2,
                num_draft_positions=4,
                num_checkpoint_layers=2,
            )
            hs = [mx.random.normal((1, 3, 32)) for _ in range(2)]
            x = mx.random.normal((1, 4, 32))
            out = drafter(hs, x)
            mx.eval(out)
            assert out.shape[-1] == vocab

    def test_single_checkpoint_layer(self):
        drafter = BlockDiffusionDrafter(
            vocab_size=50,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            num_draft_positions=4,
            num_checkpoint_layers=1,
        )
        hs = [mx.random.normal((1, 5, 32))]
        x = mx.random.normal((1, 4, 32))
        out = drafter(hs, x)
        mx.eval(out)
        assert out.shape == (1, 4, 50)

    def test_single_draft_position(self):
        drafter = BlockDiffusionDrafter(
            vocab_size=50,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            num_draft_positions=1,
            num_checkpoint_layers=2,
        )
        hs = [mx.random.normal((1, 3, 32)) for _ in range(2)]
        x = mx.random.normal((1, 1, 32))
        out = drafter(hs, x)
        mx.eval(out)
        assert out.shape == (1, 1, 50)


# ============================================================
# DFlashEngine tests
# ============================================================


class TestDFlashEngine:
    def _make_target(self, vocab_size=64, hidden_dim=32, num_layers=4):
        model = MockTargetModel(vocab_size, hidden_dim, num_layers)
        mx.eval(model.parameters())
        return model

    def _make_drafter(self, vocab_size=64, hidden_dim=32, num_draft=8, num_ckpt=3):
        drafter = BlockDiffusionDrafter(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_draft_positions=num_draft,
            num_checkpoint_layers=num_ckpt,
        )
        mx.eval(drafter.parameters())
        return drafter

    def test_init_no_drafter(self):
        target = self._make_target(num_layers=8)
        config = DFlashConfig()
        engine = DFlashEngine(target, drafter=None, config=config)
        assert engine.drafter is None
        assert len(config.checkpoint_layers) == 5

    def test_init_with_drafter(self):
        target = self._make_target()
        drafter = self._make_drafter()
        config = DFlashConfig()
        engine = DFlashEngine(target, drafter, config)
        assert engine.drafter is drafter

    def test_auto_detect_checkpoints(self):
        target = self._make_target(num_layers=32)
        config = DFlashConfig()
        engine = DFlashEngine(target, drafter=None, config=config)
        assert len(config.checkpoint_layers) == 5
        assert config.checkpoint_layers[0] == 1
        assert config.checkpoint_layers[-1] == 31

    def test_auto_detect_checkpoints_small_model(self):
        """Model with few layers should still get checkpoint layers."""

        class TinyModel:
            layers = [None, None, None]

        config = DFlashConfig()
        engine = DFlashEngine(TinyModel(), drafter=None, config=config)
        assert len(config.checkpoint_layers) == 5
        assert config.checkpoint_layers[0] == 1
        assert config.checkpoint_layers[-1] == 2

    def test_auto_detect_checkpoints_no_layers(self):
        """Model with no detectable layers uses fallback."""

        class BareboneModel:
            pass

        config = DFlashConfig()
        engine = DFlashEngine(BareboneModel(), drafter=None, config=config)
        # Falls through to num_layers=32 fallback
        assert len(config.checkpoint_layers) == 5

    def test_provided_checkpoint_layers_not_overwritten(self):
        target = self._make_target(num_layers=16)
        config = DFlashConfig(checkpoint_layers=[2, 4, 6])
        engine = DFlashEngine(target, drafter=None, config=config)
        assert config.checkpoint_layers == [2, 4, 6]

    def test_get_model_num_layers_model_model(self):
        target = self._make_target(num_layers=8)
        config = DFlashConfig(checkpoint_layers=[0])
        engine = DFlashEngine(target, drafter=None, config=config)
        assert engine._get_model_num_layers() == 8

    def test_get_model_num_layers_direct(self):
        class DirectModel:
            layers = [None] * 10

        config = DFlashConfig(checkpoint_layers=[0])
        engine = DFlashEngine(DirectModel(), drafter=None, config=config)
        assert engine._get_model_num_layers() == 10

    def test_get_model_num_layers_fallback(self):
        class UnknownModel:
            pass

        config = DFlashConfig(checkpoint_layers=[0])
        engine = DFlashEngine(UnknownModel(), drafter=None, config=config)
        assert engine._get_model_num_layers() == 32

    def test_stats_summary_format(self):
        config = DFlashConfig()
        engine = DFlashEngine(None, drafter=None, config=config)
        engine.stats.total_accepted = 100
        engine.stats.total_target_calls = 10
        engine.stats.total_drafter_calls = 10
        engine.stats.total_drafts = 150
        engine.stats.draft_times_ms = [5.0, 6.0, 7.0]
        engine.stats.verify_times_ms = [20.0, 25.0]

        summary = engine.get_stats_summary()
        assert "total_tokens_generated" in summary
        assert "acceptance_rate" in summary
        assert "tokens_per_draft_step" in summary
        assert "effective_speedup" in summary
        assert "avg_draft_time_ms" in summary
        assert "avg_verify_time_ms" in summary
        assert summary["total_tokens_generated"] == 100

    def test_stats_summary_empty(self):
        config = DFlashConfig()
        engine = DFlashEngine(None, drafter=None, config=config)
        summary = engine.get_stats_summary()
        assert summary["avg_draft_time_ms"] == "N/A"
        assert summary["avg_verify_time_ms"] == "N/A"

    def test_install_hidden_state_hooks(self):
        target = self._make_target(num_layers=8)
        config = DFlashConfig(checkpoint_layers=[0, 3, 7])
        engine = DFlashEngine(target, drafter=None, config=config)
        engine.install_hidden_state_hooks()
        assert 0 in engine._captured_hidden_states
        assert 3 in engine._captured_hidden_states
        assert 7 in engine._captured_hidden_states

    def test_install_hooks_clears_previous(self):
        target = self._make_target(num_layers=8)
        config = DFlashConfig(checkpoint_layers=[0, 3])
        engine = DFlashEngine(target, drafter=None, config=config)
        engine._captured_hidden_states = {99: "stale"}
        engine.install_hidden_state_hooks()
        assert 99 not in engine._captured_hidden_states

    def test_install_hooks_no_layers(self):
        class NoLayers:
            pass

        config = DFlashConfig(checkpoint_layers=[0, 1])
        engine = DFlashEngine(NoLayers(), drafter=None, config=config)
        engine.install_hidden_state_hooks()
        assert engine._captured_hidden_states == {}

    def test_get_checkpoint_hidden_states_with_model(self):
        target = self._make_target(vocab_size=64, hidden_dim=32, num_layers=8)
        config = DFlashConfig(checkpoint_layers=[1, 3, 5])
        engine = DFlashEngine(target, drafter=None, config=config)

        input_ids = mx.array([1, 2, 3, 4])
        hs = engine._get_checkpoint_hidden_states(input_ids)
        assert len(hs) == 3
        for h in hs:
            mx.eval(h)
            assert h.shape[0] == 1  # batch
            assert h.shape[1] == 4  # seq_len
            assert h.shape[2] == 32  # hidden_dim

    def test_get_checkpoint_hidden_states_fallback(self):
        """Model with no layers returns dummy hidden states."""

        class NoLayers:
            pass

        config = DFlashConfig(checkpoint_layers=[0, 1, 2])
        engine = DFlashEngine(NoLayers(), drafter=None, config=config)

        input_ids = mx.array([1, 2, 3])
        hs = engine._get_checkpoint_hidden_states(input_ids)
        assert len(hs) == 3
        for h in hs:
            assert h.shape == (1, 1, 4096)

    def test_target_forward_callable(self):
        target = self._make_target(vocab_size=64, hidden_dim=32)
        config = DFlashConfig(checkpoint_layers=[0])
        engine = DFlashEngine(target, drafter=None, config=config)

        input_ids = mx.array([[1, 2, 3]])
        logits = engine._target_forward(input_ids)
        mx.eval(logits)
        assert logits.shape[-1] == 64

    def test_target_forward_raises_on_unknown(self):
        class NotCallable:
            pass

        config = DFlashConfig(checkpoint_layers=[0])
        engine = DFlashEngine(NotCallable(), drafter=None, config=config)

        with pytest.raises(RuntimeError, match="Cannot determine target model forward method"):
            engine._target_forward(mx.array([[1, 2]]))

    def test_fallback_generation_produces_tokens(self):
        """Fallback mode should generate tokens autoregressively."""
        target = self._make_target(vocab_size=64, hidden_dim=32, num_layers=4)
        config = DFlashConfig()
        engine = DFlashEngine(target, drafter=None, config=config)

        prompt = mx.array([1, 2, 3])
        result = engine.generate(prompt, max_tokens=5)
        mx.eval(result)

        result_list = result.tolist()
        assert len(result_list) > 3  # prompt + at least some generated tokens
        assert result_list[:3] == [1, 2, 3]

    def test_fallback_generation_respects_max_tokens(self):
        target = self._make_target(vocab_size=64, hidden_dim=32, num_layers=4)
        config = DFlashConfig()
        engine = DFlashEngine(target, drafter=None, config=config)

        prompt = mx.array([1, 2, 3])
        result = engine.generate(prompt, max_tokens=3)
        mx.eval(result)

        result_list = result.tolist()
        new_tokens = len(result_list) - 3
        assert new_tokens <= 3

    def test_fallback_generation_with_callback(self):
        target = self._make_target(vocab_size=64, hidden_dim=32, num_layers=4)
        config = DFlashConfig()
        engine = DFlashEngine(target, drafter=None, config=config)

        callbacks = []

        def on_token(tokens, stats):
            callbacks.append(tokens)

        prompt = mx.array([1, 2, 3])
        engine.generate(prompt, max_tokens=3, callback=on_token)

        assert len(callbacks) > 0
        for cb in callbacks:
            assert isinstance(cb, list)
            assert len(cb) == 1  # fallback emits one token at a time

    def test_fallback_generation_stops_at_eos(self):
        """Fallback should stop when EOS token is generated."""
        # Create a model that always outputs token 2 (eos_token_id)

        class AlwaysEosModel:
            def __call__(self, x):
                batch, seq = x.shape
                # Return logits where token 2 is dominant
                logits = mx.zeros((batch, seq, 64))
                logits = logits.at[:, :, 2].add(100.0)
                return logits

        config = DFlashConfig()
        tokenizer = MockTokenizer()  # eos_token_id=2
        engine = DFlashEngine(AlwaysEosModel(), drafter=None, config=config, tokenizer=tokenizer)

        prompt = mx.array([1, 3, 4])
        result = engine.generate(prompt, max_tokens=50)
        mx.eval(result)

        result_list = result.tolist()
        # Should have stopped after generating EOS (token 2)
        # prompt (3) + 1 generated token
        assert len(result_list) == 4
        assert result_list[-1] == 2

    def test_generate_resets_stats(self):
        """generate() should reset stats each call."""
        target = self._make_target(vocab_size=64, hidden_dim=32, num_layers=4)
        config = DFlashConfig()
        engine = DFlashEngine(target, drafter=None, config=config)

        # First generate
        engine.generate(mx.array([1, 2]), max_tokens=2)
        # Second generate should have fresh stats
        engine.generate(mx.array([1, 2]), max_tokens=2)
        # (no crash is the test here)


# ============================================================
# DFlashEngine with drafter (draft-verify loop)
# ============================================================


class TestDFlashEngineDraftVerify:
    def _make_target(self, vocab_size=64, hidden_dim=32, num_layers=4):
        model = MockTargetModel(vocab_size, hidden_dim, num_layers)
        mx.eval(model.parameters())
        return model

    def test_draft_tokens(self):
        hidden_dim = 32
        drafter = BlockDiffusionDrafter(
            vocab_size=64,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_draft_positions=8,
            num_checkpoint_layers=3,
        )
        mx.eval(drafter.parameters())

        target = self._make_target(hidden_dim=hidden_dim, num_layers=8)
        config = DFlashConfig(num_spec_tokens=8, num_denoise_steps=2, checkpoint_layers=[1, 3, 5])
        engine = DFlashEngine(target, drafter, config)

        hidden_states = [mx.random.normal((1, 5, hidden_dim)) for _ in range(3)]
        draft = engine.draft_tokens(hidden_states)
        mx.eval(draft)

        assert draft.shape == (8,)
        assert engine.stats.total_drafter_calls == 1
        assert len(engine.stats.draft_times_ms) == 1

    def test_draft_tokens_with_temperature(self):
        hidden_dim = 32
        drafter = BlockDiffusionDrafter(
            vocab_size=64,
            hidden_dim=hidden_dim,
            num_layers=1,
            num_heads=2,
            num_draft_positions=4,
            num_checkpoint_layers=2,
        )
        mx.eval(drafter.parameters())

        target = self._make_target(hidden_dim=hidden_dim)
        config = DFlashConfig(num_spec_tokens=4, temperature=0.8, checkpoint_layers=[0, 1])
        engine = DFlashEngine(target, drafter, config)

        hs = [mx.random.normal((1, 3, hidden_dim)) for _ in range(2)]
        draft = engine.draft_tokens(hs)
        mx.eval(draft)
        assert draft.shape == (4,)

    def test_verify_tokens(self):
        target = self._make_target(vocab_size=64, hidden_dim=32, num_layers=4)
        config = DFlashConfig(num_spec_tokens=4, checkpoint_layers=[0, 1])
        engine = DFlashEngine(target, drafter=None, config=config)

        input_ids = mx.array([1, 2, 3, 4, 5])
        draft_tokens = mx.array([10, 11, 12, 13])

        accepted, num_accepted = engine.verify_tokens(input_ids, draft_tokens)
        mx.eval(accepted)

        assert num_accepted >= 0
        assert engine.stats.total_target_calls == 1
        assert engine.stats.total_drafts == 4

    def test_verify_tokens_with_temperature(self):
        target = self._make_target(vocab_size=64, hidden_dim=32, num_layers=4)
        config = DFlashConfig(num_spec_tokens=4, temperature=0.5, checkpoint_layers=[0])
        engine = DFlashEngine(target, drafter=None, config=config)

        input_ids = mx.array([1, 2, 3])
        draft_tokens = mx.array([10, 11, 12, 13])

        accepted, num_accepted = engine.verify_tokens(input_ids, draft_tokens)
        mx.eval(accepted)
        assert num_accepted >= 0
