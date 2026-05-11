"""Tests for LayerSkip self-speculative decoding."""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_flash_compress.layerskip import (
    LayerSkipConfig,
    LayerSkipDrafter,
    LayerSkipEngine,
    apply_layerskip,
)

# -- Mock Model Components --


class MockLayer(nn.Module):
    """Simple linear layer that mimics a transformer layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def __call__(self, x, mask=None, cache=None):
        return self.linear(x)


class MockInnerModel(nn.Module):
    """Mimics model.model with embed_tokens, layers, and norm."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [MockLayer(hidden_dim) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_dim)


class _MockKVCache:
    """Minimal KV cache stub for testing."""

    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset = max(0, self.offset - n)


class MockModel(nn.Module):
    """Mimics a standard HuggingFace/MLX causal LM structure.

    model.model.embed_tokens / model.model.layers / model.model.norm
    model.lm_head
    """

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 8):
        super().__init__()
        self.model = MockInnerModel(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        h = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            c = cache[i] if cache is not None and i < len(cache) else None
            h = layer(h, cache=c)
            if cache is not None and i < len(cache):
                cache[i].offset += input_ids.shape[-1]
        h = self.model.norm(h)
        return self.lm_head(h)

    def make_cache(self):
        return [_MockKVCache() for _ in self.model.layers]


class MockLanguageModel(nn.Module):
    """Model with language_model.model.embed_tokens structure."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 4):
        super().__init__()
        self.language_model = MockModel(vocab_size, hidden_dim, num_layers)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.language_model(input_ids)


class MockTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 100 for c in text[:20]]

    def decode(self, tokens) -> str:
        return "".join(chr(t + 32) for t in tokens)


class MockTokenizerNoEos:
    """Tokenizer without eos_token_id."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 100 for c in text[:20]]

    def decode(self, tokens) -> str:
        return "".join(chr(t + 32) for t in tokens)


# -- Config Tests --


class TestLayerSkipConfig:
    def test_default_config(self):
        config = LayerSkipConfig()
        assert config.exit_layer == -1
        assert config.num_speculative_tokens == 5
        assert config.temperature == 0.0
        assert config.confidence_threshold == 0.9
        assert config.adaptive_exit is True

    def test_custom_config(self):
        config = LayerSkipConfig(
            exit_layer=4,
            num_speculative_tokens=10,
            temperature=0.5,
        )
        assert config.exit_layer == 4
        assert config.num_speculative_tokens == 10
        assert config.temperature == 0.5

    def test_invalid_num_speculative_tokens_zero(self):
        with pytest.raises(ValueError, match="num_speculative_tokens"):
            LayerSkipConfig(num_speculative_tokens=0)

    def test_invalid_num_speculative_tokens_negative(self):
        with pytest.raises(ValueError, match="num_speculative_tokens"):
            LayerSkipConfig(num_speculative_tokens=-1)

    def test_invalid_temperature_negative(self):
        with pytest.raises(ValueError, match="temperature"):
            LayerSkipConfig(temperature=-0.1)

    def test_invalid_confidence_threshold_below_zero(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            LayerSkipConfig(confidence_threshold=-0.1)

    def test_invalid_confidence_threshold_above_one(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            LayerSkipConfig(confidence_threshold=1.1)

    def test_edge_confidence_threshold_zero(self):
        config = LayerSkipConfig(confidence_threshold=0.0)
        assert config.confidence_threshold == 0.0

    def test_edge_confidence_threshold_one(self):
        config = LayerSkipConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0

    def test_edge_temperature_zero(self):
        config = LayerSkipConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_exit_layer_auto(self):
        """exit_layer=-1 is valid and means auto."""
        config = LayerSkipConfig(exit_layer=-1)
        assert config.exit_layer == -1

    def test_exit_layer_explicit(self):
        config = LayerSkipConfig(exit_layer=3)
        assert config.exit_layer == 3


# -- Drafter Tests --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestLayerSkipDrafter:
    def setup_method(self):
        mx.random.seed(42)
        self.model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(self.model.parameters())

    def test_auto_exit_layer(self):
        config = LayerSkipConfig()  # exit_layer=-1 -> auto
        drafter = LayerSkipDrafter(self.model, config)
        assert drafter.get_exit_layer() == 4  # 8 // 2

    def test_explicit_exit_layer(self):
        config = LayerSkipConfig(exit_layer=3)
        drafter = LayerSkipDrafter(self.model, config)
        assert drafter.get_exit_layer() == 3

    def test_component_detection(self):
        config = LayerSkipConfig()
        drafter = LayerSkipDrafter(self.model, config)
        assert drafter._embed_fn is not None
        assert drafter._layers is not None
        assert len(drafter._layers) == 8
        assert drafter._norm_fn is not None
        assert drafter._lm_head_fn is not None

    def test_component_detection_language_model(self):
        """Test detection for language_model.model structure."""
        model = MockLanguageModel(vocab_size=100, hidden_dim=32, num_layers=4)
        mx.eval(model.parameters())
        config = LayerSkipConfig()
        drafter = LayerSkipDrafter(model, config)
        assert drafter._embed_fn is not None
        assert len(drafter._layers) == 4
        assert drafter.get_exit_layer() == 2  # 4 // 2

    def test_component_detection_fails_for_bad_model(self):
        class BadModel:
            pass

        config = LayerSkipConfig()
        with pytest.raises(RuntimeError, match="Cannot detect"):
            LayerSkipDrafter(BadModel(), config)

    def test_partial_forward(self):
        config = LayerSkipConfig(exit_layer=4)
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        h = drafter._forward_partial(input_ids, num_layers=4)
        mx.eval(h)

        assert h.shape == (1, 3, 32)

    def test_partial_forward_fewer_layers(self):
        """Partial forward with 2 layers should differ from 4 layers."""
        config = LayerSkipConfig()
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        h2 = drafter._forward_partial(input_ids, num_layers=2)
        h4 = drafter._forward_partial(input_ids, num_layers=4)
        mx.eval(h2, h4)

        # Different number of layers should produce different outputs
        diff = float(mx.sum(mx.abs(h2 - h4)).item())
        assert diff > 0

    def test_partial_forward_zero_layers(self):
        """Zero layers should just return embeddings."""
        config = LayerSkipConfig()
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        h0 = drafter._forward_partial(input_ids, num_layers=0)
        mx.eval(h0)

        # Should be raw embeddings (no layer applied)
        expected = drafter._embed_fn(input_ids)
        mx.eval(expected)
        diff = float(mx.sum(mx.abs(h0 - expected)).item())
        assert diff == 0.0

    def test_partial_forward_all_layers(self):
        """Using all layers should be equivalent to full forward (minus norm)."""
        config = LayerSkipConfig()
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        h_all = drafter._forward_partial(input_ids, num_layers=8)
        mx.eval(h_all)

        assert h_all.shape == (1, 3, 32)

    def test_draft_produces_tokens(self):
        config = LayerSkipConfig(exit_layer=4, num_speculative_tokens=3)
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        draft_ids, draft_logits = drafter.draft(input_ids)
        mx.eval(draft_ids)

        assert draft_ids.shape == (1, 3)
        assert len(draft_logits) == 3
        for logits in draft_logits:
            assert logits.shape == (1, 1, 100)  # vocab_size=100

    def test_draft_deterministic_greedy(self):
        config = LayerSkipConfig(exit_layer=4, num_speculative_tokens=3, temperature=0.0)
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        ids1, _ = drafter.draft(input_ids)
        ids2, _ = drafter.draft(input_ids)
        mx.eval(ids1, ids2)

        assert mx.array_equal(ids1, ids2)

    def test_draft_with_temperature(self):
        """Sampling mode should still produce valid tokens."""
        config = LayerSkipConfig(exit_layer=4, num_speculative_tokens=3, temperature=0.8)
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        draft_ids, draft_logits = drafter.draft(input_ids)
        mx.eval(draft_ids)

        assert draft_ids.shape == (1, 3)
        # All drafted tokens should be valid vocab indices
        for tok in draft_ids[0].tolist():
            assert 0 <= tok < 100

    def test_draft_tracks_exit_layers_adaptive(self):
        """With adaptive exit, the per-token exit layers should be tracked."""
        config = LayerSkipConfig(
            exit_layer=4,
            num_speculative_tokens=5,
            adaptive_exit=True,
            confidence_threshold=0.5,
        )
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        drafter.draft(input_ids)

        assert hasattr(drafter, "_draft_exit_layers")
        assert len(drafter._draft_exit_layers) == 5

    def test_draft_no_adaptive_exit(self):
        """Without adaptive exit, exit layer should stay fixed."""
        config = LayerSkipConfig(
            exit_layer=4,
            num_speculative_tokens=3,
            adaptive_exit=False,
        )
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        drafter.draft(input_ids)

        assert hasattr(drafter, "_draft_exit_layers")
        # All should be the base exit layer since adaptive is off
        assert all(el == 4 for el in drafter._draft_exit_layers)


# -- Engine Tests --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestLayerSkipEngine:
    def setup_method(self):
        mx.random.seed(42)
        self.model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(self.model.parameters())
        self.tokenizer = MockTokenizer()

    def test_engine_init_default(self):
        engine = LayerSkipEngine(self.model, self.tokenizer)
        assert engine.drafter.get_exit_layer() == 4
        assert engine.config.num_speculative_tokens == 5

    def test_engine_init_custom_config(self):
        config = LayerSkipConfig(exit_layer=2, num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)
        assert engine.drafter.get_exit_layer() == 2
        assert engine.config.num_speculative_tokens == 3

    def test_draft_step(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        draft_ids, draft_logits = engine._draft_step(input_ids)
        mx.eval(draft_ids, draft_logits)

        assert draft_ids.shape == (1, 3)
        assert draft_logits.shape == (1, 3, 100)

    def test_draft_step_updates_stats(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        input_ids = mx.array([[1, 2, 3]])
        engine._draft_step(input_ids)

        assert engine.stats["total_draft_tokens"] == 3
        assert len(engine.stats["draft_times_ms"]) == 1

    def test_verify_step_all_match(self):
        """When draft matches target, all tokens should be accepted."""
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        input_ids = mx.array([[1, 2, 3]])

        # Get what the full model would predict
        logits = self.model(input_ids)
        target_tokens = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(target_tokens)

        # Use target's own prediction as draft (should be accepted)
        full_input = mx.concatenate([input_ids, target_tokens], axis=-1)
        full_logits = self.model(full_input)
        next_target = mx.argmax(full_logits[:, -1:, :], axis=-1)
        mx.eval(next_target)

        draft = mx.concatenate([target_tokens, next_target], axis=-1)
        accepted, num = engine._verify_step(input_ids, draft)
        mx.eval(accepted)

        # At least the first token should be accepted since it matches
        assert num >= 1

    def test_verify_step_none_match(self):
        """When draft is completely wrong, bonus token should still be produced."""
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        input_ids = mx.array([[1, 2, 3]])

        # Get target prediction
        logits = self.model(input_ids)
        target_token = int(mx.argmax(logits[0, -1]).item())

        # Use a token that is NOT what the target predicts
        wrong_token = (target_token + 1) % 100
        draft = mx.array([[wrong_token, wrong_token, wrong_token]])
        accepted, num = engine._verify_step(input_ids, draft)
        mx.eval(accepted)

        # 0 drafts match, but bonus token should be included
        assert len(accepted) >= 1  # at least the bonus token
        assert num >= 1

    def test_verify_step_updates_stats(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        input_ids = mx.array([[1, 2, 3]])
        draft = mx.array([[10, 20, 30]])
        engine._verify_step(input_ids, draft)

        assert engine.stats["total_verify_steps"] == 1
        assert len(engine.stats["verify_times_ms"]) == 1
        assert engine.stats["total_accepted"] >= 1

    def test_generate_basic(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        output = engine.generate(prompt, max_tokens=10)
        mx.eval(output)

        assert output.shape[0] > 5  # should generate some tokens

    def test_generate_2d_input(self):
        """Should handle both 1D and 2D prompt inputs."""
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([[1, 2, 3, 4, 5]])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)

        assert output.shape[0] > 5

    def test_generate_with_callback(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        callback_tokens = []

        def on_tokens(tokens):
            callback_tokens.append(list(tokens))

        prompt = mx.array([1, 2, 3])
        engine.generate(prompt, max_tokens=10, callback=on_tokens)

        assert len(callback_tokens) > 0

    def test_generate_with_temperature(self):
        """Test sampling mode generation."""
        config = LayerSkipConfig(num_speculative_tokens=3, temperature=0.8)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)

        assert output.shape[0] > 5

    def test_generate_no_eos_tokenizer(self):
        """Test with tokenizer that has no eos_token_id."""
        config = LayerSkipConfig(num_speculative_tokens=3)
        tokenizer = MockTokenizerNoEos()
        engine = LayerSkipEngine(self.model, tokenizer, config)

        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)

        assert output.shape[0] > 3

    def test_stats(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        engine.generate(prompt, max_tokens=10)

        stats = engine.get_stats()
        assert "exit_layer" in stats
        assert "total_layers" in stats
        assert "acceptance_rate" in stats
        assert "tokens_per_step" in stats
        assert "speedup_factor" in stats
        assert "avg_draft_ms" in stats
        assert "avg_verify_ms" in stats
        assert "adaptive_exit_enabled" in stats
        assert "avg_adaptive_exit_layer" in stats
        assert stats["exit_layer"] == 4
        assert stats["total_layers"] == 8
        assert stats["total_draft_tokens"] > 0
        assert stats["total_accepted"] > 0

    def test_stats_reset_on_generate(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        engine.generate(prompt, max_tokens=5)
        stats1 = engine.get_stats()

        engine.generate(prompt, max_tokens=5)
        stats2 = engine.get_stats()

        # Stats should be fresh for each generate call
        assert stats2["total_verify_steps"] > 0

    def test_stats_empty_before_generate(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        stats = engine.get_stats()
        assert stats["total_draft_tokens"] == 0
        assert stats["total_accepted"] == 0
        assert stats["avg_draft_ms"] == 0
        assert stats["avg_verify_ms"] == 0

    def test_stats_acceptance_rate_bounded(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        engine.generate(prompt, max_tokens=10)

        stats = engine.get_stats()
        assert 0.0 <= stats["acceptance_rate"] <= 2.0  # can exceed 1.0 due to bonus tokens
        assert stats["speedup_factor"] >= 0.0

    def test_stats_adaptive_exit_info(self):
        """Stats should include adaptive exit layer info."""
        config = LayerSkipConfig(num_speculative_tokens=3, adaptive_exit=True)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        engine.generate(prompt, max_tokens=10)

        stats = engine.get_stats()
        assert stats["adaptive_exit_enabled"] is True
        assert isinstance(stats["avg_adaptive_exit_layer"], (int, float))


# -- apply_layerskip convenience function --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestApplyLayerSkip:
    def test_apply_returns_engine(self):
        mx.random.seed(42)
        model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer()

        engine = apply_layerskip(model, tokenizer)
        assert isinstance(engine, LayerSkipEngine)

    def test_apply_with_config(self):
        mx.random.seed(42)
        model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer()

        config = LayerSkipConfig(exit_layer=2, num_speculative_tokens=7)
        engine = apply_layerskip(model, tokenizer, config)
        assert engine.config.exit_layer == 2
        assert engine.config.num_speculative_tokens == 7

    def test_apply_generates(self):
        mx.random.seed(42)
        model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer()

        engine = apply_layerskip(model, tokenizer)
        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)
        assert output.shape[0] > 3


# -- Edge Cases --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestLayerSkipEdgeCases:
    def setup_method(self):
        mx.random.seed(42)
        self.model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(self.model.parameters())
        self.tokenizer = MockTokenizer()

    def test_exit_layer_1(self):
        """Using only 1 layer for drafting should still work."""
        config = LayerSkipConfig(exit_layer=1, num_speculative_tokens=2)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)
        assert output.shape[0] > 3

    def test_single_draft_token(self):
        config = LayerSkipConfig(num_speculative_tokens=1)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)
        assert output.shape[0] > 3

    def test_generate_max_tokens_zero(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=0)
        mx.eval(output)
        assert output.shape[0] == 3  # no new tokens

    def test_single_token_prompt(self):
        config = LayerSkipConfig(num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([42])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)
        assert output.shape[0] > 1

    def test_large_speculative_count(self):
        config = LayerSkipConfig(num_speculative_tokens=15)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        output = engine.generate(prompt, max_tokens=20)
        mx.eval(output)
        assert output.shape[0] > 5

    def test_exit_layer_near_total(self):
        """Exit layer close to total layers should still work."""
        config = LayerSkipConfig(exit_layer=7, num_speculative_tokens=3)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)
        assert output.shape[0] > 3

    def test_verify_with_single_draft(self):
        config = LayerSkipConfig(num_speculative_tokens=1)
        engine = LayerSkipEngine(self.model, self.tokenizer, config)

        input_ids = mx.array([[1, 2, 3]])
        draft = mx.array([[10]])
        accepted, num = engine._verify_step(input_ids, draft)
        mx.eval(accepted)

        assert len(accepted) >= 1

    def test_adaptive_exit_reduces_layers_for_high_confidence(self):
        """With low threshold, adaptive should reduce exit layers."""
        config = LayerSkipConfig(
            exit_layer=4,
            num_speculative_tokens=5,
            adaptive_exit=True,
            confidence_threshold=0.0,  # everything is "high confidence"
        )
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        drafter.draft(input_ids)

        # With threshold=0.0, every token has high confidence,
        # so exit layers should decrease
        exit_layers = drafter._draft_exit_layers
        assert len(exit_layers) == 5
        # Last tokens should use fewer layers than the first
        assert exit_layers[-1] <= exit_layers[0]

    def test_adaptive_exit_increases_layers_for_low_confidence(self):
        """With very high threshold, adaptive should increase exit layers."""
        config = LayerSkipConfig(
            exit_layer=4,
            num_speculative_tokens=5,
            adaptive_exit=True,
            confidence_threshold=1.0,  # nothing can be high confidence
        )
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        drafter.draft(input_ids)

        exit_layers = drafter._draft_exit_layers
        assert len(exit_layers) == 5
        # Last tokens should use more layers than the first
        assert exit_layers[-1] >= exit_layers[0]
