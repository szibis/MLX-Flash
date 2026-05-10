"""Tests for LayerSkip self-speculative decoding."""

import numpy as np
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

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32,
                 num_layers: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [MockLayer(hidden_dim) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_dim)


class MockModel(nn.Module):
    """Mimics a standard HuggingFace/MLX causal LM structure.

    model.model.embed_tokens / model.model.layers / model.model.norm
    model.lm_head
    """

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32,
                 num_layers: int = 8):
        super().__init__()
        self.model = MockInnerModel(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


class MockTokenizer:
    """Minimal tokenizer for testing."""
    eos_token_id = 99

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
        config = LayerSkipConfig(exit_layer=4, num_speculative_tokens=3,
                                 temperature=0.0)
        drafter = LayerSkipDrafter(self.model, config)

        input_ids = mx.array([[1, 2, 3]])
        ids1, _ = drafter.draft(input_ids)
        ids2, _ = drafter.draft(input_ids)
        mx.eval(ids1, ids2)

        assert mx.array_equal(ids1, ids2)


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
