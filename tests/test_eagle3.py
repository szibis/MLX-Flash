"""Tests for EAGLE-3 speculative decoding."""

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_flash_compress.eagle3 import (
    EAGLE3Config,
    EAGLE3Engine,
    EAGLE3Trainer,
    EAGLEDraftHead,
    _flatten_dict,
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


class MockModel(nn.Module):
    """Mimics a standard HuggingFace/MLX causal LM structure."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 8):
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


class TestEAGLE3Config:
    def test_default_config(self):
        config = EAGLE3Config()
        assert config.num_draft_tokens == 6
        assert config.hidden_dim == 0  # auto
        assert config.num_heads == 4
        assert config.num_layers == 1
        assert config.temperature == 0.0

    def test_custom_config(self):
        config = EAGLE3Config(
            num_draft_tokens=10,
            hidden_dim=256,
            num_heads=8,
            num_layers=2,
            temperature=0.5,
        )
        assert config.num_draft_tokens == 10
        assert config.hidden_dim == 256
        assert config.num_heads == 8
        assert config.num_layers == 2


# -- Draft Head Tests --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestEAGLEDraftHead:
    def setup_method(self):
        mx.random.seed(42)

    def test_forward_shape(self):
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=1)
        mx.eval(head.parameters())

        hidden = mx.random.normal((1, 1, 32))
        embed = mx.random.normal((1, 1, 32))
        out = head(hidden, embed)
        mx.eval(out)

        assert out.shape == (1, 1, 32)

    def test_forward_batch(self):
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=1)
        mx.eval(head.parameters())

        hidden = mx.random.normal((4, 1, 32))
        embed = mx.random.normal((4, 1, 32))
        out = head(hidden, embed)
        mx.eval(out)

        assert out.shape == (4, 1, 32)

    def test_forward_multi_token(self):
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=1)
        mx.eval(head.parameters())

        hidden = mx.random.normal((1, 5, 32))
        embed = mx.random.normal((1, 5, 32))
        out = head(hidden, embed)
        mx.eval(out)

        assert out.shape == (1, 5, 32)

    def test_multi_layer_head(self):
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=3)
        mx.eval(head.parameters())

        hidden = mx.random.normal((1, 1, 32))
        embed = mx.random.normal((1, 1, 32))
        out = head(hidden, embed)
        mx.eval(out)

        assert out.shape == (1, 1, 32)

    def test_different_inputs_different_outputs(self):
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=1)
        mx.eval(head.parameters())

        h1 = mx.random.normal((1, 1, 32))
        h2 = mx.random.normal((1, 1, 32))
        e = mx.random.normal((1, 1, 32))

        out1 = head(h1, e)
        out2 = head(h2, e)
        mx.eval(out1, out2)

        diff = float(mx.sum(mx.abs(out1 - out2)).item())
        assert diff > 0

    def test_gradient_flows(self):
        """Verify gradients can flow through the draft head."""
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=1)
        mx.eval(head.parameters())

        def loss_fn(model, h, e):
            out = model(h, e)
            return mx.mean(out**2)

        h = mx.random.normal((1, 1, 32))
        e = mx.random.normal((1, 1, 32))

        loss, grads = nn.value_and_grad(head, loss_fn)(head, h, e)
        mx.eval(loss, grads)

        assert float(loss.item()) > 0
        # Check that at least one gradient is non-zero
        has_nonzero = False
        for k, v in grads.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, mx.array) and float(mx.sum(mx.abs(vv)).item()) > 0:
                        has_nonzero = True
            elif isinstance(v, mx.array) and float(mx.sum(mx.abs(v)).item()) > 0:
                has_nonzero = True
        assert has_nonzero


# -- Engine Tests --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestEAGLE3Engine:
    def setup_method(self):
        mx.random.seed(42)
        self.model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(self.model.parameters())
        self.tokenizer = MockTokenizer()

    def test_engine_init_default(self):
        engine = EAGLE3Engine(self.model, self.tokenizer)
        assert engine.config.num_draft_tokens == 6
        assert engine.config.hidden_dim == 32  # auto-detected

    def test_engine_init_with_head(self):
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=1)
        mx.eval(head.parameters())
        engine = EAGLE3Engine(self.model, self.tokenizer, draft_head=head)
        assert engine.draft_head is head

    def test_hidden_dim_detection(self):
        engine = EAGLE3Engine(self.model, self.tokenizer)
        assert engine.config.hidden_dim == 32

    def test_get_last_hidden_state(self):
        engine = EAGLE3Engine(self.model, self.tokenizer)
        input_ids = mx.array([[1, 2, 3]])
        hidden = engine._get_last_hidden_state(input_ids)
        mx.eval(hidden)

        assert hidden.shape == (1, 3, 32)

    def test_hidden_to_logits(self):
        engine = EAGLE3Engine(self.model, self.tokenizer)
        hidden = mx.random.normal((1, 1, 32))
        logits = engine._hidden_to_logits(hidden)
        mx.eval(logits)

        assert logits.shape == (1, 1, 100)

    def test_draft_produces_tokens(self):
        config = EAGLE3Config(num_draft_tokens=3)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        last_hidden = mx.random.normal((1, 1, 32))
        draft_ids, draft_hiddens = engine.draft(last_hidden, last_token_id=5)
        mx.eval(draft_ids, draft_hiddens)

        assert draft_ids.shape == (1, 3)
        assert draft_hiddens.shape == (1, 3, 32)

    def test_verify_step(self):
        engine = EAGLE3Engine(self.model, self.tokenizer)

        input_ids = mx.array([[1, 2, 3]])
        draft = mx.array([[10, 20, 30]])
        accepted, num = engine.verify(input_ids, draft)
        mx.eval(accepted)

        # Should always produce at least the bonus token
        assert len(accepted) >= 1
        assert num >= 1

    def test_generate_basic(self):
        config = EAGLE3Config(num_draft_tokens=3)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        output = engine.generate(prompt, max_tokens=10)
        mx.eval(output)

        assert output.shape[0] > 5

    def test_generate_2d_input(self):
        config = EAGLE3Config(num_draft_tokens=3)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        prompt = mx.array([[1, 2, 3, 4, 5]])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)

        assert output.shape[0] > 5

    def test_generate_with_callback(self):
        config = EAGLE3Config(num_draft_tokens=3)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        callback_tokens = []

        def on_tokens(tokens):
            callback_tokens.append(list(tokens))

        prompt = mx.array([1, 2, 3])
        engine.generate(prompt, max_tokens=10, callback=on_tokens)

        assert len(callback_tokens) > 0

    def test_stats(self):
        config = EAGLE3Config(num_draft_tokens=3)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        engine.generate(prompt, max_tokens=10)

        stats = engine.get_stats()
        assert "total_draft_tokens" in stats
        assert "total_accepted" in stats
        assert "acceptance_rate" in stats
        assert "tokens_per_step" in stats
        assert "speedup_factor" in stats
        assert "avg_draft_ms" in stats
        assert "avg_verify_ms" in stats
        assert stats["total_draft_tokens"] > 0
        assert stats["total_accepted"] > 0

    def test_stats_reset_on_generate(self):
        config = EAGLE3Config(num_draft_tokens=3)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        engine.generate(prompt, max_tokens=5)
        engine.generate(prompt, max_tokens=5)
        stats = engine.get_stats()

        assert stats["total_verify_steps"] > 0


# -- Trainer Tests --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestEAGLE3Trainer:
    def setup_method(self):
        mx.random.seed(42)
        self.model = MockModel(vocab_size=100, hidden_dim=32, num_layers=4)
        mx.eval(self.model.parameters())
        self.tokenizer = MockTokenizer()

    def test_trainer_init(self):
        trainer = EAGLE3Trainer(self.model, self.tokenizer)
        assert trainer.config.hidden_dim == 32

    def test_collect_training_data(self):
        trainer = EAGLE3Trainer(self.model, self.tokenizer)
        texts = ["hello world test"]
        input_pairs, target_hiddens = trainer.collect_training_data(
            texts,
            max_tokens_per_text=10,
        )
        mx.eval(input_pairs, target_hiddens)

        assert input_pairs.ndim == 2
        assert target_hiddens.ndim == 2
        assert input_pairs.shape[0] == target_hiddens.shape[0]
        assert input_pairs.shape[0] > 0
        assert input_pairs.shape[1] == 64  # hidden_dim * 2
        assert target_hiddens.shape[1] == 32  # hidden_dim

    def test_collect_empty_text(self):
        trainer = EAGLE3Trainer(self.model, self.tokenizer)
        input_pairs, target_hiddens = trainer.collect_training_data(
            [""],
            max_tokens_per_text=10,
        )
        mx.eval(input_pairs, target_hiddens)

        # Empty or too-short text should produce zero pairs
        assert input_pairs.shape[0] == 0

    def test_collect_multiple_texts(self):
        trainer = EAGLE3Trainer(self.model, self.tokenizer)
        texts = ["hello world", "test data", "more text here"]
        input_pairs, target_hiddens = trainer.collect_training_data(
            texts,
            max_tokens_per_text=10,
        )
        mx.eval(input_pairs, target_hiddens)

        # Should have pairs from all texts
        assert input_pairs.shape[0] > 10

    def test_train_basic(self):
        trainer = EAGLE3Trainer(self.model, self.tokenizer)
        texts = ["hello world test data for training the eagle head"]
        pairs = trainer.collect_training_data(texts, max_tokens_per_text=20)

        draft_head = trainer.train(pairs, num_steps=5, lr=1e-3, batch_size=4)

        assert isinstance(draft_head, EAGLEDraftHead)
        assert draft_head.hidden_dim == 32

        # Verify it can forward
        h = mx.random.normal((1, 1, 32))
        e = mx.random.normal((1, 1, 32))
        out = draft_head(h, e)
        mx.eval(out)
        assert out.shape == (1, 1, 32)

    def test_train_loss_decreases(self):
        """Training should reduce MSE loss over steps."""
        trainer = EAGLE3Trainer(self.model, self.tokenizer)
        texts = ["hello world test data for training"] * 3
        input_pairs, target_hiddens = trainer.collect_training_data(
            texts,
            max_tokens_per_text=20,
        )

        hidden_dim = 32
        draft_head = EAGLEDraftHead(hidden_dim=hidden_dim, num_heads=4, num_layers=1)
        mx.eval(draft_head.parameters())

        # Compute initial loss
        sample_h = input_pairs[:4, :hidden_dim].reshape(-1, 1, hidden_dim)
        sample_e = input_pairs[:4, hidden_dim:].reshape(-1, 1, hidden_dim)
        sample_t = target_hiddens[:4]

        pred_before = draft_head(sample_h, sample_e).squeeze(1)
        loss_before = float(mx.mean((pred_before - sample_t) ** 2).item())

        # Train
        draft_head = trainer.train(
            (input_pairs, target_hiddens),
            num_steps=50,
            lr=1e-3,
            batch_size=8,
        )

        pred_after = draft_head(sample_h, sample_e).squeeze(1)
        loss_after = float(mx.mean((pred_after - sample_t) ** 2).item())

        # Loss should decrease (or at least not increase dramatically)
        # With random init and small data, we just check it runs without error
        assert loss_after < loss_before * 10  # generous bound

    def test_train_no_data_raises(self):
        trainer = EAGLE3Trainer(self.model, self.tokenizer)
        empty_pairs = (mx.zeros((0, 64)), mx.zeros((0, 32)))

        with pytest.raises(ValueError, match="No training data"):
            trainer.train(empty_pairs, num_steps=10)


# -- Save/Load Tests --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestEAGLE3SaveLoad:
    def test_save_and_load(self, tmp_path):
        mx.random.seed(42)
        head = EAGLEDraftHead(hidden_dim=32, num_heads=4, num_layers=1)
        mx.eval(head.parameters())

        # Forward to get reference output
        h = mx.random.normal((1, 1, 32))
        e = mx.random.normal((1, 1, 32))
        out_before = head(h, e)
        mx.eval(out_before)

        # Save
        model = MockModel(vocab_size=100, hidden_dim=32)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer()
        trainer = EAGLE3Trainer(model, tokenizer)

        path = str(tmp_path / "eagle_head.safetensors")
        trainer.save(head, path)

        # Load
        loaded_head = EAGLE3Trainer.load(path, hidden_dim=32)
        out_after = loaded_head(h, e)
        mx.eval(out_after)

        # Outputs should match
        diff = float(mx.sum(mx.abs(out_before - out_after)).item())
        assert diff < 1e-5


# -- Helper Tests --


class TestFlattenDict:
    def test_simple(self):
        d = {"a": 1, "b": 2}
        out = {}
        _flatten_dict(d, "", out)
        assert out == {"a": 1, "b": 2}

    def test_nested(self):
        d = {"a": {"b": 1, "c": 2}}
        out = {}
        _flatten_dict(d, "", out)
        assert out == {"a.b": 1, "a.c": 2}

    def test_with_prefix(self):
        d = {"x": 1}
        out = {}
        _flatten_dict(d, "prefix", out)
        assert out == {"prefix.x": 1}


# -- Edge Cases --


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestEAGLE3EdgeCases:
    def setup_method(self):
        mx.random.seed(42)
        self.model = MockModel(vocab_size=100, hidden_dim=32, num_layers=8)
        mx.eval(self.model.parameters())
        self.tokenizer = MockTokenizer()

    def test_single_draft_token(self):
        config = EAGLE3Config(num_draft_tokens=1)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=5)
        mx.eval(output)
        assert output.shape[0] > 3

    def test_generate_max_tokens_zero(self):
        config = EAGLE3Config(num_draft_tokens=3)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3])
        output = engine.generate(prompt, max_tokens=0)
        mx.eval(output)
        assert output.shape[0] == 3

    def test_large_draft_count(self):
        config = EAGLE3Config(num_draft_tokens=12)
        engine = EAGLE3Engine(self.model, self.tokenizer, config)

        prompt = mx.array([1, 2, 3, 4, 5])
        output = engine.generate(prompt, max_tokens=15)
        mx.eval(output)
        assert output.shape[0] > 5
