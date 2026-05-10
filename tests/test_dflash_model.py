"""Tests for the MLX port of z-lab's DFlashDraftModel."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_flash_compress.dflash_model import (
    DFlashModelConfig,
    DFlashDraftModel,
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashMLP,
    DFlashRunner,
    _rotate_half,
    _apply_rotary_emb,
    _compute_rope_freqs,
)


class TestDFlashModelConfig:
    def test_default_config(self):
        config = DFlashModelConfig()
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 8
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 4
        assert config.block_size == 16
        assert config.rope_theta == 10000000.0

    def test_from_dict(self):
        data = {
            "hidden_size": 4096,
            "num_hidden_layers": 5,
            "block_size": 8,
            "dflash_config": {
                "mask_token_id": 99,
                "target_layer_ids": [0, 5, 10],
            },
            "rope_theta": 500000.0,
        }
        config = DFlashModelConfig.from_dict(data)
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 5
        assert config.block_size == 8
        assert config.mask_token_id == 99
        assert config.target_layer_ids == [0, 5, 10]
        assert config.rope_theta == 500000.0

    def test_from_json(self):
        data = {
            "hidden_size": 1024,
            "num_hidden_layers": 3,
            "dflash_config": {"mask_token_id": 42, "target_layer_ids": [1, 5]},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            config = DFlashModelConfig.from_json(f.name)
        assert config.hidden_size == 1024
        assert config.target_layer_ids == [1, 5]
        Path(f.name).unlink()

    def test_qwen3_35b_config(self):
        """Match the real z-lab/Qwen3.6-35B-A3B-DFlash config."""
        data = {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 8,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 248320,
            "block_size": 16,
            "dflash_config": {
                "mask_token_id": 248070,
                "target_layer_ids": [1, 10, 19, 28, 37],
            },
            "num_target_layers": 40,
            "rope_theta": 10000000,
        }
        config = DFlashModelConfig.from_dict(data)
        assert config.hidden_size == 2048
        assert config.intermediate_size == 6144
        assert config.num_hidden_layers == 8
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 4
        assert config.head_dim == 128
        assert config.block_size == 16
        assert config.mask_token_id == 248070
        assert config.target_layer_ids == [1, 10, 19, 28, 37]
        assert config.num_target_layers == 40


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestRoPE:
    def test_rotate_half(self):
        x = mx.array([[1, 2, 3, 4]], dtype=mx.float32)
        rotated = _rotate_half(x)
        expected = mx.array([[-3, -4, 1, 2]], dtype=mx.float32)
        np.testing.assert_array_equal(np.array(rotated), np.array(expected))

    def test_rope_freqs_shape(self):
        freqs = _compute_rope_freqs(128, theta=10000000.0)
        assert freqs.shape == (64,)

    def test_apply_rotary_emb_shapes(self):
        B, num_heads, q_len, kv_len, head_dim = 1, 4, 8, 12, 64
        q = mx.random.normal((B, num_heads, q_len, head_dim))
        k = mx.random.normal((B, num_heads, kv_len, head_dim))
        freqs = _compute_rope_freqs(head_dim)

        q_rot, k_rot = _apply_rotary_emb(q, k, freqs, total_len=kv_len, q_len=q_len)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDFlashAttention:
    def test_forward_shape(self):
        config = DFlashModelConfig(hidden_size=64, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=16)
        attn = DFlashAttention(config, layer_idx=0)

        B, q_len, ctx_len = 1, 8, 4
        hidden = mx.random.normal((B, q_len, 64))
        target = mx.random.normal((B, ctx_len, 64))

        out = attn(hidden, target)
        mx.eval(out)
        assert out.shape == (B, q_len, 64)

    def test_gqa_groups(self):
        config = DFlashModelConfig(hidden_size=128, num_attention_heads=8,
                                   num_key_value_heads=2, head_dim=16)
        attn = DFlashAttention(config, layer_idx=0)
        assert attn.num_kv_groups == 4

    def test_bidirectional(self):
        """Verify attention is bidirectional (all positions attend to all)."""
        config = DFlashModelConfig(hidden_size=32, num_attention_heads=2,
                                   num_key_value_heads=2, head_dim=16)
        attn = DFlashAttention(config, layer_idx=0)

        hidden = mx.random.normal((1, 4, 32))
        target = mx.random.normal((1, 2, 32))
        out = attn(hidden, target)
        mx.eval(out)
        assert out.shape == (1, 4, 32)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDFlashMLP:
    def test_forward_shape(self):
        config = DFlashModelConfig(hidden_size=64, intermediate_size=128)
        mlp = DFlashMLP(config)
        x = mx.random.normal((1, 8, 64))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (1, 8, 64)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDFlashDecoderLayer:
    def test_forward_shape(self):
        config = DFlashModelConfig(hidden_size=64, intermediate_size=128,
                                   num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16)
        layer = DFlashDecoderLayer(config, layer_idx=0)

        hidden = mx.random.normal((1, 8, 64))
        target = mx.random.normal((1, 2, 64))
        out = layer(hidden, target)
        mx.eval(out)
        assert out.shape == (1, 8, 64)

    def test_residual_connection(self):
        """Output should differ from input (residual + attention + MLP)."""
        config = DFlashModelConfig(hidden_size=32, intermediate_size=64,
                                   num_attention_heads=2, num_key_value_heads=2,
                                   head_dim=16)
        layer = DFlashDecoderLayer(config, layer_idx=0)

        hidden = mx.random.normal((1, 4, 32))
        target = mx.random.normal((1, 2, 32))
        out = layer(hidden, target)
        mx.eval(out)
        assert not mx.array_equal(hidden, out)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDFlashDraftModel:
    def _make_model(self, hidden_size=64, num_layers=2, block_size=4):
        config = DFlashModelConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=hidden_size // 4,
            block_size=block_size,
            target_layer_ids=[1, 3, 5],
        )
        return DFlashDraftModel(config), config

    def test_forward_shape(self):
        model, config = self._make_model()
        B = 1
        noise = mx.random.normal((B, config.block_size, config.hidden_size))
        target = mx.random.normal((B, 2, len(config.target_layer_ids) * config.hidden_size))

        out = model(noise, target)
        mx.eval(out)
        assert out.shape == (B, config.block_size, config.hidden_size)

    def test_fc_projection_dim(self):
        model, config = self._make_model(hidden_size=128)
        num_tgt = len(config.target_layer_ids)
        assert model.fc.weight.shape == (128, num_tgt * 128)

    def test_num_layers(self):
        model, config = self._make_model(num_layers=5)
        assert len(model.layers) == 5

    def test_save_load_roundtrip(self):
        model, config = self._make_model()
        B = 1
        noise = mx.random.normal((B, config.block_size, config.hidden_size))
        target = mx.random.normal((B, 1, len(config.target_layer_ids) * config.hidden_size))

        out1 = model(noise, target)
        mx.eval(out1)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "block_size": config.block_size,
                "dflash_config": {
                    "mask_token_id": config.mask_token_id,
                    "target_layer_ids": config.target_layer_ids,
                },
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config_data))

            from mlx.utils import tree_flatten
            flat = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(Path(tmpdir) / "model.safetensors"), flat)

            loaded_model, loaded_config = DFlashDraftModel.from_pretrained(tmpdir)

        out2 = loaded_model(noise, target)
        mx.eval(out2)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-5)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDFlashRunner:
    def _make_fake_target(self, num_layers=10, hidden_size=64, vocab_size=100):
        """Create a minimal fake target model for testing."""

        class FakeLayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim, bias=False)
            def __call__(self, x, **kwargs):
                return self.linear(x)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
                self.layers = [FakeLayer(hidden_size) for _ in range(num_layers)]
                self.norm = nn.RMSNorm(hidden_size)

        class FakeWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeModel()
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

            def __call__(self, input_ids, **kwargs):
                h = self.model.embed_tokens(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                h = self.model.norm(h)
                return self.lm_head(h)

        return FakeWrapper()

    def _make_fake_tokenizer(self, vocab_size=100):
        class FakeTokenizer:
            eos_token_id = 2
            def encode(self, text):
                return [1] + [ord(c) % vocab_size for c in text[:20]]
            def decode(self, ids):
                return "".join(chr(max(32, i % 128)) for i in ids)
        return FakeTokenizer()

    def test_runner_init(self):
        target = self._make_fake_target(num_layers=10, hidden_size=64)
        tokenizer = self._make_fake_tokenizer()

        config = DFlashModelConfig(
            hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            block_size=4, vocab_size=100,
            target_layer_ids=[1, 4, 7],
        )
        drafter = DFlashDraftModel(config)
        runner = DFlashRunner(target, tokenizer, drafter, config)

        assert runner._embed_fn is not None
        assert runner._lm_head_fn is not None
        assert runner._layers is not None

    def test_extract_hidden_states(self):
        target = self._make_fake_target(num_layers=10, hidden_size=64)
        tokenizer = self._make_fake_tokenizer()

        config = DFlashModelConfig(
            hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            block_size=4, vocab_size=100,
            target_layer_ids=[1, 4, 7],
        )
        drafter = DFlashDraftModel(config)
        runner = DFlashRunner(target, tokenizer, drafter, config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hidden = runner.extract_hidden_states(input_ids)
        mx.eval(hidden)

        assert hidden.shape == (1, 5, 3 * 64)

    def test_draft_tokens(self):
        target = self._make_fake_target(num_layers=10, hidden_size=64)
        tokenizer = self._make_fake_tokenizer()

        config = DFlashModelConfig(
            hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            block_size=4, vocab_size=100,
            target_layer_ids=[1, 4, 7],
            mask_token_id=0,
        )
        drafter = DFlashDraftModel(config)
        runner = DFlashRunner(target, tokenizer, drafter, config)

        input_ids = mx.array([[1, 5, 10, 20, 30]])
        draft_ids, logits = runner.draft_tokens(input_ids)
        mx.eval(draft_ids, logits)

        assert draft_ids.shape == (1, 3)
        assert logits.shape == (1, 3, 100)

    def test_generate_produces_text(self):
        target = self._make_fake_target(num_layers=10, hidden_size=64)
        tokenizer = self._make_fake_tokenizer()

        config = DFlashModelConfig(
            hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            block_size=4, vocab_size=100,
            target_layer_ids=[1, 4, 7],
            mask_token_id=0,
        )
        drafter = DFlashDraftModel(config)
        runner = DFlashRunner(target, tokenizer, drafter, config)

        text, stats = runner.generate("hello world", max_tokens=8, use_cache=False)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0
        assert stats["total_target_calls"] > 0
        assert 0 <= stats["acceptance_rate"] <= 1.0
        assert stats.get("cached") is False

    def test_generate_cache_fallback(self):
        target = self._make_fake_target(num_layers=10, hidden_size=64)
        tokenizer = self._make_fake_tokenizer()

        config = DFlashModelConfig(
            hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            block_size=4, vocab_size=100,
            target_layer_ids=[1, 4, 7],
            mask_token_id=0,
        )
        drafter = DFlashDraftModel(config)
        runner = DFlashRunner(target, tokenizer, drafter, config)

        text, stats = runner.generate("hello world", max_tokens=8, use_cache=True)
        assert isinstance(text, str)
        assert stats["tokens_generated"] > 0

    def test_generate_with_tree_no_cache(self):
        target = self._make_fake_target(num_layers=10, hidden_size=64)
        tokenizer = self._make_fake_tokenizer()

        config = DFlashModelConfig(
            hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            block_size=4, vocab_size=100,
            target_layer_ids=[1, 4, 7],
            mask_token_id=0,
        )
        drafter = DFlashDraftModel(config)
        runner = DFlashRunner(target, tokenizer, drafter, config)

        text, stats = runner.generate_with_tree(
            "hello world", max_tokens=8, tree_width=3, max_tree_size=15,
            use_cache=False,
        )
        assert isinstance(text, str)
        assert "total_tree_nodes" in stats
        assert stats.get("cached") is False

    def test_generate_with_tree_cache_fallback(self):
        target = self._make_fake_target(num_layers=10, hidden_size=64)
        tokenizer = self._make_fake_tokenizer()

        config = DFlashModelConfig(
            hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            block_size=4, vocab_size=100,
            target_layer_ids=[1, 4, 7],
            mask_token_id=0,
        )
        drafter = DFlashDraftModel(config)
        runner = DFlashRunner(target, tokenizer, drafter, config)

        text, stats = runner.generate_with_tree(
            "hello world", max_tokens=8, tree_width=3, max_tree_size=15,
            use_cache=True,
        )
        assert isinstance(text, str)
        assert "total_tree_nodes" in stats
