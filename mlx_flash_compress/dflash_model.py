"""MLX port of z-lab's DFlashDraftModel architecture.

Faithfully ports the DFlash block diffusion drafter from PyTorch/HuggingFace
to Apple Silicon MLX. The drafter generates 16 draft tokens in a single forward
pass using bidirectional attention conditioned on the target model's hidden states.

Reference: z-lab/Qwen3.6-35B-A3B-DFlash (HuggingFace)
Paper: arXiv:2602.06036 (DFlash, ICLR 2026)

Architecture:
  - No embed_tokens or lm_head — uses target model's
  - fc: projects concatenated target hidden states from checkpoint layers
  - 8 DFlash decoder layers with cross-attention (Q from draft, K/V from target+draft)
  - Bidirectional attention (is_causal=False)
  - GQA with QK-norm and RoPE (YaRN scaling)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DFlashModelConfig:
    """Configuration matching z-lab's DFlash drafter config.json."""
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 8
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: list[int] = field(default_factory=lambda: [1, 10, 19, 28, 37])
    num_target_layers: int = 40
    rope_theta: float = 10000000.0
    max_position_embeddings: int = 262144

    @classmethod
    def from_dict(cls, data: dict) -> "DFlashModelConfig":
        dflash_cfg = data.get("dflash_config", {})
        return cls(
            hidden_size=data.get("hidden_size", 2048),
            intermediate_size=data.get("intermediate_size", 6144),
            num_hidden_layers=data.get("num_hidden_layers", 8),
            num_attention_heads=data.get("num_attention_heads", 32),
            num_key_value_heads=data.get("num_key_value_heads", 4),
            head_dim=data.get("head_dim", 128),
            vocab_size=data.get("vocab_size", 248320),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            hidden_act=data.get("hidden_act", "silu"),
            attention_bias=data.get("attention_bias", False),
            block_size=data.get("block_size", 16),
            mask_token_id=dflash_cfg.get("mask_token_id", 0),
            target_layer_ids=dflash_cfg.get("target_layer_ids", [1, 10, 19, 28, 37]),
            num_target_layers=data.get("num_target_layers", 40),
            rope_theta=data.get("rope_theta", 10000000.0),
            max_position_embeddings=data.get("max_position_embeddings", 262144),
        )

    @classmethod
    def from_json(cls, path: str) -> "DFlashModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _compute_rope_freqs(head_dim: int, theta: float = 10000000.0) -> mx.array:
    return 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))


def _apply_rotary_emb(
    q: mx.array, k: mx.array, freqs: mx.array, total_len: int, q_len: int
) -> tuple[mx.array, mx.array]:
    """Apply RoPE to Q and K with different sequence lengths.

    K covers all positions [0..total_len-1].
    Q covers the last q_len positions [total_len-q_len..total_len-1].
    """
    positions = mx.arange(total_len, dtype=mx.float32)
    angles = mx.outer(positions, freqs)
    cos_full = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
    sin_full = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)

    cos_full = cos_full[None, None, :, :]
    sin_full = sin_full[None, None, :, :]

    q_cos = cos_full[:, :, -q_len:, :]
    q_sin = sin_full[:, :, -q_len:, :]

    q_out = q * q_cos + _rotate_half(q) * q_sin
    k_out = k * cos_full + _rotate_half(k) * sin_full

    return q_out, k_out


class DFlashAttention(nn.Module):
    """Cross-attention: Q from draft embeddings, K/V from concat(target, draft).

    Bidirectional (no causal mask). GQA with QK-norm and RoPE.
    """

    def __init__(self, config: DFlashModelConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self._rope_freqs = _compute_rope_freqs(self.head_dim, config.rope_theta)

    def __call__(self, hidden_states: mx.array, target_hidden: mx.array) -> mx.array:
        B, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]
        total_len = ctx_len + q_len

        q = self.q_proj(hidden_states)
        q = q.reshape(B, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)

        kv_input = mx.concatenate([target_hidden, hidden_states], axis=1)
        k = self.k_proj(kv_input).reshape(B, total_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(kv_input).reshape(B, total_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_norm(k)

        q, k = _apply_rotary_emb(q, k, self._rope_freqs, total_len, q_len)

        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, q_len, -1)
        return self.o_proj(out)


class DFlashMLP(nn.Module):
    """SiLU-gated MLP matching Qwen3MLP."""

    def __init__(self, config: DFlashModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """Pre-norm residual block with cross-attention and SiLU MLP."""

    def __init__(self, config: DFlashModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: mx.array, target_hidden: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), target_hidden)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashDraftModel(nn.Module):
    """MLX port of z-lab's DFlashDraftModel.

    Forward: noise_embedding + target_hidden → refined hidden states.
    The caller projects the output to vocab logits using the target model's lm_head.
    """

    def __init__(self, config: DFlashModelConfig):
        super().__init__()
        self.config = config

        num_target_layers = len(config.target_layer_ids)
        self.fc = nn.Linear(num_target_layers * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = [DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, noise_embedding: mx.array, target_hidden: mx.array) -> mx.array:
        """Forward pass.

        Args:
            noise_embedding: Embedded draft tokens [B, block_size, hidden_size]
            target_hidden: Concatenated checkpoint hidden states
                           [B, ctx_len, num_checkpoint_layers * hidden_size]

        Returns:
            Refined hidden states [B, block_size, hidden_size]
        """
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden)

        return self.norm(hidden_states)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> tuple["DFlashDraftModel", "DFlashModelConfig"]:
        """Load pre-trained drafter from a directory with config.json + model.safetensors."""
        model_dir = Path(model_dir)
        config = DFlashModelConfig.from_json(str(model_dir / "config.json"))
        model = cls(config)

        weights_file = model_dir / "model.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"Weights not found at {weights_file}")

        weights = mx.load(str(weights_file))
        model.load_weights(list(weights.items()))
        return model, config


class DFlashRunner:
    """Wires a DFlash drafter to a target model for speculative decoding.

    Handles hidden state extraction, draft generation, and verification.
    """

    def __init__(self, target_model, tokenizer, drafter: DFlashDraftModel,
                 config: DFlashModelConfig):
        self.target = target_model
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.config = config

        self._embed_fn = None
        self._lm_head_fn = None
        self._layers = None
        self._norm_fn = None
        self._detect_target_components()

        self.stats = {
            "total_drafts": 0,
            "total_accepted": 0,
            "total_target_calls": 0,
            "draft_times_ms": [],
            "verify_times_ms": [],
        }

    def _detect_target_components(self):
        m = self.target
        if hasattr(m, "model"):
            inner = m.model
            self._embed_fn = getattr(inner, "embed_tokens", None)
            self._layers = getattr(inner, "layers", None)
            self._norm_fn = getattr(inner, "norm", None)
        else:
            self._embed_fn = getattr(m, "embed_tokens", None)
            self._layers = getattr(m, "layers", None)
            self._norm_fn = getattr(m, "norm", None)

        if hasattr(m, "lm_head"):
            self._lm_head_fn = m.lm_head
        elif hasattr(m, "model") and hasattr(m.model, "output"):
            self._lm_head_fn = m.model.output

        if self._embed_fn is None or self._layers is None:
            raise RuntimeError("Cannot detect target model's embed_tokens/layers")

    def extract_hidden_states(self, input_ids: mx.array) -> mx.array:
        """Run target model and extract hidden states at checkpoint layers.

        Returns concatenated hidden states [B, seq_len, num_layers * hidden_size].
        """
        x = self._embed_fn(input_ids)

        checkpoint_hiddens = []
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if i in self.config.target_layer_ids:
                checkpoint_hiddens.append(x)

        return mx.concatenate(checkpoint_hiddens, axis=-1)

    def draft_tokens(self, input_ids: mx.array, num_positions: int = 1) -> tuple[mx.array, mx.array]:
        """Generate draft tokens via block diffusion.

        Args:
            input_ids: Context token IDs [B, seq_len]
            num_positions: How many target positions to condition on (from the end)

        Returns:
            (draft_token_ids [B, block_size], draft_logits [B, block_size, vocab])
        """
        import time
        t0 = time.perf_counter()

        target_hidden = self.extract_hidden_states(input_ids)
        target_hidden = target_hidden[:, -num_positions:, :]

        mask_ids = mx.full(
            (input_ids.shape[0], self.config.block_size),
            self.config.mask_token_id,
            dtype=mx.int32,
        )
        noise_embedding = self._embed_fn(mask_ids)

        refined = self.drafter(noise_embedding, target_hidden)

        if self._lm_head_fn is not None:
            logits = self._lm_head_fn(refined)
        else:
            logits = refined

        draft_ids = mx.argmax(logits, axis=-1)
        mx.eval(draft_ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["draft_times_ms"].append(elapsed_ms)
        self.stats["total_drafts"] += self.config.block_size

        return draft_ids, logits

    def verify_drafts(self, input_ids: mx.array, draft_ids: mx.array) -> tuple[mx.array, int]:
        """Verify draft tokens against target model (single forward pass).

        Returns (accepted_token_ids, num_accepted).
        """
        import time
        t0 = time.perf_counter()

        full_ids = mx.concatenate([input_ids, draft_ids], axis=-1)

        x = self._embed_fn(full_ids)
        for layer in self._layers:
            x = layer(x)
        if self._norm_fn is not None:
            x = self._norm_fn(x)
        if self._lm_head_fn is not None:
            logits = self._lm_head_fn(x)
        else:
            logits = x

        seq_len = input_ids.shape[-1]
        block_size = draft_ids.shape[-1]

        verify_logits = logits[:, seq_len - 1 : seq_len + block_size - 1, :]
        target_ids = mx.argmax(verify_logits, axis=-1)
        mx.eval(target_ids)

        draft_np = draft_ids[0].tolist()
        target_np = target_ids[0].tolist()

        num_accepted = 0
        for d, t in zip(draft_np, target_np):
            if d == t:
                num_accepted += 1
            else:
                break

        bonus_idx = num_accepted
        if bonus_idx < len(target_np):
            accepted = draft_np[:num_accepted] + [target_np[bonus_idx]]
        else:
            accepted = draft_np[:num_accepted]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["verify_times_ms"].append(elapsed_ms)
        self.stats["total_target_calls"] += 1
        self.stats["total_accepted"] += len(accepted)

        return mx.array(accepted, dtype=mx.int32), len(accepted)

    def generate(self, prompt: str, max_tokens: int = 128) -> tuple[str, dict]:
        """Generate text with DFlash speculative decoding.

        Returns (generated_text, stats_summary).
        """
        import time
        self.stats = {
            "total_drafts": 0, "total_accepted": 0, "total_target_calls": 0,
            "draft_times_ms": [], "verify_times_ms": [],
        }

        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        t_start = time.perf_counter()

        while len(generated) - len(tokens) < max_tokens:
            input_ids = mx.array([generated])

            draft_ids, _ = self.draft_tokens(input_ids, num_positions=1)
            accepted, n_accepted = self.verify_drafts(input_ids, draft_ids)

            if n_accepted == 0:
                break

            generated.extend(accepted.tolist())

            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None and eos in accepted.tolist():
                break

        elapsed = time.perf_counter() - t_start
        n_gen = len(generated) - len(tokens)
        tok_per_sec = n_gen / elapsed if elapsed > 0 else 0

        import numpy as np
        summary = {
            "tokens_generated": n_gen,
            "wall_time_s": round(elapsed, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "total_target_calls": self.stats["total_target_calls"],
            "total_drafts": self.stats["total_drafts"],
            "total_accepted": self.stats["total_accepted"],
            "acceptance_rate": round(self.stats["total_accepted"] / max(1, self.stats["total_drafts"]), 3),
            "tokens_per_step": round(self.stats["total_accepted"] / max(1, self.stats["total_target_calls"]), 1),
            "avg_draft_ms": round(float(np.mean(self.stats["draft_times_ms"])), 1) if self.stats["draft_times_ms"] else 0,
            "avg_verify_ms": round(float(np.mean(self.stats["verify_times_ms"])), 1) if self.stats["verify_times_ms"] else 0,
        }

        text = self.tokenizer.decode(generated[len(tokens):])
        return text, summary
