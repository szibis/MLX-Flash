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

    def __call__(self, noise_embedding: mx.array, target_hidden: mx.array,
                 num_active_layers: int | None = None) -> mx.array:
        """Forward pass.

        Args:
            noise_embedding: Embedded draft tokens [B, block_size, hidden_size]
            target_hidden: Concatenated checkpoint hidden states
                           [B, ctx_len, num_checkpoint_layers * hidden_size]
            num_active_layers: Use only first N layers (None = all).

        Returns:
            Refined hidden states [B, block_size, hidden_size]
        """
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        active = self.layers[:num_active_layers] if num_active_layers else self.layers
        hidden_states = noise_embedding
        for layer in active:
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
                 config: DFlashModelConfig, *,
                 inference_block_size: int | None = None,
                 hidden_dtype: mx.Dtype | None = None,
                 compile_drafter: bool = False,
                 num_denoise_steps: int = 1,
                 draft_temperature: float | None = None,
                 num_active_layers: int | None = None,
                 quantize_drafter: int | None = None):
        self.target = target_model
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.config = config

        self.block_size = inference_block_size or config.block_size
        self.hidden_dtype = hidden_dtype
        self._num_denoise_steps = num_denoise_steps
        self._draft_temperature = draft_temperature
        self._num_active_layers = num_active_layers

        if quantize_drafter is not None:
            nn.quantize(self.drafter, group_size=64, bits=quantize_drafter)
            mx.eval(self.drafter.parameters())

        self._embed_fn = None
        self._lm_head_fn = None
        self._layers = None
        self._norm_fn = None
        self._inner_model = None
        self._detect_target_components()

        if compile_drafter:
            self._compiled_drafter_fn = mx.compile(self.drafter)
        else:
            self._compiled_drafter_fn = None

        self.stats = {
            "total_drafts": 0,
            "total_accepted": 0,
            "total_target_calls": 0,
            "draft_times_ms": [],
            "verify_times_ms": [],
        }

    def _detect_target_components(self):
        m = self.target

        # Try common model structures:
        # 1. model.language_model.model.{embed_tokens,layers,norm} + language_model.lm_head (multimodal)
        # 2. model.model.{embed_tokens,layers,norm} + model.lm_head (standard causal LM)
        # 3. model.{embed_tokens,layers,norm} + model.lm_head (flat)
        candidates = []
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "model"):
                candidates.append((lm.model, lm))
        if hasattr(m, "model"):
            candidates.append((m.model, m))
        candidates.append((m, m))

        for inner, outer in candidates:
            embed = getattr(inner, "embed_tokens", None)
            layers = getattr(inner, "layers", None)
            norm = getattr(inner, "norm", None)
            head = getattr(outer, "lm_head", None) or getattr(inner, "output", None)

            if embed is not None and layers is not None:
                self._embed_fn = embed
                self._layers = layers
                self._norm_fn = norm
                self._lm_head_fn = head
                self._inner_model = inner
                return

        raise RuntimeError("Cannot detect target model's embed_tokens/layers")

    def _forward_target(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Single target forward pass returning both logits and checkpoint hidden states.

        This combines extraction + verification into one pass, saving 40 layers
        of compute per speculative decoding step.

        Returns (logits [B, seq_len, vocab], hidden [B, seq_len, n_layers * hidden]).
        """
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        h = self._embed_fn(input_ids)
        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None)

        checkpoint_hiddens = []
        for i, layer in enumerate(self._layers):
            is_linear = getattr(layer, "is_linear", False)
            mask = ssm_mask if is_linear else fa_mask
            h = layer(h, mask=mask, cache=None)
            if i in self.config.target_layer_ids:
                checkpoint_hiddens.append(h)

        if self.hidden_dtype is not None:
            checkpoint_hiddens = [c.astype(self.hidden_dtype) for c in checkpoint_hiddens]
        target_hidden = mx.concatenate(checkpoint_hiddens, axis=-1)

        h = self._norm_fn(h) if self._norm_fn else h
        logits = self._lm_head_fn(h)

        return logits, target_hidden

    def extract_hidden_states(self, input_ids: mx.array) -> mx.array:
        """Extract checkpoint hidden states (convenience wrapper).

        Returns concatenated hidden states [B, seq_len, num_layers * hidden_size].
        """
        _, hidden = self._forward_target(input_ids)
        return hidden

    def draft_tokens(self, input_ids: mx.array,
                     target_hidden: mx.array | None = None,
                     num_denoise_steps: int | None = None,
                     draft_temperature: float | None = None,
                     ) -> tuple[mx.array, mx.array]:
        """Generate draft tokens via block diffusion.

        The draft block is [anchor_token, MASK, MASK, ...] where anchor is the
        last context token. The drafter conditions on ALL context hidden states.
        Returns block_size-1 draft predictions (mask positions only).

        Args:
            input_ids: Context token IDs [B, seq_len]
            target_hidden: Pre-computed checkpoint hidden states. If None,
                extracts them via a target forward pass.
            num_denoise_steps: Number of denoising iterations (default: 1).
                Each step re-embeds predicted tokens and re-runs drafter.
            draft_temperature: Temperature for draft logits (default: None = argmax).
                Values > 1.0 flatten overconfident predictions.

        Returns:
            (draft_token_ids [B, block_size-1], draft_logits [B, block_size-1, vocab])
        """
        import time
        t0 = time.perf_counter()

        if target_hidden is None:
            target_hidden = self.extract_hidden_states(input_ids)

        steps = num_denoise_steps or self._num_denoise_steps
        temp = draft_temperature or self._draft_temperature

        B = input_ids.shape[0]
        bs = self.block_size
        anchor = input_ids[:, -1:]
        mask_ids = mx.full(
            (B, bs - 1),
            self.config.mask_token_id,
            dtype=mx.int32,
        )
        block_ids = mx.concatenate([anchor, mask_ids], axis=1)

        drafter_fn = self._compiled_drafter_fn or self.drafter

        for step_i in range(steps):
            noise_embedding = self._embed_fn(block_ids)
            if self._num_active_layers is not None:
                refined = drafter_fn(noise_embedding, target_hidden,
                                     num_active_layers=self._num_active_layers)
            else:
                refined = drafter_fn(noise_embedding, target_hidden)

            if self._lm_head_fn is not None:
                logits = self._lm_head_fn(refined)
            else:
                logits = refined

            draft_logits = logits[:, 1:, :]

            if temp and temp != 1.0:
                scaled_logits = draft_logits / temp
            else:
                scaled_logits = draft_logits

            draft_ids = mx.argmax(scaled_logits, axis=-1)

            if step_i < steps - 1:
                block_ids = mx.concatenate([anchor, draft_ids], axis=1)

        mx.eval(draft_ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["draft_times_ms"].append(elapsed_ms)
        self.stats["total_drafts"] += bs - 1

        return draft_ids, draft_logits

    def verify_drafts(self, input_ids: mx.array, draft_ids: mx.array) -> tuple[mx.array, int]:
        """Verify draft tokens against target model (single forward pass).

        The target model sees [context_tokens, draft_tokens]. Logits at position
        i predict token i+1, so we compare target logits[ctx_len-1 : ctx_len-1+n_draft]
        against the draft tokens.

        Returns (accepted_token_ids, num_accepted).
        """
        import time
        t0 = time.perf_counter()

        full_ids = mx.concatenate([input_ids, draft_ids], axis=-1)
        logits = self.target(full_ids)

        ctx_len = input_ids.shape[-1]
        n_draft = draft_ids.shape[-1]

        verify_logits = logits[:, ctx_len - 1 : ctx_len - 1 + n_draft, :]
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

    def _make_cache(self):
        """Create per-layer KV/SSM cache for the target model."""
        if self._inner_model is not None and hasattr(self._inner_model, 'make_cache'):
            return self._inner_model.make_cache()

        from mlx_lm.models.cache import KVCache
        try:
            from mlx_lm.models.cache import ArraysCache
        except ImportError:
            ArraysCache = None

        cache = []
        for layer in self._layers:
            is_ssm = getattr(layer, 'is_linear', False) or getattr(layer, 'is_mamba', False)
            if is_ssm and ArraysCache is not None:
                cache.append(ArraysCache(size=2))
            else:
                cache.append(KVCache())
        return cache

    def _find_cache_refs(self, cache: list):
        """Find representative attention and SSM cache entries for mask creation."""
        fa_cache = None
        ssm_cache_ref = None
        for i, layer in enumerate(self._layers):
            if getattr(layer, 'is_linear', False):
                if ssm_cache_ref is None:
                    ssm_cache_ref = cache[i]
            else:
                if fa_cache is None:
                    fa_cache = cache[i]
        return fa_cache, ssm_cache_ref

    def _forward_target_cached(self, input_ids: mx.array, cache: list
                                ) -> tuple[mx.array, mx.array]:
        """Forward pass using KV/SSM cache. Processes only new tokens.

        Returns (logits [B, new_len, vocab], new_hidden [B, new_len, checkpoints*hidden]).
        """
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        h = self._embed_fn(input_ids)
        fa_cache, ssm_cache_ref = self._find_cache_refs(cache)

        fa_mask = create_attention_mask(h, fa_cache)
        ssm_mask = create_ssm_mask(h, ssm_cache_ref)

        checkpoint_hiddens = []
        for i, layer in enumerate(self._layers):
            is_linear = getattr(layer, 'is_linear', False)
            mask = ssm_mask if is_linear else fa_mask
            h = layer(h, mask=mask, cache=cache[i])
            if i in self.config.target_layer_ids:
                checkpoint_hiddens.append(h)

        if self.hidden_dtype is not None:
            checkpoint_hiddens = [c.astype(self.hidden_dtype) for c in checkpoint_hiddens]
        new_hidden = mx.concatenate(checkpoint_hiddens, axis=-1)

        h = self._norm_fn(h) if self._norm_fn else h
        logits = self._lm_head_fn(h)

        return logits, new_hidden

    def _forward_target_tree_cached(self, tree_token_ids: mx.array,
                                     tree_attention_mask: mx.array,
                                     cache: list) -> mx.array:
        """Forward pass for tree verification with KV cache.

        Tree nodes attend to all cached context (prefix) plus other tree nodes
        according to tree_attention_mask. Returns logits only (no hidden state
        extraction needed during tree verification).

        Args:
            tree_token_ids: [1, tree_size] token IDs for tree nodes
            tree_attention_mask: [tree_size, tree_size] boolean mask (True=attend)
            cache: per-layer cache list (will be mutated — snapshot before calling)

        Returns:
            logits [1, tree_size, vocab]
        """
        from mlx_lm.models.base import create_ssm_mask

        h = self._embed_fn(tree_token_ids)
        fa_cache, ssm_cache_ref = self._find_cache_refs(cache)

        tree_size = h.shape[1]
        cache_offset = fa_cache.offset if fa_cache is not None else 0

        prefix_part = mx.ones((tree_size, cache_offset), dtype=mx.bool_)
        full_tree_mask = mx.concatenate([prefix_part, tree_attention_mask], axis=-1)
        fa_mask = (~full_tree_mask).astype(h.dtype) * -1e9

        ssm_mask = create_ssm_mask(h, ssm_cache_ref)

        for i, layer in enumerate(self._layers):
            is_linear = getattr(layer, 'is_linear', False)
            mask = ssm_mask if is_linear else fa_mask
            h = layer(h, mask=mask, cache=cache[i])

        h = self._norm_fn(h) if self._norm_fn else h
        logits = self._lm_head_fn(h)
        return logits

    def generate(self, prompt: str, max_tokens: int = 128,
                 use_cache: bool = True,
                 accept_top_k: int = 1) -> tuple[str, dict]:
        """Generate text with DFlash speculative decoding.

        With use_cache=True (default), uses KV/SSM cache to avoid
        reprocessing the full context each step. Falls back to non-cached
        mode if the target model doesn't support caching.

        Args:
            accept_top_k: Accept draft if in target's top-k (default 1 = strict
                greedy matching). Higher values trade quality for speed.

        Returns (generated_text, stats_summary).
        """
        if use_cache:
            try:
                return self._generate_cached(prompt, max_tokens,
                                             accept_top_k=accept_top_k)
            except Exception:
                pass
        return self._generate_no_cache(prompt, max_tokens,
                                       accept_top_k=accept_top_k)

    def _generate_cached(self, prompt: str, max_tokens: int = 128,
                         accept_top_k: int = 1) -> tuple[str, dict]:
        """Cached DFlash generation — processes only new tokens per step."""
        import copy
        import time
        import numpy as np

        self.stats = {
            "total_drafts": 0, "total_accepted": 0, "total_target_calls": 0,
            "draft_times_ms": [], "verify_times_ms": [], "verify_steps": 0,
        }

        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        t_start = time.perf_counter()

        cache = self._make_cache()
        input_ids = mx.array([generated])
        prompt_logits, hidden_buffer = self._forward_target_cached(input_ids, cache)
        mx.eval(prompt_logits, hidden_buffer)
        self.stats["total_target_calls"] += 1
        last_logit = prompt_logits[:, -1:, :]

        while len(generated) - len(tokens) < max_tokens:
            all_ids = mx.array([generated])

            t_draft = time.perf_counter()
            draft_ids, _ = self.draft_tokens(all_ids, target_hidden=hidden_buffer)
            n_draft = draft_ids.shape[-1]
            self.stats["draft_times_ms"].append((time.perf_counter() - t_draft) * 1000)

            t_verify = time.perf_counter()
            cache_snapshot = copy.deepcopy(cache)

            verify_logits, verify_hidden = self._forward_target_cached(
                draft_ids, cache,
            )
            mx.eval(verify_logits)
            self.stats["total_target_calls"] += 1

            # last_logit predicts draft[0]; verify_logits[:,i,:] predicts draft[i+1]
            predictions = mx.concatenate(
                [last_logit, verify_logits[:, :-1, :]], axis=1,
            )

            draft_np = draft_ids[0].tolist()

            if accept_top_k <= 1:
                target_ids = mx.argmax(predictions, axis=-1)
                mx.eval(target_ids)
                target_np = target_ids[0].tolist()
                num_accepted = 0
                for d, t in zip(draft_np, target_np):
                    if d == t:
                        num_accepted += 1
                    else:
                        break
            else:
                mx.eval(predictions)
                pred_np = predictions[0]
                num_accepted = 0
                for i, d in enumerate(draft_np):
                    top_k_ids = mx.argpartition(pred_np[i], kth=-accept_top_k)[-accept_top_k:]
                    mx.eval(top_k_ids)
                    if d in top_k_ids.tolist():
                        num_accepted += 1
                    else:
                        break

            if num_accepted == n_draft:
                bonus = int(mx.argmax(verify_logits[0, -1]).item())
            elif num_accepted == 0:
                bonus = int(mx.argmax(last_logit[0, 0]).item())
            else:
                bonus = int(mx.argmax(verify_logits[0, num_accepted - 1]).item())

            accepted = draft_np[:num_accepted] + [bonus]

            self.stats["verify_times_ms"].append((time.perf_counter() - t_verify) * 1000)
            self.stats["total_drafts"] += n_draft
            self.stats["total_accepted"] += len(accepted)
            self.stats["verify_steps"] += 1

            # Rollback cache to pre-draft state, replay accepted + bonus
            for i in range(len(cache)):
                cache[i] = cache_snapshot[i]

            replay_ids = mx.array([accepted])
            replay_logits, replay_hidden = self._forward_target_cached(
                replay_ids, cache,
            )
            mx.eval(replay_logits, replay_hidden)
            self.stats["total_target_calls"] += 1

            hidden_buffer = mx.concatenate([hidden_buffer, replay_hidden], axis=1)
            last_logit = replay_logits[:, -1:, :]

            generated.extend(accepted)

            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None and eos in accepted:
                break

        elapsed = time.perf_counter() - t_start
        n_gen = len(generated) - len(tokens)
        tok_per_sec = n_gen / elapsed if elapsed > 0 else 0

        steps = max(1, self.stats["verify_steps"])
        summary = {
            "tokens_generated": n_gen,
            "wall_time_s": round(elapsed, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "total_target_calls": self.stats["total_target_calls"],
            "total_drafts": self.stats["total_drafts"],
            "total_accepted": self.stats["total_accepted"],
            "acceptance_rate": round(self.stats["total_accepted"] / max(1, self.stats["total_drafts"]), 3),
            "tokens_per_step": round(self.stats["total_accepted"] / steps, 1),
            "avg_draft_ms": round(float(np.mean(self.stats["draft_times_ms"])), 1) if self.stats["draft_times_ms"] else 0,
            "avg_verify_ms": round(float(np.mean(self.stats["verify_times_ms"])), 1) if self.stats["verify_times_ms"] else 0,
            "cached": True,
        }

        text = self.tokenizer.decode(generated[len(tokens):])
        return text, summary

    def _generate_no_cache(self, prompt: str, max_tokens: int = 128,
                           accept_top_k: int = 1) -> tuple[str, dict]:
        """Non-cached DFlash generation — reprocesses full context each step."""
        import time

        self.stats = {
            "total_drafts": 0, "total_accepted": 0, "total_target_calls": 0,
            "draft_times_ms": [], "verify_times_ms": [],
        }

        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        t_start = time.perf_counter()

        input_ids = mx.array([generated])
        _, target_hidden = self._forward_target(input_ids)
        self.stats["total_target_calls"] += 1

        while len(generated) - len(tokens) < max_tokens:
            input_ids = mx.array([generated])

            t_draft = time.perf_counter()
            draft_ids, _ = self.draft_tokens(input_ids, target_hidden=target_hidden)
            self.stats["draft_times_ms"].append((time.perf_counter() - t_draft) * 1000)

            t_verify = time.perf_counter()
            full_ids = mx.concatenate([input_ids, draft_ids], axis=-1)
            verify_logits, new_hidden = self._forward_target(full_ids)
            self.stats["total_target_calls"] += 1

            ctx_len = input_ids.shape[-1]
            n_draft = draft_ids.shape[-1]
            pred_logits = verify_logits[:, ctx_len - 1 : ctx_len - 1 + n_draft, :]

            draft_np = draft_ids[0].tolist()

            if accept_top_k <= 1:
                target_ids = mx.argmax(pred_logits, axis=-1)
                mx.eval(target_ids)
                target_np = target_ids[0].tolist()
                num_accepted = 0
                for d, t in zip(draft_np, target_np):
                    if d == t:
                        num_accepted += 1
                    else:
                        break
            else:
                mx.eval(pred_logits)
                num_accepted = 0
                for i, d in enumerate(draft_np):
                    top_k_ids = mx.argpartition(pred_logits[0, i], kth=-accept_top_k)[-accept_top_k:]
                    mx.eval(top_k_ids)
                    if d in top_k_ids.tolist():
                        num_accepted += 1
                    else:
                        break

            target_greedy = mx.argmax(pred_logits, axis=-1)
            mx.eval(target_greedy)
            target_np = target_greedy[0].tolist()
            bonus_idx = num_accepted
            if bonus_idx < len(target_np):
                accepted = draft_np[:num_accepted] + [target_np[bonus_idx]]
            else:
                accepted = draft_np[:num_accepted]

            self.stats["verify_times_ms"].append((time.perf_counter() - t_verify) * 1000)
            self.stats["total_drafts"] += n_draft
            self.stats["total_accepted"] += len(accepted)

            if len(accepted) == 0:
                break

            generated.extend(accepted)

            n_kept = ctx_len + len(accepted)
            target_hidden = new_hidden[:, :n_kept, :]

            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None and eos in accepted:
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
            "cached": False,
        }

        text = self.tokenizer.decode(generated[len(tokens):])
        return text, summary

    def generate_with_tree(self, prompt: str, max_tokens: int = 128,
                           tree_width: int = 5, max_tree_size: int = 60,
                           use_cache: bool = True,
                           ) -> tuple[str, dict]:
        """Generate text with DFlash + DDTree speculative decoding.

        With use_cache=True (default), uses KV/SSM cache for context and
        tree verification with a custom tree attention mask. Falls back to
        non-cached mode if the target model doesn't support caching.

        Returns (generated_text, stats_summary).
        """
        if use_cache:
            try:
                return self._generate_with_tree_cached(
                    prompt, max_tokens, tree_width, max_tree_size)
            except Exception:
                pass
        return self._generate_with_tree_no_cache(
            prompt, max_tokens, tree_width, max_tree_size)

    def _generate_with_tree_cached(self, prompt: str, max_tokens: int = 128,
                                    tree_width: int = 5, max_tree_size: int = 60,
                                    ) -> tuple[str, dict]:
        """Cached DDTree generation — tree verification with KV cache."""
        import copy
        import time
        import numpy as np
        from mlx_flash_compress.ddtree import DDTreeBuilder, DDTreeConfig

        tree_config = DDTreeConfig(
            tree_width=tree_width,
            max_tree_size=max_tree_size,
        )
        tree_builder = DDTreeBuilder(tree_config)

        self.stats = {
            "total_drafts": 0, "total_accepted": 0, "total_target_calls": 0,
            "draft_times_ms": [], "verify_times_ms": [], "verify_steps": 0,
        }

        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        t_start = time.perf_counter()

        cache = self._make_cache()
        input_ids = mx.array([generated])
        prompt_logits, hidden_buffer = self._forward_target_cached(input_ids, cache)
        mx.eval(prompt_logits, hidden_buffer)
        self.stats["total_target_calls"] += 1
        last_logit = prompt_logits[:, -1:, :]

        while len(generated) - len(tokens) < max_tokens:
            all_ids = mx.array([generated])

            t_draft = time.perf_counter()
            _, draft_logits = self.draft_tokens(all_ids, target_hidden=hidden_buffer)
            tree = tree_builder.build_tree(draft_logits)
            self.stats["draft_times_ms"].append((time.perf_counter() - t_draft) * 1000)

            if tree.size == 0:
                break

            t_verify = time.perf_counter()
            cache_snapshot = copy.deepcopy(cache)
            ctx_len = len(generated)

            tree_ids = tree.token_ids.reshape(1, -1)
            tree_logits = self._forward_target_tree_cached(
                tree_ids, tree.attention_mask, cache,
            )
            mx.eval(tree_logits)
            self.stats["total_target_calls"] += 1

            # verify_tree expects [1, ctx_len + tree_size, vocab] —
            # pad positions 0..ctx_len-2 with zeros (never accessed),
            # set ctx_len-1 = last_logit, ctx_len.. = tree_logits
            vocab = tree_logits.shape[-1]
            padding = mx.zeros((1, ctx_len - 1, vocab), dtype=tree_logits.dtype)
            full_logits = mx.concatenate([padding, last_logit, tree_logits], axis=1)

            accepted, n_accepted = tree_builder.verify_tree(
                tree, full_logits, ctx_len,
            )
            self.stats["verify_times_ms"].append((time.perf_counter() - t_verify) * 1000)
            self.stats["total_drafts"] += tree.size
            self.stats["total_accepted"] += n_accepted
            self.stats["verify_steps"] += 1

            if n_accepted == 0:
                for i in range(len(cache)):
                    cache[i] = cache_snapshot[i]
                break

            for i in range(len(cache)):
                cache[i] = cache_snapshot[i]

            replay_ids = mx.array([accepted])
            replay_logits, replay_hidden = self._forward_target_cached(
                replay_ids, cache,
            )
            mx.eval(replay_logits, replay_hidden)
            self.stats["total_target_calls"] += 1

            hidden_buffer = mx.concatenate([hidden_buffer, replay_hidden], axis=1)
            last_logit = replay_logits[:, -1:, :]

            generated.extend(accepted)

            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None and eos in accepted:
                break

        elapsed = time.perf_counter() - t_start
        n_gen = len(generated) - len(tokens)
        tok_per_sec = n_gen / elapsed if elapsed > 0 else 0

        tree_stats = tree_builder.get_stats()
        steps = max(1, self.stats["verify_steps"])
        summary = {
            "tokens_generated": n_gen,
            "wall_time_s": round(elapsed, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "total_target_calls": self.stats["total_target_calls"],
            "total_tree_nodes": self.stats["total_drafts"],
            "total_accepted": self.stats["total_accepted"],
            "acceptance_rate": round(self.stats["total_accepted"] / max(1, self.stats["total_drafts"]), 3),
            "tokens_per_step": round(self.stats["total_accepted"] / steps, 1),
            "avg_draft_ms": round(float(np.mean(self.stats["draft_times_ms"])), 1) if self.stats["draft_times_ms"] else 0,
            "avg_verify_ms": round(float(np.mean(self.stats["verify_times_ms"])), 1) if self.stats["verify_times_ms"] else 0,
            "avg_tree_size": tree_stats["avg_tree_size"],
            "avg_path_length": tree_stats["avg_path_length"],
            "cached": True,
        }

        text = self.tokenizer.decode(generated[len(tokens):])
        return text, summary

    def _generate_with_tree_no_cache(self, prompt: str, max_tokens: int = 128,
                                      tree_width: int = 5, max_tree_size: int = 60,
                                      ) -> tuple[str, dict]:
        """Non-cached DDTree generation — reprocesses full context each step."""
        import time
        import numpy as np
        from mlx_flash_compress.ddtree import DDTreeBuilder, DDTreeConfig

        tree_config = DDTreeConfig(
            tree_width=tree_width,
            max_tree_size=max_tree_size,
        )
        tree_builder = DDTreeBuilder(tree_config)

        self.stats = {
            "total_drafts": 0, "total_accepted": 0, "total_target_calls": 0,
            "draft_times_ms": [], "verify_times_ms": [],
        }

        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        t_start = time.perf_counter()

        input_ids = mx.array([generated])
        _, target_hidden = self._forward_target(input_ids)
        self.stats["total_target_calls"] += 1

        while len(generated) - len(tokens) < max_tokens:
            input_ids = mx.array([generated])
            ctx_len = input_ids.shape[-1]

            t_draft = time.perf_counter()
            _, draft_logits = self.draft_tokens(input_ids, target_hidden=target_hidden)
            tree = tree_builder.build_tree(draft_logits)
            self.stats["draft_times_ms"].append((time.perf_counter() - t_draft) * 1000)

            t_verify = time.perf_counter()
            full_ids = mx.concatenate([input_ids, tree.token_ids.reshape(1, -1)], axis=-1)
            verify_logits, new_hidden = self._forward_target(full_ids)
            self.stats["total_target_calls"] += 1

            accepted, n_accepted = tree_builder.verify_tree(
                tree, verify_logits, ctx_len
            )
            self.stats["verify_times_ms"].append((time.perf_counter() - t_verify) * 1000)
            self.stats["total_drafts"] += tree.size
            self.stats["total_accepted"] += n_accepted

            if n_accepted == 0:
                break

            generated.extend(accepted)

            n_kept = ctx_len + n_accepted
            if n_kept <= new_hidden.shape[1]:
                target_hidden = new_hidden[:, :n_kept, :]
            else:
                input_ids = mx.array([generated])
                _, target_hidden = self._forward_target(input_ids)
                self.stats["total_target_calls"] += 1

            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None and eos in accepted:
                break

        elapsed = time.perf_counter() - t_start
        n_gen = len(generated) - len(tokens)
        tok_per_sec = n_gen / elapsed if elapsed > 0 else 0

        tree_stats = tree_builder.get_stats()
        summary = {
            "tokens_generated": n_gen,
            "wall_time_s": round(elapsed, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "total_target_calls": self.stats["total_target_calls"],
            "total_tree_nodes": self.stats["total_drafts"],
            "total_accepted": self.stats["total_accepted"],
            "acceptance_rate": round(self.stats["total_accepted"] / max(1, self.stats["total_drafts"]), 3),
            "tokens_per_step": round(self.stats["total_accepted"] / max(1, self.stats["total_target_calls"]), 1),
            "avg_draft_ms": round(float(np.mean(self.stats["draft_times_ms"])), 1) if self.stats["draft_times_ms"] else 0,
            "avg_verify_ms": round(float(np.mean(self.stats["verify_times_ms"])), 1) if self.stats["verify_times_ms"] else 0,
            "avg_tree_size": tree_stats["avg_tree_size"],
            "avg_path_length": tree_stats["avg_path_length"],
            "cached": False,
        }

        text = self.tokenizer.decode(generated[len(tokens):])
        return text, summary
