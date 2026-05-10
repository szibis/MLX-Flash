#!/usr/bin/env python3
"""Micro-benchmark: SDPA vs manual attention in DFlashAttention.

Measures drafter forward pass latency in isolation, comparing both paths.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import (
    DFlashAttention, DFlashDraftModel, DFlashModelConfig, _apply_rotary_emb,
)


def manual_attention_call(self, hidden_states, target_hidden):
    """Manual matmul attention (the old code path)."""
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


def bench_drafter_forward(drafter, noise_embedding, target_hidden, n=20, warmup=5):
    """Time the drafter forward pass."""
    for _ in range(warmup):
        out = drafter(noise_embedding, target_hidden)
        mx.eval(out)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = drafter(noise_embedding, target_hidden)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    return median, times[0], times[-1]


def main():
    config = DFlashModelConfig()
    drafter = DFlashDraftModel(config)
    mx.eval(drafter.parameters())

    for ctx_len in [32, 128, 512]:
        _run_comparison(drafter, config, ctx_len, B=1)


def _run_comparison(drafter, config, ctx_len, B=1):
    block_size = config.block_size
    hidden_dim = config.hidden_size
    num_ckpt = len(config.target_layer_ids)

    target_hidden = mx.random.normal((B, ctx_len, num_ckpt * hidden_dim))
    noise_embedding = mx.random.normal((B, block_size, hidden_dim))

    m1, b1, w1 = bench_drafter_forward(drafter, noise_embedding, target_hidden)
    original_calls = {}
    for layer in drafter.layers:
        original_calls[id(layer.self_attn)] = layer.self_attn.__class__.__call__
        layer.self_attn.__class__.__call__ = manual_attention_call
    m2, b2, w2 = bench_drafter_forward(drafter, noise_embedding, target_hidden)
    for layer in drafter.layers:
        layer.self_attn.__class__.__call__ = original_calls[id(layer.self_attn)]

    speedup = m2 / m1 if m1 > 0 else 0
    delta_pct = (speedup - 1) * 100
    sign = "+" if delta_pct >= 0 else ""
    print(f"  ctx={ctx_len:4d}: SDPA={m1*1000:.2f}ms  Manual={m2*1000:.2f}ms  "
          f"| {speedup:.2f}x ({sign}{delta_pct:.1f}%)")


def _old_main():
    """Kept for reference."""
    ctx_len = 32
    block_size = config.block_size
    hidden_dim = config.hidden_size
    num_ckpt = len(config.target_layer_ids)
    B = 1

    target_hidden = mx.random.normal((B, ctx_len, num_ckpt * hidden_dim))
    noise_embedding = mx.random.normal((B, block_size, hidden_dim))

    print("=" * 60)
    print("DFlash Drafter Forward Pass: SDPA vs Manual Attention")
    print("=" * 60)
    print(f"Config: {config.num_hidden_layers} layers, "
          f"{config.num_attention_heads} heads, "
          f"{config.num_key_value_heads} KV heads (GQA {config.num_attention_heads // config.num_key_value_heads}x), "
          f"head_dim={config.head_dim}")
    print(f"Input: ctx_len={ctx_len}, block_size={block_size}")
    print()

    # --- SDPA (current) ---
    m1, b1, w1 = bench_drafter_forward(drafter, noise_embedding, target_hidden)
    print(f"SDPA (Metal kernel):   median={m1*1000:.2f}ms  "
          f"best={b1*1000:.2f}ms  worst={w1*1000:.2f}ms")

    # --- Manual attention (old code) ---
    original_calls = {}
    for layer in drafter.layers:
        original_calls[id(layer.self_attn)] = layer.self_attn.__class__.__call__
        layer.self_attn.__class__.__call__ = manual_attention_call

    m2, b2, w2 = bench_drafter_forward(drafter, noise_embedding, target_hidden)
    print(f"Manual (matmul+GQA):   median={m2*1000:.2f}ms  "
          f"best={b2*1000:.2f}ms  worst={w2*1000:.2f}ms")

    # Restore
    for layer in drafter.layers:
        layer.self_attn.__class__.__call__ = original_calls[id(layer.self_attn)]

    # --- Summary ---
    print()
    speedup = m2 / m1 if m1 > 0 else 0
    delta_pct = (speedup - 1) * 100
    sign = "+" if delta_pct >= 0 else ""
    print(f"SDPA speedup: {speedup:.2f}x ({sign}{delta_pct:.1f}%)")
    print(f"Per-step savings: {(m2 - m1)*1000:.2f}ms")


if __name__ == "__main__":
    main()
