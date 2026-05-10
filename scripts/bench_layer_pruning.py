#!/usr/bin/env python3
"""Benchmark layer pruning: drafter with N of 8 layers active."""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig


def bench(drafter, noise_embedding, target_hidden, num_layers=None, n=20, warmup=5):
    for _ in range(warmup):
        out = drafter(noise_embedding, target_hidden, num_active_layers=num_layers)
        mx.eval(out)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = drafter(noise_embedding, target_hidden, num_active_layers=num_layers)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    times.sort()
    return times[len(times) // 2]


def main():
    config = DFlashModelConfig()
    drafter = DFlashDraftModel(config)
    mx.eval(drafter.parameters())

    ctx_len = 32
    B = 1
    num_ckpt = len(config.target_layer_ids)
    target_hidden = mx.random.normal((B, ctx_len, num_ckpt * config.hidden_size))
    noise_embedding = mx.random.normal((B, config.block_size, config.hidden_size))

    print("Layer pruning: drafter forward pass time")
    print(f"{'Layers':>8s} | {'Time (ms)':>10s} | {'vs 8-layer':>10s}")
    print("-" * 35)

    base = bench(drafter, noise_embedding, target_hidden, num_layers=None)
    print(f"{'8 (all)':>8s} | {base*1000:10.2f} | {'baseline':>10s}")

    for n_layers in [6, 4, 2, 1]:
        t = bench(drafter, noise_embedding, target_hidden, num_layers=n_layers)
        speedup = base / t if t > 0 else 0
        print(f"{n_layers:>8d} | {t*1000:10.2f} | {speedup:9.2f}x")


if __name__ == "__main__":
    main()
