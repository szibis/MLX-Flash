#!/usr/bin/env python3
"""Benchmark different draft counts (block sizes) with 8-bit quantized drafter.

Tests the tradeoff: fewer drafts = faster verify but fewer tokens per step.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="def binary_search(arr, target):\n    ")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print("Loading models...")
    from mlx_lm import load
    from huggingface_hub import snapshot_download
    model, tokenizer = load("mlx-community/Qwen3.6-35B-A3B-4bit")
    drafter_path = snapshot_download("z-lab/Qwen3.6-35B-A3B-DFlash")
    print("  Done.\n")

    print("=" * 80)
    print("Draft Count vs Throughput (8-bit quantized drafter)")
    print("=" * 80)
    print(f"Prompt: {args.prompt[:50]}...")
    print()

    # Warmup
    drafter, config = DFlashDraftModel.from_pretrained(drafter_path)
    nn.quantize(drafter, group_size=64, bits=8)
    mx.eval(drafter.parameters())
    runner = DFlashRunner(model, tokenizer, drafter, config)
    runner.generate(args.prompt, max_tokens=4, use_cache=True)

    print(f"  {'block_size':>10s} | {'drafts':>6s} | {'tok/s':>8s} | {'accept':>7s} | "
          f"{'tok/step':>8s} | {'draft_ms':>8s} | {'verify_ms':>9s} | {'calls':>5s}")
    print("-" * 85)

    for bs in [16, 12, 8, 6, 4]:
        drafter_i, config_i = DFlashDraftModel.from_pretrained(drafter_path)
        nn.quantize(drafter_i, group_size=64, bits=8)
        mx.eval(drafter_i.parameters())

        results = []
        for _ in range(args.trials):
            runner = DFlashRunner(model, tokenizer, drafter_i, config_i,
                                  inference_block_size=bs)
            _, stats = runner.generate(args.prompt, max_tokens=args.max_tokens, use_cache=True)
            results.append(stats)

        results.sort(key=lambda s: s["tok_per_sec"])
        s = results[len(results) // 2]

        print(f"  {bs:>10d} | {bs-1:>6d} | {s['tok_per_sec']:8.1f} | "
              f"{s['acceptance_rate']:7.1%} | {s['tokens_per_step']:8.1f} | "
              f"{s['avg_draft_ms']:8.1f} | {s['avg_verify_ms']:9.1f} | "
              f"{s['total_target_calls']:>5d}")


if __name__ == "__main__":
    main()
