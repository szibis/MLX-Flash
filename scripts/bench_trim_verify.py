#!/usr/bin/env python3
"""Benchmark trim-based verify vs rollback+replay on real models.

Tests both approaches at different block sizes with 8-bit quantized drafter.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner


def main():
    print("Loading models...")
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen3.6-35B-A3B-4bit")
    drafter_path = snapshot_download("z-lab/Qwen3.6-35B-A3B-DFlash")
    print("  Done.\n")

    prompt = "def binary_search(arr, target):\n    "
    max_tokens = 32
    trials = 3

    # Warmup
    drafter, config = DFlashDraftModel.from_pretrained(drafter_path)
    nn.quantize(drafter, group_size=64, bits=8)
    mx.eval(drafter.parameters())
    runner = DFlashRunner(model, tokenizer, drafter, config)
    runner.generate(prompt, max_tokens=4, use_cache=True)

    print("=" * 90)
    print("Trim-Verify vs Rollback+Replay (8-bit quantized drafter)")
    print("=" * 90)
    print(
        f"  {'Config':35s} | {'tok/s':>8s} | {'accept':>7s} | {'tok/step':>8s} | "
        f"{'draft':>7s} | {'verify':>8s} | {'calls':>5s}"
    )
    print("-" * 90)

    for bs in [16, 8, 4]:
        for trim in [False, True]:
            label = f"bs={bs} {'trim' if trim else 'replay':>6s}"

            results = []
            for _ in range(trials):
                d, c = DFlashDraftModel.from_pretrained(drafter_path)
                nn.quantize(d, group_size=64, bits=8)
                mx.eval(d.parameters())
                r = DFlashRunner(model, tokenizer, d, c, inference_block_size=bs, trim_verify=trim)
                text, stats = r.generate(prompt, max_tokens=max_tokens, use_cache=True)
                results.append(stats)

            results.sort(key=lambda s: s["tok_per_sec"])
            s = results[len(results) // 2]

            print(
                f"  {label:35s} | {s['tok_per_sec']:8.1f} | "
                f"{s['acceptance_rate']:7.1%} | {s['tokens_per_step']:8.1f} | "
                f"{s['avg_draft_ms']:7.1f} | {s['avg_verify_ms']:8.1f} | "
                f"{s['total_target_calls']:>5d}"
            )

        print()


if __name__ == "__main__":
    main()
