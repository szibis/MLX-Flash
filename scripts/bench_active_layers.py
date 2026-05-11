#!/usr/bin/env python3
"""Benchmark layer pruning end-to-end on real models.

Tests DFlash with different num_active_layers values to find the optimal
tradeoff between drafter cost and acceptance rate.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="def binary_search(arr, target):\n    ")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--target", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    parser.add_argument("--drafter", default="z-lab/Qwen3.6-35B-A3B-DFlash")
    args = parser.parse_args()

    print("Loading models...")
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    model, tokenizer = load(args.target)
    drafter_path = snapshot_download(args.drafter)
    drafter, config = DFlashDraftModel.from_pretrained(drafter_path)
    print("  Done.\n")

    print("=" * 80)
    print("DFlash Layer Pruning Benchmark")
    print("=" * 80)
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Warmup
    runner = DFlashRunner(model, tokenizer, drafter, config)
    runner.generate(args.prompt, max_tokens=4, use_cache=True)

    print(
        f"  {'Layers':>8s} | {'tok/s':>8s} | {'accept':>8s} | {'tok/step':>8s} | "
        f"{'draft_ms':>9s} | {'verify_ms':>10s} | {'calls':>5s}"
    )
    print("-" * 80)

    for n_layers in [8, 6, 4, 2, 1]:
        label = "all" if n_layers == 8 else str(n_layers)
        active = None if n_layers == 8 else n_layers

        results = []
        for _ in range(args.trials):
            runner = DFlashRunner(model, tokenizer, drafter, config, num_active_layers=active)
            _, stats = runner.generate(args.prompt, max_tokens=args.max_tokens, use_cache=True)
            results.append(stats)

        results.sort(key=lambda s: s["tok_per_sec"])
        s = results[len(results) // 2]

        print(
            f"  {label:>8s} | {s['tok_per_sec']:8.1f} | "
            f"{s['acceptance_rate']:7.1%} | {s['tokens_per_step']:8.1f} | "
            f"{s['avg_draft_ms']:9.1f} | {s['avg_verify_ms']:10.1f} | "
            f"{s['total_target_calls']:>5d}"
        )


if __name__ == "__main__":
    main()
