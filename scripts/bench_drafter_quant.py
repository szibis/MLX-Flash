#!/usr/bin/env python3
"""Benchmark drafter quantization: bfloat16 vs 4-bit on real models.

Tests both drafter forward pass time and end-to-end acceptance rate.
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
    print("DFlash Drafter Quantization Benchmark")
    print("=" * 80)
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Max tokens: {args.max_tokens}")
    print()

    def run_trial(runner, trials):
        results = []
        for _ in range(trials):
            _, stats = runner.generate(args.prompt, max_tokens=args.max_tokens, use_cache=True)
            results.append(stats)
        results.sort(key=lambda s: s["tok_per_sec"])
        return results[len(results) // 2]

    def print_stats(label, s, baseline_tok=None):
        delta = ""
        if baseline_tok and baseline_tok > 0:
            ratio = s["tok_per_sec"] / baseline_tok
            sign = "+" if ratio >= 1 else ""
            delta = f" | {sign}{(ratio - 1) * 100:.0f}%"
        print(
            f"  {label:30s} | {s['tok_per_sec']:6.1f} tok/s | "
            f"accept={s['acceptance_rate']:5.1%} | tok/step={s['tokens_per_step']:.1f} | "
            f"draft={s['avg_draft_ms']:.0f}ms | verify={s['avg_verify_ms']:.0f}ms{delta}"
        )

    # Warmup
    runner = DFlashRunner(model, tokenizer, drafter, config)
    runner.generate(args.prompt, max_tokens=4, use_cache=True)

    print(
        f"  {'Config':30s} | {'tok/s':>8s} | {'accept':>7s} | {'tok/step':>8s} | "
        f"{'draft':>7s} | {'verify':>10s} | delta"
    )
    print("-" * 95)

    # Baseline: bf16 drafter, 8 layers
    s_base = run_trial(DFlashRunner(model, tokenizer, drafter, config), args.trials)
    print_stats("BASELINE (bf16, 8 layers)", s_base)
    base_tok = s_base["tok_per_sec"]

    # Quantize drafter to 4-bit
    drafter_q4, config_q4 = DFlashDraftModel.from_pretrained(drafter_path)
    nn.quantize(drafter_q4, group_size=64, bits=4)
    mx.eval(drafter_q4.parameters())

    s_q4 = run_trial(DFlashRunner(model, tokenizer, drafter_q4, config_q4), args.trials)
    print_stats("4-bit drafter, 8 layers", s_q4, base_tok)

    # Quantize to 8-bit
    drafter_q8, config_q8 = DFlashDraftModel.from_pretrained(drafter_path)
    nn.quantize(drafter_q8, group_size=64, bits=8)
    mx.eval(drafter_q8.parameters())

    s_q8 = run_trial(DFlashRunner(model, tokenizer, drafter_q8, config_q8), args.trials)
    print_stats("8-bit drafter, 8 layers", s_q8, base_tok)

    print()

    # Combo: 4-bit + 4 layers
    drafter_combo, config_combo = DFlashDraftModel.from_pretrained(drafter_path)
    nn.quantize(drafter_combo, group_size=64, bits=4)
    mx.eval(drafter_combo.parameters())

    s_combo = run_trial(DFlashRunner(model, tokenizer, drafter_combo, config_combo, num_active_layers=4), args.trials)
    print_stats("4-bit drafter, 4 layers", s_combo, base_tok)

    # Combo: 4-bit + 2 layers
    drafter_combo2, config_combo2 = DFlashDraftModel.from_pretrained(drafter_path)
    nn.quantize(drafter_combo2, group_size=64, bits=4)
    mx.eval(drafter_combo2.parameters())

    s_combo2 = run_trial(
        DFlashRunner(model, tokenizer, drafter_combo2, config_combo2, num_active_layers=2), args.trials
    )
    print_stats("4-bit drafter, 2 layers", s_combo2, base_tok)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
