#!/usr/bin/env python3
"""Benchmark individual DFlash optimizations with A/B comparison.

Loads models once, then tests each optimization independently against
a baseline. Reports delta for each change.

Usage:
  python scripts/bench_dflash_opts.py
  python scripts/bench_dflash_opts.py --prompt "def fib(n):" --max-tokens 32
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import (
    DFlashDraftModel, DFlashModelConfig, DFlashRunner,
)


def run_trial(runner, prompt, max_tokens, use_cache=True, label="", **kwargs):
    """Run a single trial and return stats."""
    text, stats = runner.generate(prompt, max_tokens=max_tokens,
                                   use_cache=use_cache)
    return stats


def run_n_trials(runner, prompt, max_tokens, n=3, use_cache=True,
                 label="baseline", **kwargs):
    """Run n trials and return median stats."""
    results = []
    for i in range(n):
        stats = run_trial(runner, prompt, max_tokens, use_cache=use_cache,
                          label=label, **kwargs)
        results.append(stats)

    # Sort by tok/s, take median
    results.sort(key=lambda s: s["tok_per_sec"])
    median = results[len(results) // 2]
    return median


def print_result(label, stats, baseline_tok_s=None):
    """Print formatted result line."""
    delta = ""
    if baseline_tok_s and baseline_tok_s > 0:
        ratio = stats["tok_per_sec"] / baseline_tok_s
        sign = "+" if ratio >= 1.0 else ""
        delta = f" | {sign}{(ratio - 1) * 100:.0f}%"

    accept = stats.get("acceptance_rate", 0)
    tok_step = stats.get("tokens_per_step", 0)
    print(f"  {label:30s} | {stats['tok_per_sec']:6.1f} tok/s | "
          f"accept={accept:5.1%} | tok/step={tok_step:.1f} | "
          f"calls={stats.get('total_target_calls', 'N/A')}{delta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="def binary_search(arr, target):\n    ")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--target", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    parser.add_argument("--drafter", default="z-lab/Qwen3.6-35B-A3B-DFlash")
    args = parser.parse_args()

    print("=" * 80)
    print("DFlash Optimization Benchmark")
    print("=" * 80)
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Max tokens: {args.max_tokens}, Trials: {args.trials}")
    print()

    # Load models once
    print("Loading models...")
    t0 = time.perf_counter()
    model, tokenizer = load(args.target)
    from huggingface_hub import snapshot_download
    drafter_path = snapshot_download(args.drafter)
    drafter, config = DFlashDraftModel.from_pretrained(drafter_path)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s\n")

    # Warmup
    runner = DFlashRunner(model, tokenizer, drafter, config)
    runner.generate(args.prompt, max_tokens=4, use_cache=False)

    print("-" * 80)
    print(f"  {'Optimization':30s} | {'tok/s':>8s} | {'accept':>7s} | {'tok/step':>8s} | {'calls':>5s} | delta")
    print("-" * 80)

    # ── BASELINE: flat cached (block_size=16) ──
    runner_base = DFlashRunner(model, tokenizer, drafter, config)
    baseline = run_n_trials(runner_base, args.prompt, args.max_tokens,
                            n=args.trials, label="baseline-cached")
    print_result("BASELINE (flat, cached, b16)", baseline)
    base_tok_s = baseline["tok_per_sec"]

    # ── BASELINE: no cache ──
    nocache = run_n_trials(runner_base, args.prompt, args.max_tokens,
                           n=args.trials, use_cache=False, label="no-cache")
    print_result("no-cache (reference)", nocache, base_tok_s)

    print()

    # ── OPT 1: block_size = 8 ──
    runner_b8 = DFlashRunner(model, tokenizer, drafter, config,
                             inference_block_size=8)
    b8 = run_n_trials(runner_b8, args.prompt, args.max_tokens,
                      n=args.trials, label="block-size-8")
    print_result("Opt1: block_size=8", b8, base_tok_s)

    # ── OPT 1b: block_size = 4 ──
    runner_b4 = DFlashRunner(model, tokenizer, drafter, config,
                             inference_block_size=4)
    b4 = run_n_trials(runner_b4, args.prompt, args.max_tokens,
                      n=args.trials, label="block-size-4")
    print_result("Opt1b: block_size=4", b4, base_tok_s)

    print()

    # ── OPT 2: hidden_dtype = float32 ──
    runner_f32 = DFlashRunner(model, tokenizer, drafter, config,
                              hidden_dtype=mx.float32)
    f32 = run_n_trials(runner_f32, args.prompt, args.max_tokens,
                       n=args.trials, label="hidden-float32")
    print_result("Opt2: hidden_dtype=float32", f32, base_tok_s)

    # ── OPT 2b: hidden_dtype = bfloat16 ──
    runner_bf16 = DFlashRunner(model, tokenizer, drafter, config,
                               hidden_dtype=mx.bfloat16)
    bf16 = run_n_trials(runner_bf16, args.prompt, args.max_tokens,
                        n=args.trials, label="hidden-bfloat16")
    print_result("Opt2b: hidden_dtype=bfloat16", bf16, base_tok_s)

    print()

    # ── OPT 3: mx.compile() drafter ──
    runner_compile = DFlashRunner(model, tokenizer, drafter, config,
                                  compile_drafter=True)
    compiled = run_n_trials(runner_compile, args.prompt, args.max_tokens,
                            n=args.trials, label="compile-drafter")
    print_result("Opt3: compile_drafter=True", compiled, base_tok_s)

    print()

    # ── OPT 4: block_size=8 + compile + float32 (combo) ──
    runner_combo = DFlashRunner(model, tokenizer, drafter, config,
                                inference_block_size=8,
                                compile_drafter=True,
                                hidden_dtype=mx.float32)
    combo = run_n_trials(runner_combo, args.prompt, args.max_tokens,
                         n=args.trials, label="combo-b8-compile-f32")
    print_result("Combo: b8+compile+f32", combo, base_tok_s)

    print()
    print("=" * 80)
    print("SUMMARY: Best result vs baseline")
    all_results = {
        "baseline": baseline,
        "block_size=8": b8,
        "block_size=4": b4,
        "hidden_f32": f32,
        "hidden_bf16": bf16,
        "compile": compiled,
        "combo": combo,
    }
    best_name = max(all_results, key=lambda k: all_results[k]["tok_per_sec"])
    best = all_results[best_name]
    speedup = best["tok_per_sec"] / base_tok_s if base_tok_s > 0 else 0
    print(f"  Best: {best_name} at {best['tok_per_sec']:.1f} tok/s "
          f"({speedup:.2f}x vs baseline)")
    print(f"  Acceptance: {best['acceptance_rate']:.1%}, "
          f"Tokens/step: {best['tokens_per_step']:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
