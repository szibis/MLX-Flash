#!/usr/bin/env python3
"""Benchmark DFlash profiles on a model.

Auto-detects model characteristics, measures AR baseline, then tests
each relevant DFlash configuration. Outputs a comparison matrix.

Usage:
  python scripts/bench_profiles.py
  python scripts/bench_profiles.py --target mlx-community/Qwen3.6-35B-A3B-4bit
  python scripts/bench_profiles.py --prompt "def fibonacci(n):" --max-tokens 64
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner
from mlx_flash_compress.dflash_profile import (
    PROFILES,
    detect_model,
    measure_ar_baseline,
    select_profile,
)


def run_dflash_trial(model, tokenizer, drafter, config, prompt, max_tokens, trials=3, **kwargs):
    """Run DFlash with given config and return median stats."""
    results = []
    for _ in range(trials):
        d, c = DFlashDraftModel.from_pretrained(drafter_path_global)
        if kwargs.get("quantize_drafter"):
            nn.quantize(d, group_size=64, bits=kwargs["quantize_drafter"])
            mx.eval(d.parameters())
        runner_kwargs = {k: v for k, v in kwargs.items() if k != "quantize_drafter"}
        r = DFlashRunner(model, tokenizer, d, c, **runner_kwargs)
        _, stats = r.generate(prompt, max_tokens=max_tokens, use_cache=True)
        results.append(stats)

    results.sort(key=lambda s: s["tok_per_sec"])
    return results[len(results) // 2]


drafter_path_global = None


def main():
    global drafter_path_global

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    parser.add_argument("--drafter", default="z-lab/Qwen3.6-35B-A3B-DFlash")
    parser.add_argument("--prompt", default="def binary_search(arr, target):\n    ")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--priority",
        choices=["auto", "quality", "speed", "balanced"],
        default="auto",
        help="Profile priority (default: auto)",
    )
    args = parser.parse_args()

    print("Loading models...")
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    model, tokenizer = load(args.target)
    drafter_path_global = snapshot_download(args.drafter)
    drafter, config = DFlashDraftModel.from_pretrained(drafter_path_global)
    print("  Done.\n")

    # === Step 1: Detect model ===
    print("=" * 90)
    print("MODEL PROFILE")
    print("=" * 90)
    profile = detect_model(model, tokenizer)
    print(f"  Category:       {profile.category}")
    print(f"  Total params:   {profile.total_params_b:.1f}B")
    print(f"  Active params:  {profile.active_params_b:.1f}B")
    print(
        f"  Layers:         {profile.num_layers} total "
        f"({profile.num_ssm_layers} SSM + {profile.num_attn_layers} attention)"
    )
    print(f"  MoE:            {profile.is_moe}")
    print(
        f"  Quantized:      {profile.is_quantized} ({profile.quant_bits}-bit)"
        if profile.is_quantized
        else f"  Quantized:      {profile.is_quantized}"
    )
    print(f"  SSM ratio:      {profile.ssm_ratio:.0%}")
    print()

    # === Step 2: Measure AR baseline ===
    print("Measuring AR baseline...")
    ar_tok_s = measure_ar_baseline(model, tokenizer, args.prompt, args.max_tokens)
    profile.ar_tok_s = round(ar_tok_s, 1)
    print(f"  AR baseline:    {profile.ar_tok_s} tok/s")
    print()

    # === Step 3: Select recommended profile ===
    recommended = select_profile(profile, args.priority)
    print(f"  Priority:       {args.priority}")
    print(f"  Recommended:    {recommended.name}")
    print(f"  DFlash value:   {profile.recommendation}")
    print(f"  Rationale:      {recommended.description}")
    print()

    # === Step 4: Test all DFlash configurations ===
    print("=" * 90)
    print("CONFIGURATION MATRIX")
    print("=" * 90)
    print(f"  Prompt: {args.prompt[:50]}...")
    print(f"  Max tokens: {args.max_tokens}, Trials: {args.trials}")
    print()

    # Warmup
    r0 = DFlashRunner(model, tokenizer, drafter, config)
    r0.generate(args.prompt, max_tokens=4, use_cache=True)

    configs = [
        ("AR baseline (no DFlash)", None),
        ("DFlash bf16 bs=16", dict(quantize_drafter=None, inference_block_size=None)),
        ("DFlash 8-bit bs=16", dict(quantize_drafter=8, inference_block_size=None)),
        ("DFlash 4-bit bs=16", dict(quantize_drafter=4, inference_block_size=None)),
        ("DFlash 8-bit bs=8", dict(quantize_drafter=8, inference_block_size=8)),
        ("DFlash 8-bit bs=4", dict(quantize_drafter=8, inference_block_size=4)),
        ("DFlash 4-bit bs=4", dict(quantize_drafter=4, inference_block_size=4)),
    ]

    print(
        f"  {'Config':28s} | {'tok/s':>8s} | {'accept':>7s} | {'tok/step':>8s} | "
        f"{'draft':>7s} | {'verify':>8s} | {'vs AR':>7s} | verdict"
    )
    print("-" * 105)

    for label, kwargs in configs:
        if kwargs is None:
            # AR baseline
            from mlx_lm import stream_generate

            ar_results = []
            for _ in range(args.trials):
                last_resp = None
                for resp in stream_generate(model, tokenizer, args.prompt, max_tokens=args.max_tokens):
                    last_resp = resp
                ar_results.append(last_resp.generation_tps if last_resp else 0)
            ar_results.sort()
            ar_median = ar_results[len(ar_results) // 2]
            print(
                f"  {'AR baseline':28s} | {ar_median:8.1f} | {'N/A':>7s} | {'1.0':>8s} | "
                f"{'—':>7s} | {'—':>8s} | {'1.00x':>7s} | reference"
            )
            ar_base = ar_median
            continue

        s = run_dflash_trial(
            model, tokenizer, drafter, config, args.prompt, args.max_tokens, trials=args.trials, **kwargs
        )

        ratio = s["tok_per_sec"] / ar_base if ar_base > 0 else 0
        if ratio >= 0.95:
            verdict = "GOOD" if ratio >= 1.0 else "OK"
        elif ratio >= 0.7:
            verdict = "marginal"
        else:
            verdict = "skip"

        marker = (
            " ◀ recommended"
            if label.startswith("DFlash")
            and (
                (recommended.quantize_drafter == kwargs.get("quantize_drafter"))
                and (recommended.inference_block_size == kwargs.get("inference_block_size"))
            )
            else ""
        )

        print(
            f"  {label:28s} | {s['tok_per_sec']:8.1f} | "
            f"{s['acceptance_rate']:7.1%} | {s['tokens_per_step']:8.1f} | "
            f"{s['avg_draft_ms']:7.1f} | {s['avg_verify_ms']:8.1f} | "
            f"{ratio:6.2f}x | {verdict}{marker}"
        )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  Model:          {args.target}")
    print(f"  Category:       {profile.category}")
    print(f"  AR baseline:    {profile.ar_tok_s} tok/s")
    print(f"  Recommended:    {recommended.name} — {recommended.description}")
    print(f"  DFlash value:   {profile.recommendation}")
    print("=" * 90)


if __name__ == "__main__":
    main()
