#!/usr/bin/env python3
"""DFlash Proof-of-Concept: real speculative decoding on Apple Silicon.

Downloads z-lab/Qwen3.6-35B-A3B-DFlash drafter (948 MB) and pairs it with
the Qwen3.6-35B-A3B target model to demonstrate DFlash block diffusion
speculative decoding on MLX.

Requirements:
  - Apple Silicon Mac with >= 24 GB RAM (target ~18 GB + drafter ~1 GB)
  - pip install mlx mlx-lm huggingface_hub

Usage:
  # Full PoC (downloads ~19 GB on first run)
  python scripts/run_dflash_poc.py

  # With a specific target model
  python scripts/run_dflash_poc.py --target mlx-community/Qwen3-30B-A3B-4bit

  # Skip download (use cached models)
  python scripts/run_dflash_poc.py --skip-download

  # Custom prompts
  python scripts/run_dflash_poc.py --prompt "def fibonacci(n):"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def download_drafter(drafter_repo: str, cache_dir: str) -> str:
    """Download z-lab DFlash drafter from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading drafter: {drafter_repo}")
    path = snapshot_download(
        drafter_repo,
        cache_dir=cache_dir,
        allow_patterns=["config.json", "*.safetensors", "dflash.py"],
    )
    print(f"  Drafter cached at: {path}")
    return path


def find_target_model(target_repo: str) -> str:
    """Find or download the target model."""
    from huggingface_hub import snapshot_download

    print(f"Ensuring target model: {target_repo}")
    path = snapshot_download(
        target_repo,
        allow_patterns=["config.json", "*.safetensors", "tokenizer*", "*.json", "*.txt"],
    )
    print(f"  Target cached at: {path}")
    return path


def run_baseline(model, tokenizer, prompts: list[str], max_tokens: int = 64) -> dict:
    """Run standard autoregressive generation for baseline comparison."""
    from mlx_lm import generate

    print("\n--- Baseline (autoregressive) ---")
    results = []

    for prompt in prompts:
        t0 = time.perf_counter()
        output = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        n_tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
        tok_s = n_tokens / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "tokens": n_tokens,
            "time_s": round(elapsed, 2),
            "tok_s": round(tok_s, 1),
        })
        print(f"  {results[-1]['prompt']}: {tok_s:.1f} tok/s ({n_tokens} tokens in {elapsed:.2f}s)")

    avg_tok_s = sum(r["tok_s"] for r in results) / len(results) if results else 0
    return {"results": results, "avg_tok_s": round(avg_tok_s, 1)}


def run_dflash(runner, prompts: list[str], max_tokens: int = 64) -> dict:
    """Run DFlash speculative decoding."""
    print("\n--- DFlash (block diffusion speculative) ---")
    results = []

    for prompt in prompts:
        text, stats = runner.generate(prompt, max_tokens=max_tokens)
        results.append({
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "stats": stats,
            "output_preview": text[:100] + "..." if len(text) > 100 else text,
        })
        print(f"  {results[-1]['prompt']}:")
        print(f"    {stats['tok_per_sec']:.1f} tok/s | "
              f"accepted {stats['acceptance_rate']:.1%} | "
              f"{stats['tokens_per_step']:.1f} tok/step | "
              f"{stats['tokens_generated']} tokens in {stats['wall_time_s']:.2f}s")

    avg_tok_s = sum(r["stats"]["tok_per_sec"] for r in results) / len(results) if results else 0
    avg_accept = sum(r["stats"]["acceptance_rate"] for r in results) / len(results) if results else 0
    return {
        "results": results,
        "avg_tok_s": round(avg_tok_s, 1),
        "avg_acceptance": round(avg_accept, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="DFlash PoC on Apple Silicon")
    parser.add_argument("--target", type=str, default="mlx-community/Qwen3-30B-A3B-4bit",
                        help="Target model (HuggingFace repo or local path)")
    parser.add_argument("--drafter", type=str, default="z-lab/Qwen3.6-35B-A3B-DFlash",
                        help="DFlash drafter model")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (otherwise uses built-in test prompts)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        print("Error: pip install mlx mlx-lm huggingface_hub")
        sys.exit(1)

    # Test prompts covering different content types
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "def binary_search(arr, target):\n    ",
            "The transformer architecture consists of an encoder and decoder, where each layer",
            "To solve the equation 3x + 7 = 22, first subtract 7 from both sides:",
        ]

    print("=" * 60)
    print("DFlash Proof-of-Concept on Apple Silicon")
    print("=" * 60)
    print(f"Target: {args.target}")
    print(f"Drafter: {args.drafter}")
    print(f"Max tokens: {args.max_tokens}")

    # Step 1: Download drafter
    if not args.skip_download:
        drafter_path = download_drafter(args.drafter, args.cache_dir)
    else:
        from huggingface_hub import scan_cache_dir
        drafter_path = args.drafter

    # Step 2: Load target model
    print(f"\nLoading target model: {args.target}")
    t0 = time.perf_counter()
    target_model, tokenizer = load(args.target)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Step 3: Load drafter
    print(f"\nLoading DFlash drafter: {args.drafter}")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner

    drafter_path_resolved = drafter_path
    if not Path(drafter_path).exists():
        from huggingface_hub import snapshot_download
        drafter_path_resolved = snapshot_download(args.drafter, args.cache_dir)

    t0 = time.perf_counter()
    drafter, drafter_config = DFlashDraftModel.from_pretrained(drafter_path_resolved)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  Config: {drafter_config.num_hidden_layers} layers, "
          f"hidden={drafter_config.hidden_size}, "
          f"block_size={drafter_config.block_size}, "
          f"target_layers={drafter_config.target_layer_ids}")

    # Step 4: Check target model compatibility
    target_num_layers = 0
    if hasattr(target_model, "model") and hasattr(target_model.model, "layers"):
        target_num_layers = len(target_model.model.layers)
    elif hasattr(target_model, "layers"):
        target_num_layers = len(target_model.layers)

    print(f"\n  Target model layers: {target_num_layers}")
    print(f"  Drafter expects target layers: {drafter_config.num_target_layers}")

    max_checkpoint = max(drafter_config.target_layer_ids)
    if max_checkpoint >= target_num_layers:
        print(f"\n  WARNING: Drafter expects checkpoint layer {max_checkpoint} but target has {target_num_layers} layers.")
        print(f"  The drafter was trained for a {drafter_config.num_target_layers}-layer target.")
        print(f"  Results may be poor due to architecture mismatch.")

    # Step 5: Run baseline
    baseline_stats = None
    if not args.skip_baseline:
        baseline_stats = run_baseline(target_model, tokenizer, prompts, args.max_tokens)

    # Step 6: Run DFlash
    runner = DFlashRunner(target_model, tokenizer, drafter, drafter_config)
    dflash_stats = run_dflash(runner, prompts, args.max_tokens)

    # Step 7: Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if baseline_stats:
        print(f"  Baseline (AR):  {baseline_stats['avg_tok_s']:.1f} tok/s")
    print(f"  DFlash:         {dflash_stats['avg_tok_s']:.1f} tok/s")
    print(f"  Acceptance:     {dflash_stats['avg_acceptance']:.1%}")

    if baseline_stats and baseline_stats["avg_tok_s"] > 0:
        speedup = dflash_stats["avg_tok_s"] / baseline_stats["avg_tok_s"]
        print(f"  Speedup:        {speedup:.2f}x")

    # Save results
    results_path = Path("dflash_poc_results.json")
    results = {
        "target": args.target,
        "drafter": args.drafter,
        "max_tokens": args.max_tokens,
        "baseline": baseline_stats,
        "dflash": dflash_stats,
    }
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
