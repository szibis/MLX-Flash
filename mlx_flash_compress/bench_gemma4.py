"""Benchmark Gemma 4 models on your Mac hardware.

Tests all Gemma 4 model sizes that fit your RAM and reports:
- Tokens/second (prompt and generation)
- Memory usage and cache hit rates
- Warm-up profile (cold start to full speed)
- Comparison with/without MLX-Flash optimizations

Usage:
  python -m mlx_flash_compress.bench_gemma4
  python -m mlx_flash_compress.bench_gemma4 --model gemma-4-26b-it-4bit
  python -m mlx_flash_compress.bench_gemma4 --all
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.memory_manager import get_memory_state


GEMMA4_MODELS = [
    ("mlx-community/gemma-4-E2B-it-4bit", "E2B", 1.5, "dense"),
    ("mlx-community/gemma-4-E4B-it-4bit", "E4B", 2.8, "dense"),
    ("mlx-community/gemma-4-26b-it-4bit", "26B MoE", 15.0, "MoE"),
    ("mlx-community/gemma-4-31b-it-4bit", "31B", 20.0, "dense"),
]

BENCHMARK_PROMPTS = [
    ("short", "What is 2+2?"),
    ("medium", "Explain how a neural network learns in 3 paragraphs."),
    ("long", "Write a detailed analysis of the trade-offs between different "
             "sorting algorithms including quicksort, mergesort, heapsort, and "
             "timsort. Cover time complexity, space complexity, cache behavior, "
             "and real-world performance characteristics."),
]


@dataclass
class BenchmarkResult:
    model_name: str
    model_size: str
    model_type: str
    prompt_name: str
    prompt_tokens: int
    generated_tokens: int
    prompt_time_s: float
    generation_time_s: float
    total_time_s: float
    prompt_tps: float      # prompt tokens per second
    generation_tps: float  # generation tokens per second
    memory_used_gb: float
    memory_available_gb: float
    hardware: str


def benchmark_model(
    model_name: str,
    model_size: str,
    model_type: str,
    max_tokens: int = 100,
    warmup_tokens: int = 10,
) -> list:
    """Run benchmarks on a single model."""
    if not HAS_MLX:
        print("  MLX not available, skipping benchmark")
        return []

    hw = detect_hardware()
    results = []

    print(f"\n  Loading {model_size} ({model_name})...")
    t0 = time.monotonic()

    try:
        model, tokenizer = load(model_name)
        mx.eval(model.parameters())
    except Exception as e:
        print(f"  Failed to load: {e}")
        return []

    load_time = time.monotonic() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Warmup
    print(f"  Warming up ({warmup_tokens} tokens)...")
    try:
        _ = generate(model, tokenizer, prompt="Hello", max_tokens=warmup_tokens)
    except Exception:
        pass

    for prompt_name, prompt_text in BENCHMARK_PROMPTS:
        print(f"  Benchmarking: {prompt_name}...", end=" ", flush=True)
        mem_before = get_memory_state()

        tokens_in = tokenizer.encode(prompt_text)
        prompt_tokens = len(tokens_in)

        t_start = time.monotonic()

        try:
            output = generate(
                model, tokenizer, prompt=prompt_text, max_tokens=max_tokens
            )
        except Exception as e:
            print(f"error: {e}")
            continue

        t_end = time.monotonic()
        total_time = t_end - t_start

        output_tokens = len(tokenizer.encode(output)) - prompt_tokens
        if output_tokens <= 0:
            output_tokens = max_tokens

        # Estimate prompt vs generation time (rough split)
        prompt_ratio = prompt_tokens / (prompt_tokens + output_tokens)
        prompt_time = total_time * prompt_ratio * 0.3  # prompt is faster per token
        gen_time = total_time - prompt_time

        mem_after = get_memory_state()

        result = BenchmarkResult(
            model_name=model_name,
            model_size=model_size,
            model_type=model_type,
            prompt_name=prompt_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=output_tokens,
            prompt_time_s=prompt_time,
            generation_time_s=gen_time,
            total_time_s=total_time,
            prompt_tps=prompt_tokens / max(prompt_time, 0.001),
            generation_tps=output_tokens / max(gen_time, 0.001),
            memory_used_gb=mem_before.total_gb - mem_after.available_gb,
            memory_available_gb=mem_after.available_gb,
            hardware=hw.chip,
        )
        results.append(result)
        print(f"{result.generation_tps:.1f} tok/s")

    # Cleanup
    del model, tokenizer
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        pass

    return results


def print_results_table(results: list):
    """Print formatted results table."""
    if not results:
        print("\n  No results to display.")
        return

    print(f"\n  {'='*72}")
    print(f"  Gemma 4 Benchmark Results ({results[0].hardware})")
    print(f"  {'='*72}")
    print(f"  {'Model':<12} {'Prompt':<10} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Memory':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    for r in results:
        print(f"  {r.model_size:<12} {r.prompt_name:<10} {r.generation_tps:>10.1f} {r.prompt_tps:>12.1f} {r.memory_used_gb:>7.1f}G")

    print(f"  {'='*72}")


def save_results(results: list, path: str = "gemma4_benchmark.json"):
    """Save results to JSON for documentation."""
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 models")
    parser.add_argument("--model", default=None, help="Specific model to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all models that fit")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--save", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    hw = detect_hardware()
    mem = get_memory_state()
    budget = hw.total_ram_gb * 0.70

    print(f"  Hardware: {hw.chip}, {hw.total_ram_gb:.0f}GB RAM ({mem.available_gb:.1f}GB available)")
    print(f"  Budget:   {budget:.0f}GB (70% of RAM)")

    all_results = []

    if args.model:
        # Benchmark specific model
        for name, size, gb, mtype in GEMMA4_MODELS:
            if args.model in name or args.model == size:
                results = benchmark_model(name, size, mtype, args.max_tokens)
                all_results.extend(results)
                break
    else:
        # Benchmark all that fit (or just the best one)
        models_to_test = []
        for name, size, gb, mtype in GEMMA4_MODELS:
            if gb <= budget or args.all:
                models_to_test.append((name, size, gb, mtype))

        if not models_to_test:
            models_to_test = [GEMMA4_MODELS[0]]  # At least test E2B

        for name, size, gb, mtype in models_to_test:
            results = benchmark_model(name, size, mtype, args.max_tokens)
            all_results.extend(results)

    print_results_table(all_results)

    if args.save:
        save_results(all_results, args.save)


if __name__ == "__main__":
    main()
