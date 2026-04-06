#!/usr/bin/env python3
"""Benchmark MLX-Flash optimization layers separately.

Measures the contribution of each optimization:
  1. Baseline MLX (no optimizations)
  2. + Memory management (pressure-aware, auto-release)
  3. + Expert caching (LCP cache, prefetch)
  4. + SSD streaming (for models > RAM)

Usage:
  python scripts/bench-optimization-layers.py
  python scripts/bench-optimization-layers.py --model mlx-community/Qwen3-30B-A3B-4bit
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. Install with: pip install mlx mlx-lm")
    exit(1)

from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.memory_manager import MemoryManager, get_memory_state


@dataclass
class LayerResult:
    layer: str
    tok_per_s: float
    memory_available_gb: float
    pressure: str
    cache_hit_rate: float
    notes: str


def bench_generate(model, tokenizer, prompt, max_tokens, runs=3):
    """Average tok/s over multiple runs."""
    tps_list = []
    for _ in range(runs):
        t0 = time.monotonic()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        mx.synchronize()
        elapsed = time.monotonic() - t0
        toks = len(tokenizer.encode(out))
        tps_list.append(toks / elapsed if elapsed > 0 else 0)
    return sum(tps_list) / len(tps_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    hw = detect_hardware()
    mem = get_memory_state()
    print(f"Hardware: {hw.chip}, {hw.total_ram_gb:.0f}GB RAM")
    print(f"Available: {mem.available_gb:.1f}GB, pressure: {mem.pressure_level}")
    print(f"Model: {args.model}")
    print(f"Config: {args.max_tokens} tokens x {args.runs} runs per layer")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    prompt = "Explain the advantages of mixture-of-experts architectures. Cover parameter efficiency, computation cost, scaling, and memory trade-offs in detail."
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        pass

    # Warmup
    _ = generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)
    mx.synchronize()

    results = []

    # --- Layer 1: Baseline MLX ---
    print("=== Layer 1: Baseline MLX (no optimizations) ===")
    mem_before = get_memory_state()
    tps = bench_generate(model, tokenizer, prompt, args.max_tokens, args.runs)
    mem_after = get_memory_state()
    results.append(LayerResult(
        layer="1-baseline",
        tok_per_s=round(tps, 1),
        memory_available_gb=round(mem_after.available_gb, 1),
        pressure=mem_after.pressure_level,
        cache_hit_rate=0.0,
        notes="Standard mlx_lm.generate, no MLX-Flash",
    ))
    print(f"  {tps:.1f} tok/s | {mem_after.available_gb:.1f}GB available | {mem_after.pressure_level}")

    # --- Layer 2: + Memory management ---
    print("\n=== Layer 2: + Memory management (pressure-aware) ===")
    mgr = MemoryManager(safety_margin_gb=2.0)
    mem_state = get_memory_state()
    if mem_state.pressure_level in ("warning", "critical"):
        release_info = mgr.auto_release_if_needed()
        print(f"  Auto-released: {release_info}")
        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:
            pass

    hints = mgr.get_optimization_hints()
    tps = bench_generate(model, tokenizer, prompt, args.max_tokens, args.runs)
    mem_after = get_memory_state()
    results.append(LayerResult(
        layer="2-mem-managed",
        tok_per_s=round(tps, 1),
        memory_available_gb=round(mem_after.available_gb, 1),
        pressure=mem_after.pressure_level,
        cache_hit_rate=0.0,
        notes=f"Memory management active, {len(hints)} hints",
    ))
    print(f"  {tps:.1f} tok/s | {mem_after.available_gb:.1f}GB available | {mem_after.pressure_level}")
    if hints:
        for h in hints[:3]:
            print(f"  Hint: {h}")

    # --- Layer 3: + Expert caching ---
    print("\n=== Layer 3: + Expert caching (LCP + prefetch) ===")
    cache_hit = 0.0
    try:
        from mlx_flash_compress.expert_streaming import enable_expert_streaming
        streaming = enable_expert_streaming(model, capacity_per_layer=64)
        streaming.warmup()
        tps = bench_generate(model, tokenizer, prompt, args.max_tokens, args.runs)
        try:
            cache_hit = streaming.cache_hit_rate()
        except Exception:
            cache_hit = 0.0
        print(f"  Expert streaming enabled, cache hit: {cache_hit:.0%}")
    except Exception as e:
        print(f"  Expert streaming not applicable: {e}")
        print("  (Model fits entirely in RAM — caching not needed)")
        tps = results[-1].tok_per_s  # Same as previous

    mem_after = get_memory_state()
    results.append(LayerResult(
        layer="3-expert-cache",
        tok_per_s=round(tps, 1),
        memory_available_gb=round(mem_after.available_gb, 1),
        pressure=mem_after.pressure_level,
        cache_hit_rate=round(cache_hit, 2),
        notes="Expert caching + prefetch (helps when model > RAM)",
    ))
    print(f"  {tps:.1f} tok/s | {mem_after.available_gb:.1f}GB available | {mem_after.pressure_level}")

    # --- Layer 4: + SSD streaming ---
    print("\n=== Layer 4: + SSD streaming (for models > RAM) ===")
    try:
        from mlx_flash_compress.expert_streaming import enable_skip_fallback
        enable_skip_fallback(model, streaming.caches if 'streaming' in dir() else {},
                            adaptive_skip_threshold=3.0)
        tps = bench_generate(model, tokenizer, prompt, args.max_tokens, args.runs)
        print(f"  SSD streaming + skip fallback enabled")
    except Exception as e:
        print(f"  SSD streaming not applicable: {e}")
        print("  (Only activates when experts must be paged from SSD)")
        tps = results[-1].tok_per_s

    mem_after = get_memory_state()
    results.append(LayerResult(
        layer="4-ssd-streaming",
        tok_per_s=round(tps, 1),
        memory_available_gb=round(mem_after.available_gb, 1),
        pressure=mem_after.pressure_level,
        cache_hit_rate=round(cache_hit, 2),
        notes="SSD streaming for experts that don't fit in RAM",
    ))
    print(f"  {tps:.1f} tok/s | {mem_after.available_gb:.1f}GB available | {mem_after.pressure_level}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print(f"  Optimization Layer Benchmark — {hw.chip}, {hw.total_ram_gb:.0f}GB")
    print("=" * 65)
    print(f"  {'Layer':<25} {'tok/s':>8} {'RAM avail':>10} {'Pressure':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10}")
    baseline = results[0].tok_per_s
    for r in results:
        speedup = r.tok_per_s / baseline if baseline > 0 else 0
        suffix = f" ({speedup:.1f}x)" if r.layer != "1-baseline" else ""
        print(f"  {r.layer:<25} {r.tok_per_s:>8.1f} {r.memory_available_gb:>9.1f}G {r.pressure:>10}{suffix}")
    print("=" * 65)

    if results[-1].tok_per_s > baseline * 1.05:
        print(f"\n  Total speedup: {results[-1].tok_per_s / baseline:.1f}x over baseline")
    else:
        print(f"\n  Model fits in RAM — optimizations mainly add memory safety.")
        print(f"  For speedup, test with a model that exceeds your RAM (30B+ on 36GB).")

    if args.save:
        with open(args.save, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n  Saved to {args.save}")


if __name__ == "__main__":
    main()
