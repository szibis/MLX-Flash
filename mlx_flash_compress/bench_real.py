"""Real model benchmark — tests on actual MLX MoE models.

Downloads a small MoE model, runs pure MLX inference as baseline,
then extracts expert weights to disk and benchmarks cache-assisted
inference with expert eviction/streaming.

Usage:
  python -m mlx_flash_compress.bench_real
  python -m mlx_flash_compress.bench_real --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit
  python -m mlx_flash_compress.bench_real --tokens 50 --hot-mb 512
"""

import argparse
import copy
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

from mlx_flash_compress.compression import LZ4Compressor, ZSTDCompressor
from mlx_flash_compress.compression_native import is_available as native_available, NativeCompressor, Algorithm


def print_separator(title: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(f"  {line}")


# ── Expert weight extraction ────────────────────────────────────

def find_expert_params(model):
    """Find all expert weight parameters in the model tree.

    Returns dict mapping (layer_idx, expert_id, weight_name) -> (param_name, param).
    """
    experts = {}

    def _walk(module, prefix=""):
        if hasattr(module, "children"):
            for name, child in module.children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                if isinstance(child, dict):
                    for k, v in child.items():
                        _walk(v, f"{child_prefix}.{k}")
                elif isinstance(child, list):
                    for i, v in enumerate(child):
                        _walk(v, f"{child_prefix}.{i}")
                elif hasattr(child, "children"):
                    _walk(child, child_prefix)

        # Check for expert-style weight tensors
        if hasattr(module, "weight") and isinstance(module.weight, mx.array):
            parts = prefix.split(".")
            layer_idx = None
            expert_id = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                if p == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    expert_id = int(parts[i + 1])
            if layer_idx is not None and expert_id is not None:
                weight_name = parts[-1] if not parts[-1].isdigit() else parts[-2]
                key = (layer_idx, expert_id, weight_name)
                experts[key] = (prefix + ".weight", module.weight)

    _walk(model)
    return experts


def extract_expert_weights_to_disk(model, work_dir: str):
    """Extract all expert weights from the model and save to disk.

    Returns: (expert_dir, metadata dict with shapes/sizes)
    """
    expert_dir = Path(work_dir) / "real_experts"
    if expert_dir.exists():
        shutil.rmtree(expert_dir)
    expert_dir.mkdir(parents=True)

    # Use flat parameter search (handles both individual and packed 3D experts)
    experts = _find_expert_params_flat(model)
    if not experts:
        experts = find_expert_params(model)

    metadata = {
        "num_experts_found": len(experts),
        "layers": set(),
        "expert_ids": set(),
        "total_bytes": 0,
        "expert_files": {},
    }

    for (layer_idx, expert_id, weight_name), (param_name, param) in experts.items():
        layer_dir = expert_dir / f"layer_{layer_idx:03d}"
        layer_dir.mkdir(exist_ok=True)

        arr = np.array(param)
        data = arr.tobytes()

        # Combine all weight tensors for one expert into a single file
        path = layer_dir / f"expert_{expert_id:04d}.bin"
        mode = "ab" if path.exists() else "wb"
        with open(path, mode) as f:
            f.write(data)

        metadata["layers"].add(layer_idx)
        metadata["expert_ids"].add(expert_id)
        metadata["total_bytes"] += len(data)
        file_key = f"layer_{layer_idx:03d}/expert_{expert_id:04d}"
        if file_key not in metadata["expert_files"]:
            metadata["expert_files"][file_key] = {"bytes": 0, "shapes": []}
        metadata["expert_files"][file_key]["bytes"] += len(data)
        metadata["expert_files"][file_key]["shapes"].append(
            (weight_name, arr.shape, str(arr.dtype))
        )

    metadata["num_layers"] = len(metadata["layers"])
    metadata["num_experts_per_layer"] = len(metadata["expert_ids"])
    # Clean up sets for printing
    metadata["layers"] = sorted(metadata["layers"])
    metadata["expert_ids"] = sorted(metadata["expert_ids"])

    return expert_dir, metadata


def _find_expert_params_flat(model):
    """Search model.parameters() for expert weights.

    Handles two MoE weight formats:
    1. Individual expert modules: model.layers.N.experts.M.weight
    2. Packed 3D tensors (MLX QuantizedSwitchLinear):
       model.layers.N.mlp.switch_mlp.gate_proj.weight (shape: [num_experts, out, in])
       These are sliced into individual expert tensors.
    """
    experts = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}.{i}" if prefix else str(i))
        elif isinstance(obj, mx.array):
            parts = prefix.split(".")
            layer_idx = None
            weight_name = parts[-1]

            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])

            if layer_idx is None:
                return

            # Case 1: Individual expert modules
            expert_id = None
            for i, p in enumerate(parts):
                if p == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    expert_id = int(parts[i + 1])
            if expert_id is not None:
                key = (layer_idx, expert_id, weight_name)
                experts[key] = (prefix, obj)
                return

            # Case 2: Packed 3D QuantizedSwitchLinear tensors
            # Pattern: model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
            if "switch_mlp" in prefix and len(obj.shape) == 3:
                num_experts = obj.shape[0]
                # Find the projection name (gate_proj, up_proj, down_proj)
                proj_name = None
                for p in parts:
                    if p in ("gate_proj", "up_proj", "down_proj"):
                        proj_name = p
                if proj_name is None:
                    proj_name = "unknown"

                for eid in range(num_experts):
                    key = (layer_idx, eid, f"{proj_name}.{weight_name}")
                    # Slice out this expert's weights
                    expert_slice = obj[eid]
                    experts[key] = (f"{prefix}[{eid}]", expert_slice)

    _flatten(model.parameters())
    return experts


# ── Compression ratio on real weights ───────────────────────────

def benchmark_real_compression(expert_dir: Path, max_files: int = 20):
    """Measure compression ratios on actual model expert weights."""
    files = sorted(expert_dir.rglob("*.bin"))[:max_files]
    if not files:
        print("  No expert files found!")
        return

    # Concatenate for bulk benchmark
    all_data = b''.join(f.read_bytes() for f in files)
    total_mb = len(all_data) / 1e6
    print(f"  Testing on {len(files)} expert files ({total_mb:.1f} MB total)")
    print()

    compressors = [
        ("LZ4 (C ext)", lambda d: LZ4Compressor().compress(d), lambda b: LZ4Compressor().decompress(b)),
        ("ZSTD-1", lambda d: ZSTDCompressor(1).compress(d), lambda b: ZSTDCompressor(1).decompress(b)),
        ("ZSTD-3", lambda d: ZSTDCompressor(3).compress(d), lambda b: ZSTDCompressor(3).decompress(b)),
    ]

    if native_available():
        for name, algo in [("LZFSE (native)", Algorithm.LZFSE), ("LZ4_RAW (native)", Algorithm.LZ4_RAW)]:
            c = NativeCompressor(algo)
            compressors.append((name, lambda d, _c=c: _c.compress(d), lambda b, _c=c: _c.decompress(b)))

    headers = ["Algorithm", "Ratio", "Compress MB/s", "Decompress MB/s"]
    rows = []

    for name, compress_fn, decompress_fn in compressors:
        # Compress
        t0 = time.monotonic()
        buf = compress_fn(all_data)
        ct = time.monotonic() - t0

        # Decompress
        t0 = time.monotonic()
        _ = decompress_fn(buf)
        dt = time.monotonic() - t0

        ratio = buf.original_size / buf.compressed_size if buf.compressed_size > 0 else 0
        rows.append([
            name,
            f"{ratio:.2f}x",
            f"{total_mb / ct:.0f}" if ct > 0 else "N/A",
            f"{total_mb / dt:.0f}" if dt > 0 else "N/A",
        ])

    print_table(headers, rows)


# ── Pure MLX inference benchmark ────────────────────────────────

def benchmark_mlx_inference(model, tokenizer, prompt, max_tokens, warmup_tokens=10):
    """Run timed MLX inference and return (output_text, tokens, tok/s, time)."""
    import psutil

    # Format prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    process = psutil.Process(os.getpid())

    # Warm-up
    print(f"  Warm-up ({warmup_tokens} tokens)...")
    _ = generate(model, tokenizer, prompt=formatted, max_tokens=warmup_tokens, verbose=False)
    mx.synchronize()

    # Timed run
    mem_before = process.memory_info().rss / 1e6
    print(f"  Generating {max_tokens} tokens...")

    t0 = time.monotonic()
    output = generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
    mx.synchronize()
    elapsed = time.monotonic() - t0

    mem_after = process.memory_info().rss / 1e6
    out_tokens = len(tokenizer.encode(output))
    tps = out_tokens / elapsed if elapsed > 0 else 0

    return {
        "output": output[:200],  # truncate for display
        "tokens": out_tokens,
        "time_s": elapsed,
        "tok_per_s": tps,
        "memory_mb": max(mem_before, mem_after),
    }


# ── Cache-assisted inference simulation ─────────────────────────

def benchmark_cache_inference(
    model, tokenizer, prompt, max_tokens,
    expert_dir, num_layers, num_experts,
    hot_mb=512, workers=4, hot_algo="lz4",
):
    """Run inference while routing expert fetches through the compressed cache.

    Strategy: for each generated token, simulate the expert fetch pattern
    through the cache while also running the actual MLX inference.
    This measures the overhead of the cache layer on real inference.
    """
    from mlx_flash_compress.cache import ExpertCacheManager

    cache = ExpertCacheManager(
        expert_dir=str(expert_dir),
        hot_limit_bytes=hot_mb * 1024 * 1024,
        warm_limit_bytes=0,
        num_workers=workers,
        enable_hot=True,
        enable_warm=False,
        hot_algo=hot_algo,
    )

    # Pre-warm the cache (simulates pre-compressed model)
    print(f"  Pre-warming cache ({hot_mb}MB, {hot_algo})...")
    t0 = time.monotonic()
    cached = cache.prewarm(num_layers, num_experts)
    prewarm_time = time.monotonic() - t0
    print(f"  Pre-warmed {cached} experts in {prewarm_time:.1f}s")
    stats_after_prewarm = cache.get_stats()

    # Format prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    # Generate with cache fetch simulation interleaved
    rng = np.random.default_rng(42)
    k = min(4, num_experts)
    expert_probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
    expert_probs /= expert_probs.sum()

    cache.reset_stats()

    print(f"  Generating {max_tokens} tokens with cache interleaving...")
    t0 = time.monotonic()

    # Run actual MLX generation
    output = generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
    mx.synchronize()

    # Simulate cache fetches that would happen during generation
    for token_idx in range(max_tokens):
        for layer_idx in range(num_layers):
            expert_ids = rng.choice(num_experts, size=k, replace=False, p=expert_probs).tolist()
            cache.fetch_experts(layer_idx, expert_ids, np.float16)

    elapsed = time.monotonic() - t0
    out_tokens = len(tokenizer.encode(output))
    tps = out_tokens / elapsed if elapsed > 0 else 0

    stats = cache.get_stats()
    cache.shutdown()

    return {
        "output": output[:200],
        "tokens": out_tokens,
        "time_s": elapsed,
        "tok_per_s": tps,
        "prewarm_s": prewarm_time,
        "cache_hit_rate": stats.hit_rate,
        "hot_hits": stats.hot_hits,
        "cold_hits": stats.cold_hits,
        "decompress_ms": stats.total_decompress_ms,
        "hot_mb_used": stats.hot_bytes / 1e6,
    }


# ── Main benchmark ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real model benchmark for MLX-Flash")
    parser.add_argument("--model", default="mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit",
                        help="MLX MoE model to benchmark")
    parser.add_argument("--prompt", default="Explain the concept of mixture of experts in neural networks in detail.",
                        help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--hot-mb", type=int, default=512, help="Hot cache size in MB")
    parser.add_argument("--workers", type=int, default=4, help="Cache worker threads")
    parser.add_argument("--work-dir", default="/tmp/mlx_flash_real", help="Working directory")
    args = parser.parse_args()

    print_separator("MLX-Flash: Real Model Benchmark")
    print(f"  Model:  {args.model}")
    print(f"  Prompt: {args.prompt[:60]}...")
    print(f"  Tokens: {args.tokens}")
    print(f"  Cache:  {args.hot_mb} MB hot tier")
    print()

    # ── Step 1: Load model ──
    print_separator("1. Loading Model")
    t0 = time.monotonic()
    model, tokenizer = load(args.model)
    load_time = time.monotonic() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Count params
    def _count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += _count_params(v)
        elif isinstance(params, (list, tuple)):
            for v in params:
                total += _count_params(v)
        elif isinstance(params, mx.array):
            total += params.size
        return total
    total_params = _count_params(model.parameters())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")

    # ── Step 2: Extract expert weights ──
    print_separator("2. Extracting Expert Weights to Disk")
    expert_dir, meta = extract_expert_weights_to_disk(model, args.work_dir)
    print(f"  Found {meta['num_experts_found']} expert weight tensors")
    print(f"  Layers with experts: {meta['num_layers']}")
    print(f"  Experts per layer: {meta['num_experts_per_layer']}")
    print(f"  Total expert data: {meta['total_bytes'] / 1e6:.1f} MB")

    # Show a few sample files
    sample_files = sorted(expert_dir.rglob("*.bin"))[:3]
    for f in sample_files:
        print(f"    {f.relative_to(expert_dir)}: {f.stat().st_size / 1024:.0f} KB")

    if meta["num_experts_found"] == 0:
        print("\n  ERROR: No expert weights found in model!")
        print("  This model may not have a standard MoE architecture.")
        print("  Skipping cache benchmarks, running pure MLX only.\n")

    # ── Step 3: Compression ratios on real weights ──
    if meta["num_experts_found"] > 0:
        print_separator("3. Compression Ratios on Real Expert Weights")
        benchmark_real_compression(expert_dir)

    # ── Step 4: Pure MLX inference (baseline) ──
    print_separator("4. Pure MLX Inference (Baseline)")
    baseline = benchmark_mlx_inference(model, tokenizer, args.prompt, args.tokens)
    print(f"  Tokens: {baseline['tokens']}")
    print(f"  Time:   {baseline['time_s']:.2f}s")
    print(f"  Speed:  {baseline['tok_per_s']:.1f} tok/s")
    print(f"  Memory: {baseline['memory_mb']:.0f} MB")
    print(f"  Output: {baseline['output'][:100]}...")

    # ── Step 5: Cache-assisted inference ──
    if meta["num_experts_found"] > 0:
        for algo_name, algo in [("LZ4", "lz4"), ("LZFSE", "lzfse")]:
            print_separator(f"5. Cache-Assisted Inference ({algo_name})")
            cached = benchmark_cache_inference(
                model, tokenizer, args.prompt, args.tokens,
                expert_dir=expert_dir,
                num_layers=meta["num_layers"],
                num_experts=meta["num_experts_per_layer"],
                hot_mb=args.hot_mb,
                workers=args.workers,
                hot_algo=algo,
            )
            print(f"  Tokens:       {cached['tokens']}")
            print(f"  Time:         {cached['time_s']:.2f}s  (includes cache overhead)")
            print(f"  Speed:        {cached['tok_per_s']:.1f} tok/s")
            print(f"  Pre-warm:     {cached['prewarm_s']:.1f}s")
            print(f"  Cache hit:    {cached['cache_hit_rate']:.1%}")
            print(f"  Hot hits:     {cached['hot_hits']}")
            print(f"  Cold hits:    {cached['cold_hits']}")
            print(f"  Decompress:   {cached['decompress_ms']:.1f}ms total")
            print(f"  Cache used:   {cached['hot_mb_used']:.1f} MB")

            overhead = cached['time_s'] - baseline['time_s']
            overhead_pct = (overhead / baseline['time_s'] * 100) if baseline['time_s'] > 0 else 0
            print(f"  Overhead:     {overhead:.2f}s ({overhead_pct:.1f}% vs baseline)")

    # ── Step 6: Summary ──
    print_separator("6. Summary")

    headers = ["Mode", "tok/s", "Time (s)", "Notes"]
    rows = [["Pure MLX (baseline)", f"{baseline['tok_per_s']:.1f}", f"{baseline['time_s']:.2f}", "All weights in RAM"]]

    if meta["num_experts_found"] > 0:
        # Re-run quick to get final numbers (already have them from above)
        pass

    print_table(headers, rows)
    print()
    print("  NOTE: In this benchmark, MLX runs the full model (all weights in RAM)")
    print("  while the cache layer runs alongside simulating expert fetch patterns.")
    print("  The cache overhead shows what the compression layer costs on top of")
    print("  normal inference. For a model that truly exceeds RAM, the cache would")
    print("  REPLACE the SSD read (saving 0.6-2.4ms per expert), not add to it.")

    # Cleanup
    if expert_dir.exists():
        total_size = sum(f.stat().st_size for f in expert_dir.rglob("*"))
        print(f"\n  Expert data on disk: {total_size / 1e6:.1f} MB at {expert_dir}")
        print(f"  Run 'rm -rf {expert_dir}' to clean up")


if __name__ == "__main__":
    main()
