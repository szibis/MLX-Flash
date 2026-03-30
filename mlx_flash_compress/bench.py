"""Benchmark harness — runs all inference modes and compares performance.

Supports two benchmark types:
  1. Synthetic: Creates fake expert weights to benchmark cache subsystem
     in isolation (no model download required, fast iteration).
  2. Model: Uses a real MLX MoE model (requires model download).

Usage:
  # Synthetic benchmark (fast, no model needed)
  python -m mlx_flash_compress.bench --synthetic

  # Real model benchmark
  python -m mlx_flash_compress.bench --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit

  # All variants with custom cache sizes
  python -m mlx_flash_compress.bench --synthetic --hot-mb 512 --warm-mb 256 --tokens 50
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mlx_flash_compress.cache import ExpertCacheManager, CacheStats, CacheTier
from mlx_flash_compress.compression import (
    CompressionAlgo,
    LZ4Compressor,
    ZSTDCompressor,
)
from mlx_flash_compress.engine import InferenceMode, InferenceResult


# ─── Synthetic expert generation ────────────────────────────────

def create_synthetic_experts(
    work_dir: str,
    num_layers: int = 8,
    num_experts: int = 64,
    expert_size_bytes: int = 256 * 1024,  # 256KB per expert (small for testing)
    dtype: np.dtype = np.float16,
    quantized: bool = True,
) -> Path:
    """Create synthetic expert weight files on disk.

    When quantized=True (default), generates data that mimics 4-bit quantized
    MoE weights: packed nibbles in uint8 with periodic scale/bias float16 groups.
    This is ~1.6-2.2x compressible, matching real quantized weight patterns.

    When quantized=False, generates random float16 (incompressible baseline).
    """
    expert_dir = Path(work_dir) / "experts"
    if expert_dir.exists():
        shutil.rmtree(expert_dir)

    expert_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)

    total_bytes = 0
    for layer_idx in range(num_layers):
        layer_dir = expert_dir / f"layer_{layer_idx:03d}"
        layer_dir.mkdir()

        for expert_id in range(num_experts):
            if quantized:
                data = _generate_quantized_expert(rng, expert_size_bytes)
            else:
                num_elements = expert_size_bytes // np.dtype(dtype).itemsize
                data = rng.normal(0, 0.02, size=num_elements).astype(dtype).tobytes()

            path = layer_dir / f"expert_{expert_id:04d}.bin"
            path.write_bytes(data)
            total_bytes += len(data)

    mode_str = "4-bit quantized" if quantized else "random float16"
    print(f"  Created {num_layers} layers x {num_experts} experts = {num_layers * num_experts} files")
    print(f"  Data type: {mode_str}")
    print(f"  Total size: {total_bytes / 1e6:.1f} MB ({expert_size_bytes / 1024:.0f} KB per expert)")
    return expert_dir


def _generate_quantized_expert(rng: np.random.Generator, size_bytes: int) -> bytes:
    """Generate data mimicking real 4-bit quantized neural network weights.

    Real quantized weight files (GGUF Q4_K_M, MLX 4-bit) achieve:
      - LZ4:  ~1.5-1.7x  (measured on Qwen, Llama, Mixtral weights)
      - ZSTD: ~2.0-2.5x

    The key to compressibility is BLOCK-LEVEL structure:
      1. Super-blocks of 256 weights share one scale (creates 128-byte patterns)
      2. Many neurons are near-dead (rows of zero-point values = 0x88 runs)
      3. Adjacent rows have correlated weight distributions
      4. The scale+min metadata is highly repetitive

    We simulate this with a block-template approach: generate a small number
    of "archetype" blocks and create variations, producing data with
    realistic LZ4/ZSTD compression characteristics.
    """
    block_size = 256  # Super-block: 128 packed bytes + metadata
    packed_per_block = block_size // 2  # 128 bytes of packed nibbles
    meta_per_block = 4  # 2 bytes scale + 2 bytes min
    bytes_per_block = packed_per_block + meta_per_block
    num_blocks = size_bytes // bytes_per_block
    remainder = size_bytes - (num_blocks * bytes_per_block)

    # Generate ~16 archetype blocks (real models have repeating patterns)
    num_archetypes = 16
    archetypes = []
    for _ in range(num_archetypes):
        # Each archetype has a characteristic distribution
        center = rng.choice([7, 8, 9])  # zero-points vary slightly
        spread = rng.uniform(0.5, 3.0)  # some layers are narrow, some wide
        sparsity = rng.uniform(0.15, 0.60)  # 15-60% sparsity

        raw = rng.laplace(loc=center, scale=spread, size=block_size)
        raw = np.clip(np.round(raw), 0, 15).astype(np.uint8)
        mask = rng.random(block_size) < sparsity
        raw[mask] = center  # zero-point

        packed = np.zeros(packed_per_block, dtype=np.uint8)
        for j in range(packed_per_block):
            packed[j] = (raw[2 * j] << 4) | raw[2 * j + 1]
        archetypes.append(packed)

    # Small pool of scale/min metadata (real models reuse these heavily)
    scale_pool = np.float16(rng.uniform(0.001, 0.03, size=32))
    min_pool = np.float16(rng.uniform(-0.02, 0.0, size=32))

    parts = []
    for b in range(num_blocks):
        # Pick an archetype and apply small noise (creates near-duplicates)
        arch_idx = b % num_archetypes
        block = archetypes[arch_idx].copy()

        # Flip ~5% of nibbles (creates variation within repeating pattern)
        num_flips = max(1, packed_per_block // 20)
        flip_indices = rng.integers(0, packed_per_block, size=num_flips)
        for idx in flip_indices:
            nibble_pos = rng.integers(0, 2)
            new_nibble = rng.integers(0, 16)
            if nibble_pos == 0:
                block[idx] = (new_nibble << 4) | (block[idx] & 0x0F)
            else:
                block[idx] = (block[idx] & 0xF0) | new_nibble

        parts.append(block.tobytes())

        # Metadata from small pool (high repetition = compresses well)
        parts.append(scale_pool[b % len(scale_pool)].tobytes())
        parts.append(min_pool[b % len(min_pool)].tobytes())

    # Dead neuron blocks: ~10% of blocks are all-zero-point (0x88)
    # Shuffle some dead blocks in (creates long compressible runs)
    result = bytearray(b''.join(parts))
    dead_block = bytes([0x88] * packed_per_block) + b'\x00\x00\x00\x00'
    num_dead = num_blocks // 10
    dead_positions = rng.choice(num_blocks, size=num_dead, replace=False)
    for pos in dead_positions:
        offset = pos * bytes_per_block
        result[offset:offset + bytes_per_block] = dead_block

    if remainder > 0:
        result.extend(b'\x00' * remainder)

    return bytes(result)


def purge_os_cache_for_dir(expert_dir: Path):
    """Try to minimize OS page cache effects for fairer SSD benchmarks.

    Uses F_NOCACHE on macOS to advise the OS not to cache file reads.
    This isn't perfect but reduces the OS cache advantage for 'SSD' reads.
    """
    import fcntl
    F_NOCACHE = 48  # macOS fcntl constant

    for path in expert_dir.rglob("*.bin"):
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            finally:
                os.close(fd)
        except (OSError, IOError):
            pass  # Not critical if this fails


# ─── Compression ratio measurement ─────────────────────────────

@dataclass
class CompressionBenchResult:
    algo: str
    original_bytes: int
    compressed_bytes: int
    ratio: float
    compress_speed_mbs: float
    decompress_speed_mbs: float
    compress_time_ms: float
    decompress_time_ms: float


def benchmark_compression_ratios(
    expert_dir: Path,
    num_samples: int = 16,
) -> list[CompressionBenchResult]:
    """Measure compression ratio and speed on actual expert weight data."""
    results = []
    lz4 = LZ4Compressor()
    zstd_low = ZSTDCompressor(level=1)
    zstd_mid = ZSTDCompressor(level=3)
    zstd_high = ZSTDCompressor(level=6)

    # Collect sample expert files
    sample_files = []
    for layer_dir in sorted(expert_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        for expert_file in sorted(layer_dir.iterdir()):
            sample_files.append(expert_file)
            if len(sample_files) >= num_samples:
                break
        if len(sample_files) >= num_samples:
            break

    if not sample_files:
        print("  WARNING: No expert files found for compression benchmark")
        return results

    compressors = [
        ("LZ4", lz4),
        ("ZSTD-1", zstd_low),
        ("ZSTD-3", zstd_mid),
        ("ZSTD-6", zstd_high),
    ]

    for name, compressor in compressors:
        total_original = 0
        total_compressed = 0
        total_compress_time = 0.0
        total_decompress_time = 0.0

        for path in sample_files:
            data = path.read_bytes()
            total_original += len(data)

            # Compress
            t0 = time.monotonic()
            buf = compressor.compress(data)
            total_compress_time += time.monotonic() - t0
            total_compressed += buf.compressed_size

            # Decompress
            t0 = time.monotonic()
            _ = compressor.decompress(buf)
            total_decompress_time += time.monotonic() - t0

        compress_speed = (total_original / 1e6) / total_compress_time if total_compress_time > 0 else 0
        decompress_speed = (total_original / 1e6) / total_decompress_time if total_decompress_time > 0 else 0

        results.append(CompressionBenchResult(
            algo=name,
            original_bytes=total_original,
            compressed_bytes=total_compressed,
            ratio=total_original / total_compressed if total_compressed > 0 else 0,
            compress_speed_mbs=compress_speed,
            decompress_speed_mbs=decompress_speed,
            compress_time_ms=total_compress_time * 1000,
            decompress_time_ms=total_decompress_time * 1000,
        ))

    # Apple native compression via libcompression (LZFSE, LZ4_RAW, native ZSTD)
    from mlx_flash_compress.compression_native import is_available, NativeCompressor, Algorithm

    if is_available():
        # Concatenate all sample data for native benchmark
        all_data = b''.join(path.read_bytes() for path in sample_files)

        native_algos = [
            ("LZFSE (native)", Algorithm.LZFSE),
            ("LZ4_RAW (native)", Algorithm.LZ4_RAW),
            ("LZ4 (native)", Algorithm.LZ4),
        ]

        for name, algo in native_algos:
            try:
                comp = NativeCompressor(algo)
                # Warm up
                nbuf = comp.compress(all_data)
                _ = comp.decompress(nbuf)

                # Benchmark (3 iterations for accuracy)
                compress_times = []
                decompress_times = []
                for _ in range(3):
                    t0 = time.monotonic()
                    nbuf = comp.compress(all_data)
                    compress_times.append(time.monotonic() - t0)

                    t0 = time.monotonic()
                    _ = comp.decompress(nbuf)
                    decompress_times.append(time.monotonic() - t0)

                avg_ct = sum(compress_times) / len(compress_times)
                avg_dt = sum(decompress_times) / len(decompress_times)
                data_mb = len(all_data) / 1e6

                results.append(CompressionBenchResult(
                    algo=name,
                    original_bytes=len(all_data),
                    compressed_bytes=nbuf.compressed_size,
                    ratio=len(all_data) / nbuf.compressed_size if nbuf.compressed_size > 0 else 0,
                    compress_speed_mbs=data_mb / avg_ct if avg_ct > 0 else 0,
                    decompress_speed_mbs=data_mb / avg_dt if avg_dt > 0 else 0,
                    compress_time_ms=avg_ct * 1000,
                    decompress_time_ms=avg_dt * 1000,
                ))
            except (RuntimeError, OSError):
                pass

    return results


# ─── Cache subsystem benchmark ──────────────────────────────────

@dataclass
class CacheBenchResult:
    mode: str
    tokens: int
    layers: int
    experts_per_token: int
    total_time_s: float
    tokens_per_second: float
    avg_layer_ms: float
    cache_hit_rate: float
    hot_hits: int
    warm_hits: int
    cold_hits: int
    hot_mb: float
    warm_mb: float
    total_decompress_ms: float
    total_ssd_read_ms: float
    evictions: int


def benchmark_cache_mode(
    expert_dir: Path,
    mode_name: str,
    num_layers: int,
    num_experts: int,
    num_tokens: int = 50,
    k: int = 4,
    hot_limit_mb: int = 256,
    warm_limit_mb: int = 128,
    num_workers: int = 4,
    enable_hot: bool = True,
    enable_warm: bool = True,
    expert_dtype: np.dtype = np.float16,
    bypass_os_cache: bool = False,
    hot_algo: str = "lz4",
    ssd_latency_ms: float = 0.0,
) -> CacheBenchResult:
    """Run cache benchmark for a single configuration."""

    cache = ExpertCacheManager(
        expert_dir=str(expert_dir),
        hot_limit_bytes=hot_limit_mb * 1024 * 1024 if enable_hot else 0,
        warm_limit_bytes=warm_limit_mb * 1024 * 1024 if enable_warm else 0,
        num_workers=num_workers,
        enable_hot=enable_hot,
        enable_warm=enable_warm,
        bypass_os_cache=bypass_os_cache,
        hot_algo=hot_algo,
        simulated_ssd_latency_ms=ssd_latency_ms,
    )

    # Power-law expert distribution (Zipf-like)
    rng = np.random.default_rng(42)
    expert_probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
    expert_probs /= expert_probs.sum()

    t0 = time.monotonic()

    for token_idx in range(num_tokens):
        for layer_idx in range(num_layers):
            expert_ids = rng.choice(
                num_experts, size=k, replace=False, p=expert_probs
            ).tolist()
            cache.fetch_experts(layer_idx, expert_ids, expert_dtype)

    total_time = time.monotonic() - t0
    stats = cache.get_stats()
    cache.shutdown()

    total_layer_calls = num_tokens * num_layers
    return CacheBenchResult(
        mode=mode_name,
        tokens=num_tokens,
        layers=num_layers,
        experts_per_token=k * num_layers,
        total_time_s=total_time,
        tokens_per_second=num_tokens / total_time if total_time > 0 else 0,
        avg_layer_ms=(total_time * 1000) / total_layer_calls if total_layer_calls > 0 else 0,
        cache_hit_rate=stats.hit_rate,
        hot_hits=stats.hot_hits,
        warm_hits=stats.warm_hits,
        cold_hits=stats.cold_hits,
        hot_mb=stats.hot_bytes / 1e6,
        warm_mb=stats.warm_bytes / 1e6,
        total_decompress_ms=stats.total_decompress_ms,
        total_ssd_read_ms=stats.total_ssd_read_ms,
        evictions=stats.evictions,
    )


# ─── Pure MLX benchmark ────────────────────────────────────────

@dataclass
class MLXBenchResult:
    mode: str
    model: str
    tokens: int
    total_time_s: float
    tokens_per_second: float
    prompt_tps: float
    generation_tps: float
    peak_memory_mb: float


def benchmark_pure_mlx(
    model_name: str,
    prompt: str = "Explain the concept of mixture of experts in neural networks.",
    max_tokens: int = 100,
) -> Optional[MLXBenchResult]:
    """Benchmark pure MLX inference (baseline)."""
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
    except ImportError:
        print("  SKIP: mlx/mlx-lm not installed")
        return None

    import psutil
    process = psutil.Process(os.getpid())

    print(f"  Loading model: {model_name}")
    t_load = time.monotonic()
    model, tokenizer = load(model_name)
    load_time = time.monotonic() - t_load
    print(f"  Model loaded in {load_time:.1f}s")

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

    # Warm-up run
    print("  Warm-up run...")
    _ = generate(model, tokenizer, prompt=formatted, max_tokens=10, verbose=False)

    # Sync MLX
    mx.synchronize()

    # Timed run
    mem_before = process.memory_info().rss / 1e6
    print(f"  Generating {max_tokens} tokens...")
    t0 = time.monotonic()
    output = generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
    mx.synchronize()
    t_done = time.monotonic()
    mem_after = process.memory_info().rss / 1e6

    out_tokens = len(tokenizer.encode(output))
    gen_time = t_done - t0

    # Estimate prompt vs generation time (rough split)
    prompt_tokens = len(tokenizer.encode(formatted))

    return MLXBenchResult(
        mode="pure_mlx",
        model=model_name,
        tokens=out_tokens,
        total_time_s=gen_time,
        tokens_per_second=out_tokens / gen_time if gen_time > 0 else 0,
        prompt_tps=prompt_tokens / 0.5,  # approximate
        generation_tps=out_tokens / gen_time if gen_time > 0 else 0,
        peak_memory_mb=max(mem_before, mem_after),
    )


# ─── Main benchmark runner ──────────────────────────────────────

def print_separator(title: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def print_table(headers: list[str], rows: list[list], align: Optional[list[str]] = None):
    """Simple table printer."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")

    # Rows
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(f"  {line}")


def run_synthetic_benchmarks(args) -> dict:
    """Run all synthetic benchmarks (no model needed)."""
    work_dir = args.work_dir

    results = {"synthetic": {}}

    # Step 1: Create synthetic experts (quantized = compressible)
    print_separator("1. Creating Synthetic Expert Weights")
    expert_dir = create_synthetic_experts(
        work_dir=work_dir,
        num_layers=args.layers,
        num_experts=args.experts,
        expert_size_bytes=args.expert_kb * 1024,
        quantized=True,
    )
    # Purge OS page cache for fairer SSD benchmarks
    print("  Purging OS page cache hints (F_NOCACHE)...")
    purge_os_cache_for_dir(expert_dir)

    # Step 2: Compression ratios
    print_separator("2. Compression Ratio Benchmark")
    comp_results = benchmark_compression_ratios(expert_dir, num_samples=min(32, args.experts))
    results["synthetic"]["compression"] = [asdict(r) for r in comp_results]

    headers = ["Algorithm", "Ratio", "Compress MB/s", "Decompress MB/s", "Compress ms", "Decompress ms"]
    rows = []
    for r in comp_results:
        rows.append([
            r.algo,
            f"{r.ratio:.2f}x",
            f"{r.compress_speed_mbs:.0f}",
            f"{r.decompress_speed_mbs:.0f}",
            f"{r.compress_time_ms:.1f}",
            f"{r.decompress_time_ms:.1f}",
        ])
    print_table(headers, rows)

    # Step 3: Cache subsystem benchmarks
    print_separator("3. Cache Subsystem Benchmark")
    print(f"  Config: {args.layers} layers, {args.experts} experts, K=4, {args.tokens} tokens")
    print(f"  Expert size: {args.expert_kb} KB, Hot cache: {args.hot_mb} MB, Warm cache: {args.warm_mb} MB")
    print()

    # (mode_name, hot_enabled, warm_enabled, hot_algo, description)
    cache_configs = [
        ("no_cache_ssd",     False, False, "lz4",    "No cache (SSD only)"),
        ("lz4_cache",        True,  False, "lz4",    "LZ4 hot cache (Python C ext)"),
        ("lzfse_cache",      True,  False, "lzfse",  "LZFSE hot cache (Apple native)"),
        ("lz4_native_cache", True,  False, "lz4_native", "LZ4 hot cache (Apple native)"),
        ("zstd_cache",       False, True,  "lz4",    "ZSTD warm cache only"),
        ("tiered_lz4+zstd",  True,  True,  "lz4",    "Tiered (LZ4 hot + ZSTD warm)"),
        ("tiered_lzfse+zstd",True,  True,  "lzfse",  "Tiered (LZFSE hot + ZSTD warm)"),
    ]

    cache_results = []
    for mode_name, hot, warm, hot_algo, description in cache_configs:
        print(f"  Running: {description}...")
        result = benchmark_cache_mode(
            expert_dir=expert_dir,
            mode_name=mode_name,
            num_layers=args.layers,
            num_experts=args.experts,
            num_tokens=args.tokens,
            k=4,
            hot_limit_mb=args.hot_mb,
            warm_limit_mb=args.warm_mb,
            num_workers=args.workers,
            enable_hot=hot,
            enable_warm=warm,
            bypass_os_cache=True,
            hot_algo=hot_algo,
            ssd_latency_ms=0.0,  # First pass: no simulated latency (OS page cache)
        )
        cache_results.append(result)

    results["synthetic"]["cache_cached"] = [asdict(r) for r in cache_results]

    # Results table
    print()
    headers = ["Mode", "tok/s", "Avg Layer ms", "Hit Rate", "Hot Hits", "Warm Hits", "Cold Hits", "Evictions"]
    rows = []
    for r in cache_results:
        rows.append([
            r.mode,
            f"{r.tokens_per_second:.2f}",
            f"{r.avg_layer_ms:.3f}",
            f"{r.cache_hit_rate:.1%}",
            str(r.hot_hits),
            str(r.warm_hits),
            str(r.cold_hits),
            str(r.evictions),
        ])
    print_table(headers, rows)

    # Speedup comparison
    baseline_tps = cache_results[0].tokens_per_second
    print()
    print("  Speedup vs SSD-only baseline:")
    for r in cache_results:
        speedup = r.tokens_per_second / baseline_tps if baseline_tps > 0 else 0
        bar = "#" * int(speedup * 20)
        print(f"    {r.mode:20s}  {speedup:.2f}x  {bar}")

    # Time breakdown
    print()
    print_separator("4. Time Breakdown")
    headers = ["Mode", "Total s", "Decompress ms", "SSD Read ms", "Decomp %", "SSD %"]
    rows = []
    for r in cache_results:
        total_ms = r.total_time_s * 1000
        decomp_pct = (r.total_decompress_ms / total_ms * 100) if total_ms > 0 else 0
        ssd_pct = (r.total_ssd_read_ms / total_ms * 100) if total_ms > 0 else 0
        rows.append([
            r.mode,
            f"{r.total_time_s:.3f}",
            f"{r.total_decompress_ms:.1f}",
            f"{r.total_ssd_read_ms:.1f}",
            f"{decomp_pct:.1f}%",
            f"{ssd_pct:.1f}%",
        ])
    print_table(headers, rows)

    # Memory usage
    print()
    print_separator("5. Cache Memory Usage")
    headers = ["Mode", "Hot MB", "Warm MB", "Total Cached MB"]
    rows = []
    for r in cache_results:
        total = r.hot_mb + r.warm_mb
        rows.append([r.mode, f"{r.hot_mb:.1f}", f"{r.warm_mb:.1f}", f"{total:.1f}"])
    print_table(headers, rows)

    # ── Step 6: Simulated real SSD latency ──
    # This simulates what happens when the model exceeds RAM:
    # real NVMe read latency per 2MB expert ≈ 0.12ms (17.5 GB/s)
    # With queue depth contention on Apple Silicon unified memory: ~0.4ms
    print_separator("6. Simulated Real SSD Latency (model exceeds RAM)")
    ssd_latency = 0.4  # ms per 2MB (Apple NVMe with memory bus contention)
    print(f"  Simulated NVMe latency: {ssd_latency} ms per 2MB expert read")
    print(f"  (Real Apple NVMe: ~17.5 GB/s = ~0.12ms/2MB, +contention = ~0.4ms)")
    print()

    # Only run key configs for the simulated benchmark
    sim_configs = [
        ("SSD_only (sim)",     False, False, "lz4",   "SSD only (simulated latency)"),
        ("LZ4_cache (sim)",    True,  False, "lz4",   "LZ4 hot cache + simulated SSD"),
        ("Tiered (sim)",       True,  True,  "lz4",   "Tiered LZ4+ZSTD + simulated SSD"),
    ]

    sim_results = []
    for mode_name, hot, warm, hot_algo, description in sim_configs:
        print(f"  Running: {description}...")
        result = benchmark_cache_mode(
            expert_dir=expert_dir,
            mode_name=mode_name,
            num_layers=args.layers,
            num_experts=args.experts,
            num_tokens=args.tokens,
            k=4,
            hot_limit_mb=args.hot_mb,
            warm_limit_mb=args.warm_mb,
            num_workers=args.workers,
            enable_hot=hot,
            enable_warm=warm,
            bypass_os_cache=True,
            hot_algo=hot_algo,
            ssd_latency_ms=ssd_latency,
        )
        sim_results.append(result)

    results["synthetic"]["cache_simulated_ssd"] = [asdict(r) for r in sim_results]

    print()
    headers = ["Mode", "tok/s", "Avg Layer ms", "Hit Rate", "Hot", "Warm", "Cold", "Speedup"]
    rows = []
    sim_baseline = sim_results[0].tokens_per_second
    for r in sim_results:
        speedup = r.tokens_per_second / sim_baseline if sim_baseline > 0 else 0
        rows.append([
            r.mode,
            f"{r.tokens_per_second:.2f}",
            f"{r.avg_layer_ms:.3f}",
            f"{r.cache_hit_rate:.1%}",
            str(r.hot_hits),
            str(r.warm_hits),
            str(r.cold_hits),
            f"{speedup:.2f}x",
        ])
    print_table(headers, rows)

    # Simulated speedup chart
    print()
    print("  Speedup vs SSD-only (simulated NVMe latency):")
    for r in sim_results:
        speedup = r.tokens_per_second / sim_baseline if sim_baseline > 0 else 0
        bar = "#" * int(speedup * 20)
        print(f"    {r.mode:22s}  {speedup:.2f}x  {bar}")

    return results


def run_model_benchmarks(args) -> dict:
    """Run benchmarks with a real MLX MoE model."""
    results = {"model": {}}

    print_separator("1. Pure MLX Baseline")
    mlx_result = benchmark_pure_mlx(
        model_name=args.model,
        max_tokens=args.tokens,
    )
    if mlx_result:
        results["model"]["pure_mlx"] = asdict(mlx_result)
        print(f"  Tokens/sec: {mlx_result.tokens_per_second:.1f}")
        print(f"  Peak memory: {mlx_result.peak_memory_mb:.0f} MB")

    return results


def run_benchmarks(args=None):
    """Main entry point for benchmarks."""
    if args is None:
        args = parse_args()
    return _run(args)


def _run(args) -> dict:
    all_results = {}

    if args.synthetic:
        all_results.update(run_synthetic_benchmarks(args))

    if args.model:
        all_results.update(run_model_benchmarks(args))

    # Save results to JSON
    results_path = Path(args.work_dir) / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLX-Flash-Compress Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick synthetic benchmark
  python -m mlx_flash_compress.bench --synthetic

  # Larger synthetic benchmark
  python -m mlx_flash_compress.bench --synthetic --layers 16 --experts 128 --tokens 100

  # Real model benchmark
  python -m mlx_flash_compress.bench --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit

  # Both synthetic and model
  python -m mlx_flash_compress.bench --synthetic --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit
        """,
    )

    parser.add_argument("--synthetic", action="store_true", help="Run synthetic benchmarks (no model needed)")
    parser.add_argument("--model", type=str, default=None, help="MLX MoE model name for real inference benchmark")
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate/simulate")
    parser.add_argument("--layers", type=int, default=8, help="Number of MoE layers (synthetic)")
    parser.add_argument("--experts", type=int, default=64, help="Number of experts per layer (synthetic)")
    parser.add_argument("--expert-kb", type=int, default=256, help="Expert size in KB (synthetic)")
    parser.add_argument("--hot-mb", type=int, default=256, help="Hot tier cache size in MB")
    parser.add_argument("--warm-mb", type=int, default=128, help="Warm tier cache size in MB")
    parser.add_argument("--workers", type=int, default=4, help="Parallel decompression workers")
    parser.add_argument("--work-dir", type=str, default="/tmp/mlx_flash_compress", help="Working directory for temp files")

    args = parser.parse_args()

    if not args.synthetic and not args.model:
        args.synthetic = True  # Default to synthetic

    return args


def main():
    args = parse_args()
    _run(args)


if __name__ == "__main__":
    main()
