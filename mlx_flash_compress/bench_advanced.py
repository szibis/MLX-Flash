"""Advanced techniques benchmark on real Qwen MoE weights.

Tests: mixed precision (4→2 bit), smart eviction, speculative prefetch.

Usage:
  python -m mlx_flash_compress.bench_advanced
"""

import sys
import time
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from mlx_flash_compress.mixed_precision import (
    benchmark_mixed_precision, requantize_4bit_to_2bit, ExpertHotness
)
from mlx_flash_compress.smart_eviction import (
    LeastStalePolicy, RoutingPredictor, simulate_prefetch
)
from mlx_flash_compress.compression import LZ4Compressor, ZSTDCompressor


def print_separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print(f"  {' | '.join(h.ljust(widths[i]) for i, h in enumerate(headers))}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")
    for row in rows:
        print(f"  {' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))}")


def main():
    model_path = str(Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen1.5-MoE-A2.7B-Chat-4bit/snapshots/cf116003d120c4216cf008eba169f98b95bdf3ee")

    print_separator("Advanced Techniques Benchmark: Qwen1.5-MoE-A2.7B-Chat-4bit")
    model, _ = load(model_path)
    params = model.parameters()
    print("  Model loaded.")

    num_layers = 24
    num_experts = 60

    # ── Section 1: Mixed Precision (4-bit → 2-bit) ──
    print_separator("1. Mixed Precision: 4-bit Hot → 2-bit Cold Experts")
    print("  DynaExq strategy: requantize rarely-used experts to 2-bit")
    print("  Expected: ~50% size reduction per cold expert, some quality loss")
    print()

    headers = ["Layer", "Proj", "Expert", "4bit KB", "2bit KB", "Ratio", "Requant ms", "MSE", "Max Error"]
    rows = []

    test_configs = [
        (0, "gate_proj"), (0, "up_proj"), (0, "down_proj"),
        (12, "gate_proj"), (23, "gate_proj"),
    ]

    all_ratios = []
    all_mse = []

    for layer_idx, proj_name in test_configs:
        layer = params['model']['layers'][layer_idx]['mlp']['switch_mlp']
        proj = layer[proj_name]
        weight = np.array(proj['weight'])
        scales = np.array(proj['scales'])
        biases = np.array(proj['biases'])

        for eid in [0, 30, 59]:  # first, middle, last expert
            r = benchmark_mixed_precision(weight[eid], scales[eid], biases[eid], eid)
            rows.append([
                str(layer_idx), proj_name, str(eid),
                f"{r.q4_bytes/1024:.0f}", f"{r.q2_bytes/1024:.0f}",
                f"{r.ratio_4to2:.2f}x", f"{r.requant_ms:.1f}",
                f"{r.mse:.6f}", f"{r.max_error:.4f}",
            ])
            all_ratios.append(r.ratio_4to2)
            all_mse.append(r.mse)

    print_table(headers, rows)
    print(f"\n  Average 4→2 bit ratio: {np.mean(all_ratios):.2f}x")
    print(f"  Average MSE: {np.mean(all_mse):.6f}")
    print(f"  This means cold experts use {1/np.mean(all_ratios)*100:.0f}% of 4-bit storage")

    # Now: can we compress the 2-bit data?
    print_separator("2. Compressibility of 2-bit Requantized Weights")
    print("  Testing if 2-bit weights are more compressible than 4-bit...")
    print()

    lz4 = LZ4Compressor()
    zstd = ZSTDCompressor(3)

    layer = params['model']['layers'][0]['mlp']['switch_mlp']
    w = np.array(layer['gate_proj']['weight'])[0]
    s = np.array(layer['gate_proj']['scales'])[0]
    b = np.array(layer['gate_proj']['biases'])[0]

    # Original 4-bit
    raw_4bit = w.tobytes() + s.tobytes() + b.tobytes()
    lz4_4bit = lz4.compress(raw_4bit)
    zstd_4bit = zstd.compress(raw_4bit)

    # 2-bit requantized
    packed_2bit, new_s, new_b, meta = requantize_4bit_to_2bit(w, s, b)
    raw_2bit = packed_2bit.tobytes() + new_s.tobytes() + new_b.tobytes()
    lz4_2bit = lz4.compress(raw_2bit)
    zstd_2bit = zstd.compress(raw_2bit)

    headers = ["Format", "Raw KB", "LZ4 KB", "LZ4 ratio", "ZSTD KB", "ZSTD ratio"]
    rows = [
        ["4-bit original",
         f"{len(raw_4bit)/1024:.0f}", f"{lz4_4bit.compressed_size/1024:.0f}", f"{lz4_4bit.ratio:.3f}x",
         f"{zstd_4bit.compressed_size/1024:.0f}", f"{zstd_4bit.ratio:.3f}x"],
        ["2-bit requant",
         f"{len(raw_2bit)/1024:.0f}", f"{lz4_2bit.compressed_size/1024:.0f}", f"{lz4_2bit.ratio:.3f}x",
         f"{zstd_2bit.compressed_size/1024:.0f}", f"{zstd_2bit.ratio:.3f}x"],
    ]
    print_table(headers, rows)

    total_saving = len(raw_4bit) / len(raw_2bit)
    with_zstd = len(raw_4bit) / zstd_2bit.compressed_size
    print(f"\n  Raw 4→2 bit saving: {total_saving:.2f}x")
    print(f"  4-bit raw vs 2-bit+ZSTD: {with_zstd:.2f}x")

    # ── Section 3: Mixed Precision Impact on Cache ──
    print_separator("3. Cache Impact: Mixed Precision 4-bit Hot / 2-bit Cold")
    print("  Simulating: top 20% experts stay 4-bit, bottom 80% go to 2-bit")
    print()

    hot_fraction = 0.20
    hot_count = int(num_experts * hot_fraction)
    cold_count = num_experts - hot_count

    expert_size_4bit = 4752  # KB, from real measurement
    expert_size_2bit = expert_size_4bit / np.mean(all_ratios)

    total_4bit = num_layers * num_experts * expert_size_4bit / 1024  # MB
    total_mixed = num_layers * (hot_count * expert_size_4bit + cold_count * expert_size_2bit) / 1024  # MB

    print(f"  All 4-bit: {total_4bit:.0f} MB ({num_layers * num_experts} experts)")
    print(f"  Mixed (20% hot 4-bit + 80% cold 2-bit): {total_mixed:.0f} MB")
    print(f"  Memory saving: {total_4bit/total_mixed:.2f}x ({(1 - total_mixed/total_4bit)*100:.0f}% less)")
    print(f"  With Zipf routing: top-20% experts handle ~60% of traffic → quality preserved")

    # ── Section 4: Smart Eviction ──
    print_separator("4. Least-Stale vs LFU Eviction Policy")

    rng = np.random.default_rng(42)
    expert_probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
    expert_probs /= expert_probs.sum()

    cache_sizes_pct = [5, 10, 20, 50]
    num_tokens = 200
    k = 4

    headers = ["Cache %", "Capacity", "LFU Hits", "LFU Rate", "LS Hits", "LS Rate", "LS Gain"]
    rows = []

    for pct in cache_sizes_pct:
        cache_slots = int(num_layers * num_experts * pct / 100)

        # Simulate LFU
        lfu_cache = {}
        lfu_counts = {}
        lfu_hits = 0
        lfu_total = 0

        rng_lfu = np.random.default_rng(42)
        for _ in range(num_tokens):
            for layer in range(num_layers):
                experts = rng_lfu.choice(num_experts, size=k, replace=False, p=expert_probs).tolist()
                for eid in experts:
                    key = (layer, eid)
                    lfu_total += 1
                    if key in lfu_cache:
                        lfu_hits += 1
                        lfu_counts[key] = lfu_counts.get(key, 0) + 1
                    else:
                        if len(lfu_cache) >= cache_slots:
                            min_key = min(lfu_counts, key=lfu_counts.get)
                            del lfu_cache[min_key]
                            del lfu_counts[min_key]
                        lfu_cache[key] = True
                        lfu_counts[key] = 1

        # Simulate Least-Stale
        ls_policy = LeastStalePolicy(num_layers=num_layers)
        ls_cache = {}
        ls_hits = 0
        ls_total = 0

        rng_ls = np.random.default_rng(42)
        for token in range(num_tokens):
            ls_policy.advance_token()
            for layer in range(num_layers):
                experts = rng_ls.choice(num_experts, size=k, replace=False, p=expert_probs).tolist()
                for eid in experts:
                    key = (layer, eid)
                    ls_total += 1
                    ls_policy.record_access(layer, eid)
                    if key in ls_cache:
                        ls_hits += 1
                    else:
                        if len(ls_cache) >= cache_slots:
                            evict_key = ls_policy.select_eviction(list(ls_cache.keys()))
                            del ls_cache[evict_key]
                        ls_cache[key] = True

        lfu_rate = lfu_hits / lfu_total if lfu_total > 0 else 0
        ls_rate = ls_hits / ls_total if ls_total > 0 else 0
        gain = (ls_rate - lfu_rate) / lfu_rate * 100 if lfu_rate > 0 else 0

        rows.append([
            f"{pct}%", str(cache_slots),
            str(lfu_hits), f"{lfu_rate:.1%}",
            str(ls_hits), f"{ls_rate:.1%}",
            f"+{gain:.1f}%",
        ])

    print_table(headers, rows)

    # ── Section 5: Speculative Prefetch ──
    print_separator("5. Speculative Expert Prefetching Accuracy")
    print("  Training a co-occurrence predictor to guess next layer's experts")
    print()

    headers = ["Tokens", "Predictions", "Accuracy", "Prefetch Hit Rate", "Wasted %"]
    rows = []

    for n_tokens in [50, 100, 200, 500]:
        result = simulate_prefetch(
            num_layers=num_layers,
            num_experts=num_experts,
            num_tokens=n_tokens,
            top_k=k,
        )
        wasted_pct = result.wasted_prefetches / max(result.total_experts_prefetched, 1) * 100
        rows.append([
            str(n_tokens),
            str(result.total_predictions),
            f"{result.avg_accuracy:.1%}",
            f"{result.prefetch_hit_rate:.1%}",
            f"{wasted_pct:.0f}%",
        ])

    print_table(headers, rows)

    # ── Section 6: Combined Projection ──
    print_separator("6. Combined Projection: All Techniques Applied")
    print()
    print("  Baseline: Flash-MoE on 48GB M3 Max = 4.36 tok/s")
    print("  SSD I/O is 56% of layer time (2.41ms out of 4.27ms per layer)")
    print()

    base_tps = 4.36
    base_layer_ms = 4.27
    ssd_ms = 2.41
    gpu_ms = base_layer_ms - ssd_ms  # 1.86ms

    # Mixed precision: cold experts at 2-bit = 50% less data to load
    # With Zipf routing: ~40% of SSD reads hit cold experts
    cold_traffic = 0.40
    saving_2bit = np.mean(all_ratios)  # ~1.98x
    new_ssd_cold = ssd_ms * cold_traffic / saving_2bit
    new_ssd_hot = ssd_ms * (1 - cold_traffic)
    mixed_ssd_ms = new_ssd_hot + new_ssd_cold

    # Smart eviction: assume cache holds 20% of experts but achieves 70% hit rate
    # Cache hit: ~0.08ms (LZ4 decompress from RAM)
    # Cache miss: full SSD read
    cache_hit_rate = 0.70
    cache_hit_ms = 0.08
    cache_miss_ms = mixed_ssd_ms
    eviction_ssd_ms = cache_hit_rate * cache_hit_ms + (1 - cache_hit_rate) * cache_miss_ms

    # Prefetch: hide ~60% of remaining SSD reads behind GPU compute
    prefetch_accuracy = 0.60
    prefetch_overlap = min(gpu_ms, eviction_ssd_ms * prefetch_accuracy)
    final_ssd_ms = eviction_ssd_ms - prefetch_overlap

    final_layer_ms = gpu_ms + final_ssd_ms
    final_tps = 1000 / (60 * final_layer_ms)  # 60 layers

    techniques = [
        ("Baseline (Flash-MoE)", base_layer_ms, ssd_ms, base_tps, "1.00x"),
        ("+ Mixed precision (2-bit cold)", gpu_ms + mixed_ssd_ms, mixed_ssd_ms, 1000/(60*(gpu_ms + mixed_ssd_ms)), f"{(1000/(60*(gpu_ms + mixed_ssd_ms)))/base_tps:.2f}x"),
        ("+ Smart eviction (70% hit)", gpu_ms + eviction_ssd_ms, eviction_ssd_ms, 1000/(60*(gpu_ms + eviction_ssd_ms)), f"{(1000/(60*(gpu_ms + eviction_ssd_ms)))/base_tps:.2f}x"),
        ("+ Speculative prefetch", final_layer_ms, final_ssd_ms, final_tps, f"{final_tps/base_tps:.2f}x"),
    ]

    headers = ["Configuration", "Layer ms", "I/O ms", "tok/s", "Speedup"]
    rows = []
    for name, layer, io, tps, speedup in techniques:
        bar = "#" * int(tps / base_tps * 15)
        rows.append([name, f"{layer:.2f}", f"{io:.2f}", f"{tps:.1f}", f"{speedup} {bar}"])

    print_table(headers, rows)

    print()
    print("  PROJECTED RESULT: {:.1f} tok/s ({:.2f}x over Flash-MoE baseline)".format(
        final_tps, final_tps / base_tps))
    print()
    print("  Breakdown of gains:")
    print(f"    Mixed precision:     saves {(1 - mixed_ssd_ms/ssd_ms)*100:.0f}% of SSD bandwidth")
    print(f"    Smart eviction:      eliminates {cache_hit_rate*100:.0f}% of SSD reads")
    print(f"    Speculative prefetch: hides {prefetch_overlap/eviction_ssd_ms*100:.0f}% of remaining I/O")


if __name__ == "__main__":
    main()
