"""Tier Optimizer: Find the optimal SSD/RAM split for a given model + hardware.

The key question: given N GB of RAM and M GB of expert weights, what's the
optimal fraction to keep in RAM (compressed) vs stream from SSD?

Variables:
  - RAM budget for expert cache (after OS, GPU buffers, KV cache)
  - Expert compression ratio (1.0x for 4-bit, 1.8x for 2-bit mixed)
  - SSD bandwidth (17.5 GB/s internal, +TB5 external)
  - Expert access frequency distribution (Zipf α parameter)
  - Number of layers, experts per layer, K (active per token)

The optimizer profiles the model's routing distribution, then sweeps
the RAM/SSD split to find the configuration that maximizes tok/s.

Usage:
  python -m mlx_flash_compress.tier_optimizer --model-gb 7 --ram-gb 2 --ssd-latency 0.6
  python -m mlx_flash_compress.tier_optimizer --model-gb 209 --ram-gb 35 --ssd-latency 2.4
"""

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class HardwareProfile:
    """Hardware characteristics for optimization."""
    total_ram_gb: float = 48.0
    os_overhead_gb: float = 6.0   # OS + GPU buffers + non-expert weights
    kv_cache_gb: float = 0.5     # KV cache for full context
    ssd_bandwidth_gbs: float = 17.5
    ssd_latency_ms_per_mb: float = 0.057  # 1MB / 17.5 GB/s
    gpu_layer_ms: float = 1.86   # non-I/O layer time
    ram_decompress_gbs: float = 25.0  # LZ4 decompress speed
    num_tb5_drives: int = 0      # external drives
    tb5_bandwidth_gbs: float = 8.0

    @property
    def available_ram_gb(self) -> float:
        return self.total_ram_gb - self.os_overhead_gb - self.kv_cache_gb

    @property
    def total_ssd_bandwidth(self) -> float:
        return self.ssd_bandwidth_gbs + self.num_tb5_drives * self.tb5_bandwidth_gbs


@dataclass
class ModelProfile:
    """Model characteristics for optimization."""
    total_expert_gb: float = 209.0
    num_layers: int = 60
    num_experts: int = 512
    k: int = 4
    expert_size_mb: float = 6.75
    zipf_alpha: float = 0.8  # routing distribution skew


@dataclass
class TierConfig:
    """A specific RAM/SSD tier configuration."""
    ram_fraction: float         # fraction of experts in RAM
    compression_ratio: float    # effective compression (1.0 = none, 1.8 = 2-bit)
    ram_experts: int
    ssd_experts: int
    ram_gb_used: float
    hit_rate: float            # fraction of accesses served from RAM
    effective_io_ms: float     # average I/O time per layer
    layer_ms: float            # total layer time
    tok_per_s: float           # projected throughput


def compute_hit_rate(
    ram_experts: int,
    total_experts: int,
    k: int,
    zipf_alpha: float,
) -> float:
    """Compute cache hit rate given RAM capacity and Zipf routing.

    With Zipf distribution, the top-N experts by frequency account for
    a disproportionate share of total accesses. If we cache the top-N
    experts in RAM, the hit rate is the cumulative probability mass of
    those N experts under the Zipf distribution.
    """
    if ram_experts >= total_experts:
        return 1.0
    if ram_experts <= 0:
        return 0.0

    # Zipf probabilities: P(rank=i) ∝ 1/(i+1)^α
    probs = np.array([(1.0 / (i + 1)) ** zipf_alpha for i in range(total_experts)])
    probs /= probs.sum()

    # Top-N experts by probability (already sorted by rank)
    top_n_mass = probs[:ram_experts].sum()

    # Hit rate: probability that all K selected experts are in the top-N
    # For each expert selection: P(in cache) = top_n_mass
    # For K independent selections: P(all in cache) = top_n_mass^K
    # But we want: average fraction of K experts that are in cache
    # E[hits/K] = top_n_mass
    return float(top_n_mass)


def optimize_tiers(
    hw: HardwareProfile,
    model: ModelProfile,
    compression_ratios: list[float] = None,
    granularity: int = 20,
) -> list[TierConfig]:
    """Sweep RAM/SSD splits and compression ratios to find optimal config.

    Returns sorted list of TierConfig (best first by tok/s).
    """
    if compression_ratios is None:
        compression_ratios = [1.0, 1.3, 1.5, 1.8, 2.0, 2.4]  # none to aggressive

    results = []

    for comp_ratio in compression_ratios:
        # How many experts fit in available RAM with this compression?
        available_mb = hw.available_ram_gb * 1024
        effective_expert_mb = model.expert_size_mb / comp_ratio
        max_ram_experts = int(available_mb / effective_expert_mb)
        max_ram_experts = min(max_ram_experts, model.num_layers * model.num_experts)

        for frac_idx in range(granularity + 1):
            ram_frac = frac_idx / granularity
            ram_experts = int(ram_frac * max_ram_experts)

            if ram_experts < 0:
                continue

            ssd_experts = model.num_layers * model.num_experts - ram_experts
            ram_gb = ram_experts * effective_expert_mb / 1024

            if ram_gb > hw.available_ram_gb:
                continue

            # Hit rate based on Zipf routing
            per_layer_experts = model.num_experts
            per_layer_ram = min(ram_experts // max(model.num_layers, 1), per_layer_experts)
            hit_rate = compute_hit_rate(per_layer_ram, per_layer_experts, model.k, model.zipf_alpha)

            # I/O time per layer
            ram_hit_ms = 0.08  # LZ4 decompress from RAM
            ssd_miss_ms = model.expert_size_mb * model.k / hw.total_ssd_bandwidth * 1000

            # Account for contention
            if hw.ssd_latency_ms_per_mb > 0:
                ssd_miss_ms = model.expert_size_mb * model.k * hw.ssd_latency_ms_per_mb

            effective_io = hit_rate * ram_hit_ms + (1 - hit_rate) * ssd_miss_ms
            layer_ms = hw.gpu_layer_ms + effective_io
            tps = 1000 / (model.num_layers * layer_ms)

            results.append(TierConfig(
                ram_fraction=ram_frac,
                compression_ratio=comp_ratio,
                ram_experts=ram_experts,
                ssd_experts=ssd_experts,
                ram_gb_used=ram_gb,
                hit_rate=hit_rate,
                effective_io_ms=effective_io,
                layer_ms=layer_ms,
                tok_per_s=tps,
            ))

    results.sort(key=lambda c: -c.tok_per_s)
    return results


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print(f"  {' | '.join(h.ljust(widths[i]) for i, h in enumerate(headers))}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")
    for row in rows:
        print(f"  {' | '.join(str(c).ljust(widths[i]) for i, c in enumerate(row))}")


def main():
    parser = argparse.ArgumentParser(description="Tier Optimizer: find optimal SSD/RAM split")
    parser.add_argument("--total-ram", type=float, default=48.0, help="Total system RAM (GB)")
    parser.add_argument("--model-gb", type=float, default=209.0, help="Total expert weights (GB)")
    parser.add_argument("--layers", type=int, default=60, help="Number of MoE layers")
    parser.add_argument("--experts", type=int, default=512, help="Experts per layer")
    parser.add_argument("--expert-mb", type=float, default=6.75, help="Expert size (MB)")
    parser.add_argument("--k", type=int, default=4, help="Active experts per token")
    parser.add_argument("--ssd-latency", type=float, default=0.057, help="SSD latency ms/MB")
    parser.add_argument("--gpu-ms", type=float, default=1.86, help="GPU layer time (ms)")
    parser.add_argument("--tb5-drives", type=int, default=0, help="External TB5 NVMe drives")
    parser.add_argument("--zipf", type=float, default=0.8, help="Zipf alpha (routing skew)")
    args = parser.parse_args()

    hw = HardwareProfile(
        total_ram_gb=args.total_ram,
        ssd_latency_ms_per_mb=args.ssd_latency,
        gpu_layer_ms=args.gpu_ms,
        num_tb5_drives=args.tb5_drives,
    )
    model = ModelProfile(
        total_expert_gb=args.model_gb,
        num_layers=args.layers,
        num_experts=args.experts,
        k=args.k,
        expert_size_mb=args.expert_mb,
        zipf_alpha=args.zipf,
    )

    print(f"\n{'=' * 70}")
    print(f"  Tier Optimizer: {args.model_gb:.0f}GB model on {args.total_ram:.0f}GB Mac")
    print(f"{'=' * 70}\n")
    print(f"  Available RAM for cache: {hw.available_ram_gb:.1f} GB")
    print(f"  Total SSD bandwidth: {hw.total_ssd_bandwidth:.1f} GB/s")
    print(f"  Model: {model.num_layers} layers × {model.num_experts} experts × {model.expert_size_mb:.2f} MB = {model.total_expert_gb:.0f} GB")
    print()

    results = optimize_tiers(hw, model)

    # Show top 10 configs
    print("  TOP 10 CONFIGURATIONS (sorted by tok/s):\n")
    headers = ["#", "Compress", "RAM GB", "RAM Experts", "Hit Rate", "I/O ms", "Layer ms", "tok/s"]
    rows = []
    for i, cfg in enumerate(results[:10]):
        rows.append([
            str(i + 1),
            f"{cfg.compression_ratio:.1f}x",
            f"{cfg.ram_gb_used:.1f}",
            str(cfg.ram_experts),
            f"{cfg.hit_rate:.1%}",
            f"{cfg.effective_io_ms:.2f}",
            f"{cfg.layer_ms:.2f}",
            f"{cfg.tok_per_s:.1f}",
        ])
    print_table(headers, rows)

    best = results[0]
    worst_ssd = [r for r in results if r.ram_experts == 0][0] if any(r.ram_experts == 0 for r in results) else results[-1]

    print(f"\n  BEST CONFIG:")
    print(f"    Compression: {best.compression_ratio:.1f}x")
    print(f"    RAM: {best.ram_gb_used:.1f} GB ({best.ram_experts} experts)")
    print(f"    SSD: {best.ssd_experts} experts")
    print(f"    Hit rate: {best.hit_rate:.1%}")
    print(f"    tok/s: {best.tok_per_s:.1f}")
    print(f"    Speedup vs all-SSD: {best.tok_per_s / worst_ssd.tok_per_s:.2f}x")

    # Sweep visualization
    print(f"\n  RAM ALLOCATION vs THROUGHPUT (compression={best.compression_ratio:.1f}x):\n")
    best_ratio_results = [r for r in results if abs(r.compression_ratio - best.compression_ratio) < 0.01]
    best_ratio_results.sort(key=lambda r: r.ram_gb_used)

    for r in best_ratio_results[::max(1, len(best_ratio_results) // 15)]:
        bar = "#" * int(r.tok_per_s / best.tok_per_s * 30)
        marker = " ← BEST" if abs(r.tok_per_s - best.tok_per_s) < 0.01 else ""
        print(f"    {r.ram_gb_used:5.1f}GB RAM  {r.hit_rate:5.1%} hit  {r.tok_per_s:5.1f} tok/s  {bar}{marker}")


if __name__ == "__main__":
    main()
