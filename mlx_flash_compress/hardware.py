"""Auto-detect Mac hardware and estimate performance for MoE inference.

Detects: chip type, RAM, GPU cores, SSD gen/speed, Neural Engine.
Builds a performance matrix for different model sizes and configurations.

Usage:
  python -m mlx_flash_compress.hardware
"""

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class MacHardware:
    """Detected Mac hardware profile."""
    chip: str = "Unknown"
    chip_family: str = "Unknown"  # M1, M2, M3, M4
    chip_tier: str = "Unknown"    # base, Pro, Max, Ultra
    cpu_cores: int = 0
    p_cores: int = 0
    e_cores: int = 0
    gpu_cores: int = 0
    neural_engine_cores: int = 16
    total_ram_gb: float = 0.0
    memory_bandwidth_gbs: float = 0.0
    ssd_size_gb: float = 0.0
    ssd_protocol: str = "Apple Fabric"
    estimated_ssd_read_gbs: float = 0.0
    macos_version: str = ""
    thunderbolt_version: int = 0
    thunderbolt_ports: int = 0

    # Estimated capabilities
    amx_tflops_fp16: float = 0.0
    ane_tops_int8: float = 0.0

    @property
    def available_ram_gb(self) -> float:
        """RAM available for expert caching (after OS + GPU + KV cache)."""
        os_overhead = min(self.total_ram_gb * 0.15, 8.0)  # 15% or 8GB max
        return max(self.total_ram_gb - os_overhead, 1.0)

    @property
    def ssd_latency_ms_per_mb(self) -> float:
        if self.estimated_ssd_read_gbs > 0:
            return 1.0 / (self.estimated_ssd_read_gbs * 1024) * 1000
        return 0.1  # fallback


# ── Chip performance databases ────────────────────────────────

CHIP_SPECS = {
    # (memory_bw_gbs, gpu_cores, ane_tops, amx_tflops, ssd_gbs, tb_version, tb_ports)
    "M1":          (68.25,  8,  11.0, 2.0,  2.8, 4, 2),
    "M1 Pro":      (200.0,  16, 11.0, 4.0,  5.0, 4, 3),
    "M1 Max":      (400.0,  32, 11.0, 8.0,  7.4, 4, 4),
    "M1 Ultra":    (800.0,  64, 22.0, 16.0, 7.4, 4, 6),
    "M2":          (100.0,  10, 15.8, 3.0,  3.5, 4, 2),
    "M2 Pro":      (200.0,  19, 15.8, 6.0,  5.0, 4, 3),
    "M2 Max":      (400.0,  38, 15.8, 12.0, 7.4, 4, 4),
    "M2 Ultra":    (800.0,  76, 31.6, 24.0, 7.4, 4, 6),
    "M3":          (100.0,  10, 18.0, 3.5,  3.5, 4, 2),
    "M3 Pro":      (150.0,  18, 18.0, 6.0,  5.0, 4, 3),
    "M3 Max":      (400.0,  40, 18.0, 12.0, 7.4, 4, 4),
    "M3 Ultra":    (800.0,  80, 36.0, 24.0, 7.4, 4, 6),
    "M4":          (120.0,  10, 38.0, 4.0,  4.0, 5, 2),
    "M4 Pro":      (273.0,  20, 38.0, 8.0,  5.0, 5, 3),
    "M4 Max":      (546.0,  40, 38.0, 16.0, 7.4, 5, 4),
    "M4 Ultra":    (819.0,  80, 76.0, 32.0, 7.4, 5, 6),
}


def detect_hardware() -> MacHardware:
    """Auto-detect current Mac hardware."""
    hw = MacHardware()

    # macOS version
    hw.macos_version = platform.mac_ver()[0]

    # Chip info via system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        hw_info = data.get("SPHardwareDataType", [{}])[0]

        hw.chip = hw_info.get("chip_type", "Unknown")

        # Parse "proc 14:10:4:0" format (total:perf:eff:?)
        cores_str = hw_info.get("number_processors", "0")
        cores_match = re.match(r"proc\s+(\d+):(\d+):(\d+)", cores_str)
        if cores_match:
            hw.cpu_cores = int(cores_match.group(1))
            hw.p_cores = int(cores_match.group(2))
            hw.e_cores = int(cores_match.group(3))
        else:
            try:
                hw.cpu_cores = int(cores_str.split()[0])
            except (ValueError, IndexError):
                pass

        # RAM
        ram_str = hw_info.get("physical_memory", "0 GB")
        ram_match = re.search(r"(\d+)\s*GB", ram_str)
        if ram_match:
            hw.total_ram_gb = float(ram_match.group(1))

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        pass

    # Identify chip family and tier
    chip_lower = hw.chip.lower().replace("apple ", "")
    for name, specs in CHIP_SPECS.items():
        if name.lower().replace(" ", "") in chip_lower.replace(" ", ""):
            hw.memory_bandwidth_gbs = specs[0]
            hw.gpu_cores = specs[1]
            hw.ane_tops_int8 = specs[2]
            hw.amx_tflops_fp16 = specs[3]
            hw.estimated_ssd_read_gbs = specs[4]
            hw.thunderbolt_version = specs[5]
            hw.thunderbolt_ports = specs[6]

            # Parse family/tier
            parts = name.split()
            hw.chip_family = parts[0]
            hw.chip_tier = parts[1] if len(parts) > 1 else "base"
            break

    # SSD size
    try:
        result = subprocess.run(
            ["diskutil", "info", "disk0"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.split("\n"):
            if "Disk Size" in line:
                size_match = re.search(r"([\d.]+)\s*TB", line)
                if size_match:
                    hw.ssd_size_gb = float(size_match.group(1)) * 1024
                else:
                    gb_match = re.search(r"([\d.]+)\s*GB", line)
                    if gb_match:
                        hw.ssd_size_gb = float(gb_match.group(1))
    except (subprocess.TimeoutExpired, KeyError):
        pass

    return hw


@dataclass
class PerformanceEstimate:
    """Estimated performance for a specific model on detected hardware."""
    model_name: str
    model_gb: float
    fits_in_ram: bool
    ram_cache_gb: float
    estimated_hit_rate: float
    estimated_tok_per_s: float
    estimated_layer_ms: float
    bottleneck: str  # "compute", "memory_bandwidth", "ssd_bandwidth"


def estimate_performance(
    hw: MacHardware,
    model_gb: float,
    model_name: str = "",
    num_layers: int = 60,
    num_experts: int = 512,
    k: int = 4,
    expert_size_mb: float = 6.75,
    gpu_layer_ms: float = 1.86,
    zipf_alpha: float = 0.8,
    compression: float = 1.0,
    ram_budget_fraction: float = 0.85,
) -> PerformanceEstimate:
    """Estimate tok/s for a model on the detected hardware."""

    available_ram = hw.available_ram_gb * ram_budget_fraction
    fits = model_gb <= available_ram

    if fits:
        # Model fits entirely in RAM — compute-bound
        # Estimate based on memory bandwidth for matmul
        layer_ms = gpu_layer_ms
        tps = 1000 / (num_layers * layer_ms)
        return PerformanceEstimate(
            model_name=model_name, model_gb=model_gb,
            fits_in_ram=True, ram_cache_gb=model_gb,
            estimated_hit_rate=1.0, estimated_tok_per_s=tps,
            estimated_layer_ms=layer_ms, bottleneck="compute",
        )

    # Model exceeds RAM — SSD streaming
    effective_expert_mb = expert_size_mb / compression
    ram_for_cache = available_ram
    max_experts = int(ram_for_cache * 1024 / effective_expert_mb)
    per_layer_cached = min(max_experts // max(num_layers, 1), num_experts)

    # Hit rate from Zipf
    if per_layer_cached >= num_experts:
        hit_rate = 1.0
    else:
        probs = np.array([(1.0 / (i + 1)) ** zipf_alpha for i in range(num_experts)])
        probs /= probs.sum()
        hit_rate = float(probs[:per_layer_cached].sum())

    # I/O time
    ssd_ms = expert_size_mb * k / hw.estimated_ssd_read_gbs / 1024 * 1000
    ram_ms = 0.08
    io_ms = hit_rate * ram_ms + (1 - hit_rate) * ssd_ms

    layer_ms = gpu_layer_ms + io_ms
    tps = 1000 / (num_layers * layer_ms)

    bottleneck = "ssd_bandwidth" if io_ms > gpu_layer_ms else "compute"

    return PerformanceEstimate(
        model_name=model_name, model_gb=model_gb,
        fits_in_ram=fits, ram_cache_gb=min(ram_for_cache, model_gb),
        estimated_hit_rate=hit_rate, estimated_tok_per_s=tps,
        estimated_layer_ms=layer_ms, bottleneck=bottleneck,
    )


# Need numpy for Zipf calculation
import numpy as np


def print_performance_matrix(hw: MacHardware):
    """Print performance estimates for common models on this hardware."""
    models = [
        ("Qwen1.5-MoE-A2.7B (4bit)", 5.0, 24, 60, 4, 4.75, 0.5),
        ("Mixtral-8x7B (4bit)", 26.0, 32, 8, 2, 3200.0, 2.0),
        ("Qwen3-235B (4bit)", 130.0, 48, 512, 4, 4.75, 1.86),
        ("DeepSeek-V3 (4bit)", 170.0, 61, 256, 8, 6.0, 2.0),
        ("Qwen3.5-397B (4bit)", 209.0, 60, 512, 4, 6.75, 1.86),
    ]

    print(f"\n  Performance Matrix for {hw.chip} ({hw.total_ram_gb:.0f}GB RAM)")
    print(f"  {'=' * 65}")
    print(f"  {'Model':<30s} {'Size':>6s} {'Fits?':>5s} {'Hit%':>5s} {'tok/s':>6s} {'Bottleneck':>12s}")
    print(f"  {'-' * 65}")

    for name, gb, layers, experts, k, expert_mb, gpu_ms in models:
        est = estimate_performance(
            hw, gb, name, layers, experts, k, expert_mb, gpu_ms,
        )
        fits = "Yes" if est.fits_in_ram else "No"
        print(f"  {name:<30s} {gb:>5.0f}G {fits:>5s} {est.estimated_hit_rate:>4.0%} {est.estimated_tok_per_s:>6.1f} {est.bottleneck:>12s}")

    # Show with compression
    print(f"\n  With 2-bit mixed precision (1.8x compression on cold experts):")
    print(f"  {'-' * 65}")
    for name, gb, layers, experts, k, expert_mb, gpu_ms in models:
        est = estimate_performance(
            hw, gb, name, layers, experts, k, expert_mb, gpu_ms,
            compression=1.8,
        )
        fits = "Yes" if est.fits_in_ram else "No"
        print(f"  {name:<30s} {gb:>5.0f}G {fits:>5s} {est.estimated_hit_rate:>4.0%} {est.estimated_tok_per_s:>6.1f} {est.bottleneck:>12s}")


def print_live_calculator(hw: MacHardware):
    """Interactive display of how RAM allocation affects performance."""
    print(f"\n  LIVE CALCULATOR: Adjust RAM budget for expert caching")
    print(f"  Hardware: {hw.chip}, {hw.total_ram_gb:.0f}GB RAM, {hw.ssd_size_gb:.0f}GB SSD")
    print(f"  Available for cache: {hw.available_ram_gb:.1f}GB")
    print()

    # Show impact of different RAM allocations for 397B model
    model_gb = 209.0
    print(f"  Model: Qwen3.5-397B (4bit) — {model_gb:.0f}GB expert weights")
    print(f"  {'RAM for cache':>14s} {'Hit Rate':>9s} {'tok/s':>7s} {'Speedup':>8s}  Visual")
    print(f"  {'-' * 60}")

    base_tps = None
    for pct in range(0, 105, 5):
        ram = hw.available_ram_gb * pct / 100
        est = estimate_performance(hw, model_gb, ram_budget_fraction=pct / 100)
        if base_tps is None:
            base_tps = est.estimated_tok_per_s
        speedup = est.estimated_tok_per_s / base_tps if base_tps > 0 else 0
        bar = "#" * int(est.estimated_tok_per_s * 3)
        print(f"  {ram:>10.1f} GB  {est.estimated_hit_rate:>8.1%} {est.estimated_tok_per_s:>6.1f}  {speedup:>6.2f}x  {bar}")


def main():
    hw = detect_hardware()

    print(f"\n{'=' * 70}")
    print(f"  MLX-Flash: Hardware Detection & Performance Calculator")
    print(f"{'=' * 70}\n")

    print(f"  Detected Hardware:")
    print(f"    Chip:            {hw.chip}")
    print(f"    CPU Cores:       {hw.p_cores}P + {hw.e_cores}E = {hw.cpu_cores} total")
    print(f"    GPU Cores:       {hw.gpu_cores}")
    print(f"    Neural Engine:   {hw.ane_tops_int8:.0f} TOPS")
    print(f"    AMX:             {hw.amx_tflops_fp16:.1f} TFLOPS FP16")
    print(f"    RAM:             {hw.total_ram_gb:.0f} GB (available for cache: {hw.available_ram_gb:.1f} GB)")
    print(f"    Memory BW:       {hw.memory_bandwidth_gbs:.0f} GB/s")
    print(f"    SSD:             {hw.ssd_size_gb:.0f} GB @ ~{hw.estimated_ssd_read_gbs:.1f} GB/s read")
    print(f"    Thunderbolt:     TB{hw.thunderbolt_version} × {hw.thunderbolt_ports} ports")
    print(f"    macOS:           {hw.macos_version}")

    print_performance_matrix(hw)
    print_live_calculator(hw)


if __name__ == "__main__":
    main()
