"""Model Browser: Discover, score, and run MoE models on your Mac.

Lists popular MoE models from Hugging Face, scores them against your
hardware, shows what fits in RAM vs needs SSD, and estimates performance.

Usage:
  python -m mlx_flash_compress.model_browser              # list all
  python -m mlx_flash_compress.model_browser --run best    # auto-run best fit
  python -m mlx_flash_compress.model_browser --run 3       # run model #3
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from mlx_flash_compress.hardware import detect_hardware, MacHardware, estimate_performance


@dataclass
class MoEModelInfo:
    name: str
    display_name: str
    size_gb: float
    num_layers: int
    num_experts: int
    top_k: int
    expert_size_mb: float
    gpu_layer_ms: float
    category: str
    description: str
    available: bool = True  # available in MLX format


KNOWN_MODELS = [
    MoEModelInfo("mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit",
        "Qwen1.5 MoE 2.7B (4bit)", 5.0, 24, 60, 4, 4.75, 0.5,
        "small", "Small MoE, great for testing. Fits on any Mac."),
    MoEModelInfo("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        "Mixtral 8x7B (4bit)", 26.0, 32, 8, 2, 3200.0, 2.0,
        "medium", "Strong general-purpose MoE. Fits on 32GB+ Macs."),
    MoEModelInfo("mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit",
        "Mixtral 8x22B (4bit)", 80.0, 56, 8, 2, 6400.0, 3.0,
        "large", "Powerful MoE. Needs 96GB+ for full RAM."),
    MoEModelInfo("deepseek-ai/DeepSeek-V2-Lite-Chat",
        "DeepSeek V2 Lite (4bit)", 10.0, 28, 64, 6, 2.0, 1.0,
        "small", "Efficient MoE with 64 experts."),
    MoEModelInfo("N/A", "Qwen3-235B MoE (4bit)", 130.0, 48, 512, 4, 4.75, 1.86,
        "huge", "Massive MoE. Needs SSD streaming.", available=False),
    MoEModelInfo("N/A", "DeepSeek V3 (4bit)", 170.0, 61, 256, 8, 6.0, 2.0,
        "huge", "State-of-the-art MoE. 170GB at 4-bit.", available=False),
    MoEModelInfo("N/A", "Qwen3.5-397B (4bit)", 209.0, 60, 512, 4, 6.75, 1.86,
        "huge", "The Flash-MoE target. 209GB from SSD.", available=False),
]


def score_model(model: MoEModelInfo, hw: MacHardware) -> dict:
    fits_ram = model.size_gb <= hw.available_ram_gb
    est = estimate_performance(hw, model.size_gb, model.display_name,
        model.num_layers, model.num_experts, model.top_k,
        model.expert_size_mb, model.gpu_layer_ms)
    est_opt = estimate_performance(hw, model.size_gb, model.display_name,
        model.num_layers, model.num_experts, model.top_k,
        model.expert_size_mb, model.gpu_layer_ms, compression=1.8)

    return {
        "fits_ram": fits_ram, "needs_ssd": not fits_ram,
        "base_tps": est.estimated_tok_per_s,
        "optimized_tps": est_opt.estimated_tok_per_s,
        "hit_rate": est.estimated_hit_rate,
        "optimized_hit_rate": est_opt.estimated_hit_rate,
        "speedup": est_opt.estimated_tok_per_s / max(est.estimated_tok_per_s, 0.1),
        "bottleneck": est.bottleneck,
    }


def print_model_browser(hw: MacHardware):
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  MODEL BROWSER — {hw.chip}, {hw.total_ram_gb:.0f}GB RAM{' ' * (38 - len(hw.chip))}║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                                                                      ║")
    print("║  #  Model                       Size  Fits?   tok/s +Cache  Gain     ║")
    print("║  ── ──────────────────────────── ───── ─────── ───── ────── ──────── ║")

    for i, model in enumerate(KNOWN_MODELS):
        score = score_model(model, hw)
        fits = "✅ RAM" if score["fits_ram"] else "💾 SSD"
        run = "▶" if model.available else " "
        gain = f"+{(score['speedup']-1)*100:.0f}%" if score["speedup"] > 1.01 else "  —"
        print(f"║ {run}{i+1:>2d} {model.display_name:<30s} {model.size_gb:>3.0f}G {fits:>7s} {score['base_tps']:>5.1f} {score['optimized_tps']:>5.1f} {gain:>7s}  ║")

    print("║                                                                      ║")
    print("║  ✅ = fits in RAM    💾 = needs SSD    ▶ = available to download      ║")
    print("║                                                                      ║")

    # Show cache impact for SSD models
    print("║  ── HOW CACHE HELPS (models exceeding RAM) ──                        ║")
    print("║                                                                      ║")

    for model in KNOWN_MODELS:
        score = score_model(model, hw)
        if score["needs_ssd"] and model.num_experts > 1:
            b1 = "█" * max(1, int(score["base_tps"] * 3))
            b2 = "█" * max(1, int(score["optimized_tps"] * 3))
            print(f"║  {model.display_name[:25]:<25s}                                     ║")
            print(f"║    No cache:  {score['base_tps']:>5.1f} tok/s  {b1:<20s}              ║")
            print(f"║    + Cache:   {score['optimized_tps']:>5.1f} tok/s  {b2:<20s}  {(score['speedup']-1)*100:>+.0f}%    ║")
            print("║                                                                      ║")

    # Quick run command
    best = None
    for model in KNOWN_MODELS:
        if model.available:
            score = score_model(model, hw)
            if score["fits_ram"] and (best is None or score["base_tps"] > best[1]["base_tps"]):
                best = (model, score)

    if best:
        m = best[0]
        print(f"║  QUICK RUN: python -m mlx_flash_compress.model_browser --run 1      ║")

    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")


def run_model(model: MoEModelInfo, tokens: int = 100):
    """Launch the model via run.py."""
    if not model.available:
        print(f"\n  {model.display_name} is not yet available in MLX format.")
        return

    print(f"\n  Launching: {model.display_name}")
    print(f"  Command: python -m mlx_flash_compress.run --model {model.name}")
    print()

    subprocess.run(
        [sys.executable, "-m", "mlx_flash_compress.run",
         "--model", model.name, "--tokens", str(tokens)],
    )


def main():
    parser = argparse.ArgumentParser(description="MLX-Flash-Compress Model Browser")
    parser.add_argument("--run", type=str, default=None, help="Run model: 'best' or number (1-7)")
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate")
    args = parser.parse_args()

    hw = detect_hardware()

    if args.run:
        if args.run == "best":
            for model in KNOWN_MODELS:
                if model.available:
                    run_model(model, args.tokens)
                    return
        else:
            try:
                idx = int(args.run) - 1
                run_model(KNOWN_MODELS[idx], args.tokens)
            except (ValueError, IndexError):
                print(f"  Invalid: {args.run}. Use 1-{len(KNOWN_MODELS)} or 'best'")
    else:
        print_model_browser(hw)


if __name__ == "__main__":
    main()
