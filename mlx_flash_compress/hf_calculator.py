"""HuggingFace Space calculator: estimate MLX-Flash compression and memory savings.

Deployable as a Gradio app on HuggingFace Spaces.
Shows compression ratio, memory requirements, and recommended settings
for any MoE model.

Run locally:
  python -m mlx_flash_compress.hf_calculator

Deploy to HF Spaces:
  Copy this file + requirements to a new HF Space with Gradio SDK.
"""

from dataclasses import dataclass


# -- Model database --

KNOWN_MODELS = {
    "Qwen3-30B-A3B": {"total_b": 30, "active_b": 3, "experts": 128, "layers": 48, "type": "MoE"},
    "Qwen3.5-35B-A3B": {"total_b": 35, "active_b": 3, "experts": 128, "layers": 64, "type": "MoE"},
    "Mixtral-8x7B": {"total_b": 47, "active_b": 13, "experts": 8, "layers": 32, "type": "MoE"},
    "Mixtral-8x22B": {"total_b": 141, "active_b": 39, "experts": 8, "layers": 56, "type": "MoE"},
    "DeepSeek-V3-671B": {"total_b": 671, "active_b": 37, "experts": 256, "layers": 61, "type": "MoE"},
    "Qwen3-4B": {"total_b": 4, "active_b": 4, "experts": 0, "layers": 32, "type": "dense"},
    "Qwen3-8B": {"total_b": 8, "active_b": 8, "experts": 0, "layers": 32, "type": "dense"},
    "Qwen3-14B": {"total_b": 14, "active_b": 14, "experts": 0, "layers": 40, "type": "dense"},
    "Llama-3-70B": {"total_b": 70, "active_b": 70, "experts": 0, "layers": 80, "type": "dense"},
}


def estimate_model(model_name: str = "", total_params_b: float = 0,
                   active_params_b: float = 0, num_experts: int = 0,
                   num_layers: int = 32, quant_bits: int = 4,
                   ram_gb: float = 36, cache_capacity_pct: float = 50) -> dict:
    """Estimate memory, compression, and performance for a model.

    Args:
        model_name: Known model name (auto-fills params) or custom
        total_params_b: Total parameters in billions
        active_params_b: Active parameters per token (= total for dense)
        num_experts: Number of experts per layer (0 for dense)
        num_layers: Number of transformer layers
        quant_bits: Quantization bits (4, 8, 16)
        ram_gb: Available RAM in GB
        cache_capacity_pct: What % of experts to cache (for MoE)

    Returns dict with all estimates.
    """
    # Auto-fill from known models
    if model_name in KNOWN_MODELS:
        info = KNOWN_MODELS[model_name]
        total_params_b = info["total_b"]
        active_params_b = info["active_b"]
        num_experts = info["experts"]
        num_layers = info["layers"]

    if total_params_b == 0:
        total_params_b = 7  # default
    if active_params_b == 0:
        active_params_b = total_params_b

    is_moe = num_experts > 0

    # Model size estimates
    bytes_per_param = quant_bits / 8
    total_size_gb = total_params_b * 1e9 * bytes_per_param / (1024 ** 3)
    active_size_gb = active_params_b * 1e9 * bytes_per_param / (1024 ** 3)

    # KV cache estimate (4096 context, fp16)
    head_dim = 128
    num_heads = max(32, int(active_params_b * 4))  # rough estimate
    kv_per_layer_mb = 2 * num_heads * head_dim * 4096 * 2 / (1024 ** 2)
    kv_total_gb = kv_per_layer_mb * num_layers / 1024

    # With KV 8-bit quantization
    kv_8bit_gb = kv_total_gb * 0.5

    # Expert caching (MoE only)
    if is_moe:
        expert_size_gb = (total_size_gb - active_size_gb) / max(num_experts * num_layers, 1) * num_experts
        cached_experts = int(num_experts * cache_capacity_pct / 100)
        cached_size_gb = expert_size_gb * cache_capacity_pct / 100
        full_load_gb = total_size_gb
        streaming_gb = active_size_gb + cached_size_gb + kv_total_gb
    else:
        expert_size_gb = 0
        cached_experts = 0
        cached_size_gb = 0
        full_load_gb = total_size_gb
        streaming_gb = total_size_gb + kv_total_gb

    # MLX-Flash optimizations
    mixed_precision_savings = 0.2 if is_moe else 0  # 20% from hot 4-bit / cold 2-bit
    entropy_savings = 0.15  # 15% from Huffman coding
    vertical_split_coverage = 2.0 if is_moe else 1.0

    optimized_size_gb = streaming_gb * (1 - mixed_precision_savings) * (1 - entropy_savings)

    # Fit analysis
    fits_full = full_load_gb < ram_gb * 0.85
    fits_streaming = streaming_gb < ram_gb * 0.85
    fits_optimized = optimized_size_gb < ram_gb * 0.85

    # Recommendation
    if fits_full:
        recommendation = "Full model fits in RAM. No streaming needed — maximum speed."
    elif fits_streaming:
        recommendation = "Fits with expert streaming. Enable MLX-Flash for best performance."
    elif fits_optimized:
        recommendation = "Fits with MLX-Flash optimizations (mixed precision + entropy coding)."
    else:
        nodes_needed = max(2, int(full_load_gb / (ram_gb * 0.8)) + 1)
        recommendation = f"Too large for single Mac. Need {nodes_needed} nodes with distributed expert parallelism."

    return {
        "model": model_name or "Custom",
        "type": "MoE" if is_moe else "Dense",
        "total_params_b": total_params_b,
        "active_params_b": active_params_b,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "quant_bits": quant_bits,
        # Sizes
        "total_size_gb": round(total_size_gb, 1),
        "active_size_gb": round(active_size_gb, 1),
        "kv_cache_gb": round(kv_total_gb, 1),
        "kv_cache_8bit_gb": round(kv_8bit_gb, 1),
        # Streaming
        "full_load_gb": round(full_load_gb, 1),
        "streaming_gb": round(streaming_gb, 1),
        "optimized_gb": round(optimized_size_gb, 1),
        "savings_vs_full_pct": round((1 - optimized_size_gb / max(full_load_gb, 0.1)) * 100, 1),
        # Expert cache
        "cached_experts": cached_experts,
        "cache_capacity_pct": cache_capacity_pct,
        "vertical_split_coverage": f"{vertical_split_coverage:.0f}x",
        # Fit
        "ram_gb": ram_gb,
        "fits_full": fits_full,
        "fits_streaming": fits_streaming,
        "fits_optimized": fits_optimized,
        "recommendation": recommendation,
    }


def format_estimate(est: dict) -> str:
    """Format estimate as a readable string."""
    lines = []
    lines.append(f"{'═' * 50}")
    lines.append(f"  Model: {est['model']} ({est['type']})")
    lines.append(f"  Params: {est['total_params_b']}B total, {est['active_params_b']}B active")
    if est['num_experts'] > 0:
        lines.append(f"  Experts: {est['num_experts']} per layer × {est['num_layers']} layers")
    lines.append(f"  Quant: {est['quant_bits']}-bit")
    lines.append(f"{'─' * 50}")
    lines.append(f"  Full model size:     {est['total_size_gb']:>6.1f} GB")
    lines.append(f"  With streaming:      {est['streaming_gb']:>6.1f} GB")
    lines.append(f"  With MLX-Flash:      {est['optimized_gb']:>6.1f} GB  ({est['savings_vs_full_pct']:.0f}% smaller)")
    lines.append(f"  KV cache (fp16):     {est['kv_cache_gb']:>6.1f} GB")
    lines.append(f"  KV cache (8-bit):    {est['kv_cache_8bit_gb']:>6.1f} GB")
    lines.append(f"{'─' * 50}")
    lines.append(f"  Your RAM: {est['ram_gb']}GB")
    lines.append(f"  Fits fully:     {'✓ YES' if est['fits_full'] else '✗ NO'}")
    lines.append(f"  Fits streaming: {'✓ YES' if est['fits_streaming'] else '✗ NO'}")
    lines.append(f"  Fits optimized: {'✓ YES' if est['fits_optimized'] else '✗ NO'}")
    lines.append(f"{'─' * 50}")
    lines.append(f"  → {est['recommendation']}")
    lines.append(f"{'═' * 50}")
    return "\n".join(lines)


def main():
    """CLI mode: interactive calculator."""
    print("\n  ⚡ MLX-Flash Model Calculator")
    print("  " + "═" * 40)

    print("\n  Known models:")
    for i, (name, info) in enumerate(KNOWN_MODELS.items(), 1):
        print(f"  {i}. {name} ({info['total_b']}B, {info['type']})")

    print(f"\n  Enter model number or name (or 'q' to quit):")

    while True:
        try:
            choice = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice.lower() in ('q', 'quit', 'exit'):
            break

        # Try as number
        model_name = None
        try:
            idx = int(choice) - 1
            names = list(KNOWN_MODELS.keys())
            if 0 <= idx < len(names):
                model_name = names[idx]
        except ValueError:
            pass

        # Try as name
        if model_name is None:
            for name in KNOWN_MODELS:
                if choice.lower() in name.lower():
                    model_name = name
                    break

        if model_name is None:
            print(f"  Unknown model: {choice}")
            continue

        # Ask for RAM
        try:
            ram_input = input("  Your RAM (GB, default 36): ").strip()
            ram_gb = float(ram_input) if ram_input else 36.0
        except ValueError:
            ram_gb = 36.0

        est = estimate_model(model_name=model_name, ram_gb=ram_gb)
        print(format_estimate(est))


if __name__ == "__main__":
    main()
