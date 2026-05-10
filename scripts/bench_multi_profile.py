#!/usr/bin/env python3
"""Profile multiple models and output a comparison matrix.

Downloads models, detects characteristics, measures AR baseline,
and recommends DFlash configuration for each.

Usage:
  python scripts/bench_multi_profile.py
  python scripts/bench_multi_profile.py --models Qwen3.5-27B gemma-4-31b
  python scripts/bench_multi_profile.py --max-ram 64
  python scripts/bench_multi_profile.py --cleanup
  python scripts/bench_multi_profile.py --list
"""

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_YAML = Path(__file__).parent / "models.yaml"


def load_model_registry():
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)


def get_system_ram_gb():
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024**3)
    except Exception:
        return 64


def cleanup_models(registry):
    """Remove all cached models listed in the registry."""
    cache_dir = os.path.expanduser(registry.get("cache_dir", "~/.cache/huggingface/hub"))
    removed = []
    for m in registry["models"]:
        model_id = m["id"]
        dir_name = "models--" + model_id.replace("/", "--")
        model_path = os.path.join(cache_dir, dir_name)
        if os.path.exists(model_path):
            size = sum(
                f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file()
            ) / (1024**3)
            removed.append((model_id, size))
            print(f"  Removing {model_id} ({size:.1f} GB)...")
            shutil.rmtree(model_path)
    if removed:
        total = sum(s for _, s in removed)
        print(f"\nRemoved {len(removed)} models, freed {total:.1f} GB")
    else:
        print("No cached models found to remove.")


def list_models(registry):
    """List all models in the registry with cache status."""
    cache_dir = os.path.expanduser(registry.get("cache_dir", "~/.cache/huggingface/hub"))
    print(f"{'Model':60s} {'Size':>8s} {'Cached':>8s} {'Category'}")
    print("-" * 100)
    for m in registry["models"]:
        model_id = m["id"]
        dir_name = "models--" + model_id.replace("/", "--")
        model_path = os.path.join(cache_dir, dir_name)
        cached = "yes" if os.path.exists(model_path) else "no"
        if cached == "yes":
            size = sum(
                f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file()
            ) / (1024**3)
            size_str = f"{size:.1f} GB"
        else:
            size_str = f"~{m.get('size_gb_approx', '?')} GB"
        print(f"  {model_id:58s} {size_str:>8s} {cached:>8s} {m.get('category_expected', '?')}")


def profile_model(model_id, prompt, max_tokens, priority="auto"):
    """Profile a single model: load, detect, measure AR, recommend profile."""
    import mlx.core as mx

    from mlx_flash_compress.dflash_profile import (
        detect_model, measure_ar_baseline, select_profile,
    )

    print(f"\n  Loading {model_id}...")
    t0 = time.perf_counter()
    from mlx_lm import load
    model, tokenizer = load(model_id)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    print(f"  Detecting model characteristics...")
    profile = detect_model(model, tokenizer)

    print(f"  Measuring AR baseline ({max_tokens} tokens)...")
    ar_tok_s = measure_ar_baseline(model, tokenizer, prompt, max_tokens)
    profile.ar_tok_s = round(ar_tok_s, 1)

    recommended = select_profile(profile, priority)

    result = {
        "model_id": model_id,
        "category": profile.category,
        "total_params_b": profile.total_params_b,
        "active_params_b": profile.active_params_b,
        "num_layers": profile.num_layers,
        "num_ssm_layers": profile.num_ssm_layers,
        "num_attn_layers": profile.num_attn_layers,
        "is_moe": profile.is_moe,
        "is_quantized": profile.is_quantized,
        "quant_bits": profile.quant_bits,
        "ssm_ratio": profile.ssm_ratio,
        "ar_tok_s": profile.ar_tok_s,
        "recommendation": profile.recommendation,
        "profile_name": recommended.name,
        "profile_desc": recommended.description,
        "load_time": round(load_time, 1),
    }

    del model, tokenizer
    gc.collect()
    clear = getattr(mx, 'clear_cache', None) or getattr(mx.metal, 'clear_cache', None)
    if clear:
        clear()

    return result


def print_summary(results):
    """Print the comparison matrix."""
    print("\n" + "=" * 120)
    print("MODEL PROFILING MATRIX")
    print("=" * 120)

    header = (f"  {'Model':40s} | {'Category':12s} | {'Params':>7s} | {'Active':>7s} | "
              f"{'Layers':>8s} | {'SSM%':>5s} | {'Quant':>5s} | {'AR tok/s':>9s} | "
              f"{'DFlash':>10s} | Profile")
    print(header)
    print("-" * 120)

    for r in results:
        model_short = r["model_id"].split("/")[-1]
        if len(model_short) > 40:
            model_short = model_short[:37] + "..."

        layers_str = f"{r['num_layers']}"
        if r["num_ssm_layers"] > 0:
            layers_str += f" ({r['num_ssm_layers']}S)"

        quant_str = f"{r['quant_bits']}bit" if r["is_quantized"] else "bf16"
        moe_marker = " MoE" if r["is_moe"] else ""

        print(f"  {model_short:40s} | {r['category']:12s} | "
              f"{r['total_params_b']:6.1f}B | {r['active_params_b']:6.1f}B | "
              f"{layers_str:>8s} | {r['ssm_ratio']:4.0%} | {quant_str:>5s} | "
              f"{r['ar_tok_s']:8.1f} | "
              f"{r['recommendation']:>10s} | {r['profile_name']}")

    print("=" * 120)

    print("\n  DFlash value legend:")
    print("    recommended  — AR < 25 tok/s, DFlash should provide significant speedup")
    print("    marginal     — AR 25-50 tok/s, DFlash may match AR but unlikely to beat it")
    print("    skip         — AR > 50 tok/s, AR is fast enough, DFlash adds overhead")

    print("\n  Profile descriptions:")
    seen = set()
    for r in results:
        if r["profile_name"] not in seen:
            seen.add(r["profile_name"])
            print(f"    {r['profile_name']:15s} — {r['profile_desc']}")


def main():
    parser = argparse.ArgumentParser(description="Profile multiple models for DFlash")
    parser.add_argument("--models", nargs="*",
                        help="Filter models by substring match (e.g. 'Qwen3.5' 'gemma')")
    parser.add_argument("--max-ram", type=int, default=None,
                        help="Max RAM in GB (auto-detected if not set)")
    parser.add_argument("--prompt", default="def binary_search(arr, target):\n    ")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--priority", choices=["auto", "quality", "speed", "balanced"],
                        default="auto", help="Profile priority (default: auto)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove all cached models from the registry")
    parser.add_argument("--list", action="store_true",
                        help="List all models with cache status")
    args = parser.parse_args()

    registry = load_model_registry()

    if args.cleanup:
        cleanup_models(registry)
        return

    if args.list:
        list_models(registry)
        return

    ram_gb = args.max_ram or get_system_ram_gb()
    print(f"System RAM: {ram_gb:.0f} GB")

    models_to_test = []
    for m in registry["models"]:
        if m.get("skip"):
            continue

        if m.get("skip_on_small_ram") and ram_gb < 128:
            print(f"  Skipping {m['id']} (needs >128GB RAM, have {ram_gb:.0f}GB)")
            continue

        approx_size = m.get("size_gb_approx", 0)
        if approx_size > ram_gb * 0.85:
            print(f"  Skipping {m['id']} (~{approx_size}GB > {ram_gb*0.85:.0f}GB limit)")
            continue

        if args.models:
            if not any(f.lower() in m["id"].lower() for f in args.models):
                continue

        models_to_test.append(m)

    print(f"\nWill profile {len(models_to_test)} models:")
    for m in models_to_test:
        print(f"  - {m['id']} (~{m.get('size_gb_approx', '?')} GB)")

    results = []
    for i, m in enumerate(models_to_test):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(models_to_test)}] {m['id']}")
        print(f"{'='*80}")

        try:
            r = profile_model(m["id"], args.prompt, args.max_tokens, args.priority)
            r["expected_category"] = m.get("category_expected", "unknown")
            results.append(r)
            print(f"  ✓ {r['category']} | AR={r['ar_tok_s']} tok/s | "
                  f"DFlash={r['recommendation']} | Profile={r['profile_name']}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({
                "model_id": m["id"],
                "category": "error",
                "total_params_b": 0,
                "active_params_b": 0,
                "num_layers": 0,
                "num_ssm_layers": 0,
                "num_attn_layers": 0,
                "is_moe": False,
                "is_quantized": False,
                "quant_bits": None,
                "ssm_ratio": 0,
                "ar_tok_s": 0,
                "recommendation": "error",
                "profile_name": "error",
                "profile_desc": str(e)[:80],
                "load_time": 0,
            })

    print_summary(results)

    # Check profile accuracy
    print("\n  Profile detection accuracy:")
    correct = 0
    total = 0
    for r in results:
        if r["category"] == "error":
            continue
        expected = r.get("expected_category", "unknown")
        actual = r["category"]
        match = "✓" if expected == actual else "✗"
        if expected == actual:
            correct += 1
        total += 1
        print(f"    {match} {r['model_id'].split('/')[-1]:40s} expected={expected:15s} actual={actual}")
    if total > 0:
        print(f"    Accuracy: {correct}/{total} ({correct/total:.0%})")


if __name__ == "__main__":
    main()
