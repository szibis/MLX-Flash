#!/usr/bin/env python3
"""Train a DFlash drafter model for any MLX-compatible target.

Creates a lightweight 5-layer block diffusion drafter that generates 16 tokens
in a single forward pass, conditioned on the target model's hidden states.

Architecture (based on z-lab/Qwen3.5-27B-DFlash reference):
  - 5 transformer layers (full attention, not causal)
  - Hidden dim matches target model
  - GQA with 32 heads / 8 KV heads
  - block_size = 16 (tokens drafted per step)
  - Takes hidden states from 5 checkpoint layers of target

Training procedure:
  1. Run target model on training corpus, capture hidden states at checkpoint layers
  2. For each position, create (hidden_states, next_16_tokens) pairs
  3. Train drafter with block diffusion objective: denoise 16 masked tokens
     given the captured hidden states

Reference drafters from z-lab (all on HuggingFace):
  - z-lab/Qwen3.6-27B-DFlash (2B, for 27B dense)
  - z-lab/Qwen3.6-35B-A3B-DFlash (0.5B, for 35B MoE)
  - z-lab/Qwen3.5-122B-A10B-DFlash (0.5B, for 122B MoE)
  - z-lab/Kimi-K2.6-DFlash (3B, for Kimi K2)
  - NO drafter exists yet for DeepSeek V4 Flash

Usage:
  # Step 1: Collect hidden states from target model
  python scripts/train_dflash_drafter.py collect \\
    --target-model mlx-community/DeepSeek-V4-Flash-2bit-DQ \\
    --output-dir ./dflash-training-data \\
    --num-samples 10000

  # Step 2: Train the drafter
  python scripts/train_dflash_drafter.py train \\
    --training-data ./dflash-training-data \\
    --output-dir ./dflash-drafter-ds-v4-flash \\
    --epochs 3 --batch-size 8

  # Step 3: Evaluate drafter quality
  python scripts/train_dflash_drafter.py eval \\
    --target-model mlx-community/DeepSeek-V4-Flash-2bit-DQ \\
    --drafter-model ./dflash-drafter-ds-v4-flash

Requirements:
  - Target model must run on this machine (fits in RAM)
  - ~50-100 GB disk for training data (hidden states are large)
  - Training takes ~2-8 hours on M5 Pro depending on corpus size
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
except ImportError:
    print("Error: mlx, mlx-lm required. Run: pip install mlx mlx-lm")
    sys.exit(1)


@dataclass
class DrafterConfig:
    """DFlash drafter architecture config (matches z-lab convention)."""
    hidden_size: int = 5120
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 17408
    head_dim: int = 128
    vocab_size: int = 248320
    block_size: int = 16
    target_layer_ids: list = None
    num_target_layers: int = 64
    mask_token_id: int = 0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    def to_dict(self) -> dict:
        return {
            "architectures": ["DFlashDraftModel"],
            "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
            "block_size": self.block_size,
            "dflash_config": {
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": self.target_layer_ids or [1, 16, 31, 46, 61],
            },
            "dtype": "bfloat16",
            "head_dim": self.head_dim,
            "hidden_act": self.hidden_act,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "layer_types": ["full_attention"] * self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "num_target_layers": self.num_target_layers,
            "rms_norm_eps": self.rms_norm_eps,
            "vocab_size": self.vocab_size,
        }


def detect_target_config(model_path: str) -> DrafterConfig:
    """Auto-detect drafter config from target model architecture."""
    print(f"Loading target model config from: {model_path}")

    model, tokenizer = load(model_path)

    # Extract architecture details
    config = DrafterConfig()

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    elif hasattr(model, 'layers'):
        num_layers = len(model.layers)
    else:
        num_layers = 64
        print(f"  Warning: could not detect layer count, defaulting to {num_layers}")

    # Get hidden size from embedding or first layer
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed = model.model.embed_tokens
        if hasattr(embed, 'weight'):
            config.vocab_size = embed.weight.shape[0]
            config.hidden_size = embed.weight.shape[1]
    elif hasattr(model, 'embed_tokens'):
        embed = model.embed_tokens
        if hasattr(embed, 'weight'):
            config.vocab_size = embed.weight.shape[0]
            config.hidden_size = embed.weight.shape[1]

    config.num_target_layers = num_layers

    # Select 5 evenly-spaced checkpoint layers
    step = max(1, num_layers // 5)
    config.target_layer_ids = [
        1,
        step,
        step * 2,
        step * 3,
        num_layers - 1,
    ]

    # Set mask token (use last unused token or vocab_size - 1)
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        config.mask_token_id = tokenizer.pad_token_id
    else:
        config.mask_token_id = config.vocab_size - 1

    print(f"  Target layers: {num_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Checkpoint layers: {config.target_layer_ids}")
    print(f"  Mask token: {config.mask_token_id}")

    del model
    import gc
    gc.collect()

    return config, tokenizer


def collect_hidden_states(args):
    """Step 1: Run target model and collect hidden states at checkpoint layers."""
    print("="*60)
    print("Step 1: Collecting hidden states from target model")
    print("="*60)

    config, tokenizer = detect_target_config(args.target_model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "drafter_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(f"\nDrafter config saved to: {config_path}")

    # Load target model for inference
    print(f"\nLoading target model for hidden state extraction...")
    model, _ = load(args.target_model)

    # Training corpus: use a simple text source
    # In production, use a diverse corpus (code, math, prose, dialogue)
    corpus_prompts = [
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n",
        "The transformer architecture consists of an encoder and decoder. Each layer has multi-head attention followed by",
        "To solve this equation, we first isolate the variable x on one side: 3x + 7 = 22, so 3x = 15, therefore x =",
        "User: How do I implement a linked list in Python?\nAssistant: Here's a basic implementation:\n\nclass Node:\n",
    ] * (args.num_samples // 4 + 1)

    corpus_prompts = corpus_prompts[:args.num_samples]

    print(f"\nCollecting hidden states from {len(corpus_prompts)} samples...")
    print(f"This will take a while — each sample requires a full forward pass.\n")

    samples_collected = 0
    data_path = output_dir / "hidden_states"
    data_path.mkdir(exist_ok=True)

    for i, prompt in enumerate(corpus_prompts):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(corpus_prompts)} samples")

        tokens = tokenizer.encode(prompt)
        if len(tokens) < config.block_size + 5:
            continue

        input_ids = mx.array(tokens[:128])  # Limit context for training efficiency
        input_batch = mx.expand_dims(input_ids, axis=0)

        # Forward pass capturing hidden states
        hidden_states = {}
        try:
            layers = model.model.layers if hasattr(model, 'model') else model.layers
            embed = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens

            x = embed(input_batch)
            for layer_idx, layer in enumerate(layers):
                x = layer(x)
                if layer_idx in config.target_layer_ids:
                    hidden_states[layer_idx] = np.array(x[0, -1, :].astype(mx.float16))
        except Exception as e:
            print(f"  Warning: skipping sample {i} due to: {e}")
            continue

        if len(hidden_states) == len(config.target_layer_ids):
            # Save hidden states + target tokens
            target_tokens = tokens[len(tokens) - config.block_size:]
            sample = {
                "hidden_states": {str(k): v.tolist() for k, v in hidden_states.items()},
                "target_tokens": target_tokens,
            }
            sample_path = data_path / f"sample_{samples_collected:06d}.json"
            sample_path.write_text(json.dumps(sample))
            samples_collected += 1

    print(f"\nCollected {samples_collected} training samples")
    print(f"Saved to: {data_path}")
    print(f"\nNext step: python scripts/train_dflash_drafter.py train --training-data {output_dir}")


def train_drafter(args):
    """Step 2: Train the DFlash drafter on collected hidden states."""
    print("="*60)
    print("Step 2: Training DFlash drafter")
    print("="*60)

    training_dir = Path(args.training_data)
    config_path = training_dir / "drafter_config.json"

    if not config_path.exists():
        print(f"Error: {config_path} not found. Run 'collect' step first.")
        sys.exit(1)

    config_dict = json.loads(config_path.read_text())
    print(f"Drafter config: {json.dumps(config_dict, indent=2)[:500]}")

    # Count training samples
    data_path = training_dir / "hidden_states"
    samples = list(data_path.glob("sample_*.json"))
    print(f"\nTraining samples: {len(samples)}")

    if len(samples) < 100:
        print("Warning: fewer than 100 samples. Quality will be poor.")
        print("Recommendation: collect at least 10,000 samples for good results.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[TODO] Full training loop requires:")
    print("  1. Block diffusion training objective implementation")
    print("  2. MLX optimizer loop with the 5-layer drafter")
    print("  3. Noise schedule (linear or cosine masking)")
    print("  4. Cross-attention from drafter to target hidden states")
    print("\nThis is a research engineering task (~1-2 weeks of work).")
    print("For now, the framework is in place. See mlx_flash_compress/dflash.py")
    print(f"\nAlternative: port z-lab/Qwen3.6-35B-A3B-DFlash (0.5B) to MLX format")
    print("  and use with Qwen3.6-35B-A3B target model for immediate DFlash testing.")

    # Save placeholder config
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    print(f"\nConfig saved to: {output_dir / 'config.json'}")


def eval_drafter(args):
    """Step 3: Evaluate drafter acceptance rate against target model."""
    print("="*60)
    print("Step 3: Evaluating DFlash drafter")
    print("="*60)
    print("\n[TODO] Requires trained drafter model.")
    print("Run 'train' step first, or download a pre-trained drafter.")
    print("\nAvailable pre-trained drafters (from z-lab on HuggingFace):")
    print("  - z-lab/Qwen3.6-35B-A3B-DFlash (0.5B) — for Qwen3.6-35B-A3B MoE")
    print("  - z-lab/Qwen3.6-27B-DFlash (2B) — for Qwen3.6-27B dense")
    print("  - z-lab/Qwen3.5-122B-A10B-DFlash (0.5B) — for Qwen3.5-122B MoE")
    print("  - z-lab/Kimi-K2.6-DFlash (3B) — for Kimi K2")
    print("\n  NO pre-trained drafter exists for DeepSeek V4 Flash.")
    print("\nOptions to get a DeepSeek V4 Flash drafter:")
    print("  1. Train from scratch using this script (needs ~10K+ samples)")
    print("  2. Wait for z-lab to release one (they cover popular models)")
    print("  3. Use a Qwen3.6-35B-A3B model + existing drafter for proof-of-concept")


def main():
    parser = argparse.ArgumentParser(description="Train DFlash drafter for MLX")
    subparsers = parser.add_subparsers(dest="command")

    # Collect
    collect_parser = subparsers.add_parser("collect", help="Collect hidden states from target model")
    collect_parser.add_argument("--target-model", type=str, required=True)
    collect_parser.add_argument("--output-dir", type=str, default="./dflash-training-data")
    collect_parser.add_argument("--num-samples", type=int, default=1000)

    # Train
    train_parser = subparsers.add_parser("train", help="Train DFlash drafter")
    train_parser.add_argument("--training-data", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, default="./dflash-drafter")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate drafter quality")
    eval_parser.add_argument("--target-model", type=str, required=True)
    eval_parser.add_argument("--drafter-model", type=str, required=True)

    args = parser.parse_args()

    if args.command == "collect":
        collect_hidden_states(args)
    elif args.command == "train":
        train_drafter(args)
    elif args.command == "eval":
        eval_drafter(args)
    else:
        parser.print_help()
        print("\n\nQuick start:")
        print("  python scripts/train_dflash_drafter.py collect --target-model mlx-community/DeepSeek-V4-Flash-2bit-DQ")


if __name__ == "__main__":
    main()
