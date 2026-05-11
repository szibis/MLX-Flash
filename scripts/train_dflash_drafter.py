#!/usr/bin/env python3
"""Train a DFlash drafter model on Apple Silicon.

Implements the full training pipeline from the DFlash paper (arXiv:2602.06036)
with key tricks:
  - Exponential loss decay: w_k = exp(-(k-1)/gamma), early positions matter more
  - Random anchor sampling: block starts from random positions, matches inference
  - Target-generated training data: train on target model's own outputs
  - BD3-LMs clipped noise schedule: t ~ U[0.3, 0.8] for block_size=16

Usage:
  # Collect hidden states from target model
  python scripts/train_dflash_drafter.py collect \
    --target-model mlx-community/Qwen3.6-35B-A3B-4bit \
    --output-dir ./dflash-training-data \
    --num-samples 500

  # Train drafter from scratch
  python scripts/train_dflash_drafter.py train \
    --training-data ./dflash-training-data \
    --output-dir ./dflash-drafter \
    --steps 5000

  # Fine-tune fc projection only (calibration for quantized targets)
  python scripts/train_dflash_drafter.py train \
    --training-data ./dflash-training-data \
    --pretrained z-lab/Qwen3.6-35B-A3B-DFlash \
    --fc-only \
    --output-dir ./calibrated-drafter \
    --steps 500

  # Evaluate drafter acceptance rate
  python scripts/train_dflash_drafter.py eval \
    --target-model mlx-community/Qwen3.6-35B-A3B-4bit \
    --drafter-model ./dflash-drafter
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
except ImportError:
    print("Error: pip install mlx mlx-lm")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner


def select_checkpoint_layers(num_layers: int, n: int = 5) -> list[int]:
    start, end = 1, num_layers - 2
    step = (end - start) / (n - 1)
    return [int(start + i * step) for i in range(n)]


# Preset configurations for known target models
DRAFTER_PRESETS = {
    "deepseek-v4-flash": {
        "target_2bit": "mlx-community/DeepSeek-V4-Flash-2bit-DQ",
        "target_4bit": "mlx-community/DeepSeek-V4-Flash-4bit",
        "drafter_layers": 5,
        "drafter_hidden": 2048,
        "block_size": 16,
        "recommended_steps": 10000,
        "recommended_samples": 1000,
        "recommended_lr": 3e-4,
        "notes": "284B total / 13B active MoE. Use 2-bit on 64GB, 4-bit on 128GB+.",
    },
    "deepseek-v4-flash-mini": {
        "target_2bit": "mlx-community/DeepSeek-V4-Flash-2bit-DQ",
        "target_4bit": "mlx-community/DeepSeek-V4-Flash-4bit",
        "drafter_layers": 3,
        "drafter_hidden": 1024,
        "block_size": 16,
        "recommended_steps": 8000,
        "recommended_samples": 800,
        "recommended_lr": 4e-4,
        "notes": "Mini drafter (3 layers, 1024 hidden) for RAM-constrained setups. "
        "Trades ~5-10% acceptance for ~60% less drafter memory.",
    },
    "qwen3.6-35b-a3b": {
        "target_4bit": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "drafter_hub": "z-lab/Qwen3.6-35B-A3B-DFlash",
        "drafter_layers": 8,
        "drafter_hidden": 2048,
        "block_size": 16,
        "recommended_steps": 5000,
        "recommended_samples": 500,
        "recommended_lr": 6e-4,
        "notes": "35B hybrid SSM+attention MoE, 3B active. z-lab drafter available.",
    },
}


def _get_available_ram_gb() -> float:
    """Return available system RAM in GB (best-effort)."""
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass
    # Fallback: try psutil
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    return 0.0


def auto_select_preset(target_model: str | None = None) -> str:
    """Automatically select the best drafter preset based on available RAM.

    Heuristic:
    - If target is DeepSeek V4 Flash:
      - >= 96 GB RAM: deepseek-v4-flash (full drafter)
      - < 96 GB RAM: deepseek-v4-flash-mini (smaller drafter)
    - Otherwise defaults to qwen3.6-35b-a3b

    Args:
        target_model: Optional target model path/name to help select.

    Returns:
        Preset name string.
    """
    ram_gb = _get_available_ram_gb()

    # Detect DeepSeek V4 Flash from target model name
    is_deepseek_v4 = False
    if target_model:
        lower = target_model.lower()
        if "deepseek" in lower and ("v4" in lower or "flash" in lower):
            is_deepseek_v4 = True

    if is_deepseek_v4:
        if ram_gb >= 96:
            return "deepseek-v4-flash"
        else:
            return "deepseek-v4-flash-mini"

    # Default
    return "qwen3.6-35b-a3b"


def detect_target(model_path: str):
    """Load target model and detect architecture."""
    print(f"Loading target model: {model_path}")
    model, tokenizer = load(model_path)

    candidates = []
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        candidates.append((model.language_model.model, model.language_model))
    if hasattr(model, "model"):
        candidates.append((model.model, model))
    candidates.append((model, model))

    for inner, outer in candidates:
        if hasattr(inner, "embed_tokens") and hasattr(inner, "layers"):
            embed = inner.embed_tokens
            layers = inner.layers
            norm = getattr(inner, "norm", None)
            lm_head = getattr(outer, "lm_head", None)
            num_layers = len(layers)

            # Detect hidden size from embedding output
            test = embed(mx.array([[0]]))
            mx.eval(test)
            hidden_size = test.shape[-1]

            checkpoint_ids = select_checkpoint_layers(num_layers)
            vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else embed.weight.shape[0]

            print(f"  {num_layers} layers, hidden={hidden_size}, vocab={vocab_size}")
            print(f"  Checkpoint layers: {checkpoint_ids}")
            return (
                model,
                tokenizer,
                embed,
                layers,
                norm,
                lm_head,
                {
                    "num_layers": num_layers,
                    "hidden_size": hidden_size,
                    "vocab_size": vocab_size,
                    "checkpoint_ids": checkpoint_ids,
                },
            )

    raise RuntimeError("Cannot detect target model architecture")


def extract_hidden_and_tokens(embed, layers, input_ids, checkpoint_ids):
    """Run target forward pass, return checkpoint hidden states."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    h = embed(input_ids)
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)

    checkpoints = []
    for i, layer in enumerate(layers):
        is_linear = getattr(layer, "is_linear", False)
        mask = ssm_mask if is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        if i in checkpoint_ids:
            checkpoints.append(h)

    return mx.concatenate(checkpoints, axis=-1)


def collect_hidden_states(args):
    """Collect hidden states + token sequences from target model."""
    print("=" * 60)
    print("Step 1: Collecting training data from target model")
    print("=" * 60)

    model, tokenizer, embed, layers, norm, lm_head, info = detect_target(args.target_model)
    if args.checkpoint_layers:
        checkpoint_ids = [int(x) for x in args.checkpoint_layers.split(",")]
        print(f"  Using explicit checkpoint layers: {checkpoint_ids}")
    else:
        checkpoint_ids = info["checkpoint_ids"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model info
    model_info = {
        "target_model": args.target_model,
        "num_target_layers": info["num_layers"],
        "hidden_size": info["hidden_size"],
        "vocab_size": info["vocab_size"],
        "checkpoint_ids": checkpoint_ids,
    }
    (output_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))

    # Load calibration prompts from file if provided, otherwise use built-in defaults
    if getattr(args, "calibration_prompts", None):
        prompts_path = Path(args.calibration_prompts)
        if not prompts_path.exists():
            print(f"Error: calibration prompts file not found: {prompts_path}")
            sys.exit(1)
        prompts = []
        with open(prompts_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Support {"prompt": "..."} or {"text": "..."} or plain string
                    if isinstance(data, dict):
                        prompt_text = data.get("prompt") or data.get("text") or ""
                    elif isinstance(data, str):
                        prompt_text = data
                    else:
                        continue
                    if prompt_text:
                        prompts.append(prompt_text)
                except json.JSONDecodeError:
                    # Treat as plain text line
                    prompts.append(line)
        print(f"  Loaded {len(prompts)} calibration prompts from {prompts_path}")
        if not prompts:
            print("Error: no valid prompts found in calibration file")
            sys.exit(1)
    else:
        # Diverse prompts covering code, math, reasoning, prose, dialogue
        prompts = [
            # Code (Python, JS, Rust, SQL, shell)
            "def fibonacci(n):\n    ",
            "import torch\nimport torch.nn as nn\n\nclass",
            "async def fetch_data(url: str) -> dict:\n    ",
            "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n",
            "class BinaryTree:\n    def __init__(self, value):\n",
            "SELECT u.name, COUNT(o.id) FROM users u JOIN",
            "function debounce(fn, delay) {\n  let timer;\n  return function(",
            "fn main() -> Result<(), Box<dyn std::error::Error>> {\n    let client =",
            "#!/bin/bash\nset -euo pipefail\n\nfor file in",
            "def train_step(model, optimizer, batch):\n    optimizer.zero_grad()\n",
            "class LRUCache:\n    def __init__(self, capacity: int):\n",
            "@app.route('/api/users', methods=['POST'])\ndef create_user():\n",
            # Math & reasoning
            "To solve the quadratic equation ax^2 + bx + c = 0,",
            "Given a matrix A of size m x n, the transpose A^T is",
            "The gradient descent algorithm works by iteratively",
            "Prove by induction that the sum of the first n natural numbers is",
            "The eigenvalues of a 2x2 matrix can be found by solving",
            "Using the chain rule, the derivative of f(g(x)) is",
            # Technical prose
            "The transformer architecture consists of an encoder and decoder, where",
            "The attention mechanism allows the model to focus on",
            "In Python, list comprehensions provide a concise way to",
            "The CAP theorem states that a distributed system cannot simultaneously",
            "Garbage collection in modern languages uses either reference counting or",
            "The difference between TCP and UDP is that TCP provides",
            # Dialogue & instruction following
            "User: How do I implement a binary search?\nAssistant:",
            "Explain the difference between a stack and a queue to",
            "Write a Python function that takes a list of integers and returns",
            "Summarize the key advantages of using a hash table for",
            # Structured output
            '{"name": "John", "age": 30, "skills": [',
            "# README\n\n## Installation\n\n```bash\npip install",
            "| Column A | Column B | Column C |\n|----------|----------|----------|\n|",
        ]

    from mlx_lm import generate

    all_sequences = []
    print(f"\nGenerating {args.num_samples} training sequences...")
    for i in range(args.num_samples):
        prompt = prompts[i % len(prompts)]
        try:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
            tokens = tokenizer.encode(prompt + output)
            if len(tokens) >= 48:
                all_sequences.append(tokens)
        except Exception as e:
            print(f"  Warning: sample {i} failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{args.num_samples} sequences")

    print(f"\nCollected {len(all_sequences)} sequences, {sum(len(s) for s in all_sequences)} total tokens")

    # Extract hidden states for each sequence
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"\nExtracting hidden states from {len(all_sequences)} sequences...")
    saved = 0
    for i, tokens in enumerate(all_sequences):
        input_ids = mx.array([tokens])
        try:
            hidden = extract_hidden_and_tokens(embed, layers, input_ids, checkpoint_ids)
            mx.eval(hidden)

            np.savez_compressed(
                str(data_dir / f"sample_{saved:06d}.npz"),
                hidden=np.array(hidden[0].astype(mx.float16)),
                tokens=np.array(tokens, dtype=np.int32),
            )
            saved += 1
        except Exception as e:
            print(f"  Warning: sample {i} extraction failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(all_sequences)}")

    print(f"\nSaved {saved} training samples to {data_dir}")
    print(f"Next: python scripts/train_dflash_drafter.py train --training-data {output_dir}")


def train_drafter(args):
    """Train or fine-tune a DFlash drafter."""
    print("=" * 60)
    print("Step 2: Training DFlash drafter")
    print("=" * 60)

    training_dir = Path(args.training_data)
    info_path = training_dir / "model_info.json"
    if not info_path.exists():
        print(f"Error: {info_path} not found. Run 'collect' step first.")
        sys.exit(1)

    model_info = json.loads(info_path.read_text())
    data_dir = training_dir / "data"
    sample_files = sorted(data_dir.glob("sample_*.npz"))
    print(f"Training samples: {len(sample_files)}")

    if not sample_files:
        print("Error: no training samples found. Run 'collect' step first.")
        sys.exit(1)

    # Load target model (needed for embed_tokens and lm_head)
    print(f"\nLoading target model: {model_info['target_model']}")
    target_model, tokenizer, embed, layers, norm, lm_head, _ = detect_target(model_info["target_model"])
    target_model.freeze()

    checkpoint_ids = model_info["checkpoint_ids"]
    hidden_size = model_info["hidden_size"]
    block_size = args.block_size

    # Create or load drafter
    if args.pretrained:
        print(f"Loading pre-trained drafter: {args.pretrained}")
        from huggingface_hub import snapshot_download

        drafter_path = snapshot_download(args.pretrained)
        drafter, config = DFlashDraftModel.from_pretrained(drafter_path)
    else:
        config = DFlashModelConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 3,
            num_hidden_layers=args.num_layers,
            num_attention_heads=max(4, hidden_size // 128),
            num_key_value_heads=max(2, hidden_size // 512),
            head_dim=128,
            vocab_size=model_info["vocab_size"],
            block_size=block_size,
            mask_token_id=0,
            target_layer_ids=checkpoint_ids,
            num_target_layers=model_info["num_target_layers"],
        )
        drafter = DFlashDraftModel(config)

    if args.fc_only:
        drafter.freeze()
        drafter.fc.unfreeze()
        drafter.hidden_norm.unfreeze()
        print("  FC-only mode: training fc + hidden_norm only")

    trainable = sum(p.size for _, p in tree_flatten(drafter.trainable_parameters()))
    print(f"  Trainable parameters: {trainable:,}")

    # Loss weights: exponential decay (DFlash paper)
    gamma = args.loss_gamma
    loss_weights = mx.array([math.exp(-k / gamma) for k in range(block_size - 1)])

    # Cosine LR schedule with warmup
    warmup_steps = min(args.warmup_steps, args.steps // 5)
    lr_schedule = optim.schedulers.join_schedules(
        [
            optim.schedulers.linear_schedule(1e-7, args.lr, warmup_steps),
            optim.schedulers.cosine_decay(args.lr, args.steps - warmup_steps, end=args.lr * 0.1),
        ],
        [warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    # Pre-load all samples into memory
    print("\nLoading training data into memory...")
    all_samples = []
    for f in sample_files:
        d = np.load(str(f))
        all_samples.append((d["hidden"], d["tokens"]))

    # Validation split (10%)
    random.shuffle(all_samples)
    n_val = max(1, len(all_samples) // 10)
    val_samples = all_samples[:n_val]
    samples = all_samples[n_val:]
    print(f"  Loaded {len(samples)} train + {n_val} validation samples")

    def compute_loss(drafter_model, noise_emb, target_hidden, target_ids):
        refined = drafter_model(noise_emb, target_hidden)
        logits = lm_head(refined)
        mask_logits = logits[:, 1:, :]
        B, N, V = mask_logits.shape
        ce = nn.losses.cross_entropy(
            mask_logits.reshape(B * N, V),
            target_ids.reshape(B * N),
            reduction="none",
        ).reshape(B, N)
        return mx.mean(ce * loss_weights[None, :N])

    loss_and_grad = nn.value_and_grad(drafter, compute_loss)

    def _sample_batch(sample_list, bs):
        hidden_np, tokens_np = random.choice(sample_list)
        seq_len = len(tokens_np)
        if seq_len < bs + 2:
            return None
        block_start = random.randint(1, seq_len - bs - 1)
        target_hidden = mx.array(hidden_np[None, :, :])
        anchor_id = int(tokens_np[block_start - 1])
        block_ids = [anchor_id] + [config.mask_token_id] * (bs - 1)
        noise_emb = embed(mx.array([block_ids]))
        target_ids = mx.array([tokens_np[block_start : block_start + bs - 1].tolist()])
        return noise_emb, target_hidden, target_ids

    def _eval_val_loss():
        val_losses = []
        for vh, vt in val_samples[:20]:
            seq_len = len(vt)
            if seq_len < block_size + 2:
                continue
            block_start = seq_len // 2
            th = mx.array(vh[None, :, :])
            anchor_id = int(vt[block_start - 1])
            bids = [anchor_id] + [config.mask_token_id] * (block_size - 1)
            ne = embed(mx.array([bids]))
            ti = mx.array([vt[block_start : block_start + block_size - 1].tolist()])
            vl = compute_loss(drafter, ne, th, ti)
            mx.eval(vl)
            val_losses.append(vl.item())
        return sum(val_losses) / max(1, len(val_losses))

    # Training loop
    print(f"\nTraining: {args.steps} steps, lr={args.lr}, block_size={block_size}")
    print(f"  Loss decay gamma={gamma}, fc_only={args.fc_only}, warmup={warmup_steps}")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    best_val_loss = float("inf")
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        batch = _sample_batch(samples, block_size)
        if batch is None:
            continue
        noise_emb, target_hidden, target_ids = batch

        loss, grads = loss_and_grad(drafter, noise_emb, target_hidden, target_ids)
        optimizer.update(drafter, grads)
        mx.eval(drafter.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % args.log_every == 0:
            window = min(len(losses), args.log_every)
            avg = sum(losses[-window:]) / window
            elapsed = time.perf_counter() - t_start
            sps = step / elapsed
            current_lr = lr_schedule(step) if callable(lr_schedule) else args.lr
            print(
                f"  Step {step:5d}/{args.steps} | loss={avg:.4f} | "
                f"lr={current_lr:.2e} | {sps:.1f} steps/s | {elapsed:.0f}s"
            )

        if step % args.save_every == 0:
            val_loss = _eval_val_loss()
            improved = " *BEST*" if val_loss < best_val_loss else ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = output_dir / "best"
                best_ckpt.mkdir(exist_ok=True)
                flat = dict(tree_flatten(drafter.parameters()))
                mx.save_safetensors(str(best_ckpt / "model.safetensors"), flat)
            print(f"    val_loss={val_loss:.4f}{improved}")

            ckpt = output_dir / f"checkpoint-{step}"
            ckpt.mkdir(exist_ok=True)
            flat = dict(tree_flatten(drafter.parameters()))
            mx.save_safetensors(str(ckpt / "model.safetensors"), flat)

    # Save final model (all parameters, not just trainable)
    print(f"\nSaving to {output_dir}")
    flat = dict(tree_flatten(drafter.parameters()))
    mx.save_safetensors(str(output_dir / "model.safetensors"), flat)

    config_data = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "dflash_config": {
            "mask_token_id": config.mask_token_id,
            "target_layer_ids": config.target_layer_ids,
        },
        "num_target_layers": config.num_target_layers,
        "rope_theta": config.rope_theta,
    }
    (output_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    final_loss = sum(losses[-10:]) / min(len(losses), 10) if losses else 0
    elapsed = time.perf_counter() - t_start
    print(f"Done: {args.steps} steps in {elapsed:.0f}s, final loss={final_loss:.4f}")


def eval_drafter(args):
    """Evaluate drafter acceptance rate against target model."""
    print("=" * 60)
    print("Step 3: Evaluating DFlash drafter")
    print("=" * 60)

    model, tokenizer, _, _, _, _, _ = detect_target(args.target_model)

    print(f"\nLoading drafter: {args.drafter_model}")
    drafter_path = Path(args.drafter_model)
    if not drafter_path.exists():
        from huggingface_hub import snapshot_download

        drafter_path = Path(snapshot_download(args.drafter_model))

    drafter, config = DFlashDraftModel.from_pretrained(str(drafter_path))
    runner = DFlashRunner(model, tokenizer, drafter, config)

    prompts = [
        "def binary_search(arr, target):\n    ",
        "The transformer architecture consists of",
        "To solve 3x + 7 = 22, first subtract 7:",
    ]

    max_tokens = args.max_tokens

    for mode_name, use_cache in [("no-cache", False), ("cached", True)]:
        print(f"\n--- Eval: {mode_name} (max_tokens={max_tokens}) ---")
        for prompt in prompts:
            text, stats = runner.generate(prompt, max_tokens=max_tokens, use_cache=use_cache)
            cached_label = " [cached]" if stats.get("cached") else ""
            print(f"  {prompt[:40]}...{cached_label}")
            print(
                f"    accept={stats['acceptance_rate']:.1%} | "
                f"tok/step={stats['tokens_per_step']:.1f} | "
                f"{stats['tok_per_sec']:.1f} tok/s"
            )


def show_preset(args):
    """Show recommended training configuration for a target model."""
    if args.name is None:
        print("Available presets:")
        for name, cfg in DRAFTER_PRESETS.items():
            print(f"\n  {name}:")
            print(f"    {cfg['notes']}")
        print("\n  auto:")
        print("    Automatically selects preset based on available RAM and target model.")
        print("\nUsage: python scripts/train_dflash_drafter.py preset <name>")
        return

    if args.name == "auto":
        target = getattr(args, "target_model", None)
        selected = auto_select_preset(target)
        ram_gb = _get_available_ram_gb()
        print(f"Auto-selected preset: {selected}")
        print(f"  Available RAM: {ram_gb:.0f} GB")
        if target:
            print(f"  Target model: {target}")
        print(f"\nShowing configuration for '{selected}':\n")
        args.name = selected

    cfg = DRAFTER_PRESETS[args.name]
    target_key = f"target_{args.quant}"
    target = cfg.get(target_key, cfg.get("target_4bit", "UNKNOWN"))

    print("=" * 60)
    print(f"DFlash Drafter Training Preset: {args.name}")
    print("=" * 60)
    print(f"\n  Target model: {target}")
    print(f"  Drafter: {cfg['drafter_layers']} layers, hidden={cfg['drafter_hidden']}")
    print(f"  Block size: {cfg['block_size']}")
    print(
        f"  Recommended: {cfg['recommended_samples']} samples, "
        f"{cfg['recommended_steps']} steps, lr={cfg['recommended_lr']}"
    )
    print(f"  Notes: {cfg['notes']}")

    if cfg.get("drafter_hub"):
        print(f"\n  Pre-trained drafter available: {cfg['drafter_hub']}")

    data_dir = f"./{args.name}-training-data"
    out_dir = f"./{args.name}-drafter"

    print("\n--- Copy-paste commands ---\n")
    print(f"# Step 1: Collect training data ({cfg['recommended_samples']} samples)")
    print("python scripts/train_dflash_drafter.py collect \\")
    print(f"  --target-model {target} \\")
    print(f"  --output-dir {data_dir} \\")
    print(f"  --num-samples {cfg['recommended_samples']}")

    print("\n# Step 2: Train drafter from scratch")
    print("python scripts/train_dflash_drafter.py train \\")
    print(f"  --training-data {data_dir} \\")
    print(f"  --output-dir {out_dir} \\")
    print(f"  --num-layers {cfg['drafter_layers']} \\")
    print(f"  --lr {cfg['recommended_lr']} \\")
    print(f"  --steps {cfg['recommended_steps']}")

    if cfg.get("drafter_hub"):
        print("\n# Alternative: FC-only calibration of existing drafter")
        print("python scripts/train_dflash_drafter.py train \\")
        print(f"  --training-data {data_dir} \\")
        print(f"  --pretrained {cfg['drafter_hub']} \\")
        print("  --fc-only \\")
        print(f"  --output-dir {out_dir}-calibrated \\")
        print("  --steps 2000 --lr 1e-4")

    print("\n# Step 3: Evaluate")
    print("python scripts/train_dflash_drafter.py eval \\")
    print(f"  --target-model {target} \\")
    print(f"  --drafter-model {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train DFlash drafter on Apple Silicon")
    subparsers = parser.add_subparsers(dest="command")

    collect_p = subparsers.add_parser("collect", help="Collect hidden states from target")
    collect_p.add_argument("--target-model", type=str, required=True)
    collect_p.add_argument("--output-dir", type=str, default="./dflash-training-data")
    collect_p.add_argument("--num-samples", type=int, default=500)
    collect_p.add_argument(
        "--checkpoint-layers", type=str, default=None, help="Comma-separated checkpoint layer IDs (e.g. 1,10,19,28,37)"
    )
    collect_p.add_argument(
        "--calibration-prompts",
        type=str,
        default=None,
        help="Path to a .jsonl file of custom prompts for calibration data collection. "
        'Each line: {"prompt": "..."} or plain text. If omitted, uses built-in diverse prompts.',
    )

    train_p = subparsers.add_parser("train", help="Train DFlash drafter")
    train_p.add_argument("--training-data", type=str, required=True)
    train_p.add_argument("--pretrained", type=str, default=None, help="Pre-trained drafter to fine-tune")
    train_p.add_argument("--output-dir", type=str, default="./dflash-drafter")
    train_p.add_argument("--steps", type=int, default=5000)
    train_p.add_argument("--lr", type=float, default=6e-4)
    train_p.add_argument("--block-size", type=int, default=16)
    train_p.add_argument("--num-layers", type=int, default=8)
    train_p.add_argument("--fc-only", action="store_true", help="Only train fc + hidden_norm (calibration)")
    train_p.add_argument("--loss-gamma", type=float, default=7.0, help="Exponential loss decay gamma")
    train_p.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps (default: 100)")
    train_p.add_argument("--save-every", type=int, default=1000)
    train_p.add_argument("--log-every", type=int, default=10)

    eval_p = subparsers.add_parser("eval", help="Evaluate drafter acceptance rate")
    eval_p.add_argument("--target-model", type=str, required=True)
    eval_p.add_argument("--drafter-model", type=str, required=True)
    eval_p.add_argument("--max-tokens", type=int, default=64)

    preset_p = subparsers.add_parser("preset", help="Show recommended config for a target model")
    preset_p.add_argument(
        "name",
        type=str,
        nargs="?",
        default=None,
        choices=list(DRAFTER_PRESETS.keys()) + ["auto"],
        help="Preset name (omit to list all, 'auto' to auto-select based on RAM)",
    )
    preset_p.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Target model name (used by 'auto' to pick the best preset)",
    )
    preset_p.add_argument("--quant", type=str, default="4bit", choices=["2bit", "4bit"], help="Quantization level")

    args = parser.parse_args()

    if args.command == "collect":
        collect_hidden_states(args)
    elif args.command == "train":
        train_drafter(args)
    elif args.command == "eval":
        eval_drafter(args)
    elif args.command == "preset":
        show_preset(args)
    else:
        parser.print_help()
        print("\n\nQuick start:")
        print("  python scripts/train_dflash_drafter.py preset deepseek-v4-flash")
        print("  python scripts/train_dflash_drafter.py collect --target-model mlx-community/Qwen3.6-35B-A3B-4bit")


if __name__ == "__main__":
    main()
