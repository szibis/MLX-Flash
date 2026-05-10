# DFlash Drafter Training Guide

How to train a custom DFlash block diffusion drafter for any target model on Apple Silicon.

## Overview

A DFlash drafter is a small (5-8 layer) transformer that predicts the next block_size tokens in a single forward pass, conditioned on hidden states from the target model. Training involves:

1. Collecting hidden states from the target model on diverse text
2. Training the drafter to predict masked tokens given those hidden states
3. Evaluating acceptance rate against the target model

## Architecture

### Drafter Model Structure

```
DFlashDraftModel:
  fc: Linear(num_checkpoints * hidden_size → hidden_size, bias=False)
  hidden_norm: RMSNorm(hidden_size)
  layers: [DFlashDecoderLayer × num_layers]
  norm: RMSNorm(hidden_size)
  # No embed_tokens or lm_head — shared with target model
```

Each `DFlashDecoderLayer`:
```
  input_layernorm → DFlashAttention → residual
  post_attention_layernorm → SiLU MLP → residual
```

`DFlashAttention` (cross-attention):
- Q: from draft hidden states
- K/V: from concat(target_hidden_states, draft_hidden_states)
- Bidirectional (is_causal=False)
- GQA with QK-norm and RoPE

### Hyperparameters by Target Model Size

| Target Model | Drafter Layers | Hidden Size | Checkpoint Layers | Block Size | Drafter Size |
|---|---:|---:|---|---:|---:|
| 7-8B dense | 5 | match target | 5 uniform | 16 | ~200 MB |
| 30-35B MoE | 8 | match target | 5 uniform | 16 | ~950 MB |
| 70B dense | 5 | match target | 5 uniform | 16 | ~1.5 GB |
| 284B MoE (DS V4) | 5-8 | match target | 5 uniform | 16 | ~1-2 GB |

### Selecting Checkpoint Layers

Choose 5 layers uniformly spaced from shallow to deep in the target model:

```python
def select_checkpoint_layers(num_target_layers: int, num_checkpoints: int = 5) -> list[int]:
    # Skip first and last layers (most generic/specialized)
    start = 1
    end = num_target_layers - 2
    step = (end - start) / (num_checkpoints - 1)
    return [int(start + i * step) for i in range(num_checkpoints)]

# Examples:
# 40-layer model → [1, 10, 19, 28, 37]
# 32-layer model → [1, 8, 15, 22, 29]
# 80-layer model → [1, 20, 39, 58, 77]
```

## Training Pipeline

A complete training script is provided at `scripts/train_dflash_drafter.py` with four subcommands: `collect`, `train`, `eval`, and `preset`.

### Step 1: Collect Hidden States

Run the target model on diverse text and capture hidden states at checkpoint layers.

```bash
python scripts/train_dflash_drafter.py collect \
  --target-model mlx-community/Qwen3.6-35B-A3B-4bit \
  --output-dir ./dflash-training-data \
  --num-samples 500 \
  --max-tokens 128
```

This generates training data by:
1. Running the target model on built-in prompts (code, prose, math, conversation)
2. Generating continuations using `mlx_lm.generate`
3. Extracting hidden states at checkpoint layers for each sequence
4. Saving `.npz` files with `tokens`, `hidden_states`, and `target_layer_ids`

### Step 2: Train the Drafter

```bash
# Train from scratch
python scripts/train_dflash_drafter.py train \
  --training-data ./dflash-training-data \
  --output-dir ./dflash-drafter \
  --steps 5000 \
  --lr 1e-4

# FC-only calibration for quantized targets (fine-tune existing drafter)
python scripts/train_dflash_drafter.py train \
  --training-data ./dflash-training-data \
  --pretrained z-lab/Qwen3.6-35B-A3B-DFlash \
  --fc-only \
  --output-dir ./calibrated-drafter \
  --steps 500
```

#### Paper Tricks (implemented in training script)

| Trick | Description | Flag |
|-------|-------------|------|
| **Exponential loss decay** | w_k = exp(-(k-1)/gamma), early positions weighted more | `--loss-gamma 7.0` |
| **Random anchor sampling** | Block starts from random position each step | Always on |
| **Target-generated data** | Train on target model's own outputs | `collect` subcommand |
| **FC-only calibration** | Only train fc + hidden_norm projections | `--fc-only` |
| **Cosine LR + warmup** | Linear warmup then cosine decay to 10% of peak | `--warmup-steps 100` |
| **Validation split** | 10% holdout with periodic val loss + best checkpoint | Automatic |
| **31 diverse prompts** | Code (5 langs), math, prose, dialogue, structured | `collect` subcommand |

Loss function with exponential decay:
```python
# w_k = exp(-(k-1)/gamma) — position 1 has weight 1.0, position 15 has weight 0.13
loss_weights = mx.array([math.exp(-k / gamma) for k in range(block_size - 1)])
per_pos_loss = nn.losses.cross_entropy(mask_logits, target, reduction="none")
loss = mx.mean(per_pos_loss * loss_weights)
```

### Step 3: Evaluate

```bash
python scripts/train_dflash_drafter.py eval \
  --target-model mlx-community/Qwen3.6-35B-A3B-4bit \
  --drafter-model ./dflash-drafter \
  --max-tokens 64
```

Evaluates both cached and non-cached modes automatically. Target metrics:
- Acceptance rate > 40% (code), > 25% (general text)
- Tokens per step > 3.0

### Presets: Quick Start for Known Models

```bash
# List available presets
python scripts/train_dflash_drafter.py preset

# Show recommended config + copy-paste commands for DeepSeek V4 Flash
python scripts/train_dflash_drafter.py preset deepseek-v4-flash --quant 2bit

# Show config for Qwen3.6-35B-A3B
python scripts/train_dflash_drafter.py preset qwen3.6-35b-a3b
```

Available presets:
| Preset | Target | Drafter Layers | Recommended Steps | Notes |
|--------|--------|---:|---:|---|
| `deepseek-v4-flash` | DS V4 Flash 2/4-bit | 5 | 10,000 | 284B MoE, highest speedup potential |
| `qwen3.6-35b-a3b` | Qwen3.6-35B-A3B 4-bit | 8 | 5,000 | z-lab drafter available for FC-only calibration |

## Training Data

### Requirements

- 1B+ tokens of diverse text (code, prose, math, conversation)
- Tokenized with the target model's tokenizer
- Sequences of 512-2048 tokens

### Recommended Sources

| Dataset | Tokens | Content Type |
|---|---:|---|
| The Stack v2 (filtered) | code | Programming languages |
| SlimPajama | general | Web, books, Wikipedia |
| OpenWebMath | math | Mathematical text |
| ShareGPT/WildChat | conversation | Chat turns |

### Data Preparation

The `collect` subcommand in `train_dflash_drafter.py` handles data preparation automatically — it generates text with the target model and extracts hidden states. For custom datasets:

```bash
# Quick start: auto-generate training data from target model
python scripts/train_dflash_drafter.py collect \
  --target-model mlx-community/Qwen3.6-35B-A3B-4bit \
  --output-dir ./dflash-training-data \
  --num-samples 500

# For large-scale training with custom datasets, prepare .npz files:
# Each file should contain:
#   tokens: int32 array of token IDs [seq_len]
#   hidden_states: float32 array [seq_len, num_checkpoints * hidden_size]
#   target_layer_ids: int32 array [num_checkpoints]
```

## Training Compute

### On Apple Silicon

| Mac | Training Time (1B tokens) | Notes |
|---|---:|---|
| M3 Max 36GB | ~48-72h | Memory-limited, small batch |
| M4 Pro 48GB | ~36-48h | Good balance |
| M5 Pro 64GB | ~24-36h | Recommended |
| M4 Ultra 192GB | ~8-12h | Fastest Mac option |

### Tips for Mac Training

1. Use `mx.compile` on the training loop for 20-30% speedup
2. Gradient accumulation over 4-8 micro-batches to simulate larger batch
3. Save checkpoints every 1000 steps (training can be interrupted and resumed)
4. Monitor memory with `mx.metal.get_active_memory()` — stay below 90% of total

## Saving and Publishing

```python
from mlx.utils import tree_flatten

# Save weights
flat = dict(tree_flatten(drafter.trainable_parameters()))
mx.save_safetensors("model.safetensors", flat)

# Save config
config_dict = {
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
Path("config.json").write_text(json.dumps(config_dict, indent=2))
```

Upload to HuggingFace:
```bash
huggingface-cli upload your-org/TargetModel-DFlash \
  model.safetensors config.json \
  --repo-type model
```

## Lessons from the PoC

### What Worked
- Architecture port is correct — manual forward matches target model exactly (logit diff = 0.0)
- Drafter loads, runs, and produces draft tokens in a single forward pass
- Combined verify+extract forward pass halves per-step cost
- KV cache delivers 2.7x throughput gain (4.1 → 11.0 tok/s) via snapshot/rollback
- DDTree improves tokens/step by 36% (2.5 → 3.4) with EAGLE-2 confidence expansion
- DDTree + KV cache now combines both gains (tree attention mask with cached context)
- Training pipeline with paper tricks, cosine LR, and validation tracking is ready

### What Limits Acceptance Rate
1. **4-bit quantization** — hidden states diverge from full-precision training data. Use `--fc-only` calibration with 500+ samples and 1000+ steps.
2. **Hybrid SSM models** — 30/40 layers are SSM/linear, not attention. 4/5 checkpoint layers land on SSM layers, producing different hidden state distributions.
3. **Small active models** — MoE models with <5B active params run AR fast enough that DFlash overhead isn't justified.
4. **Insufficient calibration data** — 50 samples / 200 steps is not enough for FC-only calibration to converge. Need 500+ samples, 1000+ steps.

### Recommendations
- Train drafters against the EXACT target model (same quantization level)
- For FC-only calibration: use 500+ samples, 1000+ steps, cosine LR with warmup
- Use `preset` subcommand to get recommended configurations for known targets
- Start with full-precision or 8-bit targets for best acceptance
- Target large dense models (70B+) or large MoE (284B+) where AR is slow
- Use KV cache (`use_cache=True`) for both flat and DDTree generation
- Use DDTree with `--tree-width 5 --tree-size 60` for structured content
