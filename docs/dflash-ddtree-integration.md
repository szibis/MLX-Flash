# DFlash + DDTree Integration for MLX-Flash

## Overview

DFlash (arXiv:2602.06036, ICLR 2026) is a speculative decoding framework that uses a lightweight block diffusion model to draft 10-16 tokens in a single forward pass. Combined with DDTree (Dynamic Draft Trees), it achieves 6-8x lossless acceleration on structured content and 96%+ draft acceptance rates.

This document describes integrating DFlash+DDTree into mlx-flash for Apple Silicon inference.

## Why This Matters for MLX-Flash

| Factor | Advantage for Apple Silicon |
|--------|----------------------------|
| Unified memory | Zero PCIe transfer cost for speculative loads — Apple Silicon's native advantage |
| Memory bandwidth bound | DFlash amortizes the bandwidth cost: verify 15 tokens in one pass instead of generating 1 |
| Expert streaming synergy | DFlash drafts predict which experts will be needed → better prefetch |
| MoE models | DeepSeek V4 Flash (284B/13B active) benefits most — drafter is tiny vs target |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLX-Flash + DFlash Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    hidden states     ┌──────────────────┐     │
│  │  Target  │ ──────────────────── │  DFlash Drafter  │     │
│  │  Model   │   layers [1,L/4,     │  (5-layer block  │     │
│  │(DeepSeek │    L/2, 3L/4, L]     │   diffusion)     │     │
│  │ V4 Flash)│                      └────────┬─────────┘     │
│  └────┬─────┘                               │               │
│       │                            draft 15 tokens           │
│       │                            (1 forward pass)          │
│       │                                     │               │
│       │         ┌───────────────────────────┘               │
│       │         ▼                                           │
│       │    ┌──────────┐                                     │
│       │    │  DDTree   │  Build tree from draft tokens       │
│       │    │  Builder  │  (top-k branching at each pos)     │
│       │    └─────┬─────┘                                     │
│       │          │                                           │
│       │     tree of candidates                               │
│       │          │                                           │
│       ▼          ▼                                           │
│  ┌─────────────────────────┐                                │
│  │   Target Model Verify   │  Single forward pass            │
│  │   (tree attention mask) │  verifies ALL candidates       │
│  └───────────┬─────────────┘                                │
│              │                                               │
│         accepted tokens (longest valid path)                 │
│              │                                               │
│              ▼                                               │
│  ┌─────────────────────────┐                                │
│  │   Expert Streaming      │  Prefetch experts for next      │
│  │   (LCP Cache update)    │  iteration using draft hints   │
│  └─────────────────────────┘                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## DFlash: How It Works

### Block Diffusion Drafting

Unlike autoregressive drafters (EAGLE, Medusa), DFlash generates ALL draft tokens in parallel:

1. **Extract hidden states** from target model at checkpoint layers
2. **Condition the drafter** on these hidden states (cross-attention)
3. **Initialize** N token positions with noise (masked tokens)
4. **Denoise in 1-2 steps** — the drafter refines all N positions simultaneously
5. **Output** N draft tokens (typically 15)

The drafter is a 5-layer transformer with bidirectional attention within the draft block and cross-attention to the target's hidden states.

### Key Properties

- **Lossless**: Every accepted token matches what greedy decoding would produce
- **Content-adaptive**: Code/reasoning → ~5.5 tokens accepted per draft (2.1x speedup); prose → ~2 tokens (1.3x)
- **Single-pass**: One drafter forward pass produces all 15 candidates (vs EAGLE's 15 sequential steps)
- **6x+ acceleration** measured on structured tasks

### MLX-Specific Advantages

- No GPU-CPU transfer overhead (unified memory)
- MLX's lazy evaluation allows fusing drafter + verifier graphs
- Expert routing predictions from drafter hidden states → better cache prefetch
- Memory-mapped weights work identically for drafter and target

## DDTree: Dynamic Draft Trees

DDTree extends DFlash by exploring multiple continuations at each position:

### Algorithm

```
1. Run DFlash drafter → get logits for positions [1..N]
2. At each position i, take top-k candidates (k=3-5)
3. Build tree structure: total candidates = sum of branches
4. Create tree attention mask (each node attends to its prefix path)
5. Run target model on entire tree in ONE forward pass
6. Accept longest valid path through tree
```

### Acceptance Rate Boost

| Method | Tokens per step | Acceptance | Effective speedup |
|--------|----------------|------------|-------------------|
| Vanilla AR | 1 | 100% | 1.0x |
| EAGLE-3 | 1 (sequential draft) | ~80% | 2.4x |
| DFlash (flat) | 15 (parallel) | ~60-78% | 3-6x |
| DFlash + DDTree | 15×3 tree | **96.4%** | **6-9x** |

### Memory Cost

DDTree's tree is verified in a single target forward pass. The additional memory is:
- Tree attention mask: O(tree_size²) — negligible for trees of 30-50 nodes
- Additional KV cache entries: O(tree_size × hidden_dim) — ~2-5 MB typical
- No additional model weights needed

## DeepSeek V4 Flash on MLX-Flash + DFlash

### Model Profile

| Property | Value |
|----------|-------|
| Total params | 284B |
| Active params/token | 13B |
| Architecture | MoE (Transformer) |
| Context length | 1M tokens |
| Experts (estimated) | ~256 routed + shared |
| Top-k routing | 2 (like V3) |
| MLX 4-bit size | ~71-80 GB on disk |
| MLX 2-bit size | ~35-40 GB on disk |
| Active weight memory | ~6.5 GB (4-bit) per token |

### Feasibility on 36GB Mac (M3 Max)

With expert streaming:
- **2-bit quant** (`mlx-community/DeepSeek-V4-Flash-2bit-DQ`): ~35 GB total weights
- Active experts per token: 13B active → ~3-4 GB at 2-bit → fits in 36 GB
- LCP cache can hold ~20 GB of hot experts
- Baseline speed: **unknown — needs benchmarking** (first priority)
- For reference: Qwen3.5-397B (209GB, similar MoE scale) gets 3.3 tok/s at 58% cache hit on this hardware

### Feasibility on Larger Macs

- **4-bit quant** (`mlx-community/DeepSeek-V4-Flash-4bit`): ~80 GB on disk
- Requires M2/M3/M4 Ultra with 128+ GB or M4 Max 128GB to fit in RAM
- Baseline speed and DFlash speedup: **unknown — no measurements exist**

## DFlash Drafter Training

A complete training pipeline is available at `scripts/train_dflash_drafter.py`. See [dflash-training-guide.md](dflash-training-guide.md) for full documentation.

To create a DFlash drafter for DeepSeek V4 Flash:

```bash
# 1. Collect hidden states from target model
python scripts/train_dflash_drafter.py collect \
  --target-model mlx-community/DeepSeek-V4-Flash-4bit \
  --output-dir ./dsv4-training-data --num-samples 500

# 2. Train from scratch with paper tricks
python scripts/train_dflash_drafter.py train \
  --training-data ./dsv4-training-data \
  --output-dir ./dsv4-dflash-drafter --steps 5000

# 3. Evaluate acceptance rate
python scripts/train_dflash_drafter.py eval \
  --target-model mlx-community/DeepSeek-V4-Flash-4bit \
  --drafter-path ./dsv4-dflash-drafter --max-tokens 64
```

Training features implemented from the DFlash paper:
- Exponential loss decay (w_k = exp(-(k-1)/7)) — early positions weighted more
- Random anchor sampling — matches inference-time protocol
- Target-generated training data — train on the model's own outputs
- FC-only calibration mode — adapt existing drafters to quantized targets

## Implementation Plan

### Phase 1: DFlash Framework (this PR)
- [x] `mlx_flash_compress/dflash.py` — Core DFlash inference loop
- [x] `mlx_flash_compress/ddtree.py` — Draft tree builder + verifier
- [x] Integration with expert streaming (prefetch from draft predictions)
- [x] Benchmark script for DeepSeek V4 Flash

### Phase 2: Drafter Models & Training
- [x] DFlash drafter on MLX (`dflash_model.py`)
- [x] PoC benchmark on Apple Silicon (M5 Pro 64GB) — see [dflash-poc-results.md](dflash-poc-results.md)
- [x] Training guide — see [dflash-training-guide.md](dflash-training-guide.md)
- [x] Full training pipeline (`scripts/train_dflash_drafter.py`) — collect/train/eval/preset subcommands
- [x] Paper tricks: exponential loss decay, random anchor sampling, target-generated data
- [x] FC-only calibration mode for quantized target models
- [x] DDTree benchmark: 3.4 tokens/step (+36% vs flat), EAGLE-2 confidence expansion
- [x] Sequoia optimal tree topology via DP
- [x] All optimizations: `hidden_dtype`, `inference_block_size`, `compile_drafter`
- [x] KV cache for flat DFlash: 2.7x speedup via snapshot/rollback
- [x] **DDTree + KV cache**: tree attention mask with cached context prefix
- [x] Scaled calibration pipeline: cosine LR, warmup, validation split, 31 diverse prompts
- [x] DeepSeek V4 Flash preset config (`preset deepseek-v4-flash`)
- [x] **Drafter quantization** (`quantize_drafter=8`): +19% throughput, zero quality loss
- [x] **Layer pruning** (`num_active_layers`): benchmarked, not recommended (acceptance drops)
- [x] **Full optimization sweep**: 12 optimizations tested, 3 accepted, 9 rejected with measured data
- [x] **SSM rollback analysis**: identified replay overhead as key bottleneck (~18ms/step, 28% of cost)
- [ ] Build SSM state capture & replay (eliminates ~18ms/step via generic recurrence replay)
- [ ] Add drafter KV caching (reduces drafter forward time)
- [ ] Train DeepSeek V4 Flash-specific drafter (5 layers, block diffusion objective)
- [ ] Publish drafter weights to `mlx-community/`

### Phase 3: Production Integration
- [ ] Wire DFlash into `serve.py` generation loop
- [ ] Add DFlash metrics to Prometheus (draft_acceptance_rate, tokens_per_draft, etc.)
- [ ] Auto-detect DFlash drafter when model is loaded
- [ ] Adaptive k (num_spec_tokens) based on content type

## Measured Results (Real Data)

### PoC: DFlash Drafter on Apple Silicon (M5 Pro 64GB)

DFlash speculative decoding on MLX. See [dflash-poc-results.md](dflash-poc-results.md) for full details.

| Method | Model | tok/s | Acceptance | Tokens/step |
|--------|-------|------:|------:|------:|
| Baseline AR | Qwen3.6-35B-A3B-4bit | 65.6 | N/A | 1.0 |
| DFlash naive (no cache, bf16) | + DFlash drafter | 3.6 | 6.4% | 1.8 |
| DFlash + KV cache (bf16) | + DFlash drafter | 40.5 | 13.3% | 4.0 |
| **DFlash + KV cache + 8-bit quant** | + DFlash drafter | **48.0** | **13.3%** | **4.0** |
| DFlash + KV cache + 8-bit + bs=4 | + DFlash drafter | 50.6 | 36.7% | 2.2 |
| DFlash + DDTree (no cache) | + DFlash drafter | 3.8 | 8.0% | 3.4 |

**Optimization stack**: naive → +KV cache (11.3x) → +8-bit drafter quant (+19%) → best: **48.0 tok/s** (0.73x AR).

**Why DFlash can't beat AR here**: target model (3B active MoE, 4-bit) runs AR at 65.6 tok/s — too fast for speculative decoding overhead to amortize. DFlash wins on slow models (AR < 20 tok/s).

**12 optimizations tested and rejected** (block_size changes, hidden_dtype, compile, multi-step denoising, temperature scaling, top-k acceptance, SDPA, layer pruning, cache trim). Full details in [dflash-poc-results.md](dflash-poc-results.md).

**Next optimization**: SSM state capture & replay to eliminate ~18ms/step replay overhead. See [dflash-poc-results.md](dflash-poc-results.md) for details.

### Prior Baselines (M3 Max 36GB)

| Model | Config | Measured tok/s | Source |
|-------|--------|---------------:|--------|
| Qwen1.5-MoE-A2.7B 4bit | Fits in RAM, no streaming | 114.9 | measured-results.md |
| Qwen3-30B-A3B 4bit | Expert streaming | 82.6 | README.md |
| Qwen3.5-397B 4bit (209GB) | Expert streaming, 58% cache hit | 3.3 | measured-results.md |

### DFlash on NVIDIA (Reference, NOT on Mac)

From AEON-7/vllm-dflash on DGX Spark (Blackwell, 128GB unified, 273 GB/s):
- Qwen3.5-27B NVFP4 + DFlash k=15: **64 tok/s** on code, **29.5 tok/s** on prose
- Acceptance per 15-token draft: ~5.5 tokens (code), ~2 tokens (prose)
- Effective decode speedup: **~2.1x** (code), **~1.3x** (prose) vs vanilla on same hardware

### Where DFlash WILL Help on Mac

DFlash speedup requires a target model where AR is slow enough that amortizing 4-7 accepted tokens over 1 verification pass wins. Candidates:

| Mac Config | Model | AR Baseline | DFlash Potential | Status |
|------------|-------|------:|---|---|
| M5 Pro 64GB | DS-V4 Flash 2bit (35 GB) | TBD | HIGH — 13B active params, slow AR | Needs drafter + baseline |
| M4 Ultra 192GB | DS-V4 Flash 4bit (80 GB) | TBD | HIGH | Needs drafter |
| M5 Pro 64GB | Qwen3.5-397B 4bit | 3.3 tok/s | VERY HIGH — AR is very slow | Needs drafter |

### What We Need Next

1. **SSM state capture & replay** — eliminates ~18ms/step replay overhead via generic recurrence replay. Projected: 48 → 58 tok/s (+21%)
2. **Drafter KV caching** — drafter attention should cache context KV across steps
3. **Train DeepSeek V4 Flash drafter** — 284B model where AR is slow → DFlash wins big
4. **Full-precision baseline** — test with bf16 target to isolate quantization impact (6-12% acceptance vs paper's 40-70%)

```bash
# Quick start: DeepSeek V4 Flash drafter
python scripts/train_dflash_drafter.py preset deepseek-v4-flash --quant 2bit
```

## References

- DFlash paper: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) (Chen, Liang, Liu — ICLR 2026)
- Block Diffusion: [arXiv:2503.09573](https://arxiv.org/abs/2503.09573) (Arriola et al., 2025)
- AEON-7/vllm-dflash: [GitHub](https://github.com/AEON-7/vllm-dflash) — reference vLLM implementation
- DFlash drafter (Qwen3.5): [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash)
- DeepSeek V4 Flash MLX: [mlx-community/DeepSeek-V4-Flash-4bit](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit)
