# Flash-MoE Technical Analysis

Source: https://github.com/danveloper/flash-moe

## Overview

Pure C/Metal inference engine running Qwen3.5-397B-A17B (397 billion parameter MoE) on a MacBook Pro M3 Max (48GB) at 4.36 tokens/second. The entire 209GB model streams from NVMe SSD.

## Model Architecture

- **60 transformer layers**: 45 GatedDeltaNet (linear attention) + 15 standard full attention
- **512 experts per layer**, K=4 activated per token (plus one shared expert)
- **Hidden dimension**: 4096
- **Expert size**: ~6.75MB each at 4-bit quantization

## Key Techniques

### 1. SSD Expert Streaming

Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only K=4 active experts per layer are loaded (~27MB total per layer). The OS page cache manages caching — no custom cache needed.

### 2. FMA-Optimized Dequant Kernel

The inner loop rearranges `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU's fused multiply-add unit do dequant+multiply in one instruction. 12% faster than naive formulation.

### 3. Deferred CMD3 Pipeline

Expert forward pass (CMD3) is submitted without waiting. GPU executes it while CPU prepares next layer. The combine + residual + norm feed directly into the next layer's attention projections.

### 4. Trust the OS

No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU. Every custom caching approach tested was slower due to GPU memory pressure or overhead. Page cache achieves ~71% hit rate naturally.

## Timing Breakdown (per layer, 4-bit)

```
Phase                Time     %
──────────────────────────────────
SSD I/O (4 experts)  2.41ms   56%
CMD1 GPU             1.22ms   29%
CMD2 GPU             0.55ms   13%
CMD3 GPU (deferred)  0.04ms    1%
CPU routing          0.003ms  <0.1%
──────────────────────────────────
Total                4.27ms
```

## Metal Shaders (shaders.metal, ~1200 lines)

- `dequant_matvec_4bit_v3`: Tiled 4-bit dequantized matrix-vector multiply with SIMD reduction, shared input cache, FMA optimization
- `fused_gate_up_swiglu`: Fused SwiGLU activation (gate × silu(up))
- `rms_norm_sum_sq` + `rms_norm_apply_bf16`: Two-pass RMS normalization
- `attn_scores_batched` + `attn_softmax_batched` + `attn_values_batched`: Batched GPU attention
- `gpu_rope`: Fused RoPE with Q deinterleave and K normalization
- `weighted_sum` + `residual_add`: MoE combine + residual (fused)

## Memory Layout

- **Non-expert weights**: 5.5GB (mmap'd, read-only, `MTLResourceStorageModeShared`)
- **Metal scratch buffers**: ~200MB
- **Total fixed**: ~6GB, leaving 42GB for OS + page cache
- **Expert data**: Streams from SSD on demand, OS page cache retains hot experts

## What Was Tested and Rejected (58 experiments)

| Approach | Impact | Reason for Rejection |
|----------|--------|---------------------|
| Metal LRU cache | +38% when removed | GPU memory pressure from buffer allocation |
| LZ4 compression | -13% | Decompress overhead on unified memory bus |
| F_RDADVISE prefetch | 0% net | SSD DMA during GPU: -73% GPU throughput |
| Temporal prediction | -18% | 25% hit rate, bandwidth waste |
| GPU LUT dequant | -2% | Indirect register access serializes |
| Private buffer compression | -20% pipeline | Blit cost 4×7MB exceeds matvec savings |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| Expert file clustering | 0% | NVMe ignores scatter at 7MB granularity |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| MTP speculative decode | break-even | MoE I/O scales per-token (unlike dense) |

## Hardware Constraints (Apple Silicon)

1. **Unified memory bus**: SSD DMA and GPU compute share ~400 GB/s bandwidth. Cannot overlap profitably.
2. **No explicit fence needed**: `MTLResourceStorageModeShared` + `commit` provides producer-consumer ordering automatically.
3. **Page cache is optimal**: The OS LRU page cache outperforms every custom caching scheme tested.
4. **NVMe command queue depth**: M3 Max supports parallel pread via GCD, achieving ~17.5 GB/s sequential read.
