# Architecture: MLX + Flash-MoE + Compressed Cache

## Problem Statement

Large Mixture-of-Experts (MoE) models like Qwen3.5-397B have 209GB+ of expert weights that far exceed the 48-128GB unified memory on Apple Silicon Macs. Flash-MoE demonstrated that streaming expert weights from NVMe SSD at 4.36 tok/s on a MacBook Pro is viable, but **56% of inference time is spent waiting for SSD I/O**.

Meanwhile, the CPU's 12+ performance cores sit almost entirely idle during inference (CPU utilization <0.1% — only used for softmax routing at 3us per layer).

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: MLX Interface (Python)                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  model = FlashMoEModel.from_pretrained("Qwen3-397B") │  │
│  │  output = model.generate("Hello", max_tokens=100)     │  │
│  │                                                       │  │
│  │  Uses: mx.array, mx.compile, mx.nn.Module             │  │
│  │  KV cache: MLX quantized (8-bit, saves 4-8x RAM)      │  │
│  │  Attention: MLX Steel GEMM (15 full-attn layers)       │  │
│  │  Norms/activations: mx.compile() auto-fused            │  │
│  └────────────────────┬──────────────────────────────────┘  │
│                       │                                     │
│  LAYER 2: Compressed Expert Cache (C/libcompression)        │
│  ┌────────────────────▼──────────────────────────────────┐  │
│  │  ExpertCacheManager                                   │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  │  │
│  │  │ LZ4 Hot  │  │ ZSTD Warm    │  │ Routing Stats  │  │  │
│  │  │ Cache    │  │ Cache        │  │ Tracker        │  │  │
│  │  │ (25GB/s  │  │ (best ratio  │  │ (frequency-    │  │  │
│  │  │  decomp) │  │  for rarely  │  │  aware LFU     │  │  │
│  │  │          │  │  used exps)  │  │  eviction)     │  │  │
│  │  └──────────┘  └──────────────┘  └────────────────┘  │  │
│  │  GCD parallel decompress on 12 P-cores                │  │
│  └────────────────────┬──────────────────────────────────┘  │
│                       │ cache miss → pread()                │
│  LAYER 3: Flash-MoE SSD Engine (C/Metal)                    │
│  ┌────────────────────▼──────────────────────────────────┐  │
│  │  NVMe pread() → 2MB-aligned buffers → MTLBuffer       │  │
│  │  Hand-tuned Metal shaders (dequant, SwiGLU, combine)   │  │
│  │  GatedDeltaNet recurrence (BLAS/AMX)                   │  │
│  │  Deferred CMD3 pipeline                                │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Flash-MoE Deep Dive

### What It Does

Pure C/Objective-C + hand-written Metal shaders running Qwen3.5-397B (397B params, 512 experts/layer, K=4 active) on a 48GB M3 Max at 4.36 tok/s. The 209GB model streams from NVMe SSD via `pread()`.

### Per-Layer Pipeline (4.27ms average)

```
Phase        Time     %     Description
─────────────────────────────────────────────────────────────
SSD I/O      2.41ms   56%   4 experts × 6.75MB via parallel pread()
CMD1 GPU     1.22ms   29%   GatedDeltaNet / attention projections
CMD2 GPU     0.55ms   13%   Routing + shared expert + o_proj
CMD3 GPU     0.04ms    1%   Expert forward (deferred, overlapped)
CPU routing  0.003ms  <0.1% Softmax + topK selection
```

### Three Command Buffer Pipeline

```
Layer N:
  [CMD1: commit + wait]     -- normalization, linear attn
  [CMD2: commit + wait]     -- attention, routing, shared expert
  [CMD3: commit, NO wait]   -- expert forward pass (DEFERRED)
    ↓ CPU immediately starts Layer N+1 prep
  [CMD1 N+1: commit + wait] -- implicitly waits for CMD3 N
                               (shared hidden_state buffer)
```

### What Flash-MoE Tried and Rejected

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Uncompressed LRU cache | -38% | GPU memory pressure from large RAM buffers |
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | SSD DMA during GPU compute: -73% GPU throughput |
| Temporal expert prediction | -18% | 25% hit rate, SSD bandwidth waste |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |

### The Critical Constraint

On Apple Silicon, SSD DMA and GPU compute share the **same memory bus** (~400 GB/s). When both run simultaneously, GPU throughput drops 73%. This means SSD prefetch during GPU compute is counterproductive. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

## MLX Deep Dive

### Architecture

```
Python API  (mlx.core / mlx.nn)
    ↓
C++ Array + Primitives  (lazy evaluation DAG)
    ↓
Scheduler + Graph Executor  (BFS topological sort)
    ↓
Backend dispatch
    ├── Metal (GPU)  — Steel GEMM, quantized kernels
    └── CPU          — Accelerate BLAS
```

### Key Features for This Project

1. **`mx.fast.metal_kernel()`** — Register custom Metal shaders in MLX's graph
2. **`gather_qmm`** — First-class MoE kernel (gather expert weights + quantized GEMM)
3. **Quantized KV cache** — 8-bit default, saves 4-8x RAM for attention
4. **`mx.compile()`** — JIT-fuses elementwise ops into single Metal kernels
5. **Lazy evaluation** — Enables pipelining CPU decompression with GPU compute

### MLX vs Hand-Written Metal

| Aspect | MLX | Hand-Written |
|--------|-----|-------------|
| Kernel perf | ~95% of hand-tuned | 100% |
| MoE dispatch | `gather_qmm` single kernel | Manual per-expert |
| Quantized ops | 2/3/4/5/6/8-bit, arch-tuned | Must write from scratch |
| Disk streaming | Not supported (requires DRAM) | Custom pread/mmap |
| Cmd buffer control | Automatic heuristic | Surgical placement |

### Why MLX Can't Replace Flash-MoE

MLX requires all weights in DRAM (or lazy-mmap'd, which is 5x slower). It has no mechanism for streaming expert weights from SSD on demand. The compressed cache bridges this gap.

## Compressed Cache Design

### Why Compression Changes the Economics

Flash-MoE tested uncompressed caching and rejected it because the cache consumed too much RAM, causing GPU memory pressure. **Compression is the key differentiator**: with 2-4x compression ratio, the same RAM budget holds 2-4x more experts, dramatically changing the cache hit rate.

### Cache Hit Economics

```
SSD read (cold):      6.75MB at 17.5 GB/s = 0.39ms per expert
                      × 4 parallel = 0.39ms (with NVMe queue depth)
                      Measured: 2.41ms (with overhead + contention)

LZ4 decompress (hot): ~550KB compressed → 2MB
                       At 25 GB/s = 0.08ms per expert
                       × 4 sequential = 0.32ms (fast-path, no thread pool)
```

### Why Decompression Avoids the Memory Bus Problem

```
Flash-MoE's failure (SSD prefetch + GPU):
  GPU executing ←→ SSD DMA writing to DRAM  (bus fight! -73% GPU)

Compressed cache (CPU decompress + GPU):
  GPU executing CMD3[N]  ← GPU L2 + memory bus for compute
  CPU decompressing[N+1] ← CPU L2 cache only (small writes)

  The CPU decompresses from compressed RAM into a staging buffer.
  Memory bus writes are short bursts (cache line writebacks), not
  sustained DMA streams. The memory controller interleaves these
  with GPU reads without the arbitration penalty of SSD DMA.
```

### Tiered Eviction

```
Access frequency:  ████████████░░░░░░░░░░░░░░ (power-law / Zipf)
                   ↑ hot experts    ↑ warm       ↑ cold (SSD)

LZ4 hot tier:   Most-accessed experts (>5 hits per 1000 tokens)
                Fast decompress (25 GB/s), moderate ratio (1.6x)

ZSTD warm tier: Infrequently-accessed experts (1-5 hits)
                Better ratio (2.2x), slower decompress (1.4 GB/s)

SSD cold tier:  Rarely-accessed experts (0 hits recently)
                Full NVMe bandwidth, no cache overhead
```

## Performance Projections

### Python Prototype (this repo, measured)

| Scenario | SSD Only | LZ4 Cache | Speedup |
|----------|----------|-----------|---------|
| OS page cache | 194 tok/s | 161 tok/s | 0.83x |
| NVMe + contention (1.5ms) | 91 tok/s | 92 tok/s | 1.01x |
| Flash-MoE calibrated (2.4ms) | 59 tok/s | 81 tok/s | **1.37x** |

### Projected C Implementation (extending Flash-MoE)

| Configuration | tok/s | vs Flash-MoE baseline |
|--------------|-------|----------------------|
| Flash-MoE baseline (4-bit, FMA) | 4.36 | — |
| + LZ4 compressed cache (85% hit) | ~6.0 | +37% |
| + MLX quantized KV cache (frees RAM) | ~6.8 | +56% |
| + Pipelined CPU decompress during GPU | ~7.2 | +65% |

## Benchmark Discoveries

### 1. Python Threading Overhead Dominates

`ThreadPoolExecutor` future submit/collect costs ~50us per call. LZ4 decompress of 550KB takes ~20us. The synchronous fast-path for cache hits (bypassing the pool) was essential.

### 2. Compress-on-Insert Is the Cold Path Killer

Naively compressing when inserting into cache makes every cold miss pay a 0.5ms+ compress penalty. Async background population is mandatory.

### 3. LZ4 Beats Everything on Decompress Speed

| Algorithm | Decompress Speed | Compress Ratio |
|-----------|-----------------|----------------|
| LZ4 (Python C ext) | 25,783 MB/s | 3.72x |
| LZ4_RAW (Apple native) | 19,852 MB/s | 3.63x |
| LZFSE (Apple native) | 908 MB/s | 6.71x |
| ZSTD-3 | 1,307 MB/s | 7.97x |

LZ4 decompresses at 20x the speed of ZSTD. For hot-tier caching, speed matters more than ratio.

### 4. LZFSE Is Not Faster Than LZ4

Despite being Apple's native format optimized for Apple Silicon, LZFSE compresses at only 251 MB/s vs LZ4's 3,583 MB/s. It's designed for better ratio (file compression), not raw speed.

### 5. The Cache Only Wins When SSD Is the Real Bottleneck

When the OS page cache serves "SSD" reads from RAM, any cache layer adds overhead. The compressed cache breaks even at ~1.5ms/2MB SSD latency and wins at Flash-MoE's measured 2.4ms.
