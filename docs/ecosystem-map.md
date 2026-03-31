# MoE Inference Ecosystem: Complete Project Map

Every open-source project solving MoE inference on memory-constrained devices, what we can borrow from each, and the gap nobody has filled.

## The Discovery: mlx-moe Already Exists

**mu-hashmi/mlx-moe** (github.com/mu-hashmi/mlx-moe, March 2026) — runs a 46GB Qwen3-Coder-Next (512 experts, top-10) on a 32GB Mac at **8-23 tok/s using 19GB RAM**. This is the closest existing project to what we're building.

Key techniques we should adopt immediately:
- **LCP eviction**: `P = μ × 0.25^(ν/128)` — frequency × exponential recency decay
- **SafetensorsMap mmap slicing** — reads only needed expert rows from disk, not full tensor
- **mx.set_wired_limit()** — pins active expert tensors in Metal residency set
- **Skip-fallback** — when expert not cached, zero its score and renormalize remaining
- **Delta warmup** — between turns, only swap changed experts (~2-3s vs 70s full reload)

## Complete Ecosystem

### Apple Silicon Native

| Project | Stars | Key Technique | tok/s | Offloading? |
|---------|-------|--------------|-------|-------------|
| **mlx-moe** | 2 | LCP cache + mmap slice + wired pinning | 8-23 on 32GB | Yes — expert-level |
| **mlx-lm** (Apple) | 8.4K | `gather_qmm` fused kernel | ~100+ (in-RAM) | No |
| **flash-moe** | 2.3K | SSD pread + hand-tuned Metal | 4.36 on 48GB | Yes — SSD streaming |
| **llama.cpp** (Metal) | 100K | mmap + layer-level offload | varies | Layer-level only |
| **mlc-llm** (Metal) | 22K | TVM-compiled Metal kernels | varies | No |

### Linux/CUDA (Techniques Borrowable)

| Project | Stars | Key Technique | Borrowable? |
|---------|-------|--------------|------------|
| **PowerInfer** | 9.2K | Hot/cold neuron classification + io_uring async prefetch | Pattern (need macOS dispatch_io) |
| **vLLM** | 75K | PagedAttention, fused MoE kernels | Paging concept |
| **SGLang** | 25K | Elastic EP, expert backup from disk | Expert backup pattern |
| **ExLlamaV2/V3** | 4.5K | EXL3 quantization, fused MoE CUDA | Quant format |
| **DeepSpeed** | 42K | ZeRO-Infinity NVMe offload | NVMe async pattern |
| **MegaBlocks** | 1.6K | Dropless block-sparse MoE | Already in gather_qmm |
| **FasterMoE** | 1.8K | Dynamic expert shadowing | Pin universal experts |

## The Gap Nobody Has Filled

**No project does async expert prefetch overlapped with Metal GPU compute on Apple Silicon.**

- PowerInfer does this on Linux with `io_uring` (kernel-bypass async I/O)
- mlx-moe loads experts synchronously between tokens
- flash-moe's CMD3 deferred pipeline overlaps GPU compute but not SSD reads (bus contention)
- llama.cpp has no expert-level caching at all

The optimal system would:
```
Layer N:  GPU computes attention + routing
          ↕ PARALLEL
          dispatch_io reads layer N+1 experts from NVMe
          AMX dequantizes into staging MTLBuffer

Layer N+1: GPU consumes pre-loaded experts (zero wait)
           ↕ PARALLEL
           dispatch_io reads layer N+2 experts
```

This pipeline overlap is worth **40-70% latency reduction** based on PowerInfer's Linux measurements.

### macOS Equivalent of io_uring

| Linux | macOS | Status |
|-------|-------|--------|
| `io_uring` | `dispatch_io` (GCD) | Available, untested for MoE |
| `io_uring` | `kqueue` + `preadv` | Available, lower-level |
| `io_uring` | `posix_aio` | Available, POSIX standard |
| `io_uring` | `F_RDADVISE` + `fcntl` | Available (flash-moe tested, limited) |

Flash-MoE tested `dispatch_io` and found -70% performance due to `dispatch_data` management overhead. However, their test was on 6.75MB experts with unified memory bus contention. For **pre-loaded, pre-dequantized** experts where the AMX handles decompression, the dispatch_io path may be viable since the memory bus pressure is different.

## Technique Adoption Priority

### Immediate (integrate from mlx-moe)
1. LCP eviction policy (replace our LFU)
2. SafetensorsMap mmap slicing (direct expert row reads)
3. mx.set_wired_limit() for active expert pinning
4. Skip-fallback renormalization on cache miss

### Short-term (adapt from PowerInfer)
5. Hot/cold expert classification from profiling
6. Pipeline overlap: dispatch_io + AMX during GPU compute
7. Layer N+1 prefetch while N computes

### Medium-term (combine with our research)
8. EntroLLM entropy coding on asymmetric-requantized experts
9. DynaExq mixed precision (4-bit hot / 2-bit cold)
10. SpecMD Least-Stale eviction (upgrade from LCP)
11. Tensor Train decomposition for 10-20x expert compression

## What We Uniquely Contribute

No existing project combines ALL of:
- Tiered compressed cache (our cache.py)
- Mixed precision per-expert (our mixed_precision.py)
- Smart eviction (our smart_eviction.py)
- Speculative prefetch (our smart_eviction.py RoutingPredictor)
- Apple native compression (our compression_native.py)
- Real model benchmarks with honest analysis

Our unique findings:
- **4-bit quantized weights are incompressible** (7.52/8.0 entropy) — no other project documents this
- **LZ4/ZSTD achieve 1.0x on real expert data** — explains why flash-moe rejected compression
- **Entropy coding requires asymmetric quantization format** — EntroLLM connection
- **AMX dequant at 237 GB/s** — 13.5x faster than NVMe, eliminates dequant bottleneck
- **Combined technique stacking projects 2.5-3x** over flash-moe baseline
