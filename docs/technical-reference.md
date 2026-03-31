# Technical Reference

## Research Foundation

MLX-Flash-Compress is built on techniques from 60+ published papers across multiple fields. Key references:

### Cache Eviction
- **LCP (Least Critical Priority)**: Production-proven in [mlx-moe](https://github.com/mu-hashmi/mlx-moe). Formula: `P = frequency × 0.25^(steps_since_last / 128)`. Combines Zipf-distributed access frequency with exponential recency decay.
- **SpecMD Least-Stale**: [arXiv:2602.03921](https://arxiv.org/abs/2602.03921) (2026) — 85x fewer collision misses vs LRU, 88% hit rate at 5% cache capacity.

### Mixed Precision
- **DynaExq**: [arXiv:2511.15015](https://arxiv.org/abs/2511.15015) (2024) — Dynamic expert quantization. Hot experts at 4-bit, cold at 2-bit. 2.73x throughput improvement.
- **MoPEQ**: [arXiv:2509.02512](https://arxiv.org/abs/2509.02512) (2025) — Per-expert precision based on activation frequency.

### Expert Compression
- **D²-MoE**: [arXiv:2502.17298](https://arxiv.org/abs/2502.17298) (ICML 2025) — Delta decomposition via Fisher information + SVD. 40-60% compression.
- **MoBE**: [arXiv:2508.05257](https://arxiv.org/abs/2508.05257) (2025) — Shared basis matrices. 24-30% reduction on 671B models.
- **QMoE**: [arXiv:2310.16795](https://arxiv.org/abs/2310.16795) (MLSys 2024) — Sub-1-bit quantization. 20x compression on 1.6T models.

### Expert Prefetching
- **Speculating Experts**: [arXiv:2603.19289](https://arxiv.org/abs/2603.19289) (2026) — 93-97% prediction accuracy using residual stream.
- **Eliseev & Mazur**: [arXiv:2312.17238](https://arxiv.org/abs/2312.17238) (2023) — Expert offloading with 60-80% prefetch hit rate on Mixtral.

### MoE Offloading Systems
- **Flash-MoE**: [github.com/danveloper/flash-moe](https://github.com/danveloper/flash-moe) — 397B on 48GB MacBook at 4.36 tok/s via NVMe streaming.
- **PowerInfer**: [github.com/SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) — Hot/cold neuron classification + io_uring async I/O.
- **MoE-Lightning**: [arXiv:2411.11217](https://arxiv.org/abs/2411.11217) (ASPLOS 2025) — 10.3x throughput via analytical pipeline optimization.
- **KTransformers**: SOSP 2025 — CPU-side expert compute, DeepSeek-R1 on 24GB GPU.

### Entropy Coding
- **EntroLLM**: [arXiv:2505.02380](https://arxiv.org/abs/2505.02380) (2025) — Huffman on quantized weights: 30% savings with asymmetric quantization.
- **ECCO**: ISCA 2025 — 4x compression via per-group entropy-aware codebooks.

### Expert Pruning/Merging
- **MoE-Pruner**: ICLR 2025 — Drop entire experts based on router statistics.
- **PuzzleMoE**: [arXiv:2511.04805](https://arxiv.org/abs/2511.04805) (2024) — 50% model size reduction via sparse merging.
- **"Super Experts"**: [arXiv:2507.23279](https://arxiv.org/abs/2507.23279) (2025) — Non-uniform precision allocation.

### Neuroscience-Inspired
- **Predictive coding** (Friston 2005) → speculative expert prefetching
- **Hebbian co-activation** → expert file layout clustering
- **Dendritic computation** → two-stage expert loading (load gate first, skip if low-impact)

### Hardware
- **Apple AMX**: Matrix coprocessor embedded in CPU P-cores, ~237 GB/s dequantization throughput
- **Apple Neural Engine**: 18 TOPS (M3), supports INT4 dequant via CoreML
- **Thunderbolt 5**: 80 Gbps per port, 4 ports on M4 Max = ~32 GB/s external bandwidth
- **Tensor Train decomposition**: Quantum-inspired classical technique, 10-37x compression potential

## Implementation Details

### C GCD Engine (`csrc/fast_cache.m`)

The hot path is implemented in C with Objective-C for Apple's GCD:

```c
// dispatch_group for parallel expert reads (4 experts simultaneously)
dispatch_group_t group = dispatch_group_create();
for (int i = 0; i < k; i++) {
    dispatch_group_async(group, io_queue, ^{
        read_expert(layer, expert_ids[i]);
    });
}
dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
```

**Measured dispatch overhead**: 5.38 microseconds per GCD dispatch (vs ~50us for Python ThreadPoolExecutor).

### LCP Priority Calculation

```
Priority(expert) = frequency × 0.25^(steps_since_last_use / 128)

Where:
  frequency = total times this expert was activated
  steps_since_last_use = current_token - last_token_that_used_this_expert
  0.25 = decay base (experts lose 75% priority every 128 tokens of non-use)
  128 = decay constant (tuned for MoE routing patterns)
```

Experts with high frequency AND recent use have highest priority (kept in cache). Experts that were frequent but haven't been used in hundreds of tokens decay to near-zero and get evicted.

### Mixed Precision (4-bit → 2-bit)

```
4-bit expert: 1,584 KB per projection (gate + up + down)
2-bit expert:   880 KB per projection (1.80x smaller)
Quality loss:   MSE 0.000059 (negligible for cold/rare experts)

Process:
  1. Unpack 4-bit nibbles from uint32
  2. Dequantize using original scale/bias
  3. Requantize to 2-bit (4 levels) with new scale/bias
  4. Pack 4 crumbs per uint8
```

### SSD Protection

```
Read workload per inference token (Flash-MoE scale, 70% cache hit):
  - 4 experts × 60 layers × 30% miss rate = 72 expert reads
  - 72 × 6.75 MB = 486 MB per token
  - At 10 tok/s = 4.86 GB/s sustained read

SSD impact: READS DO NOT DEGRADE NAND CELLS.
  - TBW (Total Bytes Written) is the wear metric
  - Inference is read-only — zero write wear
  - Only risk: thermal throttling from sustained reads
  - Protection: rate limiting + thermal monitoring + F_RDAHEAD hints
```

### Performance Model

```
tok/s = 1000 / (num_layers × layer_time_ms)

layer_time_ms = gpu_compute_ms + effective_io_ms

effective_io_ms = cache_hit_rate × 0.08ms (RAM decompress)
               + (1 - cache_hit_rate) × ssd_read_ms

ssd_read_ms = K × expert_size_mb / ssd_bandwidth_gbs × 1000

cache_hit_rate = Σ P(rank ≤ N) where P follows Zipf(α=0.8)
                 and N = cached_experts_per_layer
```
