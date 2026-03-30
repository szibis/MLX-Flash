# MoE Compression & Inference Acceleration: Research Survey

**Context**: Standard byte-level compression (LZ4/ZSTD) achieves 1.00-1.10x on 4-bit quantized MoE weights due to 7.52/8.0 bits/byte entropy. This survey covers techniques that actually work.

## Why Standard Compression Fails

4-bit quantized weights have near-maximum entropy by design. A well-calibrated quantizer maps weights to 16 bins with near-uniform utilization — maximizing information density. The scales/biases (11% of data) compress 1.6-1.8x with ZSTD, but the packed weight nibbles (89%) are fundamentally incompressible at the byte level.

**Our measured results on Qwen1.5-MoE-A2.7B-Chat-4bit:**

| Strategy | Ratio | Viable? |
|----------|-------|---------|
| Raw + LZ4 | 1.000x | No |
| Raw + ZSTD | 1.107x | Marginal |
| Nibble stream + ZSTD | 1.013x | No |
| Dict scales + ZSTD | 1.085x | No (11% of data) |
| XOR row delta + ZSTD | 1.040x | No |
| Combined (all above) | 0.994x | No (overhead > gain) |

---

## Category 1: Expert Pruning & Merging

### MoE-Pruner (ICLR 2025)
- **Idea**: Use router activation statistics to drop entire low-importance experts
- **Result**: Removes experts with minimal perplexity degradation
- **Complexity**: Low — just router stats + threshold
- **Apple Silicon fit**: High

### MergeMoE (Oct 2024)
- **Idea**: Measure expert OUTPUT similarity (not weight similarity) to decide merges
- **Result**: Hierarchical clustering reduces expert count while preserving quality
- **Complexity**: Medium

### MoE-I2 (EMNLP 2024)
- **Idea**: Two-stage — prune entire experts (inter) then SVD on survivors (intra)
- **Result**: 40-60% parameter reduction
- **Complexity**: Medium-high

### "Super Experts" (Jul 2025)
- **Idea**: Small subset of experts have disproportionate importance — keep in higher precision
- **Result**: Informs non-uniform compression (8-bit super, 2-bit rest)
- **Complexity**: Low (analysis only)

---

## Category 2: Shared Expert Substructures (Most Promising)

### D2-MoE — Delta Decompression (ICML 2025)
- **Paper**: [arxiv.org/abs/2502.17298](https://arxiv.org/abs/2502.17298)
- **Code**: [github.com/lliai/D2MoE](https://github.com/lliai/D2MoE)
- **Idea**: Merge all experts into single shared "base weight" via Fisher information, store per-expert deltas, compress deltas with SVD (they're low-rank!)
- **Result**: 13%+ over competing methods at 40-60% compression
- **Why it works when byte-compression doesn't**: SVD exploits the mathematical structure of the delta matrices (low-rank), not byte-level patterns
- **Complexity**: Medium-high

### MoBE — Mixture of Basis Experts (Aug 2025)
- **Paper**: [arxiv.org/abs/2508.05257](https://arxiv.org/abs/2508.05257)
- **Code**: [github.com/inclusionAI/MoBE](https://github.com/inclusionAI/MoBE)
- **Idea**: Decompose W = A * B where B is shared "basis matrices" across all experts, A is expert-unique
- **Result**: 24-30% parameter reduction on Qwen3-235B, DeepSeek-V3 (671B), Kimi-K2 (1T)
- **Why it works**: Exploits the fact that experts in the same layer learn similar features — the shared basis captures the common subspace
- **Complexity**: High

### MoE-SVD (ICML 2025)
- **Idea**: Direct per-expert SVD with sensitivity-guided rank allocation
- **Result**: More important experts get higher rank
- **Complexity**: Medium

---

## Category 3: Predictive Expert Loading (Latency Hiding)

### Speculating Experts (Mar 2026)
- **Paper**: [arxiv.org/abs/2603.19289](https://arxiv.org/abs/2603.19289)
- **Idea**: Use current layer's residual stream to predict next layer's experts, prefetch during current compute
- **Result**: 5-14% TPOT reduction, 93-97% prediction accuracy
- **Apple Silicon fit**: Medium (most valuable when SSD-offloading)

### SpecMD Least-Stale Eviction (Feb 2026)
- **Paper**: [arxiv.org/abs/2602.03921](https://arxiv.org/abs/2602.03921)
- **Idea**: ML-informed cache eviction that exploits predictable expert access patterns
- **Result**: 85x fewer cache misses vs LRU, 88%+ hit rates at 5% capacity, 10.7-34.7% TTFT reduction
- **Apple Silicon fit**: Very high for the SSD-overflow path

### FlashMoE SSD Cache (Jan 2026)
- **Paper**: [arxiv.org/abs/2601.17063](https://arxiv.org/abs/2601.17063)
- **Idea**: ML-based cache replacement for SSD-resident experts on edge devices
- **Result**: 51% hit rate improvement over LRU/LFU, 2.6x speedup
- **Apple Silicon fit**: Very high

---

## Category 4: Sub-4-bit Mixed Precision (Most Actionable)

### DynaExq — Dynamic Expert Quantization (Nov 2024)
- **Paper**: [arxiv.org/abs/2511.15015](https://arxiv.org/abs/2511.15015)
- **Idea**: Hot experts stay 4-bit, cold experts drop to 2-bit dynamically
- **Result**: Accuracy improvement from 73% to 77% on Qwen3-80B vs uniform 4-bit; 2.73x throughput
- **Why it works**: Rarely-used experts contribute less to output quality, tolerating aggressive quantization
- **Complexity**: Medium

### MoPEQ — Mixed Precision per Expert (Sep 2025)
- **Paper**: [arxiv.org/html/2509.02512](https://arxiv.org/html/2509.02512)
- **Idea**: Assign precision based on expert activation frequency
- **Result**: Below 4 bits average while maintaining accuracy

### QMoE — Sub-1-bit (MLSys 2024)
- **Paper**: [arxiv.org/abs/2310.16795](https://arxiv.org/abs/2310.16795)
- **Idea**: Custom codeword quantization achieving 0.8 bits/parameter
- **Result**: 1.6T model: 3142GB to 158GB (20x reduction)
- **Complexity**: Very high (custom GPU kernels, not yet on Metal)

### HOBBIT — Mixed Precision Offloading (Nov 2024)
- **Paper**: [arxiv.org/abs/2411.01433](https://arxiv.org/abs/2411.01433)
- **Idea**: Three-level system: token-level loading + layer-level prefetch + sequence-level caching
- **Result**: 3.2x speedup on Mixtral-8x7B

---

## Category 5: Expert Offloading Systems

### MoE-Lightning (ASPLOS 2025)
- **Paper**: [arxiv.org/abs/2411.11217](https://arxiv.org/abs/2411.11217)
- **Idea**: CPU-GPU-I/O pipelining with analytical roofline model
- **Result**: 10.3x throughput over SOTA offloading

### KTransformers (SOSP 2025)
- **Idea**: Keep experts in CPU RAM, compute there with AMX/AVX-512, only transfer results to GPU
- **Result**: 4.62-19.74x prefill speedup, DeepSeek-R1 on single 24GB GPU
- **Note**: x86-specific, but the concept is inverted on Apple Silicon (unified memory makes this unnecessary)

### Multi-Node Apple Silicon (ACM RACS 2025)
- **Paper**: [arxiv.org/abs/2506.23635](https://arxiv.org/abs/2506.23635)
- **Idea**: Distribute experts across Mac Studios via Thunderbolt
- **Result**: Linear scaling 2-4 nodes, 1.15x more cost-efficient than H100 cluster
- **Apple Silicon fit**: Very high

---

## Apple Silicon Specifics

The standard CPU-GPU offloading challenge (PCIe bandwidth) **does not exist** on Apple Silicon:

| x86 Problem | Apple Silicon Reality |
|-------------|---------------------|
| PCIe bottleneck (~64 GB/s) | Unified memory (~400 GB/s) |
| Need to minimize CPU-GPU transfers | CPU and GPU see same physical memory |
| Expert offloading to CPU RAM | Experts already accessible to GPU |

**The real bottleneck on Apple Silicon is model size vs RAM.** When models fit in RAM, Apple Silicon is already optimal. When they exceed RAM and spill to SSD, the bottleneck is NVMe bandwidth (17.5 GB/s).

---

## Recommended Path Forward

### Immediate (low complexity, high impact)

1. **SpecMD Least-Stale eviction** — replace our LFU with ML-informed eviction for the SSD path (88% hit at 5% cache)
2. **MoE-Pruner** — identify droppable experts from router stats, reduce working set
3. **DynaExq mixed precision** — hot experts at 4-bit, cold at 2-bit (30-40% bandwidth reduction)

### Medium-term (medium complexity)

4. **D2-MoE delta decomposition** — has open-source code, ICML 2025. Compute base expert + low-rank deltas
5. **Speculating Experts** — prefetch during GPU compute window, especially for SSD-overflow path

### Research (high complexity, transformative)

6. **MoBE shared basis** — 24-30% reduction on 671B models
7. **QMoE sub-1-bit** — 20x reduction but needs Metal kernel development
8. **Multi-node Apple Silicon** — linear scaling across Mac Studios

---

## Key References

| Paper | Venue | Year | Category |
|-------|-------|------|----------|
| D2-MoE | ICML | 2025 | Delta decomposition |
| MoBE | arXiv | 2025 | Shared basis |
| QMoE | MLSys | 2024 | Sub-1-bit quantization |
| DynaExq | arXiv | 2024 | Mixed precision |
| SpecMD | arXiv | 2026 | Cache eviction |
| Speculating Experts | arXiv | 2026 | Prefetching |
| MoE-Lightning | ASPLOS | 2025 | Offloading |
| KTransformers | SOSP | 2025 | CPU-side compute |
| FlashMoE (SSD) | arXiv | 2026 | SSD caching |
| Multi-Node Apple | RACS | 2025 | Distributed |
| MoE-Pruner | ICLR | 2025 | Expert pruning |
| HOBBIT | arXiv | 2024 | Mixed offloading |
| PuzzleMoE | arXiv | 2024 | Expert merging |
| MoE-SVD | ICML | 2025 | SVD compression |
| MoPEQ | arXiv | 2025 | Mixed precision |
