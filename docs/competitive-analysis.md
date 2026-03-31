# Competitive Analysis: MoE Inference on Consumer Hardware

## Executive Summary

Between January 2024 and March 2026, at least **8 independent open-source projects** and **12+ academic papers** attacked the same core problem: running MoE models too large for available memory. The approaches split into two camps — **pure streamers** (accept very low throughput, minimize RAM use) and **cache-first systems** (invest RAM in caching for dramatically higher throughput). MLX-Flash-Compress is in the cache-first camp.

## Direct Apple Silicon Competitors

### 1. mu-hashmi/mlx-moe (2 stars) — Closest Functional Competitor

- Expert profiles (pre-computed activation patterns) determine which experts to keep in RAM
- Universal experts (always-hot) pinned permanently
- Zero-eval dispatch: if expert not resident, skip it (graceful degradation)
- OpenAI + Anthropic-compatible API server
- 46GB Qwen3-Coder on 32GB Mac at **6-23 tok/s** using 19GB RAM
- Supports 10+ MoE architectures

**vs. Us:** Similar architecture but no Rust sidecar, no mixed precision (hot 4-bit / cold 2-bit), no async prefetch, no memory monitoring, no SSD protection. Profiles are pre-computed offline vs. our LCP online learning.

### 2. kqb/mlx-od-moe (1 star) — Shadow Model Predictor

- Memory-mapped experts via `.npy` files (OS page cache)
- **Shadow model predictor** trained on routing patterns — >90% top-8 accuracy
- 375GB Kimi-K2.5 on 192GB Mac Studio at ~70 tok/s
- M4 Max 36GB: ~45 tok/s with 55-65% hit rate

**vs. Us:** The learned routing predictor is more sophisticated than our LCP heuristic. We should consider this as a future enhancement.

### 3. rita-aga/mlx-turboquant-moe-streaming (0 stars, March 2026)

- **Pure streaming, zero caching** — pread() every expert every time
- TurboQuant: KV cache 4-5x compression (Google Research technique)
- 1.04T param model (578GB) at **0.5 tok/s using only 7.1 GB RAM**
- 953 lines of code

**vs. Us:** Opposite tradeoff — extreme minimum RAM vs. our cache-first approach. Complementary, not competing. The TurboQuant KV cache compression is worth investigating.

### 4. iahuang/cosmoe (0 stars, Feb 2026)

- Interleaves I/O with GPU compute (same async prefetch principle)
- Early-stage, no caching layer or production server

### 5. 0xSero/reap-mlx (52 stars)

- Permanently removes experts based on activation telemetry (pruning, not caching)
- Orthogonal to our approach — could be used as preprocessing before our system

## Non-Apple Silicon Competitors

### PowerInfer (SJTU, 9,207 stars, SOSP 2024)

- Hot/cold neuron locality concept (intellectual ancestor of this space)
- **Only ReLU-sparse models** — doesn't support modern SwiGLU MoE (Qwen, DeepSeek)
- 11.69x faster than llama.cpp on RTX 4090
- Not relevant for Apple Silicon + modern models

### MoE-Infinity (290 stars, arXiv:2401.14361)

- Most sophisticated research-grade system
- Sparsity-aware expert cache with activation trace analysis
- 3.1-16.7x latency improvement vs. vLLM, Ollama, DeepSpeed
- NVIDIA-only, HuggingFace drop-in API
- 290 stars, actively maintained

### tinyserve (2 stars, CUDA-only)

- 3-tier: SSD -> RAM -> GPU VRAM
- LFRU eviction + FATE cross-layer prefetch (looks ahead N layers)
- 79-94% cache hit rates, 335 tests
- OpenAI-compatible

### HOBBIT (arXiv:2411.01433, Nov 2024) — Closest Paper

- **Nearly identical architecture** to MLX-Flash-Compress
- Mixed precision per expert (hot vs. cold) + dynamic loading + adaptive prefetch + multi-dimensional caching
- Up to 9.93x speedup
- llama.cpp + NVIDIA (not MLX/Apple Silicon)

## Commercial/Mainstream

| Product | MoE Expert Caching? | SSD Streaming? |
|---------|---------------------|----------------|
| llama.cpp | No | No (model must fit in RAM+VRAM) |
| Ollama | No (built on llama.cpp) | No |
| vLLM | No | No (requires full VRAM) |
| LM Studio | No (built on llama.cpp) | No |

**Key gap:** None of the mainstream tools support expert-level caching or SSD streaming for MoE models.

### 6. jundot/omlx (NEW — 2026)

- LLM inference server with continuous batching and SSD caching
- Hybrid quantization per-layer: mxfp4/mxfp8/affine per expert
- Batched GPTQ: ~15x quantization speedup
- SpecPrefill: attention-based sparse prefill

**vs. Us:** We have more prediction techniques (residual predictor, speculative execution, Belady eviction). They have hybrid quantization format support.

### 7. ARahim3/mlx-tune (NEW — 2026)

- Fine-tuning framework for 39+ architectures including all MoE families
- Per-expert LoRA via `LoRASwitchLinear`
- Complementary, not competing — could fine-tune our compressed models

## What We Do That Nobody Else Does

1. **Rust sidecar with Mach syscall memory monitoring** (0.1ms, 210x faster) — no competitor
2. **Mixed precision per-expert on Apple Silicon** (hot 4-bit / cold 2-bit) — only HOBBIT does this, on NVIDIA
3. **Speculative expert execution** (predict → execute → verify) — no Apple Silicon competitor
4. **Residual-stream predictor** (97%+ accuracy, linear projection) — only "Speculating Experts" paper, NVIDIA
5. **Forward-looking Belady-optimal eviction** — no competitor integrates prediction into eviction
6. **15+ research techniques implemented** — most of any project in this space
7. **SSD thermal protection** (70C cutoff, sequential hints, zero writes) — no competitor
8. **Expert merging + vertical splitting** — complementary compression from both directions
9. **Adaptive top-k skipping** — dynamic compute reduction per token
10. **Combined stack** (speculative execution + Belady eviction + residual predictor + expert merging + entropy coding + Rust sidecar) — competitors implement 1-3 of these

## What Competitors Do Better (Updated)

| Feature | Who | Status |
|---------|-----|--------|
| ~~Shadow model predictor~~ | ~~kqb/mlx-od-moe~~ | **CLOSED** — we have shadow MLP + residual predictor (97%+) |
| ~~Vertical expert splitting~~ | ~~MoEpic paper~~ | **CLOSED** — implemented in vertical_split.py |
| ~~Cross-layer prefetch~~ | ~~tinyserve~~ | **CLOSED** — 3-hop lookahead in advanced_prefetch.py |
| Hidden-state predictor input | kqb/mlx-od-moe | Uses actual hidden states from model internals |
| Hybrid mxfp4/mxfp8 per expert | jundot/omlx | Per-expert format selection |
| HuggingFace drop-in | MoE-Infinity | Zero-code-change model loading |
| Model breadth | mu-hashmi/mlx-moe | 10+ architectures explicitly tested |
| Minimum RAM | rita-aga/mlx-turboquant | 7.1 GB for 1T model |
| Users/maturity | llama.cpp, Ollama | 100K+ stars, massive ecosystem |

## Competitive Position

```mermaid
quadrantChart
    title MoE Inference: Sophistication vs. Apple Silicon Support
    x-axis Low Sophistication --> High Sophistication
    y-axis No Apple Silicon --> Full Apple Silicon
    quadrant-1 Our Sweet Spot
    quadrant-2 Simpler Apple Silicon
    quadrant-3 Mainstream (no MoE caching)
    quadrant-4 Research (NVIDIA only)
    MLX-Flash-Compress: [0.75, 0.9]
    mlx-moe: [0.4, 0.85]
    mlx-od-moe: [0.5, 0.8]
    cosmoe: [0.2, 0.7]
    turboquant: [0.15, 0.75]
    MoE-Infinity: [0.85, 0.1]
    HOBBIT: [0.9, 0.05]
    PowerInfer: [0.6, 0.15]
    tinyserve: [0.65, 0.05]
    llama.cpp: [0.3, 0.5]
    Ollama: [0.2, 0.45]
```

## Research Frontier: Opportunities

### Papers to Watch

| Paper | Technique | Potential Gain |
|-------|-----------|---------------|
| HOBBIT | Multi-dimensional cache + mixed precision | Architecture validation |
| MoEpic | Vertical expert splitting | 2x cache coverage |
| FATE | Cross-layer gate correlation | Better prefetch accuracy |
| DALI | Workload-aware dynamic assignment | Optimal cache policy |
| MELINOE | Fine-tune to reduce expert churn | 1.2-3x throughput |
| Not All Models Suit Offloading | LRC metric | Know when caching helps |

### Technical Opportunities No Competitor Is Exploiting

1. **Tensor network decomposition** — 10-20x compression (research frontier)
2. **AMX dequant pipeline** — Apple's matrix coprocessor for 13x faster decompression
3. **Entropy coding (EntroLLM)** — asymmetric quantization for 30% storage savings
4. **Thunderbolt 5 striping** — 2.8x SSD bandwidth with external drives
5. **Async prefetch overlapped with Metal GPU** — explicitly unfilled gap
