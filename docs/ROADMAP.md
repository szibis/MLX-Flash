# Roadmap — Gap Analysis & Implementation Plan

Based on competitive analysis of oMLX, SwiftLM, Flash-MoE, Ollama 0.19, and Anubis (April 2026).

## Status: What We Have vs What We Need

| Feature | MLX-Flash | oMLX | SwiftLM | Ollama | Priority |
|---------|-----------|------|---------|--------|----------|
| Expert streaming (MoE) | YES | no | SSD→GPU | no | done |
| Multi-precision (FP16→Q2) | YES (7 tiers) | no | TurboQuant | no | done |
| MCP tools | YES (6 tools) | no | no | no | done |
| Ollama API compat | YES | yes | no | native | done |
| Rust sidecar | YES | no | native Swift | no | done |
| Page cache (madvise) | YES | no | no | no | done |
| **Web Dashboard** | **v0.7** | YES (/admin) | no | no | **P0** |
| **Continuous batching** | planned | YES | no | YES | **P1** |
| **SSD KV cache** | planned | YES (2-tier) | YES | YES (prefix) | **P1** |
| **Prefix caching** | planned | no | no | YES (trie) | **P2** |
| **Hardware telemetry** | planned | partial | no | no | **P2** |
| **Community benchmarks** | planned | no | no | no | **P3** |
| Native Swift/iOS | no | no | YES | no | future |

## v0.7.0 — Dashboard + Real Model Testing

### Web Dashboard (/admin) — DONE
- Embedded HTML/CSS/JS served by Rust sidecar at `/admin`
- Live charts: memory pressure, tokens/sec (canvas, no dependencies)
- Cards: model info, hardware, cache hit rate, tokens generated
- Polls `/status` + `/cache/stats` every 2 seconds
- Optimization hints display
- Dark theme, responsive, works on mobile

### Real Model Testing Framework — TODO
- docker-compose: Rust server + Python worker + test runner
- Download Gemma 4 E2B (1.5GB, smallest) for CI testing
- E2E tests: MCP tool calls, OpenAI API, Ollama API, SSE streaming
- Measure: tok/s, TTFT, memory pressure, cache hit rate
- Results auto-published to docs/

## v0.8.0 — Continuous Batching + SSD KV Cache

### Continuous Batching
- Serve multiple concurrent requests efficiently
- Share KV cache prefix across requests with same system prompt
- Inspired by: oMLX BatchTurboQuantKVCache, vLLM-MLX

### SSD KV Cache Persistence
- 2-tier: hot KV in RAM, cold KV on SSD (safetensors format)
- On cache miss: restore from SSD instead of recomputing
- TTFT improvement: 30-90s → 1-3s for repeated prefixes
- Inspired by: oMLX, Apple's LLM-in-a-Flash paper

## v0.9.0 — Prefix Caching + Hardware Telemetry

### Prefix Caching (Trie-Based)
- Compressed prefix trie for KV cache sharing
- Exact match: reuse full KV state for identical prefixes
- Partial match: reuse longest matching prefix, recompute delta
- Huge win for coding agents (same system prompt per request)
- Inspired by: Ollama's prefix caching trie, SGLang RadixAttention

### Hardware Telemetry Dashboard
- IOReport-based sampling (GPU utilization, power, thermal)
- Live charts in /admin: GPU %, power W, temperature C
- Per-layer execution timing (which layers are IO-bound)
- Inspired by: Anubis OSS (11 live charts)

## v0.10.0 — DFlash + DDTree Speculative Decoding

### DFlash Block Diffusion Drafter
- Lightweight 5-layer drafter generates 15 tokens in ONE forward pass
- Conditioned on target model hidden states at checkpoint layers
- Block diffusion denoising (2 steps) produces draft candidates
- Target: 6x+ lossless acceleration on structured content (code, reasoning)
- Paper: arXiv:2602.06036 (ICLR 2026)
- Implementation: `mlx_flash_compress/dflash.py`

### DDTree (Dynamic Draft Trees)
- Build tree of candidates from DFlash logits (top-k branching)
- Tree attention mask for single-pass verification of entire tree
- Target: 96%+ acceptance rate → 6-9x effective speedup
- Implementation: `mlx_flash_compress/ddtree.py`

### DeepSeek V4 Flash MoE Support
- 284B total / 13B active — ideal for expert streaming
- MLX models: `mlx-community/DeepSeek-V4-Flash-4bit` (80 GB), `2bit-DQ` (35 GB)
- Expert streaming profile: ~256 experts, top-2 routing
- Benchmark script: `scripts/bench_deepseek_v4_flash.py`
- Target: Run on 36 GB Mac (2-bit) with expert streaming + DFlash

### Expert Prefetch from Draft Predictions
- DFlash drafts predict future tokens → infer which experts needed next
- Prefetch experts during target verification pass (overlapped I/O)
- Expected: +15-30% cache hit rate improvement from speculative prefetch

## v1.0.0 — Production Ready

### Community Benchmark Leaderboard
- `mlx-flash-bench --submit` posts results to central store
- Ranked by: tok/s, TTFT, power efficiency, memory efficiency
- Filterable by: chip, model, quantization, RAM

### Single Binary Distribution
- Rust binary embeds Python worker launch
- `mlx-flash` command starts everything (Rust + Python)
- Homebrew formula builds both Rust and Python
- Pre-built universal binaries for download

### Stability
- 95%+ test coverage on core library
- Real model integration tests in CI (Gemma 4 E2B)
- Fuzz testing on safetensors parser
- Memory leak testing with 24h continuous generation
