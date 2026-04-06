# Changelog

All notable changes to MLX-Flash are documented here.

## [Unreleased] — v0.6.0

### Added
- **Gemma 4 as default model** — chat auto-detects best Gemma 4 model (E2B/E4B/26B MoE/31B) for your hardware
- **Page cache control** — `madvise(MADV_FREE)` via ctypes for macOS page cache eviction of expert weights
- **Pipelined execution** — phase-level IO/compute overlap (prefetch attention during norm, prefetch MLP during attention)
- **Metal kernel acceleration** — custom `.metal` shaders: fused Q4 dequant+GEMV, SwiGLU fusion, MoE expert dispatch
- **Bit-parity verification** — FP32 accumulation proves streaming output matches standard MLX exactly (0.0 delta)
- **mlx-lm monkey patching** — `apply_flash_patch()` transparently enables Flash mode for LM Studio and any mlx-lm user
- **Gemma 4 benchmark suite** — `python -m mlx_flash_compress.bench_gemma4` tests all Gemma 4 sizes on your hardware
- **Comprehensive test suite** — 45 new tests for page_cache, pipeline, bit_parity, mlx_lm_patch, kernels, chat auto-select
- **Coverage reporting** — CI now reports test coverage with 90%+ target
- **Docker and Homebrew install** — instructions in main README Quick Start
- **GitHub Release badge** — auto-updated from latest release tag
- **Homebrew version badge** — auto-updated from version.json in tap repo

### Fixed
- **Homebrew release workflow** — GITHUB_TOKEN replaced with HOMEBREW_TAP_TOKEN PAT for cross-repo push
- **Homebrew auth chain** — explicit `git remote set-url` ensures token persists for push
- **Token validation** — workflow fails fast with clear error if HOMEBREW_TAP_TOKEN is not set

### Changed
- **README rewrite** — restructured for non-technical users: "Why?" comparison table, 3-step Quick Start, Supported Models
- **Technical deep-dives moved** to `docs/internals.md`: Core Modules, Research & Techniques, Competition, Key Discoveries
- **Performance docs updated** — new techniques documented with measured/estimated gains

## [0.5.1] — 2026-03-26

### Added
- Memory safety guards — prevent Mac crashes on large model loads
- Interactive command helper — tips, autocomplete hints, welcome guide
- vLLM-MLX integration adapter — auto-configure for vLLM backends
- Distributed experts — multi-Mac expert parallelism over Thunderbolt 5 RDMA
- KV-cache sharing — PT-MoE KV-cache sharing between blocks (37.5% savings)
- Model size calculator — estimate memory for any MoE or dense model

## [0.5.0] — 2026-03-25

### Added
- Web search — DuckDuckGo search + persistent memory (Perplexity-style local)
- Model catalog — `/models` browser and `/model N` live switching
- Colorful chat UI — download progress, memory bars, emoji status

## [0.4.0] — 2026-03-24

### Added
- Colorful chat UI with download progress, memory bars, emoji status
- Smart model defaults — Qwen3-8B for chat, auto-select for serve
- Upgraded default model to Qwen3-30B-A3B-4bit

### Fixed
- Audit fixes — LICENSE, old name refs, stale docs, MCP path

## [0.3.0] — 2026-03-23

### Added
- Model download progress bar
- Auto-release workflow + Docker GHCR + CI
- Integration examples in README with collapsible sections

### Fixed
- Hardware detection in CI VMs
- CI failures with lazy imports and optional dependencies
- PyPI-friendly readme with logo, badges, ASCII diagrams

## [0.2.1] — 2026-03-22

### Fixed
- PyPI packaging with setuptools backend
- Environment gate removed from PyPI release job

## [0.2.0] — 2026-03-21

### Added
- PyPI packaging, GitHub Actions CI, Homebrew tap, integration docs
- 9 new research techniques: entropy coding, shadow predictor, cross-layer prefetch, vertical split, expert merging
- Expert streaming — GPU lookup table + pre-stacked weights for Mixtral and Qwen
- Skip-fallback and profile-based warmup
- Rust sidecar — axum HTTP proxy, mach2 memory monitoring, DashMap LCP cache, Unix socket bridge
- Adaptive memory manager — safe RAM usage without harming user workloads
- Task-aware expert profiling — per-workload caching (coding/writing/math/chat)
- OpenAI-compatible API server with KV cache quantization
- Interactive chat CLI with web search and memory
- Model browser with hardware scoring
- 254 tests (222 Python + 32 Rust), 8 benchmark suites

## [0.1.0] — 2026-03-20

### Added
- Initial release
- LCP smart cache with layer-depth biased eviction
- LZ4/ZSTD/LZFSE compression
- Mixed precision (hot 4-bit, cold 2-bit)
- SSD protection (thermal cutoff, sequential hints, zero writes)
- Hardware auto-detection (M1-M4, RAM, GPU cores, SSD speed)
- Speculative expert execution with 97%+ prediction accuracy
- Measured: 2.80x speedup with smart cache, 2.93x with async prefetch
