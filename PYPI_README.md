<p align="center">
  <img src="https://raw.githubusercontent.com/szibis/MLX-Flash/main/assets/logo.svg" width="200" alt="MLX-Flash Logo" />
</p>

<h1 align="center">MLX-Flash</h1>

<p align="center"><strong>Run AI models too large for your Mac's memory — at near-full speed.</strong></p>

<p align="center">
  <a href="https://github.com/szibis/MLX-Flash"><img src="https://img.shields.io/github/stars/szibis/MLX-Flash?style=social" alt="GitHub Stars" /></a>
  <a href="https://pypi.org/project/mlx-flash/"><img src="https://img.shields.io/pypi/v/mlx-flash" alt="PyPI" /></a>
  <a href="https://github.com/szibis/MLX-Flash/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License" /></a>
</p>

---

Your MacBook has 32-48GB of RAM, but the best AI models need 100-200GB+. MLX-Flash makes them run anyway by intelligently caching the most-needed parts in RAM and streaming the rest from your SSD.

## How It Works

Think of it like Netflix streaming: instead of downloading the entire movie before watching, you buffer what you need and stream the rest. MLX-Flash does this for AI model weights:

```
┌─────────────────────────────────┐
│    Your Mac's RAM (fast)        │
│  ┌──────────┐ ┌──────────────┐  │
│  │Hot Cache  │ │Mixed Precis. │  │
│  │85%+ hits  │ │4-bit / 2-bit │  │
│  └──────────┘ └──────────────┘  │
└──────────────┬──────────────────┘
               │ cache hit: 0.08ms
┌──────────────┴──────────────────┐
│    Smart Cache Layer            │
│  • LCP Eviction (layer-biased) │
│  • Speculative Prefetch (97%)  │
│  • Memory Monitor              │
│  • Speculative Execution       │
└──────────────┬──────────────────┘
               │ cache miss: 0.6ms
┌──────────────┴──────────────────┐
│    Your SSD (big)               │
│  Full model weights — 200GB+   │
│  Entropy-coded — 65% smaller   │
└─────────────────────────────────┘
         │
         ▼
   MLX GPU Inference
```

**Result:** A 200GB AI model runs on your 48GB Mac at **2-3x faster** than naive SSD streaming.

## Quick Start

```bash
pip install mlx-flash
```

```bash
# Interactive chat
mlx-flash-chat

# API server (works with LM Studio, Cursor, Claude Code, Codex, OpenAI SDK)
mlx-flash --port 8080

# With KV cache quantization (45% less KV memory)
mlx-flash --port 8080 --kv-bits 8

# See what models fit your hardware
mlx-flash-browse
```

## Performance

| Technique | Speedup | How It Works |
|-----------|---------|-------------|
| **LCP Smart Cache** | **2.80x** | Keeps frequently-used model parts in RAM |
| **+ Async Prefetch** | **2.93x** | Loads next part from SSD while GPU computes |
| **Mixed Precision** | **1.80x smaller** | Rarely-used parts stored at lower quality |
| **Skip Fallback** | **2.67x** | Gracefully skip uncached parts instead of waiting |
| **Speculative Execution** | **14-42% TPOT** | Execute predicted experts before router confirms |
| **Adaptive Top-K** | **10-30% compute** | Skip low-confidence secondary experts |

### Real Hardware (M3 Max 36GB)

```
Memory pressure recovery:
  Without optimization:    43.5 tok/s
  With mixed precision:   104.5 tok/s  → 2.4x faster

Cache warm-up:
  Token  0:  83.3ms (cold start)
  Token  8:   5.7ms (warming up)
  Token 24:   0.5ms (full speed) → 41x speedup
```

## What's Inside

**35 Python modules + Rust sidecar** implementing 15+ research techniques:

| Category | Modules |
|----------|---------|
| **Expert Streaming** | GPU lookup tables, speculative execution, skip-fallback, adaptive top-k |
| **Prediction (97%+)** | Residual-stream predictor, shadow MLP, cross-layer 3-hop prefetch |
| **Cache Management** | Layer-biased LCP, Belady-optimal eviction, vertical splitting, expert merging |
| **Compression** | Entropy coding (Huffman uint4), mixed precision (4-bit/2-bit) |
| **Memory** | Real-time pressure monitoring, wired memory optimization, `mx.clear_cache()` |
| **Serving** | OpenAI-compatible API, KV cache 8-bit quantization, SSE streaming |
| **Rust Sidecar** | axum HTTP/SSE, mach2 memory (0.1ms), DashMap LCP, Unix socket bridge |

## Integration

Works with any OpenAI-compatible tool:

```bash
# Start server
mlx-flash --port 8080 --preload

# Point any tool at it
# LM Studio: Settings → Server → http://localhost:8080/v1
# Cursor: Settings → Models → OpenAI Compatible → http://localhost:8080/v1
# Claude Code: OPENAI_API_BASE=http://localhost:8080/v1
# continue.dev: apiBase: http://localhost:8080/v1
```

```python
# Python SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Expert Streaming (for large MoE models)

```python
from mlx_flash_compress.expert_streaming import (
    enable_expert_streaming, enable_skip_fallback
)

# Enable streaming with 50% capacity + adaptive skipping
streaming = enable_expert_streaming(model, capacity_per_layer=64)
enable_skip_fallback(model, streaming.caches, adaptive_skip_threshold=3.0)
streaming.warmup()
```

## Research Techniques Implemented

From 15+ papers (2024-2026):

| Technique | Paper | Status |
|-----------|-------|--------|
| Expert streaming (GPU lookup) | HOBBIT arXiv:2411.01433 | Implemented |
| Residual-stream predictor (97%+) | Speculating Experts arXiv:2603.19289 | Implemented |
| Speculative execution (14-42% TPOT) | MoE-SpAc arXiv:2603.09983 | Implemented |
| Belady-optimal eviction | MoE-SpeQ arXiv:2511.14102 | Implemented |
| Cross-layer 3-hop prefetch | FATE arXiv:2502.12224 | Implemented |
| Layer-depth cache bias | FATE arXiv:2502.12224 | Implemented |
| Vertical expert splitting (2x coverage) | MoEpic paper | Implemented |
| Expert merging (15-30% fewer params) | DEK/EEP arXiv:2509.19781 | Implemented |
| Entropy coding (65% compression) | EntroLLM arXiv:2505.02380 | Implemented |
| Adaptive top-k (10-30% compute savings) | LExI arXiv:2509.02753 | Implemented |
| Mixed precision per-expert | HOBBIT arXiv:2411.01433 | Implemented |
| KV cache 8-bit quantization | mlx-moe / mlx-lm | Implemented |

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4/M5)
- **Python 3.10+**
- 16GB+ RAM (more = better caching = faster)

## Project Stats

- **15,000+ lines of code** (Python + Rust)
- **224 tests** (192 Python + 32 Rust)
- **35 Python modules** + Rust sidecar
- **15+ research papers** implemented

## Links

- [GitHub](https://github.com/szibis/MLX-Flash)
- [Documentation](https://github.com/szibis/MLX-Flash/tree/main/docs)
- [Integration Guide](https://github.com/szibis/MLX-Flash/blob/main/docs/integrations.md)
- [Competitive Analysis](https://github.com/szibis/MLX-Flash/blob/main/docs/competitive-analysis.md)

## License

MIT
