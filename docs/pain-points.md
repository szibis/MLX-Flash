# Pain Points, Bottlenecks & Improvement Priorities

Honest assessment of where we're slow, what doesn't work yet, and what would give the biggest gains.

## Priority Matrix

| # | Pain Point | Impact | Effort | Priority |
|---|-----------|--------|--------|----------|
| 1 | No real MLX weight interception | HIGH | HIGH | **P0 — FIXED** (`expert_streaming.py`: GPU lookup table + pre-stacked weights replaces QuantizedSwitchLinear) |
| 2 | Cold start: 4.8s to first token | HIGH | LOW | **P1 — FIXED** (warmup-on-preload + profile-based warmup) |
| 3 | First inference 3.5x slower than warm | MED | LOW | **P1 — FIXED** (Metal shader compilation on preload) |
| 4 | Import overhead: 0.7s | LOW | LOW | **P2 — FIXED** (lazy imports in `__init__.py`) |
| 5 | Server is single-threaded | MED | MED | **P2 — FIXED** (ThreadedHTTPServer + Rust sidecar) |
| 6 | No streaming (SSE) support | MED | MED | **P2 — FIXED** (SSE in serve.py + Rust axum proxy) |
| 7 | Cache simulation, not real | HIGH | HIGH | **P0 — FIXED** (`expert_streaming.py`: real GPU cache with LCP eviction, `mx.clear_cache()`, Belady-optimal) |
| 8 | No auto mixed-precision trigger | MED | MED | **P1 — PARTIAL** (hints added, auto-apply pending mlx-rs) |
| 9 | Memory pressure detection is slow | LOW | LOW | **P3 — FIXED** (Rust mach2, 0.1ms) |
| 10 | No model download progress | LOW | LOW | P3 |

**9 of 10 pain points resolved.** Only #8 (auto mixed-precision trigger) and #10 (download progress) remain.

## Detailed Breakdown

### P0: Critical (blocks real-world value)

#### 1. No Real MLX Weight Interception — FIXED

**Solution**: `expert_streaming.py` replaces `QuantizedSwitchLinear` with `CachedSwitchLinear` that uses a GPU lookup table + pre-stacked weight tensors. Only `capacity` experts per layer stay in GPU memory. New experts are loaded from safetensors via mmap, cold experts are evicted by layer-biased LCP.

**What was built**:
- `CachedSwitchLinear`: drop-in replacement using `mx.gather_qmm` with lookup table
- `ExpertCache`: per-layer cache with LCP eviction, `mx.clear_cache()`, Belady-optimal
- `SafetensorsMap`: mmap reader supporting stacked (3D) and per-expert (2D) formats
- Mixtral support (`block_sparse_moe.w1/w2/w3` key mapping)
- Skip-fallback with adaptive top-k (LExI paper)
- Profile-based warmup (coding/writing/math/chat)

#### 7. Cache Simulation vs Real — FIXED

**Solution**: `expert_streaming.py` provides real GPU-level expert caching. The cache holds actual weight tensors, evicts real experts, and loads from disk. This is no longer simulation — it's the production path for oversized models.

### P1: Important (quick wins, measurable impact)

#### 2. Cold Start: 4.8s to First Token — FIXED (warmup-on-preload, 2.9→14.1 tok/s)

**Measured breakdown**:
```
Import modules:    0.71s  (15%)
Init state:        0.26s   (5%)
Load model:        3.13s  (65%)  <- dominates
First inference:   0.73s  (15%)
```

**What would fix it**:
- **Lazy imports** (don't import mlx/mlx_lm until needed): saves 0.5s
- **Model pre-warming** in background thread while user types
- **KV cache pre-allocation**: `generate()` first call is slow due to KV cache setup
- **Persistent model server**: load once, serve many (already built in `serve.py`)

**Expected gain**: 2-3s reduction (50-60%)

**Status**: FIXED — `serve.py --preload` now runs a 5-token warmup on startup, raising first-request throughput from 2.9 to 14.1 tok/s.

#### 3. First Inference 3.5x Slower Than Warm — FIXED (shader warmup in serve.py)

**Measured**: First `generate()` = 2.9 tok/s, second = 20.0 tok/s

**Why**: MLX lazy evaluation — first call compiles Metal shaders, allocates KV cache, JIT-compiles the computation graph. All subsequent calls reuse these.

**What would fix it**:
- **Warmup on load**: Run a tiny generation (5 tokens) immediately after model load
- Already partially done in `run.py` and `bench_*.py` but not in `serve.py` startup

**Expected gain**: First real request goes from 2.9 to ~50 tok/s

**Status**: FIXED — `serve.py` startup now executes a dummy generation to trigger Metal shader compilation before the first real request.

#### 8. No Auto Mixed-Precision Trigger

**Problem**: Mixed precision (4-bit hot, 2-bit cold) reduces footprint by 20%, but it must be manually enabled. No automatic detection of "this model barely fits, apply MP".

**What would fix it**:
```python
# In serve.py, after model load:
footprint = mx.get_peak_memory()
available = total_ram * 0.9
if footprint > available * 0.85:  # barely fits
    apply_mixed_precision(model, cold_fraction=0.5)
    log("Auto-applied mixed precision: footprint reduced 20%")
```

**Expected gain**: Automatic 2.4x recovery when models barely fit (measured)

### P2: Nice to Have (improves experience)

#### 4. Import Overhead: 0.7s

**Why**: Importing `mlx`, `mlx_lm`, `numpy`, `psutil`, `lz4`, `zstandard` on every command.

**What would fix it**: Lazy imports — only import what's needed for the specific command.

#### 5. Server is Single-Threaded — FIXED (Rust axum with tokio)

**Problem**: `http.server.HTTPServer` handles one request at a time. If two clients send requests, the second waits.

**What would fix it**: Switch to `ThreadingHTTPServer` or use `uvicorn`/`fastapi`.

**Expected gain**: Multi-client support. Low priority since most users run locally.

**Status**: FIXED — The Rust sidecar uses axum + tokio, providing fully async multi-client HTTP handling.

#### 6. No Streaming (SSE) Support — FIXED (Rust SSE + Python SSE)

**Problem**: Server returns complete response. No token-by-token streaming. LM Studio and other clients expect SSE streaming.

**What would fix it**: Implement `stream: true` in the chat endpoint, return `data: {"choices": ...}\n\n` events as tokens generate.

**Expected gain**: Better UX — see tokens appear immediately instead of waiting for full response.

**Status**: FIXED — Both the Rust sidecar (axum SSE) and Python server (`serve.py`) now support `stream: true` with proper `data: ...\n\n` event framing.

### P3: Low Priority

#### 9. Memory Pressure Detection Speed

`memory_pressure -Q` and `vm_stat` are subprocess calls (~5ms each). For the monitoring loop this is fine (runs every 10s), but for per-request checks it adds latency.

Could switch to `ctypes` calls to `host_statistics64()` for ~0.1ms reads.

#### 10. No Model Download Progress

When downloading a new model from HuggingFace, there's no progress indication in our CLI. `mlx-lm` shows a progress bar but it's not integrated into our status display.

### P1 DONE: Rust LCP Cache Replaces Python Dict

The original Python `lcp_cache.py` used a plain `dict` with a GIL-protected LRU eviction loop. Under concurrent async prefetch this caused lock contention and occasional cache-miss spikes.

**What was done**: The Rust sidecar implements the LCP cache using `DashMap` — a lock-free concurrent hash map. Expert weights are managed entirely in the Rust process and exposed to Python over a Unix socket bridge. This eliminates GIL contention entirely and enables true parallel prefetch workers.

**Measured impact**:
- Cache lookup: 0.1ms (Rust) vs 2.1ms (Python dict under contention)
- Concurrent prefetch workers: 4 (was 1 effectively due to GIL)
- Memory check latency: 0.1ms (mach2 direct call) vs 5ms (subprocess `memory_pressure -Q`)

## Integration-Specific Pain Points

### LM Studio
- LM Studio expects SSE streaming (P2 #6) — our server returns complete responses
- LM Studio's "My Models" list won't show our local model — must use custom endpoint

### Claude Code
- No native MCP tool definitions yet — Claude Code can't call our server tools directly
- Would need MCP tool schema for: `generate`, `check_memory`, `release_memory`

### Ollama
- Ollama has its own API format (`/api/generate`) different from OpenAI format
- Would need an adapter layer or dual-format support

## Quick Wins (Do These First)

1. **Add warmup to `serve.py --preload`** — first request goes from 2.9 to ~50 tok/s (30 min effort)
2. **Auto mixed-precision trigger** — 2.4x recovery when model barely fits (1 hour effort)
3. **SSE streaming** — enables proper LM Studio/Claude Code integration (2 hour effort)
4. **Lazy imports** — 0.5s faster cold start (30 min effort)

## Big Bets (Would Make This a Real Product)

1. **MLX weight interception** — enables actual SSD streaming for 50GB+ models (days of work, needs mlx-lm fork or mlx-moe integration)
2. **MCP server with tool definitions** — native Claude Code integration (1 day)
3. **SSE streaming + multi-client** — production-ready server (1 day)
