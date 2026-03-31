# Rust Sidecar for MLX-Flash-Compress

## Summary

A Rust binary (`mlx-flash-server`) that sits in front of the Python MLX inference worker. Delivers two phases:

- **Phase 1**: HTTP proxy with native SSE streaming, macOS memory monitoring via Mach syscalls, and multi-client support.
- **Phase 2**: Expert weight cache manager using `mlx-rs` `gather_qmm`, with LCP eviction and async SSD prefetch. Communicates with Python via Unix socket + shared memory.

## Motivation

Python overhead in the current server:

| Component | Python | Rust |
|-----------|--------|------|
| Cold start | 1066ms | <50ms |
| Memory check | 21ms (subprocess) | 0.1ms (mach2 syscall) |
| Concurrent clients | 1 (GIL) | Unlimited (tokio) |
| SSE streaming | Word-chunked simulation | True SSE |
| Expert cache | GIL-bound dict | Lock-free concurrent |

MLX inference is 95% of request time and stays in Python (C++/Metal underneath). Rust handles the 5% that Python is bad at.

## Architecture

```
Clients (:8080)  -->  Rust Sidecar  -->  Python Worker (:8081 / .sock)
                      (axum + mlx-rs)    (mlx-lm generate)
```

### Phase 1: HTTP Proxy

Rust accepts OpenAI-compatible requests, proxies to Python, streams SSE back.

```
Client  --POST /v1/chat/completions-->  Rust (:8080)
                                         |
                                         |-- check memory (mach2, 0.1ms)
                                         |-- optimization hints
                                         |-- auto-release if critical
                                         |
                                         +--proxy--> Python (:8081)
                                         |           |
                                         |           +-- mlx-lm generate()
                                         |           |
                                         +<--SSE---- Python streams tokens
                                         |
Client  <--SSE stream--                  Rust forwards SSE
```

### Phase 2: Expert Cache with mlx-rs

Rust owns expert weight files on SSD. Python requests experts by index, Rust serves from cache or loads from SSD.

```
Python: "need experts [3,7,12,45] for layer 17"  -->  Unix socket  -->  Rust
Rust:   LCP cache lookup
        hit?  return from RAM (0.08ms)
        miss? async pread from SSD (0.6ms) -> cache -> return
Rust:   expert weight tensors                     -->  shared memory  -->  Python
Python: mx.array from pointer, feed into forward pass
```

The LCP eviction runs entirely in Rust with no GIL contention:
```
Priority(expert) = frequency * 0.25^(steps_since_last / 128)
```

## Crate Structure

```
mlx-flash-server/
  Cargo.toml
  src/
    main.rs              # CLI, startup, Python worker launch
    server.rs            # axum routes: /v1/chat/completions, /status, /hints, /release
    proxy.rs             # Phase 1: reqwest proxy to Python, SSE forwarding
    memory.rs            # host_statistics64() via mach2, pressure detection
    cache/
      mod.rs             # CacheManager: public API
      lcp.rs             # LCP eviction policy (lock-free with DashMap)
      prefetch.rs        # tokio::fs async SSD reads
      mixed_prec.rs      # mlx-rs quantize/dequantize for 4->2 bit
    expert_store.rs      # Phase 2: safetensors I/O, expert slicing
    protocol.rs          # Phase 2: Unix socket message types
```

## Dependencies

```toml
[dependencies]
axum = { version = "0.8", features = ["ws"] }
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["stream"] }
mlx-rs = { version = "0.25", features = ["safetensors", "metal"] }
mach2 = "0.6"
dashmap = "6"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio-stream = "0.1"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

## API Endpoints (Rust Server)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible, `stream: true` supported |
| `/v1/models` | GET | List available models |
| `/status` | GET | Memory, pressure, cache stats, hints |
| `/hints` | GET | Optimization recommendations |
| `/release` | GET | Trigger GPU memory release |
| `/health` | GET | Liveness check |
| `/cache/stats` | GET | Phase 2: LCP cache hit rate, entries, size |
| `/cache/warm` | POST | Phase 2: Pre-warm cache for a topic |

## Memory Monitor (memory.rs)

Replace Python's `subprocess("vm_stat")` + `subprocess("memory_pressure")` with direct Mach kernel calls:

```rust
use mach2::vm_statistics::vm_statistics64;
use mach2::kern_return::KERN_SUCCESS;
use mach2::mach_types::host_t;

extern "C" {
    fn host_statistics64(
        host: host_t,
        flavor: i32,
        info: *mut i32,
        count: *mut u32,
    ) -> i32;
}

pub struct MemoryState {
    pub total_gb: f64,
    pub free_gb: f64,
    pub active_gb: f64,
    pub inactive_gb: f64,
    pub wired_gb: f64,
    pub compressed_gb: f64,
    pub swap_used_gb: f64,
    pub pressure: PressureLevel,
}

pub enum PressureLevel {
    Normal,   // free > 40%
    Warning,  // free 15-40%
    Critical, // free < 15%
}
```

Cost: ~0.1ms per call vs Python's 21ms (210x faster). Can poll every 100ms without overhead.

## LCP Cache (cache/lcp.rs)

Lock-free expert cache using `DashMap`:

```rust
pub struct LcpCache {
    entries: DashMap<(u32, u32), CacheEntry>,  // (layer, expert) -> data
    capacity_bytes: AtomicUsize,
    current_bytes: AtomicUsize,
    step: AtomicU64,
}

struct CacheEntry {
    data: Vec<u8>,           // raw expert weight bytes
    frequency: AtomicU32,
    last_step: AtomicU64,
    size_bytes: usize,
}

impl LcpCache {
    pub fn fetch(&self, layer: u32, expert: u32) -> Option<&[u8]>;
    pub fn insert(&self, layer: u32, expert: u32, data: Vec<u8>);
    pub fn prefetch(&self, layer: u32, experts: &[u32]);  // async SSD
    pub fn evict_lowest(&self);
    fn priority(&self, entry: &CacheEntry) -> f64;
}
```

## Expert Weight Interception (Phase 2)

The Unix socket protocol between Rust and Python:

```rust
// Rust -> Python (or Python -> Rust)
enum Message {
    // Python asks Rust for expert weights
    FetchExperts { layer: u32, experts: Vec<u32>, request_id: u64 },
    // Rust returns expert data via shared memory
    ExpertData { request_id: u64, shm_offset: u64, sizes: Vec<u32> },
    // Python reports which experts were activated (for cache learning)
    RoutingReport { layer: u32, activated: Vec<u32>, token_idx: u64 },
    // Rust tells Python to apply mixed precision
    ApplyMixedPrecision { cold_experts: Vec<(u32, u32)> },
}
```

Shared memory region layout:
```
[header: 64 bytes]  [expert_0: N bytes]  [expert_1: N bytes]  ...
```

Python side reads expert weights via `mmap` and constructs `mx.array` from the raw pointer. MLX's unified memory model means this works without copying.

## Phase 1 Implementation Plan

1. **Scaffold Cargo project** in `mlx-flash-server/`
2. **memory.rs**: `host_statistics64` via mach2, `MemoryState` struct, pressure detection
3. **server.rs**: axum routes — `/status`, `/hints`, `/health`, `/release`
4. **proxy.rs**: reqwest proxy to Python `:8081`, SSE stream forwarding
5. **server.rs**: `/v1/chat/completions` with `stream: true` SSE support
6. **main.rs**: CLI with clap — `--port`, `--python-port`, `--model`, launch Python worker as child process
7. **Integration test**: start both processes, `curl` the Rust endpoint, verify SSE
8. **Benchmark**: compare cold start, memory check latency, concurrent requests vs Python-only

## Phase 2 Implementation Plan

1. **expert_store.rs**: Read safetensors expert weight files, slice 3D tensors per expert
2. **cache/lcp.rs**: DashMap-based LCP cache with eviction
3. **cache/prefetch.rs**: `tokio::fs::read` for async SSD reads
4. **cache/mixed_prec.rs**: mlx-rs `quantize`/`dequantize` for 4->2 bit conversion
5. **protocol.rs**: Unix socket message serialization, shared memory setup
6. **Python bridge**: Modify `cached_inference.py` to request experts from Rust via socket
7. **Integration test**: generate tokens, verify cache warm-up curve matches Python simulation
8. **Benchmark**: measure real SSD streaming vs simulated, cache hit rate, tok/s recovery

## Success Criteria

### Phase 1
- [ ] Cold start < 50ms (excl. model load)
- [ ] Memory check < 1ms
- [ ] SSE streaming works with LM Studio
- [ ] 10 concurrent clients without blocking
- [ ] `/status` returns memory + hints in < 1ms

### Phase 2
- [ ] Real expert weight loading from SSD (not simulated)
- [ ] LCP cache hit rate > 80% at steady state
- [ ] Cache warm-up visible: first token slower, converges in < 30 tokens
- [ ] Mixed precision auto-applied when pressure detected
- [ ] 50GB+ model runs on 36GB Mac via SSD streaming (the ultimate test)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| mlx-rs `gather_qmm` untested at scale | Phase 2 blocked | Test in Phase 1, have Python fallback |
| Shared memory bridge complex | Phase 2 delayed | Start with Unix socket JSON, upgrade to mmap |
| Python MLX can't accept external arrays | Phase 2 blocked | MLX unified memory + `mx.array` from pointer should work |
| Two-process coordination failures | Reliability | Rust monitors Python health, auto-restarts |
