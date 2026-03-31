# Phase 1: Rust HTTP Sidecar + Python Test Coverage

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust HTTP sidecar that proxies to the Python MLX worker with native SSE streaming, macOS memory monitoring via Mach syscalls, and multi-client support. Also bring Python test coverage from 59 to 92+ tests.

**Architecture:** Rust axum server on :8080 proxies OpenAI-compatible requests to Python serve.py on :8081. Rust handles memory monitoring (mach2 syscalls, 0.1ms), SSE streaming, concurrent clients. Python handles MLX inference (95% of work). Both processes managed by Rust main.

**Tech Stack:** Rust (axum 0.8, tokio, reqwest, mach2, clap, serde_json, tracing), Python (existing mlx_flash_compress)

**Spec:** `docs/superpowers/specs/2026-03-31-rust-sidecar-design.md`

---

## File Structure

### Rust (new)

```
mlx-flash-server/
  Cargo.toml              # workspace root with dependencies
  src/
    main.rs               # CLI args, Python worker launch, server startup
    memory.rs             # host_statistics64 via mach2, MemoryState, PressureLevel
    server.rs             # axum routes: /status, /hints, /release, /health, /v1/*
    proxy.rs              # reqwest proxy to Python, SSE stream forwarding
  tests/
    memory_test.rs        # integration: memory reads match vm_stat
    proxy_test.rs         # integration: roundtrip through Rust to Python
```

### Python tests (new)

```
tests/
  test_memory_manager_hints.py   # optimization hints, auto-release, budget
  test_serve.py                  # server status, endpoints, error handling
  test_demo_warmup.py            # warm-up simulation, topic switching
  test_cached_inference.py       # router hook, event capture, cache sim
```

---

## Task 1: Scaffold Rust Project

**Files:**
- Create: `mlx-flash-server/Cargo.toml`
- Create: `mlx-flash-server/src/main.rs`

- [ ] **Step 1: Create Cargo project**

```bash
cd /Users/slawomirskowron/github/MLX-Flash-compress
mkdir -p mlx-flash-server/src
```

- [ ] **Step 2: Write Cargo.toml**

```toml
[package]
name = "mlx-flash-server"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["stream"] }
mach2 = "0.6"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio-stream = "0.1"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

- [ ] **Step 3: Write minimal main.rs that starts and exits**

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "mlx-flash-server", about = "Rust sidecar for MLX-Flash-Compress")]
struct Args {
    #[arg(long, default_value = "8080")]
    port: u16,
    #[arg(long, default_value = "8081")]
    python_port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    tracing_subscriber::fmt::init();
    tracing::info!("mlx-flash-server starting on {}:{}", args.host, args.port);
}
```

- [ ] **Step 4: Verify it compiles and runs**

Run: `cd mlx-flash-server && cargo build 2>&1`
Expected: `Compiling mlx-flash-server ...` then `Finished`

Run: `cargo run -- --help`
Expected: Shows `--port`, `--python-port`, `--host` options

- [ ] **Step 5: Commit**

```bash
git add mlx-flash-server/
git commit -m "feat(rust): scaffold mlx-flash-server Cargo project"
```

---

## Task 2: Memory Monitor (memory.rs)

**Files:**
- Create: `mlx-flash-server/src/memory.rs`
- Modify: `mlx-flash-server/src/main.rs`

- [ ] **Step 1: Write the failing test**

Add to bottom of `memory.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_memory_state_returns_valid_values() {
        let state = get_memory_state();
        assert!(state.total_gb > 0.0, "total_gb must be positive");
        assert!(state.free_gb >= 0.0);
        assert!(state.active_gb >= 0.0);
        assert!(state.total_gb > state.free_gb);
    }

    #[test]
    fn test_pressure_classification() {
        assert_eq!(PressureLevel::from_free_pct(80.0), PressureLevel::Normal);
        assert_eq!(PressureLevel::from_free_pct(25.0), PressureLevel::Warning);
        assert_eq!(PressureLevel::from_free_pct(5.0), PressureLevel::Critical);
    }

    #[test]
    fn test_available_gb_calculation() {
        let state = MemoryState {
            total_gb: 36.0, free_gb: 4.0, active_gb: 10.0,
            inactive_gb: 8.0, wired_gb: 4.0, compressed_gb: 2.0,
            swap_used_gb: 0.0, pressure: PressureLevel::Normal,
        };
        // available = free + 50% of inactive
        assert!((state.available_gb() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_optimization_hints_critical() {
        let state = MemoryState {
            total_gb: 36.0, free_gb: 1.0, active_gb: 20.0,
            inactive_gb: 5.0, wired_gb: 8.0, compressed_gb: 2.0,
            swap_used_gb: 5.0, pressure: PressureLevel::Critical,
        };
        let hints = state.optimization_hints();
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.priority == "critical"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-flash-server && cargo test -- memory 2>&1`
Expected: FAIL — `get_memory_state` not defined

- [ ] **Step 3: Implement memory.rs**

```rust
use mach2::kern_return::KERN_SUCCESS;
use mach2::mach_init::mach_host_self;
use mach2::vm_statistics::vm_statistics64;
use serde::Serialize;
use std::mem;

const HOST_VM_INFO64: i32 = 4;
const HOST_VM_INFO64_COUNT: u32 =
    (mem::size_of::<vm_statistics64>() / mem::size_of::<i32>()) as u32;

extern "C" {
    fn host_statistics64(
        host: u32,
        flavor: i32,
        info: *mut vm_statistics64,
        count: *mut u32,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PressureLevel {
    Normal,
    Warning,
    Critical,
}

impl PressureLevel {
    pub fn from_free_pct(pct: f64) -> Self {
        if pct >= 40.0 { Self::Normal }
        else if pct >= 15.0 { Self::Warning }
        else { Self::Critical }
    }
}

#[derive(Debug, Clone, Serialize)]
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

impl MemoryState {
    pub fn available_gb(&self) -> f64 {
        (self.free_gb + self.inactive_gb * 0.5).max(0.0)
    }

    pub fn optimization_hints(&self) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();
        if self.pressure == PressureLevel::Critical {
            hints.push(OptimizationHint {
                priority: "critical".into(),
                action: "reduce_cache".into(),
                message: "Memory pressure critical. Reduce cache or close apps.".into(),
            });
        }
        if self.swap_used_gb > 2.0 {
            hints.push(OptimizationHint {
                priority: "warning".into(),
                action: "close_apps".into(),
                message: format!("{:.1}GB in swap. Close unused apps.", self.swap_used_gb),
            });
        }
        if self.pressure == PressureLevel::Normal && self.available_gb() > 8.0 {
            hints.push(OptimizationHint {
                priority: "info".into(),
                action: "expand_cache".into(),
                message: format!("{:.1}GB available. Cache could grow.", self.available_gb()),
            });
        }
        hints
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationHint {
    pub priority: String,
    pub action: String,
    pub message: String,
}

/// Read macOS memory state via host_statistics64 Mach syscall.
/// Cost: ~0.1ms (vs Python subprocess 21ms).
pub fn get_memory_state() -> MemoryState {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as f64;
    let total_bytes = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) } as f64 * page_size;

    let mut info: vm_statistics64 = unsafe { mem::zeroed() };
    let mut count = HOST_VM_INFO64_COUNT;

    let kr = unsafe {
        host_statistics64(mach_host_self(), HOST_VM_INFO64, &mut info as *mut _, &mut count)
    };

    let to_gb = |pages: u64| -> f64 { pages as f64 * page_size / (1024.0 * 1024.0 * 1024.0) };

    if kr == KERN_SUCCESS {
        let free = to_gb(info.free_count as u64);
        let total = total_bytes / (1024.0 * 1024.0 * 1024.0);
        let free_pct = if total > 0.0 { free / total * 100.0 } else { 50.0 };

        MemoryState {
            total_gb: total,
            free_gb: free,
            active_gb: to_gb(info.active_count as u64),
            inactive_gb: to_gb(info.inactive_count as u64),
            wired_gb: to_gb(info.wire_count as u64),
            compressed_gb: to_gb(info.compressor_page_count as u64),
            swap_used_gb: 0.0, // TODO: read from sysctl
            pressure: PressureLevel::from_free_pct(free_pct),
        }
    } else {
        // Fallback
        MemoryState {
            total_gb: total_bytes / (1024.0 * 1024.0 * 1024.0),
            free_gb: 0.0, active_gb: 0.0, inactive_gb: 0.0,
            wired_gb: 0.0, compressed_gb: 0.0, swap_used_gb: 0.0,
            pressure: PressureLevel::Warning,
        }
    }
}
```

Add `mod memory;` to `main.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mlx-flash-server && cargo test -- memory 2>&1`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-flash-server/src/memory.rs mlx-flash-server/src/main.rs
git commit -m "feat(rust): macOS memory monitor via mach2 host_statistics64"
```

---

## Task 3: Axum HTTP Server with Status Endpoints

**Files:**
- Create: `mlx-flash-server/src/server.rs`
- Modify: `mlx-flash-server/src/main.rs`

- [ ] **Step 1: Write failing test for /status endpoint**

Add to bottom of `server.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_status_returns_200() {
        let app = create_router(AppState::default());
        let resp = app.oneshot(
            Request::builder().uri("/status").body(Body::empty()).unwrap()
        ).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status_contains_memory() {
        let app = create_router(AppState::default());
        let resp = app.oneshot(
            Request::builder().uri("/status").body(Body::empty()).unwrap()
        ).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["memory"]["total_gb"].as_f64().unwrap() > 0.0);
        assert!(json["memory"]["pressure"].is_string());
    }

    #[tokio::test]
    async fn test_hints_returns_array() {
        let app = create_router(AppState::default());
        let resp = app.oneshot(
            Request::builder().uri("/hints").body(Body::empty()).unwrap()
        ).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["hints"].is_array());
    }

    #[tokio::test]
    async fn test_health_returns_ok() {
        let app = create_router(AppState::default());
        let resp = app.oneshot(
            Request::builder().uri("/health").body(Body::empty()).unwrap()
        ).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_models_returns_list() {
        let app = create_router(AppState::default());
        let resp = app.oneshot(
            Request::builder().uri("/v1/models").body(Body::empty()).unwrap()
        ).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["data"].is_array());
    }

    #[tokio::test]
    async fn test_cors_headers() {
        let app = create_router(AppState::default());
        let resp = app.oneshot(
            Request::builder().uri("/status").body(Body::empty()).unwrap()
        ).await.unwrap();
        assert!(resp.headers().contains_key("access-control-allow-origin"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-flash-server && cargo test -- server 2>&1`
Expected: FAIL — `create_router` not defined

- [ ] **Step 3: Implement server.rs**

```rust
use axum::{
    extract::State,
    http::{header, Method, StatusCode},
    response::Json,
    routing::get,
    Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};

use crate::memory;

#[derive(Clone)]
pub struct AppState {
    pub python_port: u16,
    pub model_name: String,
    pub start_time: Instant,
    pub request_count: Arc<std::sync::atomic::AtomicU64>,
    pub tokens_generated: Arc<std::sync::atomic::AtomicU64>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            python_port: 8081,
            model_name: "local".into(),
            start_time: Instant::now(),
            request_count: Arc::new(0.into()),
            tokens_generated: Arc::new(0.into()),
        }
    }
}

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/status", get(handle_status))
        .route("/health", get(handle_status))
        .route("/hints", get(handle_hints))
        .route("/release", get(handle_release))
        .route("/v1/models", get(handle_models))
        .layer(cors)
        .with_state(state)
}

async fn handle_status(State(state): State<AppState>) -> Json<Value> {
    let mem = memory::get_memory_state();
    let uptime = state.start_time.elapsed().as_secs();
    let reqs = state.request_count.load(std::sync::atomic::Ordering::Relaxed);
    let toks = state.tokens_generated.load(std::sync::atomic::Ordering::Relaxed);

    Json(json!({
        "model": state.model_name,
        "memory": {
            "total_gb": mem.total_gb,
            "free_gb": mem.free_gb,
            "available_gb": mem.available_gb(),
            "pressure": mem.pressure,
            "swap_used_gb": mem.swap_used_gb,
        },
        "stats": {
            "requests": reqs,
            "tokens_generated": toks,
            "uptime_s": uptime,
        },
        "optimization_hints": mem.optimization_hints(),
    }))
}

async fn handle_hints(State(_state): State<AppState>) -> Json<Value> {
    let mem = memory::get_memory_state();
    Json(json!({ "hints": mem.optimization_hints() }))
}

async fn handle_release() -> Json<Value> {
    // Memory release is a Python/MLX operation — signal the worker
    Json(json!({ "action": "signaled", "note": "GPU memory release requires Python worker" }))
}

async fn handle_models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "mlx-flash-compress",
        }]
    }))
}
```

Add `tower-http = { version = "0.6", features = ["cors"] }` to Cargo.toml dependencies.
Add `mod server;` to `main.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mlx-flash-server && cargo test -- server 2>&1`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-flash-server/
git commit -m "feat(rust): axum HTTP server with /status, /hints, /health, /v1/models"
```

---

## Task 4: HTTP Proxy to Python Worker

**Files:**
- Create: `mlx-flash-server/src/proxy.rs`
- Modify: `mlx-flash-server/src/server.rs`

- [ ] **Step 1: Write failing test for proxy request forwarding**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_build_proxy_url() {
        let url = build_proxy_url(8081, "/v1/chat/completions");
        assert_eq!(url, "http://127.0.0.1:8081/v1/chat/completions");
    }

    #[tokio::test]
    async fn test_proxy_error_when_worker_down() {
        let client = reqwest::Client::new();
        let result = proxy_request(&client, 19999, "/v1/chat/completions", b"{}").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_is_sse_request() {
        let body: serde_json::Value = serde_json::json!({"stream": true, "messages": []});
        assert!(is_stream_request(&body));

        let body2: serde_json::Value = serde_json::json!({"messages": []});
        assert!(!is_stream_request(&body2));
    }

    #[tokio::test]
    async fn test_parse_chat_request_validates_messages() {
        let body = serde_json::json!({"model": "local"});
        let result = validate_chat_request(&body);
        assert!(result.is_err());

        let body2 = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        let result2 = validate_chat_request(&body2);
        assert!(result2.is_ok());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mlx-flash-server && cargo test -- proxy 2>&1`
Expected: FAIL

- [ ] **Step 3: Implement proxy.rs**

```rust
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use reqwest::Client;
use serde_json::Value;
use std::convert::Infallible;
use tokio_stream::{Stream, StreamExt};

use crate::server::AppState;

pub fn build_proxy_url(port: u16, path: &str) -> String {
    format!("http://127.0.0.1:{}{}", port, path)
}

pub fn is_stream_request(body: &Value) -> bool {
    body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false)
}

pub fn validate_chat_request(body: &Value) -> Result<(), String> {
    match body.get("messages") {
        Some(Value::Array(arr)) if !arr.is_empty() => Ok(()),
        _ => Err("messages array is required and must not be empty".into()),
    }
}

pub async fn proxy_request(
    client: &Client, port: u16, path: &str, body: &[u8],
) -> Result<reqwest::Response, reqwest::Error> {
    let url = build_proxy_url(port, path);
    client
        .post(&url)
        .header("content-type", "application/json")
        .body(body.to_vec())
        .send()
        .await
}

/// Handler for /v1/chat/completions — proxy to Python or stream SSE
pub async fn handle_chat(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid JSON"}))).into_response(),
    };

    if let Err(e) = validate_chat_request(&parsed) {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e}))).into_response();
    }

    state.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let client = Client::new();
    let resp = match proxy_request(&client, state.python_port, "/v1/chat/completions", &body).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": format!("Python worker unavailable: {}", e)}))).into_response(),
    };

    if is_stream_request(&parsed) {
        // Forward SSE stream
        let stream = resp.bytes_stream().map(|chunk| {
            match chunk {
                Ok(bytes) => Ok(Event::default().data(String::from_utf8_lossy(&bytes).to_string())),
                Err(e) => Ok(Event::default().data(format!("error: {}", e))),
            }
        });
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        // Forward complete response
        let status = resp.status();
        let body_bytes = resp.bytes().await.unwrap_or_default();
        Response::builder()
            .status(status.as_u16())
            .header("content-type", "application/json")
            .header("access-control-allow-origin", "*")
            .body(Body::from(body_bytes))
            .unwrap()
    }
}
```

Wire into server.rs: add `use crate::proxy; .route("/v1/chat/completions", axum::routing::post(proxy::handle_chat))`

- [ ] **Step 4: Run tests**

Run: `cd mlx-flash-server && cargo test -- proxy 2>&1`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-flash-server/
git commit -m "feat(rust): HTTP proxy to Python worker with SSE forwarding"
```

---

## Task 5: CLI and Python Worker Launch

**Files:**
- Modify: `mlx-flash-server/src/main.rs`

- [ ] **Step 1: Implement main.rs with worker management**

```rust
use clap::Parser;
use std::process::{Child, Command};
use tokio::net::TcpListener;

mod memory;
mod proxy;
mod server;

#[derive(Parser)]
#[command(name = "mlx-flash-server", about = "Rust sidecar for MLX-Flash-Compress")]
struct Args {
    #[arg(long, default_value = "8080")]
    port: u16,
    #[arg(long, default_value = "8081")]
    python_port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value = "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")]
    model: String,
    #[arg(long, help = "Launch Python worker automatically")]
    launch_worker: bool,
    #[arg(long, help = "Preload model in Python worker")]
    preload: bool,
}

fn launch_python_worker(port: u16, model: &str, preload: bool) -> Option<Child> {
    let mut cmd = Command::new("python3");
    cmd.args(["-m", "mlx_flash_compress.serve",
              "--port", &port.to_string(),
              "--host", "127.0.0.1",
              "--model", model]);
    if preload {
        cmd.arg("--preload");
    }
    match cmd.spawn() {
        Ok(child) => {
            tracing::info!("Python worker started (PID {})", child.id());
            Some(child)
        }
        Err(e) => {
            tracing::error!("Failed to start Python worker: {}", e);
            None
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    tracing_subscriber::fmt::init();

    // Print startup info
    let mem = memory::get_memory_state();
    tracing::info!("Memory: {:.1}GB total, {:.1}GB free, pressure: {:?}",
        mem.total_gb, mem.free_gb, mem.pressure);

    let mut _worker: Option<Child> = None;
    if args.launch_worker {
        _worker = launch_python_worker(args.python_port, &args.model, args.preload);
        // Wait briefly for Python to start
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    let state = server::AppState {
        python_port: args.python_port,
        model_name: args.model,
        ..Default::default()
    };

    let app = server::create_router(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Listening on http://{}", addr);
    tracing::info!("Python worker at http://127.0.0.1:{}", args.python_port);
    tracing::info!("Endpoints: /v1/chat/completions, /status, /hints, /health");

    axum::serve(listener, app)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); })
        .await
        .unwrap();

    tracing::info!("Shutting down");
}
```

- [ ] **Step 2: Build and test startup**

Run: `cd mlx-flash-server && cargo build 2>&1`
Expected: builds successfully

Run: `cargo run -- --help`
Expected: shows all CLI options

- [ ] **Step 3: Commit**

```bash
git add mlx-flash-server/src/main.rs
git commit -m "feat(rust): CLI with Python worker launch and graceful shutdown"
```

---

## Task 6: Python Test — Memory Manager Hints

**Files:**
- Create: `tests/test_memory_manager_hints.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for memory manager optimization hints and auto-release."""
import pytest
from unittest.mock import patch, MagicMock
from mlx_flash_compress.memory_manager import (
    MemoryManager, MemoryState, get_memory_state,
)


class TestOptimizationHints:
    def test_hints_normal_plenty_of_ram(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        with patch('mlx_flash_compress.memory_manager.get_memory_state') as mock:
            mock.return_value = MemoryState(
                total_gb=36.0, free_gb=10.0, active_gb=8.0,
                inactive_gb=10.0, wired_gb=4.0, compressed_gb=2.0,
                swap_used_gb=0.0, pressure_level="normal",
            )
            hints = mgr.get_optimization_hints()
            priorities = [h["priority"] for h in hints]
            assert "critical" not in priorities

    def test_hints_critical_pressure(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        with patch('mlx_flash_compress.memory_manager.get_memory_state') as mock:
            mock.return_value = MemoryState(
                total_gb=36.0, free_gb=1.0, active_gb=25.0,
                inactive_gb=2.0, wired_gb=6.0, compressed_gb=2.0,
                swap_used_gb=5.0, pressure_level="critical",
            )
            hints = mgr.get_optimization_hints()
            assert any(h["priority"] == "critical" for h in hints)

    def test_hints_high_swap(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        with patch('mlx_flash_compress.memory_manager.get_memory_state') as mock:
            mock.return_value = MemoryState(
                total_gb=36.0, free_gb=4.0, active_gb=15.0,
                inactive_gb=8.0, wired_gb=5.0, compressed_gb=2.0,
                swap_used_gb=8.0, pressure_level="warning",
            )
            hints = mgr.get_optimization_hints()
            assert any("swap" in h["message"].lower() for h in hints)

    def test_auto_release_normal_does_nothing(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        result = mgr.auto_release_if_needed()
        assert result["action"] == "none"

    def test_pressure_level_detection(self):
        state = get_memory_state()
        assert state.pressure_level in ("normal", "warning", "critical")

    def test_budget_adjusts_with_pressure(self):
        mgr = MemoryManager(safety_margin_gb=2.0)
        budget1 = mgr.get_cache_budget_gb()
        assert budget1 > 0
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_memory_manager_hints.py -v`
Expected: 6 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_memory_manager_hints.py
git commit -m "test: memory manager optimization hints and auto-release"
```

---

## Task 7: Python Test — Server Endpoints

**Files:**
- Create: `tests/test_serve.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the inference server (no model loading)."""
import json
import pytest
from unittest.mock import patch, MagicMock
from mlx_flash_compress.serve import InferenceState


class TestInferenceState:
    def test_status_returns_valid_structure(self):
        state = InferenceState("test-model")
        status = state.get_status()
        assert "model" in status
        assert "hardware" in status
        assert "memory" in status
        assert "stats" in status
        assert "optimization_hints" in status

    def test_status_memory_has_required_fields(self):
        state = InferenceState("test-model")
        mem = state.get_status()["memory"]
        for field in ["total_gb", "free_gb", "available_gb", "pressure", "cache_budget_gb"]:
            assert field in mem, f"Missing field: {field}"

    def test_status_pressure_is_valid(self):
        state = InferenceState("test-model")
        pressure = state.get_status()["memory"]["pressure"]
        assert pressure in ("normal", "warning", "critical")

    def test_stats_start_at_zero(self):
        state = InferenceState("test-model")
        stats = state.get_status()["stats"]
        assert stats["requests"] == 0
        assert stats["tokens_generated"] == 0

    def test_generate_without_model_loads_model(self):
        """Verify generate triggers model loading (we don't actually load)."""
        state = InferenceState("nonexistent-model")
        # Should fail trying to load a nonexistent model
        with pytest.raises(Exception):
            state.generate([{"role": "user", "content": "hi"}], max_tokens=5)

    def test_format_messages_fallback(self):
        state = InferenceState("test")
        state.tokenizer = MagicMock()
        state.tokenizer.apply_chat_template = MagicMock(side_effect=Exception("nope"))
        result = state._format_messages([{"role": "user", "content": "hello"}])
        assert "hello" in result

    def test_optimization_hints_in_status(self):
        state = InferenceState("test")
        hints = state.get_status()["optimization_hints"]
        assert isinstance(hints, list)

    def test_model_name_in_status(self):
        state = InferenceState("my-custom-model")
        assert state.get_status()["model"] == "my-custom-model"
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_serve.py -v`
Expected: 8 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_serve.py
git commit -m "test: server endpoint structure and status validation"
```

---

## Task 8: Python Test — Demo Warmup

**Files:**
- Create: `tests/test_demo_warmup.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the warm-up demo simulation."""
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path
from mlx_flash_compress.demo_warmup import (
    create_expert_files, make_topic_routing, simulate_token,
    WarmupSession, run_warmup_session,
)
from mlx_flash_compress.lcp_cache import LCPCache


class TestExpertFiles:
    def test_create_expert_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_dir = create_expert_files(tmpdir, num_layers=2, num_experts=4,
                                             expert_size_bytes=1024)
            files = list(Path(expert_dir).rglob("*.bin"))
            assert len(files) == 8  # 2 layers * 4 experts

    def test_expert_file_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_dir = create_expert_files(tmpdir, num_layers=1, num_experts=2,
                                             expert_size_bytes=512)
            for f in Path(expert_dir).rglob("*.bin"):
                assert f.stat().st_size == 512


class TestTopicRouting:
    def test_different_topics_different_distributions(self):
        rng = np.random.default_rng(42)
        p1 = make_topic_routing("coding", 60, rng)
        p2 = make_topic_routing("writing", 60, rng)
        assert p1.shape == (60,)
        assert abs(p1.sum() - 1.0) < 1e-6
        # Different topics should have different hot experts
        top_coding = set(np.argsort(p1)[-10:])
        top_writing = set(np.argsort(p2)[-10:])
        assert top_coding != top_writing

    def test_same_topic_same_distribution(self):
        rng = np.random.default_rng(42)
        p1 = make_topic_routing("coding", 60, rng)
        p2 = make_topic_routing("coding", 60, rng)
        np.testing.assert_array_equal(p1, p2)


class TestWarmupSession:
    def test_session_records_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_dir = create_expert_files(tmpdir, num_layers=4, num_experts=8,
                                             expert_size_bytes=256)
            cache = LCPCache(str(expert_dir), capacity_bytes=4096,
                             enable_dendritic=False, enable_skip_fallback=False)
            rng = np.random.default_rng(42)
            session = run_warmup_session(cache, "test", num_tokens=10,
                                         num_layers=4, num_experts=8, k=2,
                                         ssd_latency_ms=0, rng=rng,
                                         show_every=100)
            assert len(session.token_metrics) == 10
            assert session.topic == "test"
            cache.shutdown()
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_demo_warmup.py -v`
Expected: 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_demo_warmup.py
git commit -m "test: warm-up demo simulation, topic routing, session tracking"
```

---

## Task 9: Python Test — Cached Inference Router Hook

**Files:**
- Create: `tests/test_cached_inference.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for cached inference router hooks and cache simulation."""
import pytest
from collections import defaultdict
from mlx_flash_compress.cached_inference import (
    ExpertRouter, RoutingEvent, CacheSimState,
)


class TestCacheSimState:
    def test_empty_cache_all_misses(self):
        cache = CacheSimState(capacity_experts=10)
        events = [RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0)]
        hits, misses = cache.process_token(events)
        assert hits == 0
        assert misses == 2

    def test_second_access_is_hit(self):
        cache = CacheSimState(capacity_experts=10)
        events = [RoutingEvent(layer_idx=0, expert_indices=[1], token_idx=0, timestamp=0)]
        cache.process_token(events)
        hits, misses = cache.process_token(events)
        assert hits == 1
        assert misses == 0

    def test_capacity_enforcement(self):
        cache = CacheSimState(capacity_experts=2)
        e1 = [RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0)]
        cache.process_token(e1)
        assert len(cache.cached) == 2
        e2 = [RoutingEvent(layer_idx=0, expert_indices=[3], token_idx=1, timestamp=0)]
        cache.process_token(e2)
        assert len(cache.cached) <= 2  # evicted one to make room

    def test_hit_rate_calculation(self):
        cache = CacheSimState(capacity_experts=100)
        events = [RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0)]
        cache.process_token(events)
        cache.process_token(events)
        assert cache.hit_rate == 0.5  # 2 hits out of 4 total

    def test_lcp_eviction_prefers_cold(self):
        cache = CacheSimState(capacity_experts=2)
        # Access expert 1 many times (hot)
        for i in range(5):
            cache.process_token([
                RoutingEvent(layer_idx=0, expert_indices=[1], token_idx=i, timestamp=0)
            ])
        # Access expert 2 once (cold)
        cache.process_token([
            RoutingEvent(layer_idx=0, expert_indices=[2], token_idx=5, timestamp=0)
        ])
        # Now insert expert 3 — should evict 2 (cold), not 1 (hot)
        cache.process_token([
            RoutingEvent(layer_idx=0, expert_indices=[3], token_idx=6, timestamp=0)
        ])
        assert (0, 1) in cache.cached  # hot expert 1 survives
        assert (0, 2) not in cache.cached  # cold expert 2 evicted


class TestExpertRouter:
    def test_router_initializes(self):
        router = ExpertRouter()
        assert router.token_counter == 0
        assert len(router.events) == 0

    def test_get_expert_frequencies(self):
        router = ExpertRouter()
        router.events = [
            RoutingEvent(layer_idx=0, expert_indices=[1, 2], token_idx=0, timestamp=0),
            RoutingEvent(layer_idx=0, expert_indices=[1, 3], token_idx=1, timestamp=0),
        ]
        freqs = router.get_expert_frequencies()
        assert freqs[(0, 1)] == 2
        assert freqs[(0, 2)] == 1
        assert freqs[(0, 3)] == 1
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_cached_inference.py -v`
Expected: 7 tests PASS (may need to adjust if CacheSimState internals differ)

- [ ] **Step 3: Commit**

```bash
git add tests/test_cached_inference.py
git commit -m "test: cached inference cache simulation and router events"
```

---

## Task 10: Integration Test — Full Rust+Python Roundtrip

**Files:**
- Create: `mlx-flash-server/tests/e2e_roundtrip.rs`
- Create: `tests/e2e_full_pipeline.sh`

- [ ] **Step 1: Write E2E shell script**

```bash
#!/bin/bash
# E2E test: Rust sidecar + Python worker
set -e

echo "=== E2E: Full Pipeline Test ==="

# Start Python worker
echo "Starting Python worker on :8081..."
.venv/bin/python -m mlx_flash_compress.serve --port 8081 --host 127.0.0.1 &
PYTHON_PID=$!
sleep 3

# Start Rust sidecar
echo "Starting Rust sidecar on :8080..."
./mlx-flash-server/target/release/mlx-flash-server --port 8080 --python-port 8081 &
RUST_PID=$!
sleep 1

cleanup() {
    kill $RUST_PID 2>/dev/null || true
    kill $PYTHON_PID 2>/dev/null || true
}
trap cleanup EXIT

# Test /status
echo "Testing /status..."
STATUS=$(curl -s http://localhost:8080/status)
echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['memory']['total_gb'] > 0; print('  /status OK')"

# Test /hints
echo "Testing /hints..."
HINTS=$(curl -s http://localhost:8080/hints)
echo "$HINTS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'hints' in d; print('  /hints OK')"

# Test /v1/models
echo "Testing /v1/models..."
MODELS=$(curl -s http://localhost:8080/v1/models)
echo "$MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['data']) > 0; print('  /v1/models OK')"

# Test /health
echo "Testing /health..."
curl -sf http://localhost:8080/health > /dev/null && echo "  /health OK"

echo "=== All E2E tests passed ==="
```

- [ ] **Step 2: Make executable and test (requires both servers running)**

```bash
chmod +x tests/e2e_full_pipeline.sh
```

- [ ] **Step 3: Commit**

```bash
git add tests/e2e_full_pipeline.sh
git commit -m "test: E2E pipeline test — Rust sidecar + Python worker roundtrip"
```

---

## Task 11: Run Full Test Suite and Benchmark

- [ ] **Step 1: Run all Python tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: 85+ tests PASS (59 existing + 26 new)

- [ ] **Step 2: Run all Rust tests**

Run: `cd mlx-flash-server && cargo test 2>&1`
Expected: 14+ tests PASS

- [ ] **Step 3: Benchmark memory check latency**

```bash
cd mlx-flash-server && cargo run -- --port 8080 &
sleep 1
# Measure Rust memory check
for i in $(seq 1 10); do
  curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8080/status
done
kill %1

# Compare with Python
.venv/bin/python -c "
import time
from mlx_flash_compress.memory_manager import get_memory_state
times = []
for _ in range(10):
    t0 = time.monotonic()
    get_memory_state()
    times.append((time.monotonic() - t0) * 1000)
print(f'Python avg: {sum(times)/len(times):.1f}ms')
"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Phase 1 complete — Rust sidecar with 85+ Python tests and 14+ Rust tests"
git push origin main
```
