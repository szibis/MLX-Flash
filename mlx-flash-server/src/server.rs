use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::RwLock;

use axum::{Router, extract::State, response::IntoResponse, routing::get};
use axum::http::Method;
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

use crate::cache::LcpCache;
use crate::memory;
use crate::proxy;
use crate::worker_pool::WorkerPool;

#[derive(Clone)]
pub struct AppState {
    pub python_port: u16,
    pub model_name: Arc<RwLock<String>>,
    pub start_time: Instant,
    pub request_count: Arc<AtomicU64>,
    pub tokens_generated: Arc<AtomicU64>,
    pub cache: Option<Arc<LcpCache>>,
    pub pool: Arc<WorkerPool>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            python_port: 8081,
            model_name: Arc::new(RwLock::new("local".to_string())),
            start_time: Instant::now(),
            request_count: Arc::new(AtomicU64::new(0)),
            tokens_generated: Arc::new(AtomicU64::new(0)),
            cache: None,
            pool: Arc::new(WorkerPool::single(8081)),
        }
    }
}

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/admin", get(crate::dashboard::serve_dashboard))
        .route("/chat", get(crate::chat_ui::serve_chat))
        .route("/status", get(handle_status))
        .route("/health", get(handle_status))
        .route("/hints", get(handle_hints))
        .route("/release", get(handle_release))
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", axum::routing::post(proxy::handle_chat))
        .route("/cache/stats", get(handle_cache_stats))
        .route("/workers", get(handle_workers))
        .route("/v1/models/switch", axum::routing::post(handle_model_switch))
        .route("/metrics", get(handle_metrics))
        .with_state(state)
        .layer(cors)
}

async fn handle_status(State(state): State<AppState>) -> axum::Json<Value> {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    let memory = match memory::get_memory_state() {
        Ok(m) => serde_json::to_value(&m).unwrap_or(json!({})),
        Err(e) => json!({ "error": e }),
    };

    let uptime_secs = state.start_time.elapsed().as_secs_f64();
    let model_name = state.model_name.read().await.clone();

    axum::Json(json!({
        "model": model_name,
        "memory": memory,
        "stats": {
            "requests": state.request_count.load(Ordering::Relaxed),
            "tokens_generated": state.tokens_generated.load(Ordering::Relaxed),
            "uptime_secs": uptime_secs,
        },
        "workers": state.pool.status(),
        "optimization_hints": memory::get_memory_state()
            .map(|m| serde_json::to_value(m.optimization_hints()).unwrap_or(json!([])))
            .unwrap_or(json!([])),
    }))
}

async fn handle_hints(State(state): State<AppState>) -> axum::Json<Value> {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    let hints = memory::get_memory_state()
        .map(|m| serde_json::to_value(m.optimization_hints()).unwrap_or(json!([])))
        .unwrap_or(json!([]));

    axum::Json(json!({ "hints": hints }))
}

async fn handle_release(State(state): State<AppState>) -> axum::Json<Value> {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    axum::Json(json!({
        "action": "signaled",
        "note": "Memory release signal sent to Python worker.",
    }))
}

async fn handle_models(State(state): State<AppState>) -> axum::Json<Value> {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let model_name = state.model_name.read().await.clone();

    axum::Json(json!({
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "mlx-flash-compress",
            }
        ]
    }))
}

async fn handle_workers(State(state): State<AppState>) -> axum::Json<Value> {
    axum::Json(state.pool.status())
}

async fn handle_model_switch(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> axum::Json<Value> {
    // Parse request
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return axum::Json(json!({
                "error": format!("Invalid JSON: {e}"),
            }));
        }
    };

    let new_model = match parsed.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => {
            return axum::Json(json!({
                "error": "Missing required field: model",
                "usage": {"model": "mlx-community/Qwen3-8B-4bit"},
            }));
        }
    };

    let old_model = state.model_name.read().await.clone();

    // Forward switch request to all healthy workers
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300)) // model loading can be slow
        .build()
        .unwrap();

    let mut successes = 0;
    let mut failures = Vec::new();

    for port in state.pool.ports() {
        let url = format!("http://127.0.0.1:{port}/switch");
        let switch_body = serde_json::json!({"model": &new_model}).to_string();
        match client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(switch_body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                successes += 1;
            }
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                failures.push(json!({"port": port, "status": status, "error": body}));
                state.pool.mark_unhealthy(port);
            }
            Err(e) => {
                failures.push(json!({"port": port, "error": format!("{e}")}));
                state.pool.mark_unhealthy(port);
            }
        }
    }

    // Update model name if at least one worker succeeded
    if successes > 0 {
        *state.model_name.write().await = new_model.clone();
    }

    axum::Json(json!({
        "switched": successes > 0,
        "model": if successes > 0 { &new_model } else { &old_model },
        "previous": old_model,
        "workers_updated": successes,
        "workers_failed": failures.len(),
        "failures": failures,
    }))
}

async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    use std::fmt::Write;
    let mut out = String::with_capacity(4096);

    let model_name = state.model_name.read().await.clone();
    let uptime = state.start_time.elapsed().as_secs_f64();
    let requests = state.request_count.load(Ordering::Relaxed);
    let tokens = state.tokens_generated.load(Ordering::Relaxed);

    // -- Server info --
    let _ = write!(out, "# HELP mlx_flash_info Server metadata.\n");
    let _ = write!(out, "# TYPE mlx_flash_info gauge\n");
    let _ = write!(out, "mlx_flash_info{{model=\"{model_name}\"}} 1\n\n");

    let _ = write!(out, "# HELP mlx_flash_uptime_seconds Time since server start.\n");
    let _ = write!(out, "# TYPE mlx_flash_uptime_seconds gauge\n");
    let _ = write!(out, "mlx_flash_uptime_seconds {uptime:.1}\n\n");

    // -- Request counters --
    let _ = write!(out, "# HELP mlx_flash_requests_total Total inference requests.\n");
    let _ = write!(out, "# TYPE mlx_flash_requests_total counter\n");
    let _ = write!(out, "mlx_flash_requests_total {requests}\n\n");

    let _ = write!(out, "# HELP mlx_flash_tokens_generated_total Total tokens generated.\n");
    let _ = write!(out, "# TYPE mlx_flash_tokens_generated_total counter\n");
    let _ = write!(out, "mlx_flash_tokens_generated_total {tokens}\n\n");

    // -- Memory (macOS vm_statistics64) --
    if let Ok(mem) = memory::get_memory_state() {
        let total = mem.total_gb * 1073741824.0;
        let free = mem.free_gb * 1073741824.0;
        let active = mem.active_gb * 1073741824.0;
        let inactive = mem.inactive_gb * 1073741824.0;
        let wired = mem.wired_gb * 1073741824.0;
        let compressed = mem.compressed_gb * 1073741824.0;
        let swap = mem.swap_used_gb * 1073741824.0;
        let available = mem.available_gb() * 1073741824.0;
        let used_ratio = if mem.total_gb > 0.0 { 1.0 - mem.available_gb() / mem.total_gb } else { 0.0 };
        let pressure_val = match mem.pressure {
            memory::PressureLevel::Normal => 0,
            memory::PressureLevel::Warning => 1,
            memory::PressureLevel::Critical => 2,
        };

        let _ = write!(out, "# HELP mlx_flash_memory_total_bytes Total physical RAM.\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_total_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_total_bytes {total:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_free_bytes Free (unused) RAM.\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_free_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_free_bytes {free:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_available_bytes Usable RAM (free + 50% inactive).\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_available_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_available_bytes {available:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_active_bytes Active pages.\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_active_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_active_bytes {active:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_inactive_bytes Inactive pages (reclaimable).\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_inactive_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_inactive_bytes {inactive:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_wired_bytes Wired (non-evictable) pages.\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_wired_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_wired_bytes {wired:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_compressed_bytes Compressed pages.\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_compressed_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_compressed_bytes {compressed:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_swap_used_bytes Swap space in use.\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_swap_used_bytes gauge\n");
        let _ = write!(out, "mlx_flash_memory_swap_used_bytes {swap:.0}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_used_ratio Fraction of RAM in use (0-1).\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_used_ratio gauge\n");
        let _ = write!(out, "mlx_flash_memory_used_ratio {used_ratio:.4}\n\n");

        let _ = write!(out, "# HELP mlx_flash_memory_pressure macOS memory pressure (0=normal, 1=warning, 2=critical).\n");
        let _ = write!(out, "# TYPE mlx_flash_memory_pressure gauge\n");
        let _ = write!(out, "mlx_flash_memory_pressure {pressure_val}\n\n");
    }

    // -- Worker pool --
    let _ = write!(out, "# HELP mlx_flash_workers_total Total workers in pool.\n");
    let _ = write!(out, "# TYPE mlx_flash_workers_total gauge\n");
    let _ = write!(out, "mlx_flash_workers_total {}\n\n", state.pool.len());

    let _ = write!(out, "# HELP mlx_flash_workers_healthy Number of healthy workers.\n");
    let _ = write!(out, "# TYPE mlx_flash_workers_healthy gauge\n");
    let _ = write!(out, "mlx_flash_workers_healthy {}\n\n", state.pool.healthy_count());

    let _ = write!(out, "# HELP mlx_flash_worker_inflight Current in-flight requests per worker.\n");
    let _ = write!(out, "# TYPE mlx_flash_worker_inflight gauge\n");
    let _ = write!(out, "# HELP mlx_flash_worker_requests_total Total requests served per worker.\n");
    let _ = write!(out, "# TYPE mlx_flash_worker_requests_total counter\n");
    let _ = write!(out, "# HELP mlx_flash_worker_healthy Whether worker is healthy (1) or not (0).\n");
    let _ = write!(out, "# TYPE mlx_flash_worker_healthy gauge\n");

    let pool_status = state.pool.status();
    if let Some(workers) = pool_status["workers"].as_array() {
        for w in workers {
            let port = w["port"].as_u64().unwrap_or(0);
            let inflight = w["inflight"].as_u64().unwrap_or(0);
            let total_req = w["total_requests"].as_u64().unwrap_or(0);
            let healthy = if w["healthy"].as_bool().unwrap_or(false) { 1 } else { 0 };
            let _ = write!(out, "mlx_flash_worker_inflight{{worker=\"{port}\"}} {inflight}\n");
            let _ = write!(out, "mlx_flash_worker_requests_total{{worker=\"{port}\"}} {total_req}\n");
            let _ = write!(out, "mlx_flash_worker_healthy{{worker=\"{port}\"}} {healthy}\n");
        }
        let _ = write!(out, "\n");
    }

    let _ = write!(out, "# HELP mlx_flash_sessions_active Active sticky sessions.\n");
    let _ = write!(out, "# TYPE mlx_flash_sessions_active gauge\n");
    let _ = write!(out, "mlx_flash_sessions_active {}\n\n", state.pool.session_count());

    // -- Cache --
    if let Some(ref cache) = state.cache {
        let stats = cache.stats();
        if let Ok(cs) = serde_json::to_value(&stats) {
            let hits = cs["hot_hits"].as_u64().unwrap_or(0) + cs["warm_hits"].as_u64().unwrap_or(0);
            let misses = cs["cold_hits"].as_u64().unwrap_or(0);
            let total_cache = hits + misses;
            let hit_ratio = if total_cache > 0 { hits as f64 / total_cache as f64 } else { 0.0 };
            let entries = cs["cached_experts"].as_u64().unwrap_or(0);

            let _ = write!(out, "# HELP mlx_flash_cache_hits_total Cache hits (hot+warm).\n");
            let _ = write!(out, "# TYPE mlx_flash_cache_hits_total counter\n");
            let _ = write!(out, "mlx_flash_cache_hits_total {hits}\n\n");

            let _ = write!(out, "# HELP mlx_flash_cache_misses_total Cache misses (cold).\n");
            let _ = write!(out, "# TYPE mlx_flash_cache_misses_total counter\n");
            let _ = write!(out, "mlx_flash_cache_misses_total {misses}\n\n");

            let _ = write!(out, "# HELP mlx_flash_cache_hit_ratio Cache hit ratio (0-1).\n");
            let _ = write!(out, "# TYPE mlx_flash_cache_hit_ratio gauge\n");
            let _ = write!(out, "mlx_flash_cache_hit_ratio {hit_ratio:.4}\n\n");

            let _ = write!(out, "# HELP mlx_flash_cache_entries Cached expert count.\n");
            let _ = write!(out, "# TYPE mlx_flash_cache_entries gauge\n");
            let _ = write!(out, "mlx_flash_cache_entries {entries}\n\n");
        }
    }

    axum::response::Response::builder()
        .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        .body(axum::body::Body::from(out))
        .unwrap()
}

async fn handle_cache_stats(State(state): State<AppState>) -> axum::Json<Value> {
    if let Some(ref cache) = state.cache {
        axum::Json(serde_json::to_value(cache.stats()).unwrap_or(json!({"error": "serialization failed"})))
    } else {
        axum::Json(json!({"error": "Cache not initialized", "hint": "Start with --expert-dir to enable expert caching"}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_state() -> AppState {
        AppState::default()
    }

    async fn get_json(router: Router, path: &str) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(req).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json)
    }

    #[tokio::test]
    async fn test_status_returns_200() {
        let router = create_router(test_state());
        let (status, _) = get_json(router, "/status").await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status_contains_memory() {
        let router = create_router(test_state());
        let (_, json) = get_json(router, "/status").await;
        let total_gb = json["memory"]["total_gb"].as_f64().unwrap_or(0.0);
        assert!(total_gb > 0.0, "expected total_gb > 0, got {total_gb}");
    }

    #[tokio::test]
    async fn test_hints_returns_array() {
        let router = create_router(test_state());
        let (_, json) = get_json(router, "/hints").await;
        assert!(json["hints"].is_array(), "expected hints to be an array");
    }

    #[tokio::test]
    async fn test_health_returns_ok() {
        let router = create_router(test_state());
        let (status, _) = get_json(router, "/health").await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_models_returns_list() {
        let router = create_router(test_state());
        let (_, json) = get_json(router, "/v1/models").await;
        assert!(json["data"].is_array(), "expected data to be an array");
        assert!(!json["data"].as_array().unwrap().is_empty(), "data array must not be empty");
    }

    #[tokio::test]
    async fn test_cors_headers() {
        let router = create_router(test_state());
        let req = Request::builder()
            .method("GET")
            .uri("/status")
            .header("Origin", "http://localhost:3000")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(req).await.unwrap();
        assert!(
            response.headers().contains_key("access-control-allow-origin"),
            "expected access-control-allow-origin header"
        );
    }

    async fn post_json(router: Router, path: &str, body: &str) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router.oneshot(req).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json)
    }

    #[tokio::test]
    async fn test_model_switch_requires_model_field() {
        let router = create_router(test_state());
        let (status, json) = post_json(router, "/v1/models/switch", r#"{"foo": "bar"}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert!(json["error"].as_str().unwrap().contains("model"));
    }

    #[tokio::test]
    async fn test_model_switch_rejects_invalid_json() {
        let router = create_router(test_state());
        let (status, json) = post_json(router, "/v1/models/switch", "not json").await;
        assert_eq!(status, StatusCode::OK);
        assert!(json["error"].as_str().unwrap().contains("Invalid JSON"));
    }

    #[tokio::test]
    async fn test_model_switch_reports_worker_failures() {
        // No Python worker running — switch should fail but not crash
        let router = create_router(test_state());
        let (status, json) = post_json(
            router,
            "/v1/models/switch",
            r#"{"model": "mlx-community/Qwen3-8B-4bit"}"#,
        ).await;
        assert_eq!(status, StatusCode::OK);
        // Workers unreachable → switched = false
        assert_eq!(json["switched"], false);
        assert!(json["workers_failed"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_model_name_unchanged_on_failed_switch() {
        let state = test_state();
        let router = create_router(state.clone());
        let _ = post_json(
            router,
            "/v1/models/switch",
            r#"{"model": "new-model"}"#,
        ).await;
        // Model name should remain "local" since no worker accepted the switch
        let current = state.model_name.read().await.clone();
        assert_eq!(current, "local");
    }

    #[tokio::test]
    async fn test_workers_endpoint() {
        let router = create_router(test_state());
        let (status, json) = get_json(router, "/workers").await;
        assert_eq!(status, StatusCode::OK);
        assert!(json["workers"].is_array());
        assert_eq!(json["total_count"], 1);
    }

    #[tokio::test]
    async fn test_metrics_returns_prometheus_format() {
        let router = create_router(test_state());
        let req = Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response.headers().get("content-type").unwrap().to_str().unwrap();
        assert!(ct.contains("text/plain"), "expected text/plain content type for prometheus");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("mlx_flash_uptime_seconds"), "expected uptime metric");
        assert!(text.contains("mlx_flash_requests_total"), "expected requests metric");
        assert!(text.contains("mlx_flash_memory_total_bytes"), "expected memory metric");
        assert!(text.contains("mlx_flash_workers_total"), "expected workers metric");
        assert!(text.contains("# TYPE"), "expected TYPE annotations");
        assert!(text.contains("# HELP"), "expected HELP annotations");
    }

    #[tokio::test]
    async fn test_metrics_contains_worker_labels() {
        let router = create_router(test_state());
        let req = Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("mlx_flash_worker_inflight{worker="), "expected per-worker inflight metric");
    }

    #[tokio::test]
    async fn test_dashboard_returns_html() {
        let router = create_router(test_state());
        let req = Request::builder()
            .method("GET")
            .uri("/admin")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let html = String::from_utf8_lossy(&body);
        assert!(html.contains("MLX-Flash Dashboard"), "expected dashboard HTML");
        assert!(html.contains("mem-chart"), "expected memory chart canvas");
    }
}
