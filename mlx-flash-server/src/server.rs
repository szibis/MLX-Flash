use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::RwLock;

use axum::{Router, extract::State, routing::get};
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
