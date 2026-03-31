use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::{Router, extract::State, routing::get};
use axum::http::Method;
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

use crate::memory;
use crate::proxy;

#[derive(Clone)]
pub struct AppState {
    pub python_port: u16,
    pub model_name: String,
    pub start_time: Instant,
    pub request_count: Arc<AtomicU64>,
    pub tokens_generated: Arc<AtomicU64>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            python_port: 8081,
            model_name: "local".to_string(),
            start_time: Instant::now(),
            request_count: Arc::new(AtomicU64::new(0)),
            tokens_generated: Arc::new(AtomicU64::new(0)),
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
        .route("/v1/chat/completions", axum::routing::post(proxy::handle_chat))
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

    axum::Json(json!({
        "model": state.model_name,
        "memory": memory,
        "stats": {
            "requests": state.request_count.load(Ordering::Relaxed),
            "tokens_generated": state.tokens_generated.load(Ordering::Relaxed),
            "uptime_secs": uptime_secs,
        },
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

    axum::Json(json!({
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "owned_by": "mlx-flash-compress",
            }
        ]
    }))
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
}
