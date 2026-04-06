use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio_stream::StreamExt;

use crate::server::AppState;

pub fn build_proxy_url(port: u16, path: &str) -> String {
    format!("http://127.0.0.1:{port}{path}")
}

pub fn is_stream_request(body: &serde_json::Value) -> bool {
    body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false)
}

pub fn validate_chat_request(body: &serde_json::Value) -> Result<(), String> {
    match body.get("messages") {
        Some(msgs) if msgs.is_array() && !msgs.as_array().unwrap().is_empty() => Ok(()),
        Some(_) => Err("messages must be a non-empty array".to_string()),
        None => Err("missing required field: messages".to_string()),
    }
}

pub async fn proxy_request(
    client: &reqwest::Client,
    port: u16,
    path: &str,
    body: &[u8],
) -> Result<reqwest::Response, reqwest::Error> {
    let url = build_proxy_url(port, path);
    client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec())
        .send()
        .await
}

/// Extract a session identifier from the request body.
/// Checks: "session_id" field, or falls back to first message hash for affinity.
fn extract_session_id(parsed: &serde_json::Value) -> Option<String> {
    // Explicit session_id (custom extension)
    if let Some(sid) = parsed.get("session_id").and_then(|v| v.as_str()) {
        return Some(sid.to_string());
    }
    // OpenAI "user" field — stable per user
    if let Some(user) = parsed.get("user").and_then(|v| v.as_str()) {
        return Some(user.to_string());
    }
    None
}

fn json_error(status: u16, msg: String) -> Response {
    let body = serde_json::json!({ "error": msg });
    axum::response::Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap()
}

pub async fn handle_chat(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // Parse body
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return json_error(400, format!("Invalid JSON: {e}")),
    };

    // Validate
    if let Err(e) = validate_chat_request(&parsed) {
        return json_error(400, e);
    }

    // Pick worker: session-sticky if session_id/user present, else least-connections
    let session_id = extract_session_id(&parsed);
    let worker = match &session_id {
        Some(sid) => state.pool.next_worker_for_session(sid),
        None => state.pool.next_worker(),
    };

    let worker = match worker {
        Some(w) => w,
        None => return json_error(503, "No healthy workers available".to_string()),
    };

    let port = worker.port;
    worker.inflight.fetch_add(1, Ordering::Relaxed);
    worker.total_requests.fetch_add(1, Ordering::Relaxed);

    let streaming = is_stream_request(&parsed);
    let client = reqwest::Client::new();

    let result = proxy_request(&client, port, "/v1/chat/completions", &body).await;

    // Decrement inflight after we get a response (or error)
    let response = match result {
        Err(e) => {
            state.pool.mark_unhealthy(port);
            json_error(502, format!("Python worker on port {port} unavailable: {e}"))
        }
        Ok(upstream) => {
            if streaming {
                let byte_stream = upstream.bytes_stream();
                let event_stream = byte_stream.map(|chunk| {
                    chunk
                        .map(|bytes| {
                            Event::default().data(
                                String::from_utf8_lossy(&bytes)
                                    .trim_end()
                                    .to_string(),
                            )
                        })
                        .map_err(|e| format!("stream error: {e}"))
                });

                Sse::new(event_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            } else {
                let status = upstream.status();
                let bytes = match upstream.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        worker.inflight.fetch_sub(1, Ordering::Relaxed);
                        return json_error(
                            502,
                            format!("Python worker on port {port} unavailable: {e}"),
                        );
                    }
                };
                // Extract token count from response to update Rust-side counter
                if let Ok(resp_json) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    let completion_tokens = resp_json["usage"]["completion_tokens"]
                        .as_u64()
                        .or_else(|| resp_json["mlx_flash_compress"]["tok_per_s"].as_f64().map(|_| {
                            // Fallback: estimate from mlx_flash_compress data
                            resp_json["usage"]["total_tokens"].as_u64().unwrap_or(0)
                        }))
                        .unwrap_or(0);
                    if completion_tokens > 0 {
                        state.tokens_generated.fetch_add(completion_tokens, Ordering::Relaxed);
                    }
                }
                axum::response::Response::builder()
                    .status(status.as_u16())
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(bytes))
                    .unwrap()
            }
        }
    };

    worker.inflight.fetch_sub(1, Ordering::Relaxed);
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_proxy_url() {
        let url = build_proxy_url(8081, "/v1/chat/completions");
        assert_eq!(url, "http://127.0.0.1:8081/v1/chat/completions");
    }

    #[test]
    fn test_is_sse_request() {
        let with_stream = serde_json::json!({ "stream": true, "messages": [] });
        assert!(is_stream_request(&with_stream));

        let without_stream = serde_json::json!({ "messages": [] });
        assert!(!is_stream_request(&without_stream));

        let stream_false = serde_json::json!({ "stream": false, "messages": [] });
        assert!(!is_stream_request(&stream_false));
    }

    #[test]
    fn test_parse_chat_request_validates_messages() {
        // Missing messages field → error
        let no_messages = serde_json::json!({ "model": "test" });
        assert!(validate_chat_request(&no_messages).is_err());

        // Empty messages array → error
        let empty_messages = serde_json::json!({ "messages": [] });
        assert!(validate_chat_request(&empty_messages).is_err());

        // Valid non-empty messages → ok
        let valid = serde_json::json!({
            "messages": [{ "role": "user", "content": "hello" }]
        });
        assert!(validate_chat_request(&valid).is_ok());
    }

    #[tokio::test]
    async fn test_proxy_error_when_worker_down() {
        let client = reqwest::Client::new();
        let result = proxy_request(&client, 19999, "/v1/chat/completions", b"{}").await;
        assert!(result.is_err(), "expected connection error to port 19999");
    }

    #[test]
    fn test_extract_session_id_from_session_field() {
        let body = serde_json::json!({"session_id": "abc-123", "messages": []});
        assert_eq!(extract_session_id(&body), Some("abc-123".to_string()));
    }

    #[test]
    fn test_extract_session_id_from_user_field() {
        let body = serde_json::json!({"user": "user-42", "messages": []});
        assert_eq!(extract_session_id(&body), Some("user-42".to_string()));
    }

    #[test]
    fn test_extract_session_id_prefers_session_id_over_user() {
        let body = serde_json::json!({"session_id": "sess", "user": "usr", "messages": []});
        assert_eq!(extract_session_id(&body), Some("sess".to_string()));
    }

    #[test]
    fn test_extract_session_id_returns_none_when_absent() {
        let body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(extract_session_id(&body), None);
    }
}
