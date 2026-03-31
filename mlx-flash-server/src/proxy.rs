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

pub async fn handle_chat(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // Parse body
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let error_body = serde_json::json!({ "error": format!("Invalid JSON: {e}") });
            return axum::response::Response::builder()
                .status(400)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(error_body.to_string()))
                .unwrap();
        }
    };

    // Validate
    if let Err(e) = validate_chat_request(&parsed) {
        let error_body = serde_json::json!({ "error": e });
        return axum::response::Response::builder()
            .status(400)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(error_body.to_string()))
            .unwrap();
    }

    let streaming = is_stream_request(&parsed);
    let client = reqwest::Client::new();

    match proxy_request(&client, state.python_port, "/v1/chat/completions", &body).await {
        Err(e) => {
            let error_body =
                serde_json::json!({ "error": format!("Python worker unavailable: {e}") });
            axum::response::Response::builder()
                .status(502)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(error_body.to_string()))
                .unwrap()
        }
        Ok(upstream) => {
            if streaming {
                let byte_stream = upstream.bytes_stream();
                let event_stream = byte_stream.map(|chunk| {
                    chunk
                        .map(|bytes| {
                            // Each chunk from the upstream is already an SSE line or data chunk;
                            // wrap it as a raw SSE event so axum forwards it verbatim.
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
                        let error_body = serde_json::json!({
                            "error": format!("Python worker unavailable: {e}")
                        });
                        return axum::response::Response::builder()
                            .status(502)
                            .header("Content-Type", "application/json")
                            .body(axum::body::Body::from(error_body.to_string()))
                            .unwrap();
                    }
                };
                axum::response::Response::builder()
                    .status(status.as_u16())
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(bytes))
                    .unwrap()
            }
        }
    }
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
}
