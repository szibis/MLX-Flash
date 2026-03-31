use crate::cache::{AsyncPrefetcher, LcpCache};
use crate::expert_store::ExpertStore;
use crate::protocol::{self, Message};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;

pub async fn run_socket_server(
    socket_path: &str,
    cache: Arc<LcpCache>,
    store: Arc<ExpertStore>,
) {
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", socket_path, e));
    tracing::info!("Cache socket listening at {}", socket_path);

    let prefetcher = Arc::new(AsyncPrefetcher::new(cache.clone(), store));

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let cache = cache.clone();
                let prefetcher = prefetcher.clone();
                tokio::spawn(handle_connection(stream, cache, prefetcher));
            }
            Err(e) => tracing::error!("Socket accept error: {}", e),
        }
    }
}

async fn handle_connection(
    mut stream: tokio::net::UnixStream,
    cache: Arc<LcpCache>,
    prefetcher: Arc<AsyncPrefetcher>,
) {
    let mut buf = vec![0u8; 64 * 1024];
    let mut read_buf = Vec::new();

    loop {
        match stream.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                read_buf.extend_from_slice(&buf[..n]);
                while let Some((msg, consumed)) = protocol::decode_message(&read_buf) {
                    let response = handle_message(msg, &cache, &prefetcher).await;
                    let response_bytes = protocol::encode_message(&response);
                    if let Err(e) = stream.write_all(&response_bytes).await {
                        tracing::error!("Write error: {}", e);
                        return;
                    }
                    read_buf.drain(..consumed);
                }
            }
            Err(e) => {
                tracing::error!("Read error: {}", e);
                break;
            }
        }
    }
}

async fn handle_message(
    msg: Message,
    cache: &LcpCache,
    prefetcher: &AsyncPrefetcher,
) -> Message {
    match msg {
        Message::FetchExperts { layer, experts, request_id } => {
            let mut sizes = Vec::new();
            for &expert in &experts {
                match prefetcher.fetch_or_load(layer, expert).await {
                    Ok(data) => sizes.push(data.len()),
                    Err(_) => sizes.push(0),
                }
            }
            cache.advance_step();
            Message::ExpertData { request_id, expert_sizes: sizes }
        }
        Message::RoutingReport { layer, activated, .. } => {
            prefetcher.prefetch(layer + 1, &activated).await;
            let stats = cache.stats();
            Message::CacheStatsResponse {
                entries: stats.entries,
                hit_rate: stats.hit_rate,
                bytes_used: stats.bytes_used,
            }
        }
        _ => Message::Error { request_id: 0, message: "Unhandled message type".into() },
    }
}
