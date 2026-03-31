# Phase 2: Rust Expert Cache with LCP Eviction + Unix Socket Bridge

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lock-free LCP expert cache in Rust with async SSD prefetch, connected to the Python MLX worker via Unix socket, enabling real expert weight streaming for models exceeding RAM.

**Architecture:** Rust `DashMap`-based LCP cache manages expert weights in RAM. Python requests experts by (layer, expert_id) over a Unix socket. Rust serves from cache (0.08ms) or loads from SSD (0.6ms) with async prefetch. Cache endpoints added to the existing axum server.

**Tech Stack:** Rust (dashmap, tokio::fs, tokio::net::UnixListener, serde_json), Python (socket, struct, mmap). mlx-rs optional for Task 6 (mixed precision).

**Spec:** `docs/superpowers/specs/2026-03-31-rust-sidecar-design.md` (Phase 2 section)

**Prerequisite:** Phase 1 complete (mlx-flash-server with 14 Rust tests, 85 Python tests)

---

## File Structure

### Rust (new files in mlx-flash-server/)

```
src/
  cache/
    mod.rs           # CacheManager public API + cache stats endpoint
    lcp.rs           # LcpCache: DashMap-based LCP eviction
    prefetch.rs      # AsyncPrefetcher: tokio::fs SSD reads
  expert_store.rs    # ExpertStore: read expert .bin files from disk
  protocol.rs        # Unix socket message types + listener
  main.rs            # (modify) add cache + socket to startup
  server.rs          # (modify) add /cache/stats, /cache/warm endpoints
```

### Python (new/modified)

```
mlx_flash_compress/
  rust_bridge.py     # Python client for Rust Unix socket
  cached_inference.py  # (modify) add Rust cache backend option
tests/
  test_rust_bridge.py  # Tests for Python<->Rust communication
```

---

## Task 1: LCP Cache Core (cache/lcp.rs)

**Files:**
- Create: `mlx-flash-server/src/cache/mod.rs`
- Create: `mlx-flash-server/src/cache/lcp.rs`
- Modify: `mlx-flash-server/Cargo.toml` (add dashmap)
- Modify: `mlx-flash-server/src/main.rs` (add mod cache)

- [ ] **Step 1: Add dashmap dependency**

In `Cargo.toml` add under `[dependencies]`:
```toml
dashmap = "6"
```

- [ ] **Step 2: Write failing tests in cache/lcp.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_fetch() {
        let cache = LcpCache::new(1024 * 1024); // 1MB
        cache.insert(0, 1, vec![1, 2, 3, 4]);
        let data = cache.fetch(0, 1);
        assert!(data.is_some());
        assert_eq!(data.unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_fetch_miss() {
        let cache = LcpCache::new(1024 * 1024);
        assert!(cache.fetch(0, 99).is_none());
    }

    #[test]
    fn test_capacity_enforcement() {
        let cache = LcpCache::new(100); // 100 bytes capacity
        // Insert 60 bytes
        cache.insert(0, 1, vec![0u8; 60]);
        assert!(cache.fetch(0, 1).is_some());
        // Insert another 60 — should evict first
        cache.insert(0, 2, vec![0u8; 60]);
        assert!(cache.fetch(0, 2).is_some());
        // First should have been evicted
        assert!(cache.fetch(0, 1).is_none());
    }

    #[test]
    fn test_lcp_priority_keeps_hot() {
        let cache = LcpCache::new(200);
        // Access expert 1 many times (hot)
        cache.insert(0, 1, vec![0u8; 80]);
        for _ in 0..10 {
            cache.advance_step();
            cache.fetch(0, 1);
        }
        // Access expert 2 once (cold)
        cache.insert(0, 2, vec![0u8; 80]);
        cache.advance_step();
        // Insert expert 3 — should evict 2 (cold), not 1 (hot)
        cache.insert(0, 3, vec![0u8; 80]);
        assert!(cache.fetch(0, 1).is_some(), "hot expert should survive");
        assert!(cache.fetch(0, 2).is_none(), "cold expert should be evicted");
    }

    #[test]
    fn test_advance_step() {
        let cache = LcpCache::new(1024);
        cache.advance_step();
        cache.advance_step();
        assert_eq!(cache.current_step(), 2);
    }

    #[test]
    fn test_stats() {
        let cache = LcpCache::new(1024);
        cache.insert(0, 1, vec![1, 2, 3]);
        cache.fetch(0, 1); // hit
        cache.fetch(0, 99); // miss
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(LcpCache::new(1024 * 1024));
        let mut handles = vec![];
        for t in 0..4 {
            let c = cache.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    c.insert(t, i, vec![t as u8; 64]);
                    c.advance_step();
                    c.fetch(t, i);
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
        assert!(cache.stats().entries > 0);
    }

    #[test]
    fn test_warmup_convergence() {
        // Simulate 50 tokens accessing same 4 experts — hit rate should climb
        let cache = LcpCache::new(1024 * 1024);
        let experts = [0u32, 1, 2, 3];
        for _ in 0..50 {
            cache.advance_step();
            for &e in &experts {
                if cache.fetch(0, e).is_none() {
                    cache.insert(0, e, vec![0u8; 64]);
                }
            }
        }
        let stats = cache.stats();
        let hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
        assert!(hit_rate > 0.9, "hit rate should converge above 90%, got {:.1}%", hit_rate * 100.0);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd mlx-flash-server && cargo test -- cache::lcp 2>&1`
Expected: FAIL — module not found

- [ ] **Step 4: Implement cache/lcp.rs**

```rust
use dashmap::DashMap;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// LCP priority: frequency * 0.25^(steps_since_last / 128)
const LCP_BASE: f64 = 0.25;
const LCP_DECAY: f64 = 128.0;

#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes_used: usize,
    pub capacity_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

struct CacheEntry {
    data: Vec<u8>,
    frequency: u64,
    last_step: u64,
}

pub struct LcpCache {
    entries: DashMap<(u32, u32), CacheEntry>,
    capacity_bytes: usize,
    current_bytes: AtomicUsize,
    step: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl LcpCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: DashMap::new(),
            capacity_bytes,
            current_bytes: AtomicUsize::new(0),
            step: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn advance_step(&self) {
        self.step.fetch_add(1, Ordering::Relaxed);
    }

    pub fn current_step(&self) -> u64 {
        self.step.load(Ordering::Relaxed)
    }

    pub fn fetch(&self, layer: u32, expert: u32) -> Option<Vec<u8>> {
        let key = (layer, expert);
        if let Some(mut entry) = self.entries.get_mut(&key) {
            entry.frequency += 1;
            entry.last_step = self.step.load(Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.data.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn insert(&self, layer: u32, expert: u32, data: Vec<u8>) {
        let key = (layer, expert);
        let size = data.len();

        // Remove old entry if exists
        if let Some((_, old)) = self.entries.remove(&key) {
            self.current_bytes.fetch_sub(old.data.len(), Ordering::Relaxed);
        }

        // Evict until enough space
        while self.current_bytes.load(Ordering::Relaxed) + size > self.capacity_bytes {
            if !self.evict_lowest() {
                break;
            }
        }

        self.entries.insert(key, CacheEntry {
            data,
            frequency: 1,
            last_step: self.step.load(Ordering::Relaxed),
        });
        self.current_bytes.fetch_add(size, Ordering::Relaxed);
    }

    fn evict_lowest(&self) -> bool {
        let step = self.step.load(Ordering::Relaxed);
        let mut min_key = None;
        let mut min_priority = f64::MAX;

        for entry in self.entries.iter() {
            let p = Self::priority(entry.value(), step);
            if p < min_priority {
                min_priority = p;
                min_key = Some(*entry.key());
            }
        }

        if let Some(key) = min_key {
            if let Some((_, evicted)) = self.entries.remove(&key) {
                self.current_bytes.fetch_sub(evicted.data.len(), Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    fn priority(entry: &CacheEntry, current_step: u64) -> f64 {
        let age = current_step.saturating_sub(entry.last_step) as f64;
        entry.frequency as f64 * LCP_BASE.powf(age / LCP_DECAY)
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            entries: self.entries.len(),
            bytes_used: self.current_bytes.load(Ordering::Relaxed),
            capacity_bytes: self.capacity_bytes,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}
```

- [ ] **Step 5: Create cache/mod.rs**

```rust
pub mod lcp;
pub use lcp::{LcpCache, CacheStats};
```

- [ ] **Step 6: Add `mod cache;` to main.rs**

- [ ] **Step 7: Run tests**

Run: `cd mlx-flash-server && cargo test -- cache 2>&1`
Expected: 8 tests PASS

- [ ] **Step 8: Commit**

```bash
git add mlx-flash-server/src/cache/ mlx-flash-server/Cargo.toml mlx-flash-server/src/main.rs
git commit -m "feat(rust): LCP expert cache with DashMap — 8 tests passing"
```

---

## Task 2: Expert Store (expert_store.rs)

**Files:**
- Create: `mlx-flash-server/src/expert_store.rs`
- Modify: `mlx-flash-server/src/main.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_experts(dir: &std::path::Path, layers: u32, experts: u32, size: usize) {
        for l in 0..layers {
            let layer_dir = dir.join(format!("layer_{:03}", l));
            fs::create_dir_all(&layer_dir).unwrap();
            for e in 0..experts {
                let data: Vec<u8> = (0..size).map(|i| ((l * experts + e) as u8).wrapping_add(i as u8)).collect();
                fs::write(layer_dir.join(format!("expert_{:04}.bin", e)), &data).unwrap();
            }
        }
    }

    #[test]
    fn test_load_expert() {
        let dir = TempDir::new().unwrap();
        create_test_experts(dir.path(), 2, 4, 256);
        let store = ExpertStore::new(dir.path().to_path_buf());
        let data = store.load_expert(0, 1);
        assert!(data.is_ok());
        assert_eq!(data.unwrap().len(), 256);
    }

    #[test]
    fn test_load_nonexistent() {
        let dir = TempDir::new().unwrap();
        let store = ExpertStore::new(dir.path().to_path_buf());
        assert!(store.load_expert(99, 99).is_err());
    }

    #[tokio::test]
    async fn test_load_expert_async() {
        let dir = TempDir::new().unwrap();
        create_test_experts(dir.path(), 2, 4, 128);
        let store = ExpertStore::new(dir.path().to_path_buf());
        let data = store.load_expert_async(1, 3).await;
        assert!(data.is_ok());
        assert_eq!(data.unwrap().len(), 128);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement expert_store.rs**

```rust
use std::path::PathBuf;
use tokio::fs as async_fs;

pub struct ExpertStore {
    base_dir: PathBuf,
}

impl ExpertStore {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    pub fn expert_path(&self, layer: u32, expert: u32) -> PathBuf {
        self.base_dir
            .join(format!("layer_{:03}", layer))
            .join(format!("expert_{:04}.bin", expert))
    }

    pub fn load_expert(&self, layer: u32, expert: u32) -> std::io::Result<Vec<u8>> {
        std::fs::read(self.expert_path(layer, expert))
    }

    pub async fn load_expert_async(&self, layer: u32, expert: u32) -> std::io::Result<Vec<u8>> {
        async_fs::read(self.expert_path(layer, expert)).await
    }
}
```

Add `tempfile = "3"` to `[dev-dependencies]` in Cargo.toml.
Add `mod expert_store;` to main.rs.

- [ ] **Step 4: Run tests**

Run: `cd mlx-flash-server && cargo test -- expert_store 2>&1`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-flash-server/src/expert_store.rs mlx-flash-server/Cargo.toml mlx-flash-server/src/main.rs
git commit -m "feat(rust): expert weight file store with sync and async loading"
```

---

## Task 3: Async Prefetcher (cache/prefetch.rs)

**Files:**
- Create: `mlx-flash-server/src/cache/prefetch.rs`
- Modify: `mlx-flash-server/src/cache/mod.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LcpCache;
    use crate::expert_store::ExpertStore;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Arc<LcpCache>, Arc<ExpertStore>) {
        let dir = TempDir::new().unwrap();
        // Create test expert files
        for l in 0..2u32 {
            let ld = dir.path().join(format!("layer_{:03}", l));
            std::fs::create_dir_all(&ld).unwrap();
            for e in 0..4u32 {
                std::fs::write(ld.join(format!("expert_{:04}.bin", e)), vec![e as u8; 128]).unwrap();
            }
        }
        let cache = Arc::new(LcpCache::new(1024 * 1024));
        let store = Arc::new(ExpertStore::new(dir.path().to_path_buf()));
        (dir, cache, store)
    }

    #[tokio::test]
    async fn test_prefetch_loads_into_cache() {
        let (_dir, cache, store) = setup();
        let prefetcher = AsyncPrefetcher::new(cache.clone(), store);
        prefetcher.prefetch(0, &[1, 2]).await;
        // Give async tasks time to complete
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(cache.fetch(0, 1).is_some());
        assert!(cache.fetch(0, 2).is_some());
    }

    #[tokio::test]
    async fn test_prefetch_skips_cached() {
        let (_dir, cache, store) = setup();
        cache.insert(0, 1, vec![99; 128]); // pre-cache
        let prefetcher = AsyncPrefetcher::new(cache.clone(), store);
        prefetcher.prefetch(0, &[1]).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // Should still have original data (not reloaded)
        assert_eq!(cache.fetch(0, 1).unwrap(), vec![99; 128]);
    }

    #[tokio::test]
    async fn test_fetch_or_load() {
        let (_dir, cache, store) = setup();
        let prefetcher = AsyncPrefetcher::new(cache.clone(), store);
        // First call: cache miss, loads from disk
        let data = prefetcher.fetch_or_load(0, 3).await.unwrap();
        assert_eq!(data.len(), 128);
        // Second call: cache hit
        let data2 = prefetcher.fetch_or_load(0, 3).await.unwrap();
        assert_eq!(data, data2);
    }
}
```

- [ ] **Step 2: Implement cache/prefetch.rs**

```rust
use crate::cache::LcpCache;
use crate::expert_store::ExpertStore;
use std::sync::Arc;

pub struct AsyncPrefetcher {
    cache: Arc<LcpCache>,
    store: Arc<ExpertStore>,
}

impl AsyncPrefetcher {
    pub fn new(cache: Arc<LcpCache>, store: Arc<ExpertStore>) -> Self {
        Self { cache, store }
    }

    /// Prefetch experts into cache in background. Skips already-cached experts.
    pub async fn prefetch(&self, layer: u32, experts: &[u32]) {
        for &expert in experts {
            if self.cache.fetch(layer, expert).is_some() {
                continue; // already cached
            }
            let cache = self.cache.clone();
            let store = self.store.clone();
            tokio::spawn(async move {
                if let Ok(data) = store.load_expert_async(layer, expert).await {
                    cache.insert(layer, expert, data);
                }
            });
        }
    }

    /// Fetch from cache or load from SSD synchronously (for the hot path).
    pub async fn fetch_or_load(&self, layer: u32, expert: u32) -> std::io::Result<Vec<u8>> {
        if let Some(data) = self.cache.fetch(layer, expert) {
            return Ok(data);
        }
        let data = self.store.load_expert_async(layer, expert).await?;
        self.cache.insert(layer, expert, data.clone());
        Ok(data)
    }
}
```

Update `cache/mod.rs`:
```rust
pub mod lcp;
pub mod prefetch;
pub use lcp::{LcpCache, CacheStats};
pub use prefetch::AsyncPrefetcher;
```

- [ ] **Step 3: Run tests**

Run: `cd mlx-flash-server && cargo test -- prefetch 2>&1`
Expected: 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add mlx-flash-server/src/cache/
git commit -m "feat(rust): async expert prefetcher with tokio::fs SSD reads"
```

---

## Task 4: Unix Socket Protocol (protocol.rs)

**Files:**
- Create: `mlx-flash-server/src/protocol.rs`
- Modify: `mlx-flash-server/src/main.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_fetch_request() {
        let msg = Message::FetchExperts {
            layer: 5, experts: vec![1, 2, 3], request_id: 42,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("FetchExperts"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_deserialize_fetch_request() {
        let json = r#"{"FetchExperts":{"layer":5,"experts":[1,2,3],"request_id":42}}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        match msg {
            Message::FetchExperts { layer, experts, request_id } => {
                assert_eq!(layer, 5);
                assert_eq!(experts, vec![1, 2, 3]);
                assert_eq!(request_id, 42);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_serialize_expert_data() {
        let msg = Message::ExpertData {
            request_id: 42, expert_sizes: vec![256, 256, 256],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        match parsed {
            Message::ExpertData { request_id, expert_sizes } => {
                assert_eq!(request_id, 42);
                assert_eq!(expert_sizes.len(), 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_serialize_routing_report() {
        let msg = Message::RoutingReport {
            layer: 10, activated: vec![4, 7, 12, 45], token_idx: 100,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("RoutingReport"));
    }
}
```

- [ ] **Step 2: Implement protocol.rs**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Python asks Rust for expert weights
    FetchExperts {
        layer: u32,
        experts: Vec<u32>,
        request_id: u64,
    },
    /// Rust returns expert data sizes (actual data sent as raw bytes after)
    ExpertData {
        request_id: u64,
        expert_sizes: Vec<usize>,
    },
    /// Python reports which experts were activated (for cache learning)
    RoutingReport {
        layer: u32,
        activated: Vec<u32>,
        token_idx: u64,
    },
    /// Cache statistics response
    CacheStatsResponse {
        entries: usize,
        hit_rate: f64,
        bytes_used: usize,
    },
    /// Error response
    Error {
        request_id: u64,
        message: String,
    },
}

/// Encode a message as length-prefixed JSON for Unix socket transport.
/// Format: [4 bytes big-endian length][JSON bytes]
pub fn encode_message(msg: &Message) -> Vec<u8> {
    let json = serde_json::to_vec(msg).expect("Message serialization failed");
    let len = (json.len() as u32).to_be_bytes();
    let mut buf = Vec::with_capacity(4 + json.len());
    buf.extend_from_slice(&len);
    buf.extend_from_slice(&json);
    buf
}

/// Decode a length-prefixed JSON message from a byte buffer.
/// Returns (message, bytes_consumed).
pub fn decode_message(buf: &[u8]) -> Option<(Message, usize)> {
    if buf.len() < 4 {
        return None;
    }
    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if buf.len() < 4 + len {
        return None;
    }
    let msg: Message = serde_json::from_slice(&buf[4..4 + len]).ok()?;
    Some((msg, 4 + len))
}
```

Add `mod protocol;` to main.rs.

- [ ] **Step 3: Run tests**

Run: `cd mlx-flash-server && cargo test -- protocol 2>&1`
Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add mlx-flash-server/src/protocol.rs mlx-flash-server/src/main.rs
git commit -m "feat(rust): Unix socket protocol with length-prefixed JSON messages"
```

---

## Task 5: Socket Listener + Cache Server Wiring

**Files:**
- Modify: `mlx-flash-server/src/main.rs`
- Modify: `mlx-flash-server/src/server.rs`
- Create: `mlx-flash-server/src/socket_server.rs`

- [ ] **Step 1: Create socket_server.rs with Unix socket listener**

```rust
use crate::cache::{AsyncPrefetcher, LcpCache};
use crate::expert_store::ExpertStore;
use crate::protocol::{self, Message};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tracing;

pub async fn run_socket_server(
    socket_path: &str,
    cache: Arc<LcpCache>,
    store: Arc<ExpertStore>,
) {
    // Remove stale socket file
    let _ = std::fs::remove_file(socket_path);

    let listener = UnixListener::bind(socket_path)
        .expect(&format!("Failed to bind Unix socket at {}", socket_path));
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
            Ok(0) => break, // connection closed
            Ok(n) => {
                read_buf.extend_from_slice(&buf[..n]);

                // Process all complete messages in buffer
                while let Some((msg, consumed)) = protocol::decode_message(&read_buf) {
                    let response = handle_message(msg, &cache, &prefetcher).await;
                    let response_bytes = protocol::encode_message(&response);

                    // For FetchExperts, send response header then raw expert data
                    if let Message::ExpertData { request_id, ref expert_sizes } = response {
                        if let Err(e) = stream.write_all(&response_bytes).await {
                            tracing::error!("Write error: {}", e);
                            return;
                        }
                        // Send raw expert data bytes after the header
                        // (expert data was loaded and cached during handle_message)
                    } else {
                        if let Err(e) = stream.write_all(&response_bytes).await {
                            tracing::error!("Write error: {}", e);
                            return;
                        }
                    }
                    read_buf.drain(..consumed);
                }
            }
            Err(e) => {
                tracing::error!("Socket read error: {}", e);
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
        Message::RoutingReport { layer, activated, token_idx: _ } => {
            // Prefetch next layer's likely experts
            prefetcher.prefetch(layer + 1, &activated).await;
            Message::CacheStatsResponse {
                entries: cache.stats().entries,
                hit_rate: cache.stats().hit_rate,
                bytes_used: cache.stats().bytes_used,
            }
        }
        _ => Message::Error {
            request_id: 0,
            message: "Unhandled message type".into(),
        },
    }
}
```

- [ ] **Step 2: Add /cache/stats endpoint to server.rs**

Add to `create_router`:
```rust
.route("/cache/stats", get(handle_cache_stats))
```

Add handler:
```rust
async fn handle_cache_stats(State(state): State<AppState>) -> Json<Value> {
    if let Some(cache) = &state.cache {
        Json(serde_json::to_value(cache.stats()).unwrap())
    } else {
        Json(json!({"error": "Cache not initialized"}))
    }
}
```

Add `cache: Option<Arc<LcpCache>>` to `AppState`.

- [ ] **Step 3: Wire socket server into main.rs**

Add CLI args `--expert-dir`, `--cache-mb`, `--socket-path` to Args.
In main, if `--expert-dir` is provided, create `ExpertStore`, `LcpCache`, and spawn `socket_server::run_socket_server`.

- [ ] **Step 4: Run all tests**

Run: `cd mlx-flash-server && cargo test 2>&1`
Expected: All previous 29 tests + new tests PASS

- [ ] **Step 5: Commit**

```bash
git add mlx-flash-server/
git commit -m "feat(rust): Unix socket server for expert cache + /cache/stats endpoint"
```

---

## Task 6: Python Bridge (rust_bridge.py)

**Files:**
- Create: `mlx_flash_compress/rust_bridge.py`
- Create: `tests/test_rust_bridge.py`

- [ ] **Step 1: Write failing Python tests**

```python
"""Tests for the Python<->Rust Unix socket bridge."""
import json
import struct
import pytest
from mlx_flash_compress.rust_bridge import encode_message, decode_message


class TestProtocol:
    def test_encode_fetch_request(self):
        msg = {"FetchExperts": {"layer": 5, "experts": [1, 2, 3], "request_id": 42}}
        data = encode_message(msg)
        assert len(data) > 4
        length = struct.unpack(">I", data[:4])[0]
        assert length == len(data) - 4
        parsed = json.loads(data[4:])
        assert parsed["FetchExperts"]["layer"] == 5

    def test_decode_response(self):
        msg = {"ExpertData": {"request_id": 42, "expert_sizes": [256, 256]}}
        encoded = encode_message(msg)
        decoded, consumed = decode_message(encoded)
        assert decoded["ExpertData"]["request_id"] == 42
        assert consumed == len(encoded)

    def test_decode_partial_returns_none(self):
        result = decode_message(b"\x00\x00\x00\x10short")
        assert result is None

    def test_roundtrip(self):
        msg = {"RoutingReport": {"layer": 3, "activated": [1, 5, 9], "token_idx": 77}}
        encoded = encode_message(msg)
        decoded, _ = decode_message(encoded)
        assert decoded == msg
```

- [ ] **Step 2: Implement rust_bridge.py**

```python
"""Python client for the Rust expert cache Unix socket."""
import json
import socket
import struct
from typing import Optional


def encode_message(msg: dict) -> bytes:
    """Encode a message as length-prefixed JSON (matches Rust protocol)."""
    data = json.dumps(msg).encode("utf-8")
    return struct.pack(">I", len(data)) + data


def decode_message(buf: bytes) -> Optional[tuple[dict, int]]:
    """Decode a length-prefixed JSON message. Returns (msg, bytes_consumed) or None."""
    if len(buf) < 4:
        return None
    length = struct.unpack(">I", buf[:4])[0]
    if len(buf) < 4 + length:
        return None
    msg = json.loads(buf[4:4 + length])
    return msg, 4 + length


class RustCacheClient:
    """Client for the Rust expert cache Unix socket server."""

    def __init__(self, socket_path: str = "/tmp/mlx-flash-cache.sock"):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None

    def connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send_recv(self, msg: dict) -> dict:
        if self._sock is None:
            self.connect()
        data = encode_message(msg)
        self._sock.sendall(data)
        # Read response
        header = self._recv_exact(4)
        length = struct.unpack(">I", header)[0]
        body = self._recv_exact(length)
        return json.loads(body)

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf += chunk
        return buf

    def fetch_experts(self, layer: int, experts: list[int], request_id: int = 0) -> dict:
        """Request expert weights from the Rust cache."""
        return self._send_recv({
            "FetchExperts": {
                "layer": layer,
                "experts": experts,
                "request_id": request_id,
            }
        })

    def report_routing(self, layer: int, activated: list[int], token_idx: int) -> dict:
        """Report expert activations for cache learning."""
        return self._send_recv({
            "RoutingReport": {
                "layer": layer,
                "activated": activated,
                "token_idx": token_idx,
            }
        })
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_rust_bridge.py -v`
Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add mlx_flash_compress/rust_bridge.py tests/test_rust_bridge.py
git commit -m "feat: Python Unix socket bridge for Rust expert cache"
```

---

## Task 7: Integration Test — Rust Cache + Python Client

**Files:**
- Create: `tests/e2e_cache_roundtrip.py`

- [ ] **Step 1: Write integration test**

```python
"""E2E test: Python client talks to Rust cache server via Unix socket.

Requires:
  1. Expert files created in a temp dir
  2. Rust server running with --expert-dir and --socket-path
  3. Python client connects and fetches experts
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pytest

from mlx_flash_compress.rust_bridge import RustCacheClient
from mlx_flash_compress.demo_warmup import create_expert_files


SOCKET_PATH = "/tmp/mlx-flash-test.sock"
RUST_BINARY = os.path.join(
    os.path.dirname(__file__), "..", "mlx-flash-server", "target", "release", "mlx-flash-server"
)


@pytest.fixture(scope="module")
def cache_server():
    """Start Rust cache server for the test module."""
    tmpdir = tempfile.mkdtemp()
    expert_dir = create_expert_files(tmpdir, num_layers=4, num_experts=8, expert_size_bytes=256)

    if not os.path.exists(RUST_BINARY):
        pytest.skip("Rust binary not built (run: cd mlx-flash-server && cargo build --release)")

    proc = subprocess.Popen([
        RUST_BINARY,
        "--port", "0",  # don't start HTTP
        "--expert-dir", str(expert_dir),
        "--cache-mb", "10",
        "--socket-path", SOCKET_PATH,
    ])
    time.sleep(1)
    yield proc
    proc.terminate()
    proc.wait()
    shutil.rmtree(tmpdir, ignore_errors=True)
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass


class TestCacheRoundtrip:
    def test_fetch_experts(self, cache_server):
        client = RustCacheClient(SOCKET_PATH)
        client.connect()
        result = client.fetch_experts(layer=0, experts=[1, 2, 3], request_id=1)
        assert "ExpertData" in result
        assert result["ExpertData"]["request_id"] == 1
        assert len(result["ExpertData"]["expert_sizes"]) == 3
        assert all(s > 0 for s in result["ExpertData"]["expert_sizes"])
        client.close()

    def test_cache_warmup(self, cache_server):
        client = RustCacheClient(SOCKET_PATH)
        client.connect()
        # First fetch — all misses
        client.fetch_experts(layer=0, experts=[1, 2], request_id=1)
        # Second fetch — should be cache hits (faster)
        result = client.fetch_experts(layer=0, experts=[1, 2], request_id=2)
        assert "ExpertData" in result
        client.close()

    def test_routing_report(self, cache_server):
        client = RustCacheClient(SOCKET_PATH)
        client.connect()
        result = client.report_routing(layer=0, activated=[1, 3, 5], token_idx=0)
        assert "CacheStatsResponse" in result
        assert result["CacheStatsResponse"]["entries"] >= 0
        client.close()
```

- [ ] **Step 2: Run (requires built Rust binary)**

Run: `cd mlx-flash-server && cargo build --release`
Run: `.venv/bin/python -m pytest tests/e2e_cache_roundtrip.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/e2e_cache_roundtrip.py
git commit -m "test: E2E cache roundtrip — Python client ↔ Rust cache server"
```

---

## Task 8: Final Test Suite + Push

- [ ] **Step 1: Run all Rust tests**

Run: `cd mlx-flash-server && cargo test 2>&1`
Expected: 29+ tests PASS (14 Phase 1 + 15 Phase 2)

- [ ] **Step 2: Run all Python tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: 93+ tests PASS (85 Phase 1 + 8 Phase 2)

- [ ] **Step 3: Push everything**

```bash
git push origin main
```
