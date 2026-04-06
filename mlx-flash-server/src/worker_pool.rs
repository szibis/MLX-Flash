use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A single Python inference worker process.
#[derive(Debug)]
pub struct Worker {
    pub port: u16,
    pub healthy: AtomicBool,
    pub inflight: AtomicU64,
    pub total_requests: AtomicU64,
}

impl Worker {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            healthy: AtomicBool::new(true),
            inflight: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
        }
    }
}

/// Pool of Python inference workers behind the Rust proxy.
///
/// Selection strategy: least-connections with cache affinity.
/// Among workers with the same inflight count, prefer the one that has
/// served more requests (its KV/expert cache is warmer).
#[derive(Debug)]
pub struct WorkerPool {
    workers: Vec<Arc<Worker>>,
    /// Fallback tie-breaker index to avoid always picking the first worker
    /// when all have identical stats.
    tie_breaker: AtomicUsize,
    /// Session affinity: maps session_id -> worker port.
    /// Keeps KV cache hot for multi-turn conversations.
    sessions: Mutex<HashMap<String, u16>>,
}

impl WorkerPool {
    /// Create a pool with `count` workers starting at `base_port`.
    pub fn new(base_port: u16, count: usize) -> Self {
        assert!(count > 0, "worker pool must have at least 1 worker");
        let workers = (0..count)
            .map(|i| Arc::new(Worker::new(base_port + i as u16)))
            .collect();
        Self {
            workers,
            tie_breaker: AtomicUsize::new(0),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Create a single-worker pool (backwards compatible).
    pub fn single(port: u16) -> Self {
        Self::new(port, 1)
    }

    /// Pick a worker for a specific session. Sticky: if the session was seen before
    /// and its worker is still healthy, route back to the same one (hot KV cache).
    /// If the worker is unhealthy or the session is new, pick via least-connections.
    pub fn next_worker_for_session(&self, session_id: &str) -> Option<&Arc<Worker>> {
        // Check existing affinity
        {
            let sessions = self.sessions.lock().unwrap();
            if let Some(&port) = sessions.get(session_id) {
                if let Some(w) = self.workers.iter().find(|w| w.port == port) {
                    if w.healthy.load(Ordering::Relaxed) {
                        return Some(w);
                    }
                }
            }
        }

        // No affinity or worker unhealthy — pick best worker
        let worker = self.next_worker()?;

        // Record affinity
        {
            let mut sessions = self.sessions.lock().unwrap();
            sessions.insert(session_id.to_string(), worker.port);
        }

        Some(worker)
    }

    /// Remove a session's affinity (e.g. when conversation ends).
    pub fn release_session(&self, session_id: &str) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(session_id);
    }

    /// Number of active sessions tracked.
    pub fn session_count(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }

    /// Pick the best healthy worker: lowest inflight, then warmest cache (most requests served).
    /// Returns None if all workers are unhealthy.
    pub fn next_worker(&self) -> Option<&Arc<Worker>> {
        let healthy: Vec<_> = self.workers
            .iter()
            .filter(|w| w.healthy.load(Ordering::Relaxed))
            .collect();

        if healthy.is_empty() {
            return None;
        }

        // Find minimum inflight count
        let min_inflight = healthy
            .iter()
            .map(|w| w.inflight.load(Ordering::Relaxed))
            .min()
            .unwrap();

        // Among workers with min inflight, pick the warmest (most total requests)
        let candidates: Vec<_> = healthy
            .into_iter()
            .filter(|w| w.inflight.load(Ordering::Relaxed) == min_inflight)
            .collect();

        if candidates.len() == 1 {
            return Some(candidates[0]);
        }

        // Multiple candidates with same inflight — prefer warmest cache
        let max_total = candidates
            .iter()
            .map(|w| w.total_requests.load(Ordering::Relaxed))
            .max()
            .unwrap();

        let warmest: Vec<_> = candidates
            .into_iter()
            .filter(|w| w.total_requests.load(Ordering::Relaxed) == max_total)
            .collect();

        if warmest.len() == 1 {
            return Some(warmest[0]);
        }

        // Perfect tie (e.g. all cold workers at startup) — rotate
        let idx = self.tie_breaker.fetch_add(1, Ordering::Relaxed) % warmest.len();
        Some(warmest[idx])
    }

    /// Mark a worker as unhealthy by port.
    pub fn mark_unhealthy(&self, port: u16) {
        if let Some(w) = self.workers.iter().find(|w| w.port == port) {
            w.healthy.store(false, Ordering::Relaxed);
        }
    }

    /// Mark a worker as healthy by port.
    pub fn mark_healthy(&self, port: u16) {
        if let Some(w) = self.workers.iter().find(|w| w.port == port) {
            w.healthy.store(true, Ordering::Relaxed);
        }
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn healthy_count(&self) -> usize {
        self.workers
            .iter()
            .filter(|w| w.healthy.load(Ordering::Relaxed))
            .count()
    }

    pub fn ports(&self) -> Vec<u16> {
        self.workers.iter().map(|w| w.port).collect()
    }

    /// Status for /status endpoint.
    pub fn status(&self) -> serde_json::Value {
        let workers: Vec<_> = self.workers
            .iter()
            .map(|w| {
                serde_json::json!({
                    "port": w.port,
                    "healthy": w.healthy.load(Ordering::Relaxed),
                    "inflight": w.inflight.load(Ordering::Relaxed),
                    "total_requests": w.total_requests.load(Ordering::Relaxed),
                })
            })
            .collect();
        serde_json::json!({
            "workers": workers,
            "strategy": "least-connections + cache-affinity",
            "healthy_count": self.healthy_count(),
            "total_count": self.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Construction ---

    #[test]
    fn test_new_creates_consecutive_ports() {
        let pool = WorkerPool::new(8081, 3);
        assert_eq!(pool.ports(), vec![8081, 8082, 8083]);
    }

    #[test]
    fn test_single_creates_one_worker() {
        let pool = WorkerPool::single(8081);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.ports(), vec![8081]);
    }

    #[test]
    #[should_panic(expected = "at least 1 worker")]
    fn test_new_panics_on_zero_workers() {
        WorkerPool::new(8081, 0);
    }

    #[test]
    fn test_all_workers_start_healthy() {
        let pool = WorkerPool::new(8081, 4);
        assert_eq!(pool.healthy_count(), 4);
    }

    // --- Least connections ---

    #[test]
    fn test_picks_lowest_inflight() {
        let pool = WorkerPool::new(8081, 3);
        pool.workers[0].inflight.store(5, Ordering::Relaxed);
        pool.workers[1].inflight.store(2, Ordering::Relaxed);
        pool.workers[2].inflight.store(10, Ordering::Relaxed);

        assert_eq!(pool.next_worker().unwrap().port, 8082);
    }

    #[test]
    fn test_skips_unhealthy_workers() {
        let pool = WorkerPool::new(8081, 3);
        // 8082 has lowest inflight but is unhealthy
        pool.workers[0].inflight.store(5, Ordering::Relaxed);
        pool.workers[1].inflight.store(1, Ordering::Relaxed);
        pool.workers[2].inflight.store(3, Ordering::Relaxed);
        pool.mark_unhealthy(8082);

        assert_eq!(pool.next_worker().unwrap().port, 8083);
    }

    #[test]
    fn test_returns_none_when_all_unhealthy() {
        let pool = WorkerPool::new(8081, 2);
        pool.mark_unhealthy(8081);
        pool.mark_unhealthy(8082);
        assert!(pool.next_worker().is_none());
    }

    // --- Cache affinity ---

    #[test]
    fn test_prefers_warmer_cache_on_tie() {
        let pool = WorkerPool::new(8081, 3);
        // All have 0 inflight, but 8083 has served more requests (warmer cache)
        pool.workers[0].total_requests.store(10, Ordering::Relaxed);
        pool.workers[1].total_requests.store(50, Ordering::Relaxed);
        pool.workers[2].total_requests.store(100, Ordering::Relaxed);

        assert_eq!(pool.next_worker().unwrap().port, 8083);
    }

    #[test]
    fn test_inflight_beats_warmth() {
        let pool = WorkerPool::new(8081, 2);
        // 8081: warm cache but busy, 8082: cold but idle
        pool.workers[0].total_requests.store(1000, Ordering::Relaxed);
        pool.workers[0].inflight.store(3, Ordering::Relaxed);
        pool.workers[1].total_requests.store(0, Ordering::Relaxed);
        pool.workers[1].inflight.store(0, Ordering::Relaxed);

        // Least connections wins — idle worker gets picked
        assert_eq!(pool.next_worker().unwrap().port, 8082);
    }

    #[test]
    fn test_tie_breaker_rotates_cold_workers() {
        let pool = WorkerPool::new(8081, 3);
        // All workers are identically cold — tie breaker rotates
        let p1 = pool.next_worker().unwrap().port;
        let p2 = pool.next_worker().unwrap().port;
        let p3 = pool.next_worker().unwrap().port;
        // Should cycle through all three (not always pick the first)
        let mut ports = vec![p1, p2, p3];
        ports.sort();
        ports.dedup();
        assert_eq!(ports.len(), 3, "tie-breaker should distribute across all workers");
    }

    // --- Health recovery ---

    #[test]
    fn test_mark_healthy_recovers_worker() {
        let pool = WorkerPool::new(8081, 2);
        pool.mark_unhealthy(8081);
        pool.mark_unhealthy(8082);
        assert!(pool.next_worker().is_none());

        pool.mark_healthy(8081);
        assert_eq!(pool.healthy_count(), 1);
        assert_eq!(pool.next_worker().unwrap().port, 8081);
    }

    // --- Inflight tracking ---

    #[test]
    fn test_inflight_increment_decrement() {
        let pool = WorkerPool::new(8081, 1);
        let w = pool.next_worker().unwrap();
        w.inflight.fetch_add(1, Ordering::Relaxed);
        w.inflight.fetch_add(1, Ordering::Relaxed);
        assert_eq!(w.inflight.load(Ordering::Relaxed), 2);
        w.inflight.fetch_sub(1, Ordering::Relaxed);
        assert_eq!(w.inflight.load(Ordering::Relaxed), 1);
    }

    // --- Status ---

    #[test]
    fn test_status_returns_all_workers() {
        let pool = WorkerPool::new(8081, 2);
        pool.mark_unhealthy(8082);
        let status = pool.status();
        let workers = status["workers"].as_array().unwrap();
        assert_eq!(workers.len(), 2);
        assert_eq!(status["healthy_count"], 1);
        assert_eq!(status["total_count"], 2);
        assert_eq!(status["strategy"], "least-connections + cache-affinity");
        assert!(workers[0]["healthy"].as_bool().unwrap());
        assert!(!workers[1]["healthy"].as_bool().unwrap());
    }

    // --- Session affinity ---

    #[test]
    fn test_session_sticks_to_same_worker() {
        let pool = WorkerPool::new(8081, 3);
        let w1 = pool.next_worker_for_session("sess-abc").unwrap().port;
        let w2 = pool.next_worker_for_session("sess-abc").unwrap().port;
        let w3 = pool.next_worker_for_session("sess-abc").unwrap().port;
        assert_eq!(w1, w2);
        assert_eq!(w2, w3);
    }

    #[test]
    fn test_different_sessions_can_get_different_workers() {
        let pool = WorkerPool::new(8081, 3);
        // First session gets assigned
        let s1 = pool.next_worker_for_session("sess-1").unwrap().port;
        // Simulate load on that worker so second session picks differently
        pool.workers.iter().find(|w| w.port == s1).unwrap()
            .inflight.store(5, Ordering::Relaxed);
        let s2 = pool.next_worker_for_session("sess-2").unwrap().port;
        assert_ne!(s1, s2, "different sessions should spread across workers when load differs");
    }

    #[test]
    fn test_session_reassigns_when_worker_unhealthy() {
        let pool = WorkerPool::new(8081, 2);
        let original = pool.next_worker_for_session("sess-x").unwrap().port;
        pool.mark_unhealthy(original);
        let new = pool.next_worker_for_session("sess-x").unwrap().port;
        assert_ne!(original, new, "should reassign to healthy worker");
    }

    #[test]
    fn test_session_returns_none_when_all_unhealthy() {
        let pool = WorkerPool::new(8081, 2);
        pool.next_worker_for_session("sess-y"); // assign
        pool.mark_unhealthy(8081);
        pool.mark_unhealthy(8082);
        assert!(pool.next_worker_for_session("sess-y").is_none());
    }

    #[test]
    fn test_release_session() {
        let pool = WorkerPool::new(8081, 2);
        pool.next_worker_for_session("sess-z");
        assert_eq!(pool.session_count(), 1);
        pool.release_session("sess-z");
        assert_eq!(pool.session_count(), 0);
    }

    #[test]
    fn test_session_count() {
        let pool = WorkerPool::new(8081, 2);
        pool.next_worker_for_session("a");
        pool.next_worker_for_session("b");
        pool.next_worker_for_session("c");
        assert_eq!(pool.session_count(), 3);
    }

    // --- Thread safety ---

    #[test]
    fn test_pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WorkerPool>();
    }

    #[test]
    fn test_concurrent_access() {
        let pool = Arc::new(WorkerPool::new(8081, 4));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = pool.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        let w = pool.next_worker().unwrap();
                        w.inflight.fetch_add(1, Ordering::Relaxed);
                        w.total_requests.fetch_add(1, Ordering::Relaxed);
                        w.inflight.fetch_sub(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let total: u64 = pool
            .workers
            .iter()
            .map(|w| w.total_requests.load(Ordering::Relaxed))
            .sum();
        assert_eq!(total, 800);
    }
}
