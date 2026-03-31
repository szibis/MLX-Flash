use dashmap::DashMap;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// A single cached expert weight blob.
struct CacheEntry {
    data: Vec<u8>,
    frequency: u64,
    last_step: u64,
}

/// Statistics snapshot returned by [`LcpCache::stats`].
#[derive(Debug, Serialize)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes_used: usize,
    pub capacity_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

/// Lock-free LCP (Least Critical Priority) expert weight cache backed by DashMap.
///
/// Priority formula: `P = frequency * 0.25^(steps_since_last_use / 128)`
///
/// On capacity overflow the entry with the lowest priority is evicted.
pub struct LcpCache {
    map: Arc<DashMap<(u32, u32), CacheEntry>>,
    capacity_bytes: usize,
    current_bytes: AtomicUsize,
    step: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl LcpCache {
    /// Create a new cache with the given byte capacity.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            map: Arc::new(DashMap::new()),
            capacity_bytes,
            current_bytes: AtomicUsize::new(0),
            step: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Look up `(layer, expert)` in the cache.
    ///
    /// On hit: increments frequency and updates `last_step`, returns cloned data.
    /// On miss: increments miss counter, returns `None`.
    pub fn fetch(&self, layer: u32, expert: u32) -> Option<Vec<u8>> {
        let current = self.step.load(Ordering::Relaxed);
        if let Some(mut entry) = self.map.get_mut(&(layer, expert)) {
            entry.frequency += 1;
            entry.last_step = current;
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.data.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert `data` for `(layer, expert)`, evicting lowest-priority entries
    /// until there is room.
    pub fn insert(&self, layer: u32, expert: u32, data: Vec<u8>) {
        let new_size = data.len();

        // Evict until we have room (or map is empty).
        while self.current_bytes.load(Ordering::Relaxed) + new_size > self.capacity_bytes
            && !self.map.is_empty()
        {
            self.evict_lowest();
        }

        // If the key already exists, subtract its old size first.
        if let Some(old) = self.map.get(&(layer, expert)) {
            self.current_bytes
                .fetch_sub(old.data.len(), Ordering::Relaxed);
        }

        let current = self.step.load(Ordering::Relaxed);
        self.map.insert(
            (layer, expert),
            CacheEntry {
                data,
                frequency: 1,
                last_step: current,
            },
        );
        self.current_bytes.fetch_add(new_size, Ordering::Relaxed);
    }

    /// Advance the global step counter by 1.
    pub fn advance_step(&self) {
        self.step.fetch_add(1, Ordering::Relaxed);
    }

    /// Return the current step counter value.
    pub fn current_step(&self) -> u64 {
        self.step.load(Ordering::Relaxed)
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        };
        CacheStats {
            entries: self.map.len(),
            bytes_used: self.current_bytes.load(Ordering::Relaxed),
            capacity_bytes: self.capacity_bytes,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate,
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Compute LCP priority for an entry given the current step.
    fn priority(entry: &CacheEntry, current_step: u64) -> f64 {
        let age = current_step.saturating_sub(entry.last_step) as f64;
        entry.frequency as f64 * (0.25_f64).powf(age / 128.0)
    }

    /// Remove the single entry with the lowest LCP priority.
    fn evict_lowest(&self) {
        let current = self.step.load(Ordering::Relaxed);

        // Find the key with minimum priority.
        let victim = self
            .map
            .iter()
            .min_by(|a, b| {
                let pa = Self::priority(a.value(), current);
                let pb = Self::priority(b.value(), current);
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| *r.key());

        if let Some(key) = victim {
            if let Some((_, entry)) = self.map.remove(&key) {
                self.current_bytes
                    .fetch_sub(entry.data.len(), Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_insert_and_fetch() {
        let cache = LcpCache::new(1024);
        cache.insert(0, 0, vec![1, 2, 3, 4]);
        let result = cache.fetch(0, 0);
        assert_eq!(result, Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_fetch_miss() {
        let cache = LcpCache::new(1024);
        let result = cache.fetch(99, 99);
        assert_eq!(result, None);
    }

    #[test]
    fn test_capacity_enforcement() {
        // 100-byte capacity; insert two 60-byte blobs — first should be evicted.
        let cache = LcpCache::new(100);
        let blob_a = vec![0u8; 60];
        let blob_b = vec![1u8; 60];

        cache.insert(0, 0, blob_a);
        cache.insert(0, 1, blob_b);

        // Total would be 120 > 100, so the first entry must have been evicted.
        let stats = cache.stats();
        assert!(
            stats.bytes_used <= 100,
            "bytes_used {} exceeds capacity 100",
            stats.bytes_used
        );
        assert_eq!(stats.evictions, 1);
        // Second insert survived.
        assert!(cache.fetch(0, 1).is_some());
    }

    #[test]
    fn test_lcp_priority_keeps_hot() {
        // 100-byte cache; hot expert fetched many times, cold expert fetched once.
        // When a third insert forces eviction, the cold one should go.
        let cache = LcpCache::new(100);

        cache.insert(0, 0, vec![0u8; 40]); // hot
        cache.insert(0, 1, vec![0u8; 40]); // cold

        // Heat up expert 0.
        for _ in 0..20 {
            cache.fetch(0, 0);
            cache.advance_step();
        }

        // This insert should evict the cold expert (0,1), not the hot (0,0).
        cache.insert(0, 2, vec![0u8; 40]);

        assert!(
            cache.fetch(0, 0).is_some(),
            "hot expert should survive eviction"
        );
        assert!(
            cache.fetch(0, 1).is_none(),
            "cold expert should have been evicted"
        );
    }

    #[test]
    fn test_advance_step() {
        let cache = LcpCache::new(1024);
        assert_eq!(cache.current_step(), 0);
        cache.advance_step();
        assert_eq!(cache.current_step(), 1);
        cache.advance_step();
        cache.advance_step();
        assert_eq!(cache.current_step(), 3);
    }

    #[test]
    fn test_stats() {
        let cache = LcpCache::new(1024);
        cache.insert(0, 0, vec![42u8; 16]);
        cache.insert(0, 1, vec![43u8; 8]);

        // One hit, one miss.
        let _ = cache.fetch(0, 0);
        let _ = cache.fetch(9, 9);

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.bytes_used, 24);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_concurrent_access() {
        // 4 threads performing inserts and fetches concurrently.
        let cache = Arc::new(LcpCache::new(16 * 1024 * 1024)); // 16 MB — no eviction pressure

        let handles: Vec<_> = (0..4u32)
            .map(|t| {
                let c = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for i in 0..64u32 {
                        c.insert(t, i, vec![t as u8; 64]);
                        let _ = c.fetch(t, i);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let stats = cache.stats();
        // Each thread inserted 64 entries; all should still be present (capacity is large).
        assert_eq!(stats.entries, 4 * 64);
    }

    #[test]
    fn test_warmup_convergence() {
        // Simulate 50 tokens, each accessing one of 4 experts.
        // After warm-up the hit rate should exceed 90 %.
        let cache = LcpCache::new(4 * 1024); // enough for all 4 experts (100 bytes each)

        let expert_data: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 100]).collect();

        // Token 0: cold start — all misses, populate cache.
        for (expert, data) in expert_data.iter().enumerate() {
            if cache.fetch(0, expert as u32).is_none() {
                cache.insert(0, expert as u32, data.clone());
            }
        }
        cache.advance_step();

        // Tokens 1–49: warm run — fetches only, no inserts needed.
        for _token in 1..50usize {
            for expert in 0..4u32 {
                cache.fetch(0, expert);
            }
            cache.advance_step();
        }

        let stats = cache.stats();
        assert!(
            stats.hit_rate > 0.90,
            "expected hit rate > 90%, got {:.1}%",
            stats.hit_rate * 100.0
        );
    }
}
