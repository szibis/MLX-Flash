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

    pub async fn prefetch(&self, layer: u32, experts: &[u32]) {
        for &expert in experts {
            if self.cache.fetch(layer, expert).is_some() {
                continue;
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

    pub async fn fetch_or_load(&self, layer: u32, expert: u32) -> std::io::Result<Vec<u8>> {
        if let Some(data) = self.cache.fetch(layer, expert) {
            return Ok(data);
        }
        let data = self.store.load_expert_async(layer, expert).await?;
        self.cache.insert(layer, expert, data.clone());
        Ok(data)
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_store::ExpertStore;

    fn create_test_experts(dir: &std::path::Path, layers: u32, experts: u32, size: usize) {
        for l in 0..layers {
            let layer_dir = dir.join(format!("layer_{:03}", l));
            std::fs::create_dir_all(&layer_dir).unwrap();
            for e in 0..experts {
                let data: Vec<u8> = (0..size)
                    .map(|i| ((l * experts + e) as u8).wrapping_add(i as u8))
                    .collect();
                std::fs::write(layer_dir.join(format!("expert_{:04}.bin", e)), &data).unwrap();
            }
        }
    }

    #[tokio::test]
    async fn test_prefetch_loads_into_cache() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_experts(tmp.path(), 1, 4, 64);

        let cache = Arc::new(LcpCache::new(1024 * 1024));
        let store = Arc::new(ExpertStore::new(tmp.path().to_path_buf()));
        let prefetcher = AsyncPrefetcher::new(cache.clone(), store);

        prefetcher.prefetch(0, &[0, 1, 2, 3]).await;

        // Give the spawned tasks time to complete
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        for expert in 0..4u32 {
            assert!(
                cache.fetch(0, expert).is_some(),
                "expert {} should be cached after prefetch",
                expert
            );
        }
    }

    #[tokio::test]
    async fn test_prefetch_skips_cached() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_experts(tmp.path(), 1, 2, 32);

        let cache = Arc::new(LcpCache::new(1024 * 1024));
        let store = Arc::new(ExpertStore::new(tmp.path().to_path_buf()));

        // Pre-insert specific data for expert (0, 0)
        let original_data = vec![0xABu8; 32];
        cache.insert(0, 0, original_data.clone());

        let prefetcher = AsyncPrefetcher::new(cache.clone(), store);
        prefetcher.prefetch(0, &[0]).await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Original data should be unchanged (prefetch skipped it)
        let fetched = cache.fetch(0, 0).unwrap();
        assert_eq!(fetched, original_data, "pre-cached data should not be overwritten");
    }

    #[tokio::test]
    async fn test_fetch_or_load() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_experts(tmp.path(), 1, 1, 128);

        let cache = Arc::new(LcpCache::new(1024 * 1024));
        let store = Arc::new(ExpertStore::new(tmp.path().to_path_buf()));
        let prefetcher = AsyncPrefetcher::new(cache.clone(), store);

        // First call: cache miss — loads from disk
        let data1 = prefetcher.fetch_or_load(0, 0).await.unwrap();
        assert_eq!(data1.len(), 128);

        // Second call: cache hit — should return same data
        let data2 = prefetcher.fetch_or_load(0, 0).await.unwrap();
        assert_eq!(data1, data2);

        // Verify it actually hit the cache
        let stats = cache.stats();
        assert!(stats.hits >= 1, "expected at least one cache hit");
    }
}
