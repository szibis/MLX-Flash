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

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_load_expert() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_experts(tmp.path(), 1, 2, 256);

        let store = ExpertStore::new(tmp.path().to_path_buf());
        let data = store.load_expert(0, 1).unwrap();
        assert_eq!(data.len(), 256);
    }

    #[test]
    fn test_load_nonexistent() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ExpertStore::new(tmp.path().to_path_buf());
        let result = store.load_expert(0, 0);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_expert_async() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_experts(tmp.path(), 2, 4, 128);

        let store = ExpertStore::new(tmp.path().to_path_buf());
        let data = store.load_expert_async(1, 2).await.unwrap();
        assert_eq!(data.len(), 128);

        // Verify contents match sync load
        let sync_data = store.load_expert(1, 2).unwrap();
        assert_eq!(data, sync_data);
    }
}
