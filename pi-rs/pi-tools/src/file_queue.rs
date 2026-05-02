//! File mutation queue - serialize concurrent writes/edits to same file.
//! Phase 2.9

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::future::Future;

/// Global queue thatserializes per-file mutations.
#[derive(Debug, Clone, Default)]
pub struct FileMutationQueue {
    locks: Arc<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>>,
}

impl FileMutationQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute a closure with a per-file lock held.
    /// Ensures that concurrent edits to the same file are serialized.
    pub async fn with_lock<P, F, T>(&self, path: P, f: F) -> T
    where
        P: AsRef<std::path::Path>,
        F: Future<Output = T>,
    {
        let path = path.as_ref().to_path_buf();
        let mut locks = self.locks.lock().await;
        let lock = locks
            .entry(path.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        drop(locks);

        let _guard = lock.lock().await;
        f.await
    }
}
