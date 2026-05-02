//! Fault isolation, reporting, and recovery.
//! To be implemented in Phase 4.17.

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    Loaded,
    Faulted,
    Disabled,
}

pub struct IsolationManager {
    states: Arc<tokio::sync::RwLock<HashMap<String, PluginState>>>,
}

impl IsolationManager {
    pub fn new() -> Self {
        Self {
            states: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub async fn record_fault(&self, _name: &str) {
        // Stub
    }

    pub async fn is_faulted(&self, name: &str) -> bool {
        self.states
            .read()
            .await
            .get(name)
            .copied()
            .unwrap_or(PluginState::Loaded)
            == PluginState::Faulted
    }
}
