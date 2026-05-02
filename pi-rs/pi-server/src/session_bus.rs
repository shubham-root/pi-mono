//! Cross-session communication bus.
//! To be implemented in Phase 5.5.

use anyhow::Result;

pub struct CrossSessionBus {
    // stub
}

impl CrossSessionBus {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn send(&self, _from: &str, _to: &str, _message: &[u8]) -> Result<()> {
        unimplemented!("Cross-session messaging")
    }

    pub async fn broadcast(&self, _from: &str, _message: &[u8]) -> Result<()> {
        unimplemented!("Broadcast messaging")
    }
}
