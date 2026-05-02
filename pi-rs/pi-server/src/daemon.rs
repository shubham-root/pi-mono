//! Daemon process management.
//! To be implemented in Phase 5.1.

use anyhow::Result;

pub struct PiServer {
    // stub
}

impl PiServer {
    pub async fn start(&self) -> Result<()> {
        unimplemented!("Start daemon")
    }

    pub async fn stop(&self) -> Result<()> {
        unimplemented!("Stop daemon")
    }

    pub async fn status(&self) -> Result<()> {
        unimplemented!("Daemon status")
    }
}
