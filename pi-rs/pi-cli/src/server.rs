//! Daemon control commands.
//! To be implemented in Phase 5.1.

use anyhow::Result;
use pi_server::daemon::PiServer;

pub async fn start() -> Result<()> {
    unimplemented!("Start daemon")
}

pub async fn stop() -> Result<()> {
    unimplemented!("Stop daemon")
}

pub async fn status() -> Result<()> {
    unimplemented!("Daemon status")
}
