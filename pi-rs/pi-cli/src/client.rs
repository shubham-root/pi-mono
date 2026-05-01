//! Client for connecting to the daemon.
//! To be implemented in Phase 5.3.

use anyhow::Result;
use pi_server::ipc::IpcClient;

pub struct PiClient {
    client: IpcClient,
}

impl PiClient {
    pub async fn connect() -> Result<Self> {
        unimplemented!("IPC client - Phase 5.3")
    }

    pub async fn attach(&self, session_id: &str) -> Result<()> {
        unimplemented!("Attach to session")
    }

    pub async fn send_prompt(&self, session_id: &str, prompt: &str) -> Result<()> {
        unimplemented!("Send prompt")
    }
}
