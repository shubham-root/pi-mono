//! IPC server and client.
//! To be implemented in Phase 5.2.

use anyhow::Result;

pub struct IpcServer {
    // stub
}

pub struct IpcClient {
    // stub
}

impl IpcServer {
    pub async fn start(&self) -> Result<()> {
        unimplemented!("IPC server")
    }
}

impl IpcClient {
    pub async fn connect() -> Result<Self> {
        unimplemented!("IPC client")
    }

    pub async fn attach(&self, session_id: &str) -> Result<()> {
        unimplemented!("Attach to session")
    }

    pub async fn send_prompt(&self, session_id: &str, prompt: &str) -> Result<()> {
        unimplemented!("Send prompt")
    }
}
