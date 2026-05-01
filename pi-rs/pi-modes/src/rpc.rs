//! RPC mode - JSON-RPC over stdin/stdout.
//! To be implemented in Phase 5.7.

use anyhow::Result;
use serde_json;

pub struct RpcMode {
    // stub
}

impl RpcMode {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn run(&mut self) -> Result<()> {
        unimplemented!("RPC mode - Phase 5.7")
    }
}
