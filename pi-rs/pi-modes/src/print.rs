//! Print mode - single-turn non-interactive mode.
//! To be implemented in Phase 1.7 with full tool support in Phase 2.15.

use anyhow::Result;
use pi_ai::types::{Model, Context, StreamOptions};
use pi_core::Agent;

pub struct PrintMode {
    // stub
}

impl PrintMode {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn run(&mut self, prompt: &str, model: Option<&str>) -> Result<()> {
        unimplemented!("Print mode - Phase 1.7")
    }
}
