//! System prompt assembly from templates and settings.
//! To be implemented in Phase 2.12.

use anyhow::Result;
use pi_tools::Tool;

pub struct SystemPromptBuilder {
    // stub
}

impl SystemPromptBuilder {
    pub fn new() -> Self {
        Self {}
    }

    pub fn with_cwd(mut self, _cwd: &str) -> Self {
        self
    }

    pub fn with_os_info(mut self, _os: &str) -> Self {
        self
    }

    pub fn with_tools(mut self, _tools: &[&dyn Tool]) -> Self {
        self
    }

    pub fn with_custom_instructions(mut self, _instructions: &str) -> Self {
        self
    }

    pub fn build(self) -> String {
        "You are pi, an AI coding assistant.".to_string()
    }
}
