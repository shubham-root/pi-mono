//! Plugin trait and helper types.
//! Full implementation in Phase 4.

use serde_json;
use tokio_util::sync::CancellationToken as TokioCancellationToken;

/// Trait that plugins must implement.

/// Cancellation token for aborting operations.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    inner: TokioCancellationToken,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self { inner: TokioCancellationToken::new() }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }
}
pub trait Plugin {
    /// Called when the plugin is loaded.
    fn load(&mut self) -> Result<(), anyhow::Error> {
        Ok(())
    }

    /// Called when the plugin receives an event.
    fn on_event(&self, _event: &PluginEvent) -> Result<(), anyhow::Error> {
        Ok(())
    }

    /// Called when another plugin sends a pipe message.
    fn on_pipe(&self, _from: &str, _message: &[u8]) -> Result<(), anyhow::Error> {
        Ok(())
    }

    /// Called before unload to allow state serialization.
    fn save_state(&self) -> Result<Vec<u8>, anyhow::Error> {
        Ok(Vec::new())
    }

    /// Restore state after loading.
    fn restore_state(&mut self, _state: &[u8]) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

/// Events that plugins can subscribe to.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PluginEvent {
    SessionStart,
    SessionEnd,
    UserMessage { content: String },
    AssistantMessage { content: String },
    ToolCall { name: String, input: serde_json::Value },
    ToolResult { result: String },
}

/// Context passed to tool functions.
#[derive(Debug, Clone)]
pub struct ToolContext<'a> {
    pub cwd: &'a std::path::Path,
    pub abort: Option<&'a tokio_util::sync::CancellationToken>,
}

/// Result from a tool call.
#[derive(Debug, Clone)]
pub enum ToolOutcome {
    Success(String),
    Error(String),
}
