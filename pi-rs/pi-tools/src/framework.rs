//! Framework for defining and executing tools.
//! To be implemented in Phase 2.1

use async_trait::async_trait;
use serde_json;
use std::collections::HashMap;
use std::fmt::Debug;

/// Result of a tool execution.
#[derive(Debug, Clone)]
pub enum ToolResult {
    Success(String),
    Error(String),
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self::Success(content.into())
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self::Error(content.into())
    }
}

/// Context provided to tool execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Current working directory
    pub cwd: std::path::PathBuf,
    /// Cancellation token
    pub abort: Option<std::sync::Arc<tokio_util::sync::CancellationToken>>,
    /// Environment variables
    pub env: HashMap<String, String>,
}

/// Trait that all tools must implement.
#[async_trait]
pub trait Tool: Send + Sync + Debug + 'static {
    /// Unique name identifier for the tool.
    fn name(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    fn schema(&self) -> serde_json::Value;

    /// Execute the tool with the given input.
    async fn execute(&self, ctx: &ToolContext, input: serde_json::Value) -> Result<ToolResult, anyhow::Error>;
}

/// Helper to create a tool schema from a Rust type that implements `schemars::JsonSchema`.
pub fn schema_from<T>() -> serde_json::Value {
    // Placeholder - full implementation in Phase 2.1
    serde_json::json!({"type": "object", "properties": {}})
}
