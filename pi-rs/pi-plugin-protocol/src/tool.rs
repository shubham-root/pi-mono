//! Tool schema and result types.
//!
//! Mirrors `pi_ai::ToolSchema` and `ToolResult` from `pi-tools` but
//! lives in the plugin-protocol crate so SDK languages can depend on
//! only one small crate instead of the whole `pi-ai` graph.

use serde::{Deserialize, Serialize};

/// The schema a plugin gives the host for a tool it wants to
/// register. The host forwards this verbatim to the model when
/// building a tool-use request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Machine-readable tool name the model will emit.
    pub name: String,
    /// One-line description used by the model for tool selection.
    pub description: String,
    /// JSON Schema for the tool's parameters. Must be a JSON
    /// object (`{"type":"object", "properties":...}`).
    pub parameters: serde_json::Value,
    /// Optional snippet for the "Available tools" section of the
    /// default system prompt. If omitted, the tool is still usable
    /// but doesn't get a bullet in the system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_snippet: Option<String>,
}

/// What a plugin returns after handling a `ToolCall`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Success or error.
    pub kind: ToolResultKind,
    /// The textual/JSON content delivered back to the model as the
    /// tool result. Plugin-owned tools currently return text only; a
    /// future extension will let them return images and JSON blocks.
    pub content: String,
    /// Optional structured details surfaced in the TUI (e.g. file
    /// paths touched, byte ranges modified). Not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Success vs error outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum ToolResultKind {
    Ok,
    Error,
}

impl ToolResult {
    /// Construct a successful result with the given text payload.
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            kind: ToolResultKind::Ok,
            content: content.into(),
            details: None,
        }
    }

    /// Construct an error result.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            kind: ToolResultKind::Error,
            content: content.into(),
            details: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_schema_roundtrip() {
        let s = ToolSchema {
            name: "greet".into(),
            description: "say hi".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }),
            prompt_snippet: Some("greet(name)".into()),
        };
        let j = serde_json::to_string(&s).unwrap();
        let back: ToolSchema = serde_json::from_str(&j).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn tool_result_constructors() {
        let ok = ToolResult::ok("done");
        assert_eq!(ok.kind, ToolResultKind::Ok);
        let err = ToolResult::error("nope");
        assert_eq!(err.kind, ToolResultKind::Error);
    }
}
