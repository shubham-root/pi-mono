//! Bash tool - execute a shell command and return its output.
//! Phase 2.2

use crate::Tool;
use crate::ToolContext;
use crate::ToolResult;
use anyhow::{Result, anyhow};
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use std::process::Stdio;

/// Execute a shell command securely.
#[derive(Debug, Clone)]
pub struct BashTool;

impl BashTool {
    /// Default timeout in seconds.
    const DEFAULT_TIMEOUT: u64 = 120;
}

#[async_trait::async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output. \
         Use this to run commands, scripts, or programs. \
         Be mindful of potentially destructive operations; confirm before running."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 120)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, input: serde_json::Value) -> Result<ToolResult> {
        // Parse input JSON
        let input_obj = input.as_object()
            .ok_or_else(|| anyhow!("Input must be a JSON object"))?;

        let command = input_obj.get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'command' field (string)"))?;

        let timeout_secs = input_obj.get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(Self::DEFAULT_TIMEOUT);

        // Working directory
        let cwd = &ctx.cwd;

        // Prepare command
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
           .arg(command)
           .current_dir(cwd)
           .stdin(Stdio::null())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        // If abort token is set, we'll respect it via timeout context; more advanced cancellation later

        // Execute with timeout
        match timeout(Duration::from_secs(timeout_secs), cmd.output()).await {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().unwrap_or(-1);

                let mut result = String::new();
                if !stdout.is_empty() {
                    result.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str(&stderr);
                }

                if !output.status.success() {
                    return Ok(ToolResult::Error(format!("Command exited with code {}", exit_code)));
                }

                Ok(ToolResult::success(result))
            }
            Ok(Err(e)) => {
                Err(anyhow!("Failed to execute command: {}", e))
            }
            Err(_elapsed) => {
                Ok(ToolResult::Error(format!("Command timed out after {} seconds", timeout_secs)))
            }
        }
    }
}
