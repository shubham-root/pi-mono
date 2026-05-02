//! Write tool - write content to a file.
//! Phase 2.4

use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct WriteTool;

#[async_trait::async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories if needed. \
         Returns a unified diff showing changes."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file (relative to CWD)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                }
            },
            "required": ["file", "content"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, input: serde_json::Value) -> Result<ToolResult> {
        let input_obj = input.as_object()
            .ok_or_else(|| anyhow!("Input must be a JSON object"))?;

        let file_path = input_obj.get("file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing required 'file' field (string)"))?;

        let content = input_obj.get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing required 'content' field (string)"))?;

        // Resolve path relative to CWD
        let full_path: PathBuf = if Path::new(file_path).is_absolute() {
            file_path.into()
        } else {
            ctx.cwd.join(file_path)
        };

        // Canonicalize to validate path and prevent escapes
        let canonical = match fs::canonicalize(&full_path) {
            Ok(path) => path,
            Err(_) => {
                // File likely doesn't exist yet; check parent
                let parent = full_path.parent().ok_or_else(|| anyhow!("Invalid file path"))?;
                let _ = fs::canonicalize(parent)
                    .map_err(|e| anyhow!("Invalid parent directory: {}", e))?;
                full_path.clone()
            }
        };

        // Ensure the resolved path is within allowed CWD
        if !canonical.starts_with(&ctx.cwd) {
            return Err(anyhow!("Access denied: path outside working directory"));
        }

        // Read old content if file exists
        let old_content = if canonical.exists() && canonical.is_file() {
            fs::read_to_string(&canonical).ok()
        } else {
            None
        };

        // Ensure parent directories exist
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write the new content
        fs::write(&full_path, content)?;

        // Generate simple diff
        let diff = if let Some(old) = old_content {
            let mut out = String::new();
            out.push_str(&format!("--- {}\n", file_path));
            out.push_str(&format!("+++ {}\n", file_path));
            for line in old.lines() {
                out.push('-');
                out.push_str(line);
                out.push('\n');
            }
            for line in content.lines() {
                out.push('+');
                out.push_str(line);
                out.push('\n');
            }
            out
        } else {
            format!("--- /dev/null\n+++ {}\n+{}\n", file_path, content)
        };

        Ok(ToolResult::success(diff))
    }
}
