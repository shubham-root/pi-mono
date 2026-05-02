//! Ls tool - list directory contents.
//! Phase 2.8

use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Result, anyhow};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LsTool;

#[async_trait::async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List directory contents with file types and sizes."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to list (default: current working directory)"
                }
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, input: serde_json::Value) -> Result<ToolResult> {
        let input_obj = input.as_object()
            .ok_or_else(|| anyhow!("Input must be a JSON object"))?;

        let list_path = input_obj.get("path")
            .and_then(|v| v.as_str())
            .map(Path::new)
            .unwrap_or(&ctx.cwd);

        // Resolve to absolute path and ensure it's within CWD
        let full_path = if list_path.is_absolute() {
            list_path.to_path_buf()
        } else {
            ctx.cwd.join(list_path)
        };

        let canonical = fs::canonicalize(&full_path)
            .map_err(|e| anyhow!("Failed to resolve path: {}", e))?;

        if !canonical.starts_with(&ctx.cwd) {
            return Err(anyhow!("Access denied: path outside working directory"));
        }

        let entries = match fs::read_dir(&canonical) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ToolResult::success(format!("Directory not found: {}", canonical.display())));
            }
            Err(e) => return Err(e.into()),
        };

        let mut lines = Vec::new();
        let mut max_name_len = 0;

        // Collect entries first for formatting
        let mut entries_info: Vec<(String, String)> = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let file_type = if metadata.is_dir() {
                "d"
            } else if metadata.is_file() {
                "f"
            } else {
                "?"
            };
            let size = if metadata.is_file() {
                format!("{:>8}", metadata.len())
            } else {
                "        ".to_string()
            };
            let line = format!("{}  {}  {}", file_type, size, name);
            entries_info.push((name.clone(), line));
            max_name_len = max_name_len.max(name.len());
        }

        // Sort by name
        entries_info.sort_by(|a, b| a.0.cmp(&b.0));

        for (_, line) in entries_info {
            lines.push(line);
        }

        if lines.is_empty() {
            Ok(ToolResult::success("Directory is empty".to_string()))
        } else {
            Ok(ToolResult::success(lines.join("\n")))
        }
    }
}
