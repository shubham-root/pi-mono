//! Read tool - read file contents with optional offset and limit.
//! Phase 2.3

use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};
use std::io::Read;

#[derive(Debug, Clone)]
pub struct ReadTool;

#[async_trait::async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a file with optional line offset and limit. \
         Supports text files and images (returns base64). \
         Binary files are rejected."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file (relative to CWD)"
                },
                "offset": {
                    "type": "number",
                    "description": "Starting line number (0-indexed, inclusive)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read (default: entire file)"
                }
            },
            "required": ["file"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, input: serde_json::Value) -> Result<ToolResult> {
        // Parse input
        let input_obj = input.as_object()
            .ok_or_else(|| anyhow!("Input must be a JSON object"))?;

        let file_path = input_obj.get("file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing required 'file' field (string)"))?;

        let offset: usize = input_obj.get("offset")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);

        let limit: Option<usize> = input_obj.get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        // Resolve path relative to CWD
        let cwd = &ctx.cwd;
        let full_path = if Path::new(file_path).is_absolute() {
            PathBuf::from(file_path)
        } else {
            cwd.join(file_path)
        };

        // Canonicalize to check bounds and avoid escapes
        let canonical = fs::canonicalize(&full_path)
            .map_err(|e| anyhow!("Failed to resolve '{}': {}", file_path, e))?;

        // Ensure file exists and is a file
        if !canonical.is_file() {
            return Err(anyhow!("Path '{}' is not a regular file", file_path));
        }

        // Check file size (prevent reading huge files)
        let metadata = fs::metadata(&canonical)?;
        if metadata.len() > 10 * 1024 * 1024 {
            return Ok(ToolResult::success(
                "File is too large (>10MB). Use offset/limit to read portions."
                    .to_string(),
            ));
        }

        // Detect if file is an image (by extension heuristics)
        let is_image = matches!(
            canonical.extension().and_then(|ext| ext.to_str()),
            Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp")
        );

        if is_image {
            // Return base64 representation
            let data = fs::read(&canonical)?;
            let base64 = base64::encode(&data);
            let mime = match canonical.extension().and_then(|ext| ext.to_str()) {
                Some("png") => "image/png",
                Some("jpg" | "jpeg") => "image/jpeg",
                Some("gif") => "image/gif",
                Some("webp") => "image/webp",
                Some("bmp") => "image/bmp",
                _ => "application/octet-stream",
            };
            let content = format!("data:{};base64,{}", mime, base64);
            return Ok(ToolResult::success(content));
        }

        // Read file as text
        let content = fs::read_to_string(&canonical)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::InvalidData {
                    anyhow!("File appears to be binary; cannot read as text")
                } else {
                    anyhow!("Failed to read file: {}", e)
                }
            })?;

        // Apply offset/limit by lines
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        if offset >= total_lines {
            return Ok(ToolResult::success(String::new()));
        }

        let end = match limit {
            Some(lim) => std::cmp::min(offset + lim, total_lines),
            None => total_lines,
        };

        let selected = &lines[offset..end];
        let output = selected.join("\n");

        if offset > 0 || end < total_lines {
            let mut header = String::new();
            if offset > 0 {
                header.push_str(&format!("... skipping {} lines ...\n", offset));
            }
            header.push_str(&output);
            if end < total_lines {
                header.push_str(&format!("\n... {} more lines ...", total_lines - end));
            }
            Ok(ToolResult::success(header))
        } else {
            Ok(ToolResult::success(output))
        }
    }
}
