//! Find tool - find files by glob pattern.
//! Phase 2.7

use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Result, anyhow};
use ignore::WalkBuilder;
use std::path::Path;

/// Very simple glob matcher: supports "*" suffix (e.g., "*.rs") and exact names.
fn matches_pattern(path: &Path, pattern: &str) -> bool {
    let file_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name,
        None => return false,
    };
    if pattern == "*" {
        return true;
    }
    if pattern.starts_with("*.") {
        let ext = &pattern[2..];
        if let Some(p) = file_name.rfind('.') {
            return file_name[p+1..].eq_ignore_ascii_case(ext);
        }
        return false;
    }
    // Exact match (including path separators? We'll match full path string simply)
    // For patterns with slashes, we could compare path components but skip for now.
    file_name == pattern
}

#[derive(Debug, Clone)]
pub struct FindTool;

#[async_trait::async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Respects .gitignore."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., \"*.rs\", \"**/*.ts\")"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search (default: current working directory)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, input: serde_json::Value) -> Result<ToolResult> {
        let input_obj = input.as_object()
            .ok_or_else(|| anyhow!("Input must be a JSON object"))?;

        let pattern = input_obj.get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing required 'pattern' field (string)"))?;

        let search_path = input_obj.get("path")
            .and_then(|v| v.as_str())
            .map(Path::new)
            .unwrap_or(&ctx.cwd);

        let limit: Option<usize> = input_obj.get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        // Build walker (respect .gitignore)
        let mut builder = WalkBuilder::new(search_path);
        builder.standard_filters(true);
        builder.git_ignore(true);
        builder.git_global(false);
        builder.git_exclude(true);

        let mut matches = Vec::new();

        for entry in builder.build() {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Simple glob matching using our lightweight matcher
            if matches_pattern(path, pattern) {
                matches.push(path.to_string_lossy().into_owned());
                if let Some(lim) = limit {
                    if matches.len() >= lim {
                        break;
                    }
                }
            }
        }

        if matches.is_empty() {
            Ok(ToolResult::success("No files found".to_string()))
        } else {
            Ok(ToolResult::success(matches.join("\n")))
        }
    }
}
