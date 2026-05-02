//! Grep tool - search files for a pattern using regex.
//! Phase 2.6

use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Result, anyhow};
use ignore::WalkBuilder;
use regex::Regex;
use std::path::Path;
use std::io::Read;

#[derive(Debug, Clone)]
pub struct GrepTool;

#[async_trait::async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a regex pattern across files. Respects .gitignore. \
         Returns matching lines with file paths and line numbers."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current working directory)"
                },
                "context": {
                    "type": "number",
                    "description": "Number of context lines before/after matches"
                },
                "include": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Glob patterns to include (e.g., [\"*.rs\", \"*.ts\"])"
                },
                "exclude": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Glob patterns to exclude"
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

        let regex = Regex::new(pattern)
            .map_err(|e| anyhow!("Invalid regex pattern: {}", e))?;

        let search_path = input_obj.get("path")
            .and_then(|v| v.as_str())
            .map(Path::new)
            .unwrap_or(&ctx.cwd);

        let context_lines = input_obj.get("context")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let include_patterns: Vec<&str> = input_obj.get("include")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let exclude_patterns: Vec<&str> = input_obj.get("exclude")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // Build walker with gitignore respect
        let mut builder = WalkBuilder::new(search_path);
        builder.standard_filters(true); // Respect .gitignore

        if !include_patterns.is_empty() {
            for pat in include_patterns {
                builder.add_custom_ignore_filename(pat);
            }
        }
        if !exclude_patterns.is_empty() {
            for pat in exclude_patterns {
                builder.add_custom_ignore_filename(pat);
            }
        }

        let mut results = Vec::new();

        for entry in builder.build() {
            let entry = entry?;
            if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                continue;
            }

            let path = entry.path();
            // Skip binary files heuristically
            if is_likely_binary(path)? {
                continue;
            }

            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let mut last_match_line = 0;
            for (line_num, line) in content.lines().enumerate() {
                if regex.is_match(line) {
                    // Include context lines before if needed
                    let start = line_num.saturating_sub(context_lines);
                    if last_match_line < start && !results.is_empty() {
                        results.push("...".to_string());
                    }

                    // Push context lines before
                    for before_line in content.lines().skip(start).take(line_num - start) {
                        results.push(format!("{}:{}: {}", path.display(), start + 1, before_line));
                    }

                    // Push matching line
                    results.push(format!("{}:{}: {}", path.display(), line_num + 1, line));

                    last_match_line = line_num + 1;
                }
            }
        }

        if results.is_empty() {
            Ok(ToolResult::success("No matches found".to_string()))
        } else {
            Ok(ToolResult::success(results.join("\n")))
        }
    }
}

/// Simple heuristic to detect binary files.
fn is_likely_binary(path: &Path) -> Result<bool> {
    let ext = path.extension().and_then(|s| s.to_str());
    match ext {
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" |
             "mp3" | "mp4" | "avi" | "mov" | "zip" | "tar" | "gz" | "exe" | "dll" | "so" | "dylib") => {
            return Ok(true);
        }
        _ => {
            // For text files with no extension or unknown, read first 512 bytes
            if let Ok(mut file) = std::fs::File::open(path) {
                let mut buf = [0u8; 512];
                let n = file.read(&mut buf)?;
                // Check for null bytes
                if buf[..n].contains(&0) {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}
