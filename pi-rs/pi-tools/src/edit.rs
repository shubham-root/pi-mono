//! Edit tool - perform multiple search-replace edits on a file.
//! Phase 2.5

use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct EditTool;

#[async_trait::async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Perform multiple search-replace edits on a file. \
         Each edit must be an object with `oldText` and `newText`. \
         All edits are applied against the original file content atomically."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file (relative to CWD)"
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text to find and replace"
                            },
                            "newText": {
                                "type": "string",
                                "description": "Replacement text"
                            }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["file", "edits"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, input: serde_json::Value) -> Result<ToolResult> {
        let input_obj = input.as_object()
            .ok_or_else(|| anyhow!("Input must be a JSON object"))?;

        let file_path = input_obj.get("file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing required 'file' field (string)"))?;

        let edits_array = input_obj.get("edits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("Missing or invalid 'edits' field (array)"))?;

        if edits_array.is_empty() {
            return Err(anyhow!("Edits array cannot be empty"));
        }

        if edits_array.len() > 100 {
            return Err(anyhow!("Too many edits (max 100)"));
        }

        // Resolve path relative to CWD
        let full_path: PathBuf = if Path::new(file_path).is_absolute() {
            file_path.into()
        } else {
            ctx.cwd.join(file_path)
        };

        let canonical = fs::canonicalize(&full_path)
            .map_err(|e| anyhow!("Failed to resolve '{}': {}", file_path, e))?;

        // Ensure file exists and is within CWD
        if !canonical.is_file() {
            return Err(anyhow!("Path '{}' is not a regular file", file_path));
        }
        if !canonical.starts_with(&ctx.cwd) {
            return Err(anyhow!("Access denied: file outside working directory"));
        }

        // Read original content
        let original = fs::read_to_string(&canonical)?;

        // Validate all oldText exist and are unique
        for edit in edits_array {
            let edit_obj = edit.as_object()
                .ok_or_else(|| anyhow!("Each edit must be an object"))?;
            let old_text = edit_obj.get("oldText")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("Edit missing 'oldText'"))?;

            let count = original.matches(old_text).count();
            if count == 0 {
                return Err(anyhow!("oldText not found in file: {:?}", old_text));
            }
            if count > 1 {
                return Err(anyhow!("oldText is ambiguous: found {} occurrences (must be unique)", count));
            }
        }

        // Apply all edits atomically against original content
        let mut result = original.clone();
        for (idx, edit) in edits_array.iter().enumerate() {
            let edit_obj = edit.as_object().unwrap();
            let old_text = edit_obj.get("oldText").unwrap().as_str().unwrap();
            let new_text = edit_obj.get("newText").unwrap().as_str().unwrap();

            if !result.contains(old_text) {
                return Err(anyhow!(
                    "Edit #{} failed: oldText no longer present after previous edits. Apply fewer edits at once.",
                    idx + 1
                ));
            }

            result = result.replacen(old_text, new_text, 1);
        }

        // Generate simple diff
        let mut out = String::new();
        out.push_str(&format!("--- {}\n", file_path));
        out.push_str(&format!("+++ {}\n", file_path));
        for line in original.lines() {
            out.push('-');
            out.push_str(line);
            out.push('\n');
        }
        for line in result.lines() {
            out.push('+');
            out.push_str(line);
            out.push('\n');
        }

        // Write back atomically
        fs::write(&canonical, result)?;

        Ok(ToolResult::success(out))
    }
}
