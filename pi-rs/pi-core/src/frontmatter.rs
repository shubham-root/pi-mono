//! Minimal YAML frontmatter parser.
//!
//! Parses the `---\n…\n---` preamble at the top of a markdown file
//! into a map of field → stringified value. Mirrors what the TS
//! `packages/coding-agent/src/utils/frontmatter.ts` does but only
//! for the subset of YAML the skill / prompt-template / theme
//! specs actually use: scalar string values, booleans, and numbers
//! at the top level.
//!
//! Nested objects and arrays are accepted but stored as the raw
//! source snippet (opaque to callers) so the file still loads.
//! Callers that need those fields can parse them with a real YAML
//! library.
//!
//! This keeps pi-core free of a full YAML dependency while matching
//! the TS feature surface for every frontmatter field Pi actually
//! reads today (`name`, `description`, `disable-model-invocation`,
//! `license`, `compatibility`, `allowed-tools`, and arbitrary
//! `metadata`).

use std::collections::HashMap;

/// One parsed frontmatter value.
#[derive(Debug, Clone, PartialEq)]
pub enum FrontmatterValue {
    /// A string scalar (unquoted or quoted).
    String(String),
    /// A boolean scalar (`true` / `false`, case-insensitive).
    Bool(bool),
    /// An integer scalar.
    Int(i64),
    /// A float scalar.
    Float(f64),
    /// A list or nested object we didn't try to parse further; stored
    /// as the raw source lines.
    Raw(String),
}

impl FrontmatterValue {
    /// View a string scalar, returning `None` for other variants.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
    /// View a boolean scalar.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Result of parsing a markdown file with optional frontmatter.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedFrontmatter {
    /// Key → value map of top-level scalar fields.
    pub fields: HashMap<String, FrontmatterValue>,
    /// Everything after the closing `---`, with the leading newline
    /// trimmed. If there was no frontmatter block, this is the
    /// entire input.
    pub body: String,
}

impl ParsedFrontmatter {
    /// Convenience: look up a string field.
    pub fn string(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(|v| v.as_str())
    }

    /// Convenience: look up a boolean field.
    pub fn bool(&self, key: &str) -> Option<bool> {
        self.fields.get(key).and_then(|v| v.as_bool())
    }
}

/// Parse optional frontmatter from `input`.
///
/// Contract:
/// - If `input` starts with `---\n` (or `---\r\n`), scan forward for
///   a line of exactly `---` and treat everything between as YAML.
/// - Otherwise return the whole input as the body with no fields.
/// - Parse errors inside the frontmatter produce an empty field map
///   — the file still loads; callers can validate required fields.
pub fn parse_frontmatter(input: &str) -> ParsedFrontmatter {
    let Some(rest) = strip_opening_fence(input) else {
        return ParsedFrontmatter {
            fields: HashMap::new(),
            body: input.to_string(),
        };
    };

    // Find the closing `---` line.
    let mut header_lines: Vec<&str> = Vec::new();
    let mut body_start: Option<usize> = None;
    let mut cursor = 0usize;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            body_start = Some(cursor + line.len());
            break;
        }
        header_lines.push(trimmed);
        cursor += line.len();
    }

    let Some(body_offset) = body_start else {
        // No closing fence — treat as no frontmatter.
        return ParsedFrontmatter {
            fields: HashMap::new(),
            body: input.to_string(),
        };
    };

    let fields = parse_header(&header_lines);
    let body = rest[body_offset..].to_string();
    ParsedFrontmatter { fields, body }
}

fn strip_opening_fence(input: &str) -> Option<&str> {
    if let Some(rest) = input.strip_prefix("---\n") {
        Some(rest)
    } else {
        input.strip_prefix("---\r\n")
    }
}

fn parse_header(lines: &[&str]) -> HashMap<String, FrontmatterValue> {
    let mut fields: HashMap<String, FrontmatterValue> = HashMap::new();
    let mut last_key: Option<String> = None;
    for line in lines {
        // Skip comments and blanks.
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Only consume top-level `key: value` pairs (no leading
        // whitespace). Indented lines belong to nested structures we
        // don't expand; preserve them as raw values on the nearest
        // unindented parent key.
        let indent = line.len() - line.trim_start().len();
        if indent > 0 {
            if let Some(key) = last_key.as_ref() {
                if let Some(FrontmatterValue::Raw(existing)) = fields.get_mut(key) {
                    if !existing.is_empty() {
                        existing.push('\n');
                    }
                    existing.push_str(line);
                }
            }
            continue;
        }
        let Some(colon) = line.find(':') else {
            continue;
        };
        let key = line[..colon].trim().to_string();
        let value_src = line[colon + 1..].trim();
        if value_src.is_empty() {
            // Key with an empty value — could be the start of a nested
            // value. Store as empty `Raw` so subsequent indented lines
            // append there.
            fields.insert(key.clone(), FrontmatterValue::Raw(String::new()));
        } else {
            fields.insert(key.clone(), parse_scalar(value_src));
        }
        last_key = Some(key);
    }
    fields
}

fn parse_scalar(src: &str) -> FrontmatterValue {
    // Quoted string.
    if let Some(stripped) = src
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
    {
        return FrontmatterValue::String(unescape_double_quoted(stripped));
    }
    if let Some(stripped) = src
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
    {
        return FrontmatterValue::String(stripped.replace("''", "'"));
    }
    // Bool.
    match src.to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" => return FrontmatterValue::Bool(true),
        "false" | "no" | "off" => return FrontmatterValue::Bool(false),
        _ => {}
    }
    // Int / Float.
    if let Ok(i) = src.parse::<i64>() {
        return FrontmatterValue::Int(i);
    }
    if let Ok(f) = src.parse::<f64>() {
        return FrontmatterValue::Float(f);
    }
    // Bare / unquoted string. Strip inline trailing comment.
    let cleaned = if let Some(idx) = src.find(" #") {
        src[..idx].trim_end()
    } else {
        src
    };
    FrontmatterValue::String(cleaned.to_string())
}

fn unescape_double_quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Return the body of `input` with any `---\n…\n---` preamble
/// stripped. Useful when feeding skill bodies into a message without
/// caring about the metadata.
pub fn strip_frontmatter(input: &str) -> String {
    parse_frontmatter(input).body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_string_fields() {
        let src = "---\nname: hello\ndescription: Say hi\n---\n# Body\n";
        let p = parse_frontmatter(src);
        assert_eq!(p.string("name"), Some("hello"));
        assert_eq!(p.string("description"), Some("Say hi"));
        assert_eq!(p.body.trim(), "# Body");
    }

    #[test]
    fn parses_quoted_strings_with_colons() {
        let src = "---\ndescription: \"Uses: colons, commas.\"\n---\nbody";
        let p = parse_frontmatter(src);
        assert_eq!(p.string("description"), Some("Uses: colons, commas."));
    }

    #[test]
    fn parses_boolean_and_int() {
        let src = "---\nflag: true\nage: 42\n---\n";
        let p = parse_frontmatter(src);
        assert_eq!(p.bool("flag"), Some(true));
        match p.fields.get("age") {
            Some(FrontmatterValue::Int(42)) => {}
            other => panic!("expected Int(42), got {other:?}"),
        }
    }

    #[test]
    fn handles_crlf_fence() {
        let src = "---\r\nname: hi\r\n---\r\nbody\r\n";
        let p = parse_frontmatter(src);
        assert_eq!(p.string("name"), Some("hi"));
        assert!(p.body.contains("body"));
    }

    #[test]
    fn no_fence_returns_whole_body() {
        let src = "# Just markdown\nno frontmatter\n";
        let p = parse_frontmatter(src);
        assert!(p.fields.is_empty());
        assert_eq!(p.body, src);
    }

    #[test]
    fn missing_closing_fence_returns_whole_body() {
        let src = "---\nname: hi\nbody\n";
        let p = parse_frontmatter(src);
        assert!(p.fields.is_empty());
        assert_eq!(p.body, src);
    }

    #[test]
    fn strip_frontmatter_returns_body_only() {
        let src = "---\nname: hi\n---\nhello world\n";
        assert_eq!(strip_frontmatter(src), "hello world\n");
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        let src = "---\n# comment line\n\nname: hi\n\n# trailing\n---\nbody";
        let p = parse_frontmatter(src);
        assert_eq!(p.string("name"), Some("hi"));
    }

    #[test]
    fn trailing_inline_comment_stripped_on_bare_values() {
        let src = "---\nname: hello # this is me\n---\n";
        let p = parse_frontmatter(src);
        assert_eq!(p.string("name"), Some("hello"));
    }

    #[test]
    fn handles_disable_model_invocation_alias() {
        let src = "---\nname: x\ndescription: y\ndisable-model-invocation: true\n---\n";
        let p = parse_frontmatter(src);
        assert_eq!(p.bool("disable-model-invocation"), Some(true));
    }

    #[test]
    fn nested_indented_value_captured_as_raw() {
        let src = "---\nmetadata:\n  author: alice\n  team: core\nname: x\n---\n";
        let p = parse_frontmatter(src);
        assert_eq!(p.string("name"), Some("x"));
        match p.fields.get("metadata") {
            Some(FrontmatterValue::Raw(s)) => assert!(s.contains("author: alice")),
            other => panic!("expected Raw metadata, got {other:?}"),
        }
    }
}
