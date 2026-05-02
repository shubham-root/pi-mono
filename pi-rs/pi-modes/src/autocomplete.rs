//! Autocomplete helpers for the interactive editor.
//!
//! Triggers:
//!   * `@<frag>` — file-path completion, subsequence match against cwd
//!   * `!<frag>` — bash command completion, subsequence match against PATH
//!
//! The intent is that the user types `@rea<tab>` and the editor
//! replaces the `@rea` fragment with the selected file path (e.g.
//! `@pi-core/src/agent.rs`). The completion machinery here is pure
//! data — it scans the filesystem / PATH and returns sorted
//! suggestions. The interactive renderer handles wiring the dropdown
//! and replacing text in the editor buffer.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Trigger character recognized at the start of a completion fragment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TriggerKind {
    File,
    Bash,
}

impl TriggerKind {
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '@' => Some(Self::File),
            '!' => Some(Self::Bash),
            _ => None,
        }
    }
}

/// A single completion suggestion plus the text that should be
/// inserted into the editor when the user accepts it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Suggestion {
    /// Display string shown in the dropdown.
    pub label: String,
    /// Text to splice into the editor in place of the current fragment
    /// (with the trigger prefix, e.g. `@pi-core/src/agent.rs`).
    pub insert: String,
}

/// Look up to `limit` completions for `fragment` under `cwd`, where
/// `fragment` is the text after the trigger character (without the
/// `@` prefix). Matches directories get a trailing `/` so tab
/// accepts-then-keeps-completing.
pub fn scan_file_completions(fragment: &str, cwd: &Path, limit: usize) -> Vec<Suggestion> {
    let (dir, leaf) = split_dir_leaf(fragment);
    let search_dir = if dir.is_empty() {
        cwd.to_path_buf()
    } else if Path::new(dir).is_absolute() {
        PathBuf::from(dir)
    } else {
        cwd.join(dir)
    };

    let mut out: Vec<Suggestion> = Vec::new();
    let Ok(reader) = std::fs::read_dir(&search_dir) else {
        return out;
    };

    let leaf_lower = leaf.to_lowercase();
    let mut entries: Vec<(String, bool)> = Vec::new();
    for entry in reader.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        // Hide dotfiles unless the user is explicitly completing one.
        if name.starts_with('.') && !leaf.starts_with('.') {
            continue;
        }
        if !name.to_lowercase().contains(&leaf_lower) && !is_subsequence(&leaf_lower, &name.to_lowercase()) {
            continue;
        }
        let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
        entries.push((name, is_dir));
    }

    entries.sort_by(|a, b| {
        // Directories first, then prefix matches, then case-insensitive name.
        let a_prefix = a.0.to_lowercase().starts_with(&leaf_lower);
        let b_prefix = b.0.to_lowercase().starts_with(&leaf_lower);
        b.1.cmp(&a.1)
            .then(b_prefix.cmp(&a_prefix))
            .then(a.0.to_lowercase().cmp(&b.0.to_lowercase()))
    });

    for (name, is_dir) in entries.into_iter().take(limit) {
        let label = if is_dir {
            format!("{name}/")
        } else {
            name.clone()
        };
        let insert = if dir.is_empty() {
            format!("@{label}")
        } else {
            format!("@{dir}{sep}{label}", sep = if dir.ends_with('/') { "" } else { "/" })
        };
        out.push(Suggestion { label, insert });
    }
    out
}

/// Scan `$PATH` for executables whose name contains the `fragment`
/// substring, case-insensitive. Returns up to `limit` results.
pub fn scan_bash_completions(fragment: &str, limit: usize) -> Vec<Suggestion> {
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    let mut seen: HashSet<String> = HashSet::new();
    let frag_lower = fragment.to_lowercase();
    let mut hits: Vec<String> = Vec::new();
    for dir in std::env::split_paths(&path_var) {
        let Ok(reader) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in reader.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.to_lowercase().contains(&frag_lower) {
                continue;
            }
            if !is_executable(&entry.path()) {
                continue;
            }
            if seen.insert(name.clone()) {
                hits.push(name);
            }
        }
    }
    hits.sort_by(|a, b| {
        let a_pref = a.to_lowercase().starts_with(&frag_lower);
        let b_pref = b.to_lowercase().starts_with(&frag_lower);
        b_pref.cmp(&a_pref).then(a.to_lowercase().cmp(&b.to_lowercase()))
    });
    hits.into_iter()
        .take(limit)
        .map(|n| Suggestion { label: n.clone(), insert: format!("!{n}") })
        .collect()
}

fn split_dir_leaf(frag: &str) -> (&str, &str) {
    match frag.rfind('/') {
        Some(i) => (&frag[..=i], &frag[i + 1..]),
        None => ("", frag),
    }
}

fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut it = needle.chars().peekable();
    for c in haystack.chars() {
        if it.peek() == Some(&c) {
            it.next();
        }
        if it.peek().is_none() {
            return true;
        }
    }
    it.peek().is_none()
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    std::fs::metadata(path).map(|m| m.is_file()).unwrap_or(false)
}

/// Examine the editor text at the given cursor offset and detect
/// whether the cursor sits inside a completion fragment. Returns the
/// (trigger kind, fragment-start byte offset, fragment text).
///
/// The trigger must be preceded by whitespace or start-of-buffer so
/// typing an email address like `you@example.com` does not accidentally
/// fire the file completer on every character.
pub fn detect_trigger(text: &str, cursor: usize) -> Option<(TriggerKind, usize, &str)> {
    let up_to_cursor = &text[..cursor];
    // Walk backward from cursor to find a '@' / '!' preceded by a
    // word boundary, and with only fragment-legal characters in
    // between.
    let mut idx = cursor;
    let bytes = up_to_cursor.as_bytes();
    while idx > 0 {
        let prev_idx = prev_byte(bytes, idx);
        let c = up_to_cursor[prev_idx..].chars().next()?;
        match c {
            '@' | '!' => {
                let kind = TriggerKind::from_char(c).unwrap();
                // Must be at the start of the buffer or preceded by
                // whitespace.
                let is_word_boundary = prev_idx == 0
                    || up_to_cursor[..prev_idx]
                        .chars()
                        .next_back()
                        .map(|p| p.is_whitespace())
                        .unwrap_or(true);
                if !is_word_boundary {
                    return None;
                }
                let frag_start = prev_idx + 1;
                return Some((kind, prev_idx, &text[frag_start..cursor]));
            }
            c if c.is_whitespace() => return None,
            // Fragment characters: allow typical filename chars.
            c if is_fragment_char(c) => {
                idx = prev_idx;
            }
            _ => return None,
        }
    }
    None
}

fn prev_byte(bytes: &[u8], idx: usize) -> usize {
    let mut i = idx.saturating_sub(1);
    while i > 0 && (bytes[i] & 0b1100_0000) == 0b1000_0000 {
        i -= 1;
    }
    i
}

fn is_fragment_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | '+' | '~' | '*')
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn trigger_fires_at_start_of_buffer() {
        let r = detect_trigger("@rea", 4);
        assert_eq!(r, Some((TriggerKind::File, 0, "rea")));
    }

    #[test]
    fn trigger_fires_after_whitespace() {
        let r = detect_trigger("hello @rea", 10);
        assert_eq!(r, Some((TriggerKind::File, 6, "rea")));
    }

    #[test]
    fn trigger_does_not_fire_inside_email() {
        assert!(detect_trigger("you@example", 11).is_none());
    }

    #[test]
    fn bang_triggers_bash_mode() {
        let r = detect_trigger("!ls", 3);
        assert_eq!(r, Some((TriggerKind::Bash, 0, "ls")));
    }

    #[test]
    fn file_scan_finds_entries_in_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.txt"), "x").unwrap();
        std::fs::write(dir.path().join("beta.rs"), "x").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let out = scan_file_completions("", dir.path(), 10);
        let labels: Vec<&str> = out.iter().map(|s| s.label.as_str()).collect();
        assert!(labels.contains(&"alpha.txt"));
        assert!(labels.contains(&"beta.rs"));
        assert!(labels.contains(&"sub/"));
        // Directory comes first in the sort.
        assert_eq!(labels[0], "sub/");
    }

    #[test]
    fn file_scan_filters_by_fragment() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.txt"), "x").unwrap();
        std::fs::write(dir.path().join("beta.rs"), "x").unwrap();
        let out = scan_file_completions("be", dir.path(), 10);
        let labels: Vec<&str> = out.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, vec!["beta.rs"]);
    }

    #[test]
    fn file_scan_descends_subdir() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "x").unwrap();
        let out = scan_file_completions("src/", dir.path(), 10);
        assert!(out.iter().any(|s| s.label == "main.rs"));
        assert!(out.iter().any(|s| s.insert == "@src/main.rs"));
    }
}
