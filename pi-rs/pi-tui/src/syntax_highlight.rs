//! Syntax highlighting for markdown code blocks
//! Implements Phase 3.13
//! NOTE: Simplified for now - just provides helper functions for styling code

use ratatui::style::{Color, Modifier, Style};

/// Detect language from code fence
pub fn detect_language(fence: &str) -> &'static str {
    let fence_lower = fence.to_lowercase();
    if fence_lower.contains("rust") {
        "rust"
    } else if fence_lower.contains("python") {
        "python"
    } else if fence_lower.contains("json") {
        "json"
    } else if fence_lower.contains("javascript") || fence_lower.contains("js") {
        "javascript"
    } else if fence_lower.contains("bash") || fence_lower.contains("sh") {
        "bash"
    } else {
        "text"
    }
}

/// Get color style for a keyword (3.13)
pub fn keyword_style() -> Style {
    Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD)
}

/// Get color style for strings (3.13)
pub fn string_style() -> Style {
    Style::default().fg(Color::Green)
}

/// Get color style for comments (3.13)
pub fn comment_style() -> Style {
    Style::default().fg(Color::Gray)
}

/// Get color style for commands/builtins (3.13)
pub fn command_style() -> Style {
    Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD)
}

/// Get color style for variables (3.13)
pub fn variable_style() -> Style {
    Style::default().fg(Color::Cyan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("rust"), "rust");
        assert_eq!(detect_language("python"), "python");
        assert_eq!(detect_language("bash"), "bash");
        assert_eq!(detect_language("json"), "json");
    }

    #[test]
    fn test_keyword_style() {
        let style = keyword_style();
        assert_eq!(style.fg, Some(Color::Magenta));
    }

    #[test]
    fn test_string_style() {
        let style = string_style();
        assert_eq!(style.fg, Some(Color::Green));
    }
}
