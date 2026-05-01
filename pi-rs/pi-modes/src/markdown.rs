//! Minimal markdown renderer for the TUI. Produces `Vec<Line<'static>>`
//! with colors drawn from the TypeScript `dark.json` theme so the Rust
//! TUI matches visually.
//!
//! Supported subset (covers LLM responses in practice):
//!
//!   # heading (levels 1..=6)
//!   **bold**
//!   *italic* / _italic_
//!   `inline code`
//!   ```lang ... ``` fenced blocks
//!   - / * / + / 1. lists
//!   > block quote
//!   [text](url) link
//!   ---/***/___ horizontal rule
//!
//! Intentionally not a full CommonMark implementation; goal is readable
//! chat output, not spec compliance.

use ratatui::prelude::{Color, Line, Modifier, Span, Style};

fn heading_style() -> Style {
    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
}
fn code_style() -> Style {
    Style::default().fg(Color::Cyan)
}
fn code_block_style() -> Style {
    Style::default().fg(Color::Green)
}
fn code_block_border_style() -> Style {
    Style::default().fg(Color::DarkGray)
}
fn quote_style() -> Style {
    Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC)
}
fn list_bullet_style() -> Style {
    Style::default().fg(Color::Cyan)
}
fn link_style() -> Style {
    Style::default().fg(Color::Blue).add_modifier(Modifier::UNDERLINED)
}
fn link_url_style() -> Style {
    Style::default().fg(Color::DarkGray)
}
fn hr_style() -> Style {
    Style::default().fg(Color::DarkGray)
}
fn bold_style() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}
fn italic_style() -> Style {
    Style::default().add_modifier(Modifier::ITALIC)
}

/// Render markdown text into styled lines for a ratatui Paragraph.
pub fn render_markdown(input: &str) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut in_fence: Option<String> = None;
    let mut fence_buf: Vec<String> = Vec::new();

    for raw in input.split('\n') {
        let line = raw.trim_end_matches('\r');

        if let Some(lang) = in_fence.as_ref() {
            if line.trim_start().starts_with("```") {
                out.extend(render_code_block(lang, &fence_buf));
                fence_buf.clear();
                in_fence = None;
                continue;
            }
            fence_buf.push(line.to_string());
            continue;
        }
        if let Some(rest) = line.trim_start().strip_prefix("```") {
            in_fence = Some(rest.trim().to_string());
            continue;
        }

        if let Some((level, text)) = parse_heading(line) {
            let marker = "#".repeat(level);
            out.push(Line::from(vec![
                Span::styled(format!("{marker} "), heading_style()),
                Span::styled(text.to_string(), heading_style()),
            ]));
            continue;
        }

        let t = line.trim();
        if t == "---" || t == "***" || t == "___" {
            out.push(Line::from(Span::styled("─".repeat(40), hr_style())));
            continue;
        }

        if let Some(rest) = line.trim_start().strip_prefix("> ") {
            out.push(Line::from(vec![
                Span::styled("│ ".to_string(), Style::default().fg(Color::DarkGray)),
                Span::styled(rest.to_string(), quote_style()),
            ]));
            continue;
        }

        if let Some(rest) = list_prefix_unordered(line) {
            let mut spans: Vec<Span<'static>> = Vec::new();
            spans.push(Span::styled("• ".to_string(), list_bullet_style()));
            spans.extend(render_inline(rest));
            out.push(Line::from(spans));
            continue;
        }
        if let Some((num, rest)) = list_prefix_ordered(line) {
            let mut spans: Vec<Span<'static>> = Vec::new();
            spans.push(Span::styled(format!("{num}. "), list_bullet_style()));
            spans.extend(render_inline(rest));
            out.push(Line::from(spans));
            continue;
        }

        out.push(Line::from(render_inline(line)));
    }

    if let Some(lang) = in_fence.take() {
        if !fence_buf.is_empty() {
            out.extend(render_code_block(&lang, &fence_buf));
        }
    }

    out
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.starts_with(' ') {
        return None;
    }
    Some((level, rest.trim_start()))
}

fn list_prefix_unordered(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return Some(rest);
        }
    }
    None
}

fn list_prefix_ordered(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    let digit_count = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_count == 0 || digit_count > 4 {
        return None;
    }
    let rest = &trimmed[digit_count..];
    let rest = rest.strip_prefix(". ")?;
    Some((&trimmed[..digit_count], rest))
}

/// Render inline markdown into styled spans. Anything unrecognized is
/// passed through as plain text.
pub fn render_inline(text: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut chars = text.chars().peekable();

    fn push_plain(spans: &mut Vec<Span<'static>>, buf: &mut String) {
        if !buf.is_empty() {
            spans.push(Span::raw(std::mem::take(buf)));
        }
    }

    while let Some(c) = chars.next() {
        match c {
            '`' => {
                push_plain(&mut spans, &mut buf);
                let mut code = String::new();
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == '`' {
                        break;
                    }
                    code.push(next);
                }
                if !code.is_empty() {
                    spans.push(Span::styled(code, code_style()));
                }
            }
            '*' if chars.peek() == Some(&'*') => {
                chars.next();
                push_plain(&mut spans, &mut buf);
                let mut inner = String::new();
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == '*' && chars.peek() == Some(&'*') {
                        chars.next();
                        break;
                    }
                    inner.push(next);
                }
                if !inner.is_empty() {
                    spans.push(Span::styled(inner, bold_style()));
                }
            }
            '*' | '_' => {
                push_plain(&mut spans, &mut buf);
                let close = c;
                let mut inner = String::new();
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == close {
                        break;
                    }
                    inner.push(next);
                }
                if !inner.is_empty() {
                    spans.push(Span::styled(inner, italic_style()));
                }
            }
            '[' => {
                push_plain(&mut spans, &mut buf);
                let mut label = String::new();
                let mut found_close = false;
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == ']' {
                        found_close = true;
                        break;
                    }
                    label.push(next);
                }
                if found_close && chars.peek() == Some(&'(') {
                    chars.next();
                    let mut url = String::new();
                    let mut found_url_close = false;
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next == ')' {
                            found_url_close = true;
                            break;
                        }
                        url.push(next);
                    }
                    if found_url_close {
                        spans.push(Span::styled(label, link_style()));
                        if !url.is_empty() {
                            spans.push(Span::raw(" ".to_string()));
                            spans.push(Span::styled(format!("({url})"), link_url_style()));
                        }
                        continue;
                    }
                    spans.push(Span::raw(format!("[{label}](")));
                    spans.push(Span::raw(url));
                    continue;
                }
                spans.push(Span::raw(format!("[{label}")));
                if found_close {
                    spans.push(Span::raw("]".to_string()));
                }
            }
            _ => buf.push(c),
        }
    }
    push_plain(&mut spans, &mut buf);
    spans
}

fn render_code_block(lang: &str, lines: &[String]) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let label = if lang.is_empty() { "code".to_string() } else { lang.to_string() };
    out.push(Line::from(vec![
        Span::styled("┌─ ".to_string(), code_block_border_style()),
        Span::styled(label, code_block_border_style().add_modifier(Modifier::BOLD)),
    ]));
    for line in lines {
        out.push(Line::from(vec![
            Span::styled("│ ".to_string(), code_block_border_style()),
            Span::styled(line.to_string(), code_block_style()),
        ]));
    }
    out.push(Line::from(Span::styled(
        "└".to_string(),
        code_block_border_style(),
    )));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_text(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn renders_heading() {
        let out = render_markdown("# Hello");
        assert_eq!(out.len(), 1);
        assert_eq!(line_text(&out[0]), "# Hello");
    }

    #[test]
    fn renders_unordered_list() {
        let out = render_markdown("- one\n- two");
        assert_eq!(out.len(), 2);
        assert!(line_text(&out[0]).starts_with("• "));
        assert!(line_text(&out[1]).contains("two"));
    }

    #[test]
    fn renders_fenced_code_block() {
        let out = render_markdown("```rust\nlet x = 1;\n```");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("rust"), "{text}");
        assert!(text.contains("let x = 1;"), "{text}");
    }

    #[test]
    fn renders_bold_italic_code_inline() {
        let spans = render_inline("plain **bold** *italic* `code` end");
        let has_bold = spans.iter().any(|s| {
            s.content == "bold" && s.style.add_modifier.contains(Modifier::BOLD)
        });
        let has_italic = spans.iter().any(|s| {
            s.content == "italic" && s.style.add_modifier.contains(Modifier::ITALIC)
        });
        let has_code = spans.iter().any(|s| s.content == "code");
        assert!(has_bold && has_italic && has_code, "spans: {spans:?}");
    }

    #[test]
    fn renders_link() {
        let spans = render_inline("see [pi](https://pi.dev) please");
        let texts: Vec<&str> = spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(texts.contains(&"pi"));
        assert!(
            texts.iter().any(|t| t.contains("pi.dev")),
            "spans: {texts:?}"
        );
    }

    #[test]
    fn renders_horizontal_rule() {
        let out = render_markdown("before\n---\nafter");
        assert_eq!(out.len(), 3);
        let hr_text = line_text(&out[1]);
        assert!(hr_text.chars().all(|c| c == '─'), "{hr_text}");
    }

    #[test]
    fn unclosed_fence_is_flushed() {
        let out = render_markdown("```rust\nlet x = 1;");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("let x = 1;"), "{text}");
    }
}
