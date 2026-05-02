//! Markdown renderer for the TUI. Uses `pulldown-cmark` for parsing and
//! `syntect` for syntax-highlighted fenced code blocks. Emits
//! `Vec<Line<'static>>` styled to match the TypeScript pi `dark.json`
//! theme.
//!
//! Supported constructs:
//!
//!   headings (#, ##, ...)   bold / italic
//!   inline code              fenced code blocks with syntax highlight
//!   ordered + unordered lists (nested)
//!   block quotes             tables (GFM)
//!   links [text](url)        horizontal rules
//!   hard/soft line breaks
//!
//! Not a full CommonMark renderer: we do not paginate images, we do not
//! wrap long lines to a column width (ratatui's Paragraph handles wrap),
//! and inline HTML is rendered as literal text.

use once_cell::sync::Lazy;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::prelude::{Color, Line, Modifier, Span, Style};
use syntect::easy::HighlightLines;
use syntect::highlighting::{
    Color as SynColor, FontStyle as SynFontStyle, Style as SynStyle, Theme, ThemeSet,
};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

// ---------------------------------------------------------------------
// syntect bootstrap (one-time, process-wide)
// ---------------------------------------------------------------------

static SYNTAX_SET: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);
static THEME: Lazy<Theme> = Lazy::new(|| {
    let ts = ThemeSet::load_defaults();
    ts.themes
        .get("base16-ocean.dark")
        .cloned()
        .unwrap_or_else(|| ts.themes.values().next().cloned().expect("at least one theme"))
});

// ---------------------------------------------------------------------
// theme styles (match TS dark.json)
// ---------------------------------------------------------------------

fn heading_style(level: HeadingLevel) -> Style {
    let base = Style::default().add_modifier(Modifier::BOLD);
    match level {
        HeadingLevel::H1 => base.fg(Color::Yellow),
        HeadingLevel::H2 => base.fg(Color::Cyan),
        _ => base.fg(Color::LightCyan),
    }
}
fn inline_code_style() -> Style {
    Style::default().fg(Color::Cyan)
}
fn code_block_border_style() -> Style {
    Style::default().fg(Color::DarkGray)
}
fn code_block_fallback_style() -> Style {
    Style::default().fg(Color::Green)
}
fn quote_bar_style() -> Style {
    Style::default().fg(Color::DarkGray)
}
fn quote_text_style() -> Style {
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
fn strike_style() -> Style {
    Style::default().add_modifier(Modifier::CROSSED_OUT)
}
fn table_border_style() -> Style {
    Style::default().fg(Color::DarkGray)
}
fn table_header_style() -> Style {
    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
}

// ---------------------------------------------------------------------
// public entry points
// ---------------------------------------------------------------------

/// Render a markdown document to styled ratatui lines.
pub fn render_markdown(input: &str) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(input, opts);
    let mut r = Renderer::default();
    for ev in parser {
        r.handle(ev);
    }
    r.finish()
}

/// Render a short inline markdown snippet (no block-level constructs)
/// to styled spans. Used by list items, table cells, and callers that
/// want inline emphasis only.
pub fn render_inline(input: &str) -> Vec<Span<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    let parser = Parser::new_ext(input, opts);
    let mut r = Renderer::default();
    for ev in parser {
        r.handle(ev);
    }
    // Flush any dangling inline buffer into a single line and return
    // its spans.
    r.finalize_line();
    r.lines
        .into_iter()
        .next()
        .map(|l| l.spans)
        .unwrap_or_default()
}

// ---------------------------------------------------------------------
// renderer
// ---------------------------------------------------------------------

#[derive(Default)]
struct Renderer {
    lines: Vec<Line<'static>>,
    /// Spans accumulating the current output line.
    current: Vec<Span<'static>>,
    /// Stack of active inline styles. The top is applied to new text.
    style_stack: Vec<Style>,

    /// Fence state.
    in_code_block: bool,
    code_lang: Option<String>,
    code_buf: String,

    /// List nesting stack. Entry is `(ordered_counter_or_unordered, next_idx)`.
    list_stack: Vec<ListState>,

    /// Quote nesting depth.
    quote_depth: usize,

    /// Link target tracked so the renderer can emit `(url)` after the
    /// link text closes.
    link_url: Option<String>,

    /// Table state.
    in_table_head: bool,
    table_buf: Vec<Vec<Vec<Span<'static>>>>, // rows of cells of spans
    current_row: Option<Vec<Vec<Span<'static>>>>,
    current_cell: Option<Vec<Span<'static>>>,
}

#[derive(Clone)]
struct ListState {
    ordered: Option<u64>,
}

impl Renderer {
    fn current_style(&self) -> Style {
        self.style_stack
            .last()
            .copied()
            .unwrap_or_else(Style::default)
    }

    fn push_style(&mut self, s: Style) {
        let base = self.current_style();
        self.style_stack.push(patch(base, s));
    }
    fn pop_style(&mut self) {
        self.style_stack.pop();
    }

    fn push_span(&mut self, text: String, style: Style) {
        if text.is_empty() {
            return;
        }
        // If the active cell exists (table), divert spans there.
        if let Some(cell) = self.current_cell.as_mut() {
            cell.push(Span::styled(text, style));
        } else {
            self.current.push(Span::styled(text, style));
        }
    }

    fn push_text(&mut self, text: &str) {
        let style = self.current_style();
        self.push_span(text.to_string(), style);
    }

    fn finalize_line(&mut self) {
        if self.current.is_empty() {
            self.lines.push(Line::from(""));
        } else {
            let spans = std::mem::take(&mut self.current);
            self.lines.push(Line::from(spans));
        }
    }

    fn blank_line(&mut self) {
        // Avoid emitting two consecutive blanks.
        if matches!(self.lines.last(), Some(l) if l.spans.is_empty()) {
            return;
        }
        self.lines.push(Line::from(""));
    }

    fn handle(&mut self, ev: Event<'_>) {
        match ev {
            Event::Start(tag) => self.start(tag),
            Event::End(end) => self.end(end),
            Event::Text(s) => {
                if self.in_code_block {
                    self.code_buf.push_str(&s);
                } else {
                    self.push_text(&s);
                }
            }
            Event::Code(s) => {
                let style = patch(self.current_style(), inline_code_style());
                self.push_span(s.into_string(), style);
            }
            Event::Html(s) | Event::InlineHtml(s) => {
                // Render inline HTML as plain text with dim styling.
                let style = patch(self.current_style(), Style::default().fg(Color::DarkGray));
                self.push_span(s.into_string(), style);
            }
            Event::FootnoteReference(s) => {
                let style = patch(self.current_style(), link_url_style());
                self.push_span(format!("[^{s}]"), style);
            }
            Event::SoftBreak => {
                self.push_text(" ");
            }
            Event::HardBreak => {
                self.finalize_line();
            }
            Event::Rule => {
                self.finalize_line();
                self.lines
                    .push(Line::from(Span::styled("─".repeat(40), hr_style())));
            }
            Event::TaskListMarker(checked) => {
                let mark = if checked { "[x] " } else { "[ ] " };
                let style = patch(self.current_style(), list_bullet_style());
                self.push_span(mark.to_string(), style);
            }
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                let marker = "#".repeat(level as usize);
                let style = heading_style(level);
                self.push_style(style);
                self.push_span(format!("{marker} "), style);
            }
            Tag::BlockQuote => {
                self.quote_depth += 1;
                self.push_style(quote_text_style());
            }
            Tag::CodeBlock(kind) => {
                self.finalize_line();
                self.in_code_block = true;
                self.code_lang = match kind {
                    CodeBlockKind::Fenced(lang) if !lang.is_empty() => Some(lang.into_string()),
                    _ => None,
                };
                self.code_buf.clear();
            }
            Tag::List(first) => {
                self.list_stack.push(ListState { ordered: first });
            }
            Tag::Item => {
                self.finalize_line();
                let indent = "  ".repeat(self.list_stack.len().saturating_sub(1));
                self.push_span(indent, Style::default());
                let marker = if let Some(state) = self.list_stack.last_mut() {
                    if let Some(n) = state.ordered {
                        state.ordered = Some(n + 1);
                        format!("{n}. ")
                    } else {
                        "• ".to_string()
                    }
                } else {
                    "• ".to_string()
                };
                self.push_span(marker, list_bullet_style());
            }
            Tag::Emphasis => self.push_style(italic_style()),
            Tag::Strong => self.push_style(bold_style()),
            Tag::Strikethrough => self.push_style(strike_style()),
            Tag::Link { dest_url, .. } => {
                self.link_url = Some(dest_url.into_string());
                self.push_style(link_style());
            }
            Tag::Image { dest_url, title, .. } => {
                // We can't display images; fall back to a labeled link.
                let label = if !title.is_empty() {
                    title.into_string()
                } else {
                    "image".to_string()
                };
                self.push_span(format!("[{label}]"), link_style());
                self.push_span(format!(" ({dest_url})"), link_url_style());
            }
            Tag::Table(_) => {
                self.finalize_line();
                self.table_buf.clear();
            }
            Tag::TableHead => {
                self.in_table_head = true;
                self.current_row = Some(Vec::new());
            }
            Tag::TableRow => {
                self.current_row = Some(Vec::new());
            }
            Tag::TableCell => {
                self.current_cell = Some(Vec::new());
                if self.in_table_head {
                    self.push_style(table_header_style());
                }
            }
            Tag::FootnoteDefinition(_) => {}
            Tag::MetadataBlock(_) => {}
            Tag::HtmlBlock => {}
        }
    }

    fn end(&mut self, end: TagEnd) {
        match end {
            TagEnd::Paragraph => {
                self.finalize_line();
                self.blank_line();
            }
            TagEnd::Heading(_) => {
                self.pop_style();
                self.finalize_line();
                self.blank_line();
            }
            TagEnd::BlockQuote => {
                self.pop_style();
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.blank_line();
            }
            TagEnd::CodeBlock => {
                self.emit_code_block();
                self.in_code_block = false;
                self.code_lang = None;
                self.code_buf.clear();
                self.blank_line();
            }
            TagEnd::List(_) => {
                self.list_stack.pop();
                if self.list_stack.is_empty() {
                    self.finalize_line();
                    self.blank_line();
                } else {
                    self.finalize_line();
                }
            }
            TagEnd::Item => {
                self.finalize_line();
            }
            TagEnd::Emphasis => self.pop_style(),
            TagEnd::Strong => self.pop_style(),
            TagEnd::Strikethrough => self.pop_style(),
            TagEnd::Link => {
                self.pop_style();
                if let Some(url) = self.link_url.take() {
                    if !url.is_empty() {
                        self.push_span(format!(" ({url})"), link_url_style());
                    }
                }
            }
            TagEnd::Image => {}
            TagEnd::Table => {
                self.emit_table();
                self.blank_line();
            }
            TagEnd::TableHead => {
                if let Some(row) = self.current_row.take() {
                    self.table_buf.push(row);
                }
                self.in_table_head = false;
            }
            TagEnd::TableRow => {
                if let Some(row) = self.current_row.take() {
                    self.table_buf.push(row);
                }
            }
            TagEnd::TableCell => {
                if self.in_table_head {
                    self.pop_style();
                }
                if let Some(cell) = self.current_cell.take() {
                    if let Some(row) = self.current_row.as_mut() {
                        row.push(cell);
                    }
                }
            }
            TagEnd::FootnoteDefinition
            | TagEnd::MetadataBlock(_)
            | TagEnd::HtmlBlock => {}
        }
    }

    fn emit_code_block(&mut self) {
        let label = self.code_lang.clone().unwrap_or_else(|| "code".to_string());
        self.lines.push(Line::from(vec![
            Span::styled("┌─ ".to_string(), code_block_border_style()),
            Span::styled(label.clone(), code_block_border_style().add_modifier(Modifier::BOLD)),
        ]));

        let body = std::mem::take(&mut self.code_buf);
        let syntax = SYNTAX_SET
            .find_syntax_by_token(&label)
            .or_else(|| SYNTAX_SET.find_syntax_by_extension(&label))
            .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());
        let mut hl = HighlightLines::new(syntax, &THEME);

        for line in LinesWithEndings::from(&body) {
            let highlighted: Result<Vec<(SynStyle, &str)>, _> =
                hl.highlight_line(line, &SYNTAX_SET);
            let mut spans: Vec<Span<'static>> = vec![Span::styled(
                "│ ".to_string(),
                code_block_border_style(),
            )];
            match highlighted {
                Ok(parts) => {
                    for (style, text) in parts {
                        let text = text.trim_end_matches('\n');
                        if text.is_empty() {
                            continue;
                        }
                        spans.push(Span::styled(text.to_string(), syn_to_ratatui(style)));
                    }
                }
                Err(_) => {
                    spans.push(Span::styled(
                        line.trim_end_matches('\n').to_string(),
                        code_block_fallback_style(),
                    ));
                }
            }
            self.lines.push(Line::from(spans));
        }

        self.lines
            .push(Line::from(Span::styled("└".repeat(1), code_block_border_style())));
    }

    fn emit_table(&mut self) {
        if self.table_buf.is_empty() {
            return;
        }
        let col_count = self.table_buf.iter().map(|r| r.len()).max().unwrap_or(0);
        if col_count == 0 {
            return;
        }
        // Compute width per column.
        let mut widths = vec![0usize; col_count];
        for row in &self.table_buf {
            for (ci, cell) in row.iter().enumerate() {
                let w: usize = cell.iter().map(|s| s.content.chars().count()).sum();
                if w > widths[ci] {
                    widths[ci] = w;
                }
            }
        }
        // Emit header row, divider, body rows.
        let rows_snapshot = std::mem::take(&mut self.table_buf);
        let mut it = rows_snapshot.into_iter();
        if let Some(header) = it.next() {
            self.lines.push(self.render_table_row(&header, &widths));
            self.lines.push(Line::from(Span::styled(
                widths
                    .iter()
                    .map(|w| "─".repeat(w + 2))
                    .collect::<Vec<_>>()
                    .join("┼"),
                table_border_style(),
            )));
        }
        for row in it {
            self.lines.push(self.render_table_row(&row, &widths));
        }
    }

    fn render_table_row(&self, row: &[Vec<Span<'static>>], widths: &[usize]) -> Line<'static> {
        let mut out: Vec<Span<'static>> = Vec::new();
        for (ci, w) in widths.iter().enumerate() {
            if ci > 0 {
                out.push(Span::styled(" │ ".to_string(), table_border_style()));
            } else {
                out.push(Span::styled(" ".to_string(), table_border_style()));
            }
            let empty: Vec<Span<'static>> = Vec::new();
            let cell = row.get(ci).unwrap_or(&empty);
            let mut used = 0usize;
            for s in cell {
                out.push(s.clone());
                used += s.content.chars().count();
            }
            if used < *w {
                out.push(Span::raw(" ".repeat(w - used)));
            }
        }
        Line::from(out)
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.finalize_line();
        // Strip a single trailing blank if present for tighter visuals.
        while matches!(self.lines.last(), Some(l) if l.spans.is_empty()) {
            self.lines.pop();
        }
        self.lines
    }
}

fn patch(base: Style, delta: Style) -> Style {
    let mut out = base;
    if let Some(fg) = delta.fg {
        out = out.fg(fg);
    }
    if let Some(bg) = delta.bg {
        out = out.bg(bg);
    }
    out.add_modifier |= delta.add_modifier;
    out.sub_modifier |= delta.sub_modifier;
    out
}

fn syn_to_ratatui(s: SynStyle) -> Style {
    let fg = syn_color_to_ratatui(s.foreground);
    let mut style = Style::default().fg(fg);
    if s.font_style.contains(SynFontStyle::BOLD) {
        style = style.add_modifier(Modifier::BOLD);
    }
    if s.font_style.contains(SynFontStyle::ITALIC) {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if s.font_style.contains(SynFontStyle::UNDERLINE) {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    style
}

fn syn_color_to_ratatui(c: SynColor) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}

// ---------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn line_text(l: &Line<'_>) -> String {
        l.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn renders_heading_and_paragraph() {
        let out = render_markdown("# Title\n\nHello **world**.");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("# Title"), "{text}");
        assert!(text.contains("world"), "{text}");
    }

    #[test]
    fn renders_unordered_list() {
        let out = render_markdown("- one\n- two\n- three");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("• one"), "{text}");
        assert!(text.contains("• three"), "{text}");
    }

    #[test]
    fn renders_ordered_list() {
        let out = render_markdown("1. first\n2. second");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("1. first"), "{text}");
        assert!(text.contains("2. second"), "{text}");
    }

    #[test]
    fn renders_fenced_block_with_label() {
        let out = render_markdown("```rust\nfn main() {}\n```");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("rust"), "{text}");
        assert!(text.contains("fn main()"), "{text}");
    }

    #[test]
    fn fenced_block_without_lang_gets_code_label() {
        let out = render_markdown("```\njust text\n```");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("code"), "{text}");
        assert!(text.contains("just text"), "{text}");
    }

    #[test]
    fn renders_link_with_url() {
        let out = render_markdown("see [pi](https://pi.dev)");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("pi"), "{text}");
        assert!(text.contains("pi.dev"), "{text}");
    }

    #[test]
    fn renders_horizontal_rule() {
        let out = render_markdown("before\n\n---\n\nafter");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("─"), "{text}");
    }

    #[test]
    fn renders_table() {
        let src = "| a | b |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |";
        let out = render_markdown(src);
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("a"), "{text}");
        assert!(text.contains("1"), "{text}");
        assert!(text.contains("│"), "{text}");
    }

    #[test]
    fn renders_quote() {
        let out = render_markdown("> quoted line");
        let text: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("quoted line"), "{text}");
    }

    #[test]
    fn inline_renders_bold_italic_code() {
        let spans = render_inline("plain **b** *i* `c`");
        let has_bold = spans
            .iter()
            .any(|s| s.content == "b" && s.style.add_modifier.contains(Modifier::BOLD));
        let has_italic = spans
            .iter()
            .any(|s| s.content == "i" && s.style.add_modifier.contains(Modifier::ITALIC));
        let has_code = spans.iter().any(|s| s.content == "c");
        assert!(has_bold && has_italic && has_code, "spans: {spans:?}");
    }

    #[test]
    fn strikethrough_applies_modifier() {
        let spans = render_inline("plain ~~strike~~ end");
        let has_strike = spans.iter().any(|s| {
            s.content == "strike" && s.style.add_modifier.contains(Modifier::CROSSED_OUT)
        });
        assert!(has_strike, "spans: {spans:?}");
    }
}
