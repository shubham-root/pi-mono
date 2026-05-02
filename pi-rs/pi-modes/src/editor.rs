//! Multi-line input editor with a TypeScript-pi-parity feature set:
//!
//! * character / word / line / buffer cursor movement
//! * insert, backspace, forward-delete (char + word + line)
//! * kill ring (yank)
//! * undo / redo with op-group coalescing (typing is one undo unit;
//!   a subsequent cursor move or kill finalizes the group)
//! * bracketed paste-aware bulk insert
//!
//! Cursor is a byte offset into `text` (which is always valid UTF-8).
//! Rendering helpers expose a (row, col) cursor in *character* columns
//! so the ratatui renderer can position the block cursor.

use std::time::{Duration, Instant};

/// A coalescing tag so repeated typing collapses into one undo unit
/// while any "structural" change (cursor move, kill, paste, newline)
/// closes the group and starts a new one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpGroup {
    Insert,
    Backspace,
    ForwardDelete,
    /// Any "boundary" op (paste, kill, newline, cursor move immediately
    /// after typing). Never coalesces.
    Boundary,
}

#[derive(Clone, Debug)]
struct Snapshot {
    text: String,
    cursor: usize,
}

/// Multi-line input editor. The buffer is a single `String`; newlines
/// are just `\n` characters and the renderer splits on them.
pub struct InputEditor {
    text: String,
    /// Byte offset into `text`. Always on a UTF-8 char boundary.
    cursor: usize,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    /// Op group of the last applied edit, used to decide whether the
    /// next edit can coalesce onto the top of the undo stack.
    last_group: Option<OpGroup>,
    last_edit_at: Option<Instant>,
    kill_ring: Vec<String>,
}

impl InputEditor {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            undo: Vec::new(),
            redo: Vec::new(),
            last_group: None,
            last_edit_at: None,
            kill_ring: Vec::new(),
        }
    }

    // ---------------------------------------------------------------
    // read
    // ---------------------------------------------------------------

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn cursor_byte(&self) -> usize {
        self.cursor
    }

    /// (row, col) in *character* columns, 0-indexed, for rendering.
    pub fn cursor_line_col(&self) -> (usize, usize) {
        let before = &self.text[..self.cursor];
        let row = before.bytes().filter(|b| *b == b'\n').count();
        let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let col = self.text[line_start..self.cursor].chars().count();
        (row, col)
    }

    /// Visual lines split on `\n`, owned so the renderer can attach
    /// styles freely.
    pub fn visual_lines(&self) -> Vec<String> {
        if self.text.is_empty() {
            vec![String::new()]
        } else {
            self.text.split('\n').map(|s| s.to_string()).collect()
        }
    }

    // ---------------------------------------------------------------
    // wholesale state (clear/set)
    // ---------------------------------------------------------------

    pub fn clear(&mut self) {
        self.push_undo(OpGroup::Boundary);
        self.text.clear();
        self.cursor = 0;
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        self.push_undo(OpGroup::Boundary);
        self.text = text.into();
        self.cursor = self.text.len();
    }

    // ---------------------------------------------------------------
    // insertion
    // ---------------------------------------------------------------

    pub fn insert_char(&mut self, c: char) {
        if c == '\n' {
            self.push_undo(OpGroup::Boundary);
        } else {
            self.push_undo(OpGroup::Insert);
        }
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.redo.clear();
    }

    /// Bulk insert — used for paste and autocomplete accepts. Treated
    /// as a single undo group.
    pub fn insert_str(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.push_undo(OpGroup::Boundary);
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.redo.clear();
    }

    pub fn newline(&mut self) {
        self.insert_char('\n');
    }

    // ---------------------------------------------------------------
    // deletion
    // ---------------------------------------------------------------

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.push_undo(OpGroup::Backspace);
        let prev = prev_char_boundary(&self.text, self.cursor);
        self.text.replace_range(prev..self.cursor, "");
        self.cursor = prev;
        self.redo.clear();
    }

    pub fn delete_forward(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        self.push_undo(OpGroup::ForwardDelete);
        let next = next_char_boundary(&self.text, self.cursor);
        self.text.replace_range(self.cursor..next, "");
        self.redo.clear();
    }

    /// Delete the word to the left of the cursor; stored in the kill
    /// ring so Ctrl+Y can paste it back.
    pub fn kill_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.push_undo(OpGroup::Boundary);
        let start = word_start_before(&self.text, self.cursor);
        let removed: String = self.text[start..self.cursor].to_string();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
        self.kill_ring.push(removed);
        self.redo.clear();
    }

    /// Delete the word to the right of the cursor (kill-ring stored).
    pub fn kill_word_forward(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        self.push_undo(OpGroup::Boundary);
        let end = word_end_after(&self.text, self.cursor);
        let removed: String = self.text[self.cursor..end].to_string();
        self.text.replace_range(self.cursor..end, "");
        self.kill_ring.push(removed);
        self.redo.clear();
    }

    /// Delete from cursor to the end of the current line (kill-ring).
    pub fn kill_to_line_end(&mut self) {
        let end = line_end(&self.text, self.cursor);
        if end <= self.cursor {
            // already at EOL — kill the newline itself
            if self.cursor < self.text.len() {
                self.push_undo(OpGroup::Boundary);
                let next = next_char_boundary(&self.text, self.cursor);
                let removed = self.text[self.cursor..next].to_string();
                self.text.replace_range(self.cursor..next, "");
                self.kill_ring.push(removed);
                self.redo.clear();
            }
            return;
        }
        self.push_undo(OpGroup::Boundary);
        let removed = self.text[self.cursor..end].to_string();
        self.text.replace_range(self.cursor..end, "");
        self.kill_ring.push(removed);
        self.redo.clear();
    }

    /// Delete from the start of the current line to the cursor (kill-ring).
    pub fn kill_to_line_start(&mut self) {
        let start = line_start(&self.text, self.cursor);
        if start >= self.cursor {
            return;
        }
        self.push_undo(OpGroup::Boundary);
        let removed = self.text[start..self.cursor].to_string();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
        self.kill_ring.push(removed);
        self.redo.clear();
    }

    /// Paste the most recent kill-ring entry.
    pub fn yank(&mut self) {
        if let Some(last) = self.kill_ring.last().cloned() {
            self.insert_str(&last);
        }
    }

    // ---------------------------------------------------------------
    // cursor movement (do not push undo)
    // ---------------------------------------------------------------

    pub fn move_left(&mut self) {
        self.finalize_group();
        if self.cursor > 0 {
            self.cursor = prev_char_boundary(&self.text, self.cursor);
        }
    }
    pub fn move_right(&mut self) {
        self.finalize_group();
        if self.cursor < self.text.len() {
            self.cursor = next_char_boundary(&self.text, self.cursor);
        }
    }
    pub fn move_word_left(&mut self) {
        self.finalize_group();
        self.cursor = word_start_before(&self.text, self.cursor);
    }
    pub fn move_word_right(&mut self) {
        self.finalize_group();
        self.cursor = word_end_after(&self.text, self.cursor);
    }
    pub fn move_line_start(&mut self) {
        self.finalize_group();
        self.cursor = line_start(&self.text, self.cursor);
    }
    pub fn move_line_end(&mut self) {
        self.finalize_group();
        self.cursor = line_end(&self.text, self.cursor);
    }
    pub fn move_buffer_start(&mut self) {
        self.finalize_group();
        self.cursor = 0;
    }
    pub fn move_buffer_end(&mut self) {
        self.finalize_group();
        self.cursor = self.text.len();
    }

    pub fn move_up(&mut self) {
        self.finalize_group();
        let (row, col) = self.cursor_line_col();
        if row == 0 {
            return;
        }
        self.cursor = position_for(&self.text, row - 1, col);
    }
    pub fn move_down(&mut self) {
        self.finalize_group();
        let (row, col) = self.cursor_line_col();
        let total_rows = self.text.bytes().filter(|b| *b == b'\n').count();
        if row >= total_rows {
            return;
        }
        self.cursor = position_for(&self.text, row + 1, col);
    }

    pub fn is_on_last_line(&self) -> bool {
        let total_rows = self.text.bytes().filter(|b| *b == b'\n').count();
        self.cursor_line_col().0 >= total_rows
    }
    pub fn is_on_first_line(&self) -> bool {
        self.cursor_line_col().0 == 0
    }

    // ---------------------------------------------------------------
    // undo / redo
    // ---------------------------------------------------------------

    pub fn undo(&mut self) {
        if let Some(snap) = self.undo.pop() {
            self.redo.push(Snapshot {
                text: std::mem::take(&mut self.text),
                cursor: self.cursor,
            });
            self.text = snap.text;
            self.cursor = snap.cursor;
            self.last_group = Some(OpGroup::Boundary);
        }
    }

    pub fn redo(&mut self) {
        if let Some(snap) = self.redo.pop() {
            self.undo.push(Snapshot {
                text: std::mem::take(&mut self.text),
                cursor: self.cursor,
            });
            self.text = snap.text;
            self.cursor = snap.cursor;
            self.last_group = Some(OpGroup::Boundary);
        }
    }

    fn finalize_group(&mut self) {
        self.last_group = Some(OpGroup::Boundary);
    }

    fn push_undo(&mut self, group: OpGroup) {
        let now = Instant::now();
        let should_push = match (self.last_group, self.last_edit_at) {
            (Some(prev), Some(at)) if prev == group && group != OpGroup::Boundary => {
                now.duration_since(at) > Duration::from_millis(500)
            }
            (Some(_), _) => true,
            (None, _) => true,
        };
        if should_push {
            self.undo.push(Snapshot {
                text: self.text.clone(),
                cursor: self.cursor,
            });
            // Cap undo history at 200 entries to keep memory bounded.
            if self.undo.len() > 200 {
                let drop_n = self.undo.len() - 200;
                self.undo.drain(0..drop_n);
            }
        }
        self.last_group = Some(group);
        self.last_edit_at = Some(now);
    }
}

impl Default for InputEditor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// text utilities (all byte-offset based on a valid UTF-8 String)
// ---------------------------------------------------------------------

fn prev_char_boundary(text: &str, mut idx: usize) -> usize {
    if idx == 0 {
        return 0;
    }
    idx -= 1;
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn next_char_boundary(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    idx += 1;
    while idx < text.len() && !text.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn word_start_before(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut i = pos;
    // Skip non-word chars directly to the left.
    while i > 0 {
        let p = prev_char_boundary(text, i);
        let c = text[p..].chars().next().unwrap_or(' ');
        if is_word_char(c) {
            break;
        }
        i = p;
    }
    // Skip word chars.
    while i > 0 {
        let p = prev_char_boundary(text, i);
        let c = text[p..].chars().next().unwrap_or(' ');
        if !is_word_char(c) {
            break;
        }
        i = p;
    }
    i
}

fn word_end_after(text: &str, pos: usize) -> usize {
    let len = text.len();
    let mut i = pos;
    // Skip non-word chars to the right.
    while i < len {
        let c = text[i..].chars().next().unwrap_or(' ');
        if is_word_char(c) {
            break;
        }
        i = next_char_boundary(text, i);
    }
    // Skip word chars.
    while i < len {
        let c = text[i..].chars().next().unwrap_or(' ');
        if !is_word_char(c) {
            break;
        }
        i = next_char_boundary(text, i);
    }
    i
}

fn line_start(text: &str, pos: usize) -> usize {
    text[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0)
}

fn line_end(text: &str, pos: usize) -> usize {
    text[pos..]
        .find('\n')
        .map(|i| pos + i)
        .unwrap_or_else(|| text.len())
}

/// Resolve (row, col_in_chars) to a byte offset, clamping col to the
/// target line's character length.
fn position_for(text: &str, target_row: usize, target_col: usize) -> usize {
    let mut row = 0usize;
    let mut line_start_idx = 0usize;
    let mut i = 0usize;
    let bytes = text.as_bytes();
    while row < target_row && i < bytes.len() {
        if bytes[i] == b'\n' {
            row += 1;
            i += 1;
            line_start_idx = i;
        } else {
            i += 1;
        }
    }
    if row < target_row {
        // Target row past the end — clamp to end of buffer.
        return text.len();
    }
    // Now walk chars from line_start_idx up to target_col or newline.
    let line_end_idx = text[line_start_idx..]
        .find('\n')
        .map(|j| line_start_idx + j)
        .unwrap_or_else(|| text.len());
    let mut char_count = 0usize;
    let mut byte_pos = line_start_idx;
    for (offset, _) in text[line_start_idx..line_end_idx].char_indices() {
        if char_count == target_col {
            byte_pos = line_start_idx + offset;
            return byte_pos;
        }
        char_count += 1;
        byte_pos = line_start_idx + offset + 1;
    }
    // col >= line length — go to end of line.
    line_end_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_chars_build_string() {
        let mut e = InputEditor::new();
        for c in "hello".chars() {
            e.insert_char(c);
        }
        assert_eq!(e.text(), "hello");
        assert_eq!(e.cursor_byte(), 5);
    }

    #[test]
    fn backspace_respects_unicode_boundaries() {
        let mut e = InputEditor::new();
        e.insert_char('é');
        assert_eq!(e.text(), "é");
        e.backspace();
        assert_eq!(e.text(), "");
        assert_eq!(e.cursor_byte(), 0);
    }

    #[test]
    fn word_movement_handles_punctuation() {
        let mut e = InputEditor::new();
        e.insert_str("hello world foo.bar");
        assert_eq!(e.cursor_byte(), 19);
        e.move_word_left();
        assert_eq!(&e.text()[e.cursor_byte()..], "bar");
        e.move_word_left();
        assert_eq!(&e.text()[e.cursor_byte()..], "foo.bar");
        e.move_word_left();
        assert_eq!(&e.text()[e.cursor_byte()..], "world foo.bar");
    }

    #[test]
    fn kill_word_back_fills_kill_ring() {
        let mut e = InputEditor::new();
        e.insert_str("hello world");
        e.kill_word_back();
        assert_eq!(e.text(), "hello ");
        e.yank();
        assert_eq!(e.text(), "hello world");
    }

    #[test]
    fn kill_to_line_end_stores_tail() {
        let mut e = InputEditor::new();
        e.insert_str("abc def");
        e.move_line_start();
        e.move_word_right();
        e.kill_to_line_end();
        assert_eq!(e.text(), "abc");
        e.move_buffer_end();
        e.yank();
        assert_eq!(e.text(), "abc def");
    }

    #[test]
    fn newline_navigates_up_down() {
        let mut e = InputEditor::new();
        e.insert_str("aaa\nbb\nc");
        // cursor at end
        e.move_up();
        let (row, col) = e.cursor_line_col();
        assert_eq!((row, col), (1, 1));
        e.move_up();
        let (row, col) = e.cursor_line_col();
        // col clamps to line length (1) — original was 1 on 'bb' so still 1
        assert_eq!((row, col), (0, 1));
    }

    #[test]
    fn undo_redo_roundtrip() {
        let mut e = InputEditor::new();
        e.insert_str("hello");
        e.move_left();
        e.insert_char('X');
        assert_eq!(e.text(), "hellXo");
        e.undo();
        assert_eq!(e.text(), "hello");
        e.redo();
        assert_eq!(e.text(), "hellXo");
    }

    #[test]
    fn typing_coalesces_into_one_undo_step() {
        let mut e = InputEditor::new();
        for c in "hello".chars() {
            e.insert_char(c);
        }
        e.undo();
        assert_eq!(e.text(), "");
    }

    #[test]
    fn cursor_move_closes_undo_group() {
        let mut e = InputEditor::new();
        e.insert_str("hello");
        e.move_left();
        e.insert_char('X');
        e.undo();
        assert_eq!(e.text(), "hello");
        e.undo();
        assert_eq!(e.text(), "");
    }

    #[test]
    fn position_for_clamps_col_past_eol() {
        // `a` line has len 1; navigating up from row 1 col 3 should clamp
        // to col 1 on row 0.
        let text = "a\nbbb";
        let pos = position_for(text, 0, 3);
        assert_eq!(pos, 1);
    }
}
