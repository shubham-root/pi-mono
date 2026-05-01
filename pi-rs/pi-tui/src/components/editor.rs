//! Editor component - multiline text input with history
//! Implements Phase 3.3 - interactive text editing

use super::Component;
use crate::input::KeyCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

pub struct Editor {
    lines: Vec<String>,
    cursor_row: usize,
    cursor_col: usize,
    history: Vec<String>,
    history_index: Option<usize>,
}

impl Editor {
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
            history: Vec::new(),
            history_index: None,
        }
    }

    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub fn set_text(&mut self, text: String) {
        self.lines = text.lines().map(|s| s.to_string()).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.cursor_row = self.lines.len() - 1;
        self.cursor_col = self.lines[self.cursor_row].len();
    }

    pub fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.cursor_row = 0;
        self.cursor_col = 0;
    }

    pub fn add_to_history(&mut self) {
        let text = self.text();
        if !text.is_empty() {
            self.history.push(text);
        }
        self.history_index = None;
    }

    fn move_cursor_up(&mut self) {
        if self.cursor_row > 0 {
            self.cursor_row -= 1;
            self.cursor_col = self.cursor_col.min(self.lines[self.cursor_row].len());
        }
    }

    fn move_cursor_down(&mut self) {
        if self.cursor_row < self.lines.len() - 1 {
            self.cursor_row += 1;
            self.cursor_col = self.cursor_col.min(self.lines[self.cursor_row].len());
        }
    }

    fn move_cursor_left(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            self.cursor_row -= 1;
            self.cursor_col = self.lines[self.cursor_row].len();
        }
    }

    fn move_cursor_right(&mut self) {
        let line_len = self.lines[self.cursor_row].len();
        if self.cursor_col < line_len {
            self.cursor_col += 1;
        } else if self.cursor_row < self.lines.len() - 1 {
            self.cursor_row += 1;
            self.cursor_col = 0;
        }
    }

    fn insert_char(&mut self, c: char) {
        self.lines[self.cursor_row].insert(self.cursor_col, c);
        self.cursor_col += 1;
    }

    fn backspace(&mut self) {
        if self.cursor_col > 0 {
            self.lines[self.cursor_row].remove(self.cursor_col - 1);
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            let line = self.lines.remove(self.cursor_row);
            self.cursor_row -= 1;
            self.cursor_col = self.lines[self.cursor_row].len();
            self.lines[self.cursor_row].push_str(&line);
        }
    }

    fn delete(&mut self) {
        if self.cursor_col < self.lines[self.cursor_row].len() {
            self.lines[self.cursor_row].remove(self.cursor_col);
        } else if self.cursor_row < self.lines.len() - 1 {
            let next_line = self.lines.remove(self.cursor_row + 1);
            self.lines[self.cursor_row].push_str(&next_line);
        }
    }

    fn new_line(&mut self) {
        let rest = self.lines[self.cursor_row][self.cursor_col..].to_string();
        self.lines[self.cursor_row].truncate(self.cursor_col);
        self.lines.insert(self.cursor_row + 1, rest);
        self.cursor_row += 1;
        self.cursor_col = 0;
    }
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for Editor {
    fn render(&self, frame: &mut Frame, area: Rect) {
        let lines: Vec<Line> = self
            .lines
            .iter()
            .enumerate()
            .map(|(row, line)| {
                if row == self.cursor_row {
                    // Highlight cursor position
                    let before = &line[..self.cursor_col];
                    let at = if self.cursor_col < line.len() {
                        &line[self.cursor_col..self.cursor_col + 1]
                    } else {
                        " "
                    };

                    Line::from(vec![
                        Span::raw(before),
                        Span::styled(at, Style::default().bg(Color::White).fg(Color::Black)),
                        Span::raw(&line[self.cursor_col + 1..]),
                    ])
                } else {
                    Line::from(Span::raw(line))
                }
            })
            .collect();

        let paragraph = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Input"))
            .wrap(Wrap { trim: true });

        frame.render_widget(paragraph, area);
    }

    fn handle_input(&mut self, cmd: &KeyCommand) -> bool {
        match cmd {
            KeyCommand::Char(c) => {
                self.insert_char(*c);
                true
            }
            KeyCommand::Backspace => {
                self.backspace();
                true
            }
            KeyCommand::Delete => {
                self.delete();
                true
            }
            KeyCommand::ArrowUp => {
                self.move_cursor_up();
                true
            }
            KeyCommand::ArrowDown => {
                self.move_cursor_down();
                true
            }
            KeyCommand::ArrowLeft => {
                self.move_cursor_left();
                true
            }
            KeyCommand::ArrowRight => {
                self.move_cursor_right();
                true
            }
            KeyCommand::Enter => {
                self.new_line();
                true
            }
            KeyCommand::Home => {
                self.cursor_col = 0;
                true
            }
            KeyCommand::End => {
                self.cursor_col = self.lines[self.cursor_row].len();
                true
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Editor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_editor_new() {
        let editor = Editor::new();
        assert_eq!(editor.text(), "");
        assert_eq!(editor.cursor_row, 0);
        assert_eq!(editor.cursor_col, 0);
    }

    #[test]
    fn test_editor_insert_char() {
        let mut editor = Editor::new();
        editor.insert_char('a');
        assert_eq!(editor.text(), "a");
        assert_eq!(editor.cursor_col, 1);
    }

    #[test]
    fn test_editor_backspace() {
        let mut editor = Editor::new();
        editor.insert_char('a');
        editor.insert_char('b');
        editor.backspace();
        assert_eq!(editor.text(), "a");
        assert_eq!(editor.cursor_col, 1);
    }

    #[test]
    fn test_editor_new_line() {
        let mut editor = Editor::new();
        editor.insert_char('a');
        editor.new_line();
        editor.insert_char('b');
        assert_eq!(editor.text(), "a\nb");
        assert_eq!(editor.cursor_row, 1);
    }
}
