//! Terminal backend - raw mode, event handling, fullscreen viewport.
//!
//! Uses ratatui's default (fullscreen) viewport in the alternate
//! screen so pi owns the whole terminal while it's running and
//! restores the user's shell on exit. This lets the renderer
//! re-paint every past message on every frame, which is what
//! makes toggles like Ctrl+I (tool output collapsed / expanded)
//! work retroactively across the entire conversation instead of
//! only affecting future turns.
//!
//! The previous inline-viewport approach (Phase 3.24) traded
//! retroactive re-rendering for native terminal scrollback. That
//! trade-off turned out to be wrong for the toggle UX — once a
//! turn is promoted into the terminal's scrollback, ratatui can't
//! touch it, so the Ctrl+I state was frozen per turn at the time
//! of promotion. Going back to alt-screen restores TS-pi parity:
//! everything lives in the viewport; scroll with PgUp / PgDn /
//! mouse wheel; toggles re-render the whole transcript.

use anyhow::Result;
use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste,
        EnableMouseCapture, Event, KeyEvent, KeyboardEnhancementFlags, MouseEvent,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use std::io::Stdout;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Terminal events (keyboard, mouse, resize)
#[derive(Debug, Clone)]
pub enum TerminalEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),
    Paste(String),
    Tick,
}

pub struct Terminal {
    terminal: ratatui::Terminal<CrosstermBackend<Stdout>>,
    event_rx: mpsc::Receiver<TerminalEvent>,
    _event_tx: mpsc::Sender<TerminalEvent>,
}

impl Terminal {
    pub fn new() -> Result<Self> {
        // Restore terminal on panic so a crash doesn't leave the
        // user's shell in raw mode + alt-screen.
        let panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let _ = Self::restore_terminal();
            panic_hook(panic);
        }));

        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        execute!(stdout, EnableBracketedPaste)?;
        // Mouse capture lets us receive wheel events (for scrolling
        // the conversation) and drag events (for the scrollbar).
        execute!(stdout, EnableMouseCapture)?;
        // Kitty keyboard protocol lets us distinguish Shift+Enter,
        // Ctrl+I from Tab, and catch key releases; query first so we
        // don't error on terminals that don't implement it.
        if matches!(supports_keyboard_enhancement(), Ok(true)) {
            let _ = execute!(
                stdout,
                PushKeyboardEnhancementFlags(
                    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                        | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS,
                )
            );
        }

        let backend = CrosstermBackend::new(stdout);
        let terminal = ratatui::Terminal::new(backend)?;

        let (tx, rx) = mpsc::channel();
        let tx_clone = tx.clone();
        thread::spawn(move || {
            let tick_rate = Duration::from_millis(100);
            loop {
                if event::poll(tick_rate).unwrap_or(false) {
                    if let Ok(event) = event::read() {
                        let terminal_event = match event {
                            Event::Key(key) => TerminalEvent::Key(key),
                            Event::Mouse(mouse) => TerminalEvent::Mouse(mouse),
                            Event::Resize(w, h) => TerminalEvent::Resize(w, h),
                            Event::Paste(s) => TerminalEvent::Paste(s),
                            _ => continue,
                        };
                        if tx_clone.send(terminal_event).is_err() {
                            break;
                        }
                    }
                }
                let _ = tx_clone.send(TerminalEvent::Tick);
            }
        });

        Ok(Self {
            terminal,
            event_rx: rx,
            _event_tx: tx,
        })
    }

    pub fn size(&self) -> (u16, u16) {
        self.terminal
            .size()
            .map(|f| (f.width, f.height))
            .unwrap_or((80, 24))
    }

    pub fn next_event(&mut self) -> Option<TerminalEvent> {
        self.event_rx.try_recv().ok()
    }

    pub fn poll_event(&mut self, timeout: Duration) -> Option<TerminalEvent> {
        self.event_rx.recv_timeout(timeout).ok()
    }

    pub fn draw<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&mut Frame),
    {
        self.terminal.draw(f)?;
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        self.terminal.clear()?;
        Ok(())
    }

    fn restore_terminal() -> Result<()> {
        disable_raw_mode()?;
        let mut stdout = std::io::stdout();
        let _ = execute!(stdout, PopKeyboardEnhancementFlags);
        let _ = execute!(stdout, DisableBracketedPaste);
        let _ = execute!(stdout, DisableMouseCapture);
        let _ = execute!(stdout, LeaveAlternateScreen);
        let _ = execute!(stdout, crossterm::cursor::Show);
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = Self::restore_terminal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_event_types() {
        let _ = TerminalEvent::Tick;
    }
}
