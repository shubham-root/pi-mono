//! Terminal backend - raw mode, event handling, inline viewport.
//!
//! Uses ratatui's `Viewport::Inline` so pi owns a fixed-height region
//! at the bottom of the terminal while the scrollback above holds
//! the normal shell history PLUS any messages pi promotes to it.
//! This gives us native terminal scrolling for conversation history
//! (mouse wheel / trackpad / terminal's own scroll UI) for free.

use anyhow::Result;
use crossterm::{
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyEvent,
        KeyboardEnhancementFlags, MouseEvent, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement},
};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use ratatui::{TerminalOptions, Viewport};
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

/// Default height for the inline viewport. Chosen to fit:
///   1 row  - last-call stats
///   1 row  - upper boundary rule
///   1..=7 rows - editor
///   1 row  - lower boundary rule
///   1 row  - hint line
///   1 row  - pwd (branch)
///   1 row  - context / model
/// Plus a couple of extra rows of slack for overlays / active streaming
/// content. Overlays and tall editor content momentarily exceed this
/// and cause ratatui to resize the viewport.
const INLINE_VIEWPORT_ROWS: u16 = 14;

pub struct Terminal {
    terminal: ratatui::Terminal<CrosstermBackend<Stdout>>,
    event_rx: mpsc::Receiver<TerminalEvent>,
    _event_tx: mpsc::Sender<TerminalEvent>,
}

impl Terminal {
    pub fn new() -> Result<Self> {
        // Restore terminal on panic so a crash doesn't leave the
        // user's shell in raw mode.
        let panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let _ = Self::restore_terminal();
            panic_hook(panic);
        }));

        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        // Bracketed paste lets us receive multi-line paste as a single
        // event instead of a sequence of Enter keypresses. Mouse capture
        // is intentionally NOT enabled because inline mode wants the
        // terminal's native scrollback to keep working when the user
        // scrolls up with their trackpad / wheel; capturing the events
        // would steal them from the terminal.
        execute!(stdout, EnableBracketedPaste)?;
        // Kitty keyboard protocol lets us distinguish Shift+Enter and
        // catch key releases; query before pushing so we don't error
        // on terminals that don't implement it.
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
        // Inline viewport: pi owns a fixed-height region at the bottom
        // of the terminal. Content above it (including anything we
        // promote via `insert_before`) lives in the terminal's native
        // scrollback so the user's existing scroll workflow just works.
        let terminal = ratatui::Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(INLINE_VIEWPORT_ROWS),
            },
        )?;

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

    /// Promote a block of content *above* the inline viewport into the
    /// terminal's native scrollback. The caller renders the full
    /// block (via ratatui widgets) inside the closure. Used for
    /// completed conversation turns so the user's terminal scrollback
    /// naturally accumulates chat history.
    pub fn insert_before<F>(&mut self, height: u16, draw_fn: F) -> Result<()>
    where
        F: FnOnce(&mut Buffer),
    {
        self.terminal.insert_before(height, draw_fn)?;
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
        // Inline viewport doesn't use the alternate screen, so we only
        // need to tear down the protocol opt-ins.
        execute!(stdout, DisableBracketedPaste)?;
        // Leave the inline viewport on screen as final output — users
        // see their last interaction preserved in the terminal
        // instead of an empty cleared frame.
        let _ = execute!(stdout, crossterm::cursor::Show);
        let _ = writeln!(stdout);
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = Self::restore_terminal();
    }
}

use std::io::Write as _;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_event_types() {
        let _ = TerminalEvent::Tick;
    }
}
