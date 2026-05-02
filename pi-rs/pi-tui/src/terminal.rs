//! Terminal backend - raw mode, screen buffer, event handling.
//! Implements Phase 3.1 using crossterm for cross-platform support.

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
use ratatui::prelude::*;
use std::io::Stdout;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use ratatui::backend::CrosstermBackend;

/// Terminal events (keyboard, mouse, resize)
#[derive(Debug, Clone)]
pub enum TerminalEvent {
    /// Key press event
    Key(KeyEvent),
    /// Mouse event
    Mouse(MouseEvent),
    /// Terminal resize event (width, height)
    Resize(u16, u16),
    /// Bracketed paste from the terminal.
    Paste(String),
    /// Tick for animation/updates
    Tick,
}

/// Terminal backend using crossterm
pub struct Terminal {
    terminal: ratatui::Terminal<CrosstermBackend<Stdout>>,
    event_rx: mpsc::Receiver<TerminalEvent>,
    _event_tx: mpsc::Sender<TerminalEvent>,
}

impl Terminal {
    /// Create a new terminal instance
    pub fn new() -> Result<Self> {
        // Setup panic hook to restore terminal on panic
        let panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let _ = Self::restore_terminal();
            panic_hook(panic);
        }));

        // Enable raw mode and setup terminal
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )?;
        // Kitty keyboard protocol: lets us distinguish Shift+Enter from
        // plain Enter, catch key *releases*, and disambiguate the
        // classic Esc=Ctrl+[ overload. Querying first avoids errors on
        // terminals that don't implement it.
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

        // Setup event channel
        let (tx, rx) = mpsc::channel();

        // Spawn event listener thread
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

                // Send tick event every tick_rate
                let _ = tx_clone.send(TerminalEvent::Tick);
            }
        });

        Ok(Self {
            terminal,
            event_rx: rx,
            _event_tx: tx,
        })
    }

    /// Get terminal size (width, height)
    pub fn size(&self) -> (u16, u16) {
        self.terminal
            .size()
            .map(|f| (f.width, f.height))
            .unwrap_or((80, 24))
    }

    /// Get next terminal event (non-blocking)
    pub fn next_event(&mut self) -> Option<TerminalEvent> {
        self.event_rx.try_recv().ok()
    }

    /// Wait for next terminal event (blocking with timeout)
    pub fn poll_event(&mut self, timeout: Duration) -> Option<TerminalEvent> {
        match self.event_rx.recv_timeout(timeout) {
            Ok(event) => Some(event),
            Err(_) => None,
        }
    }

    /// Draw a frame using the provided closure
    pub fn draw<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&mut Frame),
    {
        self.terminal.draw(f)?;
        Ok(())
    }

    /// Clear the screen
    pub fn clear(&mut self) -> Result<()> {
        self.terminal.clear()?;
        Ok(())
    }

    /// Restore terminal to normal mode
    fn restore_terminal() -> Result<()> {
        disable_raw_mode()?;
        let mut stdout = std::io::stdout();
        // Pop the keyboard-enhancement flags first so the terminal is
        // left in whatever state the user's shell expects. The pop is
        // a no-op on terminals that never received the push, so it's
        // safe to unconditionally execute.
        let _ = execute!(stdout, PopKeyboardEnhancementFlags);
        execute!(
            stdout,
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen
        )?;
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
        // Verify event enum variants exist
        let _ = TerminalEvent::Tick;
    }
}
