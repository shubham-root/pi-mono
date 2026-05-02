//! Event loop - async event dispatcher coordinating terminal, input, and agent.
//! Implements Phase 3.4

use crate::input::{parse_key_event, KeyCommand};
use crate::terminal::{Terminal, TerminalEvent};
use anyhow::Result;
use crossterm::event::KeyEvent;
use pi_ai::types::{Message, Content};
use pi_core::Agent;
use std::sync::mpsc;
use std::time::Duration;
use tokio::sync::mpsc as async_mpsc;

/// Application event
#[derive(Debug, Clone)]
pub enum AppEvent {
    Key(KeyCommand),
    Paste(String),
    Mouse(crossterm::event::MouseEvent),
    AgentMessage(String),
    AgentToolCall { tool: String, args: String },
    AgentToolResult { result: String },
    Resize(u16, u16),
    Tick,
}

/// Event loop state
pub struct EventLoop {
    terminal: Terminal,
    agent: Option<Agent>,
    event_tx: mpsc::Sender<AppEvent>,
    event_rx: mpsc::Receiver<AppEvent>,
}

impl EventLoop {
    /// Create a new event loop
    pub fn new() -> Result<Self> {
        let terminal = Terminal::new()?;
        let (tx, rx) = mpsc::channel();

        Ok(Self {
            terminal,
            agent: None,
            event_tx: tx,
            event_rx: rx,
        })
    }

    /// Set the agent
    pub fn set_agent(&mut self, agent: Agent) {
        self.agent = Some(agent);
    }

    /// Dispatch a key command to handlers
    pub fn dispatch_key(&self, key: KeyCommand) -> Result<()> {
        self.event_tx.send(AppEvent::Key(key))?;
        Ok(())
    }

    /// Send agent message event
    pub fn send_agent_message(&self, msg: String) -> Result<()> {
        self.event_tx.send(AppEvent::AgentMessage(msg))?;
        Ok(())
    }

    /// Poll for the next event
    pub fn poll_event(&mut self, timeout: Duration) -> Option<AppEvent> {
        // Check terminal events first
        if let Some(term_event) = self.terminal.poll_event(timeout) {
            return Some(match term_event {
                TerminalEvent::Key(key) => AppEvent::Key(parse_key_event(key)),
                TerminalEvent::Resize(w, h) => AppEvent::Resize(w, h),
                TerminalEvent::Mouse(m) => AppEvent::Mouse(m),
                TerminalEvent::Paste(s) => AppEvent::Paste(s),
                TerminalEvent::Tick => AppEvent::Tick,
            });
        }

        // Check app events
        self.event_rx.try_recv().ok()
    }

    /// Run the event loop (blocking)
    pub async fn run<F>(&mut self, mut handler: F) -> Result<()>
    where
        F: FnMut(&mut Self, AppEvent) -> Result<bool>, // true to continue, false to exit
    {
        loop {
            // Poll for events with 100ms timeout
            if let Some(event) = self.poll_event(Duration::from_millis(100)) {
                let should_continue = handler(self, event)?;
                if !should_continue {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get terminal for drawing
    pub fn terminal(&mut self) -> &mut Terminal {
        &mut self.terminal
    }

    /// Get agent reference
    pub fn agent(&self) -> Option<&Agent> {
        self.agent.as_ref()
    }

    /// Get mutable agent reference
    pub fn agent_mut(&mut self) -> Option<&mut Agent> {
        self.agent.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_event_key() {
        let key_cmd = KeyCommand::Char('a');
        let event = AppEvent::Key(key_cmd.clone());
        match event {
            AppEvent::Key(cmd) => assert_eq!(cmd, key_cmd),
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_app_event_agent_message() {
        let msg = "Hello from agent".to_string();
        let event = AppEvent::AgentMessage(msg.clone());
        match event {
            AppEvent::AgentMessage(m) => assert_eq!(m, msg),
            _ => panic!("Wrong event type"),
        }
    }
}
