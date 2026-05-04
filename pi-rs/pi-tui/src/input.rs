//! Input parser - convert terminal events to semantic key commands.
//! Implements Phase 3.2 with support for Kitty keyboard protocol and fallback.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use serde::{Deserialize, Serialize};

/// Semantic key command
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyCommand {
    // Navigation
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    PageUp,
    PageDown,
    Home,
    End,

    // Editing
    Enter,
    Backspace,
    Delete,
    Tab,
    Escape,

    // Characters
    Char(char),

    // Function keys
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,

    // Ctrl combinations
    CtrlA,
    CtrlB,
    CtrlC,
    CtrlD,
    CtrlE,
    CtrlI,
    CtrlJ,
    CtrlK,
    CtrlL,
    CtrlN,
    CtrlO,
    CtrlP,
    CtrlQ,
    CtrlS,
    CtrlT,
    CtrlU,
    CtrlV,
    CtrlW,
    CtrlX,
    CtrlY,
    CtrlZ,
    CtrlBackslash, // Ctrl+\

    // Alt combinations
    AltA,
    AltB,
    AltC,
    AltD,
    AltE,
    AltF,
    AltG,
    AltH,
    AltI,
    AltJ,
    AltK,
    AltL,
    AltM,
    AltN,
    AltO,
    AltP,
    AltQ,
    AltR,
    AltS,
    AltT,
    AltU,
    AltV,
    AltW,
    AltX,
    AltY,
    AltZ,

    // Special Alt combinations
    AltEnter,

    // Shift combinations
    ShiftTab,
    ShiftEnter,
    ShiftHome,
    ShiftEnd,
    ShiftPageUp,
    ShiftPageDown,
    ShiftCtrlP,
    ShiftArrowLeft,
    ShiftArrowRight,
    ShiftArrowUp,
    ShiftArrowDown,

    // Other
    Unknown,
}

/// Parse crossterm KeyEvent to semantic KeyCommand
pub fn parse_key_event(event: KeyEvent) -> KeyCommand {
    // With the Kitty keyboard protocol enabled we start receiving
    // `Release` and `Repeat` events; ignore releases so keys aren't
    // processed twice, keep presses and repeats.
    if matches!(event.kind, KeyEventKind::Release) {
        return KeyCommand::Unknown;
    }
    match (event.code, event.modifiers) {
        // Navigation
        (KeyCode::Up, KeyModifiers::NONE) => KeyCommand::ArrowUp,
        (KeyCode::Down, KeyModifiers::NONE) => KeyCommand::ArrowDown,
        (KeyCode::Left, KeyModifiers::NONE) => KeyCommand::ArrowLeft,
        (KeyCode::Right, KeyModifiers::NONE) => KeyCommand::ArrowRight,
        (KeyCode::PageUp, KeyModifiers::NONE) => KeyCommand::PageUp,
        (KeyCode::PageDown, KeyModifiers::NONE) => KeyCommand::PageDown,
        (KeyCode::Home, KeyModifiers::NONE) => KeyCommand::Home,
        (KeyCode::End, KeyModifiers::NONE) => KeyCommand::End,

        // Editing
        (KeyCode::Enter, KeyModifiers::NONE) => KeyCommand::Enter,
        (KeyCode::Backspace, KeyModifiers::NONE) => KeyCommand::Backspace,
        (KeyCode::Delete, KeyModifiers::NONE) => KeyCommand::Delete,
        (KeyCode::Tab, KeyModifiers::NONE) => KeyCommand::Tab,
        (KeyCode::Tab, KeyModifiers::SHIFT) => KeyCommand::ShiftTab,
        (KeyCode::Enter, KeyModifiers::SHIFT) => KeyCommand::ShiftEnter,
        (KeyCode::Home, KeyModifiers::SHIFT) => KeyCommand::ShiftHome,
        (KeyCode::End, KeyModifiers::SHIFT) => KeyCommand::ShiftEnd,
        (KeyCode::PageUp, KeyModifiers::SHIFT) => KeyCommand::ShiftPageUp,
        (KeyCode::PageDown, KeyModifiers::SHIFT) => KeyCommand::ShiftPageDown,
        (KeyCode::Left, KeyModifiers::SHIFT) => KeyCommand::ShiftArrowLeft,
        (KeyCode::Right, KeyModifiers::SHIFT) => KeyCommand::ShiftArrowRight,
        (KeyCode::Up, KeyModifiers::SHIFT) => KeyCommand::ShiftArrowUp,
        (KeyCode::Down, KeyModifiers::SHIFT) => KeyCommand::ShiftArrowDown,
        (KeyCode::Esc, _) => KeyCommand::Escape,

        // Function keys
        (KeyCode::F(1), _) => KeyCommand::F1,
        (KeyCode::F(2), _) => KeyCommand::F2,
        (KeyCode::F(3), _) => KeyCommand::F3,
        (KeyCode::F(4), _) => KeyCommand::F4,
        (KeyCode::F(5), _) => KeyCommand::F5,
        (KeyCode::F(6), _) => KeyCommand::F6,
        (KeyCode::F(7), _) => KeyCommand::F7,
        (KeyCode::F(8), _) => KeyCommand::F8,
        (KeyCode::F(9), _) => KeyCommand::F9,
        (KeyCode::F(10), _) => KeyCommand::F10,
        (KeyCode::F(11), _) => KeyCommand::F11,
        (KeyCode::F(12), _) => KeyCommand::F12,
        
        // Special combinations (must come before general Char case)
        (KeyCode::Enter, KeyModifiers::ALT) => KeyCommand::AltEnter,
        (KeyCode::Char('p'), KeyModifiers::CONTROL | KeyModifiers::SHIFT) => KeyCommand::ShiftCtrlP,

        // Character keys
        (KeyCode::Char(c), KeyModifiers::NONE) => KeyCommand::Char(c),
        (KeyCode::Char(c), KeyModifiers::SHIFT) => KeyCommand::Char(c.to_ascii_uppercase()),

        // Ctrl combinations
        (KeyCode::Char('a'), KeyModifiers::CONTROL) => KeyCommand::CtrlA,
        (KeyCode::Char('b'), KeyModifiers::CONTROL) => KeyCommand::CtrlB,
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => KeyCommand::CtrlC,
        (KeyCode::Char('d'), KeyModifiers::CONTROL) => KeyCommand::CtrlD,
        (KeyCode::Char('e'), KeyModifiers::CONTROL) => KeyCommand::CtrlE,
        // Ctrl+I only fires as its own command on terminals that
        // implement the Kitty keyboard protocol (disambiguate
        // escape codes). On classic terminals the byte 0x09 is
        // returned for both Ctrl+I and Tab, so crossterm yields
        // `KeyCode::Tab` there and this arm never matches — Tab
        // semantics are preserved.
        (KeyCode::Char('i'), KeyModifiers::CONTROL) => KeyCommand::CtrlI,
        (KeyCode::Char('j'), KeyModifiers::CONTROL) => KeyCommand::CtrlJ,
        (KeyCode::Char('k'), KeyModifiers::CONTROL) => KeyCommand::CtrlK,
        (KeyCode::Char('l'), KeyModifiers::CONTROL) => KeyCommand::CtrlL,
        (KeyCode::Char('n'), KeyModifiers::CONTROL) => KeyCommand::CtrlN,
        (KeyCode::Char('o'), KeyModifiers::CONTROL) => KeyCommand::CtrlO,
        (KeyCode::Char('p'), KeyModifiers::CONTROL) => KeyCommand::CtrlP,
        (KeyCode::Char('q'), KeyModifiers::CONTROL) => KeyCommand::CtrlQ,
        (KeyCode::Char('s'), KeyModifiers::CONTROL) => KeyCommand::CtrlS,
        (KeyCode::Char('t'), KeyModifiers::CONTROL) => KeyCommand::CtrlT,
        (KeyCode::Char('u'), KeyModifiers::CONTROL) => KeyCommand::CtrlU,
        (KeyCode::Char('v'), KeyModifiers::CONTROL) => KeyCommand::CtrlV,
        (KeyCode::Char('w'), KeyModifiers::CONTROL) => KeyCommand::CtrlW,
        (KeyCode::Char('x'), KeyModifiers::CONTROL) => KeyCommand::CtrlX,
        (KeyCode::Char('y'), KeyModifiers::CONTROL) => KeyCommand::CtrlY,
        (KeyCode::Char('z'), KeyModifiers::CONTROL) => KeyCommand::CtrlZ,
        (KeyCode::Char('\\'), KeyModifiers::CONTROL) => KeyCommand::CtrlBackslash,

        // Alt combinations
        (KeyCode::Char('a'), KeyModifiers::ALT) => KeyCommand::AltA,
        (KeyCode::Char('b'), KeyModifiers::ALT) => KeyCommand::AltB,
        (KeyCode::Char('c'), KeyModifiers::ALT) => KeyCommand::AltC,
        (KeyCode::Char('d'), KeyModifiers::ALT) => KeyCommand::AltD,
        (KeyCode::Char('e'), KeyModifiers::ALT) => KeyCommand::AltE,
        (KeyCode::Char('f'), KeyModifiers::ALT) => KeyCommand::AltF,
        (KeyCode::Char('g'), KeyModifiers::ALT) => KeyCommand::AltG,
        (KeyCode::Char('h'), KeyModifiers::ALT) => KeyCommand::AltH,
        (KeyCode::Char('i'), KeyModifiers::ALT) => KeyCommand::AltI,
        (KeyCode::Char('j'), KeyModifiers::ALT) => KeyCommand::AltJ,
        (KeyCode::Char('k'), KeyModifiers::ALT) => KeyCommand::AltK,
        (KeyCode::Char('l'), KeyModifiers::ALT) => KeyCommand::AltL,
        (KeyCode::Char('m'), KeyModifiers::ALT) => KeyCommand::AltM,
        (KeyCode::Char('n'), KeyModifiers::ALT) => KeyCommand::AltN,
        (KeyCode::Char('o'), KeyModifiers::ALT) => KeyCommand::AltO,
        (KeyCode::Char('p'), KeyModifiers::ALT) => KeyCommand::AltP,
        (KeyCode::Char('q'), KeyModifiers::ALT) => KeyCommand::AltQ,
        (KeyCode::Char('r'), KeyModifiers::ALT) => KeyCommand::AltR,
        (KeyCode::Char('s'), KeyModifiers::ALT) => KeyCommand::AltS,
        (KeyCode::Char('t'), KeyModifiers::ALT) => KeyCommand::AltT,
        (KeyCode::Char('u'), KeyModifiers::ALT) => KeyCommand::AltU,
        (KeyCode::Char('v'), KeyModifiers::ALT) => KeyCommand::AltV,
        (KeyCode::Char('w'), KeyModifiers::ALT) => KeyCommand::AltW,
        (KeyCode::Char('x'), KeyModifiers::ALT) => KeyCommand::AltX,
        (KeyCode::Char('y'), KeyModifiers::ALT) => KeyCommand::AltY,
        (KeyCode::Char('z'), KeyModifiers::ALT) => KeyCommand::AltZ,

        _ => KeyCommand::Unknown,
    }
}

pub struct InputParser;

impl InputParser {
    /// Parse keyboard event to command
    pub fn parse(event: KeyEvent) -> KeyCommand {
        parse_key_event(event)
    }

    /// Check if command is an editing command
    pub fn is_edit_command(cmd: &KeyCommand) -> bool {
        matches!(
            cmd,
            KeyCommand::Char(_)
                | KeyCommand::Backspace
                | KeyCommand::Delete
                | KeyCommand::Enter
                | KeyCommand::Tab
        )
    }

    /// Check if command is a navigation command
    pub fn is_nav_command(cmd: &KeyCommand) -> bool {
        matches!(
            cmd,
            KeyCommand::ArrowUp
                | KeyCommand::ArrowDown
                | KeyCommand::ArrowLeft
                | KeyCommand::ArrowRight
                | KeyCommand::Home
                | KeyCommand::End
                | KeyCommand::PageUp
                | KeyCommand::PageDown
        )
    }

    /// Check if command should exit the application
    pub fn is_exit_command(cmd: &KeyCommand) -> bool {
        matches!(cmd, KeyCommand::CtrlC | KeyCommand::CtrlD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    #[test]
    fn test_parse_char() {
        let event = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        let cmd = parse_key_event(event);
        assert_eq!(cmd, KeyCommand::Char('a'));
    }

    #[test]
    fn test_parse_ctrl_c() {
        let event = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        let cmd = parse_key_event(event);
        assert_eq!(cmd, KeyCommand::CtrlC);
    }

    #[test]
    fn test_parse_arrow_up() {
        let event = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);
        let cmd = parse_key_event(event);
        assert_eq!(cmd, KeyCommand::ArrowUp);
    }

    #[test]
    fn test_is_edit_command() {
        assert!(InputParser::is_edit_command(&KeyCommand::Char('a')));
        assert!(InputParser::is_edit_command(&KeyCommand::Backspace));
        assert!(!InputParser::is_edit_command(&KeyCommand::ArrowUp));
    }

    #[test]
    fn test_is_exit_command() {
        assert!(InputParser::is_exit_command(&KeyCommand::CtrlC));
        assert!(InputParser::is_exit_command(&KeyCommand::CtrlD));
        assert!(!InputParser::is_exit_command(&KeyCommand::Char('a')));
    }
}
