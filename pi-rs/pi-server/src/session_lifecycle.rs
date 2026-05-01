//! Session lifecycle management for the daemon.
//! To be implemented in Phase 5.4.

use anyhow::Result;
use pi_core::session::Session;
use std::collections::HashMap;

pub struct SessionManager {
    sessions: HashMap<String, Session>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    pub fn create(&mut self, model: &str) -> String {
        let session = Session::new(model, ".");
        let id = session.id().as_str().to_string();
        self.sessions.insert(id.clone(), session);
        id
    }

    pub fn get(&self, id: &str) -> Option<&Session> {
        self.sessions.get(id)
    }

    pub fn list(&self) -> Vec<(&str, &Session)> {
        self.sessions.iter().map(|(id, s)| (id.as_str(), s)).collect()
    }
}
