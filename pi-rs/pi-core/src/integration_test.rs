//! Integration tests for Agent print mode with tools.
//! These tests verify the agent loop works correctly with the tool dispatch system.

#[cfg(test)]
mod integration_tests {
    use crate::Agent;
    use pi_ai::types::{Message, Content};
    use pi_tools::{Tool, ToolContext, ToolResult};
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::json;

    /// Mock tool that counts calls and returns fixed responses
    #[derive(Clone, Debug)]
    struct CountingTool {
        call_count: std::sync::Arc<std::sync::Mutex<usize>>,
        name: String,
    }

    impl CountingTool {
        fn new(name: &str) -> Self {
            Self {
                call_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
                name: name.to_string(),
            }
        }

        fn get_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A tool that counts how many times it's called"
        }

        fn schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "A message" }
                }
            })
        }

        async fn execute(&self, _context: &ToolContext, _input: serde_json::Value) -> Result<ToolResult> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            Ok(ToolResult::Success(format!("Tool executed successfully (call #{})", count)))
        }
    }

    #[test]
    fn test_agent_creation() {
        let agent = Agent::new("claude-3-sonnet");
        assert_eq!(agent.messages().len(), 0);
    }

    #[test]
    fn test_agent_with_tools() {
        let tool1 = CountingTool::new("tool1");
        let tool2 = CountingTool::new("tool2");
        
        let agent = Agent::new("claude-3-sonnet")
            .with_tool(Box::new(tool1))
            .with_tool(Box::new(tool2));

        assert_eq!(agent.tools_count(), 2);
    }

    #[test]
    fn test_agent_with_api_key() {
        let agent = Agent::new("claude-3-sonnet")
            .with_api_key("test-key");
        // Should not panic, just verify it doesn't crash
        assert_eq!(agent.tools_count(), 0);
    }

    #[test]
    fn test_agent_with_system_prompt() {
        let agent = Agent::new("claude-3-sonnet")
            .with_system_prompt("You are a helpful assistant");
        assert_eq!(agent.tools_count(), 0);
    }

    #[test]
    fn test_agent_clear_messages() {
        let mut agent = Agent::new("claude-3-sonnet");
        // Manually add a message for testing
        let msg = Message::User(vec![Content::Text {
            text: "Hello".to_string(),
            cache_control: None,
        }]);
        // We can't directly push, but we can verify clear works on empty
        agent.clear();
        assert_eq!(agent.messages().len(), 0);
    }

    #[test]
    fn test_build_tool_schemas() -> Result<()> {
        let tool = CountingTool::new("test_tool");
        let agent = Agent::new("claude-3-sonnet")
            .with_tool(Box::new(tool));

        let schemas = agent.build_tool_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "test_tool");
        assert_eq!(schemas[0].description, "A tool that counts how many times it's called");
        Ok(())
    }

    #[test]
    fn test_agent_config_max_turns() {
        let mut agent = Agent::new("claude-3-sonnet");
        agent.config_mut().max_turns = 5;
        assert_eq!(agent.config().max_turns, 5);
    }

    #[test]
    fn test_agent_config_temperature() {
        let mut agent = Agent::new("claude-3-sonnet");
        agent.config_mut().temperature = Some(0.5);
        assert_eq!(agent.config().temperature, Some(0.5));
    }

    #[test]
    fn test_agent_config_allow_tools() {
        let mut agent = Agent::new("claude-3-sonnet");
        agent.config_mut().allow_tools = false;
        assert!(!agent.config().allow_tools);
    }

    // Note: Full end-to-end tests with real streaming would require:
    // 1. Mocking the provider::stream function
    // 2. Or using environment variables to enable integration tests
    // These are deferred to a later phase when we add more comprehensive
    // test infrastructure with fixtures and mock providers.
}
