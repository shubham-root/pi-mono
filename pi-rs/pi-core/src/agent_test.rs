//! Tests for Agent tool dispatch loop and multi-turn conversation.

#[cfg(test)]
mod tests {
    use crate::Agent;
    use pi_ai::types::*;
    use pi_tools::{Tool, ToolContext, ToolResult};
    use anyhow::Result;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken as TokioCancellationToken;

    /// Mock tool for testing that counts invocations.
    #[derive(Clone, Debug)]
    struct MockTool {
        call_count: Arc<Mutex<usize>>,
        name: String,
        response: String,
    }

    impl MockTool {
        fn new(name: &str, response: &str) -> Self {
            Self {
                call_count: Arc::new(Mutex::new(0)),
                name: name.to_string(),
                response: response.to_string(),
            }
        }

        fn get_call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Mock tool for testing"
        }

        fn schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                }
            })
        }

        async fn execute(&self, _context: &ToolContext, _input: serde_json::Value) -> Result<ToolResult> {
            *self.call_count.lock().unwrap() += 1;
            Ok(ToolResult::Success(self.response.clone()))
        }
    }

    /// Mock provider that emits tool calls in sequence.
    struct MockProvider {
        responses: Vec<Vec<StreamEvent>>,
        call_index: Arc<Mutex<usize>>,
    }

    impl MockProvider {
        fn new() -> Self {
            Self {
                responses: Vec::new(),
                call_index: Arc::new(Mutex::new(0)),
            }
        }

        fn add_response(&mut self, events: Vec<StreamEvent>) {
            self.responses.push(events);
        }

        async fn stream_mock(
            &self,
        ) -> Result<Vec<StreamEvent>, anyhow::Error> {
            let idx = {
                let mut i = self.call_index.lock().unwrap();
                let current = *i;
                if current < self.responses.len() {
                    *i += 1;
                    current
                } else {
                    self.responses.len() - 1
                }
            };
            Ok(self.responses[idx].clone())
        }
    }

    #[tokio::test]
    async fn test_agent_single_tool_call() -> Result<()> {
        // Create agent with one tool
        let tool = MockTool::new("calculator", "42");
        let mut agent = Agent::new("claude-3-sonnet").with_api_key("fake-key");
        agent = agent.with_tool(Box::new(tool.clone()));

        // Note: Since we can't easily mock the provider in the agent loop without
        // refactoring, we'll test the tool execution path directly via the prompt.
        // For now, this is a placeholder that verifies the agent can be created with tools.
        
        assert_eq!(agent.tools_count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_agent_max_turns_limit() -> Result<()> {
        let mut agent = Agent::new("claude-3-sonnet").with_api_key("fake-key");
        agent.config_mut().max_turns = 3;

        assert_eq!(agent.config().max_turns, 3);
        Ok(())
    }

    #[tokio::test]
    async fn test_agent_with_multiple_tools() -> Result<()> {
        let tool1 = MockTool::new("tool1", "result1");
        let tool2 = MockTool::new("tool2", "result2");
        let tool3 = MockTool::new("tool3", "result3");

        let mut agent = Agent::new("claude-3-sonnet").with_api_key("fake-key");
        agent = agent
            .with_tool(Box::new(tool1))
            .with_tool(Box::new(tool2))
            .with_tool(Box::new(tool3));

        assert_eq!(agent.tools_count(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn test_build_tool_schemas() -> Result<()> {
        let tool = MockTool::new("test_tool", "response");
        let mut agent = Agent::new("claude-3-sonnet").with_api_key("fake-key");
        agent = agent.with_tool(Box::new(tool));

        let schemas = agent.build_tool_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "test_tool");
        assert_eq!(schemas[0].description, "Mock tool for testing");
        Ok(())
    }

    #[test]
    fn test_message_with_tool_calls() {
        let tool_call = ToolCall {
            id: "call-1".to_string(),
            name: "calculator".to_string(),
            input: serde_json::json!({"a": 1, "b": 2}),
        };

        let content = Content::ToolUse {
            id: "call-1".to_string(),
            name: "calculator".to_string(),
            input: serde_json::json!({"a": 1, "b": 2}),
            cache_control: None,
        };

        let message = Message::Assistant(vec![content]);
        
        match message {
            Message::Assistant(contents) => {
                assert_eq!(contents.len(), 1);
                match &contents[0] {
                    Content::ToolUse {
                        id,
                        name,
                        ..
                    } => {
                        assert_eq!(id, "call-1");
                        assert_eq!(name, "calculator");
                    }
                    _ => panic!("Expected ToolUse content"),
                }
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_tool_result_message() {
        let result_content = Content::Text {
            text: "Tool executed successfully".to_string(),
            cache_control: None,
        };

        let tool_result = Message::Tool {
            tool_use_id: "call-1".to_string(),
            content: vec![result_content],
            is_error: Some(false),
        };

        match tool_result {
            Message::Tool {
                tool_use_id,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "call-1");
                assert_eq!(is_error, Some(false));
            }
            _ => panic!("Expected Tool message"),
        }
    }

    #[test]
    fn test_tool_result_error() {
        let error_content = Content::Text {
            text: "[ERROR] Tool failed: file not found".to_string(),
            cache_control: None,
        };

        let tool_result = Message::Tool {
            tool_use_id: "call-2".to_string(),
            content: vec![error_content],
            is_error: Some(true),
        };

        match tool_result {
            Message::Tool {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "call-2");
                assert_eq!(is_error, Some(true));
                match &content[0] {
                    Content::Text { text, .. } => {
                        assert!(text.starts_with("[ERROR]"));
                    }
                    _ => panic!("Expected Text content"),
                }
            }
            _ => panic!("Expected Tool message"),
        }
    }

    #[test]
    fn test_message_history_with_tools() {
        let user_msg = Message::User(vec![Content::Text {
            text: "Calculate 1+1".to_string(),
            cache_control: None,
        }]);

        let assistant_msg = Message::Assistant(vec![
            Content::Text {
                text: "I'll calculate that.".to_string(),
                cache_control: None,
            },
            Content::ToolUse {
                id: "call-1".to_string(),
                name: "calculator".to_string(),
                input: serde_json::json!({"a": 1, "b": 1}),
                cache_control: None,
            },
        ]);

        let tool_result_msg = Message::Tool {
            tool_use_id: "call-1".to_string(),
            content: vec![Content::Text {
                text: "2".to_string(),
                cache_control: None,
            }],
            is_error: Some(false),
        };

        let mut history = vec![];
        history.push(user_msg);
        history.push(assistant_msg);
        history.push(tool_result_msg);

        assert_eq!(history.len(), 3);
        match &history[2] {
            Message::Tool { is_error, .. } => {
                assert_eq!(*is_error, Some(false));
            }
            _ => panic!("Expected Tool message at index 2"),
        }
    }

    #[test]
    fn test_stream_event_tool_call_delta() {
        let delta = ToolCallDelta {
            id: Some("call-1".to_string()),
            name: Some("bash".to_string()),
            input: Some(serde_json::json!({"command": "ls"})),
        };

        let event = StreamEvent::ToolCallDelta { delta };

        match event {
            StreamEvent::ToolCallDelta { delta } => {
                assert_eq!(delta.id, Some("call-1".to_string()));
                assert_eq!(delta.name, Some("bash".to_string()));
                assert!(delta.input.is_some());
            }
            _ => panic!("Expected ToolCallDelta event"),
        }
    }

    #[test]
    fn test_stream_event_serialization() {
        let event = StreamEvent::TextDelta {
            delta: "Hello".to_string(),
            thinking: None,
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: StreamEvent = serde_json::from_str(&json).unwrap();

        match deserialized {
            StreamEvent::TextDelta { delta, .. } => {
                assert_eq!(delta, "Hello");
            }
            _ => panic!("Expected TextDelta event"),
        }
    }
}
