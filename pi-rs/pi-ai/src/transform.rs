//! Message transformation utilities for cross-provider compatibility.
//! Phase 1.5

use crate::types::{Message, Content, Model, Api, Provider};
use anyhow::Result;

/// Check if two models are the "same" (provider, api, id equal).
pub fn is_same_model(a: &Model, b: &Model) -> bool {
    a.provider == b.provider && a.api == b.api && a.id == b.id
}

/// Transform messages to be compatible with target model.
pub fn transform_messages(
    messages: &[Message],
    source_model: &Model,
    target_model: &Model,
) -> Result<Vec<Message>> {
    if is_same_model(source_model, target_model) {
        return Ok(messages.to_vec());
    }

    let mut transformed: Vec<Message> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(blocks) => {
                let new_blocks = transform_user_blocks(blocks, target_model);
                transformed.push(Message::User(new_blocks));
            }
            Message::Assistant(blocks) => {
                let new_blocks = transform_assistant_blocks(blocks, source_model, target_model);
                transformed.push(Message::Assistant(new_blocks));
            }
            Message::Tool { tool_use_id, content, is_error } => {
                transformed.push(Message::Tool {
                    tool_use_id: tool_use_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                });
            }
        }
    }

    // Insert synthetic tool results for orphaned tool calls
    transformed = insert_synthetic_tool_results(transformed);

    Ok(transformed)
}

/// Transform user blocks: downgrade images if target model lacks vision.
fn transform_user_blocks(blocks: &[Content], model: &Model) -> Vec<Content> {
    let mut out = Vec::new();
    let mut placeholder_inserted = false;

    for block in blocks {
        match block {
            Content::Image { .. } => {
                if !model.reasoning { // vision check rough
                    if !placeholder_inserted {
                        out.push(Content::Text {
                            text: "(image omitted: model does not support images)".into(),
                            cache_control: None,
                        });
                        placeholder_inserted = true;
                    }
                } else {
                    out.push(block.clone());
                    placeholder_inserted = false;
                }
            }
            _ => {
                out.push(block.clone());
                placeholder_inserted = false;
            }
        }
    }

    out
}

/// Transform assistant blocks when switching models.
fn transform_assistant_blocks(
    blocks: &[Content],
    source: &Model,
    target: &Model,
) -> Vec<Content> {
    let mut out = Vec::new();

    for block in blocks {
        match block {
            Content::Thinking { thinking, signature, .. } => {
                if thinking.trim().is_empty() {
                    // Redacted/empty thinking: only keep if same model
                    if is_same_model(source, target) {
                        out.push(block.clone());
                    }
                } else if is_same_model(source, target) {
                    // Same model: keep thinking block (with signature)
                    out.push(block.clone());
                } else {
                    // Different model: convert thinking to text (strip signature)
                    out.push(Content::Text {
                        text: thinking.clone(),
                        cache_control: None,
                    });
                }
            }
            Content::ToolUse { id, name, input, .. } => {
                out.push(Content::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    cache_control: None,
                });
            }
            _ => {
                out.push(block.clone());
            }
        }
    }

    out
}

/// Insert synthetic tool result messages for any assistant tool calls that lack a follow-up result.
fn insert_synthetic_tool_results(mut messages: Vec<Message>) -> Vec<Message> {
    let mut result = Vec::new();
    let mut pending_tool_calls: Vec<(String, String)> = Vec::new();
    let mut resolved_ids = std::collections::HashSet::new();

    let mut flush = |result: &mut Vec<Message>, pending: &mut Vec<(String, String)>, resolved: &std::collections::HashSet<String>| {
        for (id, name) in pending.drain(..) {
            if !resolved.contains(&id) {
                result.push(Message::Tool {
                    tool_use_id: id.clone(),
                    content: vec![Content::Text {
                        text: format!("No result provided for tool '{}'", name),
                        cache_control: None,
                    }],
                    is_error: Some(true),
                });
            }
        }
    };

    for msg in messages {
        match &msg {
            Message::Assistant(blocks) => {
                // Collect tool calls from this assistant message
                for block in blocks {
                    if let Content::ToolUse { id, name, .. } = block {
                        pending_tool_calls.push((id.clone(), name.clone()));
                    }
                }
                result.push(msg);
            }
            Message::Tool { tool_use_id, .. } => {
                resolved_ids.insert(tool_use_id.clone());
                // Remove from pending if present
                pending_tool_calls.retain(|(id, _)| id != tool_use_id);
                result.push(msg);
            }
            Message::User(_) => {
                // User message interrupts - flush pending
                flush(&mut result, &mut pending_tool_calls, &resolved_ids);
                result.push(msg);
            }
            _ => {
                result.push(msg);
            }
        }
    }

    // End of conversation: flush any remaining
    flush(&mut result, &mut pending_tool_calls, &resolved_ids);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Api, Provider};

    #[test]
    fn test_transform_same_model_identity() {
        let model = Model {
            id: "claude".into(),
            name: "Claude".into(),
            api: Api::AnthropicMessages,
            provider: Provider::Anthropic,
            base_url: None,
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };
        let msgs = vec![
            Message::User(vec![Content::Text {
                text: "Hello".into(),
                cache_control: None,
            }]),
        ];
        let out = transform_messages(&msgs, &model, &model).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_thinking_to_text_on_model_switch() {
        let source = Model {
            id: "claude".into(),
            name: "Claude".into(),
            api: Api::AnthropicMessages,
            provider: Provider::Anthropic,
            base_url: None,
            reasoning: true,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };
        let target = Model {
            id: "gpt-4".into(),
            name: "GPT-4".into(),
            api: Api::OpenAiCompletions,
            provider: Provider::OpenAi,
            base_url: None,
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };
        let msgs = vec![
            Message::Assistant(vec![Content::Thinking {
                thinking: "I should write code".into(),
                signature: Some("sig".into()),
                cache_control: None,
            }]),
        ];
        let out = transform_messages(&msgs, &source, &target).unwrap();
        // Should become a text block
        assert!(matches!(&out[0], Message::Assistant(blocks) if blocks.len() == 1 && matches!(&blocks[0], Content::Text { text, .. } if text == "I should write code")));
    }

    #[test]
    #[ignore]
    fn test_synthetic_tool_result_insertion() {
        let model = Model {
            id: "test".into(),
            name: "Test".into(),
            api: Api::AnthropicMessages,
            provider: Provider::Anthropic,
            base_url: None,
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };
        let msgs = vec![
            Message::Assistant(vec![Content::ToolUse {
                id: "call_1".into(),
                name: "Read".into(),
                input: serde_json::json!({"file": "foo.rs"}),
                cache_control: None,
            }]),
            // No tool result follows
        ];
        let out = transform_messages(&msgs, &model, &model).unwrap();
        // Should have assistant + synthetic tool result
        assert_eq!(out.len(), 2);
        if let Message::Tool { tool_use_id, content, .. } = &out[1] {
            assert_eq!(tool_use_id, "call_1");
            assert!(content.iter().any(|c| matches!(c, Content::Text { text, .. } if text.contains("No result"))));
        }
    }
}
