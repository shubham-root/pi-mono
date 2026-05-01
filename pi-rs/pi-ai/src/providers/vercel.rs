//! Vercel AI Gateway - Gateway provider with support for many LLM providers.
//!
//! Routes requests to underlying providers (OpenAI, Anthropic, Cohere, etc.)
//! based on model selection, providing a unified streaming API.

use crate::types::*;
use anyhow::{anyhow, Result};
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::ErrorKind;
use tokio::io::AsyncBufRead;
use futures::StreamExt;
use tokio_util::io::StreamReader;

#[derive(Debug, Serialize)]
struct VercelRequest {
    model: String,
    messages: Vec<VercelMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<VercelTool>>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct VercelMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct VercelTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: VercelFunction,
}

#[derive(Debug, Serialize)]
struct VercelFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct VercelResponse {
    choices: Vec<VercelChoice>,
    #[serde(default)]
    usage: Option<VercelUsage>,
}

#[derive(Debug, Deserialize)]
struct VercelChoice {
    delta: Option<VercelDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VercelDelta {
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<VercelToolCall>>,
}

#[derive(Debug, Deserialize)]
struct VercelToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: VercelToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct VercelToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct VercelUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Convert pi messages to Vercel format
fn convert_messages(messages: &[Message]) -> Vec<VercelMessage> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            Message::User(content) => {
                let text = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if text.is_empty() {
                    None
                } else {
                    Some(VercelMessage {
                        role: "user".to_string(),
                        content: text,
                    })
                }
            }
            Message::Assistant(content) => {
                let text = format_assistant_content(content);
                Some(VercelMessage {
                    role: "assistant".to_string(),
                    content: text,
                })
            }
            Message::Tool { content, .. } => {
                let text = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Some(VercelMessage {
                    role: "user".to_string(),
                    content: text,
                })
            }
        })
        .collect()
}

fn format_assistant_content(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text, .. } => Some(text.clone()),
            Content::Thinking { thinking, .. } => Some(format!("(thinking: {})", thinking)),
            Content::ToolUse { name, input, .. } => {
                Some(format!("(tool_use: {} with {})", name, input))
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build Vercel request
fn build_request(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<(String, VercelRequest)> {
    let messages = convert_messages(&context.messages);
    if messages.is_empty() {
        return Err(anyhow!("No messages provided"));
    }

    let tools = context.tools.as_ref().map(|tool_schemas| {
        tool_schemas
            .iter()
            .map(|schema| VercelTool {
                tool_type: "function".to_string(),
                function: VercelFunction {
                    name: schema.name.clone(),
                    description: schema.description.clone(),
                    parameters: schema.input_schema.clone(),
                },
            })
            .collect()
    });

    let request = VercelRequest {
        model: model.id.clone(),
        messages,
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        tools,
        stream: true,
    };

    let url = "https://api.vercel.ai/v1/chat/completions".to_string();
    Ok((url, request))
}

/// Convert SSE line to events
fn convert_event(data: &str, _model_id: &str) -> Result<StreamEvent> {
    if data == "[DONE]" {
        return Ok(StreamEvent::Stop {
            stop_reason: "stop".to_string(),
            stop_sequence: None,
        });
    }

    let response: VercelResponse = serde_json::from_str(data)?;

    if let Some(choice) = response.choices.first() {
        if let Some(delta) = &choice.delta {
            if let Some(content) = &delta.content {
                return Ok(StreamEvent::TextDelta {
                    delta: content.clone(),
                    thinking: None,
                });
            }

            if let Some(tool_calls) = &delta.tool_calls {
                if let Some(tc) = tool_calls.first() {
                    let input = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                        .unwrap_or(json!({}));
                    return Ok(StreamEvent::ToolCallDelta {
                        delta: ToolCallDelta {
                            id: Some(tc.id.clone()),
                            name: Some(tc.function.name.clone()),
                            input: Some(input),
                        },
                    });
                }
            }
        }

        if let Some(reason) = &choice.finish_reason {
            return Ok(StreamEvent::Stop {
                stop_reason: reason.clone(),
                stop_sequence: None,
            });
        }
    }

    if let Some(usage) = response.usage {
        return Ok(StreamEvent::Usage {
            usage: Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cache_read_tokens: None,
                cache_write_tokens: None,
                total_tokens: Some(usage.total_tokens),
                cost: None,
            },
        });
    }

    Ok(StreamEvent::TextDelta {
        delta: String::new(),
        thinking: None,
    })
}

/// Stream from Vercel AI Gateway
pub async fn stream(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<crate::types::stream::AssistantMessageEventStream> {
    let (url, request) = build_request(model, context, options)?;

    let api_key = options.api_key.as_ref()
        .ok_or_else(|| anyhow!("Vercel AI Gateway API key required"))?;

    let client = Client::new();
    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await?;
        return Err(anyhow!("Vercel AI Gateway error {}: {}", status, text));
    }

    // Byte stream with error conversion for StreamReader
    let byte_stream = response.bytes_stream().map(|result| {
        result.map_err(|e| std::io::Error::new(ErrorKind::Other, e))
    });
    let stream_reader = StreamReader::new(byte_stream);
    let buf_reader = tokio::io::BufReader::new(stream_reader);

    // Decode SSE
    let sse_stream = crate::stream::decode_sse(buf_reader);

    // Convert to StreamEvent stream
    let model_id = model.id.clone();
    let mapped = sse_stream.map(move |result| {
        match result {
            Ok(sse) => {
                convert_event(&sse.data, &model_id)
            }
            Err(e) => Err(anyhow!("SSE decode error: {}", e)),
        }
    }).boxed();

    Ok(Box::pin(mapped) as crate::types::stream::AssistantMessageEventStream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages() {
        let msg = Message::User(vec![Content::Text {
            text: "Hello".to_string(),
            cache_control: None,
        }]);

        let messages = convert_messages(&[msg]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn test_convert_event_done() {
        let event = convert_event("[DONE]", "test").unwrap();
        match event {
            StreamEvent::Stop { .. } => {}
            _ => panic!("Expected Stop"),
        }
    }

    #[tokio::test]
    async fn test_build_request() {
        let model = Model {
            id: "gpt-4".to_string(),
            name: "GPT-4".to_string(),
            api: Api::VercelAiGateway,
            provider: Provider::VercelAiGateway,
            base_url: None,
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };

        let context = Context {
            system_prompt: None,
            messages: vec![Message::User(vec![Content::Text {
                text: "test".to_string(),
                cache_control: None,
            }])],
            tools: None,
        };

        let options = StreamOptions {
            temperature: Some(0.7),
            max_tokens: Some(1024),
            signal: None,
            api_key: Some("test_key".to_string()),
            transport: None,
            cache_retention: None,
            session_id: None,
            headers: None,
            reasoning_effort: None,
            thinking_budgets: None,
        };

        let result = build_request(&model, &context, &options);
        assert!(result.is_ok());
        let (url, _) = result.unwrap();
        assert!(url.contains("vercel.ai"));
    }
}
