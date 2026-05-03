//! Amazon Bedrock Converse provider.
//!
//! This targets the **non-streaming** `/model/{id}/converse` endpoint and
//! synthesizes a [`StreamEvent`] sequence from the JSON response. That
//! avoids the binary `application/vnd.amazon.eventstream` wire format that
//! `/converse-stream` uses; our SSE decoder cannot parse it and we don't
//! want to ship a separate event-stream framer just for Bedrock.
//!
//! Approach adapted from the Dicklesworthstone/pi_agent_rust reference at
//! <https://github.com/Dicklesworthstone/pi_agent_rust/blob/main/src/providers/bedrock.rs>,
//! translated to this crate's type system (`Message` / `Content` /
//! `StreamEvent`).
//!
//! Supported authentication (in priority order):
//!   1. `AWS_BEARER_TOKEN_BEDROCK` environment variable (simple and
//!      recommended for per-project credentials).
//!   2. AWS SigV4 via `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
//!      (optionally `AWS_SESSION_TOKEN`). Uses the existing
//!      [`crate::providers::sigv4`] helper for signing.

use crate::providers::sigv4;
use crate::types::*;
use anyhow::{Result, anyhow};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Bedrock Converse request body. Strict schema: `modelId` lives in the URL
/// path (not the body), `maxTokens`/`temperature` go under
/// `inferenceConfig`, and `tools` go under `toolConfig`. Any extraneous
/// key produces `400 Malformed input request: extraneous key [X]`.
#[derive(Debug, Serialize, Default)]
struct BedrockConverseRequest {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<BedrockSystemContent>,
    messages: Vec<BedrockMessage>,
    #[serde(rename = "inferenceConfig", skip_serializing_if = "Option::is_none")]
    inference_config: Option<BedrockInferenceConfig>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    tool_config: Option<BedrockToolConfig>,
}

#[derive(Debug, Serialize)]
struct BedrockSystemContent {
    text: String,
}

#[derive(Debug, Serialize)]
struct BedrockMessage {
    role: &'static str,
    content: Vec<BedrockContent>,
}

/// Bedrock content block. The wire shape is an object with exactly one of
/// `text`, `image`, `toolUse`, or `toolResult` as the single key (no
/// `type` discriminator). `#[serde(untagged)]` gives us that directly.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum BedrockContent {
    Text {
        text: String,
    },
    ToolUse {
        #[serde(rename = "toolUse")]
        tool_use: BedrockToolUseBlock,
    },
    ToolResult {
        #[serde(rename = "toolResult")]
        tool_result: BedrockToolResultBlock,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BedrockToolUseBlock {
    tool_use_id: String,
    name: String,
    input: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BedrockToolResultBlock {
    tool_use_id: String,
    content: Vec<BedrockToolResultContent>,
    status: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum BedrockToolResultContent {
    Text { text: String },
}

#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct BedrockInferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct BedrockToolConfig {
    tools: Vec<BedrockToolDef>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BedrockToolDef {
    tool_spec: BedrockToolSpec,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BedrockToolSpec {
    name: String,
    description: String,
    /// Converse requires `inputSchema.json = <schema>` (a nested object),
    /// not the schema directly. Anyone who forgets the wrapper hits
    /// `Malformed input request: required key [inputSchema.json] is missing`.
    input_schema: BedrockInputSchema,
}

#[derive(Debug, Serialize)]
struct BedrockInputSchema {
    json: Value,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BedrockConverseResponse {
    #[serde(default)]
    output: Option<BedrockResponseOutput>,
    #[serde(default, rename = "stopReason")]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<BedrockUsage>,
}

#[derive(Debug, Deserialize)]
struct BedrockResponseOutput {
    message: BedrockResponseMessage,
}

#[derive(Debug, Deserialize)]
struct BedrockResponseMessage {
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    content: Vec<BedrockResponseContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BedrockResponseContent {
    Text {
        text: String,
    },
    ToolUse {
        #[serde(rename = "toolUse")]
        tool_use: BedrockResponseToolUse,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockResponseToolUse {
    tool_use_id: String,
    name: String,
    #[serde(default)]
    input: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

fn convert_messages(messages: &[Message]) -> Vec<BedrockMessage> {
    let mut out = Vec::new();
    for msg in messages {
        match msg {
            Message::User(blocks) => {
                let content = user_blocks_to_bedrock(blocks);
                if !content.is_empty() {
                    out.push(BedrockMessage {
                        role: "user",
                        content,
                    });
                }
            }
            Message::Assistant(blocks) => {
                let content = assistant_blocks_to_bedrock(blocks);
                if !content.is_empty() {
                    out.push(BedrockMessage {
                        role: "assistant",
                        content,
                    });
                }
            }
            Message::Tool {
                tool_use_id,
                content,
                is_error,
            } => {
                let text = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                // Bedrock expects tool results as a user-role message with a
                // toolResult content block linked back to the original
                // toolUseId. Status is "success" by default; "error" when
                // the caller flagged the result as an error.
                out.push(BedrockMessage {
                    role: "user",
                    content: vec![BedrockContent::ToolResult {
                        tool_result: BedrockToolResultBlock {
                            tool_use_id: tool_use_id.clone(),
                            content: vec![BedrockToolResultContent::Text { text }],
                            status: if is_error.unwrap_or(false) { "error" } else { "success" },
                        },
                    }],
                });
            }
        }
    }
    out
}

fn user_blocks_to_bedrock(blocks: &[Content]) -> Vec<BedrockContent> {
    let mut out = Vec::new();
    for block in blocks {
        match block {
            Content::Text { text, .. } if !text.trim().is_empty() => {
                out.push(BedrockContent::Text { text: text.clone() });
            }
            // Multimodal blocks could be wired here (Bedrock supports image
            // + document content blocks on Converse). For now we stringify
            // them so the model at least sees there was a reference.
            Content::Image { .. } | Content::Audio { .. } | Content::Video { .. } | Content::Pdf { .. } => {
                out.push(BedrockContent::Text {
                    text: "[attachment omitted: Bedrock multimodal wiring TODO]".to_string(),
                });
            }
            _ => {}
        }
    }
    out
}

fn assistant_blocks_to_bedrock(blocks: &[Content]) -> Vec<BedrockContent> {
    let mut out = Vec::new();
    for block in blocks {
        match block {
            Content::Text { text, .. } if !text.trim().is_empty() => {
                out.push(BedrockContent::Text { text: text.clone() });
            }
            Content::ToolUse {
                id, name, input, ..
            } => {
                out.push(BedrockContent::ToolUse {
                    tool_use: BedrockToolUseBlock {
                        tool_use_id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    },
                });
            }
            // Thinking blocks are internal to Anthropic; Bedrock Converse
            // has no equivalent slot in the assistant content array.
            _ => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Build request body + URL
// ---------------------------------------------------------------------------

fn region_from_env() -> String {
    std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string())
}

fn build_request(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<(String, Vec<u8>)> {
    let messages = convert_messages(&context.messages);
    if messages.is_empty() {
        return Err(anyhow!("No messages provided"));
    }

    let system = context
        .system_prompt
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            vec![BedrockSystemContent {
                text: s.to_string(),
            }]
        })
        .unwrap_or_default();

    let inference_config = if options.max_tokens.is_some() || options.temperature.is_some() {
        Some(BedrockInferenceConfig {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
        })
    } else {
        None
    };

    let tool_config = context.tools.as_ref().map(|tool_schemas| BedrockToolConfig {
        tools: tool_schemas
            .iter()
            .map(|schema| BedrockToolDef {
                tool_spec: BedrockToolSpec {
                    name: schema.name.clone(),
                    description: schema.description.clone(),
                    input_schema: BedrockInputSchema {
                        json: schema.input_schema.clone(),
                    },
                },
            })
            .collect(),
    });

    let body = BedrockConverseRequest {
        system,
        messages,
        inference_config,
        tool_config,
    };

    // Default to the runtime endpoint in the request's region. The registry
    // may ship a `base_url` that is region-specific (`bedrock-runtime.us-east-1...`);
    // we honor it when present and otherwise compute one from the
    // region env.
    let region = region_from_env();
    let default_host = format!("https://bedrock-runtime.{}.amazonaws.com", region);
    let base = model
        .base_url
        .as_deref()
        .unwrap_or(&default_host)
        .trim_end_matches('/');
    let url = format!("{}/model/{}/converse-stream", base, model.id);

    let body_bytes = serde_json::to_vec(&body)?;
    Ok((url, body_bytes))
}

// ---------------------------------------------------------------------------
// Synthesize stream events from the JSON response
// ---------------------------------------------------------------------------

fn map_stop_reason(stop_reason: Option<&str>) -> String {
    // Agent loop only cares that Stop eventually arrives; the string is
    // surfaced in a final `StopReason` field on the assistant message.
    match stop_reason.unwrap_or("end_turn") {
        "tool_use" => "tool_use".to_string(),
        "max_tokens" => "length".to_string(),
        "guardrail_intervened" | "content_filtered" => "content_filter".to_string(),
        other => other.to_string(),
    }
}

fn response_to_events(
    response: BedrockConverseResponse,
    model_id: &str,
) -> Vec<Result<StreamEvent>> {
    let mut events: Vec<Result<StreamEvent>> = Vec::new();

    let usage = response.usage.as_ref().map(|u| {
        let total = if u.total_tokens > 0 {
            u.total_tokens
        } else {
            u.input_tokens + u.output_tokens
        };
        Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
            total_tokens: Some(total),
            cost: None,
        }
    });

    events.push(Ok(StreamEvent::Start {
        model: model_id.to_string(),
        usage: usage.clone(),
    }));

    if let Some(output) = response.output {
        for block in output.message.content {
            match block {
                BedrockResponseContent::Text { text } => {
                    if !text.is_empty() {
                        events.push(Ok(StreamEvent::TextDelta {
                            delta: text,
                            thinking: None,
                        }));
                    }
                }
                BedrockResponseContent::ToolUse { tool_use } => {
                    events.push(Ok(StreamEvent::ToolCallDelta {
                        delta: ToolCallDelta {
                            id: Some(tool_use.tool_use_id),
                            name: Some(tool_use.name),
                            input: Some(tool_use.input),
                        },
                    }));
                }
            }
        }
    }

    if let Some(u) = usage {
        events.push(Ok(StreamEvent::Usage { usage: u }));
    }

    events.push(Ok(StreamEvent::Stop {
        stop_reason: map_stop_reason(response.stop_reason.as_deref()),
        stop_sequence: None,
    }));

    events
}

// ---------------------------------------------------------------------------
// HTTP call
// ---------------------------------------------------------------------------

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<crate::types::stream::AssistantMessageEventStream> {
    // Retry loop: some Bedrock-hosted models (notably Claude Opus 4.5+
    // and Sonnet 4.6+) reject the `temperature` parameter with a
    // 400 "temperature is deprecated for this model" error. Rather than
    // maintaining an ever-growing list of models that forbid the param,
    // we catch the first 400 that names it and retry once without the
    // parameter. Registry-side we already drop temperature for reasoning
    // models, so this is a belt-and-suspenders safety net.
    let mut current_options = options.clone();
    let response = loop {
        let (url, body) = build_request(model, context, &current_options)?;
        let client = Client::new();

        let bearer = std::env::var("AWS_BEARER_TOKEN_BEDROCK")
            .ok()
            .or_else(|| current_options.api_key.clone());

        let response = if let Some(token) = bearer {
            client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .body(body)
                .send()
                .await?
        } else {
            // SigV4 path. Reuse the in-crate signer.
            let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| {
                anyhow!("Bedrock auth missing: set AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID")
            })?;
            let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
                .map_err(|_| anyhow!("Bedrock auth missing: AWS_SECRET_ACCESS_KEY not set"))?;
            let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
            let region = region_from_env();

            let body_str = std::str::from_utf8(&body)
                .map_err(|e| anyhow!("Bedrock request body is not UTF-8: {e}"))?;
            let host = url::Url::parse(&url)
                .ok()
                .and_then(|u| u.host_str().map(ToString::to_string))
                .ok_or_else(|| anyhow!("could not parse Bedrock URL host: {url}"))?;

            let signing_headers: Vec<(String, String)> = vec![
                ("host".to_string(), host.clone()),
                ("content-type".to_string(), "application/json".to_string()),
                ("accept".to_string(), "application/json".to_string()),
            ];

            let params = sigv4::SigV4Params {
                access_key,
                secret_key,
                session_token,
                region,
                service: "bedrock".to_string(),
            };

            let signed = sigv4::sign_request("POST", &url, &signing_headers, body_str, &params)?;

            let mut rb = client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .header("X-Amz-Date", signed.x_amz_date)
                .header("Authorization", signed.authorization_header);
            if let Some(token) = signed.x_amz_security_token {
                rb = rb.header("X-Amz-Security-Token", token);
            }
            rb.body(body).send().await?
        };

        if response.status().is_success() {
            break response;
        }

        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        let temperature_rejected = status.as_u16() == 400
            && current_options.temperature.is_some()
            && text.to_lowercase().contains("'temperature'");
        if temperature_rejected {
            current_options.temperature = None;
            continue;
        }
        return Err(anyhow!("Bedrock API error {status}: {text}"));
    };

    // `/converse-stream` returns
    // `application/vnd.amazon.eventstream` — a binary frame format
    // where each event is a message with typed headers and a JSON
    // payload. Decode frames incrementally and map them to
    // `StreamEvent`s as they arrive so the UI renders tokens in real
    // time.
    let byte_stream = response.bytes_stream();
    let model_id = model.id.clone();
    let event_stream = build_bedrock_event_stream(byte_stream, model_id);
    Ok(Box::pin(event_stream) as crate::types::stream::AssistantMessageEventStream)
}

/// Incremental adapter from a byte-stream of event-stream frames to
/// `StreamEvent`s. Handles the Bedrock-specific set of event types:
///
///   * `messageStart`       -> emits `StreamEvent::Start`
///   * `contentBlockStart`  -> records a new content block (text, tool_use)
///   * `contentBlockDelta`  -> emits `TextDelta` or accumulates toolUse input
///   * `contentBlockStop`   -> flushes the tool-use delta if any
///   * `messageStop`        -> emits `StreamEvent::Stop`
///   * `metadata`           -> emits `StreamEvent::Usage`
///
/// Bedrock streams tool `input` as incremental JSON snippets on each
/// `contentBlockDelta`. We accumulate them into a single string per
/// block index and parse once on `contentBlockStop`, then emit a
/// `ToolCallDelta` that downstream agents consume.
fn build_bedrock_event_stream(
    byte_stream: impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
    model_id: String,
) -> impl futures::Stream<Item = Result<StreamEvent>> + Send {
    use crate::providers::eventstream::EventStreamDecoder;
    use futures::stream;
    use std::collections::HashMap;

    struct State {
        decoder: EventStreamDecoder,
        model_id: String,
        // block_index -> (tool_use_id, name, accumulated_json)
        tool_blocks: HashMap<u32, (String, String, String)>,
        /// Bedrock emits `messageStop` BEFORE `metadata` (which
        /// carries usage). The agent loop breaks on `Stop`, so
        /// we buffer the stop reason here and flush it AFTER the
        /// usage event is emitted (or at stream end). This makes
        /// the event order observed downstream: ...deltas...,
        /// Usage, Stop — matching Anthropic/OpenAI semantics and
        /// ensuring the TUI's `AgentEvent::Usage` handler runs
        /// before the turn completes.
        pending_stop: Option<String>,
    }
    let state = State {
        decoder: EventStreamDecoder::new(),
        model_id,
        tool_blocks: HashMap::new(),
        pending_stop: None,
    };

    // Pin the byte stream before handing it to `unfold` so it can be
    // polled across await points inside the closure.
    let byte_stream = Box::pin(byte_stream);

    stream::unfold((byte_stream, state, Vec::<StreamEvent>::new(), false), move |(
        mut byte_stream,
        mut state,
        mut pending,
        mut done,
    )| async move {
        loop {
            if let Some(ev) = pending.pop() {
                return Some((Ok(ev), (byte_stream, state, pending, done)));
            }
            if done {
                return None;
            }
            // Pull the next complete frame from the decoder; if the
            // buffer doesn't have one yet, read more bytes from the
            // network.
            match state.decoder.next_message() {
                Err(e) => return Some((Err(e), (byte_stream, state, pending, true))),
                Ok(Some(msg)) => {
                    match decode_bedrock_message(
                        &msg,
                        &mut state.tool_blocks,
                        &mut state.pending_stop,
                        &state.model_id,
                    ) {
                        Ok(events) => {
                            // Reverse so popping yields in order.
                            pending.extend(events.into_iter().rev());
                            continue;
                        }
                        Err(e) => {
                            return Some((Err(e), (byte_stream, state, pending, true)));
                        }
                    }
                }
                Ok(None) => {
                    use futures::StreamExt;
                    match byte_stream.next().await {
                        Some(Ok(chunk)) => {
                            state.decoder.push(&chunk);
                            continue;
                        }
                        Some(Err(e)) => {
                            return Some((
                                Err(anyhow!("Bedrock body stream error: {e}")),
                                (byte_stream, state, pending, true),
                            ));
                        }
                        None => {
                            // Stream ended. If we buffered a Stop
                            // while waiting for a metadata event that
                            // never arrived, flush it now so the
                            // agent loop terminates cleanly.
                            if let Some(reason) = state.pending_stop.take() {
                                return Some((
                                    Ok(StreamEvent::Stop {
                                        stop_reason: reason,
                                        stop_sequence: None,
                                    }),
                                    (byte_stream, state, pending, true),
                                ));
                            }
                            done = true;
                            continue;
                        }
                    }
                }
            }
        }
    })
}

fn decode_bedrock_message(
    msg: &crate::providers::eventstream::EventStreamMessage,
    tool_blocks: &mut std::collections::HashMap<u32, (String, String, String)>,
    pending_stop: &mut Option<String>,
    model_id: &str,
) -> Result<Vec<StreamEvent>> {
    let Some(event_type) = msg.event_type() else {
        return Ok(Vec::new());
    };
    // Some event types that Bedrock returns indicate the overall
    // model errored (throttling, auth, etc.). The payload is a JSON
    // object with a `message` field. Surface as an Error event so
    // the agent loop stops cleanly.
    match event_type {
        "messageStart" => Ok(vec![StreamEvent::Start {
            model: model_id.to_string(),
            usage: None,
        }]),
        "contentBlockStart" => {
            // JSON: { contentBlockIndex, start: { toolUse: { toolUseId, name } | {} } }
            let v: serde_json::Value = serde_json::from_slice(&msg.payload)
                .map_err(|e| anyhow!("decode contentBlockStart: {e}"))?;
            let idx = v.get("contentBlockIndex").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            if let Some(tu) = v.get("start").and_then(|s| s.get("toolUse")) {
                let id = tu.get("toolUseId").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let name = tu.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
                tool_blocks.insert(idx, (id, name, String::new()));
            }
            Ok(Vec::new())
        }
        "contentBlockDelta" => {
            // JSON: { contentBlockIndex, delta: { text: "..." } | { toolUse: { input: "<partial json>" } } }
            let v: serde_json::Value = serde_json::from_slice(&msg.payload)
                .map_err(|e| anyhow!("decode contentBlockDelta: {e}"))?;
            let idx = v.get("contentBlockIndex").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let delta = v.get("delta").cloned().unwrap_or(serde_json::Value::Null);
            if let Some(text) = delta.get("text").and_then(|x| x.as_str()) {
                if !text.is_empty() {
                    return Ok(vec![StreamEvent::TextDelta {
                        delta: text.to_string(),
                        thinking: None,
                    }]);
                }
            }
            if let Some(tu) = delta.get("toolUse") {
                if let Some(snippet) = tu.get("input").and_then(|x| x.as_str()) {
                    if let Some((_, _, buf)) = tool_blocks.get_mut(&idx) {
                        buf.push_str(snippet);
                    }
                }
            }
            Ok(Vec::new())
        }
        "contentBlockStop" => {
            let v: serde_json::Value = serde_json::from_slice(&msg.payload)
                .map_err(|e| anyhow!("decode contentBlockStop: {e}"))?;
            let idx = v.get("contentBlockIndex").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            // If this block was a tool_use, flush the accumulated input.
            if let Some((id, name, buf)) = tool_blocks.remove(&idx) {
                let input: serde_json::Value = if buf.trim().is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&buf).unwrap_or_else(|_| serde_json::json!({}))
                };
                return Ok(vec![StreamEvent::ToolCallDelta {
                    delta: crate::types::ToolCallDelta {
                        id: Some(id),
                        name: Some(name),
                        input: Some(input),
                    },
                }]);
            }
            Ok(Vec::new())
        }
        "messageStop" => {
            // Buffer the stop reason and wait for the metadata
            // event (usage) before emitting it. The agent loop
            // breaks on `Stop`, so emitting Stop here would hide
            // the usage event that Bedrock sends AFTER this one.
            let v: serde_json::Value = serde_json::from_slice(&msg.payload)
                .unwrap_or_else(|_| serde_json::json!({}));
            let stop_reason = v
                .get("stopReason")
                .and_then(|x| x.as_str())
                .unwrap_or("end_turn")
                .to_string();
            *pending_stop = Some(stop_reason);
            Ok(Vec::new())
        }
        "metadata" => {
            let v: serde_json::Value = serde_json::from_slice(&msg.payload)
                .map_err(|e| anyhow!("decode metadata: {e}"))?;
            let usage = v.get("usage").cloned().unwrap_or(serde_json::Value::Null);
            let input = usage.get("inputTokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let output = usage.get("outputTokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let total = usage.get("totalTokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let mut events = vec![StreamEvent::Usage {
                usage: crate::types::Usage {
                    input_tokens: input,
                    output_tokens: output,
                    total_tokens: if total > 0 { Some(total) } else { None },
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    cost: None,
                },
            }];
            // Flush the buffered Stop now that usage has been
            // delivered to the consumer.
            if let Some(reason) = pending_stop.take() {
                events.push(StreamEvent::Stop {
                    stop_reason: reason,
                    stop_sequence: None,
                });
            }
            Ok(events)
        }
        // Error / exception events. Bedrock wraps these in specific
        // event-types like "validationException", "throttlingException",
        // "modelStreamErrorException", etc. Surface as Error events.
        et if et.ends_with("Exception") || et == "internalServerException" => {
            let v: serde_json::Value = serde_json::from_slice(&msg.payload)
                .unwrap_or_else(|_| serde_json::json!({}));
            let message = v
                .get("message")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            Ok(vec![StreamEvent::Error {
                error: format!("Bedrock {et}: {message}"),
                code: None,
            }])
        }
        _ => Ok(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_text_content_has_no_type_discriminator() {
        let msg = BedrockMessage {
            role: "user",
            content: vec![BedrockContent::Text {
                text: "hi".into(),
            }],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""text":"hi""#), "got {json}");
        assert!(!json.contains(r#""type""#), "got {json}");
    }

    #[test]
    fn serialize_tool_use_content_uses_camel_case_tool_use() {
        let msg = BedrockMessage {
            role: "assistant",
            content: vec![BedrockContent::ToolUse {
                tool_use: BedrockToolUseBlock {
                    tool_use_id: "call_abc".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "pwd"}),
                },
            }],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""toolUse""#), "got {json}");
        assert!(json.contains(r#""toolUseId":"call_abc""#), "got {json}");
        assert!(json.contains(r#""name":"bash""#), "got {json}");
    }

    #[test]
    fn serialize_tool_result_links_by_id_and_uses_success_status() {
        let msg = BedrockMessage {
            role: "user",
            content: vec![BedrockContent::ToolResult {
                tool_result: BedrockToolResultBlock {
                    tool_use_id: "call_abc".into(),
                    content: vec![BedrockToolResultContent::Text {
                        text: "/tmp\n".into(),
                    }],
                    status: "success",
                },
            }],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""toolResult""#), "got {json}");
        assert!(json.contains(r#""toolUseId":"call_abc""#), "got {json}");
        assert!(json.contains(r#""status":"success""#), "got {json}");
    }

    #[test]
    fn request_body_has_expected_top_level_keys_only() {
        // Regression: original shipped `modelId` / `maxTokens` / `tools` at
        // the top level, which Bedrock rejects with "extraneous key [X]".
        let ctx = Context {
            system_prompt: Some("you are helpful".into()),
            messages: vec![Message::User(vec![Content::Text {
                text: "hello".into(),
                cache_control: None,
            }])],
            tools: None,
        };
        let options = StreamOptions {
            temperature: Some(0.7),
            max_tokens: Some(128),
            signal: None,
            api_key: None,
            transport: None,
            cache_retention: None,
            session_id: None,
            headers: None,
            reasoning_effort: None,
            thinking_budgets: None,
        };
        let model = Model {
            id: "us.anthropic.claude-haiku-4-5-20251001-v1:0".into(),
            name: "Claude Haiku".into(),
            api: Api::BedrockConverseStream,
            provider: Provider::AmazonBedrock,
            base_url: Some("https://bedrock-runtime.us-east-1.amazonaws.com".into()),
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };

        let (url, body) = build_request(&model, &ctx, &options).unwrap();
        assert!(
            url.ends_with("/converse-stream"),
            "url should use /converse-stream endpoint, got {url}"
        );
        assert!(
            url.contains("/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/"),
            "model id should be URL-encoded into the path, got {url}"
        );

        let parsed: Value = serde_json::from_slice(&body).unwrap();
        // Allowed keys only. `modelId`, `tools` at the top level, `maxTokens`
        // at the top level \u2014 all forbidden.
        let obj = parsed.as_object().unwrap();
        let keys: Vec<&String> = obj.keys().collect();
        for forbidden in ["modelId", "maxTokens", "tools", "temperature"] {
            assert!(
                !keys.iter().any(|k| k.as_str() == forbidden),
                "top-level `{forbidden}` is forbidden; got keys: {keys:?}"
            );
        }
        assert!(obj.contains_key("messages"));
        assert!(obj.contains_key("system"));
        assert!(obj.contains_key("inferenceConfig"));
        let ic = &obj["inferenceConfig"];
        assert_eq!(ic["maxTokens"], 128);
        assert_eq!(ic["temperature"], 0.7);
    }

    #[test]
    fn response_to_events_maps_text_and_tool_use() {
        let response: BedrockConverseResponse = serde_json::from_value(serde_json::json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Running it."},
                        {"toolUse": {"toolUseId": "call_1", "name": "bash", "input": {"command": "pwd"}}}
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 12, "outputTokens": 8, "totalTokens": 20}
        })).unwrap();

        let events = response_to_events(response, "test-model");
        let text = events
            .iter()
            .filter_map(|e| match e.as_ref().ok() {
                Some(StreamEvent::TextDelta { delta, .. }) => Some(delta.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "Running it.");

        let tool = events
            .iter()
            .find_map(|e| match e.as_ref().ok() {
                Some(StreamEvent::ToolCallDelta { delta }) => Some(delta.clone()),
                _ => None,
            })
            .expect("tool call");
        assert_eq!(tool.id.as_deref(), Some("call_1"));
        assert_eq!(tool.name.as_deref(), Some("bash"));
        assert_eq!(tool.input.as_ref().unwrap()["command"], "pwd");

        let saw_stop = events
            .iter()
            .any(|e| matches!(e.as_ref().ok(), Some(StreamEvent::Stop { .. })));
        assert!(saw_stop, "must emit a Stop event");
    }
}
