//! Probe an OpenAI-wire-compatible provider using the exact code path the
//! agent uses. Prints every `StreamEvent` that comes out of the converter so
//! we can verify text deltas are captured and Stop arrives.
//!
//! Run:
//!
//! ```text
//! OPENROUTER_API_KEY=... \
//!   cargo run -p pi-ai --example probe_openai_stream -- \
//!   stepfun/step-3.5-flash https://openrouter.ai/api/v1 512
//! ```

use futures::StreamExt;
use pi_ai::providers::openai;
use pi_ai::types::{Api, Content, Context, Message, Model, Provider, StreamEvent, StreamOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "stepfun/step-3.5-flash".to_string());
    let base_url = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());
    let max_tokens: u32 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let api_key = std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .map_err(|_| {
            anyhow::anyhow!("set OPENROUTER_API_KEY (or OPENAI_API_KEY) before running")
        })?;

    let model = Model {
        id: model_id.clone(),
        name: model_id.clone(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenRouter,
        base_url: Some(base_url.clone()),
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
            text: "Reply with exactly: Hello world!".into(),
            cache_control: None,
        }])],
        tools: None,
    };

    let options = StreamOptions {
        temperature: Some(0.7),
        max_tokens: Some(max_tokens),
        signal: None,
        api_key: Some(api_key),
        transport: None,
        cache_retention: None,
        session_id: None,
        headers: None,
        reasoning_effort: None,
        thinking_budgets: None,
    };

    eprintln!("probe: model={} base_url={} max_tokens={}", model_id, base_url, max_tokens);

    let mut stream = openai::stream(&model, &context, &options).await?;

    let mut accumulated_text = String::new();
    let mut accumulated_thinking = String::new();
    let mut event_count = 0usize;
    let mut saw_stop = false;

    while let Some(result) = stream.next().await {
        event_count += 1;
        match result {
            Ok(event) => match &event {
                StreamEvent::TextDelta { delta, .. } => {
                    accumulated_text.push_str(delta);
                    eprintln!("  TextDelta({:?})", delta);
                }
                StreamEvent::ThinkingDelta { delta, .. } => {
                    accumulated_thinking.push_str(delta);
                    eprintln!("  ThinkingDelta({:?})", delta);
                }
                StreamEvent::ToolCallDelta { delta } => {
                    eprintln!("  ToolCallDelta(name={:?}, input={:?})", delta.name, delta.input);
                }
                StreamEvent::Usage { usage } => {
                    eprintln!(
                        "  Usage(in={}, out={}, total={:?})",
                        usage.input_tokens, usage.output_tokens, usage.total_tokens
                    );
                }
                StreamEvent::Stop { stop_reason, .. } => {
                    eprintln!("  Stop(reason={})", stop_reason);
                    saw_stop = true;
                }
                other => eprintln!("  other: {:?}", other),
            },
            Err(e) => {
                eprintln!("  ERROR: {}", e);
                return Err(e);
            }
        }
    }

    eprintln!();
    eprintln!("=== SUMMARY ===");
    eprintln!("events:      {}", event_count);
    eprintln!("saw_stop:    {}", saw_stop);
    eprintln!("text_len:    {}", accumulated_text.len());
    eprintln!("thinking_len:{}", accumulated_thinking.len());
    eprintln!();
    eprintln!("=== TEXT ===");
    println!("{}", accumulated_text);
    eprintln!();
    eprintln!("=== THINKING ===");
    eprintln!("{}", accumulated_thinking);

    Ok(())
}
