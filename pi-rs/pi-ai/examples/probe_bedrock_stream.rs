//! Probe the Bedrock /converse-stream endpoint and dump every decoded
//! event-stream frame's headers + payload so we can verify the exact
//! `:event-type` names the service sends on the wire.
//!
//! ```text
//! AWS_BEARER_TOKEN_BEDROCK=... \
//!   cargo run -p pi-ai --example probe_bedrock_stream -- \
//!   us.anthropic.claude-haiku-4-5-20251001-v1:0
//! ```

use futures::StreamExt;
use pi_ai::providers::eventstream::EventStreamDecoder;
use reqwest::Client;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string());

    let region =
        std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-2".to_string());
    let url = format!(
        "https://bedrock-runtime.{region}.amazonaws.com/model/{}/converse-stream",
        model_id
    );

    let body = serde_json::json!({
        "messages": [{"role":"user","content":[{"text":"say hi briefly"}]}],
        "inferenceConfig": {"maxTokens": 64}
    });

    let token = std::env::var("AWS_BEARER_TOKEN_BEDROCK")?;
    let resp = Client::new()
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .body(serde_json::to_vec(&body)?)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        eprintln!("{status}: {text}");
        std::process::exit(1);
    }

    eprintln!("content-type: {:?}", resp.headers().get("content-type"));
    let mut stream = resp.bytes_stream();
    let mut decoder = EventStreamDecoder::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        decoder.push(&chunk);
        while let Some(msg) = decoder.next_message()? {
            eprintln!("=== frame ===");
            for (k, v) in &msg.headers {
                eprintln!("  hdr {k} = {v}");
            }
            let body = std::str::from_utf8(&msg.payload).unwrap_or("<non-utf8>");
            eprintln!("  payload: {body}");
        }
    }

    Ok(())
}
