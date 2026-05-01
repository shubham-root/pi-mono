//! Dump raw HTTP SSE bytes from OpenRouter, byte-for-byte, so we can see the
//! exact line separators. Used to diagnose why the SSE decoder concatenates
//! events.

use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "stepfun/step-3.5-flash",
        "messages": [{"role": "user", "content": "Reply with exactly: Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 256
    });

    let response = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    eprintln!("status: {}", response.status());
    eprintln!("--- body (escaped) ---");

    let mut bytes_stream = response.bytes_stream();
    let mut chunk_idx = 0;
    while let Some(chunk) = bytes_stream.next().await {
        let chunk = chunk?;
        chunk_idx += 1;
        eprintln!("chunk #{chunk_idx} len={}", chunk.len());
        // Escape control chars so we can see CR/LF/blank-line patterns.
        let escaped: String = chunk
            .iter()
            .map(|&b| match b {
                b'\r' => "\\r".to_string(),
                b'\n' => "\\n\n".to_string(),
                b'\t' => "\\t".to_string(),
                0x20..=0x7e => (b as char).to_string(),
                _ => format!("\\x{:02x}", b),
            })
            .collect();
        eprintln!("{escaped}");
        eprintln!("--- end chunk ---");
        if chunk_idx >= 6 {
            break;
        }
    }

    Ok(())
}
