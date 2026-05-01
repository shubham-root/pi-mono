//! SSE (Server-Sent Events) parser for streaming LLM responses.
//! Generic SSE parser supporting multi-line data fields.

use anyhow::Result;
use futures::stream::{Stream, StreamExt};
use tokio::io::AsyncBufReadExt;
use tokio::sync::mpsc;
use bytes::Bytes;

/// A server-sent event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event type (from `event:` field), or None for default.
    pub event: Option<String>,
    /// Data lines concatenated.
    pub data: String,
    /// Raw lines (for debugging).
    pub raw: Vec<String>,
}

impl SseEvent {
    /// Parse the data field as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        if self.data.is_empty() {
            return Err(anyhow::anyhow!("empty data"));
        }
        Ok(serde_json::from_str(&self.data)?)
    }
}

/// State machine for SSE decoding.
#[derive(Default)]
struct SseDecoder {
    event: Option<String>,
    data: Vec<String>,
    raw: Vec<String>,
}

impl SseDecoder {
    fn new() -> Self {
        Self::default()
    }

    fn feed_line(&mut self, line: &str) -> Option<SseEvent> {
        self.raw.push(line.to_string());

        if line.starts_with(':') || line.is_empty() {
            // Comment or blank line, ignore
            return None;
        }

        let delimiter_idx = line.find(':').unwrap_or(line.len());
        let field = &line[..delimiter_idx];
        let value = if delimiter_idx < line.len() {
            let v = line[delimiter_idx + 1..].trim_start();
            v
        } else {
            ""
        };

        match field {
            "event" => {
                self.event = Some(value.to_string());
            }
            "data" => {
                self.data.push(value.to_string());
            }
            _ => {
                // ignore other fields
            }
        }

        None
    }

    fn flush(&mut self) -> Option<SseEvent> {
        if self.event.is_none() && self.data.is_empty() {
            return None;
        }
        // Consume the accumulated state. The previous implementation cloned
        // `self.data` via `.join("\n")` but never cleared it, so subsequent
        // events carried the data from every prior event concatenated —
        // producing doubled JSON payloads (e.g. two chunks glued together)
        // that the provider converter rejected with a trailing-characters
        // parse error. That was the root cause of OpenRouter/StepFun prompts
        // appearing to return empty text in the TUI.
        let event = SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
            raw: std::mem::take(&mut self.raw),
        };
        Some(event)
    }
}

/// Create a stream of SSE events from an async byte stream.
///
/// `reader` implements `AsyncBufRead` (e.g., `tokio::io::BufReader` over HTTP response).
/// Returns a `Stream` of `Result<SseEvent, anyhow::Error>`.
pub fn decode_sse<R>(mut reader: R) -> impl Stream<Item = Result<SseEvent>>
where
    R: tokio::io::AsyncBufRead + Unpin + Send + 'static,
{
    let (tx, rx) = mpsc::channel(64);

    tokio::spawn(async move {
        let mut decoder = SseDecoder::new();
        let mut line_buf = String::new();

        loop {
            line_buf.clear();
            match reader.read_line(&mut line_buf).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    // read_line appends to the String, includes newline
                    let trimmed = line_buf.trim_end();

                    if trimmed.is_empty() {
                        // Blank line flushes event
                        if let Some(event) = decoder.flush() {
                            let _ = tx.send(Ok(event)).await;
                        }
                    } else {
                        decoder.feed_line(trimmed);
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("SSE read error: {}", e))).await;
                    break;
                }
            }
        }

        // Flush any pending event at EOF
        if let Some(event) = decoder.flush() {
            let _ = tx.send(Ok(event)).await;
        }
    });

    tokio_stream::wrappers::ReceiverStream::new(rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_sse() {
        let input = "event: message_start\ndata: {\"type\":\"message_start\"}\n\ndata: {\"type\":\"content_block_delta\"}\n\n";
        let cursor = std::io::Cursor::new(input.as_bytes());
        let reader = tokio::io::BufReader::new(cursor);

        let mut stream = decode_sse(reader);
        let ev1 = stream.next().await.unwrap().unwrap();
        assert_eq!(ev1.event, Some("message_start".to_string()));
        // Strict: data must contain only event 1's payload, not leaked bytes.
        assert_eq!(ev1.data, r#"{"type":"message_start"}"#);

        let ev2 = stream.next().await.unwrap().unwrap();
        assert_eq!(ev2.event, None);
        assert_eq!(ev2.data, r#"{"type":"content_block_delta"}"#);
    }

    /// Regression: previously `flush()` didn't clear `self.data`, so each
    /// emitted event's `data` accumulated every prior event's data joined
    /// by newlines. Downstream JSON parsers then saw two concatenated JSON
    /// objects separated by a newline and rejected with a "trailing
    /// characters" error, making streaming providers (OpenRouter / Vercel /
    /// Groq / ...) appear to return empty responses.
    #[tokio::test]
    async fn decoder_does_not_leak_data_between_events() {
        let input: Vec<u8> =
            b"data: {\"i\":1}\n\ndata: {\"i\":2}\n\ndata: {\"i\":3}\n\n".to_vec();
        let cursor = std::io::Cursor::new(input);
        let reader = tokio::io::BufReader::new(cursor);
        let mut stream = decode_sse(reader);

        let e1 = stream.next().await.unwrap().unwrap();
        assert_eq!(e1.data, r#"{"i":1}"#);

        let e2 = stream.next().await.unwrap().unwrap();
        assert_eq!(e2.data, r#"{"i":2}"#, "event 2 must not inherit event 1 data");

        let e3 = stream.next().await.unwrap().unwrap();
        assert_eq!(e3.data, r#"{"i":3}"#, "event 3 must not inherit earlier data");
    }

    /// SSE comments (lines starting with `:`) must be ignored and must not
    /// count as a data line in the current event. OpenRouter sends
    /// `: OPENROUTER PROCESSING` keepalives that could break downstream
    /// JSON parsing if they were included in `data`.
    #[tokio::test]
    async fn decoder_ignores_comment_keepalives() {
        let input: Vec<u8> = b": OPENROUTER PROCESSING\n\n: OPENROUTER PROCESSING\n\ndata: {\"i\":42}\n\n"
            .to_vec();
        let cursor = std::io::Cursor::new(input);
        let reader = tokio::io::BufReader::new(cursor);
        let mut stream = decode_sse(reader);

        // Keepalive-only blocks produce no output because decoder has neither
        // `event` nor `data` set at flush time.
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.data, r#"{"i":42}"#);
        assert!(first.event.is_none());
    }
}
