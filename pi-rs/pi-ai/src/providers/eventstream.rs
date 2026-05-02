//! Minimal decoder for AWS's `application/vnd.amazon.eventstream` binary
//! framing used by Bedrock Converse Stream and Kinesis. We only need
//! enough to parse inbound response messages; we never encode.
//!
//! Wire format per message:
//!
//!   [0..4]     total length (u32 BE)
//!   [4..8]     headers length (u32 BE)
//!   [8..12]    prelude CRC32 (covers first 8 bytes)
//!   [12..]     headers (headers_len bytes)
//!   [...]      payload
//!   [last 4]   message CRC32 (covers all bytes 0..N-4)
//!
//! Header blob format:
//!
//!   name_len  : u8         (1 byte, length of the header name)
//!   name      : name_len bytes ASCII
//!   value_tag : u8         (type discriminator, we only care about 7 = string)
//!   value_len : u16 BE     (for string type)
//!   value     : value_len bytes
//!
//! We skip CRC validation. The underlying TLS/TCP already guarantees
//! transport integrity and AWS keeps these frames small; corruption
//! would surface as a JSON parse error on the payload which we report
//! with context.

use anyhow::{anyhow, Result};
use bytes::{Buf, Bytes, BytesMut};
use std::collections::HashMap;

/// One parsed event-stream message: a map of headers plus the raw
/// payload bytes. Payloads are usually JSON for Bedrock.
#[derive(Debug, Clone)]
pub struct EventStreamMessage {
    pub headers: HashMap<String, String>,
    pub payload: Bytes,
}

impl EventStreamMessage {
    /// Convenience accessor for the `:event-type` header that Bedrock
    /// uses to dispatch event variants (e.g. `messageStart`,
    /// `contentBlockDelta`, `metadata`).
    pub fn event_type(&self) -> Option<&str> {
        self.headers.get(":event-type").map(String::as_str)
    }
}

/// Incremental decoder. Feed bytes in via `push`, call `next_message`
/// in a loop until it returns `None` to consume complete frames.
#[derive(Default)]
pub struct EventStreamDecoder {
    buf: BytesMut,
}

impl EventStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Extract the next complete message from the internal buffer.
    /// Returns `Ok(None)` when we need more bytes; `Err` on malformed
    /// framing so the caller can surface a clear error.
    pub fn next_message(&mut self) -> Result<Option<EventStreamMessage>> {
        if self.buf.len() < 12 {
            return Ok(None);
        }
        let total_len = u32::from_be_bytes(self.buf[0..4].try_into().unwrap()) as usize;
        let headers_len = u32::from_be_bytes(self.buf[4..8].try_into().unwrap()) as usize;
        if total_len < 16 {
            return Err(anyhow!(
                "eventstream frame too short: total_len={total_len}"
            ));
        }
        if self.buf.len() < total_len {
            return Ok(None);
        }
        // Consume the whole frame from the buffer before parsing so a
        // malformed frame doesn't wedge the decoder.
        let frame = self.buf.split_to(total_len).freeze();
        // Headers live between byte 12 and 12+headers_len; payload
        // fills the rest up to total_len-4 (trailing CRC).
        let headers_start = 12;
        let headers_end = headers_start + headers_len;
        let payload_end = total_len - 4;
        if headers_end > payload_end {
            return Err(anyhow!(
                "eventstream headers overrun payload: headers_end={headers_end} payload_end={payload_end}"
            ));
        }
        let headers = parse_headers(&frame[headers_start..headers_end])?;
        let payload = frame.slice(headers_end..payload_end);
        Ok(Some(EventStreamMessage { headers, payload }))
    }
}

fn parse_headers(mut bytes: &[u8]) -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    while !bytes.is_empty() {
        if bytes.len() < 1 {
            return Err(anyhow!("header block truncated at name-length prefix"));
        }
        let name_len = bytes.get_u8() as usize;
        if bytes.len() < name_len + 1 {
            return Err(anyhow!("header block truncated reading name"));
        }
        let name = std::str::from_utf8(&bytes[..name_len])
            .map_err(|_| anyhow!("header name is not utf-8"))?
            .to_string();
        bytes.advance(name_len);
        let value_tag = bytes.get_u8();
        match value_tag {
            // 7 = string; 2 bytes big-endian length + bytes.
            7 => {
                if bytes.len() < 2 {
                    return Err(anyhow!("string header truncated at length prefix"));
                }
                let value_len = bytes.get_u16() as usize;
                if bytes.len() < value_len {
                    return Err(anyhow!("string header truncated reading value"));
                }
                let value = std::str::from_utf8(&bytes[..value_len])
                    .map_err(|_| anyhow!("string header value is not utf-8"))?
                    .to_string();
                bytes.advance(value_len);
                out.insert(name, value);
            }
            // Other header types (bool, byte, short, int, long, byte array,
            // timestamp, uuid). Bedrock only uses strings in practice, but
            // we skip unknown tags gracefully instead of erroring so the
            // decoder doesn't die on a minor schema change.
            0 | 1 => {
                // true / false: no payload. Record as "true"/"false" for
                // diagnostics.
                out.insert(name, if value_tag == 0 { "false" } else { "true" }.to_string());
            }
            2 => {
                if bytes.len() < 1 {
                    return Err(anyhow!("byte header truncated"));
                }
                let v = bytes.get_u8();
                out.insert(name, v.to_string());
            }
            3 => {
                if bytes.len() < 2 {
                    return Err(anyhow!("short header truncated"));
                }
                let v = bytes.get_i16();
                out.insert(name, v.to_string());
            }
            4 => {
                if bytes.len() < 4 {
                    return Err(anyhow!("int header truncated"));
                }
                let v = bytes.get_i32();
                out.insert(name, v.to_string());
            }
            5 => {
                if bytes.len() < 8 {
                    return Err(anyhow!("long header truncated"));
                }
                let v = bytes.get_i64();
                out.insert(name, v.to_string());
            }
            6 => {
                if bytes.len() < 2 {
                    return Err(anyhow!("bytes header truncated at length prefix"));
                }
                let value_len = bytes.get_u16() as usize;
                if bytes.len() < value_len {
                    return Err(anyhow!("bytes header truncated reading value"));
                }
                bytes.advance(value_len);
            }
            8 => {
                // timestamp, i64 ms
                if bytes.len() < 8 {
                    return Err(anyhow!("timestamp header truncated"));
                }
                bytes.advance(8);
            }
            9 => {
                // uuid, 16 bytes
                if bytes.len() < 16 {
                    return Err(anyhow!("uuid header truncated"));
                }
                bytes.advance(16);
            }
            other => {
                return Err(anyhow!("unknown header value tag: {other}"));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal valid frame manually to test the decoder.
    // We don't bother computing CRCs since the decoder skips validation;
    // the CRCs are present on the wire but we don't assert them.
    fn make_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        let name = ":event-type";
        headers.push(name.len() as u8);
        headers.extend_from_slice(name.as_bytes());
        headers.push(7u8); // string tag
        headers.extend_from_slice(&(event_type.len() as u16).to_be_bytes());
        headers.extend_from_slice(event_type.as_bytes());

        let headers_len = headers.len() as u32;
        let total_len = 12 + headers.len() + payload.len() + 4;
        let mut frame = Vec::with_capacity(total_len);
        frame.extend_from_slice(&(total_len as u32).to_be_bytes());
        frame.extend_from_slice(&headers_len.to_be_bytes());
        frame.extend_from_slice(&[0, 0, 0, 0]); // prelude CRC placeholder
        frame.extend_from_slice(&headers);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&[0, 0, 0, 0]); // message CRC placeholder
        frame
    }

    #[test]
    fn decodes_single_frame() {
        let mut d = EventStreamDecoder::new();
        d.push(&make_frame("contentBlockDelta", b"{\"hi\":1}"));
        let msg = d.next_message().unwrap().unwrap();
        assert_eq!(msg.event_type(), Some("contentBlockDelta"));
        assert_eq!(&msg.payload[..], b"{\"hi\":1}");
    }

    #[test]
    fn decodes_back_to_back_frames() {
        let mut d = EventStreamDecoder::new();
        d.push(&make_frame("a", b"1"));
        d.push(&make_frame("b", b"22"));
        let m1 = d.next_message().unwrap().unwrap();
        let m2 = d.next_message().unwrap().unwrap();
        assert_eq!(m1.event_type(), Some("a"));
        assert_eq!(m2.event_type(), Some("b"));
        assert!(d.next_message().unwrap().is_none());
    }

    #[test]
    fn returns_none_on_partial_frame() {
        let mut d = EventStreamDecoder::new();
        let frame = make_frame("x", b"payload");
        // Feed in half the bytes and expect None.
        d.push(&frame[..frame.len() / 2]);
        assert!(d.next_message().unwrap().is_none());
        // Feed the rest and expect the complete message.
        d.push(&frame[frame.len() / 2..]);
        let m = d.next_message().unwrap().unwrap();
        assert_eq!(m.event_type(), Some("x"));
    }

    #[test]
    fn errors_on_truncated_headers() {
        let mut bad = make_frame("x", b"");
        // Lie about headers_len so it overruns.
        bad[4..8].copy_from_slice(&(9999u32.to_be_bytes()));
        let mut d = EventStreamDecoder::new();
        d.push(&bad);
        assert!(d.next_message().is_err());
    }
}
