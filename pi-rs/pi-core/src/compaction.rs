//! Context compaction - detect overflow and summarize old messages to stay under limits.
//! Implements Phase 2.14 with LLM-based summarization.

use anyhow::Result;
use pi_ai::types::Message;

/// Estimated tokens per message (for quick estimation without full tokenization)
const TOKENS_PER_CHAR: f32 = 0.25; // ~4 chars per token on average

/// Compaction result with metadata
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Summary of compacted messages
    pub summary: String,
    /// Number of messages replaced
    pub messages_compacted: usize,
    /// Tokens saved
    pub tokens_saved: u32,
    /// Tokens used for summary itself
    pub tokens_used: u32,
}

/// Estimate token count for a message
pub fn estimate_tokens(msg: &Message) -> u32 {
    let text = match msg {
        Message::User(contents) => {
            contents
                .iter()
                .filter_map(|c| match c {
                    pi_ai::types::Content::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
        Message::Assistant(contents) => {
            contents
                .iter()
                .filter_map(|c| match c {
                    pi_ai::types::Content::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
        Message::Tool { content, .. } => {
            content
                .iter()
                .filter_map(|c| match c {
                    pi_ai::types::Content::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
    };

    ((text.len() as f32) * TOKENS_PER_CHAR) as u32
}

/// Estimate total context tokens
pub fn estimate_context_tokens(messages: &[Message]) -> u32 {
    messages.iter().map(estimate_tokens).sum()
}

/// Detect if context overflow is likely (approaching limit)
pub fn should_compact(messages: &[Message], context_window: u32, threshold_percent: f32) -> bool {
    let tokens = estimate_context_tokens(messages);
    let threshold = ((context_window as f32) * threshold_percent) as u32;
    tokens > threshold
}

/// Build a compaction summary prompt
fn build_compaction_prompt(messages_to_summarize: &[Message]) -> String {
    let mut prompt = String::from(
        "Please provide a concise summary of the following conversation history. \
        Include:\n\
        - Key decisions made\n\
        - Important context or requirements\n\
        - Summary of work completed\n\
        - Any unresolved issues\n\n\
        Conversation history:\n\n"
    );

    for msg in messages_to_summarize {
        match msg {
            Message::User(contents) => {
                for content in contents {
                    if let pi_ai::types::Content::Text { text, .. } = content {
                        prompt.push_str("User: ");
                        prompt.push_str(text);
                        prompt.push('\n');
                    }
                }
            }
            Message::Assistant(contents) => {
                for content in contents {
                    if let pi_ai::types::Content::Text { text, .. } = content {
                        prompt.push_str("Assistant: ");
                        prompt.push_str(text);
                        prompt.push('\n');
                    }
                }
            }
            Message::Tool { content, .. } => {
                for c in content {
                    if let pi_ai::types::Content::Text { text, .. } = c {
                        prompt.push_str("Tool result: ");
                        prompt.push_str(text);
                        prompt.push('\n');
                    }
                }
            }
        }
    }

    prompt.push_str("\n\nPlease provide a brief, factual summary (2-3 paragraphs maximum).");
    prompt
}

/// Context usage information
#[derive(Debug, Clone)]
pub struct ContextUsage {
    /// Tokens used for input
    pub input_tokens: u32,
    /// Tokens used for output
    pub output_tokens: u32,
    /// Total tokens (sum of input + output)
    pub total_tokens: u32,
    /// Estimated remaining tokens before overflow
    pub remaining_tokens: u32,
}

/// Prepare messages for compaction by identifying candidates
pub fn prepare_compaction(
    messages: &[Message],
    context_window: u32,
    keep_most_recent: usize,
) -> Option<(Vec<Message>, Vec<Message>)> {
    let current_tokens = estimate_context_tokens(messages);

    // Only compact if we're using more than 70% of context
    if current_tokens < ((context_window as f32) * 0.7) as u32 {
        return None;
    }

    // Keep at least the most recent N messages
    if messages.len() <= keep_most_recent {
        return None;
    }

    let compaction_point = messages.len() - keep_most_recent;
    let to_compact = messages[..compaction_point].to_vec();
    let to_keep = messages[compaction_point..].to_vec();

    Some((to_compact, to_keep))
}

/// Calculate context tokens from usage
pub fn calculate_context_tokens(input: u32, output: u32, context_window: u32) -> ContextUsage {
    let total = input + output;
    let remaining = context_window.saturating_sub(total);

    ContextUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        remaining_tokens: remaining,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::Content;

    #[test]
    fn test_estimate_tokens() {
        let msg = Message::User(vec![Content::Text {
            text: "Hello, how are you?".to_string(),
            cache_control: None,
        }]);

        let tokens = estimate_tokens(&msg);
        assert!(tokens > 0);
        assert!(tokens < 100); // Should be small for short text
    }

    #[test]
    fn test_estimate_context_tokens() {
        let messages = vec![
            Message::User(vec![Content::Text {
                text: "Hello".to_string(),
                cache_control: None,
            }]),
            Message::User(vec![Content::Text {
                text: "How are you?".to_string(),
                cache_control: None,
            }]),
        ];

        let tokens = estimate_context_tokens(&messages);
        assert!(tokens > 0);
    }

    #[test]
    fn test_should_compact_below_threshold() {
        let messages = vec![Message::User(vec![Content::Text {
            text: "Short".to_string(),
            cache_control: None,
        }])];

        // 200k token window, threshold 70% = 140k
        let should_compact = should_compact(&messages, 200_000, 0.7);
        assert!(!should_compact);
    }

    #[test]
    fn test_should_compact_above_threshold() {
        let long_text = "a".repeat(50000); // Large text = many tokens
        let messages = vec![Message::User(vec![Content::Text {
            text: long_text,
            cache_control: None,
        }])];

        // 10k token window, threshold 70% = 7k
        let should_compact = should_compact(&messages, 10_000, 0.7);
        assert!(should_compact);
    }

    #[test]
    fn test_prepare_compaction_too_small() {
        let messages = vec![
            Message::User(vec![Content::Text {
                text: "Hi".to_string(),
                cache_control: None,
            }]),
            Message::User(vec![Content::Text {
                text: "Bye".to_string(),
                cache_control: None,
            }]),
        ];

        // Not enough messages or context
        let result = prepare_compaction(&messages, 200_000, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_prepare_compaction_keeps_recent() {
        let long_text = "a".repeat(150000); // Very large text to trigger compaction
        let messages = vec![
            Message::User(vec![Content::Text {
                text: long_text.clone(),
                cache_control: None,
            }]),
            Message::User(vec![Content::Text {
                text: long_text.clone(),
                cache_control: None,
            }]),
            Message::User(vec![Content::Text {
                text: "Recent message".to_string(),
                cache_control: None,
            }]),
        ];

        // Use context window of 100k tokens; with 2 messages of 150k chars each = ~75k tokens, should trigger at 70% threshold
        let result = prepare_compaction(&messages, 100_000, 1);
        assert!(result.is_some(), "Should trigger compaction with large messages");

        let (to_compact, to_keep) = result.unwrap();
        assert_eq!(to_compact.len(), 2);
        assert_eq!(to_keep.len(), 1);
    }

    #[test]
    fn test_build_compaction_prompt() {
        let messages = vec![
            Message::User(vec![Content::Text {
                text: "Create a file".to_string(),
                cache_control: None,
            }]),
            Message::Assistant(vec![Content::Text {
                text: "Done".to_string(),
                cache_control: None,
            }]),
        ];

        let prompt = build_compaction_prompt(&messages);
        assert!(prompt.contains("User: Create a file"));
        assert!(prompt.contains("Assistant: Done"));
        assert!(prompt.contains("summary"));
    }

    #[test]
    fn test_calculate_context_tokens() {
        let usage = calculate_context_tokens(1000, 500, 200_000);
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.output_tokens, 500);
        assert_eq!(usage.total_tokens, 1500);
        assert_eq!(usage.remaining_tokens, 198_500);
    }

    #[test]
    fn test_calculate_context_tokens_overflow() {
        let usage = calculate_context_tokens(150_000, 100_000, 200_000);
        assert_eq!(usage.total_tokens, 250_000);
        assert_eq!(usage.remaining_tokens, 0); // Saturating sub
    }
}
