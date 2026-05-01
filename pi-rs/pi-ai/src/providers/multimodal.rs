//! Provider-specific multimodal content handling.
//!
//! Each provider has different capabilities and APIs for handling images, audio, video, etc.
//! This module provides utilities for converting pi's unified multimodal format to each provider's specific format.

use crate::types::{Content, MediaSource};
use serde_json::{json, Value};

/// OpenAI multimodal content converter
pub mod openai {
    use super::*;

    /// Convert pi Content to OpenAI message content format
    ///
    /// OpenAI supports:
    /// - text (all models)
    /// - image_url (GPT-4V, GPT-4 Turbo, GPT-4o)
    /// - image + detail level (low/high/auto)
    pub fn convert_content(content: &[Content]) -> Vec<Value> {
        let mut result = Vec::new();

        for c in content {
            match c {
                Content::Text { text, .. } => {
                    result.push(json!({
                        "type": "text",
                        "text": text,
                    }));
                }
                Content::Image { source, .. } => {
                    match source {
                        MediaSource::Url { url } => {
                            result.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": url,
                                    "detail": "auto",
                                }
                            }));
                        }
                        MediaSource::Base64 { media_type, data } => {
                            result.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{};base64,{}", media_type, data),
                                    "detail": "auto",
                                }
                            }));
                        }
                    }
                }
                Content::Audio { .. } | Content::Video { .. } | Content::Pdf { .. } => {
                    // OpenAI chat doesn't support these yet, skip
                    // Could add note: "Model doesn't support audio/video/PDF input"
                }
                _ => {} // Skip thinking, tooluse, etc.
            }
        }

        result
    }
}

/// Anthropic multimodal content converter
pub mod anthropic {
    use super::*;

    /// Convert pi Content to Anthropic message content format
    ///
    /// Anthropic supports:
    /// - text
    /// - image (source type: base64, media_type required)
    /// - PDF (via vision - Claude 3 models)
    pub fn convert_content(content: &[Content]) -> Vec<Value> {
        let mut result = Vec::new();

        for c in content {
            match c {
                Content::Text { text, .. } => {
                    result.push(json!({
                        "type": "text",
                        "text": text,
                    }));
                }
                Content::Image { source, .. } => {
                    match source {
                        MediaSource::Url { url } => {
                            // Anthropic can fetch from URL (requires public access)
                            result.push(json!({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                }
                            }));
                        }
                        MediaSource::Base64 { media_type, data } => {
                            result.push(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                }
                            }));
                        }
                    }
                }
                Content::Pdf { source, .. } => {
                    // Claude 3 models support PDF
                    match source {
                        MediaSource::Url { url } => {
                            result.push(json!({
                                "type": "document",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                },
                                "format": "pdf"
                            }));
                        }
                        MediaSource::Base64 { data, .. } => {
                            result.push(json!({
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": data,
                                },
                                "format": "pdf"
                            }));
                        }
                    }
                }
                Content::Audio { .. } | Content::Video { .. } => {
                    // Not yet supported by Anthropic
                }
                _ => {}
            }
        }

        result
    }
}

/// OpenRouter multimodal content converter
pub mod openrouter {
    use super::*;

    /// Convert pi Content to OpenRouter format
    ///
    /// OpenRouter is OpenAI-compatible but also routes to other providers.
    /// Supports:
    /// - Providers with vision: GPT-4V, Claude 3, Llama 2 Vision, etc.
    /// - Provider-specific modality support varies
    pub fn convert_content(content: &[Content]) -> Vec<Value> {
        // OpenRouter uses OpenAI format for messages, so reuse that
        openai::convert_content(content)
    }
}

/// Google Generative AI multimodal converter
pub mod google {
    use super::*;

    /// Convert pi Content to Google Generative AI format (Gemini)
    ///
    /// Google supports:
    /// - text
    /// - inline_data (images, audio, video, PDF as base64)
    /// - file_data (reference to files uploaded to Google AI)
    pub fn convert_content(content: &[Content]) -> Vec<Value> {
        let mut result = Vec::new();

        for c in content {
            match c {
                Content::Text { text, .. } => {
                    result.push(json!({
                        "text": text,
                    }));
                }
                Content::Image { source, .. } => {
                    match source {
                        MediaSource::Url { url } => {
                            // Google prefers inline_data but can fetch from URL
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": "image/jpeg", // Should detect from URL
                                    "data": url,
                                }
                            }));
                        }
                        MediaSource::Base64 { media_type, data } => {
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": media_type,
                                    "data": data,
                                }
                            }));
                        }
                    }
                }
                Content::Audio { source, .. } => {
                    match source {
                        MediaSource::Url { url } => {
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": url,
                                }
                            }));
                        }
                        MediaSource::Base64 { media_type, data } => {
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": media_type,
                                    "data": data,
                                }
                            }));
                        }
                    }
                }
                Content::Video { source, .. } => {
                    match source {
                        MediaSource::Url { url } => {
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": "video/mp4",
                                    "data": url,
                                }
                            }));
                        }
                        MediaSource::Base64 { media_type, data } => {
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": media_type,
                                    "data": data,
                                }
                            }));
                        }
                    }
                }
                Content::Pdf { source, .. } => {
                    match source {
                        MediaSource::Url { url } => {
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": "application/pdf",
                                    "data": url,
                                }
                            }));
                        }
                        MediaSource::Base64 { data, .. } => {
                            result.push(json!({
                                "inline_data": {
                                    "mime_type": "application/pdf",
                                    "data": data,
                                }
                            }));
                        }
                    }
                }
                _ => {}
            }
        }

        result
    }
}

/// Bedrock multimodal content converter
pub mod bedrock {
    use super::*;

    /// Convert pi Content to Bedrock format
    ///
    /// Bedrock supports different modalities depending on model.
    /// Claude 3 models support images via base64.
    pub fn convert_content(content: &[Content]) -> Vec<Value> {
        let mut result = Vec::new();

        for c in content {
            match c {
                Content::Text { text, .. } => {
                    result.push(json!({
                        "type": "text",
                        "text": text,
                    }));
                }
                Content::Image { source, .. } => {
                    match source {
                        MediaSource::Url { url } => {
                            // Bedrock can fetch from URL
                            result.push(json!({
                                "type": "image",
                                "source": {
                                    "bytes": url,
                                    // Could be better - Bedrock uses specific format
                                }
                            }));
                        }
                        MediaSource::Base64 { media_type, data } => {
                            result.push(json!({
                                "type": "image",
                                "source": {
                                    "bytes": data,
                                    "format": media_type.split('/').last().unwrap_or("jpeg"),
                                }
                            }));
                        }
                    }
                }
                _ => {
                    // Other modalities not yet supported via Bedrock
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_text_conversion() {
        let content = vec![Content::Text {
            text: "Hello".to_string(),
            cache_control: None,
        }];

        let result = openai::convert_content(&content);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "text");
        assert_eq!(result[0]["text"], "Hello");
    }

    #[test]
    fn test_openai_image_url_conversion() {
        let content = vec![Content::Image {
            source: MediaSource::Url {
                url: "https://example.com/image.jpg".to_string(),
            },
            cache_control: None,
        }];

        let result = openai::convert_content(&content);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "image_url");
        assert!(result[0]["image_url"]["url"].as_str().unwrap().contains("https://"));
    }

    #[test]
    fn test_anthropic_text_conversion() {
        let content = vec![Content::Text {
            text: "Hello".to_string(),
            cache_control: None,
        }];

        let result = anthropic::convert_content(&content);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "text");
    }

    #[test]
    fn test_anthropic_image_base64_conversion() {
        let content = vec![Content::Image {
            source: MediaSource::Base64 {
                media_type: "image/jpeg".to_string(),
                data: "abc123".to_string(),
            },
            cache_control: None,
        }];

        let result = anthropic::convert_content(&content);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "image");
        assert_eq!(result[0]["source"]["type"], "base64");
    }

    #[test]
    fn test_google_multimodal_conversion() {
        let content = vec![
            Content::Text {
                text: "Describe this".to_string(),
                cache_control: None,
            },
            Content::Image {
                source: MediaSource::Base64 {
                    media_type: "image/png".to_string(),
                    data: "xyz789".to_string(),
                },
                cache_control: None,
            },
        ];

        let result = google::convert_content(&content);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["text"], "Describe this");
        assert_eq!(result[1]["inline_data"]["mime_type"], "image/png");
    }
}
