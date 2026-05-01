//! Output modality support for image, video, and audio generation.
//!
//! Extends StreamEvent and adds generation-specific event types for:
//! - Image generation (DALL-E, etc.)
//! - Video generation (Kling, etc.)
//! - Audio generation / TTS (OpenAI TTS, etc.)

use crate::types::Usage;
use serde::{Deserialize, Serialize};

/// Generation status for async operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GenerationStatus {
    Queued,
    Processing,
    Completed,
    Failed,
}

/// Image generation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerated {
    pub url: Option<String>,          // URL to generated image
    pub base64: Option<String>,       // Base64-encoded image
    pub prompt: String,               // Original prompt
    pub revised_prompt: Option<String>, // Revised by model
    pub size: Option<String>,         // e.g., "1024x1024"
    pub quality: Option<String>,      // e.g., "hd"
}

/// Video generation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerated {
    pub url: Option<String>,          // URL to generated video
    pub base64: Option<String>,       // Base64 video data
    pub duration: Option<f32>,        // Duration in seconds
    pub format: Option<String>,       // e.g., "mp4"
    pub resolution: Option<String>,   // e.g., "1920x1080"
}

/// Audio generation event (TTS output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenerated {
    pub data: String,                 // Base64-encoded audio
    pub format: String,               // e.g., "mp3", "wav", "opus"
    pub duration: Option<f32>,        // Duration in seconds
}

/// Extended stream events for generation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenerationStreamEvent {
    /// Image generation started
    ImageGenerationStart {
        count: u32,
        size: Option<String>,
    },

    /// Image generation progress (for async)
    ImageGenerationProgress {
        current: u32,
        total: u32,
        status: GenerationStatus,
    },

    /// Image generated
    ImageGenerated(ImageGenerated),

    /// Multiple images generated
    ImagesGenerated(Vec<ImageGenerated>),

    /// Video generation started
    VideoGenerationStart {
        duration: Option<f32>,
        resolution: Option<String>,
    },

    /// Video generation progress
    VideoGenerationProgress {
        percent: u32,
        status: GenerationStatus,
        eta_seconds: Option<u32>,
    },

    /// Video generated
    VideoGenerated(VideoGenerated),

    /// Audio generation started
    AudioGenerationStart {
        voice: Option<String>,
        speed: Option<f32>,
    },

    /// Audio generated (TTS)
    AudioGenerated(AudioGenerated),

    /// Async job created (for polling-based generation)
    AsyncJobCreated {
        job_id: String,
        status: GenerationStatus,
    },

    /// Job status update
    AsyncJobProgress {
        job_id: String,
        status: GenerationStatus,
        percent: u32,
    },

    /// Job completed with results
    AsyncJobCompleted {
        job_id: String,
        urls: Vec<String>,  // Results from completed job
    },

    /// Generation error
    GenerationError {
        error: String,
        code: Option<String>,
        job_id: Option<String>,
    },

    /// Generation complete with usage info
    GenerationComplete {
        usage: Option<Usage>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_generated_serialization() {
        let img = ImageGenerated {
            url: Some("https://example.com/image.png".to_string()),
            base64: None,
            prompt: "A cat".to_string(),
            revised_prompt: None,
            size: Some("1024x1024".to_string()),
            quality: Some("hd".to_string()),
        };

        let json = serde_json::to_string(&img).unwrap();
        assert!(json.contains("\"url\""));
        assert!(json.contains("\"A cat\""));
    }

    #[test]
    fn test_generation_event_status() {
        let event = GenerationStreamEvent::ImageGenerationProgress {
            current: 1,
            total: 4,
            status: GenerationStatus::Processing,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("image_generation_progress"));
        assert!(json.contains("processing")); // lowercase due to rename_all
    }

    #[test]
    fn test_audio_generated() {
        let audio = AudioGenerated {
            data: "base64encodedsound".to_string(),
            format: "mp3".to_string(),
            duration: Some(30.5),
        };

        let json = serde_json::to_string(&audio).unwrap();
        assert!(json.contains("\"mp3\""));
    }

    #[test]
    fn test_async_job_created() {
        let event = GenerationStreamEvent::AsyncJobCreated {
            job_id: "job_12345".to_string(),
            status: GenerationStatus::Queued,
        };

        match event {
            GenerationStreamEvent::AsyncJobCreated { job_id, status } => {
                assert_eq!(job_id, "job_12345");
                assert!(matches!(status, GenerationStatus::Queued));
            }
            _ => panic!("Wrong event type"),
        }
    }
}
