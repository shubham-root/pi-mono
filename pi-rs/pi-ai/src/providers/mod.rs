//! Provider implementations.
//! Dispatches to correct provider based on model API.

use crate::types::{Model, StreamOptions, Api, Context};
use anyhow::Result;
use futures::Stream;

pub mod anthropic;
pub mod openai;
pub mod faux;
pub mod openrouter;
pub mod vercel;
pub mod bedrock;
pub mod sigv4;
pub mod multimodal;

/// Stream from the appropriate provider.
pub async fn stream(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<crate::types::stream::AssistantMessageEventStream> {
    match model.api {
        Api::AnthropicMessages => {
            anthropic::stream(model, context, options).await
        }
        Api::OpenAiCompletions => {
            openai::stream(model, context, options).await
        }
        Api::OpenRouterMessages => {
            openrouter::stream(model, context, options).await
        }
        Api::VercelAiGateway => {
            vercel::stream(model, context, options).await
        }
        Api::BedrockConverseStream => {
            bedrock::stream(model, context, options).await
        }
        Api::Faux => {
            Err(anyhow::anyhow!("Faux provider requires FauxProvider::new() - use in tests directly"))
        }
        _ => Err(anyhow::anyhow!("Provider not yet implemented for API: {:?}", model.api)),
    }
}
