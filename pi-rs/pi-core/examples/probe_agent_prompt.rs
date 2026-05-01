//! End-to-end probe: drive the real `pi_core::Agent` against a live
//! OpenRouter (or any OpenAI-compatible) provider and print the response.
//! Mirrors the exact code path the interactive TUI uses, minus the UI.
//!
//! Run:
//!
//! ```text
//! OPENROUTER_API_KEY=... \
//!   cargo run -p pi-core --example probe_agent_prompt -- \
//!   openrouter/stepfun/step-3.5-flash "Reply with exactly: Hi"
//! ```
//!
//! Model id accepts qualified `provider_id/model_id` so the registry
//! disambiguates providers that share ids.

use pi_core::model_registry::ModelRegistry;
use pi_core::Agent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "openrouter/stepfun/step-3.5-flash".to_string());
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Reply with exactly: Hello world!".to_string());

    let registry = ModelRegistry::global();
    let (provider, _model) = {
        if let Some((pid, mid)) = model_id.split_once('/') {
            registry
                .find_by_provider(pid, mid)
                .or_else(|| registry.find_model(&model_id))
                .ok_or_else(|| anyhow::anyhow!("model {model_id} not in registry"))?
        } else {
            registry
                .find_model(&model_id)
                .ok_or_else(|| anyhow::anyhow!("model {model_id} not in registry"))?
        }
    };

    let (env_var, api_key) = provider
        .resolve_env_key()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no env key for provider '{}' - expected one of: {}",
                provider.id,
                provider.env_vars.join(", ")
            )
        })?;

    eprintln!(
        "probe: model={} provider={} env={}",
        model_id, provider.id, env_var
    );

    let mut agent = Agent::new(&model_id).with_api_key(&api_key);

    let started = std::time::Instant::now();
    let response = agent.prompt(&prompt).await?;
    let elapsed = started.elapsed();

    eprintln!(
        "=== RESPONSE ({} chars in {:.2}s) ===",
        response.len(),
        elapsed.as_secs_f32()
    );
    println!("{response}");
    Ok(())
}
