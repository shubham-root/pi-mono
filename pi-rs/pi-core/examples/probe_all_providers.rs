//! Live probe: run a tiny prompt through every provider in the registry
//! whose env key is configured in the current shell. Prints a per-provider
//! pass/fail summary so regressions in the streaming path surface quickly.
//!
//! Usage:
//!
//! ```text
//! OPENROUTER_API_KEY=... ANTHROPIC_API_KEY=... \
//!   cargo run -p pi-core --example probe_all_providers
//! ```
//!
//! Providers without a configured env key are skipped (printed as SKIP).
//! This does not require a daemon; it hits the live APIs directly through
//! the same Agent code path the TUI uses.

use pi_core::model_registry::ModelRegistry;
use pi_core::Agent;
use std::time::Duration;

/// Preferred default model per provider. Mirrors the TypeScript
/// `defaultModelPerProvider` map so the probe hits a model the provider is
/// actually set up to serve. Some providers — notably Amazon Bedrock —
/// reject bare model IDs and require a region-prefixed inference profile
/// (`us.anthropic...`), which alphabetical ordering doesn't pick.
fn preferred_model_for(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        "amazon-bedrock" => Some("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic" => Some("claude-haiku-4-5"),
        "openai" => Some("gpt-4o"),
        "openrouter" => Some("openai/gpt-4o-mini"),
        "google" => Some("gemini-2.0-flash-001"),
        "xai" => Some("grok-4-fast"),
        "groq" => Some("llama-3.3-70b-versatile"),
        "cerebras" => Some("llama-3.3-70b"),
        "deepseek" => Some("deepseek-chat"),
        "mistral" => Some("mistral-small-latest"),
        "fireworks" => Some("accounts/fireworks/models/llama-v3p3-70b-instruct"),
        "vercel-ai-gateway" => Some("anthropic/claude-haiku-4-5"),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct ProbeOutcome {
    provider: String,
    model_id: String,
    env_var: Option<String>,
    result: Result<(String, Duration), String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Reply with exactly: hi".to_string());

    let registry = ModelRegistry::global();
    let mut outcomes: Vec<ProbeOutcome> = Vec::new();

    for provider in registry.providers() {
        let env = provider.resolve_env_key();
        let Some((env_var, _key)) = env else {
            outcomes.push(ProbeOutcome {
                provider: provider.id.clone(),
                model_id: "-".to_string(),
                env_var: None,
                result: Err("no env var configured".to_string()),
            });
            continue;
        };

        // Prefer a curated default model for each provider when available;
        // fall back to the first-declared model otherwise. This matters for
        // providers like Bedrock where the alphabetically-first model
        // requires an inference profile not present in the bare ID.
        let preferred = preferred_model_for(&provider.id)
            .and_then(|id| provider.models.iter().find(|m| m.id == id));
        let model = match preferred.or_else(|| provider.models.first()) {
            Some(m) => m,
            None => {
                outcomes.push(ProbeOutcome {
                    provider: provider.id.clone(),
                    model_id: "-".to_string(),
                    env_var: Some(env_var),
                    result: Err("provider has no models declared".to_string()),
                });
                continue;
            }
        };
        let qualified = format!("{}/{}", provider.id, model.id);

        eprintln!("--- probing {qualified} via {env_var} ---");
        let result = run_probe(&qualified, &prompt).await;
        outcomes.push(ProbeOutcome {
            provider: provider.id.clone(),
            model_id: model.id.clone(),
            env_var: Some(env_var),
            result,
        });
    }

    println!();
    println!("=== PROBE SUMMARY ===");
    let name_w = outcomes.iter().map(|o| o.provider.len()).max().unwrap_or(8);
    let model_w = outcomes.iter().map(|o| o.model_id.len()).max().unwrap_or(8).min(32);
    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;
    for o in &outcomes {
        let status = match &o.result {
            Ok((text, dur)) => {
                passed += 1;
                let preview: String = text.chars().take(40).collect();
                format!(
                    "OK   [{:>5.2}s] len={:<5} preview={:?}",
                    dur.as_secs_f32(),
                    text.len(),
                    preview
                )
            }
            Err(e) if o.env_var.is_none() => {
                skipped += 1;
                format!("SKIP no-auth")
            }
            Err(e) => {
                failed += 1;
                format!("FAIL {e}")
            }
        };
        let model_trim: String = o.model_id.chars().take(model_w).collect();
        println!(
            "{:<name_w$}  {:<model_w$}  {}",
            o.provider,
            model_trim,
            status,
            name_w = name_w,
            model_w = model_w
        );
    }
    println!("--- {passed} passed, {failed} failed, {skipped} skipped ---");
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

async fn run_probe(model_id: &str, prompt: &str) -> Result<(String, Duration), String> {
    // Resolve the correct env key for this model via the registry.
    let registry = ModelRegistry::global();
    let (provider, _model) = registry
        .find_model(model_id)
        .or_else(|| {
            model_id
                .split_once('/')
                .and_then(|(p, m)| registry.find_by_provider(p, m))
        })
        .ok_or_else(|| "model not in registry".to_string())?;
    let (_var, key) = provider
        .resolve_env_key()
        .ok_or_else(|| "env key missing".to_string())?;

    let mut agent = Agent::new(model_id).with_api_key(&key);

    // Enforce a wall-clock ceiling so a stuck provider doesn't block the
    // whole matrix.
    let start = std::time::Instant::now();
    let outcome = tokio::time::timeout(
        Duration::from_secs(30),
        agent.prompt(prompt),
    )
    .await;

    match outcome {
        Ok(Ok(text)) => Ok((text, start.elapsed())),
        Ok(Err(e)) => Err(format!("{e}")),
        Err(_) => Err("timeout after 30s".to_string()),
    }
}
