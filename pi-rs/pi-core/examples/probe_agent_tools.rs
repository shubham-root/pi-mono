//! Probe a tool-using turn through the full Agent loop.

use pi_core::model_registry::ModelRegistry;
use pi_core::Agent;
use pi_tools::{BashTool, EditTool, FindTool, GrepTool, LsTool, ReadTool, WriteTool};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "openrouter/openai/gpt-4o-mini".to_string());
    let prompt = args.get(2).cloned().unwrap_or_else(|| {
        "What directory am I currently in? Use the bash tool with `pwd` to find out.".to_string()
    });

    let registry = ModelRegistry::global();
    let (provider, _model) = if let Some((pid, mid)) = model_id.split_once('/') {
        registry
            .find_by_provider(pid, mid)
            .or_else(|| registry.find_model(&model_id))
            .ok_or_else(|| anyhow::anyhow!("model not in registry: {model_id}"))?
    } else {
        registry
            .find_model(&model_id)
            .ok_or_else(|| anyhow::anyhow!("model not in registry: {model_id}"))?
    };
    let (env_var, api_key) = provider
        .resolve_env_key()
        .ok_or_else(|| anyhow::anyhow!("no env key for {}", provider.id))?;

    eprintln!("probe: model={model_id} provider={} env={env_var}", provider.id);

    let mut agent = Agent::new(&model_id)
        .with_api_key(&api_key)
        .with_tool(Box::new(BashTool))
        .with_tool(Box::new(ReadTool))
        .with_tool(Box::new(WriteTool))
        .with_tool(Box::new(EditTool))
        .with_tool(Box::new(GrepTool))
        .with_tool(Box::new(FindTool))
        .with_tool(Box::new(LsTool));

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
