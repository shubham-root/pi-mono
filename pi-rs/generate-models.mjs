#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const tsModelsPath = path.join(__dirname, '../packages/ai/src/models.generated.ts');
const rustModelsPath = path.join(__dirname, 'pi-modes/src/models.rs');

// Read TS file
const tsContent = fs.readFileSync(tsModelsPath, 'utf-8');

// Parse models - extract key info
// Match: id: "...", name: "...", api: "...", provider: "..."
const modelRegex = /id:\s*"([^"]+)",[\s\S]*?name:\s*"([^"]+)",[\s\S]*?provider:\s*"([^"]+)"/g;
let match;
const models = [];

while ((match = modelRegex.exec(tsContent)) !== null) {
  const [, id, name, provider] = match;
  models.push({ id, name, provider });
}

console.log(`Found ${models.length} models`);

// Sort by provider then name
models.sort((a, b) => {
  if (a.provider !== b.provider) return a.provider.localeCompare(b.provider);
  return a.name.localeCompare(b.name);
});

// Generate Rust code
let rustCode = `//! Model registry auto-generated from packages/ai/src/models.generated.ts
//! Run: node generate-models.mjs

pub fn get_all_models() -> Vec<(&'static str, &'static str, &'static str)> {
    // (id, name, provider)
    vec![
`;

for (const model of models) {
  rustCode += `        ("${model.id}", "${model.name}", "${model.provider}"),\n`;
}

rustCode += `    ]
}

pub fn get_models_for_provider(provider: &str) -> Vec<(&'static str, &'static str)> {
    get_all_models()
        .into_iter()
        .filter(|(_, _, p)| p == &provider)
        .map(|(id, name, _)| (id, name))
        .collect()
}

pub fn get_providers() -> Vec<&'static str> {
    let mut providers: Vec<&str> = get_all_models()
        .into_iter()
        .map(|(_, _, p)| p)
        .collect();
    providers.sort();
    providers.dedup();
    providers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_not_empty() {
        assert!(!get_all_models().is_empty());
    }

    #[test]
    fn test_providers_not_empty() {
        assert!(!get_providers().is_empty());
    }

    #[test]
    fn test_anthropic_models() {
        let anthropic = get_models_for_provider("anthropic");
        assert!(!anthropic.is_empty());
    }

    #[test]
    fn test_openai_models() {
        let openai = get_models_for_provider("openai");
        assert!(!openai.is_empty());
    }
}
`;

// Write Rust file
fs.writeFileSync(rustModelsPath, rustCode);

const providers = new Set(models.map(m => m.provider));
console.log(`✅ Generated ${models.length} models in ${rustModelsPath}`);
console.log(`📊 Providers: ${providers.size}`);
console.log(`   Providers: ${Array.from(providers).sort().join(', ')}`);
