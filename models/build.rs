//! Build script that parses models.toml and generates Rust code.

use std::env;
use std::fs;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TomlData {
    models: Vec<TomlModel>,
}

#[derive(Debug, Deserialize)]
struct TomlModel {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    name: String,
    provider: String,
    context_window: u32,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    tier: String,
    #[serde(default)]
    capabilities: Vec<String>,
}

fn main() {
    println!("cargo::rerun-if-changed=models.toml");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let toml_path = Path::new(&manifest_dir).join("models.toml");
    let toml_content = fs::read_to_string(&toml_path).expect("Failed to read models.toml");

    let data: TomlData = toml::from_str(&toml_content).expect("Failed to parse models.toml");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("generated.rs");

    let mut code = String::new();

    // Generate MODELS array using core types
    code.push_str("const MODELS: &[aither_core::llm::model::ModelInfo] = &[\n");
    for model in &data.models {
        code.push_str(&format!(
            "    aither_core::llm::model::ModelInfo {{\n        id: {:?},\n        name: {:?},\n        provider: {:?},\n        context_window: {},\n        max_output_tokens: {},\n        tier: {},\n        abilities: &[{}],\n    }},\n",
            model.id,
            model.name,
            model.provider,
            model.context_window,
            match model.max_output_tokens {
                Some(v) => format!("Some({v})"),
                None => "None".to_string(),
            },
            tier_variant(&model.tier),
            model.capabilities.iter().map(|c| ability_variant(c)).collect::<Vec<_>>().join(", "),
        ));
    }
    code.push_str("];\n\n");

    // Generate ALIASES map as a static slice of (alias, canonical_id) pairs
    let mut aliases: Vec<(String, String)> = Vec::new();
    for model in &data.models {
        for alias in &model.aliases {
            aliases.push((alias.to_lowercase(), model.id.clone()));
        }
    }
    // Sort for binary search
    aliases.sort_by(|a, b| a.0.cmp(&b.0));

    code.push_str("const ALIASES: &[(&str, &str)] = &[\n");
    for (alias, canonical) in &aliases {
        code.push_str(&format!("    ({:?}, {:?}),\n", alias, canonical));
    }
    code.push_str("];\n");

    fs::write(&out_path, code).expect("Failed to write generated.rs");
}

fn tier_variant(tier: &str) -> &'static str {
    match tier.to_lowercase().as_str() {
        "flagship" => "aither_core::llm::model::ModelTier::Flagship",
        "balanced" => "aither_core::llm::model::ModelTier::Balanced",
        "fast" => "aither_core::llm::model::ModelTier::Fast",
        _ => panic!("Unknown tier: {tier}"),
    }
}

fn ability_variant(cap: &str) -> &'static str {
    match cap.to_lowercase().as_str() {
        "vision" => "aither_core::llm::model::Ability::Vision",
        "tool_use" => "aither_core::llm::model::Ability::ToolUse",
        "audio" => "aither_core::llm::model::Ability::Audio",
        "video" => "aither_core::llm::model::Ability::Video",
        "pdf" => "aither_core::llm::model::Ability::Pdf",
        "web_search" => "aither_core::llm::model::Ability::WebSearch",
        "code_execution" => "aither_core::llm::model::Ability::CodeExecution",
        "reasoning" => "aither_core::llm::model::Ability::Reasoning",
        "image_generation" => "aither_core::llm::model::Ability::ImageGeneration",
        _ => panic!("Unknown capability: {cap}"),
    }
}
