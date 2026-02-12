//! Build script that parses model registry TOML files and generates Rust code.

use std::env;
use std::fs;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TomlData<T> {
    models: Vec<T>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmTomlModel {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    name: String,
    provider: String,
    context_window: u32,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    tier: Option<String>,
    #[serde(default)]
    tiers: Vec<String>,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    reasoning_efforts: Vec<String>,
    #[serde(default)]
    reasoning_effort_default: Option<String>,
    #[serde(default)]
    adaptive_reasoning: bool,
    #[serde(default)]
    reasoning_budget_tokens_min: Option<u32>,
    #[serde(default)]
    reasoning_budget_tokens_max: Option<u32>,
    #[serde(default)]
    outdated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageTomlModel {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    name: String,
    provider: String,
    context_window: u32,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    image_max_resolution: Option<String>,
    #[serde(default)]
    outdated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingTomlModel {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    name: String,
    provider: String,
    context_window: u32,
    #[serde(default)]
    capabilities: Vec<String>,
    embedding_dimensions: u32,
    #[serde(default)]
    outdated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RerankerTomlModel {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    name: String,
    provider: String,
    context_window: u32,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    reranker_max_documents: Option<u32>,
    #[serde(default)]
    outdated: bool,
}

#[derive(Debug, Clone)]
struct UnifiedModel {
    id: String,
    aliases: Vec<String>,
    name: String,
    provider: String,
    kind: &'static str,
    context_window: u32,
    max_output_tokens: Option<u32>,
    tier: Option<String>,
    tiers: Vec<String>,
    capabilities: Vec<String>,
    reasoning_efforts: Vec<String>,
    reasoning_effort_default: Option<String>,
    adaptive_reasoning: bool,
    reasoning_budget_tokens_min: Option<u32>,
    reasoning_budget_tokens_max: Option<u32>,
    embedding_dimensions: Option<u32>,
    image_max_resolution: Option<String>,
    reranker_max_documents: Option<u32>,
    outdated: bool,
}

fn main() {
    println!("cargo::rerun-if-changed=models_llm.toml");
    println!("cargo::rerun-if-changed=models_image.toml");
    println!("cargo::rerun-if-changed=models_embedding.toml");
    println!("cargo::rerun-if-changed=models_reranker.toml");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let llm_data: TomlData<LlmTomlModel> = read_toml(&manifest_dir, "models_llm.toml");
    let image_data: TomlData<ImageTomlModel> = read_toml(&manifest_dir, "models_image.toml");
    let embedding_data: TomlData<EmbeddingTomlModel> =
        read_toml(&manifest_dir, "models_embedding.toml");
    let reranker_data: TomlData<RerankerTomlModel> =
        read_toml(&manifest_dir, "models_reranker.toml");

    let mut models = Vec::new();

    models.extend(llm_data.models.into_iter().map(|m| UnifiedModel {
        id: m.id,
        aliases: m.aliases,
        name: m.name,
        provider: m.provider,
        kind: "llm",
        context_window: m.context_window,
        max_output_tokens: m.max_output_tokens,
        tier: m.tier,
        tiers: m.tiers,
        capabilities: m.capabilities,
        reasoning_efforts: m.reasoning_efforts,
        reasoning_effort_default: m.reasoning_effort_default,
        adaptive_reasoning: m.adaptive_reasoning,
        reasoning_budget_tokens_min: m.reasoning_budget_tokens_min,
        reasoning_budget_tokens_max: m.reasoning_budget_tokens_max,
        embedding_dimensions: None,
        image_max_resolution: None,
        reranker_max_documents: None,
        outdated: m.outdated,
    }));

    models.extend(image_data.models.into_iter().map(|m| UnifiedModel {
        id: m.id,
        aliases: m.aliases,
        name: m.name,
        provider: m.provider,
        kind: "image",
        context_window: m.context_window,
        max_output_tokens: None,
        tier: None,
        tiers: Vec::new(),
        capabilities: m.capabilities,
        reasoning_efforts: Vec::new(),
        reasoning_effort_default: None,
        adaptive_reasoning: false,
        reasoning_budget_tokens_min: None,
        reasoning_budget_tokens_max: None,
        embedding_dimensions: None,
        image_max_resolution: m.image_max_resolution,
        reranker_max_documents: None,
        outdated: m.outdated,
    }));

    models.extend(embedding_data.models.into_iter().map(|m| UnifiedModel {
        id: m.id,
        aliases: m.aliases,
        name: m.name,
        provider: m.provider,
        kind: "embedding",
        context_window: m.context_window,
        max_output_tokens: None,
        tier: None,
        tiers: Vec::new(),
        capabilities: m.capabilities,
        reasoning_efforts: Vec::new(),
        reasoning_effort_default: None,
        adaptive_reasoning: false,
        reasoning_budget_tokens_min: None,
        reasoning_budget_tokens_max: None,
        embedding_dimensions: Some(m.embedding_dimensions),
        image_max_resolution: None,
        reranker_max_documents: None,
        outdated: m.outdated,
    }));

    models.extend(reranker_data.models.into_iter().map(|m| UnifiedModel {
        id: m.id,
        aliases: m.aliases,
        name: m.name,
        provider: m.provider,
        kind: "reranker",
        context_window: m.context_window,
        max_output_tokens: None,
        tier: None,
        tiers: Vec::new(),
        capabilities: m.capabilities,
        reasoning_efforts: Vec::new(),
        reasoning_effort_default: None,
        adaptive_reasoning: false,
        reasoning_budget_tokens_min: None,
        reasoning_budget_tokens_max: None,
        embedding_dimensions: None,
        image_max_resolution: None,
        reranker_max_documents: m.reranker_max_documents,
        outdated: m.outdated,
    }));

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("generated.rs");

    let mut code = String::new();

    code.push_str("const MODELS: &[aither_core::llm::model::ModelInfo] = &[\n");
    for model in &models {
        if model.kind != "llm" {
            continue;
        }

        let tiers: Vec<&str> = if !model.tiers.is_empty() {
            model.tiers.iter().map(|s| s.as_str()).collect()
        } else if let Some(ref t) = model.tier {
            vec![t.as_str()]
        } else {
            panic!("LLM model {} has no tier or tiers defined", model.id);
        };

        let tiers_code = tiers
            .iter()
            .map(|t| tier_variant(t))
            .collect::<Vec<_>>()
            .join(", ");

        let ability_variants = model
            .capabilities
            .iter()
            .filter_map(|c| ability_variant(c))
            .collect::<Vec<_>>()
            .join(", ");

        code.push_str(&format!(
            "    aither_core::llm::model::ModelInfo {{\n        id: {:?},\n        name: {:?},\n        provider: {:?},\n        context_window: {},\n        max_output_tokens: {},\n        tiers: &[{}],\n        abilities: &[{}],\n        outdated: {},\n    }},\n",
            model.id,
            model.name,
            model.provider,
            model.context_window,
            match model.max_output_tokens {
                Some(v) => format!("Some({v})"),
                None => "None".to_string(),
            },
            tiers_code,
            ability_variants,
            model.outdated,
        ));
    }
    code.push_str("];\n\n");

    let mut aliases: Vec<(String, String)> = Vec::new();
    for model in &models {
        for alias in &model.aliases {
            aliases.push((alias.to_lowercase(), model.id.clone()));
        }
    }
    aliases.sort_by(|a, b| a.0.cmp(&b.0));

    code.push_str("const ALIASES: &[(&str, &str)] = &[\n");
    for (alias, canonical) in &aliases {
        code.push_str(&format!("    ({:?}, {:?}),\n", alias, canonical));
    }
    code.push_str("];\n\n");

    code.push_str("const MODEL_CAPABILITIES: &[(&str, &[&str])] = &[\n");
    for model in &models {
        let caps = model
            .capabilities
            .iter()
            .map(|c| format!("{:?}", c.to_lowercase()))
            .collect::<Vec<_>>()
            .join(", ");
        code.push_str(&format!("    ({:?}, &[{}]),\n", model.id, caps));
    }
    code.push_str("];\n\n");

    code.push_str("const MODEL_KINDS: &[(&str, &str)] = &[\n");
    for model in &models {
        code.push_str(&format!("    ({:?}, {:?}),\n", model.id, model.kind));
    }
    code.push_str("];\n\n");

    code.push_str("const MODEL_META: &[(&str, &str, &str, u32)] = &[\n");
    for model in &models {
        code.push_str(&format!(
            "    ({:?}, {:?}, {:?}, {}),\n",
            model.id, model.name, model.provider, model.context_window
        ));
    }
    code.push_str("];\n\n");

    code.push_str("const MODEL_REASONING_EFFORTS: &[(&str, &[&str])] = &[\n");
    for model in &models {
        let efforts = model
            .reasoning_efforts
            .iter()
            .map(|c| format!("{:?}", c.to_lowercase()))
            .collect::<Vec<_>>()
            .join(", ");
        code.push_str(&format!("    ({:?}, &[{}]),\n", model.id, efforts));
    }
    code.push_str("];\n\n");

    code.push_str(
        "const MODEL_REASONING_META: &[(&str, Option<&str>, bool, Option<u32>, Option<u32>)] = &[\n",
    );
    for model in &models {
        let default_effort_code = model
            .reasoning_effort_default
            .as_ref()
            .map(|v| format!("Some({:?})", v.to_lowercase()))
            .unwrap_or_else(|| "None".to_string());
        let min_code = model
            .reasoning_budget_tokens_min
            .map(|v| format!("Some({v})"))
            .unwrap_or_else(|| "None".to_string());
        let max_code = model
            .reasoning_budget_tokens_max
            .map(|v| format!("Some({v})"))
            .unwrap_or_else(|| "None".to_string());
        code.push_str(&format!(
            "    ({:?}, {}, {}, {}, {}),\n",
            model.id, default_effort_code, model.adaptive_reasoning, min_code, max_code
        ));
    }
    code.push_str("];\n\n");

    code.push_str(
        "const MODEL_SPECIALIZED_META: &[(&str, Option<u32>, Option<&str>, Option<u32>)] = &[\n",
    );
    for model in &models {
        let emb_code = model
            .embedding_dimensions
            .map(|v| format!("Some({v})"))
            .unwrap_or_else(|| "None".to_string());
        let img_code = model
            .image_max_resolution
            .as_ref()
            .map(|v| format!("Some({:?})", v))
            .unwrap_or_else(|| "None".to_string());
        let rerank_code = model
            .reranker_max_documents
            .map(|v| format!("Some({v})"))
            .unwrap_or_else(|| "None".to_string());
        code.push_str(&format!(
            "    ({:?}, {}, {}, {}),\n",
            model.id, emb_code, img_code, rerank_code
        ));
    }
    code.push_str("];\n");

    fs::write(&out_path, code).expect("Failed to write generated.rs");
}

fn read_toml<T: for<'de> Deserialize<'de>>(manifest_dir: &str, file_name: &str) -> T {
    let path = Path::new(manifest_dir).join(file_name);
    let content =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {}", file_name));
    toml::from_str(&content).unwrap_or_else(|_| panic!("Failed to parse {}", file_name))
}

fn tier_variant(tier: &str) -> &'static str {
    match tier.to_lowercase().as_str() {
        "flagship" => "aither_core::llm::model::ModelTier::Flagship",
        "balanced" => "aither_core::llm::model::ModelTier::Balanced",
        "fast" => "aither_core::llm::model::ModelTier::Fast",
        _ => panic!("Unknown tier: {tier}"),
    }
}

fn ability_variant(cap: &str) -> Option<&'static str> {
    match cap.to_lowercase().as_str() {
        "vision" => Some("aither_core::llm::model::Ability::Vision"),
        "tool_use" => Some("aither_core::llm::model::Ability::ToolUse"),
        "audio" => Some("aither_core::llm::model::Ability::Audio"),
        "video" => Some("aither_core::llm::model::Ability::Video"),
        "pdf" => Some("aither_core::llm::model::Ability::Pdf"),
        "web_search" => Some("aither_core::llm::model::Ability::WebSearch"),
        "code_execution" => Some("aither_core::llm::model::Ability::CodeExecution"),
        "reasoning" => Some("aither_core::llm::model::Ability::Reasoning"),
        "image_generation" => Some("aither_core::llm::model::Ability::ImageGeneration"),
        "always_reasoning" => None,
        "embedding" => None,
        "reranker" => None,
        _ => panic!("Unknown capability: {cap}"),
    }
}
