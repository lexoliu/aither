//! Shared conversion logic for `LiteLLM` and `OpenRouter` JSON.
//!
//! This module is included via `#[path]` in build.rs so the same parsing
//! logic runs at compile time and runtime. It depends only on `serde`,
//! `serde_json`, and sibling types.

use serde::Deserialize;

/// Raw `LiteLLM` entry as it appears in the JSON.
#[derive(Debug, Deserialize)]
pub struct RawLiteLLMEntry {
    /// Provider slug as `LiteLLM` reports it.
    #[serde(default)]
    pub litellm_provider: Option<String>,
    /// What the model does: `chat`, `embedding`, `image_generation`, and so on.
    #[serde(default)]
    pub mode: Option<String>,
    /// Largest prompt the model accepts, in tokens.
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    /// Largest completion the model will produce, in tokens.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Combined limit, used when the input/output split is unknown.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// USD charged per input token.
    #[serde(default)]
    pub input_cost_per_token: Option<f64>,
    /// USD charged per output token.
    #[serde(default)]
    pub output_cost_per_token: Option<f64>,
    /// USD per input token served from the prompt cache.
    #[serde(default)]
    pub cache_read_input_token_cost: Option<f64>,
    /// USD per input token written to the prompt cache.
    #[serde(default)]
    pub cache_creation_input_token_cost: Option<f64>,
    /// USD per reasoning token, where billed separately.
    #[serde(default)]
    pub output_cost_per_reasoning_token: Option<f64>,
    /// USD per audio input token.
    #[serde(default)]
    pub input_cost_per_audio_token: Option<f64>,
    /// Date the provider retires the model, as `YYYY-MM-DD`.
    #[serde(default)]
    pub deprecation_date: Option<String>,
    // supports_* flags
    /// Whether the model can call tools.
    #[serde(default)]
    pub supports_function_calling: Option<bool>,
    /// Whether the model accepts images.
    #[serde(default)]
    pub supports_vision: Option<bool>,
    /// Whether the model accepts audio.
    #[serde(default)]
    pub supports_audio_input: Option<bool>,
    /// Whether the model can emit audio.
    #[serde(default)]
    pub supports_audio_output: Option<bool>,
    /// Whether the model accepts video.
    #[serde(default)]
    pub supports_video_input: Option<bool>,
    /// Whether the provider offers built-in web search.
    #[serde(default)]
    pub supports_web_search: Option<bool>,
    /// Whether the model accepts PDFs directly.
    #[serde(default)]
    pub supports_pdf_input: Option<bool>,
    /// Whether the model exposes reasoning or thinking output.
    #[serde(default)]
    pub supports_reasoning: Option<bool>,
    /// Whether the model supports computer-use tooling.
    #[serde(default)]
    pub supports_computer_use: Option<bool>,
    /// Whether the provider caches prompt prefixes.
    #[serde(default)]
    pub supports_prompt_caching: Option<bool>,
    /// Whether an assistant turn may be prefilled.
    #[serde(default)]
    pub supports_assistant_prefill: Option<bool>,
}

/// Known provider prefix → canonical provider mapping.
const PROVIDER_PREFIXES: &[(&str, &str)] = &[
    ("openai/", "openai"),
    ("anthropic/", "anthropic"),
    ("gemini/", "gemini"),
    ("deepseek/", "deepseek"),
    ("xai/", "xai"),
    ("mistral/", "mistral"),
    ("meta-llama/", "meta"),
    ("bedrock/", "bedrock"),
    ("azure/", "azure"),
    ("azure_ai/", "azure"),
    ("vertex_ai/", "vertex_ai"),
    ("groq/", "groq"),
    ("together_ai/", "together_ai"),
    ("fireworks_ai/", "fireworks_ai"),
    ("replicate/", "replicate"),
    ("cohere/", "cohere"),
    ("cohere_chat/", "cohere"),
    ("perplexity/", "perplexity"),
    ("github_copilot/", "github_copilot"),
    ("ollama/", "ollama"),
    ("nvidia_nim/", "nvidia_nim"),
    ("sambanova/", "sambanova"),
    ("cerebras/", "cerebras"),
    ("dashscope/", "alibaba"),
];

/// Strip known provider prefixes to get the canonical model ID.
#[must_use]
pub fn strip_provider_prefix(litellm_key: &str) -> &str {
    for &(prefix, _) in PROVIDER_PREFIXES {
        if let Some(rest) = litellm_key.strip_prefix(prefix) {
            return rest;
        }
    }
    litellm_key
}

/// The Claude models Anthropic's effort documentation lists for a
/// per-message effort change (an effort-only `system` message under the
/// `mid-conversation-output-config-2026-07-01` beta). `LiteLLM` carries no
/// such flag, so it is curated here.
const ANTHROPIC_PER_MESSAGE_EFFORT_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
];

/// The Claude models whose Messages API takes a system message inside
/// `messages`. `LiteLLM` carries no such flag, so it is curated here.
const ANTHROPIC_MID_CONVERSATION_SYSTEM_MODELS: &[&str] = &[
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-mythos-5-1",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
];

/// Whether the Claude API takes a mid-conversation system message for
/// `canonical_id` on the platform `provider` names.
#[must_use]
pub fn takes_mid_conversation_system(provider: &str, canonical_id: &str) -> bool {
    provider == "anthropic" && ANTHROPIC_MID_CONVERSATION_SYSTEM_MODELS.contains(&canonical_id)
}

/// The platforms Anthropic documents the per-message effort change on: the
/// Claude API and Google Cloud.
const ANTHROPIC_PER_MESSAGE_EFFORT_PLATFORMS: &[&str] = &["anthropic", "vertex_ai"];

/// The first GPT generation `OpenAI` documents the Responses API
/// `configuration_update` input item for; every later generation takes it.
const OPENAI_FIRST_PER_MESSAGE_EFFORT_GENERATION: u32 = 6;

/// Whether the vendor documents a per-message reasoning effort change for
/// `canonical_id` on the platform `provider` names.
#[must_use]
pub fn takes_per_message_effort(provider: &str, canonical_id: &str) -> bool {
    if ANTHROPIC_PER_MESSAGE_EFFORT_PLATFORMS.contains(&provider) {
        // Vertex AI pins a version after `@`.
        let model = canonical_id
            .split_once('@')
            .map_or(canonical_id, |(model, _)| model);
        return ANTHROPIC_PER_MESSAGE_EFFORT_MODELS.contains(&model);
    }
    if provider == "openai" {
        return gpt_generation(canonical_id)
            .is_some_and(|generation| generation >= OPENAI_FIRST_PER_MESSAGE_EFFORT_GENERATION);
    }
    false
}

/// The major generation of a `gpt-<N>…` model id.
fn gpt_generation(model: &str) -> Option<u32> {
    let rest = model.strip_prefix("gpt-")?;
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest.get(..digits)?.parse().ok()
}

/// Map a `LiteLLM` `litellm_provider` string to a provider tag.
#[must_use]
pub fn parse_provider(provider_str: &str) -> String {
    let mapped = match provider_str {
        "openai" | "text-completion-openai" | "chatgpt" => "openai",
        "anthropic" => "anthropic",
        "gemini" | "palm" => "gemini",
        "deepseek" => "deepseek",
        "xai" => "xai",
        "mistral" | "codestral" | "text-completion-codestral" => "mistral",
        "meta_llama" => "meta",
        s if s.starts_with("bedrock") => "bedrock",
        "azure" | "azure_ai" | "azure_text" => "azure",
        s if s.starts_with("vertex_ai") => "vertex_ai",
        "groq" => "groq",
        "together_ai" => "together_ai",
        s if s.starts_with("fireworks_ai") => "fireworks_ai",
        "replicate" => "replicate",
        "cohere" | "cohere_chat" => "cohere",
        "perplexity" => "perplexity",
        "github_copilot" => "github_copilot",
        "dashscope" => "alibaba",
        other => return other.to_string(),
    };
    mapped.to_string()
}

/// Map a `LiteLLM` `mode` string to a mode tag.
#[must_use]
pub fn parse_mode(mode_str: &str) -> Option<&'static str> {
    match mode_str {
        "chat" | "responses" => Some("chat"),
        "embedding" => Some("embedding"),
        "image_generation" | "image_edit" => Some("image_generation"),
        "audio_transcription" => Some("audio_transcription"),
        "audio_speech" => Some("audio_speech"),
        "rerank" => Some("rerank"),
        "moderation" => Some("moderation"),
        "completion" => Some("completion"),
        "video_generation" => Some("video_generation"),
        "search" | "vector_store" => Some("search"),
        "ocr" => Some("ocr"),
        _ => None,
    }
}

/// Intermediate representation used by both build.rs and runtime.
#[derive(Debug, Clone)]
pub struct ConvertedEntry {
    /// Model id exactly as `LiteLLM` keys it, provider prefix included.
    pub litellm_id: String,
    /// Model id with the provider prefix stripped.
    pub canonical_id: String,
    /// Canonical provider slug.
    pub provider: String,
    /// What the model does, as a canonical mode string.
    pub mode: String,
    /// Largest prompt the model accepts, in tokens.
    pub max_input_tokens: Option<u32>,
    /// Largest completion the model produces, in tokens.
    pub max_output_tokens: Option<u32>,
    /// USD per input token.
    pub input_cost_per_token: f64,
    /// USD per output token.
    pub output_cost_per_token: f64,
    /// USD per cached input token read.
    pub cache_read_per_token: Option<f64>,
    /// USD per input token written to the cache.
    pub cache_write_per_token: Option<f64>,
    /// USD per reasoning token, where billed separately.
    pub reasoning_per_token: Option<f64>,
    /// USD per image token.
    pub image_per_token: Option<f64>,
    /// Capability names this model advertises.
    pub abilities: Vec<&'static str>,
    /// Retirement date, as `YYYY-MM-DD`.
    pub deprecation_date: Option<String>,
}

/// Convert a single `LiteLLM` JSON entry into our intermediate representation.
#[must_use]
pub fn convert_litellm_entry(key: &str, raw: &RawLiteLLMEntry) -> Option<ConvertedEntry> {
    let provider_str = raw.litellm_provider.as_deref()?;
    let mode_str = raw.mode.as_deref()?;
    let mode = parse_mode(mode_str)?;
    let provider = parse_provider(provider_str);
    let canonical_id = strip_provider_prefix(key).to_string();

    let max_input = raw.max_input_tokens;
    let max_output = raw.max_output_tokens.or(raw.max_tokens);

    let mut abilities = Vec::new();
    if raw.supports_function_calling == Some(true) {
        abilities.push("ToolUse");
    }
    if raw.supports_vision == Some(true) {
        abilities.push("Vision");
    }
    if raw.supports_audio_input == Some(true) {
        abilities.push("Audio");
    }
    if raw.supports_audio_output == Some(true) {
        abilities.push("AudioOutput");
    }
    if raw.supports_video_input == Some(true) {
        abilities.push("Video");
    }
    if raw.supports_web_search == Some(true) {
        abilities.push("WebSearch");
    }
    if raw.supports_pdf_input == Some(true) {
        abilities.push("Pdf");
    }
    if raw.supports_reasoning == Some(true) {
        abilities.push("Reasoning");
    }
    if raw.supports_computer_use == Some(true) {
        abilities.push("ComputerUse");
    }
    if raw.supports_prompt_caching == Some(true) {
        abilities.push("PromptCaching");
    }
    if raw.supports_assistant_prefill == Some(true) {
        abilities.push("AssistantPrefill");
    }
    if mode == "image_generation" {
        abilities.push("ImageGeneration");
    }
    if takes_per_message_effort(&provider, &canonical_id) {
        abilities.push("PerMessageEffort");
    }
    if takes_mid_conversation_system(&provider, &canonical_id) {
        abilities.push("MidConversationSystem");
    }

    Some(ConvertedEntry {
        litellm_id: key.to_string(),
        canonical_id,
        provider,
        mode: mode.to_string(),
        max_input_tokens: max_input,
        max_output_tokens: max_output,
        input_cost_per_token: raw.input_cost_per_token.unwrap_or(0.0),
        output_cost_per_token: raw.output_cost_per_token.unwrap_or(0.0),
        cache_read_per_token: raw.cache_read_input_token_cost,
        cache_write_per_token: raw.cache_creation_input_token_cost,
        reasoning_per_token: raw.output_cost_per_reasoning_token,
        image_per_token: raw.input_cost_per_audio_token,
        abilities,
        deprecation_date: raw.deprecation_date.clone(),
    })
}

/// Parse the full `LiteLLM` JSON blob and return converted entries.
///
/// # Panics
///
/// Panics if `json_bytes` is not a JSON object. The only caller supplies either
/// the committed snapshot or a payload already validated by the caller.
#[must_use]
pub fn parse_litellm_json(json_bytes: &[u8]) -> Vec<ConvertedEntry> {
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(json_bytes).expect("invalid LiteLLM JSON");

    let mut entries = Vec::new();
    for (key, value) in &map {
        if key == "sample_spec" {
            continue;
        }
        let Ok(raw) = serde_json::from_value::<RawLiteLLMEntry>(value.clone()) else {
            continue;
        };
        if let Some(entry) = convert_litellm_entry(key, &raw) {
            entries.push(entry);
        }
    }
    entries
}
