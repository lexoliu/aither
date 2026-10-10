//! Model registry with indexed lookup.

use std::collections::HashMap;
use std::sync::OnceLock;

use aither_core::llm::model::Ability;

use crate::convert;
use crate::tier::{ChatPriceBreakpoints, ModelTier, classify_entry};
use crate::types::{ModelEntry, ModelMode, Pricing, Provider};

// The registry table below is emitted by `build.rs` from the LiteLLM snapshot.
// It is machine-written and never edited by hand; the generator applies its own
// `#[allow]` attributes because style lints on generated text can only be
// "fixed" by changing the generator's formatting.
include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// A searchable collection of model entries.
#[derive(Debug)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
    index: HashMap<String, usize>,
    chat_price_breakpoints: Option<ChatPriceBreakpoints>,
}

impl ModelRegistry {
    /// Returns the compile-time bundled registry.
    #[must_use]
    pub fn bundled() -> &'static Self {
        static INSTANCE: OnceLock<ModelRegistry> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            let entries: Vec<ModelEntry> = ENTRIES.to_vec();
            Self::from_entries(entries)
        })
    }

    /// Build a registry from parsed `LiteLLM` JSON bytes.
    #[must_use]
    pub fn from_litellm_json(json: &[u8]) -> Self {
        let converted = convert::parse_litellm_json(json);
        let entries = converted.into_iter().filter_map(convert_to_entry).collect();
        Self::from_entries(entries)
    }

    fn from_entries(entries: Vec<ModelEntry>) -> Self {
        let mut index = HashMap::with_capacity(entries.len() * 2);
        // Canonical IDs index first-party records only. Unprefixed keys
        // (`claude-sonnet-4-5`) come first so a bare query resolves the
        // vendor's own record rather than whichever hosted variant
        // (`github_copilot/claude-sonnet-4`, `bedrock/…`) appears first.
        for (i, entry) in entries.iter().enumerate() {
            if entry.litellm_id() == entry.id() {
                index.entry(entry.id().to_lowercase()).or_insert(i);
            }
        }
        for (i, entry) in entries.iter().enumerate() {
            if is_first_party(entry) {
                index.entry(entry.id().to_lowercase()).or_insert(i);
            }
        }
        // Every entry stays reachable by its full `LiteLLM` key.
        for (i, entry) in entries.iter().enumerate() {
            index.entry(entry.litellm_id().to_lowercase()).or_insert(i);
        }
        let chat_price_breakpoints = ChatPriceBreakpoints::compute(entries.iter());
        Self {
            entries,
            index,
            chat_price_breakpoints,
        }
    }

    /// Capability/cost tier of a model, classified against this registry's
    /// chat-model price distribution. Unknown models classify as
    /// [`ModelTier::Balanced`].
    #[must_use]
    pub fn tier(&self, model_id: &str) -> ModelTier {
        self.lookup(model_id)
            .map_or(ModelTier::Balanced, |entry| self.tier_of(entry))
    }

    /// Capability/cost tier of a specific registry entry.
    #[must_use]
    pub fn tier_of(&self, entry: &ModelEntry) -> ModelTier {
        classify_entry(entry, self.chat_price_breakpoints)
    }

    /// Exact match on the `LiteLLM` key, then version-suffix normalization.
    ///
    /// Normalization strips trailing `-vN[:M]`, `-YYYYMMDD`, `-YYYY-MM-DD`,
    /// and `@YYYYMMDD` suffixes, so dated or versioned aliases resolve to
    /// the base model's record. It never binds a bare query to a
    /// provider-scoped key (`github_copilot/claude-sonnet-4` only answers
    /// `github_copilot/` lookups) and never changes the model family: an
    /// unknown or retired ID returns `None`.
    #[must_use]
    pub fn lookup(&self, model_id: &str) -> Option<&ModelEntry> {
        let mut id = model_id.to_lowercase();
        loop {
            if let Some(&idx) = self.index.get(id.as_str()) {
                return Some(&self.entries[idx]);
            }
            let stripped = strip_version_suffix(&id)?;
            id = stripped.to_owned();
        }
    }

    /// All models from a given provider.
    pub fn models_for_provider(&self, provider: &Provider) -> impl Iterator<Item = &ModelEntry> {
        let provider = provider.clone();
        self.entries
            .iter()
            .filter(move |e| *e.provider() == provider)
    }

    /// All models matching a given mode.
    pub fn models_by_mode(&self, mode: ModelMode) -> impl Iterator<Item = &ModelEntry> {
        self.entries.iter().filter(move |e| e.mode() == mode)
    }

    /// All models with a specific ability.
    pub fn models_with_ability(&self, ability: Ability) -> impl Iterator<Item = &ModelEntry> {
        self.entries.iter().filter(move |e| e.has_ability(ability))
    }

    /// Iterate over all entries.
    pub fn all(&self) -> impl Iterator<Item = &ModelEntry> {
        self.entries.iter()
    }

    /// Total number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// `LiteLLM` prefixes under which a vendor publishes its own models
/// (`gemini/gemini-2.5-flash` is Google's record for its own model).
/// Hosted variants — `azure/`, `bedrock/`, `github_copilot/`, … — are
/// reachable by their full key but do not claim the bare canonical ID.
const FIRST_PARTY_PREFIXES: &[&str] = &[
    "anthropic/",
    "cohere/",
    "cohere_chat/",
    "dashscope/",
    "deepseek/",
    "gemini/",
    "meta-llama/",
    "mistral/",
    "openai/",
    "xai/",
];

/// Whether an entry is a first-party record allowed to claim the bare
/// canonical ID: either its key carries no provider prefix at all, or the
/// prefix is the vendor's own namespace.
fn is_first_party(entry: &ModelEntry) -> bool {
    entry.litellm_id() == entry.id()
        || FIRST_PARTY_PREFIXES
            .iter()
            .any(|prefix| entry.litellm_id().starts_with(prefix))
}

/// Strip one trailing version suffix from a model ID: `-vN` / `-vN:M`
/// (Bedrock), `-YYYYMMDD` / `-YYYY-MM-DD` (dated snapshots), or `@YYYYMMDD`
/// (Vertex). Returns the base ID, or `None` when no suffix is recognized.
fn strip_version_suffix(id: &str) -> Option<&str> {
    let bytes = id.as_bytes();
    let n = bytes.len();
    if let Some((base, suffix)) = id.rsplit_once('-')
        && let Some(rest) = suffix.strip_prefix('v')
        && rest
            .split(':')
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
    {
        return Some(base);
    }
    if n > 9 && matches!(bytes[n - 9], b'-' | b'@') && bytes[n - 8..].iter().all(u8::is_ascii_digit)
    {
        return Some(&id[..n - 9]);
    }
    if n > 11
        && bytes[n - 11] == b'-'
        && bytes[n - 6] == b'-'
        && bytes[n - 3] == b'-'
        && bytes[n - 10..n - 6].iter().all(u8::is_ascii_digit)
        && bytes[n - 5..n - 3].iter().all(u8::is_ascii_digit)
        && bytes[n - 2..].iter().all(u8::is_ascii_digit)
    {
        return Some(&id[..n - 11]);
    }
    None
}

/// Convert a `ConvertedEntry` from the shared convert module into a `ModelEntry`.
fn convert_to_entry(c: convert::ConvertedEntry) -> Option<ModelEntry> {
    let provider = parse_provider_enum(&c.provider);
    let mode = parse_mode_enum(&c.mode)?;

    let mut pricing = crate::types::Pricing::new(c.input_cost_per_token, c.output_cost_per_token);
    if let Some(v) = c.cache_read_per_token {
        pricing = pricing.with_cache_read(v);
    }
    if let Some(v) = c.cache_write_per_token {
        pricing = pricing.with_cache_write(v);
    }
    if let Some(v) = c.reasoning_per_token {
        pricing = pricing.with_reasoning(v);
    }
    if let Some(v) = c.image_per_token {
        pricing = pricing.with_image(v);
    }

    let abilities: Vec<Ability> = c
        .abilities
        .iter()
        .filter_map(|a| parse_ability(a))
        .collect();

    let mut entry = ModelEntry::new(c.litellm_id, c.canonical_id, provider, mode)
        .with_pricing(pricing)
        .with_abilities(abilities);

    if let Some(max_in) = c.max_input_tokens {
        entry = entry.with_max_input_tokens(max_in);
    }
    if let Some(max_out) = c.max_output_tokens {
        entry = entry.with_max_output_tokens(max_out);
    }
    if let Some(date) = c.deprecation_date {
        entry = entry.with_deprecation_date(date);
    }

    Some(entry)
}

fn parse_provider_enum(s: &str) -> Provider {
    match s {
        "openai" => Provider::OpenAI,
        "anthropic" => Provider::Anthropic,
        "gemini" => Provider::Google,
        "deepseek" => Provider::DeepSeek,
        "xai" => Provider::XAI,
        "mistral" => Provider::Mistral,
        "meta" => Provider::Meta,
        "alibaba" => Provider::Alibaba,
        "bedrock" => Provider::Bedrock,
        "azure" => Provider::Azure,
        "vertex_ai" => Provider::VertexAI,
        "groq" => Provider::Groq,
        "together_ai" => Provider::Together,
        "fireworks_ai" => Provider::Fireworks,
        "replicate" => Provider::Replicate,
        "cohere" => Provider::Cohere,
        "perplexity" => Provider::Perplexity,
        "github_copilot" => Provider::Copilot,
        other => Provider::Other(std::borrow::Cow::Owned(other.to_string())),
    }
}

fn parse_mode_enum(s: &str) -> Option<ModelMode> {
    match s {
        "chat" => Some(ModelMode::Chat),
        "embedding" => Some(ModelMode::Embedding),
        "image_generation" => Some(ModelMode::ImageGeneration),
        "audio_transcription" => Some(ModelMode::AudioTranscription),
        "audio_speech" => Some(ModelMode::AudioSpeech),
        "rerank" => Some(ModelMode::Rerank),
        "moderation" => Some(ModelMode::Moderation),
        "completion" => Some(ModelMode::Completion),
        "video_generation" => Some(ModelMode::VideoGeneration),
        "search" => Some(ModelMode::Search),
        "ocr" => Some(ModelMode::Ocr),
        _ => None,
    }
}

fn parse_ability(s: &str) -> Option<Ability> {
    match s {
        "ToolUse" => Some(Ability::ToolUse),
        "Vision" => Some(Ability::Vision),
        "Audio" => Some(Ability::Audio),
        "AudioOutput" => Some(Ability::AudioOutput),
        "Video" => Some(Ability::Video),
        "WebSearch" => Some(Ability::WebSearch),
        "Pdf" => Some(Ability::Pdf),
        "CodeExecution" => Some(Ability::CodeExecution),
        "Reasoning" => Some(Ability::Reasoning),
        "ImageGeneration" => Some(Ability::ImageGeneration),
        "ComputerUse" => Some(Ability::ComputerUse),
        "PromptCaching" => Some(Ability::PromptCaching),
        "AssistantPrefill" => Some(Ability::AssistantPrefill),
        _ => None,
    }
}
