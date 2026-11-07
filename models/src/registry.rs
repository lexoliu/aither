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
    /// `@YYYYMMDD`, and compact `-MMDD` checkpoint dates, rewrites a
    /// `vMAJOR-MINOR` token to `vMAJOR.MINOR`, and collapses `vMAJOR.MINOR`
    /// to `vMAJOR` when the dotted form is also a miss, so dated or versioned
    /// aliases resolve to the base model's record. It never binds a bare
    /// query to a provider-scoped key (`github_copilot/claude-sonnet-4` only
    /// answers `github_copilot/` lookups) and never changes the model family:
    /// an unknown or retired ID returns `None`.
    #[must_use]
    pub fn lookup(&self, model_id: &str) -> Option<&ModelEntry> {
        let mut id = model_id.to_lowercase();
        loop {
            if let Some(&idx) = self.index.get(id.as_str()) {
                return Some(&self.entries[idx]);
            }
            id = next_lookup_form(&id)?;
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

/// Next more-general form of a lookup id, or `None` when none apply.
///
/// Order: trailing version/date suffix, then `vMAJOR-MINOR` → `vMAJOR.MINOR`,
/// then `vMAJOR.MINOR` → `vMAJOR`. Each step is tried only after the previous
/// form missed in the index.
fn next_lookup_form(id: &str) -> Option<String> {
    if let Some(stripped) = strip_version_suffix(id) {
        return Some(stripped.to_owned());
    }
    let minor = find_v_minor(id)?;
    if minor.sep_byte == b'-' {
        let mut dotted = id.to_owned();
        dotted.replace_range(minor.sep..=minor.sep, ".");
        return Some(dotted);
    }
    let mut collapsed = String::with_capacity(id.len() - (minor.minor_end - minor.sep));
    collapsed.push_str(&id[..minor.sep]);
    collapsed.push_str(&id[minor.minor_end..]);
    Some(collapsed)
}

/// Strip one trailing version suffix from a model ID: `-vN` / `-vN:M`
/// (Bedrock), `-YYYYMMDD` / `-YYYY-MM-DD` (dated snapshots), `@YYYYMMDD`
/// (Vertex), or compact `-MMDD` (month 01–12, day 01–31). Returns the base
/// ID, or `None` when no suffix is recognized.
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
    if n > 5 && bytes[n - 5] == b'-' && is_compact_month_day(&bytes[n - 4..]) {
        return Some(&id[..n - 5]);
    }
    None
}

/// Four ASCII digits that are a calendar month and day (`01`–`12`, `01`–`31`).
fn is_compact_month_day(digits: &[u8]) -> bool {
    if digits.len() != 4 || !digits.iter().all(u8::is_ascii_digit) {
        return false;
    }
    let month = (digits[0] - b'0') * 10 + (digits[1] - b'0');
    let day = (digits[2] - b'0') * 10 + (digits[3] - b'0');
    (1..=12).contains(&month) && (1..=31).contains(&day)
}

/// Location of a `vMAJOR-MINOR` or `vMAJOR.MINOR` token inside a model id.
struct VMinor {
    /// Byte index of the `-` or `.` between major and minor.
    sep: usize,
    /// Byte index after the minor digits.
    minor_end: usize,
    /// The separator between major and minor.
    sep_byte: u8,
}

/// First `vMAJOR-MINOR` / `vMAJOR.MINOR` token whose minor is one or two digits.
fn find_v_minor(id: &str) -> Option<VMinor> {
    let bytes = id.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let at_token_start = i == 0 || bytes[i - 1] == b'-' || bytes[i - 1] == b'/';
        if bytes[i] != b'v' || !at_token_start {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        if j >= bytes.len() || !bytes[j].is_ascii_digit() {
            i += 1;
            continue;
        }
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        let Some(&sep_byte) = bytes.get(j) else {
            i += 1;
            continue;
        };
        if sep_byte != b'-' && sep_byte != b'.' {
            i += 1;
            continue;
        }
        let mut k = j + 1;
        let minor_start = k;
        while k < bytes.len() && bytes[k].is_ascii_digit() {
            k += 1;
        }
        let minor_len = k - minor_start;
        if !(1..=2).contains(&minor_len) {
            i += 1;
            continue;
        }
        if k != bytes.len() && bytes[k] != b'-' {
            i += 1;
            continue;
        }
        return Some(VMinor {
            sep: j,
            minor_end: k,
            sep_byte,
        });
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
        "PerMessageEffort" => Some(Ability::PerMessageEffort),
        "MidConversationSystem" => Some(Ability::MidConversationSystem),
        _ => None,
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Registry built from a small hand-written `LiteLLM` dataset, so lookup
    /// and tier behaviour is tested independently of the bundled snapshot,
    /// which upstream refreshes can change at any time.
    pub fn fixture() -> ModelRegistry {
        ModelRegistry::from_litellm_json(include_bytes!("../testdata/litellm_fixture.json"))
    }

    fn litellm_id_of(registry: &ModelRegistry, query: &str) -> Option<String> {
        registry
            .lookup(query)
            .map(|entry| entry.litellm_id().to_owned())
    }

    /// Mid-conversation system messages follow Anthropic's documented model
    /// list.
    #[test]
    fn mid_conversation_system_follows_the_documented_models() {
        for (id, expected) in [
            ("claude-opus-5-5", true),
            ("claude-opus-4-8", true),
            ("claude-sonnet-5-5", true),
            ("claude-fable-5", true),
            ("claude-sonnet-5", false),
            ("gpt-6-sol", false),
        ] {
            let entry = crate::lookup(id).unwrap_or_else(|| panic!("{id} is in the snapshot"));
            assert_eq!(
                entry.has_ability(Ability::MidConversationSystem),
                expected,
                "{id}"
            );
        }
    }

    /// Per-message effort follows the vendors' documented model lists, on
    /// the platforms they document it for.
    #[test]
    fn per_message_effort_follows_the_documented_models_and_platforms() {
        use crate::convert::takes_per_message_effort;
        for (id, expected) in [
            ("claude-opus-5-5", true),
            ("claude-sonnet-5-5", true),
            ("claude-fable-5-1", true),
            ("claude-sonnet-5", false),
            ("claude-fable-5", false),
            ("gpt-6-sol", true),
            ("gpt-6.1-sol", true),
            ("gpt-5", false),
        ] {
            let entry = crate::lookup(id).unwrap_or_else(|| panic!("{id} is in the snapshot"));
            assert_eq!(
                entry.has_ability(Ability::PerMessageEffort),
                expected,
                "{id}"
            );
        }
        assert!(takes_per_message_effort(
            "vertex_ai",
            "claude-opus-5-5@default"
        ));
        assert!(!takes_per_message_effort(
            "bedrock",
            "anthropic.claude-opus-5-5"
        ));
        assert!(!takes_per_message_effort("azure", "gpt-6-sol"));
    }

    #[test]
    fn fixture_skips_sample_spec_and_unknown_modes() {
        let registry = fixture();
        assert_eq!(registry.len(), 13);
        assert!(registry.lookup("sample_spec").is_none());
        assert!(registry.lookup("fixture-unknown-mode").is_none());
    }

    #[test]
    fn exact_lookup_returns_converted_entry() {
        let registry = fixture();
        let entry = registry.lookup("fixture-gpt").expect("fixture-gpt exists");
        assert_eq!(entry.id(), "fixture-gpt");
        assert_eq!(*entry.provider(), Provider::OpenAI);
        assert_eq!(entry.mode(), ModelMode::Chat);
        assert_eq!(entry.max_input_tokens(), Some(128_000));
        assert!(entry.has_ability(Ability::Vision));
        assert!(entry.has_ability(Ability::ToolUse));
        assert!(entry.pricing().input_per_token() > 0.0);
        assert!(entry.pricing().output_per_token() > 0.0);
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let registry = fixture();
        assert_eq!(
            litellm_id_of(&registry, "FIXTURE-GPT").as_deref(),
            Some("fixture-gpt")
        );
        assert_eq!(
            litellm_id_of(&registry, "GITHUB_COPILOT/FIXTURE-HOSTED").as_deref(),
            Some("github_copilot/fixture-hosted")
        );
    }

    #[test]
    fn dated_snapshot_with_own_record_matches_exactly() {
        let registry = fixture();
        assert_eq!(
            litellm_id_of(&registry, "fixture-gpt-2024-05-13").as_deref(),
            Some("fixture-gpt-2024-05-13")
        );
    }

    #[test]
    fn version_suffixes_normalize_to_base_model() {
        let registry = fixture();
        for alias in [
            "fixture-gpt-21000101",
            "fixture-gpt-2099-01-01",
            "fixture-gpt@20240513",
            "fixture-gpt-v2",
            "fixture-gpt-v2:1",
            "fixture-gpt-1017",
        ] {
            assert_eq!(
                litellm_id_of(&registry, alias).as_deref(),
                Some("fixture-gpt"),
                "{alias} should normalize to fixture-gpt"
            );
        }
        // Normalization stays within the queried family.
        assert_eq!(
            litellm_id_of(&registry, "fixture-gpt-mini-20250101").as_deref(),
            Some("fixture-gpt-mini")
        );
    }

    #[test]
    fn hyphenated_v_minor_and_compact_date_normalize_to_base() {
        let registry = fixture();
        for alias in [
            "fixture-v4-flash-1017",
            "fixture-v4-1-flash",
            "fixture-v4.1-flash",
            "fixture-v4-1-flash-1017",
        ] {
            assert_eq!(
                litellm_id_of(&registry, alias).as_deref(),
                Some("fixture-v4-flash"),
                "{alias} should normalize to fixture-v4-flash"
            );
            assert_eq!(
                registry
                    .lookup(alias)
                    .and_then(ModelEntry::max_input_tokens),
                Some(1_000_000),
                "{alias} should inherit the 1M window"
            );
        }
    }

    #[test]
    fn hyphenated_v_minor_prefers_dotted_record_when_present() {
        let registry = ModelRegistry::from_litellm_json(
            br#"{
                "fixture-v4-flash": {
                    "litellm_provider": "deepseek",
                    "mode": "chat",
                    "max_input_tokens": 1000000,
                    "input_cost_per_token": 3e-07,
                    "output_cost_per_token": 1.2e-06
                },
                "fixture-v4.1-flash": {
                    "litellm_provider": "deepseek",
                    "mode": "chat",
                    "max_input_tokens": 1048576,
                    "input_cost_per_token": 3e-07,
                    "output_cost_per_token": 1.2e-06
                }
            }"#,
        );
        assert_eq!(
            litellm_id_of(&registry, "fixture-v4-1-flash").as_deref(),
            Some("fixture-v4.1-flash")
        );
        assert_eq!(
            registry
                .lookup("fixture-v4-1-flash")
                .and_then(ModelEntry::max_input_tokens),
            Some(1_048_576)
        );
    }

    #[test]
    fn compact_month_day_must_be_a_valid_calendar_date() {
        let registry = fixture();
        assert!(registry.lookup("fixture-gpt-1301").is_none());
        assert!(registry.lookup("fixture-gpt-0001").is_none());
        assert!(registry.lookup("fixture-gpt-4096").is_none());
    }

    #[test]
    fn numeric_minor_without_v_prefix_does_not_collapse() {
        let registry = fixture();
        assert!(registry.lookup("fixture-gpt-4-1").is_none());
        assert!(registry.lookup("fixture-gpt-4.1").is_none());
    }

    #[test]
    fn normalization_never_changes_model_family() {
        let registry = fixture();
        assert!(registry.lookup("fixture-gpt-turbo").is_none());
        assert!(registry.lookup("fixture-gpt-mini-turbo").is_none());
    }

    #[test]
    fn unknown_id_returns_none() {
        let registry = fixture();
        assert!(registry.lookup("no-such-model-xyz").is_none());
        assert!(registry.lookup("no-such-model-xyz-20250101").is_none());
    }

    #[test]
    fn bare_id_does_not_bind_provider_scoped_record() {
        let registry = fixture();
        // Only a hosted (`github_copilot/`, `azure/`) record exists, so the
        // bare ID — dated or not — is a miss rather than a fallback.
        assert!(registry.lookup("fixture-hosted").is_none());
        assert!(registry.lookup("fixture-hosted-20250514").is_none());
        assert!(registry.lookup("fixture-azure-only").is_none());
        assert!(registry.lookup("fixture-bedrock-v1:0").is_none());
    }

    #[test]
    fn provider_scoped_query_normalizes_within_its_namespace() {
        let registry = fixture();
        assert_eq!(
            litellm_id_of(&registry, "github_copilot/fixture-hosted-20250514").as_deref(),
            Some("github_copilot/fixture-hosted")
        );
        assert_eq!(
            litellm_id_of(&registry, "azure/fixture-azure-only").as_deref(),
            Some("azure/fixture-azure-only")
        );
        assert_eq!(
            litellm_id_of(&registry, "bedrock/fixture-bedrock-v1:0").as_deref(),
            Some("bedrock/fixture-bedrock-v1:0")
        );
        // Another provider's namespace does not answer for this one.
        assert!(registry.lookup("azure/fixture-hosted").is_none());
    }

    #[test]
    fn first_party_prefix_claims_bare_id() {
        let registry = fixture();
        assert_eq!(
            litellm_id_of(&registry, "fixture-gemini").as_deref(),
            Some("gemini/fixture-gemini")
        );
        assert_eq!(
            litellm_id_of(&registry, "fixture-gemini-2099-01-01").as_deref(),
            Some("gemini/fixture-gemini")
        );
    }

    #[test]
    fn unprefixed_record_wins_bare_id_over_first_party_prefix() {
        let registry = fixture();
        // `anthropic/fixture-claude-mid` precedes the unprefixed record in
        // iteration order; the unprefixed key still owns the bare ID.
        assert_eq!(
            litellm_id_of(&registry, "fixture-claude-mid").as_deref(),
            Some("fixture-claude-mid")
        );
        assert_eq!(
            litellm_id_of(&registry, "anthropic/fixture-claude-mid").as_deref(),
            Some("anthropic/fixture-claude-mid")
        );
    }

    #[test]
    fn query_helpers_filter_by_mode_ability_and_provider() {
        let registry = fixture();
        let image: Vec<_> = registry
            .models_by_mode(ModelMode::ImageGeneration)
            .map(ModelEntry::litellm_id)
            .collect();
        assert_eq!(image, ["fixture-image"]);
        let reasoning: Vec<_> = registry
            .models_with_ability(Ability::Reasoning)
            .map(ModelEntry::litellm_id)
            .collect();
        assert_eq!(reasoning, ["fixture-flagship"]);
        let copilot: Vec<_> = registry
            .models_for_provider(&Provider::Copilot)
            .map(ModelEntry::litellm_id)
            .collect();
        assert_eq!(copilot, ["github_copilot/fixture-hosted"]);
    }
}
