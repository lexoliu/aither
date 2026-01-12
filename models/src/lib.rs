//! Model registry with capabilities, context windows, and metadata for popular LLMs.
//!
//! This crate provides a database of known LLM models with their capabilities,
//! context window sizes, and performance tiers. Use this when provider APIs
//! don't expose model metadata.
//!
//! # Example
//!
//! ```rust
//! use aither_models::{lookup, ModelTier, Ability};
//!
//! // Lookup by canonical ID
//! if let Some(info) = lookup("gpt-4o") {
//!     println!("Context window: {}", info.context_window);
//!     println!("Tiers: {:?}", info.tiers);
//!     println!("Is fast tier: {}", info.has_tier(ModelTier::Fast));
//!     println!("Has vision: {}", info.abilities.contains(&Ability::Vision));
//! }
//!
//! // Lookup by alias
//! if let Some(info) = lookup("deepseek-chat") {
//!     assert_eq!(info.id, "deepseek-v3"); // Resolved to canonical ID
//! }
//! ```

// Re-export types from core for convenience
pub use aither_core::llm::model::{Ability, ModelInfo, ModelTier};

// Include generated code from build.rs
include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Look up model info by ID or alias.
///
/// Performs matching in this order:
/// 1. Exact match on canonical ID (case-insensitive)
/// 2. Exact match on alias (case-insensitive)
/// 3. Prefix match on canonical ID (e.g., "gpt-4o-2024-05-13" matches "gpt-4o")
#[must_use]
pub fn lookup(model_id: &str) -> Option<&'static ModelInfo> {
    let model_lower = model_id.to_lowercase();

    // 1. Exact match on canonical ID
    if let Some(info) = MODELS
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(&model_lower))
    {
        return Some(info);
    }

    // 2. Check aliases (binary search since ALIASES is sorted)
    if let Ok(idx) = ALIASES.binary_search_by_key(&model_lower.as_str(), |(alias, _)| *alias) {
        let canonical_id = ALIASES[idx].1;
        if let Some(info) = MODELS.iter().find(|m| m.id == canonical_id) {
            return Some(info);
        }
    }

    // 3. Prefix match - find the longest matching canonical ID
    MODELS
        .iter()
        .filter(|m| model_lower.starts_with(&m.id.to_lowercase()))
        .max_by_key(|m| m.id.len())
}

/// Get all models for a provider.
#[must_use]
pub fn models_for_provider(provider: &str) -> impl Iterator<Item = &'static ModelInfo> {
    let provider_lower = provider.to_lowercase();
    MODELS
        .iter()
        .filter(move |m| m.provider.eq_ignore_ascii_case(&provider_lower))
}

/// Get all models with a specific ability.
#[must_use]
pub fn models_with_ability(ability: Ability) -> impl Iterator<Item = &'static ModelInfo> {
    MODELS
        .iter()
        .filter(move |m| m.abilities.contains(&ability))
}

/// Get all models of a specific tier.
#[must_use]
pub fn models_by_tier(tier: ModelTier) -> impl Iterator<Item = &'static ModelInfo> {
    MODELS.iter().filter(move |m| m.has_tier(tier))
}

/// Get all known models.
#[must_use]
pub fn all_models() -> impl Iterator<Item = &'static ModelInfo> {
    MODELS.iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_exact() {
        let info = lookup("gpt-4o").unwrap();
        assert_eq!(info.id, "gpt-4o");
        assert_eq!(info.context_window, 128_000);
        assert!(info.has_tier(ModelTier::Fast));
    }

    #[test]
    fn test_lookup_prefix() {
        let info = lookup("gpt-4o-2024-05-13").unwrap();
        assert_eq!(info.id, "gpt-4o");
    }

    #[test]
    fn test_lookup_case_insensitive() {
        let info = lookup("GPT-4O").unwrap();
        assert_eq!(info.id, "gpt-4o");
    }

    #[test]
    fn test_lookup_alias() {
        // deepseek-chat is an alias for deepseek-v3
        let info = lookup("deepseek-chat").unwrap();
        assert_eq!(info.id, "deepseek-v3");
    }

    #[test]
    fn test_lookup_alias_case_insensitive() {
        let info = lookup("DEEPSEEK-CHAT").unwrap();
        assert_eq!(info.id, "deepseek-v3");
    }

    #[test]
    fn test_lookup_claude_dated() {
        let info = lookup("claude-3-5-sonnet-20241022").unwrap();
        assert_eq!(info.id, "claude-3-5-sonnet");
        assert_eq!(info.context_window, 200_000);
    }

    #[test]
    fn test_lookup_gemini() {
        let info = lookup("gemini-2.5-flash").unwrap();
        assert_eq!(info.context_window, 1_048_576);
        assert!(info.abilities.contains(&Ability::Vision));
    }

    #[test]
    fn test_models_for_provider() {
        let openai_models: Vec<_> = models_for_provider("openai").collect();
        assert!(openai_models.len() > 5);
        assert!(openai_models.iter().all(|m| m.provider == "openai"));
    }

    #[test]
    fn test_models_with_ability() {
        let vision_models: Vec<_> = models_with_ability(Ability::Vision).collect();
        assert!(!vision_models.is_empty());
        assert!(
            vision_models
                .iter()
                .all(|m| m.abilities.contains(&Ability::Vision))
        );
    }

    #[test]
    fn test_models_by_tier() {
        let flagship: Vec<_> = models_by_tier(ModelTier::Flagship).collect();
        assert!(!flagship.is_empty());
        assert!(flagship.iter().all(|m| m.has_tier(ModelTier::Flagship)));
    }

    #[test]
    fn test_deepseek_reasoner_alias() {
        // deepseek-reasoner is an alias for deepseek-r1
        let info = lookup("deepseek-reasoner").unwrap();
        assert_eq!(info.id, "deepseek-r1");
        assert!(info.abilities.contains(&Ability::Reasoning));
    }
}
