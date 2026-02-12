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

/// Resolve model ID or alias to canonical model ID.
#[must_use]
fn canonical_model_id(model_id: &str) -> Option<&'static str> {
    let model_lower = model_id.to_lowercase();

    if let Some((id, _, _, _)) = MODEL_META
        .iter()
        .find(|(id, _, _, _)| id.eq_ignore_ascii_case(&model_lower))
    {
        return Some(*id);
    }

    if let Ok(idx) = ALIASES.binary_search_by_key(&model_lower.as_str(), |(alias, _)| *alias) {
        return Some(ALIASES[idx].1);
    }

    MODEL_META
        .iter()
        .filter(|(id, _, _, _)| model_lower.starts_with(&id.to_lowercase()))
        .max_by_key(|(id, _, _, _)| id.len())
        .map(|(id, _, _, _)| *id)
}

/// Look up LLM model info by ID or alias.
///
/// Non-LLM models (image/embedding/reranker) intentionally return `None` here.
#[must_use]
pub fn lookup(model_id: &str) -> Option<&'static ModelInfo> {
    let canonical_id = canonical_model_id(model_id)?;
    MODELS
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(canonical_id))
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

/// Returns metadata-only capability labels for a model id/alias.
#[must_use]
pub fn metadata_capabilities(model_id: &str) -> &'static [&'static str] {
    let Some(id) = canonical_model_id(model_id) else {
        return &[];
    };

    MODEL_CAPABILITIES
        .iter()
        .find_map(|(model_id, caps)| {
            if model_id.eq_ignore_ascii_case(id) {
                Some(*caps)
            } else {
                None
            }
        })
        .unwrap_or(&[])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelMeta {
    pub kind: &'static str,
    pub name: &'static str,
    pub provider: &'static str,
    pub context_window: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecializedMeta {
    pub embedding_dimensions: Option<u32>,
    pub image_max_resolution: Option<&'static str>,
    pub reranker_max_documents: Option<u32>,
}

#[must_use]
pub fn model_kind(model_id: &str) -> Option<&'static str> {
    let id = canonical_model_id(model_id)?;
    MODEL_KINDS.iter().find_map(|(model_id, kind)| {
        if model_id.eq_ignore_ascii_case(id) {
            Some(*kind)
        } else {
            None
        }
    })
}

#[must_use]
pub fn model_meta(model_id: &str) -> Option<ModelMeta> {
    let id = canonical_model_id(model_id)?;
    MODEL_META
        .iter()
        .find_map(|(model_id, name, provider, context_window)| {
            if model_id.eq_ignore_ascii_case(id) {
                Some(ModelMeta {
                    kind: model_kind(id).unwrap_or("llm"),
                    name,
                    provider,
                    context_window: *context_window,
                })
            } else {
                None
            }
        })
}

#[must_use]
pub fn specialized_meta(model_id: &str) -> SpecializedMeta {
    let Some(id) = canonical_model_id(model_id) else {
        return SpecializedMeta {
            embedding_dimensions: None,
            image_max_resolution: None,
            reranker_max_documents: None,
        };
    };

    MODEL_SPECIALIZED_META
        .iter()
        .find_map(
            |(model_id, embedding_dimensions, image_max_resolution, reranker_max_documents)| {
                if model_id.eq_ignore_ascii_case(id) {
                    Some(SpecializedMeta {
                        embedding_dimensions: *embedding_dimensions,
                        image_max_resolution: *image_max_resolution,
                        reranker_max_documents: *reranker_max_documents,
                    })
                } else {
                    None
                }
            },
        )
        .unwrap_or(SpecializedMeta {
            embedding_dimensions: None,
            image_max_resolution: None,
            reranker_max_documents: None,
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningMeta {
    pub default_effort: Option<&'static str>,
    pub adaptive_reasoning: bool,
    pub budget_tokens_min: Option<u32>,
    pub budget_tokens_max: Option<u32>,
}

#[must_use]
pub fn reasoning_efforts(model_id: &str) -> &'static [&'static str] {
    let Some(id) = canonical_model_id(model_id) else {
        return &[];
    };

    MODEL_REASONING_EFFORTS
        .iter()
        .find_map(|(model_id, efforts)| {
            if model_id.eq_ignore_ascii_case(id) {
                Some(*efforts)
            } else {
                None
            }
        })
        .unwrap_or(&[])
}

#[must_use]
pub fn reasoning_meta(model_id: &str) -> ReasoningMeta {
    let Some(id) = canonical_model_id(model_id) else {
        return ReasoningMeta {
            default_effort: None,
            adaptive_reasoning: false,
            budget_tokens_min: None,
            budget_tokens_max: None,
        };
    };

    MODEL_REASONING_META
        .iter()
        .find_map(|(model_id, default_effort, adaptive_reasoning, min, max)| {
            if model_id.eq_ignore_ascii_case(id) {
                Some(ReasoningMeta {
                    default_effort: *default_effort,
                    adaptive_reasoning: *adaptive_reasoning,
                    budget_tokens_min: *min,
                    budget_tokens_max: *max,
                })
            } else {
                None
            }
        })
        .unwrap_or(ReasoningMeta {
            default_effort: None,
            adaptive_reasoning: false,
            budget_tokens_min: None,
            budget_tokens_max: None,
        })
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
