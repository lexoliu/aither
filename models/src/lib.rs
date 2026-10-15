//! Model registry with capabilities, context windows, and metadata for popular LLMs.
//!
//! This crate provides a database of known AI models sourced from `LiteLLM` upstream,
//! with their capabilities, context window sizes, pricing, and provider information.
//!
//! # Example
//!
//! ```rust
//! use aither_models::{lookup, Ability};
//!
//! if let Some(entry) = lookup("gpt-4o") {
//!     assert!(entry.max_input_tokens().unwrap_or(0) > 0);
//!     assert!(entry.has_ability(Ability::Vision));
//! }
//! ```

pub mod convert;
mod registry;
#[cfg(feature = "remote")]
pub mod remote;
pub(crate) mod tier;
pub mod types;

pub use aither_core::llm::model::Ability;
pub use registry::ModelRegistry;
pub use tier::ModelTier;
pub use types::{ModelEntry, ModelMode, Pricing, Provider};

/// Look up a model by `LiteLLM` ID or a dated/versioned alias.
///
/// Returns `None` when the ID is unknown to the registry — see
/// [`ModelRegistry::lookup`] for the normalization rules.
#[must_use]
pub fn lookup(model_id: &str) -> Option<&'static ModelEntry> {
    ModelRegistry::bundled().lookup(model_id)
}

/// Get all models for a provider.
pub fn models_for_provider(provider: &Provider) -> impl Iterator<Item = &'static ModelEntry> + '_ {
    ModelRegistry::bundled().models_for_provider(provider)
}

/// Get all models with a specific ability.
pub fn models_with_ability(ability: Ability) -> impl Iterator<Item = &'static ModelEntry> {
    ModelRegistry::bundled().models_with_ability(ability)
}

/// Get all models matching a mode.
pub fn models_by_mode(mode: ModelMode) -> impl Iterator<Item = &'static ModelEntry> {
    ModelRegistry::bundled().models_by_mode(mode)
}

/// Get all known models.
pub fn all_models() -> impl Iterator<Item = &'static ModelEntry> {
    ModelRegistry::bundled().all()
}

/// Capability/cost tier of a model in the bundled registry.
///
/// Unknown models classify as [`ModelTier::Balanced`].
#[must_use]
pub fn tier(model_id: &str) -> ModelTier {
    ModelRegistry::bundled().tier(model_id)
}

#[cfg(test)]
mod tests {
    //! The bundled snapshot is refreshed from `LiteLLM` upstream, so these
    //! tests only assert properties every valid snapshot has. Lookup and
    //! tier behaviour is tested on a fixture in `registry` and `tier`.

    use super::*;

    #[test]
    fn bundled_matches_runtime_conversion_of_snapshot() {
        let runtime =
            ModelRegistry::from_litellm_json(include_bytes!("../data/litellm_snapshot.json"));
        let bundled: Vec<_> = all_models().map(ModelEntry::litellm_id).collect();
        let converted: Vec<_> = runtime.all().map(ModelEntry::litellm_id).collect();
        assert_eq!(bundled, converted);
    }

    #[test]
    fn bundled_registry_is_populated() {
        let registry = ModelRegistry::bundled();
        assert!(
            registry.len() > 100,
            "Expected 100+ models, got {}",
            registry.len()
        );
        assert_ne!(models_by_mode(ModelMode::Chat).count(), 0);
    }

    #[test]
    fn every_entry_is_reachable_by_its_litellm_id() {
        for entry in all_models() {
            assert!(
                lookup(entry.litellm_id()).is_some(),
                "{} is not reachable by its own key",
                entry.litellm_id()
            );
            assert!(
                entry.litellm_id().ends_with(entry.id()),
                "{} does not end with its canonical id {}",
                entry.litellm_id(),
                entry.id()
            );
        }
    }

    #[test]
    fn query_helpers_return_only_matching_entries() {
        assert!(models_with_ability(Ability::Vision).all(|m| m.has_ability(Ability::Vision)));
        assert!(models_by_mode(ModelMode::Chat).all(|m| m.mode() == ModelMode::Chat));
        assert!(models_for_provider(&Provider::OpenAI).all(|m| *m.provider() == Provider::OpenAI));
    }

    #[test]
    fn image_generation_entries_classify_as_image_tier() {
        for entry in models_by_mode(ModelMode::ImageGeneration) {
            assert!(entry.has_ability(Ability::ImageGeneration));
            assert_eq!(
                ModelRegistry::bundled().tier_of(entry),
                ModelTier::ImageGeneration
            );
        }
    }
}
