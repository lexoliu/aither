//! Model tier classification inferred from registry metadata.
//!
//! Tiers group models by capability/cost so applications can offer
//! "advanced / balanced / fast" selection without hardcoding model lists.
//! Classification uses the model mode, the output-price distribution of all
//! chat models in the registry, and the context window as fallback signal.

use serde::{Deserialize, Serialize};

use crate::types::{ModelEntry, ModelMode};

/// Capability/cost tier of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    /// Flagship models - highest capability
    Advanced,
    /// Balanced models - good capability/cost ratio
    #[default]
    Balanced,
    /// Fast models - optimized for speed
    Fast,
    /// Image generation models
    ImageGeneration,
}

impl ModelTier {
    /// All tiers for iteration.
    pub const ALL: &'static [Self] = &[
        Self::Advanced,
        Self::Balanced,
        Self::Fast,
        Self::ImageGeneration,
    ];

    /// Human-readable tier name.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Advanced => "Advanced (Flagship)",
            Self::Balanced => "Balanced",
            Self::Fast => "Fast",
            Self::ImageGeneration => "Image Generation",
        }
    }

    /// Short machine-friendly tier name.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Advanced => "advanced",
            Self::Balanced => "balanced",
            Self::Fast => "fast",
            Self::ImageGeneration => "image",
        }
    }
}

/// Output-price tertile breakpoints over a registry's chat models.
///
/// Computed once per registry: prices at the lower and upper third of the
/// distribution. Chat models priced at or below the lower breakpoint classify
/// as [`ModelTier::Fast`], at or above the upper as [`ModelTier::Advanced`].
#[derive(Debug, Clone, Copy)]
pub struct ChatPriceBreakpoints {
    fast_ceiling: f64,
    advanced_floor: f64,
}

impl ChatPriceBreakpoints {
    /// Derive breakpoints from the chat-model price distribution.
    ///
    /// Returns `None` when fewer than three priced chat models exist, in
    /// which case classification falls back to context-window heuristics.
    pub fn compute<'a>(entries: impl Iterator<Item = &'a ModelEntry>) -> Option<Self> {
        let mut prices: Vec<f64> = entries
            .filter(|info| info.mode() == ModelMode::Chat)
            .map(|info| info.pricing().output_per_token())
            .filter(|price| price.is_finite() && *price > 0.0)
            .collect();
        if prices.len() < 3 {
            return None;
        }

        prices.sort_by(f64::total_cmp);
        let low_index = prices.len() / 3;
        let high_index = (prices.len() * 2) / 3;
        Some(Self {
            fast_ceiling: prices[low_index],
            advanced_floor: prices[high_index],
        })
    }

    pub fn classify(self, info: &ModelEntry) -> Option<ModelTier> {
        let output_price = info.pricing().output_per_token();
        if !output_price.is_finite() || output_price <= 0.0 {
            return None;
        }
        if output_price <= self.fast_ceiling {
            return Some(ModelTier::Fast);
        }
        if output_price >= self.advanced_floor {
            return Some(ModelTier::Advanced);
        }
        Some(ModelTier::Balanced)
    }
}

/// Classify an entry given precomputed chat-price breakpoints.
pub fn classify_entry(info: &ModelEntry, breakpoints: Option<ChatPriceBreakpoints>) -> ModelTier {
    match info.mode() {
        ModelMode::ImageGeneration => return ModelTier::ImageGeneration,
        ModelMode::Chat => {}
        _ => return ModelTier::Balanced,
    }

    if let Some(tier) = breakpoints.and_then(|b| b.classify(info)) {
        return tier;
    }

    match info.max_input_tokens() {
        Some(tokens) if tokens <= 32_768 => ModelTier::Fast,
        Some(tokens) if tokens >= 200_000 => ModelTier::Advanced,
        _ => ModelTier::Balanced,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::tests::fixture;

    #[test]
    fn fixture_registry_classifies_by_mode_and_price() {
        let registry = fixture();
        assert_eq!(registry.tier("fixture-image"), ModelTier::ImageGeneration);
        assert_eq!(registry.tier("fixture-embedding"), ModelTier::Balanced);
        // The most expensive chat model lands in the top price tertile,
        // the cheapest in the bottom one.
        assert_eq!(registry.tier("fixture-flagship"), ModelTier::Advanced);
        assert_eq!(registry.tier("fixture-gpt-mini"), ModelTier::Fast);
    }

    #[test]
    fn unpriced_chat_model_falls_back_to_context_window() {
        let registry = fixture();
        // 128k context sits between the Fast and Advanced context bounds.
        assert_eq!(
            registry.tier("github_copilot/fixture-hosted"),
            ModelTier::Balanced
        );
    }

    #[test]
    fn unknown_model_defaults_to_balanced() {
        assert_eq!(
            fixture().tier("definitely-not-a-real-model-xyz"),
            ModelTier::Balanced
        );
    }

    #[test]
    fn serde_roundtrip_matches_lowercase() {
        let json = serde_json::to_string(&ModelTier::Advanced).unwrap();
        assert_eq!(json, "\"advanced\"");
        let tier: ModelTier = serde_json::from_str("\"fast\"").unwrap();
        assert_eq!(tier, ModelTier::Fast);
    }
}
