//! AI language model configuration and profiling types.
//!
//! This module provides types for configuring AI language models, including
//! parameters for model behavior, pricing information, and capability profiles.
//!
//! # Examples
//!
//! ## Creating a model profile
//!
//! ```rust
//! use ai_types::llm::model::{Profile, Ability, Pricing};
//!
//! let mut pricing = Pricing::default();
//!
//! pricing.prompt = 0.01; // $0.01 per 1K prompt tokens
//! pricing.completion = 0.03; // $0.03 per 1K completion tokens
//! pricing.request = 0.01; // $0.01 per request
//! pricing.image = 0.1; // $0.1 per image
//! pricing.web_search = 0.05; // $0.05 per web search
//! pricing.internal_reasoning = 0.003; // $0.003 per internal reasoning
//! pricing.input_cache_read = 0.0005; // $0.0005 per input cache read
//! pricing.input_cache_write = 0.001; // $0.001 per input cache write
//!
//! let profile = Profile::new("gpt-4", "GPT-4 model", 8192)
//!     .with_ability(Ability::ToolUse)
//!     .with_ability(Ability::Vision)
//!     .with_pricing(pricing);
//! ```
//!
//! ## Configuring model parameters
//!
//! ```rust
//! use ai_types::llm::model::Parameters;
//!
//! let params = Parameters::default()
//!     .temperature(0.7)
//!     .top_p(0.9)
//!     .max_tokens(1000)
//!     .seed(42);
//! ```

use alloc::{string::String, vec::Vec};
use schemars::Schema;

/// Parameters for configuring the behavior of a language model.
///
/// This struct contains various parameters that can be used to control
/// how a language model generates responses. All parameters are optional
/// and use the builder pattern for easy configuration.
///
/// # Examples
///
/// ```rust
/// use ai_types::llm::model::Parameters;
///
/// let params = Parameters::default()
///     .temperature(0.7)
///     .top_p(0.9)
///     .max_tokens(1000)
///     .seed(42);
/// ```
#[derive(Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Parameters {
    /// Sampling temperature.
    ///
    /// Controls randomness in generation. Higher values (e.g., 1.0) make output more random,
    /// lower values (e.g., 0.1) make it more deterministic.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    ///
    /// Only consider tokens with cumulative probability up to this value.
    /// Typical values are between 0.9 and 1.0.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    ///
    /// Only consider the k most likely tokens at each step.
    pub top_k: Option<u32>,
    /// Frequency penalty to reduce repetition.
    ///
    /// Positive values penalize tokens that have already appeared.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty to encourage new tokens.
    ///
    /// Positive values encourage the model to talk about new topics.
    pub presence_penalty: Option<f32>,
    /// Repetition penalty to penalize repeated tokens.
    ///
    /// Values > 1.0 discourage repetition, values < 1.0 encourage it.
    pub repetition_penalty: Option<f32>,
    /// Minimum probability for nucleus sampling.
    ///
    /// Alternative to `top_p` that sets a minimum threshold for token probabilities.
    pub min_p: Option<f32>,
    /// Top-a sampling parameter.
    ///
    /// Adaptive sampling that adjusts the number of considered tokens.
    pub top_a: Option<f32>,
    /// Random seed for reproducibility.
    ///
    /// Use the same seed to get deterministic outputs.
    pub seed: Option<u32>,
    /// Maximum number of tokens to generate.
    ///
    /// Limits the length of the generated response.
    pub max_tokens: Option<u32>,
    /// Biases for specific logits.
    ///
    /// Each tuple contains a token string and its bias value.
    pub logit_bias: Option<Vec<(String, f32)>>,
    /// Whether to return log probabilities.
    ///
    /// When true, the model returns probability information for tokens.
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return.
    ///
    /// Only used when logprobs is true.
    pub top_logprobs: Option<u8>,
    /// Stop sequences to end generation.
    ///
    /// Generation stops when any of these strings are encountered.
    pub stop: Option<Vec<String>>,
    /// Tool choices available to the model.
    ///
    /// Specifies which tools the model is allowed to use.
    pub tool_choice: Option<Vec<String>>,

    /// Whether to enable structured outputs.
    ///
    /// When true, the model will attempt to return outputs in a structured format (e.g., JSON).
    pub structured_outputs: bool,

    /// The expected response format schema.
    ///
    /// When set, the model will attempt to return outputs matching this schema.
    pub response_format: Option<Schema>,
}

macro_rules! impl_with_methods {
    (
        impl $ty:ty {
            $($field:ident : $field_ty:ty),* $(,)?
        }
    ) => {
        impl $ty {
            $(
                /// Sets the parameter value using a builder pattern.
                ///
                /// # Arguments
                ///
                /// * `value` - The value to set for this parameter
                #[allow(clippy::missing_const_for_fn)]
                #[must_use] pub fn $field(mut self, value: $field_ty) -> Self {
                    self.$field = Some(value);
                    self
                }
            )*
        }
    };
}

impl_with_methods! {
    impl Parameters {
        temperature: f32,
        top_p: f32,
        top_k: u32,
        frequency_penalty: f32,
        presence_penalty: f32,
        repetition_penalty: f32,
        min_p: f32,
        top_a: f32,
        seed: u32,
        max_tokens: u32,
        logit_bias: Vec<(String, f32)>,
        logprobs: bool,
        top_logprobs: u8,
        stop: Vec<String>,
    }
}

/// Represents a language model's profile, including its name, description, abilities, context length, and optional pricing.
///
/// A model profile provides comprehensive information about a language model's
/// capabilities, limitations, and pricing structure. This allows applications
/// to make informed decisions about which model to use for specific tasks.
///
/// # Examples
///
/// ```rust
/// use ai_types::llm::model::{Profile, Ability, Pricing};
///
/// let profile = Profile::new("gpt-4", "GPT-4 Turbo", 128000)
///     .with_ability(Ability::ToolUse)
///     .with_ability(Ability::Vision);
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Profile {
    /// The name of the model.
    pub name: String,
    /// The author of the model.
    pub author: String,
    /// The slug of the model.
    pub slug: String,
    /// A description of the model.
    pub description: String,
    /// The abilities supported by the model.
    pub abilities: Vec<Ability>,
    /// The maximum context length supported by the model.
    pub context_length: u32,
    /// Optional pricing information for the model.
    pub pricing: Option<Pricing>,
}

/// Pricing information for a model's various capabilities (unit: USD).
///
/// This struct contains detailed pricing information for different aspects
/// of model usage. All prices are in USD and typically represent costs
/// per unit (token, request, image, etc.).
///
/// # Examples
///
/// ```rust
/// use ai_types::llm::model::Pricing;
///
/// let mut pricing = Pricing::default();
///
/// pricing.prompt = 0.01; // $0.01 per 1K prompt tokens
/// pricing.completion = 0.03; // $0.03 per 1K completion tokens
/// pricing.image = 0.25; // $0.25 per image
/// pricing.web_search = 0.005; // $0.005 per search
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Pricing {
    /// Price per prompt token.
    pub prompt: f64,
    /// Price per completion token.
    pub completion: f64,
    /// Price per request.
    pub request: f64,
    /// Price per image processed.
    pub image: f64,
    /// Price per web search.
    pub web_search: f64,
    /// Price for internal reasoning.
    pub internal_reasoning: f64,
    /// Price for reading from input cache.
    pub input_cache_read: f64,
    /// Price for writing to input cache.
    pub input_cache_write: f64,
}

/// Indicates which parameters are supported by a model.
///
/// This struct is used to communicate which configuration parameters
/// a specific model supports, allowing applications to adjust their
/// requests accordingly.
///
/// # Examples
///
/// ```rust
/// use ai_types::llm::model::SupportedParameters;
///
/// let mut support = SupportedParameters::default();
///
/// support.temperature = true;
/// support.max_tokens = true;
/// support.top_p = true;
/// support.stop = true;
/// support.seed = true;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[allow(clippy::struct_excessive_bools)]
#[non_exhaustive]
pub struct SupportedParameters {
    /// Whether `max_tokens` is supported.
    pub max_tokens: bool,
    /// Whether temperature is supported.
    pub temperature: bool,
    /// Whether `top_p` is supported.
    pub top_p: bool,
    /// Whether reasoning is supported.
    pub reasoning: bool,
    /// Whether including reasoning is supported.
    pub include_reasoning: bool,
    /// Whether structured outputs are supported.
    pub structured_outputs: bool,
    /// Whether response format is supported.
    pub response_format: bool,
    /// Whether stop sequences are supported.
    pub stop: bool,
    /// Whether frequency penalty is supported.
    pub frequency_penalty: bool,
    /// Whether presence penalty is supported.
    pub presence_penalty: bool,
    /// Whether seed is supported.
    pub seed: bool,
}

impl Profile {
    /// Creates a new `Profile` with the given name, description, and context length.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the model (e.g., "gpt-4", "claude-3-opus")
    /// * `description` - A human-readable description of the model
    /// * `context_length` - Maximum number of tokens the model can process
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::model::Profile;
    ///
    /// let profile = Profile::new("gpt-4", "GPT-4 Turbo", 128000);
    /// ```
    pub fn new(
        name: impl Into<String>,
        author: impl Into<String>,
        slug: impl Into<String>,
        description: impl Into<String>,
        context_length: u32,
    ) -> Self {
        Self {
            name: name.into(),
            author: author.into(),
            slug: slug.into(),
            description: description.into(),
            abilities: Vec::new(),
            context_length,
            pricing: None,
        }
    }

    /// Adds a single ability to the profile.
    ///
    /// # Arguments
    ///
    /// * `ability` - The ability to add to this profile
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::model::{Profile, Ability};
    ///
    /// let profile = Profile::new("vision-model", "A vision-capable model", 8192)
    ///     .with_ability(Ability::Vision);
    /// ```
    #[must_use]
    pub fn with_ability(self, ability: Ability) -> Self {
        self.with_abilities([ability])
    }

    /// Adds multiple abilities to the profile.
    ///
    /// # Arguments
    ///
    /// * `abilities` - An iterable collection of abilities to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::model::{Profile, Ability};
    ///
    /// let abilities = [Ability::ToolUse, Ability::Vision, Ability::Audio];
    /// let profile = Profile::new("multimodal", "A multimodal model", 32768)
    ///     .with_abilities(abilities);
    /// ```
    #[must_use]
    pub fn with_abilities(mut self, abilities: impl IntoIterator<Item = Ability>) -> Self {
        self.abilities.extend(abilities);
        self
    }

    /// Sets the pricing information for the profile.
    ///
    /// # Arguments
    ///
    /// * `pricing` - The pricing structure for this model
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ai_types::llm::model::{Profile, Pricing};
    ///
    /// let mut pricing = Pricing::default();
    ///
    /// pricing.prompt = 0.01;
    /// pricing.completion = 0.03;
    ///
    /// let profile = Profile::new("paid-model", "A paid model", 4096)
    ///     .with_pricing(pricing);
    /// ```
    #[must_use]
    pub const fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }
}

/// Represents the capabilities that a language model may support.
///
/// This enum defines the various advanced capabilities that modern language
/// models can possess beyond basic text generation. These capabilities can
/// be used to determine which models are suitable for specific use cases.
///
/// # Examples
///
/// ```rust
/// use ai_types::llm::model::Ability;
///
/// // Check if a model supports vision
/// let abilities = [Ability::Vision, Ability::ToolUse];
/// let has_vision = abilities.contains(&Ability::Vision);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Ability {
    /// The model can use external tools/functions.
    ToolUse,
    /// The model can process and understand images.
    Vision,
    /// The model can process and understand audio.
    Audio,
    /// The model can perform web searches natively.
    WebSearch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_creation() {
        let profile = Profile::new("Test model", "test", "test-model", "A test model", 4096);

        assert_eq!(profile.name, "test-model");
        assert_eq!(profile.description, "A test model");
        assert_eq!(profile.context_length, 4096);
        assert!(profile.abilities.is_empty());
        assert!(profile.pricing.is_none());
    }

    #[test]
    fn profile_with_single_ability() {
        let profile = Profile::new(
            "Test vision model",
            "test",
            "vision-model",
            "A vision model",
            8192,
        )
        .with_ability(Ability::Vision);

        assert_eq!(profile.abilities.len(), 1);
        assert_eq!(profile.abilities[0], Ability::Vision);
    }

    #[test]
    fn profile_with_multiple_abilities() {
        let abilities = [Ability::ToolUse, Ability::Vision, Ability::Audio];
        let profile = Profile::new(
            "Test",
            "test",
            "multimodal-model",
            "A multimodal model",
            16384,
        )
        .with_abilities(abilities);

        assert_eq!(profile.abilities.len(), 3);
        assert_eq!(profile.abilities, abilities);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn profile_with_pricing() {
        let pricing = Pricing {
            prompt: 0.0001,
            completion: 0.0002,
            request: 0.001,
            image: 0.01,
            web_search: 0.005,
            internal_reasoning: 0.0003,
            input_cache_read: 0.00005,
            input_cache_write: 0.0001,
        };

        let profile = Profile::new(
            "Test paid model",
            "test",
            "paid-model",
            "A paid model",
            2048,
        )
        .with_pricing(pricing);

        assert!(profile.pricing.is_some());
        let profile_pricing = profile.pricing.unwrap();
        assert_eq!(profile_pricing.prompt, 0.0001);
        assert_eq!(profile_pricing.completion, 0.0002);
        assert_eq!(profile_pricing.request, 0.001);
        assert_eq!(profile_pricing.image, 0.01);
        assert_eq!(profile_pricing.web_search, 0.005);
        assert_eq!(profile_pricing.internal_reasoning, 0.0003);
        assert_eq!(profile_pricing.input_cache_read, 0.00005);
        assert_eq!(profile_pricing.input_cache_write, 0.0001);
    }

    #[test]
    fn profile_builder_pattern() {
        let pricing = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let profile = Profile::new("Test", "test", "full-model", "A full-featured model", 32768)
            .with_ability(Ability::ToolUse)
            .with_ability(Ability::Vision)
            .with_abilities([Ability::Audio, Ability::WebSearch])
            .with_pricing(pricing);

        assert_eq!(profile.name, "full-model");
        assert_eq!(profile.description, "A full-featured model");
        assert_eq!(profile.context_length, 32768);
        assert_eq!(profile.abilities.len(), 4);
        assert!(profile.abilities.contains(&Ability::ToolUse));
        assert!(profile.abilities.contains(&Ability::Vision));
        assert!(profile.abilities.contains(&Ability::Audio));
        assert!(profile.abilities.contains(&Ability::WebSearch));
        assert!(profile.pricing.is_some());
    }

    #[test]
    fn ability_equality() {
        assert_eq!(Ability::ToolUse, Ability::ToolUse);
        assert_eq!(Ability::Vision, Ability::Vision);
        assert_eq!(Ability::Audio, Ability::Audio);
        assert_eq!(Ability::WebSearch, Ability::WebSearch);

        assert_ne!(Ability::ToolUse, Ability::Vision);
        assert_ne!(Ability::Audio, Ability::WebSearch);
    }

    #[test]
    fn ability_debug() {
        let ability = Ability::ToolUse;
        let debug_str = alloc::format!("{ability:?}");
        assert!(debug_str.contains("ToolUse"));
    }

    #[test]
    fn profile_debug() {
        let profile = Profile::new("Test model", "test", "debug-model", "A debug model", 1024);
        let debug_str = alloc::format!("{profile:?}");
        assert!(debug_str.contains("debug-model"));
        assert!(debug_str.contains("A debug model"));
        assert!(debug_str.contains("1024"));
    }

    #[test]
    fn profile_clone() {
        let original = Profile::new("Test model", "test", "original", "Original model", 2048)
            .with_ability(Ability::Vision);
        let cloned = original.clone();

        assert_eq!(original.name, cloned.name);
        assert_eq!(original.description, cloned.description);
        assert_eq!(original.context_length, cloned.context_length);
        assert_eq!(original.abilities, cloned.abilities);
    }

    #[test]
    fn pricing_debug() {
        let pricing = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let debug_str = alloc::format!("{pricing:?}");
        assert!(debug_str.contains("0.001"));
        assert!(debug_str.contains("0.002"));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn pricing_clone() {
        let original = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };
        let cloned = original.clone();

        assert_eq!(original.prompt, cloned.prompt);
        assert_eq!(original.completion, cloned.completion);
        assert_eq!(original.request, cloned.request);
        assert_eq!(original.image, cloned.image);
        assert_eq!(original.web_search, cloned.web_search);
        assert_eq!(original.internal_reasoning, cloned.internal_reasoning);
        assert_eq!(original.input_cache_read, cloned.input_cache_read);
        assert_eq!(original.input_cache_write, cloned.input_cache_write);
    }

    #[test]
    fn pricing_equality() {
        let pricing1 = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let pricing2 = Pricing {
            prompt: 0.001,
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        let pricing3 = Pricing {
            prompt: 0.002, // Different value
            completion: 0.002,
            request: 0.01,
            image: 0.1,
            web_search: 0.05,
            internal_reasoning: 0.003,
            input_cache_read: 0.0005,
            input_cache_write: 0.001,
        };

        assert_eq!(pricing1, pricing2);
        assert_ne!(pricing1, pricing3);
    }

    #[test]
    fn supported_parameters() {
        let params = SupportedParameters {
            max_tokens: true,
            temperature: true,
            top_p: false,
            structured_outputs: true,
            stop: true,
            presence_penalty: true,
            ..Default::default()
        };

        assert!(params.max_tokens);
        assert!(params.temperature);
        assert!(!params.top_p);
    }

    #[test]
    fn parameters_debug() {
        let params = Parameters::default()
            .temperature(0.7)
            .top_p(0.9)
            .top_k(40)
            .seed(42)
            .max_tokens(1000);

        let debug_str = alloc::format!("{params:?}");
        assert!(debug_str.contains("0.7"));
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("1000"));
    }
}
