//! The parameters a host gives the requests aither sends each model.
//!
//! A request aither builds starts from the parameters of the model that
//! serves it, as the host set them, and aither then sets the fields it owns
//! for that request:
//!
//! - [`cache`](Parameters::cache) on an agent's turns and compactions, from
//!   its [`PromptCache`](crate::context::PromptCache);
//! - [`structured_outputs`](Parameters::structured_outputs) on a structured
//!   request — a compaction, or the subagent router;
//! - [`max_tokens`](Parameters::max_tokens) and [`stream`](Parameters::stream)
//!   on a prompt-cache keep-alive.
//!
//! Every other field reaches the provider exactly as the host set it.

use aither_core::llm::model::Parameters;

use crate::model_group::ModelTier;

/// The parameters of each model tier of something created on one model.
///
/// The advanced tier is the model it was created with. A tier given a model
/// of its own carries that model's parameters; a tier still on the created-
/// with model runs with that model's parameters, whenever either was set.
///
/// Each set is boxed: they are set once and only cloned onto requests, and
/// held inline they would weigh on every future that holds an agent or its
/// builder.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TierParameters {
    /// The parameters of the model created with.
    shared: Box<Parameters>,
    /// The balanced tier's own model's parameters, once it has one.
    balanced: Option<Box<Parameters>>,
    /// The fast tier's own model's parameters, once it has one.
    fast: Option<Box<Parameters>>,
}

impl TierParameters {
    /// Sets the parameters of the model created with.
    pub fn set_shared(&mut self, parameters: Parameters) {
        *self.shared = parameters;
    }

    /// Gives the balanced tier a model of its own, with `parameters`.
    pub fn set_balanced(&mut self, parameters: Parameters) {
        self.balanced = Some(Box::new(parameters));
    }

    /// Gives the fast tier a model of its own, with `parameters`.
    pub fn set_fast(&mut self, parameters: Parameters) {
        self.fast = Some(Box::new(parameters));
    }

    /// The parameters of the model behind `tier`.
    pub fn get(&self, tier: ModelTier) -> &Parameters {
        match tier {
            ModelTier::Advanced => &self.shared,
            ModelTier::Balanced => self.balanced.as_deref().unwrap_or(&self.shared),
            ModelTier::Fast => self.fast.as_deref().unwrap_or(&self.shared),
        }
    }
}

/// A model and the parameters of the requests aither sends it, boxed like
/// [`TierParameters`].
#[derive(Debug, Clone)]
pub struct WithParameters<M> {
    pub model: M,
    pub parameters: Box<Parameters>,
}

/// Checks on what a request carried to the provider.
#[cfg(test)]
pub mod test_support {
    use aither_core::llm::LLMRequest;
    use aither_core::llm::model::{Parameters, ReasoningEffort};

    /// Parameters a host gives a model, setting no field aither owns.
    pub fn host_parameters(effort: ReasoningEffort, temperature: f32) -> Parameters {
        Parameters::default()
            .reasoning_effort(effort)
            .include_reasoning(true)
            .temperature(temperature)
            .websearch(true)
    }

    /// `request`'s parameters without the cache controls and structured
    /// output aither sets per request: what is left of the host's.
    pub fn host_part(request: &LLMRequest) -> Parameters {
        let mut parameters = request.parameters().clone().without_cache();
        parameters.structured_outputs = false;
        parameters
    }

    /// Asserts `request` carries the prompt-cache controls of an agent's
    /// request: an `OpenAI` cache key and Claude breakpoints.
    pub fn assert_cache_controls(request: &LLMRequest) {
        let cache = &request.parameters().cache;
        assert!(
            cache
                .openai
                .as_ref()
                .is_some_and(|openai| openai.key.is_some()),
            "the request carries the prompt-cache key: {cache:?}"
        );
        assert!(
            cache.claude.is_some(),
            "the request carries the Claude cache breakpoints: {cache:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::model::ReasoningEffort;

    #[test]
    fn a_tier_on_the_created_with_model_takes_its_parameters_in_either_order() {
        let shared = Parameters::default().reasoning_effort(ReasoningEffort::High);
        let fast = Parameters::default().temperature(0.1);

        let mut before = TierParameters::default();
        before.set_fast(fast.clone());
        before.set_shared(shared.clone());
        let mut after = TierParameters::default();
        after.set_shared(shared.clone());
        after.set_fast(fast.clone());

        for tiers in [before, after] {
            assert_eq!(tiers.get(ModelTier::Advanced), &shared);
            assert_eq!(tiers.get(ModelTier::Balanced), &shared);
            assert_eq!(tiers.get(ModelTier::Fast), &fast);
        }
    }
}
