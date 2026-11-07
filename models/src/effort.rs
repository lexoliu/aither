//! The reasoning effort each model runs at when a request sets none.
//!
//! `LiteLLM` records no default effort, so the vendors' documented defaults are
//! curated here, for the models whose effort can change mid-conversation:
//! clearing the effort there is an update to this level.

use aither_core::llm::model::ReasoningEffort;

/// The documented default effort, by model id.
///
/// Anthropic's effort documentation gives `medium` for Claude Opus 5.5 and
/// Claude Haiku 5.5 and `high` for every other current model; `OpenAI`'s
/// reasoning guide gives `medium` for GPT-6 Sol, GPT-6 Luna and GPT-6.1 Sol,
/// and states none for GPT-6 Astra.
const DOCUMENTED_DEFAULTS: &[(&str, ReasoningEffort)] = &[
    ("claude-fable-5-1", ReasoningEffort::High),
    ("claude-mythos-5-1", ReasoningEffort::High),
    ("claude-opus-5-5", ReasoningEffort::Medium),
    ("claude-opus-5", ReasoningEffort::High),
    ("claude-sonnet-5-5", ReasoningEffort::High),
    ("claude-haiku-5-5", ReasoningEffort::Medium),
    ("gpt-6-sol", ReasoningEffort::Medium),
    ("gpt-6-luna", ReasoningEffort::Medium),
    ("gpt-6.1-sol", ReasoningEffort::Medium),
];

/// The reasoning effort `model_id` runs at when a request sets none, as its
/// vendor documents it; `None` when no default is recorded.
///
/// The id is resolved like [`lookup`](crate::lookup) resolves it, so a dated
/// alias or a version pin after `@` finds its model.
#[must_use]
pub fn default_reasoning_effort(model_id: &str) -> Option<ReasoningEffort> {
    let id = crate::lookup(model_id).map_or(model_id, crate::ModelEntry::id);
    let id = id.split_once('@').map_or(id, |(model, _)| model);
    DOCUMENTED_DEFAULTS
        .iter()
        .find_map(|(model, effort)| (*model == id).then_some(*effort))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documented_defaults_resolve_and_undocumented_ones_are_absent() {
        for (id, expected) in [
            ("claude-opus-5-5", Some(ReasoningEffort::Medium)),
            ("claude-sonnet-5-5", Some(ReasoningEffort::High)),
            ("claude-opus-5-5@default", Some(ReasoningEffort::Medium)),
            ("gpt-6-sol", Some(ReasoningEffort::Medium)),
            ("gpt-6-astra", None),
            ("gpt-5", None),
        ] {
            assert_eq!(default_reasoning_effort(id), expected, "{id}");
        }
    }
}
