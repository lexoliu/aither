//! Provider-opaque reasoning state.

use alloc::string::String;

/// Reasoning state that a provider requires to be handed back unmodified.
///
/// Current models do not treat their own reasoning as throwaway. Anthropic
/// requires thinking blocks to be replayed "complete and unmodified" across a
/// tool-use turn, `OpenAI`'s stateless flow needs the encrypted reasoning item
/// echoed back into the next request, and Gemini attaches a thought signature
/// to each function call. Dropping any of them degrades multi-turn tool use
/// quietly: the model loses the reasoning that produced the very call it is
/// being handed a result for.
///
/// The payload is provider-encoded and deliberately opaque. Core never parses
/// it, and neither should anything else — the encoding belongs to the provider,
/// the meaning belongs to the provider, and the only correct operation is to
/// give it back. Encoding the whole block rather than its parts is what lets
/// one type carry Anthropic's `thinking` *and* `redacted_thinking` blocks,
/// whose shapes differ, without core knowing either.
///
/// # Provider tagging
///
/// State is tied to the model that produced it. Anthropic documents that
/// thinking blocks must be stripped when switching models, and cross-provider
/// replay is at best ignored and at worst rejected. Every value therefore
/// records its origin, and providers read it back with
/// [`payload_for`](Self::payload_for) rather than trusting whatever the
/// conversation happens to carry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReasoningState {
    provider: String,
    payload: String,
}

impl ReasoningState {
    /// Creates reasoning state tagged with the provider that produced it.
    ///
    /// `provider` should be the stable short name the provider crate uses for
    /// itself (`"claude"`, `"openai"`, `"gemini"`), and `payload` its own
    /// encoding of the block, which it alone will decode.
    #[must_use]
    pub fn new(provider: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            payload: payload.into(),
        }
    }

    /// The provider that produced this state.
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The opaque payload, without checking its origin.
    ///
    /// Prefer [`payload_for`](Self::payload_for) when about to put the value on
    /// the wire; this accessor is for logging and inspection.
    #[must_use]
    pub fn payload(&self) -> &str {
        &self.payload
    }

    /// The payload, but only if this state came from `provider`.
    ///
    /// Returns `None` for foreign state so callers drop it instead of replaying
    /// it into an API that cannot verify it.
    #[must_use]
    pub fn payload_for(&self, provider: &str) -> Option<&str> {
        (self.provider == provider).then_some(self.payload.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn payload_is_returned_only_to_its_own_provider() {
        let state = ReasoningState::new("claude", "opaque-signature");
        assert_eq!(state.payload_for("claude"), Some("opaque-signature"));
        assert_eq!(state.payload_for("openai"), None);
        assert_eq!(state.payload_for("gemini"), None);
    }

    /// The documented behaviour when switching models mid-conversation is to
    /// strip foreign reasoning, so filtering must leave a usable remainder
    /// rather than failing the whole turn.
    #[test]
    fn filtering_a_mixed_conversation_keeps_only_native_state() {
        let states = [
            ReasoningState::new("claude", "a"),
            ReasoningState::new("openai", "b"),
            ReasoningState::new("claude", "c"),
        ];
        let kept: Vec<&str> = states
            .iter()
            .filter_map(|state| state.payload_for("claude"))
            .collect();
        assert_eq!(kept, ["a", "c"]);
    }
}
