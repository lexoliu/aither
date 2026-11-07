//! How a change to an agent's reasoning effort reaches the provider.
//!
//! A request-level effort change restarts the provider's prompt cache. A model
//! with [`Ability::PerMessageEffort`](aither_core::llm::model::Ability::PerMessageEffort)
//! takes the change inside the conversation instead, as a
//! [`Message::Configuration`](aither_core::llm::Message::Configuration): the
//! request-level effort stays the one the first request sent, every earlier
//! byte stays as it was, and the update stays in the window like any other
//! message, so every later request carries it. A compaction keeps the last
//! update it would drop (see [`Context::apply_handoff`]), so the effort in
//! force survives it.
//!
//! Any other model gets the new effort at the request level, which restarts
//! its cache; the window's updates are left out of its requests, since it
//! cannot read them.
//!
//! [`Context::apply_handoff`]: crate::context::Context::apply_handoff

use aither_core::llm::model::ReasoningEffort;

/// The request-level effort an agent sends: fixed by its first request while
/// its model takes per-message updates, and the effort it wants otherwise.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EffortAnchor {
    /// No request has been planned yet.
    #[default]
    Unsent,
    /// The request-level effort the requests carry, `None` for none at all.
    Sent(Option<ReasoningEffort>),
}

impl EffortAnchor {
    /// Plans the next request for the effort `wanted`, given whether the model
    /// takes per-message updates and the effort the window's last update set,
    /// and returns the per-message update to append before the request.
    ///
    /// On a model that takes them, a wanted effort different from the one in
    /// force becomes an update and the request-level effort is left alone.
    /// Wanting no effort at all cannot be said per message, so it, like every
    /// change on a model without per-message updates, moves the request-level
    /// effort; updates already in the window stay there, which keeps every
    /// later byte valid, and remain in force on a model that reads them.
    pub fn plan(
        &mut self,
        wanted: Option<ReasoningEffort>,
        per_message: bool,
        last_update: Option<ReasoningEffort>,
    ) -> Option<ReasoningEffort> {
        if !per_message || wanted.is_none() {
            *self = Self::Sent(wanted);
            return None;
        }
        let request = match *self {
            Self::Unsent => {
                *self = Self::Sent(wanted);
                wanted
            }
            Self::Sent(request) => request,
        };
        let in_force = last_update.or(request);
        wanted.filter(|wanted| in_force != Some(*wanted))
    }

    /// The request-level effort for a request whose parameters want
    /// `wanted`: the one the last plan settled on, or `wanted` before any.
    pub const fn request_effort(self, wanted: Option<ReasoningEffort>) -> Option<ReasoningEffort> {
        match self {
            Self::Unsent => wanted,
            Self::Sent(request) => request,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ReasoningEffort::{High, Low, Medium};

    #[test]
    fn a_per_message_model_keeps_the_first_request_level_effort() {
        let mut anchor = EffortAnchor::default();
        assert_eq!(anchor.plan(Some(High), true, None), None);
        assert_eq!(anchor.plan(Some(Low), true, None), Some(Low));
        assert_eq!(anchor.request_effort(Some(Low)), Some(High));
        // Once the update is in the window, nothing more is needed.
        assert_eq!(anchor.plan(Some(Low), true, Some(Low)), None);
        // Back to the request-level effort is still a change from the update.
        assert_eq!(anchor.plan(Some(High), true, Some(Low)), Some(High));
        assert_eq!(anchor.request_effort(Some(High)), Some(High));
    }

    #[test]
    fn any_other_model_gets_the_new_effort_at_the_request_level() {
        let mut anchor = EffortAnchor::default();
        anchor.plan(Some(High), false, None);
        assert_eq!(anchor.plan(Some(Low), false, Some(Medium)), None);
        assert_eq!(anchor.request_effort(Some(Low)), Some(Low));
    }

    #[test]
    fn clearing_the_effort_moves_the_request_level_effort() {
        let mut anchor = EffortAnchor::default();
        anchor.plan(Some(High), true, None);
        assert_eq!(anchor.plan(None, true, None), None);
        assert_eq!(anchor.request_effort(None), None);
    }

    #[test]
    fn before_any_plan_the_wanted_effort_goes_out() {
        assert_eq!(EffortAnchor::default().request_effort(Some(Low)), Some(Low));
    }
}
