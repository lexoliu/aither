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
//! force survives it. Clearing the effort there is an update to the level the
//! model runs at when a request sets none, its documented default.
//!
//! Any other model gets the new effort at the request level, which restarts
//! its cache; the window's updates are left out of its requests, since it
//! cannot read them.
//!
//! [`Context::apply_handoff`]: crate::context::Context::apply_handoff

use aither_core::llm::model::ReasoningEffort;

/// How a model takes an effort change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortChanges {
    /// Only at the request level.
    RequestLevel,
    /// Inside the conversation; `default` is the level the model runs at
    /// when a request sets none, when it is recorded.
    PerMessage {
        /// The model's documented default effort.
        default: Option<ReasoningEffort>,
    },
}

/// Clearing the effort on a model whose default effort is not recorded: the
/// update that would undo the one in force has no level to name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownDefaultEffort;

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
    /// Plans the next request for the effort `wanted` on a model that takes
    /// changes as `changes` says, given the effort the window's last update
    /// set, and returns the per-message update to append before the request.
    ///
    /// On a model that takes them, the request-level effort never moves: an
    /// effort different from the one in force becomes an update, and wanting
    /// none becomes an update to the model's default.
    ///
    /// # Errors
    ///
    /// Returns [`UnknownDefaultEffort`] when the effort is cleared on a model
    /// whose default is not recorded while another level is in force.
    pub fn plan(
        &mut self,
        wanted: Option<ReasoningEffort>,
        changes: EffortChanges,
        last_update: Option<ReasoningEffort>,
    ) -> Result<Option<ReasoningEffort>, UnknownDefaultEffort> {
        let EffortChanges::PerMessage { default } = changes else {
            *self = Self::Sent(wanted);
            return Ok(None);
        };
        let request = match *self {
            Self::Unsent => {
                *self = Self::Sent(wanted);
                wanted
            }
            Self::Sent(request) => request,
        };
        let in_force = last_update.or(request).or(default);
        let wanted = wanted.or(default);
        if wanted == in_force {
            return Ok(None);
        }
        wanted.map(Some).ok_or(UnknownDefaultEffort)
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

    const PER_MESSAGE: EffortChanges = EffortChanges::PerMessage {
        default: Some(Medium),
    };

    #[test]
    fn a_per_message_model_keeps_the_first_request_level_effort() {
        let mut anchor = EffortAnchor::default();
        assert_eq!(anchor.plan(Some(High), PER_MESSAGE, None), Ok(None));
        assert_eq!(anchor.plan(Some(Low), PER_MESSAGE, None), Ok(Some(Low)));
        assert_eq!(anchor.request_effort(Some(Low)), Some(High));
        // Once the update is in the window, nothing more is needed.
        assert_eq!(anchor.plan(Some(Low), PER_MESSAGE, Some(Low)), Ok(None));
        // Back to the request-level effort is still a change from the update.
        assert_eq!(
            anchor.plan(Some(High), PER_MESSAGE, Some(Low)),
            Ok(Some(High))
        );
        assert_eq!(anchor.request_effort(Some(High)), Some(High));
    }

    #[test]
    fn clearing_on_a_per_message_model_updates_to_its_default() {
        let mut anchor = EffortAnchor::default();
        anchor.plan(Some(High), PER_MESSAGE, None).expect("planned");
        assert_eq!(anchor.plan(None, PER_MESSAGE, Some(Low)), Ok(Some(Medium)));
        assert_eq!(
            anchor.request_effort(None),
            Some(High),
            "the request-level effort stays as first sent"
        );
        // The default in force needs nothing more.
        assert_eq!(anchor.plan(None, PER_MESSAGE, Some(Medium)), Ok(None));
    }

    #[test]
    fn clearing_without_a_known_default_fails() {
        let unknown = EffortChanges::PerMessage { default: None };
        let mut anchor = EffortAnchor::default();
        anchor.plan(Some(High), unknown, None).expect("planned");
        assert_eq!(
            anchor.plan(None, unknown, Some(Low)),
            Err(UnknownDefaultEffort)
        );
        // With nothing in force but the unset request level, clearing is no
        // change at all.
        let mut unset = EffortAnchor::default();
        assert_eq!(unset.plan(None, unknown, None), Ok(None));
        assert_eq!(unset.plan(None, unknown, None), Ok(None));
    }

    #[test]
    fn any_other_model_gets_the_new_effort_at_the_request_level() {
        let mut anchor = EffortAnchor::default();
        anchor
            .plan(Some(High), EffortChanges::RequestLevel, None)
            .expect("planned");
        assert_eq!(
            anchor.plan(None, EffortChanges::RequestLevel, Some(Medium)),
            Ok(None)
        );
        assert_eq!(anchor.request_effort(None), None);
    }

    #[test]
    fn before_any_plan_the_wanted_effort_goes_out() {
        assert_eq!(EffortAnchor::default().request_effort(Some(Low)), Some(Low));
    }
}
