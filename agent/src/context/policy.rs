//! The single context-management policy.
//!
//! Every context decision the agent makes — when to fork a compaction, when
//! to stop and wait for one, when an idle gap is long enough that rebuilding
//! the window is cheaper than keeping it, and how much of a handoff may be
//! spent on restored detail — is a threshold on this one struct. There is no
//! second knob, no per-level strategy and no preservation config: a single
//! policy keeps the behaviour predictable and the prompt cache analysable.

use std::time::Duration;

/// Thresholds that govern context compaction, pausing and reassembly.
///
/// Fractions are of the calibrated context window
/// ([`ContextBudget`](super::ContextBudget)), not of the raw model context
/// length: the budget subtracts what the response and the tool surface need.
///
/// # Example
///
/// ```
/// use aither_agent::ContextPolicy;
///
/// let policy = ContextPolicy::default();
/// assert!(policy.compact_at < policy.pause_at);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextPolicy {
    /// Window usage fraction at which a background compaction is forked.
    ///
    /// The fork happens at a request boundary and runs concurrently with the
    /// next model turns, so this threshold is deliberately far from full: it
    /// buys the turns needed to finish the handoff without stalling the run.
    pub compact_at: f32,

    /// Window usage fraction at which the run stops and waits for compaction.
    ///
    /// Reaching this point means compaction did not finish in time. The agent
    /// awaits the in-flight handoff rather than sending a request that the
    /// provider would reject for length.
    pub pause_at: f32,

    /// Idle gap after which the next request reassembles the window.
    ///
    /// Once the provider's prompt cache has expired, the cached prefix is
    /// worthless, so the cheapest moment to rewrite the window — folding in
    /// staged system blocks and dropping superseded tool output — is the
    /// first request after the gap.
    pub idle_reassemble_after: Duration,

    /// Fraction of the window a restored-context block may occupy.
    ///
    /// After compaction the agent re-reads the files and documents the
    /// handoff names. This caps that restoration so a handoff cannot
    /// reintroduce the pressure it was supposed to relieve.
    pub restore_budget: f32,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            compact_at: 0.6,
            pause_at: 0.9,
            idle_reassemble_after: Duration::from_secs(300),
            restore_budget: 0.15,
        }
    }
}

impl ContextPolicy {
    /// Whether `usage` has reached the compaction fork threshold.
    #[must_use]
    pub fn should_compact(&self, usage: f32) -> bool {
        usage >= self.compact_at
    }

    /// Whether `usage` has reached the pause threshold.
    #[must_use]
    pub fn should_pause(&self, usage: f32) -> bool {
        usage >= self.pause_at
    }

    /// Whether `idle` is long enough to justify a reassembly.
    #[must_use]
    pub fn should_reassemble(&self, idle: Duration) -> bool {
        idle >= self.idle_reassemble_after
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_thresholds_are_ordered() {
        let policy = ContextPolicy::default();
        assert!(policy.compact_at < policy.pause_at);
        assert!(policy.pause_at < 1.0);
        assert!(policy.restore_budget < policy.compact_at);
    }

    #[test]
    fn thresholds_are_inclusive() {
        let policy = ContextPolicy::default();
        assert!(policy.should_compact(0.6));
        assert!(!policy.should_compact(0.59));
        assert!(policy.should_pause(0.9));
        assert!(!policy.should_pause(0.89));
        assert!(policy.should_reassemble(Duration::from_secs(300)));
        assert!(!policy.should_reassemble(Duration::from_secs(299)));
    }
}
