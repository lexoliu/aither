//! The single context-management policy.
//!
//! Every context decision the agent makes — when to fork a compaction, when
//! to stop and wait for one, when an idle gap is long enough that rebuilding
//! the window is cheaper than keeping it, and how much of a handoff may be
//! spent on restored detail — is a threshold on this one struct. There is no
//! second knob, no per-level strategy and no preservation config: a single
//! policy keeps the behaviour predictable and the prompt cache analysable.
//!
//! Compaction points are token counts. A fraction of the window is the right
//! unit only for small windows: on a million-token model it would compact at
//! 600k, far past where a long context stays useful, so the auto-compact
//! point is the smaller of a fraction and an absolute ceiling, and the
//! distances around it are fixed token counts that do not stretch with the
//! window.

use std::time::Duration;

use num_traits::ToPrimitive as _;

/// The share of the auto-compact point a fork may start from.
const FORK_HEADROOM: f64 = 0.9;

/// Thresholds that govern context compaction and pausing.
///
/// The window is the calibrated one of the [`ContextBudget`]: it subtracts
/// what the response and the tool surface need from the model's raw context
/// length. From it the policy derives three points, each in tokens:
///
/// - the *auto-compact point*, `min(window * compact_at, compact_ceiling)`;
/// - the *start point*, [`start_distance`](Self::start_distance) before it,
///   where a background compaction is forked;
/// - the *pause point*, [`safe_buffer`](Self::safe_buffer) before it, where
///   the loop waits for the compaction instead of sending more.
///
/// The distances are fixed counts, so a window too small for them (a start
/// distance past the point itself) forks from the first request; a host with
/// such a window sets distances that fit it.
///
/// # Example
///
/// ```
/// use aither_agent::ContextPolicy;
///
/// let policy = ContextPolicy::default();
/// assert_eq!(policy.compact_point(1_000_000), 400_000);
/// assert_eq!(policy.compact_point(200_000), 160_000);
/// assert!(policy.start_point(200_000) < policy.pause_point(200_000));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextPolicy {
    /// Fraction of the window at which the conversation is compacted.
    ///
    /// Capped by [`compact_ceiling`](Self::compact_ceiling).
    pub compact_at: f32,

    /// Most tokens the conversation may hold before it is compacted,
    /// whatever the window: a long context stops being useful well before a
    /// large window is full.
    pub compact_ceiling: usize,

    /// Tokens before the auto-compact point at which a background compaction
    /// is forked.
    ///
    /// The fork happens at a request boundary and runs concurrently with the
    /// next model turns, so this distance buys the turns needed to finish the
    /// handoff without stalling the run. It summarises everything before the
    /// fork; the turn keeps running.
    pub start_distance: usize,

    /// Tokens before the auto-compact point at which the loop stops and
    /// waits for the compaction to land, or runs one when none is in flight.
    ///
    /// A fixed count rather than a fraction, so the margin is the same on
    /// every window: it must hold the compaction request's own instruction
    /// and the handoff it produces.
    pub safe_buffer: usize,

    /// Tokens before the auto-compact point at which the model is told once
    /// to save its work and call `compact` itself.
    ///
    /// Larger than [`start_distance`](Self::start_distance), so the model has
    /// the turns in between to reach a clean point before the harness forks a
    /// compaction of its own. Only an agent with the `compact` tool raises
    /// the alert.
    pub warn_distance: usize,

    /// Fraction of the window a restored-context block may occupy.
    ///
    /// After compaction the agent re-reads the files and documents the
    /// handoff names. This caps that restoration so a handoff cannot
    /// reintroduce the pressure it was supposed to relieve.
    pub restore_budget: f32,

    /// How long before the cache lifetime ends an idle session is compacted.
    ///
    /// The compaction then reads the conversation from the warm cache, and
    /// the session resumes on a much smaller prefix instead of paying to write
    /// the whole context again. Counted back from the end of the
    /// conversation's [`CacheLifetime`](super::CacheLifetime): six minutes
    /// compacts at 54 minutes under a one-hour lifetime.
    pub idle_compact_before_expiry: Duration,

    /// Tokens a tool output may hold before failure recovery moves it to a
    /// file, leaving the path and a short head in its place.
    pub spill_threshold: usize,

    /// Tokens of a spilled tool output that stay in place as its head.
    pub spill_head: usize,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            compact_at: 0.8,
            compact_ceiling: 400_000,
            start_distance: 50_000,
            safe_buffer: 5_000,
            warn_distance: 80_000,
            restore_budget: 0.15,
            idle_compact_before_expiry: Duration::from_secs(6 * 60),
            spill_threshold: 4_000,
            spill_head: 200,
        }
    }
}

impl ContextPolicy {
    /// The auto-compact point for a window of `window` tokens:
    /// `min(window * compact_at, compact_ceiling)`.
    #[must_use]
    pub fn compact_point(&self, window: u32) -> usize {
        let fraction = f64::from(window) * f64::from(self.compact_at.clamp(0.0, 1.0));
        to_tokens(fraction).min(self.compact_ceiling)
    }

    /// The token count at which a background compaction is forked.
    #[must_use]
    pub fn start_point(&self, window: u32) -> usize {
        self.compact_point(window)
            .saturating_sub(self.start_distance)
    }

    /// The token count at which the model is told to save its work and call
    /// `compact`.
    #[must_use]
    pub fn warn_point(&self, window: u32) -> usize {
        self.compact_point(window)
            .saturating_sub(self.warn_distance)
    }

    /// Whether the window holds enough to warn the model of the coming
    /// compaction. A window of unknown size never warns.
    #[must_use]
    pub fn should_warn(&self, tokens: usize, window: u32) -> bool {
        window > 0 && tokens >= self.warn_point(window)
    }

    /// The token count at which the loop waits for compaction.
    #[must_use]
    pub fn pause_point(&self, window: u32) -> usize {
        self.compact_point(window).saturating_sub(self.safe_buffer)
    }

    /// Whether the window holds enough to fork a background compaction.
    ///
    /// A budget with no window yet never reports it: thresholds must not fire
    /// on a window the agent has not learned the size of.
    #[must_use]
    pub fn should_compact(&self, tokens: usize, window: u32) -> bool {
        window > 0 && tokens >= self.start_point(window)
    }

    /// Whether the room left before the auto-compact point is down to the
    /// safe buffer, so the loop must wait for a compaction.
    #[must_use]
    pub fn should_pause(&self, tokens: usize, window: u32) -> bool {
        window > 0 && tokens >= self.pause_point(window)
    }

    /// Whether a fork may start from a context holding `tokens`: it is
    /// refused within a tenth of the auto-compact point, where the main agent
    /// and the fork would each go on to compact the same context.
    #[must_use]
    pub fn allows_fork(&self, tokens: usize, window: u32) -> bool {
        let limit = to_f64(self.compact_point(window)) * FORK_HEADROOM;
        to_f64(tokens) < limit
    }
}

/// Widens a token count for ratio arithmetic, saturating rather than wrapping.
fn to_f64(tokens: usize) -> f64 {
    tokens.to_f64().unwrap_or(f64::MAX)
}

/// Narrows a non-negative token estimate back to a count.
fn to_tokens(value: f64) -> usize {
    value.floor().to_usize().unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_auto_compact_point_is_the_lesser_of_the_fraction_and_the_ceiling() {
        let policy = ContextPolicy::default();
        assert_eq!(policy.compact_point(1_000_000), 400_000);
        assert_eq!(policy.compact_point(200_000), 160_000);
        assert_eq!(policy.compact_point(500_000), 400_000);
        assert_eq!(policy.compact_point(0), 0);
    }

    #[test]
    fn the_fraction_and_the_ceiling_are_configurable() {
        let policy = ContextPolicy {
            compact_at: 0.5,
            compact_ceiling: 100_000,
            ..ContextPolicy::default()
        };
        assert_eq!(policy.compact_point(1_000_000), 100_000);
        assert_eq!(policy.compact_point(100_000), 50_000);
    }

    #[test]
    fn default_points_are_ordered() {
        let policy = ContextPolicy::default();
        for window in [100_000, 200_000, 1_000_000] {
            assert!(policy.warn_point(window) < policy.start_point(window));
            assert!(policy.start_point(window) < policy.pause_point(window));
            assert!(policy.pause_point(window) < policy.compact_point(window));
        }
        assert!(policy.restore_budget < policy.compact_at);
    }

    #[test]
    fn the_start_and_pause_points_sit_a_fixed_distance_before_the_compact_point() {
        let policy = ContextPolicy::default();
        assert_eq!(policy.start_point(1_000_000), 350_000);
        assert_eq!(policy.pause_point(1_000_000), 395_000);
        assert_eq!(policy.start_point(200_000), 110_000);
        assert_eq!(policy.pause_point(200_000), 155_000);
    }

    #[test]
    fn thresholds_are_inclusive() {
        let policy = ContextPolicy::default();
        assert!(policy.should_compact(110_000, 200_000));
        assert!(!policy.should_compact(109_999, 200_000));
        assert!(policy.should_pause(155_000, 200_000));
        assert!(!policy.should_pause(154_999, 200_000));
    }

    #[test]
    fn a_window_of_unknown_size_never_crosses_a_threshold() {
        let policy = ContextPolicy::default();
        assert!(!policy.should_compact(1_000_000, 0));
        assert!(!policy.should_pause(1_000_000, 0));
    }

    #[test]
    fn a_fork_is_refused_within_a_tenth_of_the_compaction_point() {
        let policy = ContextPolicy::default();
        assert!(policy.allows_fork(143_000, 200_000));
        assert!(!policy.allows_fork(144_000, 200_000));
        assert!(!policy.allows_fork(155_000, 200_000));
    }
}
