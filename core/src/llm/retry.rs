//! How often, and after how long, a failed model request is sent again.

use core::time::Duration;

use super::Failure;

/// How a model request that failed with a retryable [`Failure`] is retried.
///
/// The `retry`-th retry (0-based) waits `initial_delay * multiplier^retry`,
/// capped at `max_delay`; a delay the upstream requested
/// ([`Failure::retry_after`]) replaces the backoff, under the same cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetryPolicy {
    /// How many times a failed request is sent again (0 = never).
    pub max_retries: u32,
    /// The delay before the first retry.
    pub initial_delay: Duration,
    /// The longest delay before any retry.
    pub max_delay: Duration,
    /// The factor each following retry's delay grows by.
    pub multiplier: u32,
}

impl Default for RetryPolicy {
    /// Three retries, starting at 500 ms and doubling up to 30 s.
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            multiplier: 2,
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries.
    #[must_use]
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Whether a request that failed with `failure` after `retries` retries
    /// is sent again.
    #[must_use]
    pub const fn allows(&self, retries: u32, failure: &Failure) -> bool {
        retries < self.max_retries && failure.is_retryable()
    }

    /// The delay before the `retry`-th retry (0-based) of a request that
    /// failed with `failure`.
    #[must_use]
    pub fn delay(&self, retry: u32, failure: &Failure) -> Duration {
        failure
            .retry_after
            .unwrap_or_else(|| {
                self.initial_delay
                    .saturating_mul(self.multiplier.saturating_pow(retry))
            })
            .min(self.max_delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_up_to_the_cap() {
        let policy = RetryPolicy::default();
        let failure = Failure::network();
        assert_eq!(policy.delay(0, &failure), Duration::from_millis(500));
        assert_eq!(policy.delay(2, &failure), Duration::from_secs(2));
        assert_eq!(policy.delay(u32::MAX, &failure), policy.max_delay);
    }

    #[test]
    fn a_requested_delay_replaces_the_backoff_under_the_cap() {
        let policy = RetryPolicy::default();
        let soon = Failure::transient().with_retry_after(Some(Duration::from_secs(4)));
        assert_eq!(policy.delay(0, &soon), Duration::from_secs(4));
        let late = Failure::transient().with_retry_after(Some(Duration::from_secs(600)));
        assert_eq!(policy.delay(0, &late), policy.max_delay);
    }

    #[test]
    fn only_retryable_failures_within_the_limit_are_allowed() {
        let policy = RetryPolicy::default();
        assert!(policy.allows(0, &Failure::network()));
        assert!(policy.allows(2, &Failure::transient()));
        assert!(!policy.allows(3, &Failure::network()));
        assert!(!policy.allows(0, &Failure::permanent()));
        assert!(!policy.allows(0, &Failure::quota()));
        assert!(!RetryPolicy::none().allows(0, &Failure::network()));
    }
}
