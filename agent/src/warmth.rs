//! Whether the provider's prompt cache is still warm, and for how long.
//!
//! Keep-alive, idle compaction and reassembly in the agent, and a host's
//! choice to retry on the same provider or to spend a cache-priced sidecar
//! call, all hinge on one question about the cache entry. [`CacheWarmth`] is
//! its single answer: the agent stamps it on every request and keep-alive,
//! and everyone else reads it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::clock::{AnyClock, Instant};
use crate::context::CacheLifetime;

/// Marks an entry that was never started.
const NOT_STARTED: u64 = u64::MAX;

/// The state of a prompt-cache entry at one moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// Nothing is cached: no request has been sent yet, or the entry's
    /// lifetime has passed since it was last started.
    Cold,
    /// The entry can still be read.
    Warm {
        /// How long until the entry goes cold; `None` when it never does
        /// ([`CacheLifetime::Permanent`]).
        cold_in: Option<Duration>,
    },
}

#[derive(Debug)]
struct Inner {
    lifetime: CacheLifetime,
    clock: AnyClock,
    /// The latest of the last request start and the last keep-alive send, in
    /// nanoseconds on `clock`, or [`NOT_STARTED`].
    entry_started_nanos: AtomicU64,
}

/// A shared view of one agent's prompt-cache entry.
///
/// Clones observe the same entry: a handle taken from
/// [`Agent::cache_warmth`](crate::Agent::cache_warmth) before a later request
/// sees that request refresh the entry.
#[derive(Debug, Clone)]
pub struct CacheWarmth {
    inner: Arc<Inner>,
}

impl CacheWarmth {
    /// A handle on an entry that was never started.
    pub(crate) fn new(lifetime: CacheLifetime, clock: AnyClock) -> Self {
        Self {
            inner: Arc::new(Inner {
                lifetime,
                clock,
                entry_started_nanos: AtomicU64::new(NOT_STARTED),
            }),
        }
    }

    /// Records that a request or keep-alive started the entry at `at`.
    pub(crate) fn start_entry(&self, at: Instant) {
        self.inner
            .entry_started_nanos
            .store(at.as_nanos(), Ordering::Relaxed);
    }

    /// A handle on a never-started entry with another lifetime, same clock.
    #[cfg(test)]
    pub(crate) fn with_lifetime(&self, lifetime: CacheLifetime) -> Self {
        Self::new(lifetime, self.inner.clock.clone())
    }

    /// Records that a request or keep-alive started the entry just now.
    pub(crate) fn start_entry_now(&self) {
        self.start_entry(self.inner.clock.now());
    }

    /// How long ago the entry was started; `None` before the first request.
    fn idle(&self) -> Option<Duration> {
        match self.inner.entry_started_nanos.load(Ordering::Relaxed) {
            NOT_STARTED => None,
            nanos => Some(self.inner.clock.since(Instant::from_nanos(nanos))),
        }
    }

    /// Whether the entry was started and has since expired, as opposed to
    /// never having been started.
    pub(crate) fn has_expired(&self) -> bool {
        match (self.inner.lifetime, self.idle()) {
            (CacheLifetime::Limited(lifetime), Some(idle)) => idle >= lifetime,
            _ => false,
        }
    }

    /// The entry's state now. [`CacheState::Cold`] before the first request.
    #[must_use]
    pub fn state(&self) -> CacheState {
        let Some(idle) = self.idle() else {
            return CacheState::Cold;
        };
        match self.inner.lifetime {
            CacheLifetime::Permanent => CacheState::Warm { cold_in: None },
            CacheLifetime::Limited(lifetime) => match lifetime.checked_sub(idle) {
                Some(remaining) if !remaining.is_zero() => CacheState::Warm {
                    cold_in: Some(remaining),
                },
                _ => CacheState::Cold,
            },
        }
    }

    /// Whether the entry can still be read.
    #[must_use]
    pub fn is_warm(&self) -> bool {
        matches!(self.state(), CacheState::Warm { .. })
    }

    /// How long until the entry goes cold; `Some` only when it is warm and
    /// has a limited lifetime.
    #[must_use]
    pub fn cold_in(&self) -> Option<Duration> {
        match self.state() {
            CacheState::Warm { cold_in } => cold_in,
            CacheState::Cold => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManualClock;

    const HOUR: Duration = Duration::from_secs(3600);

    fn warmth(lifetime: CacheLifetime) -> (CacheWarmth, ManualClock) {
        let clock = ManualClock::new();
        (
            CacheWarmth::new(lifetime, AnyClock::new(clock.clone())),
            clock,
        )
    }

    #[test]
    fn it_is_cold_before_the_first_request() {
        let (warmth, _clock) = warmth(CacheLifetime::Limited(HOUR));
        assert_eq!(warmth.state(), CacheState::Cold);
        assert!(!warmth.is_warm());
        assert_eq!(warmth.cold_in(), None);
        assert!(!warmth.has_expired());

        let (permanent, _clock) = self::warmth(CacheLifetime::Permanent);
        assert_eq!(permanent.state(), CacheState::Cold);
    }

    #[test]
    fn a_warm_entry_goes_cold_once_its_lifetime_passes() {
        let (warmth, clock) = warmth(CacheLifetime::Limited(HOUR));
        clock.advance(Duration::from_secs(10));
        warmth.start_entry_now();
        assert_eq!(warmth.cold_in(), Some(HOUR));

        clock.advance(Duration::from_secs(600));
        assert!(warmth.is_warm());
        assert_eq!(
            warmth.state(),
            CacheState::Warm {
                cold_in: Some(Duration::from_secs(3000))
            }
        );

        clock.advance(Duration::from_secs(3000) - Duration::from_nanos(1));
        assert_eq!(warmth.cold_in(), Some(Duration::from_nanos(1)));
        assert!(!warmth.has_expired());

        clock.advance(Duration::from_nanos(1));
        assert_eq!(warmth.state(), CacheState::Cold);
        assert_eq!(warmth.cold_in(), None);
        assert!(warmth.has_expired());
    }

    #[test]
    fn a_permanent_entry_never_goes_cold() {
        let (warmth, clock) = warmth(CacheLifetime::Permanent);
        warmth.start_entry_now();
        clock.advance(Duration::from_secs(100 * 86_400));
        assert_eq!(warmth.state(), CacheState::Warm { cold_in: None });
        assert!(warmth.is_warm());
        assert_eq!(warmth.cold_in(), None);
        assert!(!warmth.has_expired());
    }

    #[test]
    fn a_clone_sees_later_requests() {
        let (warmth, clock) = warmth(CacheLifetime::Limited(HOUR));
        let early = warmth.clone();
        warmth.start_entry_now();
        clock.advance(Duration::from_secs(3000));
        assert_eq!(early.cold_in(), Some(Duration::from_secs(600)));

        warmth.start_entry_now();
        assert_eq!(early.cold_in(), Some(HOUR));
    }
}
