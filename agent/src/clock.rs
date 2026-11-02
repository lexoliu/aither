//! The time source behind the agent's idle timing.
//!
//! Idle compaction, idle reassembly and prompt-cache keep-alive all measure
//! how long ago the provider's cache entry was started. They read the time
//! through a [`Clock`] passed to the builder, so a host can drive them from a
//! clock it advances itself.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A source of monotonic time.
pub trait Clock: Send + Sync + 'static {
    /// The current instant.
    fn now(&self) -> Instant;
}

/// The operating system's monotonic clock; the default.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// A clock that only moves when told to.
///
/// Clones share one timeline, so a test keeps a handle, gives the builder
/// another, and advances time without waiting.
#[derive(Debug, Clone)]
pub struct ManualClock {
    inner: Arc<ManualInner>,
}

#[derive(Debug)]
struct ManualInner {
    base: Instant,
    advanced_nanos: AtomicU64,
}

impl ManualClock {
    /// Creates a clock standing at the current system instant.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ManualInner {
                base: Instant::now(),
                advanced_nanos: AtomicU64::new(0),
            }),
        }
    }

    /// Moves the clock forward by `by`.
    ///
    /// # Panics
    ///
    /// Panics when `by` exceeds `u64::MAX` nanoseconds (about 584 years).
    pub fn advance(&self, by: Duration) {
        let nanos = u64::try_from(by.as_nanos()).expect("advance fits in u64 nanoseconds");
        self.inner
            .advanced_nanos
            .fetch_add(nanos, Ordering::Relaxed);
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        self.inner.base + Duration::from_nanos(self.inner.advanced_nanos.load(Ordering::Relaxed))
    }
}

/// A type-erased [`Clock`], what the agent stores.
///
/// The agent already carries five type parameters; the clock is read a few
/// times per turn, so erasing it costs nothing that matters.
#[derive(Clone)]
pub struct AnyClock(Arc<dyn Clock>);

impl AnyClock {
    /// Erases `clock`.
    #[must_use]
    pub fn new(clock: impl Clock) -> Self {
        Self(Arc::new(clock))
    }

    /// The current instant on the wrapped clock.
    #[must_use]
    pub fn now(&self) -> Instant {
        self.0.now()
    }

    /// Time elapsed since `earlier`, zero when `earlier` is in the future.
    #[must_use]
    pub fn since(&self, earlier: Instant) -> Duration {
        self.now().saturating_duration_since(earlier)
    }
}

impl Default for AnyClock {
    fn default() -> Self {
        Self::new(SystemClock)
    }
}

impl fmt::Debug for AnyClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyClock")
    }
}
