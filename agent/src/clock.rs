//! The time source behind the prompt cache's warmth.
//!
//! [`CacheWarmth`](crate::CacheWarmth) measures how long ago the provider's
//! cache entry was started. It reads the time through a [`Clock`] passed to
//! the builder, so a host can drive it from a clock it advances itself.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use jiff::{SignedDuration, Timestamp};

/// A point on a [`Clock`]'s timeline.
///
/// Opaque and only meaningful against other instants of the same clock. Unlike
/// [`std::time::Instant`], which does not advance while the machine sleeps on
/// the platforms the agent supports, the timeline of the [`SystemClock`] keeps
/// counting through sleep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant {
    nanos: u64,
}

impl Instant {
    /// Builds an instant `nanos` nanoseconds after the clock's origin.
    pub(crate) const fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Nanoseconds since the clock's origin.
    pub(crate) const fn as_nanos(self) -> u64 {
        self.nanos
    }

    /// Time from `earlier` to `self`, or `None` when `earlier` is later.
    #[must_use]
    pub const fn checked_duration_since(self, earlier: Self) -> Option<Duration> {
        match self.nanos.checked_sub(earlier.nanos) {
            Some(nanos) => Some(Duration::from_nanos(nanos)),
            None => None,
        }
    }

    /// Time from `earlier` to `self`, zero when `earlier` is later.
    #[must_use]
    pub const fn duration_since(self, earlier: Self) -> Duration {
        Duration::from_nanos(self.nanos.saturating_sub(earlier.nanos))
    }

    /// `self` moved forward by `by`, or `None` on overflow.
    #[must_use]
    pub fn checked_add(self, by: Duration) -> Option<Self> {
        let nanos = u64::try_from(by.as_nanos()).ok()?;
        self.nanos.checked_add(nanos).map(Self::from_nanos)
    }

    /// `self` moved back by `by`, or `None` when that precedes the origin.
    #[must_use]
    pub fn checked_sub(self, by: Duration) -> Option<Self> {
        let nanos = u64::try_from(by.as_nanos()).ok()?;
        self.nanos.checked_sub(nanos).map(Self::from_nanos)
    }
}

/// A source of monotonic time.
pub trait Clock: Send + Sync + 'static {
    /// The current instant.
    fn now(&self) -> Instant;

    /// The current wall-clock time, stamped on the user messages the model
    /// reads.
    fn utc_now(&self) -> Timestamp;
}

/// The operating system's monotonic clock that keeps counting while the
/// machine sleeps; the default.
///
/// Reads `CLOCK_MONOTONIC` on Apple targets (which counts sleep, unlike
/// `CLOCK_UPTIME_RAW`, the source of [`std::time::Instant`] there),
/// `CLOCK_BOOTTIME` on Linux and Android, other Unix targets' `CLOCK_MONOTONIC`,
/// and the biased interrupt time (`QueryInterruptTime`) on Windows.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::from_nanos(system_nanos())
    }

    fn utc_now(&self) -> Timestamp {
        Timestamp::now()
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
const SYSTEM_CLOCK: rustix::time::ClockId = rustix::time::ClockId::Boottime;

#[cfg(all(unix, not(any(target_os = "linux", target_os = "android"))))]
const SYSTEM_CLOCK: rustix::time::ClockId = rustix::time::ClockId::Monotonic;

#[cfg(unix)]
fn system_nanos() -> u64 {
    let now = rustix::time::clock_gettime(SYSTEM_CLOCK);
    let secs = u64::try_from(now.tv_sec).expect("the system clock is not negative");
    let subsec = u64::try_from(now.tv_nsec).expect("tv_nsec is not negative");
    secs.checked_mul(1_000_000_000)
        .and_then(|nanos| nanos.checked_add(subsec))
        .expect("the system clock fits in u64 nanoseconds")
}

#[cfg(windows)]
fn system_nanos() -> u64 {
    let mut ticks = 0_u64;
    // SAFETY: `ticks` is a valid, writable `u64` for the whole call.
    unsafe { windows_sys::Win32::System::WindowsProgramming::QueryInterruptTime(&raw mut ticks) };
    // The count is in units of 100 nanoseconds.
    ticks
        .checked_mul(100)
        .expect("the interrupt time fits in u64 nanoseconds")
}

/// A clock that only moves when told to.
///
/// Clones share one timeline starting at its origin, so a test keeps a handle,
/// gives the builder another, and advances time without waiting.
#[derive(Debug, Clone)]
pub struct ManualClock {
    advanced_nanos: Arc<AtomicU64>,
    origin_utc: Timestamp,
}

impl ManualClock {
    /// Creates a clock standing at its origin.
    #[must_use]
    pub fn new() -> Self {
        Self::at_utc(Timestamp::UNIX_EPOCH)
    }

    /// Creates a clock standing at its origin, whose wall-clock time there is
    /// `origin_utc`.
    #[must_use]
    pub fn at_utc(origin_utc: Timestamp) -> Self {
        Self {
            advanced_nanos: Arc::new(AtomicU64::new(0)),
            origin_utc,
        }
    }

    /// Moves the clock forward by `by`.
    ///
    /// # Panics
    ///
    /// Panics when `by` exceeds `u64::MAX` nanoseconds (about 584 years).
    pub fn advance(&self, by: Duration) {
        let nanos = u64::try_from(by.as_nanos()).expect("advance fits in u64 nanoseconds");
        self.advanced_nanos.fetch_add(nanos, Ordering::Relaxed);
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        Instant::from_nanos(self.advanced_nanos.load(Ordering::Relaxed))
    }

    fn utc_now(&self) -> Timestamp {
        let advanced = i64::try_from(self.advanced_nanos.load(Ordering::Relaxed))
            .expect("a manual clock stays within i64 nanoseconds");
        self.origin_utc
            .checked_add(SignedDuration::from_nanos(advanced))
            .expect("a manual clock stays within the timestamp range")
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

    /// The current wall-clock time on the wrapped clock.
    #[must_use]
    pub fn utc_now(&self) -> Timestamp {
        self.0.utc_now()
    }

    /// Time elapsed since `earlier`, zero when `earlier` is in the future.
    #[must_use]
    pub fn since(&self, earlier: Instant) -> Duration {
        self.now().duration_since(earlier)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_clock_advances() {
        let clock = SystemClock;
        let first = clock.now();
        std::thread::sleep(Duration::from_millis(20));
        assert!(clock.now().duration_since(first) >= Duration::from_millis(20));
    }

    #[test]
    fn instants_do_arithmetic() {
        let at = Instant::from_nanos(10_000);
        let later = at.checked_add(Duration::from_micros(5)).unwrap();
        assert_eq!(later.duration_since(at), Duration::from_micros(5));
        assert_eq!(at.duration_since(later), Duration::ZERO);
        assert_eq!(at.checked_duration_since(later), None);
        assert_eq!(later.checked_sub(Duration::from_micros(5)), Some(at));
        assert_eq!(at.checked_sub(Duration::from_secs(1)), None);
    }
}
