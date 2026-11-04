//! The foreground window of a terminal, adjustable while commands run.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// How long a `terminal` call waits in the foreground before a still-running
/// command promotes to a background task.
///
/// The window is a normal one plus an optional longer one that applies while
/// it is held. Every clone shares both, so a terminal, the terminals derived
/// from it and a host that lengthens the window while the machine is loaded
/// all see one setting; a change applies to the next foreground wait.
#[derive(Debug, Clone)]
pub struct ForegroundWait(Arc<Shared>);

#[derive(Debug)]
struct Shared {
    normal_nanos: AtomicU64,
    /// The longer window in force, or `0` when the normal one applies.
    extended_nanos: AtomicU64,
}

fn nanos(window: Duration) -> u64 {
    u64::try_from(window.as_nanos()).expect("a foreground window fits in u64 nanoseconds")
}

impl ForegroundWait {
    /// A window of `normal` with no longer window in force.
    #[must_use]
    pub fn new(normal: Duration) -> Self {
        Self(Arc::new(Shared {
            normal_nanos: AtomicU64::new(nanos(normal)),
            extended_nanos: AtomicU64::new(0),
        }))
    }

    /// Changes the normal window for every handle.
    pub fn set_normal(&self, normal: Duration) {
        self.0.normal_nanos.store(nanos(normal), Ordering::Relaxed);
    }

    /// The normal window, whether or not a longer one is in force.
    #[must_use]
    pub fn normal(&self) -> Duration {
        Duration::from_nanos(self.0.normal_nanos.load(Ordering::Relaxed))
    }

    /// Puts a longer window in force until [`Self::restore`]. A zero
    /// `window` leaves the normal one in force.
    pub fn extend(&self, window: Duration) {
        self.0
            .extended_nanos
            .store(nanos(window), Ordering::Relaxed);
    }

    /// Ends the longer window; the normal one applies again.
    pub fn restore(&self) {
        self.0.extended_nanos.store(0, Ordering::Relaxed);
    }

    /// The window a foreground wait that starts now uses.
    #[must_use]
    pub fn get(&self) -> Duration {
        match self.0.extended_nanos.load(Ordering::Relaxed) {
            0 => self.normal(),
            extended => Duration::from_nanos(extended),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_longer_window_applies_until_restored() {
        let wait = ForegroundWait::new(Duration::from_secs(10));
        let shared = wait.clone();
        assert_eq!(wait.get(), Duration::from_secs(10));
        shared.extend(Duration::from_secs(60));
        assert_eq!(wait.get(), Duration::from_secs(60));
        assert_eq!(wait.normal(), Duration::from_secs(10));
        shared.restore();
        assert_eq!(wait.get(), Duration::from_secs(10));
    }
}
