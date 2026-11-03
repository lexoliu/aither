//! Network reachability and system sleep/wake notifications.
//!
//! [`Connectivity::start`] watches the operating system for two things an
//! agent that streams from a provider needs to know: whether the machine can
//! reach a network, and whether it is going to sleep or has just woken. A
//! stream opened before sleep sits on a dead socket after wake; this is the
//! signal to abandon it.
//!
//! | Platform | Reachability | Sleep and wake |
//! |---|---|---|
//! | macOS | [`nw_path_monitor`] (Network.framework) | [`IORegisterForSystemPower`] (`IOKit`) |
//! | Linux | rtnetlink link and route groups, see [`rtnetlink(7)`] | logind [`PrepareForSleep`] over D-Bus |
//! | Windows | [`NotifyNetworkConnectivityHintChange`] | [`PowerRegisterSuspendResumeNotification`] |
//!
//! Monitors run on their own threads or OS callbacks and publish through
//! plain channels, so nothing here depends on an async runtime.
//!
//! [`nw_path_monitor`]: https://developer.apple.com/documentation/network/nw_path_monitor_t
//! [`IORegisterForSystemPower`]: https://developer.apple.com/documentation/iokit/1557114-ioregisterforsystempower
//! [`rtnetlink(7)`]: https://man7.org/linux/man-pages/man7/rtnetlink.7.html
//! [`PrepareForSleep`]: https://www.freedesktop.org/software/systemd/man/latest/org.freedesktop.login1.html
//! [`NotifyNetworkConnectivityHintChange`]: https://learn.microsoft.com/en-us/windows/win32/api/netioapi/nf-netioapi-notifynetworkconnectivityhintchange
//! [`PowerRegisterSuspendResumeNotification`]: https://learn.microsoft.com/en-us/windows/win32/api/powrprof/nf-powrprof-powerregistersuspendresumenotification

use std::{future::Future, sync::Arc, time::Duration};

use futures_core::Stream;

mod hub;
mod platform;

use hub::{Shared, Sink};

/// How long [`Connectivity::start`] waits for the operating system to report
/// the first reachability.
const FIRST_REPORT_TIMEOUT: Duration = Duration::from_secs(5);

/// A change the operating system reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Change {
    /// What changed.
    pub kind: ChangeKind,
    /// When this process received the notification.
    pub at: jiff::Timestamp,
}

/// What changed on the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeKind {
    /// The system is about to sleep.
    Sleeping,
    /// The system woke from sleep. The network is often still down and every
    /// socket opened before sleep is dead.
    Woke,
    /// The machine gained a route to the network.
    Online,
    /// The machine lost its route to the network.
    Offline,
}

/// Why [`Connectivity::start`] failed.
#[derive(Debug, thiserror::Error)]
pub enum ConnectivityError {
    /// This operating system has no connectivity monitor.
    #[error("connectivity monitoring is not supported on this platform")]
    Unsupported,
    /// The operating system refused a monitor registration.
    #[error("cannot register the {monitor} monitor: {reason}")]
    Register {
        /// The monitor that failed.
        monitor: &'static str,
        /// The operating system's explanation.
        reason: String,
    },
    /// The operating system never reported the initial reachability.
    #[error("the operating system did not report network reachability within {0:?}")]
    NoInitialState(Duration),
}

/// A handle on the machine's reachability and power state.
///
/// Clones share one monitor; it stops when the last clone is dropped.
#[derive(Clone)]
pub struct Connectivity {
    shared: Arc<Shared>,
    // Held for its `Drop`, which stops the monitor.
    _monitor: Arc<Option<platform::Monitor>>,
}

impl std::fmt::Debug for Connectivity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connectivity")
            .field("online", &self.is_online())
            .finish_non_exhaustive()
    }
}

impl Connectivity {
    /// Starts watching the operating system.
    ///
    /// Returns once the first reachability is known, so [`is_online`]
    /// is accurate immediately. The first report is the initial state, not a
    /// [`Change`].
    ///
    /// # Errors
    ///
    /// Fails when the platform has no monitor, the operating system refuses a
    /// registration, or it reports no reachability within five seconds.
    ///
    /// [`is_online`]: Self::is_online
    pub fn start() -> Result<Self, ConnectivityError> {
        let shared = Shared::new();
        let monitor = platform::Monitor::start(Sink::new(shared.clone()))?;
        if !shared.block_until_known(FIRST_REPORT_TIMEOUT) {
            return Err(ConnectivityError::NoInitialState(FIRST_REPORT_TIMEOUT));
        }
        Ok(Self {
            shared,
            _monitor: Arc::new(Some(monitor)),
        })
    }

    /// A handle driven by the returned [`Controller`] instead of the
    /// operating system, for tests of hosts that react to connectivity.
    #[must_use]
    pub fn manual(online: bool) -> (Self, Controller) {
        let shared = Shared::new();
        let sink = Sink::new(shared.clone());
        sink.set_online(online);
        (
            Self {
                shared,
                _monitor: Arc::new(None),
            },
            Controller { sink },
        )
    }

    /// Whether the machine currently has a route to the network.
    #[must_use]
    pub fn is_online(&self) -> bool {
        self.shared.is_online()
    }

    /// Resolves once the machine is online; at once when it already is.
    pub fn online(&self) -> impl Future<Output = ()> + Send + use<> {
        let shared = self.shared.clone();
        async move { shared.until_online().await }
    }

    /// Every [`Change`] after this call.
    ///
    /// A subscriber that falls more than 64 changes behind loses the oldest.
    pub fn subscribe(&self) -> impl Stream<Item = Change> + Send + use<> {
        self.shared.subscribe()
    }
}

/// Injects changes into a [`Connectivity::manual`] handle.
#[derive(Clone)]
pub struct Controller {
    sink: Sink,
}

impl std::fmt::Debug for Controller {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Controller").finish_non_exhaustive()
    }
}

impl Controller {
    /// Reports that the system is going to sleep.
    pub fn sleep(&self) {
        self.sink.sleeping();
    }

    /// Reports that the system woke.
    pub fn wake(&self) {
        self.sink.woke();
    }

    /// Reports the machine's reachability; a repeat of the current state is
    /// not a change.
    pub fn set_online(&self, online: bool) {
        self.sink.set_online(online);
    }
}

#[cfg(test)]
mod tests {
    use futures_lite::{StreamExt, future};

    use super::*;

    #[test]
    fn manual_reports_the_initial_state_without_a_change() {
        let (connectivity, _controller) = Connectivity::manual(true);
        assert!(connectivity.is_online());
        let (connectivity, _controller) = Connectivity::manual(false);
        assert!(!connectivity.is_online());
    }

    #[tokio::test]
    async fn subscribers_see_every_change_after_subscribing() {
        let (connectivity, controller) = Connectivity::manual(true);
        controller.sleep();
        let mut changes = Box::pin(connectivity.subscribe());
        controller.set_online(true);
        controller.wake();
        controller.set_online(false);
        controller.set_online(true);
        for expected in [ChangeKind::Woke, ChangeKind::Offline, ChangeKind::Online] {
            assert_eq!(changes.next().await.unwrap().kind, expected);
        }
    }

    #[tokio::test]
    async fn online_is_ready_when_online() {
        let (connectivity, _controller) = Connectivity::manual(true);
        connectivity.online().await;
    }

    #[tokio::test]
    async fn online_waits_for_online() {
        let (connectivity, controller) = Connectivity::manual(false);
        let mut pending = Box::pin(connectivity.online());
        assert!(future::poll_once(&mut pending).await.is_none());
        controller.wake();
        assert!(future::poll_once(&mut pending).await.is_none());
        controller.set_online(true);
        pending.await;
        assert!(connectivity.is_online());
    }
}
