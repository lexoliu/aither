//! State shared between the platform monitor and every [`Connectivity`] handle.
//!
//! [`Connectivity`]: crate::Connectivity

use std::{
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

use async_broadcast::{InactiveReceiver, Receiver, Sender, TrySendError};
use event_listener::{Event, Listener};

use crate::{Change, ChangeKind};

const UNKNOWN: u8 = 0;
const OFFLINE: u8 = 1;
const ONLINE: u8 = 2;

/// Changes a slow subscriber can fall behind by before the oldest are dropped.
const SUBSCRIBER_BACKLOG: usize = 64;

pub struct Shared {
    state: AtomicU8,
    changed: Event,
    changes: Sender<Change>,
    /// Keeps the channel open while no one is subscribed; an inactive
    /// receiver queues nothing.
    _open: InactiveReceiver<Change>,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        let (mut changes, receiver) = async_broadcast::broadcast(SUBSCRIBER_BACKLOG);
        // A monitor thread must never block on a subscriber: when one lags,
        // the oldest change is dropped for it.
        changes.set_overflow(true);
        Arc::new(Self {
            state: AtomicU8::new(UNKNOWN),
            changed: Event::new(),
            changes,
            _open: receiver.deactivate(),
        })
    }

    pub fn is_online(&self) -> bool {
        self.state.load(Ordering::SeqCst) == ONLINE
    }

    pub fn subscribe(&self) -> Receiver<Change> {
        self.changes.new_receiver()
    }

    /// Blocks the calling thread until a platform monitor has reported the
    /// first reachability, or `timeout` passes. Returns whether it is known.
    pub fn block_until_known(&self, timeout: Duration) -> bool {
        let known = || self.state.load(Ordering::SeqCst) != UNKNOWN;
        if known() {
            return true;
        }
        let listener = self.changed.listen();
        if known() {
            return true;
        }
        listener.wait_timeout(timeout);
        known()
    }

    /// Resolves once the machine is online.
    pub async fn until_online(&self) {
        loop {
            if self.is_online() {
                return;
            }
            let listener = self.changed.listen();
            if self.is_online() {
                return;
            }
            listener.await;
        }
    }
}

/// The publishing end handed to a platform monitor.
#[derive(Clone)]
pub struct Sink(Arc<Shared>);

impl Sink {
    pub const fn new(shared: Arc<Shared>) -> Self {
        Self(shared)
    }

    /// Records the current reachability. The first report is the initial
    /// state and is not a change; a repeat of the current state is ignored.
    pub fn set_online(&self, online: bool) {
        let new = if online { ONLINE } else { OFFLINE };
        let old = self.0.state.swap(new, Ordering::SeqCst);
        self.0.changed.notify(usize::MAX);
        if old != UNKNOWN && old != new {
            self.publish(if online {
                ChangeKind::Online
            } else {
                ChangeKind::Offline
            });
        }
    }

    pub fn sleeping(&self) {
        self.publish(ChangeKind::Sleeping);
    }

    pub fn woke(&self) {
        self.publish(ChangeKind::Woke);
    }

    fn publish(&self, kind: ChangeKind) {
        let change = Change {
            kind,
            at: jiff::Timestamp::now(),
        };
        match self.0.changes.try_broadcast(change) {
            // `Inactive`: nobody is subscribed.
            Ok(_) | Err(TrySendError::Inactive(_)) => {}
            Err(error) => tracing::warn!(%error, ?kind, "connectivity change not delivered"),
        }
    }
}
