//! Linux: rtnetlink for reachability, logind for sleep and wake.
//!
//! - rtnetlink groups and route dumps: <https://man7.org/linux/man-pages/man7/rtnetlink.7.html>
//! - logind `PrepareForSleep`:
//!   <https://www.freedesktop.org/software/systemd/man/latest/org.freedesktop.login1.html>
//!
//! Both watchers run on one dedicated thread that stops when the [`Monitor`]
//! drops.

mod decode;
mod logind;
mod route;

use std::thread::JoinHandle;

use futures_lite::future;

use crate::{ConnectivityError, hub::Sink};

/// Watches reachability and power state until dropped.
pub struct Monitor {
    stop: async_channel::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Monitor {
    pub fn start(sink: Sink) -> Result<Self, ConnectivityError> {
        let routes = route::Watcher::open(&sink)?;
        let (stop, stopped) = async_channel::bounded::<()>(1);
        let (ready_tx, ready_rx) = async_channel::bounded(1);
        let thread = std::thread::Builder::new()
            .name("aither-connectivity".into())
            .spawn(move || run(routes, &sink, &stopped, &ready_tx))
            .map_err(|error| ConnectivityError::Register {
                monitor: "linux watcher thread",
                reason: error.to_string(),
            })?;
        // The thread reports whether the system bus connection came up.
        match ready_rx.recv_blocking() {
            Ok(Ok(())) => Ok(Self {
                stop,
                thread: Some(thread),
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => Err(ConnectivityError::Register {
                monitor: "linux watcher thread",
                reason: "the thread exited before reporting".into(),
            }),
        }
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop.close();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The watcher thread: runs both watchers until `stopped` closes.
#[allow(
    clippy::disallowed_methods,
    reason = "this is a dedicated OS thread, nothing else runs on it"
)]
fn run(
    routes: route::Watcher,
    sink: &Sink,
    stopped: &async_channel::Receiver<()>,
    ready: &async_channel::Sender<Result<(), ConnectivityError>>,
) {
    future::block_on(async {
        let logind = match logind::Watcher::connect().await {
            Ok(logind) => logind,
            Err(error) => {
                let _ = ready.send(Err(error)).await;
                return;
            }
        };
        let _ = ready.send(Ok(())).await;
        let watchers = future::zip(routes.watch(sink), logind.watch(sink));
        // `stopped` only ever closes; nothing is sent on it.
        future::or(
            async {
                let _ = stopped.recv().await;
            },
            async {
                let _ = watchers.await;
            },
        )
        .await;
    });
}
