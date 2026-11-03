//! Sleep and wake from logind's `PrepareForSleep(b)` signal on the system
//! bus: `true` just before the system suspends, `false` just after it
//! resumes.
//!
//! <https://www.freedesktop.org/software/systemd/man/latest/org.freedesktop.login1.html>

use futures_lite::StreamExt;
use zbus::{Connection, Proxy, proxy::SignalStream};

use crate::{ConnectivityError, hub::Sink};

const SERVICE: &str = "org.freedesktop.login1";
const PATH: &str = "/org/freedesktop/login1";
const INTERFACE: &str = "org.freedesktop.login1.Manager";

pub struct Watcher {
    signals: SignalStream<'static>,
}

impl Watcher {
    /// Connects to the system bus and subscribes to `PrepareForSleep`.
    pub async fn connect() -> Result<Self, ConnectivityError> {
        let register = |error: zbus::Error| ConnectivityError::Register {
            monitor: "logind PrepareForSleep",
            reason: error.to_string(),
        };
        let connection = Connection::system().await.map_err(register)?;
        let proxy = Proxy::new(&connection, SERVICE, PATH, INTERFACE)
            .await
            .map_err(register)?;
        let signals = proxy
            .receive_signal("PrepareForSleep")
            .await
            .map_err(register)?;
        Ok(Self { signals })
    }

    /// Publishes every sleep and wake until the bus connection ends.
    pub async fn watch(mut self, sink: &Sink) {
        while let Some(message) = self.signals.next().await {
            match message.body().deserialize::<bool>() {
                Ok(true) => sink.sleeping(),
                Ok(false) => sink.woke(),
                Err(error) => tracing::warn!(%error, "malformed PrepareForSleep signal"),
            }
        }
        tracing::error!("system bus connection closed, sleep and wake are no longer tracked");
    }
}
