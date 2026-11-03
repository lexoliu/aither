//! The monitor of the platform this crate is built for.

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::Monitor;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::Monitor;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::Monitor;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod unsupported {
    use crate::{ConnectivityError, hub::Sink};

    /// The platform has no monitor.
    pub struct Monitor;

    impl Monitor {
        pub fn start(_sink: Sink) -> Result<Self, ConnectivityError> {
            Err(ConnectivityError::Unsupported)
        }
    }
}
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub use unsupported::Monitor;
