//! Built-in commands for the bash sandbox.
//!
//! These commands are always available and provide special functionality:
//! - `reload`: Request to load file content back into context

mod reload;

pub use reload::{ReloadCommand, ReloadResponse};

use leash::IpcRouter;

/// Creates an IPC router with all built-in commands registered.
#[must_use]
pub fn builtin_router() -> IpcRouter {
    IpcRouter::new().register(ReloadCommand::default())
}
