//! Built-in commands for the bash sandbox.
//!
//! These commands are always available and provide special functionality:
//! - `ask`: Query a fast LLM about piped content
//! - `reload`: Request to load file content back into context

mod ask;
mod reload;

pub use ask::AskCommand;
pub use reload::{ReloadCommand, ReloadResponse};

use leash::IpcRouter;

/// Creates an IPC router with built-in commands (excluding ask, which needs LLM).
#[must_use]
pub fn builtin_router() -> IpcRouter {
    IpcRouter::new().register(ReloadCommand::default())
}
