//! Built-in commands for the bash sandbox.
//!
//! These commands are always available and provide special functionality:
//! - `ask`: Query a fast LLM about piped content

mod ask;

pub use ask::AskCommand;

use leash::IpcRouter;

/// Creates an IPC router with built-in commands (excluding ask, which needs LLM).
#[must_use]
pub fn builtin_router() -> IpcRouter {
    IpcRouter::new()
}
