//! Built-in commands for the bash sandbox.
//!
//! These commands are always available and provide special functionality:
//! - `ask`: Query a fast LLM about piped content

mod ask;
mod stop;
mod tasks;

pub use ask::AskCommand;
pub use stop::StopTool;
pub use tasks::TasksTool;

use leash::IpcRouter;

/// Creates an IPC router with built-in commands (excluding ask, which needs LLM).
#[must_use]
pub fn builtin_router() -> IpcRouter {
    IpcRouter::new()
}
