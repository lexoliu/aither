//! Built-in tools for the bash sandbox.
//!
//! These tools are always available and provide special functionality:
//! - `ask`: Query a fast LLM about piped content
//! - `tasks`: List background tasks (running, completed, failed, killed)
//! - `stop`: Terminate a background task by PID

mod ask;
mod stop;
mod tasks;

pub use ask::AskCommand;
pub use stop::{StopArgs, StopTool};
pub use tasks::{TasksArgs, TasksTool};

use leash::IpcRouter;

/// Creates an IPC router with built-in commands (excluding ask, which needs LLM).
#[must_use]
pub fn builtin_router() -> IpcRouter {
    IpcRouter::new()
}
