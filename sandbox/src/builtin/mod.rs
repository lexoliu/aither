//! Built-in sandbox tools.

mod ask;
mod terminal;

pub use ask::AskCommand;
pub use terminal::{InputTerminalArgs, InputTerminalTool, KillTerminalArgs, KillTerminalTool};

use leash::IpcRouter;

/// Creates an IPC router with built-in commands (excluding ask, which needs LLM).
#[must_use]
pub fn builtin_router() -> IpcRouter {
    IpcRouter::new()
}
