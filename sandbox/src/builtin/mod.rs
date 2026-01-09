//! Built-in tools for the bash sandbox.
//!
//! These tools are always available and provide special functionality:
//! - `ask`: Query a fast LLM about piped content
//! - `tasks`: List background tasks (running, completed, failed, killed)
//! - `kill`: Terminate a background task by PID

mod ask;
mod kill_tool;
mod tasks;

pub use ask::AskCommand;
pub use kill_tool::{KillArgs, KillTool};
pub use tasks::{TasksArgs, TasksTool};
