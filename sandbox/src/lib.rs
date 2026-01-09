//! Bash-based tool execution with leash sandboxing.
//!
//! This crate provides a bash-centric tool execution model where:
//! - LLM has a single `bash` tool with different permission modes
//! - All tools are exposed as CLI commands via IPC
//! - Commands can be piped and composed freely
//! - Outputs are stored with URLs for context management
//!
//! # Setting Up the Bash Tool
//!
//! ```rust,ignore
//! use aither_sandbox::{BashTool, permission::AllowAll};
//!
//! // Creates a random four-word working directory (e.g., amber-forest-thunder-pearl/)
//! // with outputs/ subdirectory and IPC command wrapper scripts
//! let tool = BashTool::new(AllowAll).await?;
//!
//! // Access the working directory path
//! println!("Working dir: {}", tool.working_dir().display());
//! ```
//!
//! # Registering Tools as IPC Commands
//!
//! Tools can be made available to bash scripts running in the sandbox:
//!
//! ```rust,ignore
//! use aither_sandbox::{configure_tool, register_tool_command};
//! use aither_sandbox::builtin::builtin_router;
//!
//! // 1. Configure tools (stores handlers in global registry)
//! configure_tool(websearch_tool);
//! configure_tool(webfetch_tool);
//!
//! // 2. Create router with built-in commands (reload)
//! let router = builtin_router();
//!
//! // 3. Register tool commands with the router
//! let router = register_tool_command(router, "websearch");
//! let router = register_tool_command(router, "webfetch");
//! ```
//!
//! # Built-in Commands
//!
//! The [`builtin`] module provides commands always available in the sandbox:
//!
//! - `reload <url>` - Request to load file content back into agent context

#![allow(clippy::module_name_repetitions)]

mod bash;
mod command;
mod output;

/// Built-in IPC commands (ask, reload).
pub mod builtin;

/// Permission handling for bash modes.
pub mod permission;

pub use bash::{BackgroundTaskReceiver, BashArgs, BashError, BashResult, BashTool, CompletedTask};
pub use command::{
    configure_raw_handler, configure_tool, create_child_bash_tool, get_tool_help,
    get_tool_primary_arg, get_tool_stdin_arg, is_tool_configured, register_tool_command,
    registered_tool_names, set_bash_tool_factory, DynBashTool, DynToolHandler, ToolCallCommand,
    ToolCommand,
};
pub use output::{Content, OutputEntry, OutputFormat, OutputStore, PendingUrl};
pub use permission::{BashMode, PermissionHandler};
