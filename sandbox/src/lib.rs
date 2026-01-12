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
//! use aither_sandbox::{BashTool, ToolRegistryBuilder, permission::AllowAll};
//! use std::sync::Arc;
//!
//! // Creates a random four-word working directory (e.g., amber-forest-thunder-pearl/)
//! // with outputs/ subdirectory and IPC command wrapper scripts
//! let tool = BashTool::new_in(std::env::temp_dir(), AllowAll, executor).await?;
//! let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
//! let tool = tool.with_registry(registry);
//!
//! // Access the working directory path
//! tracing::info!("Working dir: {}", tool.working_dir().display());
//! ```
//!
//! # Registering Tools as IPC Commands
//!
//! Tools can be made available to bash scripts running in the sandbox:
//!
//! ```rust,ignore
//! use aither_sandbox::{ToolRegistryBuilder, register_tool_command};
//! use aither_sandbox::builtin::builtin_router;
//!
//! // 1. Configure tools
//! let mut registry = ToolRegistryBuilder::new();
//! registry.configure_tool(websearch_tool);
//! registry.configure_tool(webfetch_tool);
//! let registry = std::sync::Arc::new(registry.build("./outputs"));
//!
//! // 2. Create router with built-in commands (reload)
//! let router = builtin_router();
//!
//! // 3. Register tool commands with the router
//! let router = register_tool_command(router, registry.clone(), "websearch");
//! let router = register_tool_command(router, registry.clone(), "webfetch");
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
mod naming;
mod output;

/// Built-in IPC commands (ask, reload).
pub mod builtin;

/// Background job registry for tracking tasks.
pub mod job_registry;

/// Permission handling for bash modes.
pub mod permission;

pub use bash::{
    BackgroundTaskReceiver, BashArgs, BashError, BashResult, BashTool, BashToolFactory,
    BashToolFactoryError, BashToolFactoryReceiver, CompletedTask, Configured, Unconfigured,
    bash_tool_factory_channel,
};
pub use command::{
    DynBashTool, DynToolHandler, IpcToolCommand, ToolCallCommand, ToolCommand, ToolRegistry,
    ToolRegistryBuilder, cli_to_json, register_tool_command, register_tool_direct,
    schema_to_help,
};
pub use job_registry::{JobInfo, JobRegistry, JobStatus};
pub use output::{Content, OutputEntry, OutputFormat, OutputStore, PendingUrl};
pub use permission::{BashMode, PermissionHandler};
