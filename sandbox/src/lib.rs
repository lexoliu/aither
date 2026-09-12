//! Terminal-first command execution with heel sandboxing.
//!
//! This crate provides a terminal-first execution model where:
//! - LLM has a single `terminal` tool with different permission modes
//! - All tools are exposed as CLI commands via IPC
//! - Commands can be piped and composed freely
//! - Outputs are stored with URLs for context management
//!
//! # Setting Up the Terminal Tool
//!
//! ```rust,ignore
//! use aither_sandbox::{TerminalTool, ToolRegistryBuilder, permission::NoopPermissionHandler};
//! use std::sync::Arc;
//!
//! // Creates a random four-word working directory (e.g., amber-forest-thunder-pearl/)
//! // with outputs/ subdirectory and IPC command wrapper scripts
//! let tool = TerminalTool::new_in(std::env::temp_dir(), NoopPermissionHandler, executor).await?;
//! let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
//! let tool = tool.with_registry(registry);
//!
//! // Access the working directory path
//! tracing::info!("Working dir: {}", tool.working_dir().display());
//! ```
//!
//! # Registering Tools as IPC Commands
//!
//! Tools can be made available to terminal sessions running in the sandbox:
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

mod bollard_exec;
mod command;
mod container;
mod naming;
mod output;
mod output_compress;
mod shell_session;
mod stdin_watch;
mod terminal;

/// Built-in IPC commands (ask, reload).
pub mod builtin;

/// Background job registry for tracking tasks.
pub mod job_registry;

/// Declarative host-based network policy.
pub mod network_policy;

/// Permission handling for terminal modes.
pub mod permission;

pub use bollard_exec::BollardContainerExec;
pub use command::{
    CommandEnvelope, CommandPayload, DynTerminalTool, DynToolHandler, IpcToolCommand,
    ToolCallCommand, ToolCommand, ToolRegistry, ToolRegistryBuilder, cli_to_json,
    register_ipc_gateway_command, register_tool_command, register_tool_direct, schema_to_help,
};
pub use container::{
    ContainerImageSpec, ContainerLaunchSpec, ContainerRuntimeKind, MountAccess, MountRoot,
    MountRootError, MountSpec, RuntimePreference,
};
/// Re-export of heel's rolling network audit log for terminal-tool configuration.
pub use heel::NetworkAuditLog;
pub use job_registry::{JobInfo, JobRegistry, JobStatus, TerminalDelta};
pub use network_policy::{
    CompiledNetworkPolicy, NetworkPolicyConfig, NetworkPolicyError, NetworkPolicyMode,
};
pub use output::{Content, MediaResolution, OutputEntry, OutputFormat, OutputStore, PendingUrl};
pub use permission::{PermissionHandler, TerminalMode};
pub use shell_session::{
    AnyContainerExec, ContainerExec, ContainerExecOutcome, ContainerShellRuntime, ShellBackend,
    ShellRuntimeAvailability, ShellSessionRegistry, SshRuntimeProfile, SshServer,
    SshSessionAuthorizer, bootstrap_ssh_runtime,
};
pub use stdin_watch::{
    STDIN_WATCH_INTERVAL, TERMINAL_STDIN_BLOCKED_NOTICE, detect_stdin_blocked_for_local_pid,
    is_waiting_on_stdin,
};
pub use terminal::{
    BackgroundReason, BackgroundTaskReceiver, CompletedTask, Configured, PermissionEvent,
    PermissionEventReceiver, PermissionEventStage, TerminalArgs, TerminalError,
    TerminalExecutionMode, TerminalResult, TerminalTool, TerminalToolFactory,
    TerminalToolFactoryError, TerminalToolFactoryReceiver, Unconfigured, open_network_audit_log,
    terminal_tool_factory_channel,
};
