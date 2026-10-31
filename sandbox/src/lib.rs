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
mod kill_guard;
mod naming;
mod output;
mod output_compress;
mod process_tree;
mod shell_session;
mod stdin_watch;
mod subagent;
mod terminal;
mod tool_distribution;
mod tool_environment;
mod tool_runtime;
mod tool_transport;

/// Built-in IPC commands (ask, reload).
pub mod builtin;

/// Background job registry for tracking tasks.
pub mod job_registry;

/// Jobserver token pool brokering across commands.
#[cfg(all(unix, feature = "jobserver"))]
pub mod jobserver;

/// Declarative host-based network policy.
pub mod network_policy;

/// Permission handling for terminal modes.
pub mod permission;

pub use bollard_exec::BollardContainerExec;
pub use command::{
    AnyCommandGate, AnyCommandObserver, CommandCall, CommandEnvelope, CommandGate, CommandObserver,
    CommandPayload, CommandRole, DynTerminalTool, DynToolHandler, GateDecision, IpcToolCommand,
    ToolCallCommand, ToolCommand, ToolRegistry, ToolRegistryBuilder, cli_to_json,
    register_ipc_gateway_command, register_tool_command, register_tool_direct, schema_to_help,
    tool_output_to_payload,
};
pub use container::{
    ContainerImageSpec, ContainerLaunchSpec, ContainerPath, ContainerRuntimeKind, MountAccess,
    MountRoot, MountRootError, MountSpec, RelativeContainerPath, RuntimePreference,
};
/// Re-export of heel's rolling network audit log for terminal-tool configuration.
pub use heel::NetworkAuditLog;
/// Re-export of heel's scheduling priority types for terminal-tool configuration.
pub use heel::{CpuPriority, IoPriority, NiceLevel, ProcessPriority};
pub use job_registry::{
    JobInfo, JobRegistry, JobRegistryService, JobStatus, TerminalDelta, job_registry_channel,
};
#[cfg(all(unix, feature = "jobserver"))]
pub use jobserver::{
    CommandFifo, JobBroker, JobBrokerService, JobPool, JobserverError, job_broker_channel,
};
pub use kill_guard::CommandRefusal;
pub use network_policy::{
    CompiledNetworkPolicy, NetworkPolicyConfig, NetworkPolicyError, NetworkPolicyMode,
};
pub use output::{MediaResolution, OutputFormat, OutputStore};
pub use permission::{PermissionHandler, PermissionRequest, Review, TerminalMode};
pub use process_tree::{
    JOB_MARKER_VAR, JobProcess, KillTreeError, ProcessTableError, kill_process_tree,
};
pub use shell_session::{
    AnyContainerExec, AnySshSessionAuthorizer, BridgeAddress, BridgeAddressError, ContainerExec,
    ContainerExecOutcome, ContainerIpcRoute, ContainerShellRuntime, ExecLifetime, ShellBackend,
    ShellRuntimeAvailability, ShellSessionRegistry, SshAuthorizationError, SshRuntimeProfile,
    SshServer, SshSessionAuthorizer, bootstrap_ssh_runtime,
};
pub use stdin_watch::StdinWatchSignal;
pub use subagent::{
    CommitLedger, CommitLedgerError, CommitMark, ImportBudget, SubagentCommit, SubagentGrant,
    SubagentGrantError,
};
pub use terminal::{
    BackgroundTaskEvent, BackgroundTaskReceiver, BackgroundTaskSender, CompletedCommand,
    CompletedTask, Configured, DEFAULT_FOREGROUND_WAIT, DEFAULT_LONG_JOB_ALERT_AFTER,
    DEFAULT_SUBAGENT_IMPORT_LIMIT, DEFAULT_TIMEOUT_CAP, DescendantsKilled, DirectoryReset,
    DirectoryResetReason, JobAlert, LongRunningJob, PermissionEvent, PermissionEventReceiver,
    PermissionEventStage, StartedBackgroundTask, StdinBlockedJob, StdinProbeFailedJob,
    TerminalArgs, TerminalBinding, TerminalError, TerminalExecutionMode, TerminalTool,
    TerminalToolFactory, TerminalToolFactoryError, Unconfigured, background_task_channel,
    open_network_audit_log,
};
pub use tool_distribution::{
    HostProgramArtifact, PinnedArtifact, ToolArtifact, ToolArtifactFormat, ToolArtifactLocation,
    ToolDistribution,
};
pub use tool_runtime::{
    DistributedTool, HostProgram, HostProgramBinary, HostProgramPath, ManagedTool,
    PreparedEnvironment, PreparedTools, ToolBinary, ToolBundle, ToolCatalog, ToolRuntimeError,
    ToolTarget,
};
pub use tool_transport::{
    ContainerToolTransport, SshToolTransport, ToolEnvironmentInfo, ToolTransport,
    probe_tool_environment,
};
