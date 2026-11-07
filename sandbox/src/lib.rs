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
//! // Creates a working directory heel names with four random words (e.g.,
//! // amber-forest-thunder-pearl/), not a leased id, with an outputs/
//! // subdirectory and IPC command wrapper scripts; the ids the model reads are
//! // leased from `with_id_allocator`'s allocator
//! let tool = TerminalTool::new_in(std::env::temp_dir(), NoopPermissionHandler, executor).await?;
//! let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
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
mod command_set;
mod container;
#[cfg(all(unix, feature = "jobserver"))]
mod container_jobserver;
mod foreground_wait;
mod grants;
mod heel_binary;
mod memory;
pub mod monitor;
mod notices;
mod outage;
mod output;
mod output_compress;
mod policy;
mod process_tree;
mod script_language;
mod session_policy;
mod shell_session;
mod shell_syntax;
mod stdin_watch;
mod subagent;
mod terminal;
mod tool_distribution;
mod tool_environment;
mod tool_python;
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

/// Isolated git for tests, shared with dependent crates through the `test-git` feature.
#[cfg(any(test, feature = "test-git"))]
pub mod test_git;

/// Id allocators for tests, shared with dependent crates through the
/// `test-ids` feature.
#[cfg(any(test, feature = "test-ids"))]
pub mod test_ids;

pub use bollard_exec::BollardContainerExec;
pub use command::{
    AnyCommandGate, AnyCommandObserver, CommandCall, CommandEnvelope, CommandGate, CommandObserver,
    CommandPayload, CommandRole, CommandUsage, DynTerminalTool, DynToolHandler, GateDecision,
    IpcToolCommand, KeptExchangeReceiver, POSITIONAL_KEYWORD, PositionalLayoutError,
    ToolCallCommand, ToolCommand, ToolRegistry, ToolRegistryBuilder, cli_to_json,
    output_saved_text, register_ipc_gateway_command, register_tool_command, register_tool_direct,
    schema_to_help, tool_output_to_payload, validate_positional_layout,
};
pub use command_set::{AnyCommandSet, CommandSet};
pub use container::{
    ContainerImageSpec, ContainerLaunchSpec, ContainerPath, ContainerRuntimeKind, MountAccess,
    MountRoot, MountRootError, MountSpec, RelativeContainerPath, RuntimePreference,
};
pub use foreground_wait::ForegroundWait;
pub use grants::{GitDirAccess, GrantSet, TerminalGrants};
/// Re-export of heel's rolling network audit log for terminal-tool configuration.
pub use heel::NetworkAuditLog;
/// Re-export of heel's scheduling priority types for terminal-tool configuration.
pub use heel::{CpuPriority, IoPriority, NiceLevel, ProcessPriority};
pub use job_registry::{
    Delivery, JobInfo, JobPlace, JobRegistry, JobRegistryService, JobStatus, KillCause,
    OwedCompletion, PendingCompletion, TerminalDelta, TerminalReadError, UnchangedTaskRead,
    job_registry_channel,
};
#[cfg(all(unix, feature = "jobserver"))]
pub use jobserver::{
    CommandFifo, JobBroker, JobBrokerService, JobPool, JobserverError, job_broker_channel,
};
pub use memory::{
    AnyContainerMemory, AnyFootprintReader, AnyFootprintRegistry, CgroupMemory, ContainerMemory,
    ContainerSnapshot, FileFootprintRegistry, FootprintError, FootprintReader, FootprintRegistry,
    JOB_MARKER_ENV, JobFootprint, JobMemory, MemoryDomain, MemorySize, MemorySnapshot,
    ProbedContainerMemory, ResponseWindow, SystemFootprints,
};
pub use monitor::{
    MonitorArgs, MonitorEnd, MonitorEnded, MonitorIncrement, MonitorRefusal, MonitorSource,
    MonitorStalled, MonitorTool,
};
pub use network_policy::{
    CompiledNetworkPolicy, NetworkPolicyConfig, NetworkPolicyError, NetworkPolicyMode,
};
pub use notices::{
    AnySandboxNotices, CommandKilled, DefaultSandboxNotices, MonitorStarted, Refused,
    ReservedForMain, SandboxNotices,
};
pub use output::{MediaResolution, OutputFormat, OutputStore};
pub use permission::{PermissionHandler, PermissionRequest, Review, TerminalMode};
pub use policy::{
    AnyCommandPolicy, CommandPolicy, PolicyDecision, PolicyRefusal, ScriptCheck, ScriptEnd,
    ScriptOrigin, ScriptOutcome,
};
pub use process_tree::{
    JOB_MARKER_VAR, JobProcess, KillTreeError, ProcessTableError, kill_process_tree,
};
pub use script_language::ScriptLanguage;
pub use session_policy::SessionSandboxPolicy;
pub use shell_session::{
    AnyContainerExec, AnySshSessionAuthorizer, BridgeAddress, BridgeAddressError, ContainerExec,
    ContainerExecOutcome, ContainerIpcRoute, ContainerProbe, ContainerProbeError,
    ContainerShellRuntime, ExecLifetime, ShellBackend, ShellRuntimeAvailability,
    ShellSessionRegistry, SshAuthorizationError, SshRuntimeProfile, SshServer,
    SshSessionAuthorizer, bootstrap_ssh_runtime,
};
pub use shell_syntax::{
    CommandId, LineId, PipelineId, PipelineStage, ShellCommand, ShellParseError, ShellScript,
};
pub use stdin_watch::StdinWatchSignal;
pub use subagent::{
    AnySubagentSpaces, CommitLedger, CommitLedgerError, CommitMark, ImportBudget, PreparedSpace,
    RunSpace, SpacePreparationError, SubagentCommit, SubagentGrant, SubagentGrantError,
    SubagentSpaces,
};
pub use terminal::{
    BackgroundTaskEvent, BackgroundTaskReceiver, BackgroundTaskSender, CompletedCommand,
    CompletedTask, Configured, DEFAULT_FOREGROUND_WAIT, DEFAULT_LONG_JOB_ALERT_AFTER,
    DEFAULT_RESPONSE_WINDOW, DEFAULT_SUBAGENT_IMPORT_LIMIT, DEFAULT_TIMEOUT_CAP, DescendantsKilled,
    DirectoryReset, DirectoryResetReason, JobAlert, LongRunningJob, PermissionEvent,
    PermissionEventReceiver, PermissionEventStage, StartedBackgroundTask, StdinBlockedJob,
    StdinProbeFailedJob, TerminalArgs, TerminalBinding, TerminalEnvSource, TerminalError,
    TerminalExecutionMode, TerminalRunError, TerminalRunner, TerminalScript, TerminalTool,
    TerminalToolFactory, TerminalToolFactoryError, TimeoutKill, Unconfigured,
    background_task_channel, open_network_audit_log,
};
pub use tool_distribution::{
    HostProgramArtifact, PinnedArtifact, ToolArtifact, ToolArtifactFormat, ToolArtifactLocation,
    ToolDistribution,
};
pub use tool_runtime::{
    DistributedTool, HostProgram, HostProgramPath, ImageTool, ImageToolDefect, ImageTools,
    ManagedTool, PreparedEnvironment, PreparedTools, PythonPin, PythonRuntime, ToolBinary,
    ToolBundle, ToolCatalog, ToolRuntimeError, ToolTarget,
};
pub use tool_transport::{
    ContainerToolTransport, SshToolTransport, ToolEnvironmentInfo, ToolTransport,
    probe_tool_environment,
};
