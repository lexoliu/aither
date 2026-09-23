//! Terminal command tool implementation with heel sandboxing.
//!
//! Provides the main `terminal` tool that LLMs use to execute terminal work.
//! Despite the historical tool name, execution does not have to go through a
//! literal `terminal` process: simple commands may be executed directly, while shell
//! syntax falls back to a real shell when needed.
//! Commands run in a sandbox with configurable permission modes.
//!
//! Each `TerminalTool` creates a shared working directory with four random words
//! (e.g., `amber-forest-thunder-pearl/`). All terminal executions share this
//! directory, but each execution gets a fresh sandbox (new TTY/process).

#[cfg(unix)]
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::{
    borrow::Cow,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use aither_core::llm::{IntoToolResult, Tool, ToolResult};
use askama::Template;
use async_channel::{Receiver, Sender};
#[cfg(unix)]
use async_io::Async;
use executor_core::{Executor, Task};
#[cfg(unix)]
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::{
    AllowAll, Audited, DomainRequest, IpcRouter, NetworkAuditLog, NetworkPolicy, Sandbox,
    SandboxConfig, SecurityConfig, StdioConfig, WorkingDir,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shell_words::split as split_shell_words;
use tracing::{debug, info, warn};

use crate::{
    builtin::builtin_router,
    command::ToolRegistry,
    job_registry::{JobRegistry, job_registry_channel},
    output::{
        Content, INLINE_OUTPUT_LIMIT, MediaResolution, OutputEntry, OutputFormat, OutputStore,
        save_raw_to_file,
    },
    permission::{PermissionError, PermissionHandler, TerminalMode},
    shell_session::{
        ContainerShellRuntime, ShellBackend, ShellRuntimeAvailability, ShellSessionRegistry,
        SshRuntimeProfile, bootstrap_ssh_runtime,
    },
    stdin_watch::{
        STDIN_WATCH_INTERVAL, TERMINAL_STDIN_BLOCKED_NOTICE, detect_stdin_blocked_for_local_pid,
    },
};

fn erase_terminal_tool<T>(tool: T) -> crate::command::DynTerminalToolEntry
where
    T: Tool + 'static,
{
    use crate::command::{DynTerminalToolEntry, DynToolHandler};
    use aither_core::llm::tool::ToolDefinition;

    let definition = ToolDefinition::new(&tool);
    let tool = Arc::new(tool);
    let handler: DynToolHandler = Arc::new(move |args: &str| {
        let tool = tool.clone();
        let args = args.to_string();
        Box::pin(async move {
            let parsed = match serde_json::from_str::<T::Arguments>(&args) {
                Ok(parsed) => parsed,
                Err(error) => return ToolResult::error(format!("Parse error: {error}")),
            };
            match tool.call(parsed).await {
                Ok(output) => output
                    .into_tool_result()
                    .unwrap_or_else(|error| ToolResult::error(format!("Error: {error}"))),
                Err(error) => ToolResult::error(format!("Error: {error}")),
            }
        })
    });

    DynTerminalToolEntry {
        definition,
        handler,
    }
}

/// Generate a random four-word ID (e.g., "amber-forest-thunder-pearl").
fn random_task_id() -> String {
    crate::naming::random_word_slug(4)
}

/// Opens a daily-rotated network audit log under `dir`, creating the directory if needed.
///
/// Keeps at most `max_files` rotated files. Attach the log with
/// [`TerminalTool::with_network_audit_log`] to record every sandboxed
/// network policy decision.
///
/// # Errors
/// Returns an error when the directory cannot be created or the log cannot
/// be opened.
pub async fn open_network_audit_log(
    dir: impl AsRef<Path>,
    max_files: usize,
) -> Result<NetworkAuditLog, TerminalError> {
    let dir = dir.as_ref();
    async_fs::create_dir_all(dir).await?;
    NetworkAuditLog::rolling_daily(dir, max_files).map_err(|error| {
        TerminalError::SandboxSetup(format!("failed to open network audit log: {error}"))
    })
}

const fn supports_stdin_blocked_notice(backend: ShellBackend) -> bool {
    matches!(backend, ShellBackend::Container)
        || (cfg!(target_os = "linux") && matches!(backend, ShellBackend::Local))
}

#[derive(Clone)]
struct PermissionNetworkPolicy<P> {
    permission_handler: Arc<P>,
}

impl<P: PermissionHandler + 'static> NetworkPolicy for PermissionNetworkPolicy<P> {
    async fn check(&self, request: &DomainRequest) -> bool {
        self.permission_handler
            .check_domain(request.host(), request.port())
            .await
    }
}

fn parse_direct_command(script: &str) -> Result<(String, Vec<String>), TerminalError> {
    let parts = split_shell_words(script)
        .map_err(|error| TerminalError::Execution(format!("failed to parse command: {error}")))?;
    let Some((program, args)) = parts.split_first() else {
        return Err(TerminalError::Execution(
            "command must not be empty".to_string(),
        ));
    };
    Ok((program.clone(), args.to_vec()))
}

enum TerminalLaunch {
    Direct { program: String, args: Vec<String> },
    Shell { program: String, args: Vec<String> },
}

fn script_requires_shell(script: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for ch in script.chars() {
        if escaped {
            escaped = false;
            continue;
        }

        match ch {
            '\\' if !in_single => escaped = true,
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '\n' | '\r' | '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' | '*' | '?' | '['
            | ']' | '{' | '}' | '!' | '~'
                if !in_single && !in_double =>
            {
                return true;
            }
            _ => {}
        }
    }

    false
}

fn looks_like_env_assignment(token: &str) -> bool {
    let Some((name, _value)) = token.split_once('=') else {
        return false;
    };
    !name.is_empty() && name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn is_shell_builtin(program: &str) -> bool {
    matches!(
        program,
        "alias"
            | "bg"
            | "break"
            | "cd"
            | "command"
            | "continue"
            | "eval"
            | "exec"
            | "exit"
            | "export"
            | "fg"
            | "jobs"
            | "read"
            | "readonly"
            | "return"
            | "set"
            | "shift"
            | "source"
            | "times"
            | "trap"
            | "type"
            | "ulimit"
            | "umask"
            | "unalias"
            | "unset"
    )
}

fn shell_launch(script: &str) -> (String, Vec<String>) {
    #[cfg(windows)]
    {
        let shell = std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string());
        return (shell, vec!["/C".to_string(), script.to_string()]);
    }

    #[cfg(not(windows))]
    {
        ("sh".to_string(), vec!["-c".to_string(), script.to_string()])
    }
}

fn build_terminal_launch(script: &str) -> Result<TerminalLaunch, TerminalError> {
    let script = script.trim();
    if script.is_empty() {
        return Err(TerminalError::Execution(
            "command must not be empty".to_string(),
        ));
    }

    if script_requires_shell(script) {
        let (program, args) = shell_launch(script);
        return Ok(TerminalLaunch::Shell { program, args });
    }

    let (program, args) = parse_direct_command(script)?;
    if looks_like_env_assignment(&program) || is_shell_builtin(&program) {
        let (program, args) = shell_launch(script);
        return Ok(TerminalLaunch::Shell { program, args });
    }

    Ok(TerminalLaunch::Direct { program, args })
}

async fn ensure_mode_allowed<P: PermissionHandler>(
    permission_handler: &P,
    mode: TerminalMode,
    script: &str,
) -> Result<(), TerminalError> {
    if !mode.requires_approval() {
        return Ok(());
    }

    let allowed = permission_handler.check(mode, script).await?;
    if allowed {
        Ok(())
    } else {
        Err(TerminalError::PermissionDenied(mode))
    }
}

/// Execute terminal commands in a sandboxed environment.
///
/// The primary interface to all system capabilities. Scripts run in a sandbox
/// with full read access to the host filesystem but writes contained to a
/// dedicated working directory.
///
/// ## Output Handling
///
/// By default, stdout is **compressed** before being returned. Compression
/// is semantics-preserving: the meaning is identical to the raw output, but
/// the representation may differ. Specifically:
///
/// - **JSON** outputs (arrays of objects or single objects) are automatically
///   converted to **TSV** with dot-notation flattened column headers, but
///   only when the TSV is smaller than the original JSON.
/// - **Source code** outputs (detected via content analysis) are folded
///   using syntax-aware block folding with line numbers. The full raw
///   output is saved to a file whose URL is included in the result.
/// - **Empty lines** and **invisible/control characters** are stripped.
///
/// If you need the exact verbatim output (e.g., for checksums, binary
/// protocols, or diffing), either set `raw: true`, pipe through further
/// processing in the script, or redirect to a file within the script.
///
/// Large outputs are automatically saved to file to manage context. When this
/// happens, you receive the file path and can process it using standard Unix
/// tools (head, tail, grep, less) or pipe through `ask` for summarization.
///
/// ## Built-in Commands
///
/// IPC-backed commands are configured by the host application and are exposed
/// inside the terminal runtime as ordinary CLI commands. Inspect the
/// application-provided command list and each command's `--help` output instead
/// of assuming a fixed built-in command inventory.
///
/// ## Execution Modes
///
/// `terminal` is stateless. Each call selects its own runtime mode.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TerminalArgs {
    /// Short human-readable explanation of what this command is doing.
    /// This is shown in the UI instead of echoing the shell source.
    pub description: String,

    /// The terminal command payload to execute.
    pub script: String,

    /// Runtime execution mode.
    /// - "default": local runtime execution with network enabled
    /// - "sandboxed": local runtime execution with network denied
    /// - "unsafe": direct host execution (heel profile only)
    /// - "ssh": execute on a preconfigured SSH server (`ssh_server_id` required when multiple are configured)
    #[serde(default)]
    pub mode: TerminalExecutionMode,

    /// SSH server identifier used when `mode` is `ssh`.
    /// Optional when exactly one SSH server is configured.
    #[serde(default)]
    pub ssh_server_id: Option<String>,

    /// Expected output format for proper handling.
    /// - "text" (default): plain text
    /// - "image": image data (auto-loaded to context)
    /// - "video": video data
    /// - "binary": binary data
    /// - "auto": auto-detect
    #[serde(default)]
    pub expect: OutputFormat,

    /// Requested delivery resolution for media returned to the model.
    /// - "auto" (default): preserve the source resolution
    /// - "low": downscale to a 512px bounding box
    /// - "medium": downscale to a 1024px bounding box
    /// - "high": downscale to a 2048px bounding box
    /// - "native": preserve original media bytes without resizing
    #[serde(default)]
    pub resolution: MediaResolution,

    /// Per-command timeout in seconds.
    /// - 0: run in background immediately
    /// - >0: run foreground up to timeout, then move to background on timeout
    pub timeout: u64,

    /// Maximum number of output lines to include inline.
    /// For foreground-complete executions, this is the inline line budget
    /// before output offloads to file.
    /// For timeout-promoted background executions, only the first `max_lines`
    /// are returned immediately, while full output is redirected to file.
    /// Clamped to 800 max.
    /// Default: 200.
    #[serde(default = "default_max_lines")]
    pub max_lines: u32,

    /// When true, skip all output compression and return the verbatim
    /// stdout bytes. Use this when you need exact byte-level fidelity
    /// (checksums, binary protocols, diff inputs). Default: false.
    #[serde(default)]
    pub raw: bool,
}

/// Requested terminal execution backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TerminalExecutionMode {
    /// Use the tool default.
    #[default]
    Default,
    /// Use the sandboxed backend.
    Sandboxed,
    /// Use unsafe local execution.
    Unsafe,
    /// Use SSH execution.
    Ssh,
}

const fn default_max_lines() -> u32 {
    200
}

const MAX_LINES_CEILING: u32 = 800;

/// Why a foreground terminal command was promoted to the background.
///
/// Propagated to the UI so the background shelf can render a meaningful
/// caption ("Promoted after 2s timeout" vs "Waiting for input" vs explicit).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackgroundReason {
    /// Exceeded the per-call foreground timeout budget.
    Timeout {
        /// Timeout in seconds as configured by the caller.
        configured_seconds: u64,
    },
    /// stdin-blocked detector fired — process is waiting for input that
    /// will not arrive via the foreground path.
    StdinBlocked {
        /// Human-readable notice from the detector, suitable for display.
        notice: String,
    },
    /// Caller explicitly asked to run in the background (`timeout == 0`).
    Explicit,
}

/// Result of a terminal execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalResult {
    /// stdout output.
    pub stdout: OutputEntry,

    /// stderr output (if non-empty).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<OutputEntry>,

    /// Exit code of the script.
    pub exit_code: i32,

    /// Task ID for background execution.
    /// Four random words like "amber-forest-thunder-pearl".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,

    /// Status for background tasks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    /// Why the command was moved to the background (only populated when
    /// `status == "running"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_reason: Option<BackgroundReason>,
}

enum ForegroundDecision {
    Completed(Box<Result<TerminalResult, String>>),
    PromoteToBackground(BackgroundReason),
}

#[derive(Clone)]
struct BackgroundStartup {
    tx: async_channel::Sender<Result<(), String>>,
    reported: Arc<AtomicBool>,
}

impl BackgroundStartup {
    fn new(tx: async_channel::Sender<Result<(), String>>) -> Self {
        Self {
            tx,
            reported: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn report_ready(&self) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Ok(())).await;
        }
    }

    async fn report_failure(&self, error: &TerminalError) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Err(error.to_string())).await;
        }
    }
}

/// A completed background task result.
#[derive(Debug)]
pub struct CompletedTask {
    /// The task ID (four random words like "amber-forest-thunder-pearl").
    pub task_id: String,
    /// The original script that was executed.
    pub script: String,
    /// The result of execution.
    pub result: Result<TerminalResult, TerminalError>,
}

/// Stage of a permission-lifecycle event for `terminal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionEventStage {
    /// Permission request is waiting for a decision.
    Waiting,
    /// Permission request has been resolved.
    Resolved,
}

/// Permission-lifecycle event emitted by `terminal` while waiting for approval.
#[derive(Debug, Clone)]
pub struct PermissionEvent {
    /// Requested terminal mode.
    pub mode: TerminalMode,
    /// Script awaiting permission.
    pub script: String,
    /// Permission lifecycle stage.
    pub stage: PermissionEventStage,
}

/// Receiver for completed background tasks.
///
/// This can be cloned and used independently of the `TerminalTool` to poll for
/// completed background tasks. Multiple receivers share the same channel,
/// so completed tasks are distributed among receivers.
#[derive(Clone)]
pub struct BackgroundTaskReceiver {
    rx: Receiver<CompletedTask>,
    running_tasks: Arc<AtomicUsize>,
}

impl BackgroundTaskReceiver {
    /// Takes all completed background tasks without blocking.
    ///
    /// Returns an empty vector if no tasks have completed.
    #[must_use]
    pub fn take_completed(&self) -> Vec<CompletedTask> {
        let mut completed = Vec::new();
        while let Ok(task) = self.rx.try_recv() {
            completed.push(task);
        }
        completed
    }

    /// Checks if there are any completed tasks waiting.
    #[must_use]
    pub fn has_completed(&self) -> bool {
        !self.rx.is_empty()
    }

    /// Checks if there are any background tasks still running.
    ///
    /// This works by checking the sender count - `TerminalTool` holds one sender,
    /// and each running background task holds a cloned sender.
    #[must_use]
    pub fn has_running(&self) -> bool {
        self.running_tasks.load(Ordering::Acquire) > 0
    }

    /// Waits for the next completed task.
    ///
    /// Returns `None` if the channel is closed (all senders dropped).
    pub async fn recv(&self) -> Option<CompletedTask> {
        self.rx.recv().await.ok()
    }

    /// Waits for a completed task with a timeout.
    ///
    /// Returns `None` if no task completes within the timeout or if the channel is closed.
    pub async fn recv_timeout(&self, duration: std::time::Duration) -> Option<CompletedTask> {
        futures_lite::future::or(async { self.rx.recv().await.ok() }, async {
            futures_lite::future::yield_now().await;
            async_io::Timer::after(duration).await;
            None
        })
        .await
    }
}

impl std::fmt::Debug for BackgroundTaskReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundTaskReceiver")
            .field("pending", &self.rx.len())
            .field("running_tasks", &self.running_tasks.load(Ordering::Acquire))
            .finish()
    }
}

/// Receiver for permission wait/resume events.
#[derive(Clone)]
pub struct PermissionEventReceiver {
    rx: Receiver<PermissionEvent>,
}

impl PermissionEventReceiver {
    /// Takes pending permission events without blocking.
    #[must_use]
    pub fn take_pending(&self) -> Vec<PermissionEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    /// Waits for the next permission event.
    pub async fn recv(&self) -> Option<PermissionEvent> {
        self.rx.recv().await.ok()
    }
}

impl std::fmt::Debug for PermissionEventReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermissionEventReceiver")
            .field("pending", &self.rx.len())
            .finish()
    }
}

/// Factory for creating child terminal tools asynchronously.
#[derive(Clone, Debug)]
pub struct TerminalToolFactory {
    tx: Sender<Sender<crate::command::DynTerminalTool>>,
}

/// Receiver that serves terminal tool creation requests.
#[derive(Debug)]
pub struct TerminalToolFactoryReceiver {
    rx: Receiver<Sender<crate::command::DynTerminalTool>>,
}

/// Errors that can occur when requesting a child terminal tool.
#[derive(Debug, thiserror::Error)]
pub enum TerminalToolFactoryError {
    /// The factory service is not running.
    #[error("terminal tool factory is not available")]
    Unavailable,
    /// The factory failed to return a tool.
    #[error("terminal tool factory failed to return a tool")]
    NoResponse,
}

/// Creates a factory channel pair for spawning child terminal tools.
#[must_use]
pub fn terminal_tool_factory_channel() -> (TerminalToolFactory, TerminalToolFactoryReceiver) {
    let (tx, rx) = async_channel::unbounded();
    (
        TerminalToolFactory { tx },
        TerminalToolFactoryReceiver { rx },
    )
}

impl TerminalToolFactory {
    /// Requests a new child terminal tool from the factory service.
    ///
    /// # Errors
    /// Returns an error when the factory service is unavailable or does not respond.
    pub async fn create(
        &self,
    ) -> Result<crate::command::DynTerminalTool, TerminalToolFactoryError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(reply_tx)
            .await
            .map_err(|_| TerminalToolFactoryError::Unavailable)?;
        reply_rx
            .recv()
            .await
            .map_err(|_| TerminalToolFactoryError::NoResponse)
    }
}

impl TerminalToolFactoryReceiver {
    async fn serve<P, E>(self, terminal_tool: TerminalTool<P, E, Configured>)
    where
        P: PermissionHandler + 'static,
        E: Executor + Clone + 'static,
    {
        while let Ok(reply_tx) = self.rx.recv().await {
            let tool = terminal_tool.child().to_dyn();
            if reply_tx.send(tool).await.is_err() {
                warn!("terminal tool factory response channel dropped");
            }
        }
    }
}

/// The terminal tool for executing scripts in a sandbox.
///
/// Creates a shared working directory with four random words that persists
/// across all executions. Each execution creates a fresh sandbox but shares
/// the working directory.
///
/// The executor type `E` determines how async tasks are spawned for the IPC server.
/// Use `TokioExecutor` when running in a tokio runtime.
/// Marker type for a terminal tool without a configured registry.
#[derive(Clone, Debug)]
pub struct Unconfigured;

/// Marker type for a terminal tool with a configured registry.
#[derive(Clone, Debug)]
pub struct Configured {
    registry: Arc<ToolRegistry>,
}

/// The terminal tool for executing scripts in a sandbox.
pub struct TerminalTool<P, E, State = Unconfigured> {
    /// Shared working directory (four random words, e.g., `amber-forest-thunder-pearl/`)
    working_dir: PathBuf,
    /// Runtime registry for container and ssh execution metadata.
    shell_sessions: ShellSessionRegistry,
    /// Permission handler wrapped in Arc for sharing between parent and child tools.
    permission_handler: Arc<P>,
    executor: E,
    output_store: Arc<OutputStore>,
    job_registry: JobRegistry,
    /// Channel for receiving completed background tasks.
    completed_rx: Receiver<CompletedTask>,
    /// Channel for sending completed background tasks (cloned for each background task).
    completed_tx: Sender<CompletedTask>,
    /// Number of still-running spawned executions associated with this tool.
    running_tasks: Arc<AtomicUsize>,
    /// Channel for permission lifecycle events.
    permission_rx: Receiver<PermissionEvent>,
    /// Channel sender for permission lifecycle events.
    permission_tx: Sender<PermissionEvent>,
    /// Additional paths that should be writable in the sandbox.
    writable_paths: Vec<PathBuf>,
    /// Additional paths that should be readable (but not writable) in the sandbox.
    readable_paths: Vec<PathBuf>,
    /// Rolling audit log recording every sandboxed network policy decision.
    network_audit: Option<NetworkAuditLog>,
    /// Tool registry state.
    registry: State,
}

// Manual Clone impl because P doesn't need to be Clone (we use Arc<P>)
impl<P, E: Clone, State: Clone> Clone for TerminalTool<P, E, State> {
    fn clone(&self) -> Self {
        Self {
            working_dir: self.working_dir.clone(),
            shell_sessions: self.shell_sessions.clone(),
            permission_handler: self.permission_handler.clone(),
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            job_registry: self.job_registry.clone(),
            completed_rx: self.completed_rx.clone(),
            completed_tx: self.completed_tx.clone(),
            running_tasks: self.running_tasks.clone(),
            permission_rx: self.permission_rx.clone(),
            permission_tx: self.permission_tx.clone(),
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
            network_audit: self.network_audit.clone(),
            registry: self.registry.clone(),
        }
    }
}

impl<P, E: Executor + Clone + 'static> TerminalTool<P, E, Unconfigured> {
    /// Injects the shared runtime registry used by `terminal` execution mode resolution.
    #[must_use]
    pub fn with_shell_sessions(mut self, sessions: ShellSessionRegistry) -> Self {
        self.shell_sessions = sessions;
        self
    }

    /// Sets dynamic runtime availability for terminal execution.
    #[must_use]
    pub fn with_shell_runtime_availability(
        mut self,
        availability: crate::shell_session::ShellRuntimeAvailability,
    ) -> Self {
        self.shell_sessions = self.shell_sessions.with_availability(availability);
        self
    }

    /// Adds container runtime metadata for container-backed terminal execution.
    #[must_use]
    pub fn with_container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.shell_sessions = self.shell_sessions.with_container_runtime(runtime);
        self
    }

    /// Records every sandboxed network policy decision — allowed or denied —
    /// into the given rolling audit log.
    #[must_use]
    pub fn with_network_audit_log(mut self, log: NetworkAuditLog) -> Self {
        self.network_audit = Some(log);
        self
    }

    /// Creates a new terminal tool with permission handler and executor.
    ///
    /// Creates a random four-word working directory under the specified parent directory.
    /// The executor is used to spawn async tasks for the IPC server.
    ///
    /// # Errors
    /// Returns an error when the working directory or output store cannot be created.
    pub async fn new_in(
        parent_dir: impl AsRef<std::path::Path>,
        permission_handler: P,
        executor: E,
    ) -> Result<Self, TerminalError> {
        let parent_dir = parent_dir.as_ref();

        // Ensure parent directory exists
        async_fs::create_dir_all(parent_dir).await?;

        // Create random four-word working directory
        let working_dir = WorkingDir::random_in(parent_dir).map_err(|e| {
            TerminalError::SandboxSetup(format!("failed to create working dir: {e}"))
        })?;
        let working_dir_path = working_dir.path().to_path_buf();

        info!(working_dir = %working_dir_path.display(), "created terminal tool working directory");

        // Create outputs directory inside working dir
        let outputs_dir = working_dir_path.join("outputs");
        async_fs::create_dir_all(&outputs_dir).await?;

        // Create output store
        let output_store = Arc::new(OutputStore::new(&outputs_dir).await?);

        // Start job registry service
        let (job_registry, job_registry_service) = job_registry_channel();
        executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();

        // Create channel for background task completion (unbounded to not block spawned tasks)
        let (completed_tx, completed_rx) = async_channel::unbounded();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (permission_tx, permission_rx) = async_channel::unbounded();

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(ShellRuntimeAvailability::default()),
            permission_handler: Arc::new(permission_handler),
            executor,
            output_store,
            job_registry,
            completed_rx,
            completed_tx,
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            network_audit: None,
            registry: Unconfigured,
        })
    }

    /// Creates a new terminal tool with permission handler and executor,
    /// using the provided directory directly as the working directory.
    ///
    /// Unlike `new_in` which creates a random subdirectory, this method
    /// uses the exact path provided. Use this when you want explicit control
    /// over the working directory location.
    ///
    /// # Errors
    /// Returns an error when the working directory or output store cannot be created.
    pub async fn new_exact(
        working_dir: impl AsRef<std::path::Path>,
        permission_handler: P,
        executor: E,
    ) -> Result<Self, TerminalError> {
        let working_dir_path = working_dir.as_ref().to_path_buf();

        // Ensure working directory exists
        async_fs::create_dir_all(&working_dir_path).await?;

        // Create outputs directory inside working dir
        let outputs_dir = working_dir_path.join("outputs");
        async_fs::create_dir_all(&outputs_dir).await?;

        // Create output store
        let output_store = Arc::new(OutputStore::new(&outputs_dir).await?);

        // Start job registry service
        let (job_registry, job_registry_service) = job_registry_channel();
        executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();

        // Create channel for background task completion (unbounded to not block spawned tasks)
        let (completed_tx, completed_rx) = async_channel::unbounded();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (permission_tx, permission_rx) = async_channel::unbounded();

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(ShellRuntimeAvailability::default()),
            permission_handler: Arc::new(permission_handler),
            executor,
            output_store,
            job_registry,
            completed_rx,
            completed_tx,
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            network_audit: None,
            registry: Unconfigured,
        })
    }

    /// Attaches a tool registry to this terminal tool, enabling IPC command dispatch.
    #[must_use]
    pub fn with_registry(self, registry: Arc<ToolRegistry>) -> TerminalTool<P, E, Configured> {
        TerminalTool {
            working_dir: self.working_dir,
            shell_sessions: self.shell_sessions,
            permission_handler: self.permission_handler,
            executor: self.executor,
            output_store: self.output_store,
            job_registry: self.job_registry,
            completed_rx: self.completed_rx,
            completed_tx: self.completed_tx,
            running_tasks: self.running_tasks,
            permission_rx: self.permission_rx,
            permission_tx: self.permission_tx,
            writable_paths: self.writable_paths,
            readable_paths: self.readable_paths,
            network_audit: self.network_audit,
            registry: Configured { registry },
        }
    }
}

impl<P, E, State> TerminalTool<P, E, State>
where
    E: Executor + Clone + 'static,
    State: Clone,
{
    /// Adds additional writable paths to the sandbox configuration.
    ///
    /// These paths will be writable in sandboxed and network modes.
    #[must_use]
    pub fn with_writable_paths(
        mut self,
        paths: impl IntoIterator<Item = impl Into<PathBuf>>,
    ) -> Self {
        self.writable_paths
            .extend(paths.into_iter().map(Into::into));
        self
    }

    /// Adds additional readable (but not writable) paths to the sandbox configuration.
    ///
    /// These paths will be readable in all sandbox modes, even in strict
    /// filesystem mode where reads outside the sandbox are normally denied.
    #[must_use]
    pub fn with_readable_paths(
        mut self,
        paths: impl IntoIterator<Item = impl Into<PathBuf>>,
    ) -> Self {
        self.readable_paths
            .extend(paths.into_iter().map(Into::into));
        self
    }

    /// Creates a child `TerminalTool` that shares the same sandbox and permission handler
    /// but has independent background task tracking.
    ///
    /// Use this to create terminal tools for subagents that:
    /// - Share the same working directory and output store
    /// - Share the same permission handler (security policies enforced consistently)
    /// - Have independent completion channels (no message mixup)
    #[must_use]
    pub fn child(&self) -> Self {
        let (completed_tx, completed_rx) = async_channel::unbounded();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (permission_tx, permission_rx) = async_channel::unbounded();
        let (job_registry, job_registry_service) = job_registry_channel();
        self.executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();
        Self {
            working_dir: self.working_dir.clone(),
            shell_sessions: self.shell_sessions.clone(),
            permission_handler: self.permission_handler.clone(), // Arc clone - shares handler
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            job_registry,
            completed_rx,
            completed_tx,
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
            network_audit: self.network_audit.clone(),
            registry: self.registry.clone(),
        }
    }

    /// Returns the working directory path.
    pub const fn working_dir(&self) -> &PathBuf {
        &self.working_dir
    }

    /// Returns the outputs directory path.
    pub fn outputs_dir(&self) -> PathBuf {
        self.working_dir.join("outputs")
    }

    /// Returns the output store.
    pub const fn output_store(&self) -> &Arc<OutputStore> {
        &self.output_store
    }

    /// Returns the job registry handle.
    pub fn job_registry(&self) -> JobRegistry {
        self.job_registry.clone()
    }

    /// Returns a receiver for completed background tasks.
    ///
    /// The returned `BackgroundTaskReceiver` can be used independently of the `TerminalTool`
    /// to poll for completed background tasks. This is useful for integrating with
    /// the Agent's main loop.
    pub fn background_receiver(&self) -> BackgroundTaskReceiver {
        BackgroundTaskReceiver {
            rx: self.completed_rx.clone(),
            running_tasks: self.running_tasks.clone(),
        }
    }

    /// Returns a receiver for permission lifecycle events.
    pub fn permission_receiver(&self) -> PermissionEventReceiver {
        PermissionEventReceiver {
            rx: self.permission_rx.clone(),
        }
    }

    /// Takes all completed background tasks without blocking.
    ///
    /// Returns an empty vector if no tasks have completed.
    /// The agent should call this after each turn to check for completed background tasks.
    pub fn take_completed(&self) -> Vec<CompletedTask> {
        let mut completed = Vec::new();
        while let Ok(task) = self.completed_rx.try_recv() {
            completed.push(task);
        }
        completed
    }

    /// Checks if there are any pending background tasks.
    pub fn has_pending_tasks(&self) -> bool {
        !self.completed_rx.is_empty() || self.running_tasks.load(Ordering::Acquire) > 0
    }
}

impl<P, E> TerminalTool<P, E, Configured>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    const fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry.registry
    }

    /// Starts a background service that produces child terminal tools on demand.
    pub fn start_factory_service(&self, receiver: TerminalToolFactoryReceiver) {
        let tool = self.clone();
        self.executor
            .spawn(async move { receiver.serve(tool).await })
            .detach();
    }

    /// Converts this `TerminalTool` into a type-erased `DynTerminalTool`.
    ///
    /// This is useful for creating child terminal tools for subagents where
    /// the concrete type cannot be known at compile time.
    pub fn to_dyn(self) -> crate::command::DynTerminalTool {
        use crate::builtin::{InputTerminalTool, KillTerminalTool, ReadTerminalDeltaTool};
        use crate::command::DynTerminalTool;

        let background_receiver = self.background_receiver();
        let permission_receiver = self.permission_receiver();
        let job_registry = self.job_registry();
        let working_dir = self.working_dir().clone();
        let entries = vec![
            erase_terminal_tool(self),
            erase_terminal_tool(KillTerminalTool::new(job_registry.clone())),
            erase_terminal_tool(InputTerminalTool::new(job_registry.clone())),
            erase_terminal_tool(ReadTerminalDeltaTool::new(job_registry.clone())),
        ];

        DynTerminalTool {
            entries,
            background_receiver,
            permission_receiver,
            job_registry,
            working_dir,
        }
    }

    async fn resolve_execution_target(
        &self,
        arguments: &TerminalArgs,
    ) -> anyhow::Result<ExecutionTarget> {
        match arguments.mode {
            TerminalExecutionMode::Default => self.resolve_default_target().await,
            TerminalExecutionMode::Sandboxed => self.resolve_sandboxed_target(),
            TerminalExecutionMode::Unsafe => self.resolve_unsafe_target(),
            TerminalExecutionMode::Ssh => {
                self.resolve_ssh_target(arguments.ssh_server_id.as_deref())
                    .await
            }
        }
    }

    async fn resolve_default_target(&self) -> anyhow::Result<ExecutionTarget> {
        match self.shell_sessions.resolve_local_backend() {
            Ok(backend) => Ok(ExecutionTarget {
                backend,
                mode: TerminalMode::Sandboxed,
                ssh_target: None,
                ssh_runtime: None,
                container_runtime: self.container_runtime_for_backend(backend)?,
            }),
            Err(local_error) => {
                self.shell_sessions
                    .ensure_ssh_available()
                    .map_err(anyhow::Error::msg)?;
                let server = self
                    .shell_sessions
                    .default_ssh_server()
                    .map_err(|ssh_error| anyhow::anyhow!("{local_error}; {ssh_error}"))?;
                let runtime =
                    bootstrap_ssh_runtime(&server.target, &self.shell_sessions.ssh_authorizer())
                        .await?;
                Ok(ExecutionTarget {
                    backend: ShellBackend::Ssh,
                    mode: TerminalMode::Sandboxed,
                    ssh_target: Some(server.target),
                    ssh_runtime: Some(runtime),
                    container_runtime: None,
                })
            }
        }
    }

    fn resolve_sandboxed_target(&self) -> anyhow::Result<ExecutionTarget> {
        let backend = self
            .shell_sessions
            .resolve_local_backend()
            .map_err(anyhow::Error::msg)?;
        Ok(ExecutionTarget {
            backend,
            mode: TerminalMode::Sandboxed,
            ssh_target: None,
            ssh_runtime: None,
            container_runtime: self.container_runtime_for_backend(backend)?,
        })
    }

    fn resolve_unsafe_target(&self) -> anyhow::Result<ExecutionTarget> {
        let backend = self
            .shell_sessions
            .resolve_local_backend()
            .map_err(anyhow::Error::msg)?;
        if !matches!(backend, ShellBackend::Local) {
            return Err(anyhow::anyhow!(
                "unsafe mode is only available in heel local runtime"
            ));
        }
        Ok(ExecutionTarget {
            backend,
            mode: TerminalMode::Unsafe,
            ssh_target: None,
            ssh_runtime: None,
            container_runtime: None,
        })
    }

    async fn resolve_ssh_target(
        &self,
        ssh_server_id: Option<&str>,
    ) -> anyhow::Result<ExecutionTarget> {
        self.shell_sessions
            .ensure_ssh_available()
            .map_err(anyhow::Error::msg)?;
        let server = match ssh_server_id {
            Some(server_id) => self
                .shell_sessions
                .resolve_ssh_server(server_id)
                .map_err(anyhow::Error::msg)?,
            None => self
                .shell_sessions
                .default_ssh_server()
                .map_err(anyhow::Error::msg)?,
        };
        let runtime =
            bootstrap_ssh_runtime(&server.target, &self.shell_sessions.ssh_authorizer()).await?;
        Ok(ExecutionTarget {
            backend: ShellBackend::Ssh,
            mode: TerminalMode::Sandboxed,
            ssh_target: Some(server.target),
            ssh_runtime: Some(runtime),
            container_runtime: None,
        })
    }

    fn container_runtime_for_backend(
        &self,
        backend: ShellBackend,
    ) -> anyhow::Result<Option<ContainerShellRuntime>> {
        if matches!(backend, ShellBackend::Container) {
            self.shell_sessions
                .container_runtime()
                .cloned()
                .map(Some)
                .ok_or_else(|| anyhow::anyhow!("missing container runtime for container backend"))
        } else {
            Ok(None)
        }
    }

    async fn ensure_permission_allowed(
        &self,
        mode: TerminalMode,
        script: &str,
    ) -> anyhow::Result<()> {
        let permission_waits = self
            .permission_handler
            .will_wait_for_approval(mode, script)
            .await;
        if permission_waits {
            self.emit_permission_event(mode, script, PermissionEventStage::Waiting)
                .await;
        }
        let permission_result =
            ensure_mode_allowed(self.permission_handler.as_ref(), mode, script).await;
        if permission_waits {
            self.emit_permission_event(mode, script, PermissionEventStage::Resolved)
                .await;
        }
        permission_result.map_err(anyhow::Error::new)
    }

    async fn emit_permission_event(
        &self,
        mode: TerminalMode,
        script: &str,
        stage: PermissionEventStage,
    ) {
        let _ = self
            .permission_tx
            .send(PermissionEvent {
                mode,
                script: script.to_string(),
                stage,
            })
            .await;
    }

    fn spawn_terminal_execution(&self, request: TerminalSpawnRequest) -> SpawnedTerminal {
        let TerminalSpawnRequest {
            target,
            task_id,
            execution_id,
            script,
            expect,
            resolution,
            max_lines,
            raw,
            timeout,
        } = request;
        let ExecutionTarget {
            backend,
            mode,
            ssh_target,
            ssh_runtime,
            container_runtime,
        } = target;
        let (result_tx, result_rx) = async_channel::bounded(1);
        let (startup_tx, startup_rx) = async_channel::bounded(1);
        let (stdin_blocked_tx, stdin_blocked_rx) = async_channel::bounded(1);
        let background_mode = Arc::new(AtomicBool::new(timeout == 0));
        let stdin_blocked_notice = if supports_stdin_blocked_notice(backend) {
            Some(stdin_blocked_tx)
        } else {
            None
        };

        let background_startup = BackgroundStartup::new(startup_tx);
        let background_mode_for_spawn = background_mode.clone();
        let background_mode_for_completion = background_mode.clone();
        let running_tasks = self.running_tasks.clone();
        let completed_tx = self.completed_tx.clone();

        running_tasks.fetch_add(1, Ordering::AcqRel);
        self.executor
            .spawn(execute_terminal_task(SpawnedTerminalTask {
                working_dir: self.working_dir.clone(),
                writable_paths: self.writable_paths.clone(),
                readable_paths: self.readable_paths.clone(),
                executor: self.executor.clone(),
                registry: self.registry().clone(),
                permission_handler: self.permission_handler.clone(),
                job_registry: self.job_registry.clone(),
                task_id,
                execution_id,
                background_mode: background_mode_for_spawn,
                script,
                mode,
                backend,
                ssh_target,
                ssh_runtime,
                container_runtime,
                stdin_blocked_notice,
                background_startup,
                expect,
                resolution,
                store_dir: self.output_store.dir().to_path_buf(),
                max_lines,
                raw,
                result_tx,
                completed_tx,
                background_mode_for_completion,
                running_tasks,
                network_audit: self.network_audit.clone(),
            }))
            .detach();

        SpawnedTerminal {
            result_rx,
            startup_rx,
            stdin_blocked_rx,
            background_mode,
        }
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> Tool
    for TerminalTool<P, E, Configured>
{
    fn name(&self) -> Cow<'static, str> {
        "terminal".into()
    }

    type Arguments = TerminalArgs;
    type Res = ToolResult;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<Self::Res> {
        if arguments.description.trim().is_empty() {
            return Err(anyhow::anyhow!("terminal description must not be empty"));
        }
        let task_id = random_task_id();
        let script = arguments.script.clone();
        let execution_id = format!("exec-{task_id}");
        let expect = arguments.expect;
        let resolution = arguments.resolution;
        let max_lines = arguments.max_lines.min(MAX_LINES_CEILING) as usize;
        let raw = arguments.raw;
        let timeout = arguments.timeout;

        let ExecutionTarget {
            backend,
            mode,
            ssh_target,
            ssh_runtime,
            container_runtime,
        } = self.resolve_execution_target(&arguments).await?;
        self.ensure_permission_allowed(mode, &script).await?;
        let store_dir = self.output_store.dir().to_path_buf();
        let spawned = self.spawn_terminal_execution(TerminalSpawnRequest {
            target: ExecutionTarget {
                backend,
                mode,
                ssh_target,
                ssh_runtime,
                container_runtime,
            },
            task_id: task_id.clone(),
            execution_id,
            script,
            expect,
            resolution,
            max_lines,
            raw,
            timeout,
        });

        if timeout == 0 {
            let reason = BackgroundReason::Explicit;
            wait_for_background_startup(&spawned.startup_rx).await?;
            return background_terminal_result(
                &self.job_registry,
                &store_dir,
                &task_id,
                max_lines,
                reason,
            )
            .await;
        }

        let configured_timeout_seconds = timeout;
        let immediate = wait_for_foreground_decision(
            &spawned.result_rx,
            &spawned.stdin_blocked_rx,
            timeout,
            configured_timeout_seconds,
        )
        .await;

        let (reason, pre_registered) = match immediate {
            Some(ForegroundDecision::Completed(result)) => match *result {
                Ok(mut result) => {
                    result.task_id = None;
                    result.status = None;
                    result.background_reason = None;
                    return ToolResult::json(&result);
                }
                Err(err) => return Err(anyhow::anyhow!(err)),
            },
            Some(ForegroundDecision::PromoteToBackground(reason)) => (reason, false),
            None => (BackgroundReason::Explicit, false),
        };

        tracing::info!(
            target: "may::background_promotion",
            reason = ?reason,
            configured_timeout_seconds,
            "promoting terminal command to background"
        );

        spawned.background_mode.store(true, Ordering::Release);
        if !pre_registered {
            wait_for_background_startup(&spawned.startup_rx).await?;
        }
        background_terminal_result(&self.job_registry, &store_dir, &task_id, max_lines, reason)
            .await
    }
}

async fn wait_for_background_startup(
    startup_rx: &Receiver<Result<(), String>>,
) -> anyhow::Result<()> {
    match startup_rx.recv().await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(anyhow::anyhow!(err)),
        Err(_) => Err(anyhow::anyhow!(
            "background startup channel dropped before registration"
        )),
    }
}

async fn wait_for_foreground_decision(
    result_rx: &Receiver<Result<TerminalResult, String>>,
    stdin_blocked_rx: &Receiver<String>,
    timeout: u64,
    configured_timeout_seconds: u64,
) -> Option<ForegroundDecision> {
    let timeout = std::time::Duration::from_secs(timeout);
    futures_lite::future::or(
        async {
            result_rx
                .recv()
                .await
                .ok()
                .map(Box::new)
                .map(ForegroundDecision::Completed)
        },
        async {
            futures_lite::future::or(
                async {
                    match stdin_blocked_rx.recv().await {
                        Ok(notice) => Some(ForegroundDecision::PromoteToBackground(
                            BackgroundReason::StdinBlocked { notice },
                        )),
                        Err(_) => std::future::pending().await,
                    }
                },
                async {
                    async_io::Timer::after(timeout).await;
                    if let Ok(result) = result_rx.try_recv() {
                        return Some(ForegroundDecision::Completed(Box::new(result)));
                    }
                    Some(ForegroundDecision::PromoteToBackground(
                        BackgroundReason::Timeout {
                            configured_seconds: configured_timeout_seconds,
                        },
                    ))
                },
            )
            .await
        },
    )
    .await
}

async fn background_terminal_result(
    job_registry: &JobRegistry,
    store_dir: &Path,
    task_id: &str,
    max_lines: usize,
    reason: BackgroundReason,
) -> aither_core::Result<ToolResult> {
    let stdout = start_background_output_redirect(
        job_registry,
        store_dir,
        task_id,
        max_lines,
        Some(&reason),
    )
    .await?;
    ToolResult::json(&TerminalResult {
        stdout,
        stderr: None,
        exit_code: 0,
        task_id: Some(task_id.to_string()),
        status: Some("running".to_string()),
        background_reason: Some(reason),
    })
}

async fn start_background_output_redirect(
    job_registry: &JobRegistry,
    store_dir: &Path,
    task_id: &str,
    max_lines: usize,
    promotion_reason: Option<&BackgroundReason>,
) -> Result<OutputEntry, anyhow::Error> {
    let url = save_raw_to_file(store_dir, &[]).await?;
    let output_path = store_dir.join(url.strip_prefix("outputs/").unwrap_or(&url));
    let output_path_string = output_path.display().to_string();
    let snapshot = job_registry
        .start_output_redirect(task_id, output_path)
        .await
        .map_err(anyhow::Error::msg)?;
    let (preview, truncated) = preview_first_lines(&snapshot, max_lines);
    let text = match (
        promotion_reason.map(background_reason_preview),
        preview.is_empty(),
    ) {
        (Some(reason), true) => reason,
        (Some(reason), false) => format!("{reason}\n{preview}"),
        (None, true) => "(no output yet)".to_string(),
        (None, false) => preview,
    };
    Ok(OutputEntry::Stored {
        path: output_path_string,
        url,
        content: Some(Content::Text { text, truncated }),
    })
}

fn background_reason_preview(reason: &BackgroundReason) -> String {
    match reason {
        BackgroundReason::Timeout { configured_seconds } => {
            format!("Foreground timeout of {configured_seconds}s elapsed — promoted to background.")
        }
        BackgroundReason::StdinBlocked { notice } => notice.clone(),
        BackgroundReason::Explicit => "Started in background (timeout=0 requested).".to_string(),
    }
}

struct StandaloneExecution<P, E> {
    working_dir: PathBuf,
    writable_paths: Vec<PathBuf>,
    readable_paths: Vec<PathBuf>,
    executor: E,
    registry: Arc<ToolRegistry>,
    permission_handler: Arc<P>,
    job_registry: JobRegistry,
    task_id: String,
    execution_id: String,
    background_mode: Arc<AtomicBool>,
    script: String,
    mode: TerminalMode,
    backend: ShellBackend,
    ssh_target: Option<String>,
    ssh_runtime: Option<SshRuntimeProfile>,
    container_runtime: Option<ContainerShellRuntime>,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: BackgroundStartup,
    network_audit: Option<NetworkAuditLog>,
    expect: OutputFormat,
    resolution: MediaResolution,
    store_dir: PathBuf,
    max_lines: usize,
    raw: bool,
}

struct StandaloneStart<'a, P, E> {
    working_dir: &'a Path,
    writable_paths: &'a [PathBuf],
    readable_paths: &'a [PathBuf],
    executor: E,
    registry: &'a Arc<ToolRegistry>,
    permission_handler: Arc<P>,
    job_registry: &'a JobRegistry,
    task_id: &'a str,
    execution_id: &'a str,
    script: &'a str,
    mode: TerminalMode,
    backend: ShellBackend,
    ssh_target: Option<&'a str>,
    ssh_runtime: Option<SshRuntimeProfile>,
    container_runtime: Option<&'a ContainerShellRuntime>,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: &'a BackgroundStartup,
    network_audit: Option<NetworkAuditLog>,
}

struct LocalBackgroundExecution<'a, E> {
    working_dir: &'a Path,
    writable_paths: &'a [PathBuf],
    readable_paths: &'a [PathBuf],
    executor: E,
    registry: &'a Arc<ToolRegistry>,
    task_id: &'a str,
    execution_id: &'a str,
    script: &'a str,
    mode: TerminalMode,
    job_registry: &'a JobRegistry,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: &'a BackgroundStartup,
}

struct ContainerBackgroundExecution<'a, E> {
    executor: E,
    registry: &'a Arc<ToolRegistry>,
    task_id: &'a str,
    execution_id: &'a str,
    script: &'a str,
    mode: TerminalMode,
    container_runtime: Option<&'a ContainerShellRuntime>,
    ipc_commands: &'a [String],
    job_registry: &'a JobRegistry,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: &'a BackgroundStartup,
}

struct SshBackgroundExecution<'a> {
    task_id: &'a str,
    execution_id: &'a str,
    script: &'a str,
    mode: TerminalMode,
    ssh_target: Option<&'a str>,
    ssh_runtime: Option<SshRuntimeProfile>,
    job_registry: &'a JobRegistry,
    background_startup: &'a BackgroundStartup,
}

struct CompletedProcessOutput<'a> {
    job_registry: &'a JobRegistry,
    pid: u32,
    output: std::process::Output,
    background_output: bool,
    store_dir: &'a Path,
    expect: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    raw: bool,
}

struct ExecutionTarget {
    backend: ShellBackend,
    mode: TerminalMode,
    ssh_target: Option<String>,
    ssh_runtime: Option<SshRuntimeProfile>,
    container_runtime: Option<ContainerShellRuntime>,
}

struct TerminalSpawnRequest {
    target: ExecutionTarget,
    task_id: String,
    execution_id: String,
    script: String,
    expect: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    raw: bool,
    timeout: u64,
}

struct SpawnedTerminal {
    result_rx: Receiver<Result<TerminalResult, String>>,
    startup_rx: Receiver<Result<(), String>>,
    stdin_blocked_rx: Receiver<String>,
    background_mode: Arc<AtomicBool>,
}

struct SpawnedTerminalTask<P, E> {
    working_dir: PathBuf,
    writable_paths: Vec<PathBuf>,
    readable_paths: Vec<PathBuf>,
    executor: E,
    registry: Arc<ToolRegistry>,
    permission_handler: Arc<P>,
    job_registry: JobRegistry,
    task_id: String,
    execution_id: String,
    background_mode: Arc<AtomicBool>,
    script: String,
    mode: TerminalMode,
    backend: ShellBackend,
    ssh_target: Option<String>,
    ssh_runtime: Option<SshRuntimeProfile>,
    container_runtime: Option<ContainerShellRuntime>,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: BackgroundStartup,
    expect: OutputFormat,
    resolution: MediaResolution,
    store_dir: PathBuf,
    max_lines: usize,
    raw: bool,
    result_tx: Sender<Result<TerminalResult, String>>,
    completed_tx: Sender<CompletedTask>,
    background_mode_for_completion: Arc<AtomicBool>,
    running_tasks: Arc<AtomicUsize>,
    network_audit: Option<NetworkAuditLog>,
}

async fn execute_terminal_task<P, E>(task: SpawnedTerminalTask<P, E>)
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    let SpawnedTerminalTask {
        working_dir,
        writable_paths,
        readable_paths,
        executor,
        registry,
        permission_handler,
        job_registry,
        task_id,
        execution_id,
        background_mode,
        script,
        mode,
        backend,
        ssh_target,
        ssh_runtime,
        container_runtime,
        stdin_blocked_notice,
        background_startup,
        expect,
        resolution,
        store_dir,
        max_lines,
        raw,
        result_tx,
        completed_tx,
        background_mode_for_completion,
        running_tasks,
        network_audit,
    } = task;
    let result = execute_script_standalone(StandaloneExecution {
        working_dir,
        writable_paths,
        readable_paths,
        executor,
        registry,
        permission_handler,
        job_registry,
        task_id: task_id.clone(),
        execution_id,
        background_mode,
        script: script.clone(),
        mode,
        backend,
        ssh_target,
        ssh_runtime,
        container_runtime,
        stdin_blocked_notice,
        background_startup,
        network_audit,
        expect,
        resolution,
        store_dir,
        max_lines,
        raw,
    })
    .await;

    let quick_result = match &result {
        Ok(ok) => Ok(ok.clone()),
        Err(err) => Err(err.to_string()),
    };
    let _ = result_tx.send(quick_result).await;
    if background_mode_for_completion.load(Ordering::Acquire) {
        let _ = completed_tx
            .send(CompletedTask {
                task_id,
                script,
                result,
            })
            .await;
    }
    running_tasks.fetch_sub(1, Ordering::AcqRel);
}

/// Standalone script execution that can be spawned in a background task.
async fn execute_script_standalone<P, E>(
    request: StandaloneExecution<P, E>,
) -> Result<TerminalResult, TerminalError>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    let StandaloneExecution {
        working_dir,
        writable_paths,
        readable_paths,
        executor,
        registry,
        permission_handler,
        job_registry,
        task_id,
        execution_id,
        background_mode,
        script,
        mode,
        backend,
        ssh_target,
        ssh_runtime,
        container_runtime,
        stdin_blocked_notice,
        background_startup,
        network_audit,
        expect,
        resolution,
        store_dir,
        max_lines,
        raw,
    } = request;

    info!(
        script_len = script.len(),
        ?mode,
        "executing background terminal command"
    );
    debug!(script = %script, "script content");
    let start_result = start_standalone_execution(StandaloneStart {
        working_dir: &working_dir,
        writable_paths: &writable_paths,
        readable_paths: &readable_paths,
        executor,
        registry: &registry,
        permission_handler,
        job_registry: &job_registry,
        task_id: &task_id,
        execution_id: &execution_id,
        script: &script,
        mode,
        backend,
        ssh_target: ssh_target.as_deref(),
        ssh_runtime,
        container_runtime: container_runtime.as_ref(),
        stdin_blocked_notice,
        background_startup: &background_startup,
        network_audit,
    })
    .await;
    let (pid, output) = match start_result {
        Ok(started) => started,
        Err(error) => {
            background_startup.report_failure(&error).await;
            return Err(error);
        }
    };

    let result = save_completed_process_output(CompletedProcessOutput {
        job_registry: &job_registry,
        pid,
        output,
        background_output: background_mode.load(Ordering::Acquire),
        store_dir: &store_dir,
        expect,
        resolution,
        max_lines,
        raw,
    })
    .await?;
    let exit_code = result.exit_code;

    #[cfg(unix)]
    {
        debug!(exit_code, "background script completed");
    }
    #[cfg(not(unix))]
    {
        debug!(exit_code, "background script completed");
    }

    let output_path = result.stdout.stored_path(&store_dir);
    job_registry.complete(pid, exit_code, output_path).await;

    Ok(result)
}

async fn start_standalone_execution<P, E>(
    request: StandaloneStart<'_, P, E>,
) -> Result<(u32, std::process::Output), TerminalError>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    if matches!(request.backend, ShellBackend::Container) {
        let ipc_commands = request.registry.registered_tool_names();
        return execute_container_background(ContainerBackgroundExecution {
            executor: request.executor,
            registry: request.registry,
            task_id: request.task_id,
            execution_id: request.execution_id,
            script: request.script,
            mode: request.mode,
            container_runtime: request.container_runtime,
            ipc_commands: &ipc_commands,
            job_registry: request.job_registry,
            stdin_blocked_notice: request.stdin_blocked_notice,
            background_startup: request.background_startup,
        })
        .await;
    }
    if matches!(request.backend, ShellBackend::Ssh) {
        return execute_ssh_background(SshBackgroundExecution {
            task_id: request.task_id,
            execution_id: request.execution_id,
            script: request.script,
            mode: request.mode,
            ssh_target: request.ssh_target,
            ssh_runtime: request.ssh_runtime,
            job_registry: request.job_registry,
            background_startup: request.background_startup,
        })
        .await;
    }
    let local = LocalBackgroundExecution {
        working_dir: request.working_dir,
        writable_paths: request.writable_paths,
        readable_paths: request.readable_paths,
        executor: request.executor,
        registry: request.registry,
        task_id: request.task_id,
        execution_id: request.execution_id,
        script: request.script,
        mode: request.mode,
        job_registry: request.job_registry,
        stdin_blocked_notice: request.stdin_blocked_notice,
        background_startup: request.background_startup,
    };
    match request.mode {
        TerminalMode::Sandboxed => {
            execute_audited_sandboxed_background(
                local,
                PermissionNetworkPolicy {
                    permission_handler: request.permission_handler,
                },
                request.network_audit,
            )
            .await
        }
        TerminalMode::Unsafe => execute_unsafe_background(local).await,
    }
}

/// Runs a sandboxed execution, recording policy verdicts when an audit log is
/// configured. Unsafe mode bypasses the proxy entirely, so it cannot be
/// audited at this layer.
async fn execute_audited_sandboxed_background<E, N>(
    local: LocalBackgroundExecution<'_, E>,
    policy: N,
    audit: Option<NetworkAuditLog>,
) -> Result<(u32, std::process::Output), TerminalError>
where
    E: Executor + Clone + 'static,
    N: NetworkPolicy + 'static,
{
    match audit {
        Some(log) => execute_sandboxed_background(local, Audited::new(policy, log)).await,
        None => execute_sandboxed_background(local, policy).await,
    }
}

async fn save_completed_process_output(
    request: CompletedProcessOutput<'_>,
) -> Result<TerminalResult, TerminalError> {
    let CompletedProcessOutput {
        job_registry,
        pid,
        output,
        background_output,
        store_dir,
        expect,
        resolution,
        max_lines,
        raw,
    } = request;
    let stdout = save_completed_stdout(
        job_registry,
        pid,
        &output.stdout,
        CompletedStdoutFormat {
            background_output,
            store_dir,
            expect,
            resolution,
            max_lines,
            raw,
        },
    )
    .await?;
    let stderr = save_completed_stderr(job_registry, pid, store_dir, &output.stderr).await?;
    let exit_code = output.status.code().unwrap_or(-1);

    Ok(TerminalResult {
        stdout,
        stderr,
        exit_code,
        task_id: None,
        status: None,
        background_reason: None,
    })
}

struct CompletedStdoutFormat<'a> {
    background_output: bool,
    store_dir: &'a Path,
    expect: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    raw: bool,
}

async fn save_completed_stdout(
    job_registry: &JobRegistry,
    pid: u32,
    stdout: &[u8],
    format: CompletedStdoutFormat<'_>,
) -> Result<OutputEntry, TerminalError> {
    let byte_limit = Some(INLINE_OUTPUT_LIMIT);
    let data_to_save = if format.background_output {
        Cow::Borrowed(stdout)
    } else {
        compressed_stdout_data(format.store_dir, stdout, format.expect, format.raw).await
    };

    match crate::output::save_text_with_line_limit(
        format.store_dir,
        data_to_save.as_ref(),
        format.expect,
        format.resolution,
        format.max_lines,
        byte_limit,
    )
    .await
    {
        Ok(entry) => Ok(entry),
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            Err(TerminalError::Io(err))
        }
    }
}

async fn compressed_stdout_data<'a>(
    store_dir: &Path,
    stdout: &'a [u8],
    expect: OutputFormat,
    raw: bool,
) -> Cow<'a, [u8]> {
    let is_text = matches!(expect, OutputFormat::Text | OutputFormat::Auto);
    let compressed = if !raw && is_text && !stdout.is_empty() {
        std::str::from_utf8(stdout).map_or(None, crate::output_compress::compress_text)
    } else {
        None
    };

    if let Some(ref compressed) = compressed
        && let Some(ref raw_text) = compressed.raw_for_file
        && let Err(err) = crate::output::save_raw_to_file(store_dir, raw_text.as_bytes()).await
    {
        warn!(error = %err, "failed to save raw source code output");
    }

    compressed.map_or(Cow::Borrowed(stdout), |compressed| {
        Cow::Owned(compressed.text.into_bytes())
    })
}

async fn save_completed_stderr(
    job_registry: &JobRegistry,
    pid: u32,
    store_dir: &Path,
    stderr: &[u8],
) -> Result<Option<OutputEntry>, TerminalError> {
    if stderr.is_empty() {
        return Ok(None);
    }
    match OutputStore::save_to_dir_with_limit(
        store_dir,
        stderr,
        OutputFormat::Text,
        MediaResolution::Auto,
        Some(INLINE_OUTPUT_LIMIT),
    )
    .await
    {
        Ok(entry) => Ok(Some(entry)),
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            Err(TerminalError::Io(err))
        }
    }
}

async fn execute_sandboxed_background<E, N>(
    context: LocalBackgroundExecution<'_, E>,
    policy: N,
) -> Result<(u32, std::process::Output), TerminalError>
where
    E: Executor + Clone + 'static,
    N: NetworkPolicy + 'static,
{
    let runtime_environment = prepare_session_runtime_environment(context.working_dir).await?;
    let mut config_builder = SandboxConfig::builder()
        .network(policy)
        .working_dir(context.working_dir)
        .writable_paths(context.writable_paths)
        .readable_paths(context.readable_paths)
        .security(SecurityConfig::interactive());
    if let Some(router) = create_ipc_router(context.registry) {
        config_builder = config_builder.ipc(router);
    }
    let config = config_builder.build();

    let sandbox = Sandbox::with_config_and_executor(config, context.executor.clone())
        .await
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let launch = build_terminal_launch(context.script)?;
    let (program, args) = match &launch {
        TerminalLaunch::Direct { program, args } | TerminalLaunch::Shell { program, args } => {
            (program, args)
        }
    };

    let child = sandbox
        .command(program)
        .args(args)
        .envs(runtime_environment)
        .stdin(StdioConfig::Piped)
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped)
        .spawn()
        .await
        .map_err(|e| TerminalError::Execution(e.to_string()))?;

    let pid = child.id();
    context
        .job_registry
        .register(
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.mode,
            None,
        )
        .await;
    context.background_startup.report_ready().await;

    let output = collect_local_process_output(
        child,
        context.executor,
        context.job_registry,
        pid,
        context.stdin_blocked_notice,
        "missing stdout pipe for sandbox process",
        "missing stderr pipe for sandbox process",
    )
    .await?;

    Ok((pid, output))
}

async fn execute_unsafe_background<E: Executor + Clone + 'static>(
    context: LocalBackgroundExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let runtime_environment = prepare_session_runtime_environment(context.working_dir).await?;
    let mut config_builder = SandboxConfig::builder()
        .network(AllowAll)
        .working_dir(context.working_dir)
        .writable_paths(context.writable_paths)
        .readable_paths(context.readable_paths)
        .security(SecurityConfig::interactive());
    if let Some(router) = create_ipc_gateway_router(context.registry) {
        config_builder = config_builder.ipc(router);
    }
    let config = config_builder.build();

    let sandbox = Sandbox::with_config_and_executor(config, context.executor.clone())
        .await
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let launch = build_terminal_launch(context.script)?;
    let (program, args) = match &launch {
        TerminalLaunch::Direct { program, args } | TerminalLaunch::Shell { program, args } => {
            (program, args)
        }
    };

    let child = sandbox
        .command(program)
        .args(args)
        .envs(runtime_environment)
        .stdin(StdioConfig::Piped)
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped)
        .spawn()
        .await
        .map_err(|e| TerminalError::Execution(e.to_string()))?;

    let pid = child.id();
    context
        .job_registry
        .register(
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.mode,
            None,
        )
        .await;
    context.background_startup.report_ready().await;

    let output = collect_local_process_output(
        child,
        context.executor,
        context.job_registry,
        pid,
        context.stdin_blocked_notice,
        "missing stdout pipe for unsafe process",
        "missing stderr pipe for unsafe process",
    )
    .await?;

    Ok((pid, output))
}

async fn collect_local_process_output<E: Executor + Clone + 'static>(
    mut child: heel::Child,
    executor: E,
    job_registry: &JobRegistry,
    pid: u32,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
    missing_stdout_message: &'static str,
    missing_stderr_message: &'static str,
) -> Result<std::process::Output, TerminalError> {
    if let Some(stdin) = child.take_stdin() {
        let input_tx = spawn_terminal_stdin_writer(stdin);
        job_registry.attach_terminal_input(pid, input_tx).await;
    }

    let stdout = child
        .take_stdout()
        .ok_or_else(|| TerminalError::Execution(missing_stdout_message.into()))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| TerminalError::Execution(missing_stderr_message.into()))?;

    let (chunk_tx, chunk_rx) = async_channel::unbounded();
    spawn_terminal_reader(stdout, chunk_tx.clone(), false);
    spawn_terminal_reader(stderr, chunk_tx.clone(), true);
    drop(chunk_tx);
    executor
        .spawn(drain_terminal_chunks(job_registry.clone(), pid, chunk_rx))
        .detach();

    let status = match wait_for_local_process_exit(&mut child, pid, stdin_blocked_notice).await {
        Ok(status) => status,
        Err(error) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            return Err(error);
        }
    };
    wait_for_terminal_stream_close(job_registry, pid).await;
    let (stdout, stderr) = job_registry.terminal_output(pid).await.ok_or_else(|| {
        TerminalError::Execution(format!("missing terminal output for pid {pid}"))
    })?;
    let output = std::process::Output {
        status,
        stdout,
        stderr,
    };

    Ok(output)
}

async fn wait_for_local_process_exit(
    child: &mut heel::Child,
    pid: u32,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
) -> Result<std::process::ExitStatus, TerminalError> {
    if stdin_blocked_notice.is_none() {
        return child
            .wait()
            .await
            .map_err(|error| TerminalError::Execution(error.to_string()));
    }

    let mut notice_sent = false;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| TerminalError::Execution(error.to_string()))?
        {
            return Ok(status);
        }

        if !notice_sent {
            match detect_stdin_blocked_for_local_pid(pid).await {
                Ok(true) => {
                    if let Some(notice_tx) = stdin_blocked_notice.as_ref() {
                        let _ = notice_tx.try_send(TERMINAL_STDIN_BLOCKED_NOTICE.to_string());
                    }
                    notice_sent = true;
                }
                Ok(false) => {}
                Err(error) => {
                    tracing::debug!(error = %error, pid, "stdin watchdog probe failed");
                }
            }
        }

        async_io::Timer::after(STDIN_WATCH_INTERVAL).await;
    }
}

async fn execute_container_background<E: Executor + Clone + 'static>(
    context: ContainerBackgroundExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let container_runtime = context.container_runtime.ok_or_else(|| {
        TerminalError::Execution("missing container runtime for container backend".into())
    })?;
    let exec = container_runtime.exec();

    // Use lower 32 bits of a UUID as a synthetic PID for job tracking.
    let pid = u32::from_le_bytes(
        uuid::Uuid::new_v4().as_u128().to_le_bytes()[..4]
            .try_into()
            .expect("uuid byte slice has four bytes"),
    );
    let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
    let (input_tx, input_rx) = async_channel::unbounded::<Vec<u8>>();
    context
        .job_registry
        .register(
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.mode,
            None,
        )
        .await;
    context
        .job_registry
        .attach_terminal_input(pid, input_tx)
        .await;
    context.background_startup.report_ready().await;
    context.job_registry.attach_kill_switch(pid, kill_tx).await;

    let ipc_bridge = if context.ipc_commands.is_empty() {
        None
    } else {
        Some(start_container_ipc_bridge(
            context.executor,
            Arc::clone(context.registry),
        )?)
    };
    let wrapped_script = wrap_container_script(
        context.script,
        context.ipc_commands,
        ipc_bridge.as_ref().map(|bridge| ContainerIpcEndpoint {
            host: container_runtime.ipc_host(),
            port: bridge.port(),
        }),
    )?;

    let execution = exec
        .exec_boxed(
            container_runtime.container_id(),
            &wrapped_script,
            "/workspace",
            kill_rx,
            input_rx,
            context.stdin_blocked_notice,
        )
        .await;

    if let Some(bridge) = ipc_bridge {
        bridge.stop().await;
    }

    match execution {
        Ok(crate::shell_session::ContainerExecOutcome::Completed(output)) => {
            if !output.stdout.is_empty() {
                context
                    .job_registry
                    .append_stdout(pid, output.stdout.clone())
                    .await;
            }
            if !output.stderr.is_empty() {
                context
                    .job_registry
                    .append_stderr(pid, output.stderr.clone())
                    .await;
            }
            context.job_registry.close_stdout(pid).await;
            context.job_registry.close_stderr(pid).await;
            Ok((pid, output))
        }
        Ok(crate::shell_session::ContainerExecOutcome::Killed) => {
            context.job_registry.close_stdout(pid).await;
            context.job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution("container job killed".to_string()))
        }
        Err(err) => {
            context.job_registry.fail(pid, &err, None).await;
            context.job_registry.close_stdout(pid).await;
            context.job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution(err))
        }
    }
}

async fn execute_ssh_background(
    context: SshBackgroundExecution<'_>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let target = context
        .ssh_target
        .ok_or_else(|| TerminalError::Execution("missing ssh target".to_string()))?;
    let runtime = context
        .ssh_runtime
        .ok_or_else(|| TerminalError::Execution("missing ssh runtime profile".to_string()))?;

    let remote_cmd = match (runtime, context.mode) {
        (SshRuntimeProfile::Heel { binary }, TerminalMode::Sandboxed) => {
            let (program, args) = shell_launch(context.script);
            let escaped_args = args
                .iter()
                .map(|arg| shell_escape(arg))
                .collect::<Vec<_>>()
                .join(" ");
            if escaped_args.is_empty() {
                format!(
                    "{} run --network allow -- {}",
                    shell_escape(&binary),
                    shell_escape(&program),
                )
            } else {
                format!(
                    "{} run --network allow -- {} {}",
                    shell_escape(&binary),
                    shell_escape(&program),
                    escaped_args,
                )
            }
        }
        (SshRuntimeProfile::Heel { .. }, TerminalMode::Unsafe) => {
            return Err(TerminalError::Execution(
                "unsafe mode is not supported for ssh backend".to_string(),
            ));
        }
    };

    let mut command = std::process::Command::new("ssh");
    command
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg(remote_cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);

    let child = async_process::Command::from(command)
        .spawn()
        .map_err(|e| TerminalError::Execution(e.to_string()))?;

    let pid = child.id();
    context
        .job_registry
        .register(
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.mode,
            None,
        )
        .await;
    context.background_startup.report_ready().await;

    match child.output().await {
        Ok(output) => {
            if !output.stdout.is_empty() {
                context
                    .job_registry
                    .append_stdout(pid, output.stdout.clone())
                    .await;
            }
            if !output.stderr.is_empty() {
                context
                    .job_registry
                    .append_stderr(pid, output.stderr.clone())
                    .await;
            }
            context.job_registry.close_stdout(pid).await;
            context.job_registry.close_stderr(pid).await;
            Ok((pid, output))
        }
        Err(err) => {
            context.job_registry.fail(pid, &err.to_string(), None).await;
            context.job_registry.close_stdout(pid).await;
            context.job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution(err.to_string()))
        }
    }
}

fn shell_escape(value: &str) -> String {
    let escaped = value.replace('\'', "'\"'\"'");
    let mut output = String::with_capacity(escaped.len() + 2);
    output.push('\'');
    output.push_str(&escaped);
    output.push('\'');
    output
}

async fn prepare_session_runtime_environment(
    working_dir: &Path,
) -> Result<Vec<(String, String)>, TerminalError> {
    let session_tmp = working_dir.join("tmp");
    let session_cache = working_dir.join(".cache");
    let bun_cache = session_cache.join("bun");
    let session_config = working_dir.join(".config");
    let playwright_cache = session_cache.join("ms-playwright");
    let python_path = working_dir.join("skills").join("python");

    for directory in [
        &session_tmp,
        &session_cache,
        &bun_cache,
        &session_config,
        &playwright_cache,
    ] {
        async_fs::create_dir_all(directory).await?;
    }

    let tmp = path_environment_value("TMPDIR", &session_tmp)?;
    let cache = path_environment_value("XDG_CACHE_HOME", &session_cache)?;
    let bun = path_environment_value("BUN_INSTALL_CACHE_DIR", &bun_cache)?;
    let config = path_environment_value("XDG_CONFIG_HOME", &session_config)?;
    let playwright = path_environment_value("PLAYWRIGHT_BROWSERS_PATH", &playwright_cache)?;
    let home = path_environment_value("HOME", working_dir)?;
    let python = std::env::var_os("PYTHONPATH").map_or_else(
        || Ok::<_, TerminalError>(python_path.clone().into_os_string()),
        |existing| {
            std::env::join_paths(
                std::iter::once(python_path.clone()).chain(std::env::split_paths(&existing)),
            )
            .map_err(|error| {
                TerminalError::Execution(format!("failed to construct session PYTHONPATH: {error}"))
            })
        },
    )?;
    let python = python.into_string().map_err(|value| {
        TerminalError::Execution(format!(
            "session PYTHONPATH is not valid UTF-8: {}",
            value.to_string_lossy()
        ))
    })?;

    Ok(vec![
        ("TMPDIR".to_string(), tmp.clone()),
        ("TMP".to_string(), tmp.clone()),
        ("TEMP".to_string(), tmp),
        ("HOME".to_string(), home),
        ("XDG_CACHE_HOME".to_string(), cache),
        ("XDG_CONFIG_HOME".to_string(), config),
        ("BUN_INSTALL_CACHE_DIR".to_string(), bun),
        ("PLAYWRIGHT_BROWSERS_PATH".to_string(), playwright),
        ("PYTHONPATH".to_string(), python),
    ])
}

fn path_environment_value(name: &str, path: &Path) -> Result<String, TerminalError> {
    path.to_str().map(str::to_string).ok_or_else(|| {
        TerminalError::Execution(format!(
            "session environment path for {name} is not valid UTF-8: {}",
            path.display()
        ))
    })
}

#[derive(Template)]
#[template(path = "container_ipc_wrapper.sh", escape = "none")]
struct ContainerIpcWrapperTemplate<'a> {
    escaped_commands: &'a str,
    ipc_host: &'a str,
    ipc_port: u16,
    script: &'a str,
}

struct ContainerIpcEndpoint<'a> {
    host: &'a str,
    port: u16,
}

fn wrap_container_script(
    script: &str,
    ipc_commands: &[String],
    ipc_endpoint: Option<ContainerIpcEndpoint<'_>>,
) -> Result<String, TerminalError> {
    if ipc_commands.is_empty() {
        return Ok(script.to_string());
    }

    let endpoint = ipc_endpoint.ok_or_else(|| {
        TerminalError::Execution(
            "missing container IPC endpoint for wrapped script execution".to_string(),
        )
    })?;

    for name in ipc_commands {
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            let mut message = String::from("invalid ipc command name for container wrapper: ");
            message.push_str(name);
            return Err(TerminalError::Execution(message));
        }
    }

    let escaped_commands = ipc_commands
        .iter()
        .map(|name| shell_escape(name))
        .collect::<Vec<_>>()
        .join(" ");
    ContainerIpcWrapperTemplate {
        escaped_commands: &escaped_commands,
        ipc_host: endpoint.host,
        ipc_port: endpoint.port,
        script,
    }
    .render()
    .map_err(|error| {
        let error_text = error.to_string();
        let mut message = String::from("failed to render container IPC wrapper: ");
        message.push_str(error_text.as_str());
        TerminalError::Execution(message)
    })
}

struct ContainerIpcBridge {
    shutdown_tx: Sender<()>,
    port: u16,
}

impl ContainerIpcBridge {
    const fn port(&self) -> u16 {
        self.port
    }

    async fn stop(self) {
        tracing::debug!("stopping container IPC bridge");
        let _ = self.shutdown_tx.send(()).await;
    }
}

#[cfg(unix)]
fn start_container_ipc_bridge<E: Executor + Clone + 'static>(
    executor: E,
    registry: Arc<ToolRegistry>,
) -> Result<ContainerIpcBridge, TerminalError> {
    let listener = StdTcpListener::bind("127.0.0.1:0").map_err(|e| {
        TerminalError::Execution(format!("failed to bind container IPC tcp port: {e}"))
    })?;
    let local_addr = listener.local_addr().map_err(|e| {
        TerminalError::Execution(format!("failed to resolve container IPC tcp endpoint: {e}"))
    })?;
    tracing::debug!(
        bind = %local_addr,
        "starting container IPC bridge"
    );
    listener.set_nonblocking(true).map_err(|e| {
        TerminalError::Execution(format!(
            "failed to set IPC tcp listener nonblocking mode: {e}"
        ))
    })?;
    let listener = Async::new(listener).map_err(|e| {
        TerminalError::Execution(format!(
            "failed to register IPC tcp listener with async reactor: {e}"
        ))
    })?;

    let (shutdown_tx, shutdown_rx) = async_channel::bounded::<()>(1);
    let bridge_executor = executor.clone();
    executor
        .spawn(async move {
            tracing::debug!(bind = %local_addr, "container IPC bridge listening");
            if let Err(error) =
                run_container_ipc_bridge(listener, registry, shutdown_rx, bridge_executor).await
            {
                tracing::debug!(error = %error, "container IPC bridge stopped with error");
            }
            tracing::debug!(bind = %local_addr, "container IPC bridge stopped");
        })
        .detach();

    Ok(ContainerIpcBridge {
        shutdown_tx,
        port: local_addr.port(),
    })
}

#[cfg(not(unix))]
fn start_container_ipc_bridge<E: Executor + Clone + 'static>(
    _executor: E,
    _registry: Arc<ToolRegistry>,
) -> Result<ContainerIpcBridge, TerminalError> {
    Err(TerminalError::Execution(
        "container IPC bridge is only supported on unix hosts".to_string(),
    ))
}

#[cfg(unix)]
enum ContainerIpcBridgeEvent {
    Accept(std::io::Result<(Async<StdTcpStream>, std::net::SocketAddr)>),
    Shutdown,
}

#[cfg(unix)]
async fn run_container_ipc_bridge<E: Executor + Clone + 'static>(
    listener: Async<StdTcpListener>,
    registry: Arc<ToolRegistry>,
    shutdown_rx: Receiver<()>,
    executor: E,
) -> Result<(), String> {
    loop {
        let event = futures_lite::future::or(
            async { ContainerIpcBridgeEvent::Accept(listener.accept().await) },
            async {
                let _ = shutdown_rx.recv().await;
                ContainerIpcBridgeEvent::Shutdown
            },
        )
        .await;

        match event {
            ContainerIpcBridgeEvent::Shutdown => break,
            ContainerIpcBridgeEvent::Accept(Ok((stream, _addr))) => {
                tracing::debug!("container IPC bridge accepted connection");
                let registry = registry.clone();
                executor
                    .spawn(async move {
                        if let Err(error) = handle_container_ipc_connection(stream, registry).await
                        {
                            tracing::debug!(error = %error, "container IPC connection failed");
                        }
                    })
                    .detach();
            }
            ContainerIpcBridgeEvent::Accept(Err(error))
                if error.kind() == std::io::ErrorKind::Interrupted => {}
            ContainerIpcBridgeEvent::Accept(Err(error)) => {
                return Err(format!("container IPC accept failed: {error}"));
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
async fn handle_container_ipc_connection(
    mut stream: Async<StdTcpStream>,
    registry: Arc<ToolRegistry>,
) -> Result<(), String> {
    loop {
        let mut length_bytes = [0_u8; 4];
        match stream.read_exact(&mut length_bytes).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(format!("failed to read IPC request length: {error}")),
        }
        let body_length = u32::from_be_bytes(length_bytes) as usize;
        if body_length == 0 || body_length > 16 * 1024 * 1024 {
            return Err(format!("invalid IPC request length: {body_length}"));
        }

        let mut body = vec![0_u8; body_length];
        stream
            .read_exact(&mut body)
            .await
            .map_err(|e| format!("failed to read IPC request body: {e}"))?;

        if body.is_empty() {
            write_container_ipc_error(&mut stream, "empty IPC request body").await?;
            continue;
        }

        let method_length = body[0] as usize;
        if method_length == 0 {
            write_container_ipc_error(&mut stream, "empty IPC method name").await?;
            continue;
        }
        if body.len() < 1 + method_length {
            write_container_ipc_error(&mut stream, "invalid IPC request framing").await?;
            continue;
        }

        let method = std::str::from_utf8(&body[1..=method_length])
            .map_err(|e| format!("invalid IPC method name utf8: {e}"))?;
        let params = &body[1 + method_length..];
        let cli_args = decode_container_ipc_args(params)?;
        let response = registry.query_tool_handler(method, &cli_args).await;
        write_container_ipc_success(&mut stream, &response).await?;
    }
}

#[cfg(unix)]
fn decode_container_ipc_args(params: &[u8]) -> Result<Vec<String>, String> {
    let parsed: serde_json::Value =
        rmp_serde::from_slice(params).map_err(|e| format!("invalid IPC params: {e}"))?;
    let args = parsed
        .as_object()
        .ok_or_else(|| "container IPC params must be a map".to_string())?;
    let args = args
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<std::collections::HashMap<_, _>>();
    Ok(crate::command::flatten_args_to_cli(&args))
}

#[cfg(unix)]
async fn write_container_ipc_success(
    stream: &mut Async<StdTcpStream>,
    response: &crate::command::CommandEnvelope,
) -> Result<(), String> {
    let payload = rmp_serde::to_vec(response)
        .map_err(|e| format!("failed to encode IPC success payload: {e}"))?;
    write_container_ipc_response(stream, true, &payload).await
}

#[cfg(unix)]
async fn write_container_ipc_error(
    stream: &mut Async<StdTcpStream>,
    message: &str,
) -> Result<(), String> {
    let payload = rmp_serde::to_vec(&message.to_string())
        .map_err(|e| format!("failed to encode IPC error payload: {e}"))?;
    write_container_ipc_response(stream, false, &payload).await
}

#[cfg(unix)]
async fn write_container_ipc_response(
    stream: &mut Async<StdTcpStream>,
    success: bool,
    payload: &[u8],
) -> Result<(), String> {
    let body_length = 1usize
        .checked_add(payload.len())
        .ok_or_else(|| "IPC response payload length overflow".to_string())?;
    let response_length = u32::try_from(body_length)
        .map_err(|_| format!("IPC response body too large: {body_length} bytes"))?;

    stream
        .write_all(&response_length.to_be_bytes())
        .await
        .map_err(|e| format!("failed to write IPC response length: {e}"))?;
    stream
        .write_all(&[u8::from(success)])
        .await
        .map_err(|e| format!("failed to write IPC success flag: {e}"))?;
    stream
        .write_all(payload)
        .await
        .map_err(|e| format!("failed to write IPC payload: {e}"))?;
    stream
        .flush()
        .await
        .map_err(|e| format!("failed to flush IPC payload: {e}"))
}

fn preview_first_lines(data: &[u8], max_lines: usize) -> (String, bool) {
    if data.is_empty() || max_lines == 0 {
        return (String::new(), !data.is_empty() && max_lines == 0);
    }

    let text = String::from_utf8_lossy(data);
    let mut preview = String::new();
    let mut total_lines = 0usize;

    for line in text.lines() {
        total_lines += 1;
        if total_lines <= max_lines {
            if !preview.is_empty() {
                preview.push('\n');
            }
            preview.push_str(line);
        }
    }

    let truncated = total_lines > max_lines;
    (preview, truncated)
}

enum TerminalChunk {
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
    StdoutClosed,
    StderrClosed,
}

fn spawn_terminal_reader<R>(
    mut reader: R,
    tx: async_channel::Sender<TerminalChunk>,
    is_stderr: bool,
) where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buffer = vec![0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => {
                    let _ = tx.send_blocking(if is_stderr {
                        TerminalChunk::StderrClosed
                    } else {
                        TerminalChunk::StdoutClosed
                    });
                    break;
                }
                Ok(count) => {
                    let chunk = buffer[..count].to_vec();
                    let _ = tx.send_blocking(if is_stderr {
                        TerminalChunk::Stderr(chunk)
                    } else {
                        TerminalChunk::Stdout(chunk)
                    });
                }
                Err(error) => {
                    warn!(error = %error, is_stderr, "terminal reader failed");
                    let _ = tx.send_blocking(if is_stderr {
                        TerminalChunk::StderrClosed
                    } else {
                        TerminalChunk::StdoutClosed
                    });
                    break;
                }
            }
        }
    });
}

fn spawn_terminal_stdin_writer<W>(mut writer: W) -> async_channel::Sender<Vec<u8>>
where
    W: Write + Send + 'static,
{
    let (tx, rx) = async_channel::unbounded::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(bytes) = rx.recv_blocking() {
            if bytes.is_empty() {
                continue;
            }
            if let Err(error) = writer.write_all(&bytes) {
                warn!(error = %error, "terminal stdin write failed");
                break;
            }
            if let Err(error) = writer.flush() {
                warn!(error = %error, "terminal stdin flush failed");
                break;
            }
        }
    });
    tx
}

async fn drain_terminal_chunks(
    job_registry: JobRegistry,
    pid: u32,
    rx: async_channel::Receiver<TerminalChunk>,
) {
    while let Ok(chunk) = rx.recv().await {
        match chunk {
            TerminalChunk::Stdout(bytes) => job_registry.append_stdout(pid, bytes).await,
            TerminalChunk::Stderr(bytes) => job_registry.append_stderr(pid, bytes).await,
            TerminalChunk::StdoutClosed => job_registry.close_stdout(pid).await,
            TerminalChunk::StderrClosed => job_registry.close_stderr(pid).await,
        }
    }
}

async fn wait_for_terminal_stream_close(job_registry: &JobRegistry, pid: u32) {
    while !job_registry.terminal_streams_closed(pid).await {
        async_io::Timer::after(Duration::from_millis(10)).await;
    }
}

/// Creates the IPC router with built-in and tool commands (standalone version).
fn create_ipc_router(registry: &Arc<ToolRegistry>) -> Option<IpcRouter> {
    let mut router = builtin_router();

    // Register all configured tools as IPC commands
    let tool_names = registry.registered_tool_names();
    tracing::info!(tools = ?tool_names, "Creating IPC router with registered tools");
    for name in tool_names {
        router = crate::register_tool_command(router, Arc::clone(registry), &name);
    }

    let has_methods = router.methods().next().is_some();
    has_methods.then_some(router)
}

fn create_ipc_gateway_router(registry: &Arc<ToolRegistry>) -> Option<IpcRouter> {
    let mut router = crate::register_ipc_gateway_command(IpcRouter::new(), Arc::clone(registry));

    // In unsafe mode, keep tool commands usable (websearch/webfetch/ask/task/todo...),
    // but never override native shell task/process commands like kill/jobs.
    let blocked = ["kill", "jobs"];
    let tool_names = registry.registered_tool_names();
    for name in tool_names {
        if blocked.contains(&name.as_str()) {
            continue;
        }
        router = crate::register_tool_command(router, Arc::clone(registry), &name);
    }

    let has_methods = router.methods().next().is_some();
    has_methods.then_some(router)
}

/// Errors that can occur during terminal execution.
#[derive(Debug, thiserror::Error)]
pub enum TerminalError {
    /// Permission denied for the requested mode.
    #[error(
        "sandbox blocked {mode} execution; approval is required to escalate this terminal command",
        mode = .0.description()
    )]
    PermissionDenied(TerminalMode),

    /// Permission check failed.
    #[error("permission error: {0}")]
    Permission(#[from] PermissionError),

    /// Sandbox setup failed.
    #[error("sandbox setup failed: {0}")]
    SandboxSetup(String),

    /// Script execution failed.
    #[error("execution failed: {0}")]
    Execution(String),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl<P, E, State> std::fmt::Debug for TerminalTool<P, E, State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalTool")
            .field("working_dir", &self.working_dir)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use super::*;
    use crate::ToolRegistryBuilder;
    use crate::builtin::{InputTerminalArgs, InputTerminalTool};
    use heel::DenyAll;

    fn parse_terminal_tool_result(result: &ToolResult) -> TerminalResult {
        serde_json::from_str(
            &result
                .render_for_model()
                .expect("terminal result should render"),
        )
        .expect("terminal tool output should decode")
    }

    #[cfg(unix)]
    #[test]
    fn shell_launch_uses_a_stable_posix_contract() {
        let script = "printf '%s\\n' hello | sed -n '1p'";
        let (program, args) = shell_launch(script);

        assert_eq!(program, "sh");
        assert_eq!(args, ["-c", script]);
    }

    #[tokio::test]
    async fn session_runtime_environment_is_prepared_without_wrapping_the_script() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let environment = prepare_session_runtime_environment(dir.path())
            .await
            .expect("session runtime environment should be prepared");
        let value = |name: &str| {
            environment
                .iter()
                .find_map(|(key, value)| (key == name).then_some(value.as_str()))
                .unwrap_or_else(|| panic!("missing session environment variable {name}"))
        };

        assert_eq!(value("HOME"), dir.path().to_str().expect("UTF-8 temp path"));
        assert_eq!(
            value("TMPDIR"),
            dir.path().join("tmp").to_str().expect("UTF-8 temp path")
        );
        assert_eq!(value("TMP"), value("TMPDIR"));
        assert_eq!(value("TEMP"), value("TMPDIR"));
        assert!(dir.path().join("tmp").is_dir());
        assert!(dir.path().join(".cache/bun").is_dir());
        assert!(dir.path().join(".cache/ms-playwright").is_dir());
        assert!(dir.path().join(".config").is_dir());
    }

    fn output_text(output: &OutputEntry) -> Option<&str> {
        match output {
            OutputEntry::Inline { content } | OutputEntry::Loaded { content, .. } => {
                match content {
                    Content::Text { text, .. } => Some(text.as_str()),
                    Content::Image { .. } => None,
                }
            }
            OutputEntry::Stored { content, .. } => match content.as_ref() {
                Some(Content::Text { text, .. }) => Some(text.as_str()),
                Some(Content::Image { .. }) | None => None,
            },
            OutputEntry::Empty => None,
        }
    }

    async fn wait_for_completed_task(
        receiver: &BackgroundTaskReceiver,
        task_id: &str,
    ) -> CompletedTask {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(task) = receiver
                    .take_completed()
                    .into_iter()
                    .find(|task| task.task_id == task_id)
                {
                    return task;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("background task should complete after terminal input")
    }

    #[derive(Default)]
    struct TestPermissionHandler {
        mode_checks: AtomicUsize,
        domain_checks: AtomicUsize,
        allow_domain: bool,
    }

    impl PermissionHandler for TestPermissionHandler {
        fn check(
            &self,
            mode: TerminalMode,
            _script: &str,
        ) -> impl std::future::Future<Output = Result<bool, PermissionError>> + Send {
            self.mode_checks.fetch_add(1, AtomicOrdering::Relaxed);
            std::future::ready(Ok(match mode {
                TerminalMode::Sandboxed => true,
                TerminalMode::Unsafe => false,
            }))
        }

        fn check_domain(
            &self,
            _domain: &str,
            _port: u16,
        ) -> impl std::future::Future<Output = bool> + Send {
            self.domain_checks.fetch_add(1, AtomicOrdering::Relaxed);
            std::future::ready(self.allow_domain)
        }
    }

    #[tokio::test]
    async fn test_terminal_args_defaults() {
        let args: TerminalArgs = serde_json::from_str(
            r#"{"description":"print a greeting","script":"echo hello","timeout":1}"#,
        )
        .unwrap();
        assert_eq!(args.expect, OutputFormat::Text);
        assert_eq!(args.resolution, MediaResolution::Auto);
        assert_eq!(args.timeout, 1);
        assert_eq!(args.mode, TerminalExecutionMode::Default);
    }

    #[tokio::test]
    async fn test_terminal_result_serialization() {
        use crate::output::Content;

        let result = TerminalResult {
            stdout: OutputEntry::Stored {
                url: "outputs/bold-oak-calm-river.txt".to_string(),
                path: "/tmp/outputs/bold-oak-calm-river.txt".to_string(),
                content: Some(Content::Text {
                    text: "preview text".to_string(),
                    truncated: true,
                }),
            },
            stderr: None,
            exit_code: 0,
            task_id: None,
            status: None,
            background_reason: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("bold-oak-calm-river.txt"));
        assert!(!json.contains("stderr")); // should be skipped when None
        assert!(!json.contains("task_id")); // should be skipped when None

        // Test inline output (no URL)
        let result_inline = TerminalResult {
            stdout: OutputEntry::Inline {
                content: Content::Text {
                    text: "done".to_string(),
                    truncated: false,
                },
            },
            stderr: None,
            exit_code: 0,
            task_id: None,
            status: None,
            background_reason: None,
        };
        let json_inline = serde_json::to_string(&result_inline).unwrap();
        assert!(!json_inline.contains("\"url\""));
        assert!(json_inline.contains("\"content\""));

        // Test background task result
        let result_background = TerminalResult {
            stdout: OutputEntry::Empty,
            stderr: None,
            exit_code: 0,
            task_id: Some("amber-forest-thunder-pearl".to_string()),
            status: Some("running".to_string()),
            background_reason: Some(BackgroundReason::Timeout {
                configured_seconds: 30,
            }),
        };
        let json_bg = serde_json::to_string(&result_background).unwrap();
        assert!(json_bg.contains("\"task_id\":\"amber-forest-thunder-pearl\""));
        assert!(json_bg.contains("\"status\":\"running\""));
    }

    #[tokio::test]
    async fn test_terminal_args_timeout() {
        let args: TerminalArgs = serde_json::from_str(
            r#"{"description":"print a greeting","script":"echo hello","timeout":0}"#,
        )
        .unwrap();
        assert_eq!(args.timeout, 0);

        let args_default = serde_json::from_str::<TerminalArgs>(
            r#"{"description":"print a greeting","script":"echo hello"}"#,
        )
        .unwrap_err();
        assert!(args_default.to_string().contains("timeout"));

        let missing_description =
            serde_json::from_str::<TerminalArgs>(r#"{"script":"echo hello","timeout":1}"#)
                .unwrap_err();
        assert!(missing_description.to_string().contains("description"));
    }

    #[test]
    fn wrap_container_script_refreshes_command_hash_table() {
        let wrapped = wrap_container_script(
            "websearch \"gold price\"",
            &[String::from("websearch")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
        )
        .expect("wrap script");
        assert!(wrapped.contains("hash -r;"));
        assert!(wrapped.contains("MAY_IPC_DIR"));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));
    }

    #[test]
    fn wrap_container_script_preserves_user_script_semantics() {
        let script = "echo '$5,040' && subagent --subagent .skills/slide/subagents/art_direction.md --prompt 'x'";
        let wrapped = wrap_container_script(
            script,
            &[String::from("subagent")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
        )
        .expect("wrap script");
        assert!(wrapped.contains(script));
        assert!(!wrapped.contains("set -euo pipefail"));
    }

    #[test]
    fn wrap_container_script_rejects_invalid_ipc_command_names() {
        let error = wrap_container_script(
            "echo ok",
            &[String::from("bad name")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
        )
        .expect_err("invalid command name must fail");
        assert!(
            error
                .to_string()
                .contains("invalid ipc command name for container wrapper")
        );
    }

    #[test]
    fn wrap_container_script_renders_expected_endpoint() {
        let wrapped = wrap_container_script(
            "environment --query deployment",
            &[String::from("environment"), String::from("session")],
            Some(ContainerIpcEndpoint {
                host: "host.containers.internal",
                port: 43123,
            }),
        )
        .expect("wrap script");
        assert!(wrapped.contains("tcp://host.containers.internal:43123"));
        assert!(wrapped.contains("for cmd in 'environment' 'session'; do"));
    }

    #[tokio::test]
    async fn ensure_mode_allowed_requires_unsafe_approval() {
        let handler = TestPermissionHandler {
            allow_domain: true,
            ..Default::default()
        };
        let err = ensure_mode_allowed(&handler, TerminalMode::Unsafe, "rm -rf /tmp/demo")
            .await
            .expect_err("unsafe mode should be denied");
        assert!(matches!(
            err,
            TerminalError::PermissionDenied(TerminalMode::Unsafe)
        ));
        assert_eq!(handler.mode_checks.load(AtomicOrdering::Relaxed), 1);
    }

    #[tokio::test]
    async fn permission_network_policy_delegates_domain_checks() {
        let handler = Arc::new(TestPermissionHandler {
            allow_domain: true,
            ..Default::default()
        });
        let policy = PermissionNetworkPolicy {
            permission_handler: handler.clone(),
        };
        let request = DomainRequest::new("example.com", 443);
        assert!(policy.check(&request).await);
        assert_eq!(handler.domain_checks.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn sandboxed_execution_mode_deserializes() {
        let mode: TerminalExecutionMode =
            serde_json::from_str("\"sandboxed\"").expect("sandboxed mode must deserialize");
        assert_eq!(mode, TerminalExecutionMode::Sandboxed);
    }

    #[tokio::test]
    async fn child_terminal_exposes_controls_with_isolated_jobs() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize");
        tool.job_registry()
            .register(
                424_242,
                "parent-task",
                "parent-execution",
                "sleep 30",
                TerminalMode::Sandboxed,
                None,
            )
            .await;

        let child = tool.child();
        assert!(child.job_registry().list().await.is_empty());
        let registry = Arc::new(ToolRegistryBuilder::new().build(child.outputs_dir()));
        let names = child
            .with_registry(registry)
            .to_dyn()
            .into_entries()
            .into_iter()
            .map(|entry| entry.definition.name().to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "terminal",
                "terminal_kill",
                "terminal_input",
                "terminal_read"
            ]
        );
    }

    #[tokio::test]
    async fn sandboxed_command_output_executes_pwd() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let config = SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(dir.path())
            .security(SecurityConfig::interactive())
            .build();
        let sandbox = Sandbox::with_config_and_executor(config, executor_core::tokio::TokioGlobal)
            .await
            .expect("sandbox should initialize");

        let output = sandbox
            .command("/bin/pwd")
            .output()
            .await
            .expect("sandbox command output should succeed");
        eprintln!("PWD OUTPUT STATUS: {:?}", output.status);
        eprintln!(
            "PWD OUTPUT STDOUT: {:?}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert_eq!(output.status.code(), Some(0));
    }

    #[tokio::test]
    async fn sandboxed_terminal_executes_pwd() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize");
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let job_registry = tool.job_registry();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(StandaloneExecution {
            working_dir: tool.working_dir().clone(),
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            executor: executor_core::tokio::TokioGlobal,
            registry,
            permission_handler: Arc::new(TestPermissionHandler::default()),
            job_registry,
            task_id: "task-test".to_string(),
            execution_id: "exec-test".to_string(),
            background_mode: Arc::new(AtomicBool::new(false)),
            script: "/bin/pwd".to_string(),
            mode: TerminalMode::Sandboxed,
            backend: ShellBackend::Local,
            ssh_target: None,
            ssh_runtime: None,
            container_runtime: None,
            stdin_blocked_notice: None,
            background_startup: BackgroundStartup::new(startup_tx),
            network_audit: None,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            store_dir: tool.outputs_dir(),
            max_lines: 50,
            raw: false,
        })
        .await;
        match result {
            Ok(ok) => {
                eprintln!("PWD RESULT: {ok:?}");
                assert_eq!(ok.exit_code, 0, "unexpected terminal result: {ok:?}");
            }
            Err(error) => panic!("sandboxed pwd should succeed, got error: {error}"),
        }
    }

    #[tokio::test]
    async fn failed_command_returns_json_with_nonzero_exit_code() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);

        let result = tool
            .call(TerminalArgs {
                description: "exit with non-zero status".to_string(),
                script: "exit 42".to_string(),
                mode: TerminalExecutionMode::Sandboxed,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                timeout: 30,
                max_lines: 50,
                raw: false,
            })
            .await
            .expect("terminal call should not be promoted to transport error on non-zero exit");

        assert!(
            !result.is_error(),
            "non-zero exit must not produce ToolResult::Error, got: {result:?}"
        );
        let payload = parse_terminal_tool_result(&result);
        assert_eq!(payload.exit_code, 42, "exit code should round-trip");
        assert!(
            payload.task_id.is_none(),
            "foreground completion must not carry a task id"
        );
        assert!(
            payload.status.is_none(),
            "foreground completion must not carry a running status"
        );
    }

    #[tokio::test]
    async fn background_terminal_accepts_terminal_input() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        let background_receiver = tool.background_receiver();
        let input_tool = InputTerminalTool::new(tool.job_registry());

        let result = tool
            .call(TerminalArgs {
                description: "wait for terminal input and echo it back".to_string(),
                script: "printf 'name? '; read name; printf 'hello %s\\n' \"$name\"".to_string(),
                mode: TerminalExecutionMode::Sandboxed,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                timeout: 0,
                max_lines: 50,
                raw: false,
            })
            .await
            .expect("terminal call should succeed");

        let payload = parse_terminal_tool_result(&result);
        assert_eq!(payload.status.as_deref(), Some("running"));
        let task_id = payload
            .task_id
            .clone()
            .expect("timeout=0 should return a background task id");

        input_tool
            .call(InputTerminalArgs {
                task_id: task_id.clone(),
                input: "lexo".to_string(),
                append_newline: true,
            })
            .await
            .expect("terminal_input should succeed");

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("background task should complete successfully after stdin input");
        let stdout =
            output_text(&terminal_result.stdout).expect("completed task should include stdout");
        assert!(
            stdout.contains("hello lexo"),
            "stdin input should reach the terminal process, got: {stdout}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn killing_background_terminal_kills_descendant_processes() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        let child_pid_path = dir.path().join("child.pid");

        let result = tool
            .call(TerminalArgs {
                description: "start a descendant process".to_string(),
                script: "sleep 30 & echo $! > child.pid; wait".to_string(),
                mode: TerminalExecutionMode::Sandboxed,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                timeout: 0,
                max_lines: 50,
                raw: false,
            })
            .await
            .expect("terminal call should succeed");
        let payload = parse_terminal_tool_result(&result);
        assert_eq!(payload.status.as_deref(), Some("running"));

        let mut child_pid = None;
        for _ in 0..100 {
            match async_fs::read_to_string(&child_pid_path).await {
                Ok(value) => {
                    child_pid = Some(value.trim().parse::<i32>().expect("child pid should parse"));
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    async_io::Timer::after(std::time::Duration::from_millis(20)).await;
                }
                Err(error) => panic!("failed to read child pid: {error}"),
            }
        }
        let child_pid = child_pid.expect("descendant pid should be written before timeout");

        assert_eq!(tool.job_registry().kill_running().await, 1);

        for _ in 0..100 {
            let exists = unsafe { libc::kill(child_pid, 0) } == 0;
            if !exists {
                return;
            }
            async_io::Timer::after(std::time::Duration::from_millis(20)).await;
        }
        panic!("descendant process {child_pid} survived terminal cancellation");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn sandboxed_terminal_auto_promotes_stdin_wait_and_accepts_input() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        let background_receiver = tool.background_receiver();
        let input_tool = InputTerminalTool::new(tool.job_registry());

        let result = tool
            .call(TerminalArgs {
                description: "wait for terminal input and echo it back".to_string(),
                script: "printf 'name? '; read name; printf 'hello %s\\n' \"$name\"".to_string(),
                mode: TerminalExecutionMode::Sandboxed,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                timeout: 30,
                max_lines: 50,
                raw: false,
            })
            .await
            .expect("terminal call should succeed");

        let payload = parse_terminal_tool_result(&result);
        assert_eq!(payload.status.as_deref(), Some("running"));
        let task_id = payload
            .task_id
            .clone()
            .expect("stdin-blocked process should promote to background");
        let notice = output_text(&payload.stdout).expect("background preview should include text");
        assert!(
            notice.starts_with(TERMINAL_STDIN_BLOCKED_NOTICE),
            "unexpected promotion notice: {notice}"
        );

        input_tool
            .call(InputTerminalArgs {
                task_id: task_id.clone(),
                input: "lexo".to_string(),
                append_newline: true,
            })
            .await
            .expect("terminal_input should succeed");

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("background task should complete successfully after stdin input");
        let stdout =
            output_text(&terminal_result.stdout).expect("completed task should include stdout");
        assert!(
            stdout.contains("hello lexo"),
            "stdin input should reach the terminal process, got: {stdout}"
        );
    }
}
