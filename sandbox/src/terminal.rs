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

use std::{
    borrow::Cow,
    io::{Read, Write},
    net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use aither_core::llm::ToolContext;
use aither_core::llm::{
    CommandOutcome, CommandResult, CommandStream, IntoToolResult, StoredFile, Tool, ToolResult,
};
use askama::Template;
use async_channel::{Receiver, Sender};
use async_io::Async;
use executor_core::{Executor, Task};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::{
    Access, AllowAll, Audited, CpuPriority, DomainRequest, Grant, IoPriority, IpcRouter,
    NetworkAuditLog, NetworkPolicy, ProcessPriority, Sandbox, SandboxConfig, SecurityConfig,
    StdioConfig, WorkingDir,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shell_words::split as split_shell_words;
use tracing::{debug, info, warn};

#[cfg(all(unix, feature = "jobserver"))]
use crate::jobserver::{JobBroker, JobPool, job_broker_channel};
use crate::{
    ContainerPath, PreparedTools, ToolDistribution, ToolRuntimeError,
    builtin::builtin_router,
    command::ToolRegistry,
    job_registry::{JobRegistry, JobStatus, job_registry_channel},
    output::{
        INLINE_OUTPUT_LIMIT, MediaResolution, OutputFormat, OutputStore, empty_stream,
        save_raw_to_file, stored_path,
    },
    permission::{PermissionError, PermissionHandler, TerminalMode},
    shell_session::{
        BridgeAddress, ContainerShellRuntime, ShellBackend, ShellRuntimeAvailability,
        ShellSessionRegistry, SshRuntimeProfile,
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
    let handler: DynToolHandler = Arc::new(move |args: &str, cx: ToolContext| {
        let tool = tool.clone();
        let args = args.to_string();
        Box::pin(async move {
            let parsed = match serde_json::from_str::<T::Arguments>(&args) {
                Ok(parsed) => parsed,
                Err(error) => return ToolResult::error(format!("Parse error: {error}")),
            };
            match tool.call(parsed, cx).await {
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
/// A call runs in the foreground and is killed — with its whole process
/// tree — after 10 seconds. Run long-running commands with
/// `terminal_background`, which takes an explicit job timeout.
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

    /// Maximum number of output lines to include inline.
    /// For foreground executions, this is the inline line budget
    /// before output offloads to file.
    /// For background executions started with `terminal_background`, only
    /// the first `max_lines` are returned immediately, while full output is
    /// redirected to file.
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

/// Run a terminal command as a background job.
///
/// The command starts detached and this call returns immediately with the
/// job's task id; `terminal` only runs commands in the foreground and kills
/// them after 10 seconds. Use `terminal_read` for incremental output,
/// `terminal_input` for stdin, and `terminal_kill` to stop the job early.
///
/// `timeout` is the job's maximum runtime in seconds: the job and its
/// process tree are killed when it passes, and it may be at most 120
/// minutes. To wait longer than that, schedule a wake-up (cron) rather than
/// starting a longer job.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TerminalBackgroundArgs {
    /// Every argument a foreground `terminal` call takes.
    #[serde(flatten)]
    pub terminal: TerminalArgs,
    /// Maximum runtime of the background job in seconds. The job is killed
    /// when it passes. At most 7200 (120 minutes).
    pub timeout: u64,
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

/// How long a foreground `terminal` command may run before it is killed
/// with its process tree.
pub const DEFAULT_FOREGROUND_LIMIT: Duration = Duration::from_secs(10);
/// Largest `timeout` a `terminal_background` job accepts.
pub const DEFAULT_BACKGROUND_TIMEOUT_CAP: Duration = Duration::from_secs(120 * 60);
/// A background job still running after this long produces one system alert.
pub const DEFAULT_LONG_JOB_ALERT_AFTER: Duration = Duration::from_secs(40 * 60);

/// Time bounds applied to every execution this terminal spawns.
#[derive(Debug, Clone, Copy)]
struct TerminalLimits {
    /// Foreground kill limit for `terminal` calls.
    foreground: Duration,
    /// Maximum accepted `timeout` for `terminal_background` jobs.
    background_timeout_cap: Duration,
    /// Runtime after which a background job produces a long-running alert.
    long_job_alert: Duration,
}

impl Default for TerminalLimits {
    fn default() -> Self {
        Self {
            foreground: DEFAULT_FOREGROUND_LIMIT,
            background_timeout_cap: DEFAULT_BACKGROUND_TIMEOUT_CAP,
            long_job_alert: DEFAULT_LONG_JOB_ALERT_AFTER,
        }
    }
}

/// A background job that crossed the long-running threshold.
///
/// Sent through the terminal's alert channel once per job; the agent turns
/// it into a durable system alert.
#[derive(Debug, Clone)]
pub struct LongRunningJob {
    /// Task ID of the job (four random words like "amber-forest-thunder-pearl").
    pub task_id: String,
    /// The script the job runs.
    pub script: String,
    /// How long the job had been running when the alert fired.
    pub elapsed: Duration,
}

/// A command that ran to completion.
#[derive(Debug, Clone)]
pub struct CompletedCommand {
    /// stdout output.
    pub stdout: CommandStream,

    /// stderr output (if non-empty).
    pub stderr: Option<CommandStream>,

    /// Exit code of the script.
    pub exit_code: i32,
}

impl From<CompletedCommand> for CommandResult {
    fn from(command: CompletedCommand) -> Self {
        Self {
            stdout: command.stdout,
            stderr: command.stderr,
            outcome: CommandOutcome::Exited {
                code: command.exit_code,
            },
        }
    }
}

/// A background job that has just been started by `terminal_background`.
///
/// Sent through the terminal's started-task channel once per launched job;
/// the agent turns it into a durable reminder and a `background_task_started`
/// event.
#[derive(Debug, Clone)]
pub struct StartedBackgroundTask {
    /// Task ID of the job (four random words like "amber-forest-thunder-pearl").
    pub task_id: String,
    /// First lines of the redirected output; empty when nothing was written yet.
    pub output_preview: String,
    /// File receiving the full redirected output.
    pub output_file: StoredFile,
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
    pub result: Result<CompletedCommand, TerminalError>,
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
    alert_rx: Receiver<LongRunningJob>,
    started_rx: Receiver<StartedBackgroundTask>,
    running_tasks: Arc<AtomicUsize>,
}

/// Sender half of a [`BackgroundTaskReceiver`] channel.
///
/// A `TerminalTool` owns the sender side internally; the exposed pair lets
/// tests and embedders drive a receiver without a live terminal backend:
/// `task_started`/`task_finished` move the running counter the receiver's
/// `has_running` reads, and the `send_*` methods deliver the same payloads
/// spawned executions produce.
#[derive(Clone, Debug)]
pub struct BackgroundTaskSender {
    completed_tx: Sender<CompletedTask>,
    alert_tx: Sender<LongRunningJob>,
    started_tx: Sender<StartedBackgroundTask>,
    running_tasks: Arc<AtomicUsize>,
}

impl BackgroundTaskSender {
    /// Records a task start: `has_running` on the paired receiver turns true.
    pub fn task_started(&self) {
        self.running_tasks.fetch_add(1, Ordering::AcqRel);
    }

    /// Records a task end: decrements the running counter.
    ///
    /// # Panics
    ///
    /// Panics when more tasks are finished than were started — the counter
    /// can never go negative, so a mismatched pair is a bug in the driver.
    pub fn task_finished(&self) {
        let previous = self.running_tasks.fetch_sub(1, Ordering::AcqRel);
        assert!(
            previous > 0,
            "task_finished without a matching task_started"
        );
    }

    /// Delivers a completed-task event to the receiver.
    pub async fn send_completed(&self, task: CompletedTask) {
        let _ = self.completed_tx.send(task).await;
    }

    /// Delivers a long-running-job alert to the receiver.
    pub async fn send_long_running(&self, alert: LongRunningJob) {
        let _ = self.alert_tx.send(alert).await;
    }

    /// Delivers a background-task-started event to the receiver.
    pub async fn send_started(&self, task: StartedBackgroundTask) {
        let _ = self.started_tx.send(task).await;
    }
}

/// Creates a standalone [`BackgroundTaskSender`]/[`BackgroundTaskReceiver`]
/// pair disconnected from any `TerminalTool`, mirroring
/// [`job_registry_channel`](crate::job_registry_channel).
#[must_use]
pub fn background_task_channel() -> (BackgroundTaskSender, BackgroundTaskReceiver) {
    let (completed_tx, rx) = async_channel::unbounded();
    let (alert_tx, alert_rx) = async_channel::unbounded();
    let (started_tx, started_rx) = async_channel::unbounded();
    let running_tasks = Arc::new(AtomicUsize::new(0));
    (
        BackgroundTaskSender {
            completed_tx,
            alert_tx,
            started_tx,
            running_tasks: running_tasks.clone(),
        },
        BackgroundTaskReceiver {
            rx,
            alert_rx,
            started_rx,
            running_tasks,
        },
    )
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

    /// Takes all pending long-running job alerts without blocking.
    ///
    /// Each alert names a background job that crossed the terminal's
    /// long-running threshold; a job produces at most one.
    #[must_use]
    pub fn take_long_running_alerts(&self) -> Vec<LongRunningJob> {
        let mut alerts = Vec::new();
        while let Ok(alert) = self.alert_rx.try_recv() {
            alerts.push(alert);
        }
        alerts
    }

    /// Takes all freshly started background tasks without blocking.
    ///
    /// Each entry names a job `terminal_background` has just launched; a job
    /// produces exactly one.
    #[must_use]
    pub fn take_started(&self) -> Vec<StartedBackgroundTask> {
        let mut started = Vec::new();
        while let Ok(task) = self.started_rx.try_recv() {
            started.push(task);
        }
        started
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
            .field("pending_alerts", &self.alert_rx.len())
            .field("pending_started", &self.started_rx.len())
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

/// Registry state that can be rebound to a different job registry.
///
/// Sealed: declared `pub` only so it may bound the public `TerminalTool`
/// impl; it lives in this private module and is not re-exported.
pub trait RegistryState: Clone {
    /// Returns this state with its terminal controls bound to `jobs`.
    fn for_jobs(&self, jobs: JobRegistry, output_dir: PathBuf) -> Self;
}

impl RegistryState for Unconfigured {
    fn for_jobs(&self, _jobs: JobRegistry, _output_dir: PathBuf) -> Self {
        Self
    }
}

impl RegistryState for Configured {
    fn for_jobs(&self, jobs: JobRegistry, output_dir: PathBuf) -> Self {
        Self {
            registry: Arc::new(
                ToolRegistry::clone(&self.registry).with_terminal_controls(jobs, output_dir),
            ),
        }
    }
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
    /// Channel for receiving long-running job alerts.
    alert_rx: Receiver<LongRunningJob>,
    /// Channel for sending long-running job alerts (cloned for each watchdog).
    alert_tx: Sender<LongRunningJob>,
    /// Channel for receiving freshly started background tasks.
    started_rx: Receiver<StartedBackgroundTask>,
    /// Channel for announcing freshly started background tasks.
    started_tx: Sender<StartedBackgroundTask>,
    /// Time bounds applied to spawned executions.
    limits: TerminalLimits,
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
    /// Scheduling priority applied to every sandboxed execution.
    priority: ProcessPriority,
    /// Shared jobserver token broker, when configured.
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
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
            alert_rx: self.alert_rx.clone(),
            alert_tx: self.alert_tx.clone(),
            started_rx: self.started_rx.clone(),
            started_tx: self.started_tx.clone(),
            limits: self.limits,
            running_tasks: self.running_tasks.clone(),
            permission_rx: self.permission_rx.clone(),
            permission_tx: self.permission_tx.clone(),
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
            network_audit: self.network_audit.clone(),
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver.clone(),
            registry: self.registry.clone(),
        }
    }
}

impl<P, E: Executor + Clone + 'static> TerminalTool<P, E, Unconfigured> {
    /// Injects the runtime configuration used by `terminal` execution mode resolution.
    ///
    /// The registry is configuration to prepare with [`Self::initialize_tools`];
    /// a registry that is already prepared is installed with
    /// [`Self::adopt_prepared_tools`], which checks it against this terminal.
    ///
    /// # Panics
    /// Panics if `sessions` is already prepared.
    #[must_use]
    pub fn with_shell_sessions(mut self, sessions: ShellSessionRegistry) -> Self {
        assert!(
            sessions.prepared_tools.is_none(),
            "a prepared shell session registry is installed with adopt_prepared_tools"
        );
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
        let (alert_tx, alert_rx) = async_channel::unbounded();
        let (started_tx, started_rx) = async_channel::unbounded();
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
            alert_rx,
            alert_tx,
            started_rx,
            started_tx,
            limits: TerminalLimits::default(),
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            network_audit: None,
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
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
        let (alert_tx, alert_rx) = async_channel::unbounded();
        let (started_tx, started_rx) = async_channel::unbounded();
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
            alert_rx,
            alert_tx,
            started_rx,
            started_tx,
            limits: TerminalLimits::default(),
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            network_audit: None,
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            registry: Unconfigured,
        })
    }

    /// Attaches a tool registry to this terminal tool, enabling IPC command dispatch.
    #[must_use]
    pub fn with_registry(self, registry: Arc<ToolRegistry>) -> TerminalTool<P, E, Configured> {
        let registry = Arc::new(
            Arc::unwrap_or_clone(registry)
                .with_terminal_controls(self.job_registry(), self.outputs_dir()),
        );
        TerminalTool {
            working_dir: self.working_dir,
            shell_sessions: self.shell_sessions,
            permission_handler: self.permission_handler,
            executor: self.executor,
            output_store: self.output_store,
            job_registry: self.job_registry,
            completed_rx: self.completed_rx,
            completed_tx: self.completed_tx,
            alert_rx: self.alert_rx,
            alert_tx: self.alert_tx,
            started_rx: self.started_rx,
            started_tx: self.started_tx,
            limits: self.limits,
            running_tasks: self.running_tasks,
            permission_rx: self.permission_rx,
            permission_tx: self.permission_tx,
            writable_paths: self.writable_paths,
            readable_paths: self.readable_paths,
            network_audit: self.network_audit,
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver,
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

    /// Lowers the scheduling priority of the commands this tool runs.
    ///
    /// The priority is applied to every sandbox the tool creates — one-shot
    /// commands, persistent shell sessions and background jobs — and is
    /// inherited by everything they spawn. Container executions run the script
    /// through the image's own `heel run`, so the image needs a `heel` binary
    /// when a non-normal priority is configured. SSH executions are not
    /// affected: the remote host runs its own `heel`, managed outside this
    /// configuration.
    #[must_use]
    pub const fn with_process_priority(mut self, priority: ProcessPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Shares a jobserver token pool across every command this tool runs.
    ///
    /// `pool` is the host-provided shared pool (a `fifo:` jobserver). Each
    /// command — foreground or background — gets its own fifo beneath
    /// `fifo_dir` (created if missing, broker-owned, kept out of command
    /// working directories), exported through
    /// `MAKEFLAGS=--jobserver-auth=fifo:PATH`. The broker moves tokens
    /// between the pool and each fifo on demand and returns every token
    /// once the command's whole process tree is gone: the fifo's read end
    /// reports EOF only when its last writer — including surviving
    /// grandchildren — has exited. Commands never see the shared pool
    /// directly, so killed commands cannot leak tokens.
    ///
    /// Only local executions can open a host fifo; container and SSH
    /// commands fail while a pool is configured rather than running
    /// unbounded.
    ///
    /// # Errors
    /// Returns an error when the fifo directory cannot be created or the
    /// broker's readiness poller cannot be set up.
    #[cfg(all(unix, feature = "jobserver"))]
    pub fn with_jobserver_pool(
        mut self,
        pool: JobPool,
        fifo_dir: impl AsRef<Path>,
    ) -> Result<Self, TerminalError> {
        let (broker, service) = job_broker_channel(pool, fifo_dir)?;
        self.executor
            .spawn(async move { service.serve().await })
            .detach();
        self.jobserver = Some(broker);
        Ok(self)
    }

    /// Sets how long a foreground `terminal` command may run before it is
    /// killed with its process tree. Defaults to [`DEFAULT_FOREGROUND_LIMIT`].
    #[must_use]
    pub const fn with_foreground_limit(mut self, limit: Duration) -> Self {
        self.limits.foreground = limit;
        self
    }

    /// Sets the largest `timeout` a `terminal_background` job accepts.
    /// Defaults to [`DEFAULT_BACKGROUND_TIMEOUT_CAP`].
    #[must_use]
    pub const fn with_background_timeout_cap(mut self, cap: Duration) -> Self {
        self.limits.background_timeout_cap = cap;
        self
    }

    /// Sets how long a background job may run before it produces one system
    /// alert naming it. Defaults to [`DEFAULT_LONG_JOB_ALERT_AFTER`].
    #[must_use]
    pub const fn with_long_job_alert_after(mut self, after: Duration) -> Self {
        self.limits.long_job_alert = after;
        self
    }

    /// Initializes tools before agent construction or command execution.
    ///
    /// # Errors
    /// Rejects preparation failures and writable grants that expose managed tools.
    pub async fn initialize_tools(
        mut self,
        distribution: &ToolDistribution,
        cache_root: &Path,
    ) -> Result<Self, ToolRuntimeError>
    where
        P: PermissionHandler,
        State: Send,
    {
        self.shell_sessions = self
            .shell_sessions
            .initialize_tools(distribution, cache_root)
            .await?;
        if let Some(tools) = self.local_home_tools()? {
            tools.ensure_readonly(self.writable_grants()).await?;
            tools.verify_local_versions(self.executor.clone()).await?;
        }
        Ok(self)
    }

    /// Adopts a registry that [`Self::initialize_tools`] already prepared for
    /// another terminal.
    ///
    /// Only the checks that depend on this terminal run again: its working
    /// directory and writable paths must not expose the managed tool
    /// directory. Artifact downloads, integrity hashing, version probes and SSH
    /// bootstrap — with its connection consent — are not repeated.
    ///
    /// # Errors
    /// Rejects a registry that is not prepared, one whose home backend,
    /// container or SSH server set changed after preparation, and writable
    /// grants that expose the managed tools.
    pub async fn adopt_prepared_tools(
        mut self,
        sessions: ShellSessionRegistry,
    ) -> Result<Self, ToolRuntimeError>
    where
        P: PermissionHandler,
        State: Send,
    {
        sessions.prepared_catalog()?;
        self.shell_sessions = sessions;
        if let Some(tools) = self.local_home_tools()? {
            tools.ensure_readonly(self.writable_grants()).await?;
        }
        Ok(self)
    }

    /// Prepared tools of the local home backend, when this terminal runs on it.
    fn local_home_tools(&self) -> Result<Option<&PreparedTools<PathBuf>>, ToolRuntimeError> {
        match self.shell_sessions.resolve_local_backend() {
            Ok(ShellBackend::Local) => self.shell_sessions.local_tools().map(Some),
            _ => Ok(None),
        }
    }

    /// The paths this terminal grants write access to, which must not expose
    /// the managed tools.
    fn writable_grants(&self) -> impl Iterator<Item = &PathBuf> + Send {
        std::iter::once(&self.working_dir).chain(self.writable_paths.iter())
    }

    /// Initialized session configuration shared with child terminals.
    #[must_use]
    pub const fn shell_sessions(&self) -> &ShellSessionRegistry {
        &self.shell_sessions
    }

    /// Operational context from successfully initialized environments.
    ///
    /// # Errors
    /// Fails when no tools are prepared or a backend configuration has changed.
    pub fn prepared_tools_context(&self) -> Result<String, ToolRuntimeError> {
        self.shell_sessions.prepared_catalog()?.render()
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
            alert_rx: self.alert_rx.clone(),
            started_rx: self.started_rx.clone(),
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

impl<P, E, State> TerminalTool<P, E, State>
where
    E: Executor + Clone + 'static,
    State: RegistryState,
{
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
        let (alert_tx, alert_rx) = async_channel::unbounded();
        let (started_tx, started_rx) = async_channel::unbounded();
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
            completed_rx,
            completed_tx,
            alert_rx,
            alert_tx,
            started_rx,
            started_tx,
            limits: self.limits,
            running_tasks,
            permission_rx,
            permission_tx,
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
            network_audit: self.network_audit.clone(),
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver.clone(),
            registry: self
                .registry
                .for_jobs(job_registry.clone(), self.outputs_dir()),
            job_registry,
        }
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
    ///
    /// # Panics
    /// Panics when execution environments have not completed tool initialization.
    pub fn to_dyn(self) -> crate::command::DynTerminalTool {
        use crate::command::DynTerminalTool;

        let background_receiver = self.background_receiver();
        let permission_receiver = self.permission_receiver();
        let job_registry = self.job_registry();
        let working_dir = self.working_dir().clone();
        let command_names = self.registry().registered_tool_names();
        let prepared_tools = self
            .shell_sessions
            .prepared_catalog()
            .expect("terminal tools must be initialized before exposing child capabilities")
            .clone();
        let background_tool = self.background_tool();
        let entries = vec![
            erase_terminal_tool(self),
            erase_terminal_tool(background_tool),
        ];

        DynTerminalTool {
            entries,
            background_receiver,
            permission_receiver,
            job_registry,
            working_dir,
            command_names,
            prepared_tools,
        }
    }

    fn resolve_execution_target(
        &self,
        mode: TerminalExecutionMode,
        ssh_server_id: Option<&str>,
    ) -> Result<ExecutionTarget, TerminalError> {
        match mode {
            TerminalExecutionMode::Default => self.resolve_default_target(),
            TerminalExecutionMode::Sandboxed => self.resolve_home_target(TerminalMode::Sandboxed),
            TerminalExecutionMode::Unsafe => self.resolve_home_target(TerminalMode::Unsafe),
            TerminalExecutionMode::Ssh => self.resolve_ssh_target(ssh_server_id),
        }
    }

    fn resolve_default_target(&self) -> Result<ExecutionTarget, TerminalError> {
        if self.shell_sessions.resolve_local_backend().is_ok() {
            self.resolve_home_target(TerminalMode::Sandboxed)
        } else {
            self.resolve_ssh_target(None)
        }
    }

    fn resolve_home_target(&self, mode: TerminalMode) -> Result<ExecutionTarget, TerminalError> {
        let backend = self
            .shell_sessions
            .resolve_local_backend()
            .map_err(TerminalError::InvalidTarget)?;
        if mode == TerminalMode::Unsafe && backend != ShellBackend::Local {
            return Err(TerminalError::InvalidTarget(
                "unsafe mode is only available in heel local runtime".into(),
            ));
        }
        let backend = match backend {
            ShellBackend::Local => {
                ExecutionBackend::Local(self.shell_sessions.local_tools()?.clone())
            }
            ShellBackend::Container => ExecutionBackend::Container {
                runtime: self
                    .shell_sessions
                    .container_runtime()
                    .cloned()
                    .ok_or(ToolRuntimeError::MissingContainerRuntime)?,
                tools: self.shell_sessions.container_tools()?.clone(),
            },
            ShellBackend::Ssh => unreachable!("the home backend is never SSH"),
        };
        Ok(ExecutionTarget { mode, backend })
    }

    fn resolve_ssh_target(
        &self,
        ssh_server_id: Option<&str>,
    ) -> Result<ExecutionTarget, TerminalError> {
        self.shell_sessions
            .ensure_ssh_available()
            .map_err(TerminalError::InvalidTarget)?;
        let server = ssh_server_id
            .map_or_else(
                || self.shell_sessions.default_ssh_server(),
                |id| self.shell_sessions.resolve_ssh_server(id),
            )
            .map_err(TerminalError::InvalidTarget)?;
        let (runtime, tools) = self.shell_sessions.ssh_tools(&server)?;
        Ok(ExecutionTarget {
            mode: TerminalMode::Sandboxed,
            backend: ExecutionBackend::Ssh {
                target: server.target,
                runtime: runtime.clone(),
                tools: tools.clone(),
            },
        })
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
            background,
        } = request;
        let ExecutionTarget { mode, backend } = target;
        let (result_tx, result_rx) = async_channel::bounded(1);
        let (startup_tx, startup_rx) = async_channel::bounded(1);
        let (stdin_blocked_tx, stdin_blocked_rx) = async_channel::bounded(1);
        // Only a foreground call reads the detector's notice: it becomes
        // guidance inside the kill result. A background job's stdin is
        // served by `terminal_input`.
        let stdin_blocked_notice = if background || !supports_stdin_blocked_notice(backend.kind()) {
            None
        } else {
            Some(stdin_blocked_tx)
        };

        let background_startup = BackgroundStartup::new(startup_tx);
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
                background,
                script,
                mode,
                backend,
                stdin_blocked_notice,
                background_startup,
                expect,
                resolution,
                store_dir: self.output_store.dir().to_path_buf(),
                max_lines,
                raw,
                result_tx,
                completed_tx,
                running_tasks,
                network_audit: self.network_audit.clone(),
                priority: self.priority,
                #[cfg(all(unix, feature = "jobserver"))]
                jobserver: self.jobserver.clone(),
            }))
            .detach();

        SpawnedTerminal {
            result: result_rx,
            startup: startup_rx,
            stdin_blocked_notice: stdin_blocked_rx,
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

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let prepared = self.prepare_execution(&arguments).await?;
        let spawned = self.spawn_terminal_execution(TerminalSpawnRequest {
            target: prepared.target,
            task_id: prepared.task_id.clone(),
            execution_id: prepared.execution_id,
            script: arguments.script,
            expect: arguments.expect,
            resolution: arguments.resolution,
            max_lines: prepared.max_lines,
            raw: arguments.raw,
            background: false,
        });
        // Registration precedes `report_ready` on every backend, so once
        // startup returns the job is killable through the registry and the
        // foreground clock only measures the command's own runtime.
        wait_for_background_startup(&spawned.startup).await?;
        match wait_for_foreground_result(&spawned.result, self.limits.foreground).await {
            Some(Ok(result)) => CommandResult::from(result).into_tool_result(),
            Some(Err(err)) => Err(anyhow::anyhow!(err)),
            None => {
                self.finish_killed_foreground(&spawned, &prepared.task_id)
                    .await
            }
        }
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> TerminalTool<P, E, Configured> {
    /// Returns the `terminal_background` native tool that starts background
    /// jobs on this terminal's sandbox and job registry.
    #[must_use]
    pub fn background_tool(&self) -> TerminalBackgroundTool<P, E> {
        TerminalBackgroundTool {
            terminal: self.clone(),
        }
    }

    /// Shared front half of `terminal` and `terminal_background`: argument
    /// validation, target resolution, permission and the spawned job's ids.
    async fn prepare_execution(
        &self,
        arguments: &TerminalArgs,
    ) -> anyhow::Result<ExecutionPreparation> {
        if arguments.description.trim().is_empty() {
            return Err(anyhow::anyhow!("terminal description must not be empty"));
        }
        let target =
            self.resolve_execution_target(arguments.mode, arguments.ssh_server_id.as_deref())?;
        if let ExecutionBackend::Local(tools) = &target.backend {
            tools
                .ensure_readonly(
                    std::iter::once(&self.working_dir).chain(self.writable_paths.iter()),
                )
                .await?;
        }
        self.ensure_permission_allowed(target.mode, &arguments.script)
            .await?;
        let task_id = random_task_id();
        Ok(ExecutionPreparation {
            target,
            execution_id: format!("exec-{task_id}"),
            task_id,
            store_dir: self.output_store.dir().to_path_buf(),
            max_lines: arguments.max_lines.min(MAX_LINES_CEILING) as usize,
        })
    }

    /// Starts `arguments` as a background job, arming its kill watchdog and
    /// the long-running alert.
    async fn run_background(
        &self,
        arguments: TerminalBackgroundArgs,
    ) -> aither_core::Result<ToolResult> {
        let cap = self.limits.background_timeout_cap;
        let timeout = Duration::from_secs(arguments.timeout);
        if arguments.timeout == 0 || timeout > cap {
            return Err(anyhow::anyhow!(
                "terminal_background timeout must be between 1 and {} seconds",
                cap.as_secs()
            ));
        }
        let prepared = self.prepare_execution(&arguments.terminal).await?;
        let spawned = self.spawn_terminal_execution(TerminalSpawnRequest {
            target: prepared.target,
            task_id: prepared.task_id.clone(),
            execution_id: prepared.execution_id,
            script: arguments.terminal.script,
            expect: arguments.terminal.expect,
            resolution: arguments.terminal.resolution,
            max_lines: prepared.max_lines,
            raw: arguments.terminal.raw,
            background: true,
        });
        wait_for_background_startup(&spawned.startup).await?;
        self.spawn_kill_watchdog(&prepared.task_id, timeout);
        self.spawn_long_job_alert(&prepared.task_id);
        background_terminal_result(
            &self.job_registry,
            &prepared.store_dir,
            &prepared.task_id,
            prepared.max_lines,
            &self.started_tx,
        )
        .await
    }

    /// Kills the job through the registry once its timeout passes.
    fn spawn_kill_watchdog(&self, task_id: &str, timeout: Duration) {
        let job_registry = self.job_registry.clone();
        let task_id = task_id.to_string();
        self.executor
            .spawn(async move {
                async_io::Timer::after(timeout).await;
                if let Some(job) = job_registry.get_by_task_id(&task_id).await
                    && matches!(job.status, JobStatus::Running)
                {
                    job_registry.kill(job.pid).await;
                }
            })
            .detach();
    }

    /// Emits one [`LongRunningJob`] when the job crosses the configured
    /// long-running threshold.
    fn spawn_long_job_alert(&self, task_id: &str) {
        let job_registry = self.job_registry.clone();
        let alert_tx = self.alert_tx.clone();
        let task_id = task_id.to_string();
        let alert_after = self.limits.long_job_alert;
        self.executor
            .spawn(async move {
                async_io::Timer::after(alert_after).await;
                if let Some(job) = job_registry.get_by_task_id(&task_id).await
                    && matches!(job.status, JobStatus::Running)
                {
                    let _ = alert_tx
                        .send(LongRunningJob {
                            task_id,
                            script: job.script,
                            elapsed: job.started_at.elapsed(),
                        })
                        .await;
                }
            })
            .detach();
    }

    /// Kills the registered job behind a foreground call whose limit expired
    /// and shapes the killed result the model receives.
    async fn finish_killed_foreground(
        &self,
        spawned: &SpawnedTerminal,
        task_id: &str,
    ) -> aither_core::Result<ToolResult> {
        let stdin_notice = spawned.stdin_blocked_notice.try_recv().ok();
        if !self.job_registry.kill_by_task_id(task_id).await
            && let Ok(result) = spawned.result.try_recv()
        {
            // The command finished between the limit expiring and the kill.
            return match result {
                Ok(result) => CommandResult::from(result).into_tool_result(),
                Err(err) => Err(anyhow::anyhow!(err)),
            };
        }
        let mut notice = format!(
            "Killed after running {}s in the foreground. Use terminal_background to run a command longer than that.",
            self.limits.foreground.as_secs()
        );
        if let Some(stdin_notice) = stdin_notice {
            notice = format!("{notice}\n{stdin_notice}");
        }
        let partial = spawned.result.recv().await.ok().and_then(Result::ok);
        let (stdout, stderr) = partial.map_or((empty_stream(), None), |partial| {
            (partial.stdout, partial.stderr)
        });
        CommandResult {
            stdout,
            stderr,
            outcome: CommandOutcome::Killed {
                task_id: task_id.to_string(),
                notice,
            },
        }
        .into_tool_result()
    }
}

/// Native tool that runs a command as a background job.
///
/// Wraps a configured [`TerminalTool`]: jobs share its sandbox, working
/// directory and job registry. Created by [`TerminalTool::background_tool`]
/// and registered next to `terminal`.
pub struct TerminalBackgroundTool<P, E> {
    terminal: TerminalTool<P, E, Configured>,
}

impl<P, E: Clone> Clone for TerminalBackgroundTool<P, E> {
    fn clone(&self) -> Self {
        Self {
            terminal: self.terminal.clone(),
        }
    }
}

impl<P, E> std::fmt::Debug for TerminalBackgroundTool<P, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalBackgroundTool")
            .finish_non_exhaustive()
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> Tool
    for TerminalBackgroundTool<P, E>
{
    fn name(&self) -> Cow<'static, str> {
        "terminal_background".into()
    }

    fn description(&self) -> Cow<'static, str> {
        format!(
            "Run a terminal command as a background job, detached from the \
             foreground call. Returns immediately with the job's task id; use \
             terminal_read for incremental output, terminal_input for stdin, \
             and terminal_kill to stop the job early. The required `timeout` \
             (seconds) is the job's maximum runtime — it and its process tree \
             are killed when it passes — and may be at most {} minutes. To \
             wait longer than that, schedule a wake-up (cron) rather than \
             starting a longer job.",
            self.terminal.limits.background_timeout_cap.as_secs() / 60
        )
        .into()
    }

    type Arguments = TerminalBackgroundArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        self.terminal.run_background(arguments).await
    }
}

struct ExecutionPreparation {
    target: ExecutionTarget,
    task_id: String,
    execution_id: String,
    store_dir: PathBuf,
    max_lines: usize,
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

/// Races the result channel against the foreground limit.
///
/// Returns `None` when the limit expired (or the channel was dropped, which
/// the caller treats the same way — the job is finished or killed).
async fn wait_for_foreground_result(
    result_rx: &Receiver<Result<CompletedCommand, String>>,
    limit: Duration,
) -> Option<Result<CompletedCommand, String>> {
    futures_lite::future::or(async { result_rx.recv().await.ok() }, async {
        async_io::Timer::after(limit).await;
        result_rx.try_recv().ok()
    })
    .await
}

async fn background_terminal_result(
    job_registry: &JobRegistry,
    store_dir: &Path,
    task_id: &str,
    max_lines: usize,
    started_tx: &Sender<StartedBackgroundTask>,
) -> aither_core::Result<ToolResult> {
    let redirect =
        start_background_output_redirect(job_registry, store_dir, task_id, max_lines).await?;
    let _ = started_tx
        .send(StartedBackgroundTask {
            task_id: task_id.to_string(),
            output_preview: redirect.preview.clone(),
            output_file: redirect.file.clone(),
        })
        .await;
    let stdout = if redirect.truncated {
        CommandStream::Stored {
            file: redirect.file.clone(),
            preview: redirect.preview,
            truncated: true,
        }
    } else {
        CommandStream::Inline(redirect.preview)
    };
    CommandResult {
        stdout,
        stderr: None,
        outcome: CommandOutcome::Running {
            task_id: task_id.to_string(),
            output: redirect.file,
        },
    }
    .into_tool_result()
}

/// Output file a background job was redirected to, with what it holds so far.
struct BackgroundRedirect {
    file: StoredFile,
    /// First lines already written; empty when nothing was.
    preview: String,
    truncated: bool,
}

async fn start_background_output_redirect(
    job_registry: &JobRegistry,
    store_dir: &Path,
    task_id: &str,
    max_lines: usize,
) -> Result<BackgroundRedirect, anyhow::Error> {
    let (url, output_path) = save_raw_to_file(store_dir, &[]).await?;
    let file = StoredFile {
        url,
        path: output_path.display().to_string(),
    };
    let snapshot = job_registry
        .start_output_redirect(task_id, output_path)
        .await
        .map_err(anyhow::Error::msg)?;
    let (preview, truncated) = preview_first_lines(&snapshot, max_lines);
    Ok(BackgroundRedirect {
        file,
        preview: background_preview(&preview),
        truncated,
    })
}

/// The preview of what a background job has written so far: empty when it
/// wrote only whitespace, otherwise the text with its trailing whitespace cut
/// and the first line's indentation kept.
fn background_preview(preview: &str) -> String {
    if preview.trim().is_empty() {
        String::new()
    } else {
        preview.trim_end().to_string()
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
    background: bool,
    script: String,
    mode: TerminalMode,
    backend: ExecutionBackend,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: BackgroundStartup,
    network_audit: Option<NetworkAuditLog>,
    priority: ProcessPriority,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
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
    backend: &'a ExecutionBackend,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: &'a BackgroundStartup,
    network_audit: Option<NetworkAuditLog>,
    priority: ProcessPriority,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: &'a Option<JobBroker>,
}

struct LocalBackgroundExecution<'a, E> {
    prepared_tools: &'a PreparedTools<PathBuf>,
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
    priority: ProcessPriority,
    #[cfg(all(unix, feature = "jobserver"))]
    job_fifo: Option<crate::jobserver::CommandFifo>,
}

struct ContainerBackgroundExecution<'a, E> {
    executor: E,
    registry: &'a Arc<ToolRegistry>,
    task_id: &'a str,
    execution_id: &'a str,
    script: &'a str,
    mode: TerminalMode,
    container_runtime: &'a ContainerShellRuntime,
    prepared_tools: &'a PreparedTools<ContainerPath>,
    ipc_commands: &'a [String],
    job_registry: &'a JobRegistry,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: &'a BackgroundStartup,
    priority: ProcessPriority,
}

struct SshBackgroundExecution<'a> {
    prepared_tools: &'a PreparedTools<PathBuf>,
    task_id: &'a str,
    execution_id: &'a str,
    script: &'a str,
    mode: TerminalMode,
    ssh_target: &'a str,
    ssh_runtime: &'a SshRuntimeProfile,
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

/// Where a command runs, with the tools prepared for exactly that environment.
enum ExecutionBackend {
    /// The local sandbox, whose tools live at a host path.
    Local(PreparedTools<PathBuf>),
    /// A container, whose tools are mounted at a container path.
    Container {
        runtime: ContainerShellRuntime,
        tools: PreparedTools<ContainerPath>,
    },
    /// A remote host over SSH.
    Ssh {
        target: String,
        runtime: SshRuntimeProfile,
        tools: PreparedTools<PathBuf>,
    },
}

impl ExecutionBackend {
    const fn kind(&self) -> ShellBackend {
        match self {
            Self::Local(_) => ShellBackend::Local,
            Self::Container { .. } => ShellBackend::Container,
            Self::Ssh { .. } => ShellBackend::Ssh,
        }
    }
}

struct ExecutionTarget {
    mode: TerminalMode,
    backend: ExecutionBackend,
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
    background: bool,
}

struct SpawnedTerminal {
    result: Receiver<Result<CompletedCommand, String>>,
    startup: Receiver<Result<(), String>>,
    stdin_blocked_notice: Receiver<String>,
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
    background: bool,
    script: String,
    mode: TerminalMode,
    backend: ExecutionBackend,
    stdin_blocked_notice: Option<Sender<String>>,
    background_startup: BackgroundStartup,
    expect: OutputFormat,
    resolution: MediaResolution,
    store_dir: PathBuf,
    max_lines: usize,
    raw: bool,
    result_tx: Sender<Result<CompletedCommand, String>>,
    completed_tx: Sender<CompletedTask>,
    running_tasks: Arc<AtomicUsize>,
    network_audit: Option<NetworkAuditLog>,
    priority: ProcessPriority,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
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
        background,
        script,
        mode,
        backend,
        stdin_blocked_notice,
        background_startup,
        expect,
        resolution,
        store_dir,
        max_lines,
        raw,
        result_tx,
        completed_tx,
        running_tasks,
        network_audit,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
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
        background,
        script: script.clone(),
        mode,
        backend,
        stdin_blocked_notice,
        background_startup,
        network_audit,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
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
    if background {
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
) -> Result<CompletedCommand, TerminalError>
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
        background,
        script,
        mode,
        backend,
        stdin_blocked_notice,
        background_startup,
        network_audit,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
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
        backend: &backend,
        stdin_blocked_notice,
        background_startup: &background_startup,
        network_audit,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver: &jobserver,
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
        background_output: background,
        store_dir: &store_dir,
        expect,
        resolution,
        max_lines,
        raw,
    })
    .await?;
    let exit_code = result.exit_code;

    debug!(exit_code, "background script completed");

    let output_path = stored_path(&result.stdout);
    job_registry.complete(pid, exit_code, output_path).await;

    Ok(result)
}

/// Creates the private jobserver fifo for one command under the broker.
///
/// Only local commands can open a host fifo. Container and SSH executions
/// fail rather than running unbounded while a pool is configured.
#[cfg(all(unix, feature = "jobserver"))]
async fn open_command_fifo(
    broker: &JobBroker,
    backend: &ExecutionBackend,
) -> Result<crate::jobserver::CommandFifo, TerminalError> {
    match backend {
        ExecutionBackend::Local(_) => Ok(broker.open_fifo().await?),
        ExecutionBackend::Container { .. } | ExecutionBackend::Ssh { .. } => {
            Err(TerminalError::Execution(
                "jobserver token pools are supported for local executions only".into(),
            ))
        }
    }
}

async fn start_standalone_execution<P, E>(
    request: StandaloneStart<'_, P, E>,
) -> Result<(u32, std::process::Output), TerminalError>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    #[cfg(all(unix, feature = "jobserver"))]
    let job_fifo = match request.jobserver {
        Some(broker) => Some(open_command_fifo(broker, request.backend).await?),
        None => None,
    };
    let prepared_tools = match request.backend {
        ExecutionBackend::Local(tools) => tools,
        ExecutionBackend::Container { runtime, tools } => {
            let ipc_commands = request.registry.registered_tool_names();
            return execute_container_background(ContainerBackgroundExecution {
                executor: request.executor,
                registry: request.registry,
                task_id: request.task_id,
                execution_id: request.execution_id,
                script: request.script,
                mode: request.mode,
                container_runtime: runtime,
                prepared_tools: tools,
                ipc_commands: &ipc_commands,
                job_registry: request.job_registry,
                stdin_blocked_notice: request.stdin_blocked_notice,
                background_startup: request.background_startup,
                priority: request.priority,
            })
            .await;
        }
        ExecutionBackend::Ssh {
            target,
            runtime,
            tools,
        } => {
            return execute_ssh_background(SshBackgroundExecution {
                prepared_tools: tools,
                task_id: request.task_id,
                execution_id: request.execution_id,
                script: request.script,
                mode: request.mode,
                ssh_target: target,
                ssh_runtime: runtime,
                job_registry: request.job_registry,
                background_startup: request.background_startup,
            })
            .await;
        }
    };
    let local = LocalBackgroundExecution {
        prepared_tools,
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
        priority: request.priority,
        #[cfg(all(unix, feature = "jobserver"))]
        job_fifo,
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
) -> Result<CompletedCommand, TerminalError> {
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

    Ok(CompletedCommand {
        stdout,
        stderr,
        exit_code,
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
) -> Result<CommandStream, TerminalError> {
    let byte_limit = Some(INLINE_OUTPUT_LIMIT);
    let data_to_save = if format.background_output {
        Cow::Borrowed(stdout)
    } else {
        compressed_stdout_data(format.store_dir, stdout, format.expect, format.raw).await?
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

/// Runs the output cleaning pipeline over a completed command's stdout.
///
/// `raw: true` bypasses it entirely, as do non-text formats and output that
/// is not valid UTF-8, so the model sees exactly the bytes the command wrote.
async fn compressed_stdout_data<'a>(
    store_dir: &Path,
    stdout: &'a [u8],
    expect: OutputFormat,
    raw: bool,
) -> Result<Cow<'a, [u8]>, TerminalError> {
    let is_text = matches!(expect, OutputFormat::Text | OutputFormat::Auto);
    if raw || !is_text || stdout.is_empty() {
        return Ok(Cow::Borrowed(stdout));
    }
    let Ok(text) = std::str::from_utf8(stdout) else {
        return Ok(Cow::Borrowed(stdout));
    };

    let cleaned = crate::output_compress::compress_text(store_dir, text)
        .await
        .map_err(TerminalError::Io)?;
    Ok(cleaned.map_or(Cow::Borrowed(stdout), |cleaned| {
        Cow::Owned(cleaned.into_bytes())
    }))
}

async fn save_completed_stderr(
    job_registry: &JobRegistry,
    pid: u32,
    store_dir: &Path,
    stderr: &[u8],
) -> Result<Option<CommandStream>, TerminalError> {
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

fn local_path_grants<'a>(
    writable: &'a [PathBuf],
    readable: &'a [PathBuf],
    tools: &'a PreparedTools<PathBuf>,
    job_fifo: Option<&'a Path>,
) -> impl Iterator<Item = Grant> + 'a {
    writable
        .iter()
        .map(|path| Grant::new(path, Access::WRITE))
        .chain(readable.iter().map(|path| Grant::new(path, Access::READ)))
        .chain(std::iter::once(Grant::new(tools.bin_dir(), Access::EXEC)))
        .chain(job_fifo.map(|path| Grant::new(path, Access::WRITE)))
}

/// Adds a command's private jobserver variables to its process environment.
#[cfg(all(unix, feature = "jobserver"))]
fn with_jobserver_environment(
    mut environment: Vec<(String, String)>,
    fifo: Option<&crate::jobserver::CommandFifo>,
) -> Vec<(String, String)> {
    if let Some(fifo) = fifo {
        environment.extend(fifo.environment());
    }
    environment
}

async fn execute_sandboxed_background<E, N>(
    context: LocalBackgroundExecution<'_, E>,
    policy: N,
) -> Result<(u32, std::process::Output), TerminalError>
where
    E: Executor + Clone + 'static,
    N: NetworkPolicy + 'static,
{
    let runtime_environment =
        prepare_session_runtime_environment(context.working_dir, context.prepared_tools).await?;
    #[cfg(all(unix, feature = "jobserver"))]
    let runtime_environment =
        with_jobserver_environment(runtime_environment, context.job_fifo.as_ref());
    #[cfg(all(unix, feature = "jobserver"))]
    let job_fifo_path = context
        .job_fifo
        .as_ref()
        .map(crate::jobserver::CommandFifo::path);
    #[cfg(not(all(unix, feature = "jobserver")))]
    let job_fifo_path: Option<&Path> = None;
    let config_builder = SandboxConfig::builder()
        .network(policy)
        .working_dir(context.working_dir)
        .grants(local_path_grants(
            context.writable_paths,
            context.readable_paths,
            context.prepared_tools,
            job_fifo_path,
        ))
        .security(SecurityConfig::interactive())
        .priority(context.priority);
    // `ipc` changes the builder's type, so both arms build where they are.
    let config = match create_ipc_router(context.registry) {
        Some(router) => config_builder.ipc(router).build(),
        None => config_builder.build(),
    };

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
    let runtime_environment =
        prepare_session_runtime_environment(context.working_dir, context.prepared_tools).await?;
    #[cfg(all(unix, feature = "jobserver"))]
    let runtime_environment =
        with_jobserver_environment(runtime_environment, context.job_fifo.as_ref());
    #[cfg(all(unix, feature = "jobserver"))]
    let job_fifo_path = context
        .job_fifo
        .as_ref()
        .map(crate::jobserver::CommandFifo::path);
    #[cfg(not(all(unix, feature = "jobserver")))]
    let job_fifo_path: Option<&Path> = None;
    let config_builder = SandboxConfig::builder()
        .network(AllowAll)
        .working_dir(context.working_dir)
        .grants(local_path_grants(
            context.writable_paths,
            context.readable_paths,
            context.prepared_tools,
            job_fifo_path,
        ))
        .security(SecurityConfig::interactive())
        .priority(context.priority);
    // `ipc` changes the builder's type, so both arms build where they are.
    let config = match create_ipc_gateway_router(context.registry) {
        Some(router) => config_builder.ipc(router).build(),
        None => config_builder.build(),
    };

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
    let container_runtime = context.container_runtime;
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
    context.job_registry.attach_kill_switch(pid, kill_tx).await;
    context.background_startup.report_ready().await;

    let ipc_bridge = if context.ipc_commands.is_empty() {
        None
    } else {
        Some(start_container_ipc_bridge(
            context.executor,
            Arc::clone(context.registry),
            container_runtime.ipc_route().bind(),
        )?)
    };
    let script = context.prepared_tools.wrap_script(context.script)?;
    let wrapped_script = wrap_container_script(
        &script,
        context.ipc_commands,
        ipc_bridge.as_ref().map(|bridge| ContainerIpcEndpoint {
            host: container_runtime.ipc_route().host(),
            port: bridge.port(),
        }),
        context.priority,
        context.prepared_tools.bin_dir(),
        CONTAINER_EXEC_WORKING_DIR,
    )?;

    let execution = exec
        .exec_boxed(
            container_runtime.container_id(),
            &wrapped_script,
            CONTAINER_EXEC_WORKING_DIR,
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

#[derive(Template)]
#[template(path = "ssh_tool_execution.sh", escape = "none")]
struct SshToolExecutionTemplate<'a> {
    binary: Cow<'a, str>,
    tools_dir: Cow<'a, str>,
    program: Cow<'a, str>,
    arguments: String,
}

async fn execute_ssh_background(
    context: SshBackgroundExecution<'_>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let target = context.ssh_target;
    let remote_cmd = match (context.ssh_runtime, context.mode) {
        (SshRuntimeProfile::Heel { binary }, TerminalMode::Sandboxed) => {
            let script = context.prepared_tools.wrap_script(context.script)?;
            let (program, args) = shell_launch(&script);
            let directory = context.prepared_tools.script_bin_dir()?;
            SshToolExecutionTemplate {
                binary: shell_words::quote(binary),
                tools_dir: shell_words::quote(directory),
                program: shell_words::quote(&program),
                arguments: shell_words::join(&args),
            }
            .render()
            .map_err(ToolRuntimeError::from)?
        }
        (SshRuntimeProfile::Heel { .. }, TerminalMode::Unsafe) => {
            return Err(TerminalError::InvalidTarget(
                "unsafe mode is not supported for ssh backend".into(),
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
    tools: &PreparedTools<PathBuf>,
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

    let sandbox_path = heel::default_sandbox_path();
    let path = tools
        .prepend_path(Some(std::ffi::OsStr::new(&sandbox_path)))?
        .into_string()
        .map_err(|_| ToolRuntimeError::InvalidCache(tools.bin_dir().clone()))?;
    Ok(vec![
        ("PATH".to_string(), path),
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

#[derive(Clone, Copy)]
struct ContainerIpcEndpoint<'a> {
    host: &'a str,
    port: u16,
}

/// Working directory container executions run their scripts in.
const CONTAINER_EXEC_WORKING_DIR: &str = "/workspace";

#[derive(Template)]
#[template(path = "container_heel_run.sh", escape = "none")]
struct ContainerHeelRunTemplate<'a> {
    heel_bin: &'a str,
    resolve_heel: bool,
    escaped_working_dir: &'a str,
    escaped_tools_dir: &'a str,
    escaped_ipc_endpoint: Option<String>,
    nice: Option<u8>,
    io_priority: Option<&'static str>,
    escaped_script: &'a str,
}

/// Renders the command that runs `script` inside the container at the
/// configured scheduling priority.
///
/// The priority goes through heel's own sandbox configuration: the container
/// image's `heel` binary runs the script under `heel run`, so the settings are
/// the same flags the CLI takes everywhere — `nice`/`ionice` are never
/// reimplemented here. `heel_bin` is the shell expression that resolves the
/// image's `heel` (`$MAY_IPC_BIN` once the IPC wrapper has resolved it, or
/// `$MAY_HEEL_BIN` the template resolves itself when `resolve_heel` is set)
/// and `working_dir` is the exact directory the execution runs in, matching
/// the directory the container exec applies.
///
/// The container is already the isolation boundary, so the invocation asks for
/// `permissive` isolation: the script keeps the filesystem permissions it had
/// as the container caller. The ruleset still exec-denies everything outside
/// the system directories, so `--executable` grants the mounted tools
/// directory read-execute — without it every prepared tool fails with
/// Permission denied. `PATH` and `HOME` are forwarded so the generated
/// `heel ipc` shims still resolve and the script sees its usual environment.
/// Networking keeps heel's deny-all default, and no rule is bent to pass the
/// IPC transport: `--ipc-endpoint` has heel relay the sandbox's local IPC
/// socket to the bridge endpoint from outside the sandbox, so the sandboxed
/// `heel ipc` never needs TCP access of its own.
fn heel_run_container_command(
    script: &str,
    priority: ProcessPriority,
    heel_bin: &str,
    resolve_heel: bool,
    ipc_endpoint: Option<ContainerIpcEndpoint<'_>>,
    tools_dir: &ContainerPath,
    working_dir: &str,
) -> Result<String, TerminalError> {
    let nice = match priority.cpu() {
        CpuPriority::Nice(level) => Some(level.get()),
        CpuPriority::Normal => None,
    };
    let io_priority = match priority.io() {
        IoPriority::Utility => Some("utility"),
        IoPriority::Background => Some("background"),
        IoPriority::Normal => None,
    };
    let escaped_working_dir = shell_escape(working_dir);
    let escaped_tools_dir = shell_escape(tools_dir.as_str());
    let escaped_ipc_endpoint = ipc_endpoint
        .map(|endpoint| format!("tcp://{}:{}", endpoint.host, endpoint.port))
        .map(|endpoint| shell_escape(&endpoint));
    let escaped_script = shell_escape(script);
    ContainerHeelRunTemplate {
        heel_bin,
        resolve_heel,
        escaped_working_dir: &escaped_working_dir,
        escaped_tools_dir: &escaped_tools_dir,
        escaped_ipc_endpoint,
        nice,
        io_priority,
        escaped_script: &escaped_script,
    }
    .render()
    .map_err(|error| {
        TerminalError::Execution(format!("failed to render container heel command: {error}"))
    })
}

fn prioritized_container_script(
    script: &str,
    priority: ProcessPriority,
    tools_dir: &ContainerPath,
    working_dir: &str,
) -> Result<String, TerminalError> {
    if priority.is_normal() {
        return Ok(script.to_string());
    }
    // Outside the IPC wrapper no `heel` binary has been resolved yet; the
    // template resolves it the same way the wrapper does and fails when the
    // image lacks one.
    heel_run_container_command(
        script,
        priority,
        "\"$MAY_HEEL_BIN\"",
        true,
        None,
        tools_dir,
        working_dir,
    )
}

fn wrap_container_script(
    script: &str,
    ipc_commands: &[String],
    ipc_endpoint: Option<ContainerIpcEndpoint<'_>>,
    priority: ProcessPriority,
    tools_dir: &ContainerPath,
    working_dir: &str,
) -> Result<String, TerminalError> {
    if ipc_commands.is_empty() {
        return prioritized_container_script(script, priority, tools_dir, working_dir);
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
            return Err(TerminalError::Execution(format!(
                "invalid ipc command name for container wrapper: {name}"
            )));
        }
    }

    let escaped_commands = ipc_commands
        .iter()
        .map(|name| shell_escape(name))
        .collect::<Vec<_>>()
        .join(" ");
    let script = if priority.is_normal() {
        script.to_string()
    } else {
        // The generated shims are written under the shared temp directory,
        // which stays execute-denied even at `permissive` isolation: the
        // render below grants them read-and-execute so the sandboxed `PATH`
        // resolves them — the wrapper writes them before `heel run`, so the
        // sandbox itself never writes the directory.
        heel_run_container_command(
            script,
            priority,
            "\"$MAY_IPC_BIN\"",
            false,
            Some(endpoint),
            tools_dir,
            working_dir,
        )?
    };
    ContainerIpcWrapperTemplate {
        escaped_commands: &escaped_commands,
        ipc_host: endpoint.host,
        ipc_port: endpoint.port,
        script: &script,
    }
    .render()
    .map_err(|error| {
        TerminalError::Execution(format!("failed to render container IPC wrapper: {error}"))
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

fn start_container_ipc_bridge<E: Executor + Clone + 'static>(
    executor: E,
    registry: Arc<ToolRegistry>,
    bind: BridgeAddress,
) -> Result<ContainerIpcBridge, TerminalError> {
    // The bridge executes host IPC commands for whoever connects, so it binds
    // only the address the container route names — never a wildcard.
    let listener = StdTcpListener::bind(SocketAddr::new(bind.ip(), 0)).map_err(|e| {
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

enum ContainerIpcBridgeEvent {
    Accept(std::io::Result<(Async<StdTcpStream>, std::net::SocketAddr)>),
    Shutdown,
}

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
        let envelope = registry.query_tool_handler(method, &cli_args).await;
        match crate::command::envelope_to_cli(envelope) {
            Ok(rendered) => write_container_ipc_success(&mut stream, &rendered).await?,
            Err(error) => {
                write_container_ipc_error(&mut stream, &error.to_string()).await?;
            }
        }
    }
}

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

async fn write_container_ipc_success(
    stream: &mut Async<StdTcpStream>,
    rendered: &str,
) -> Result<(), String> {
    let payload = rmp_serde::to_vec_named(rendered)
        .map_err(|e| format!("failed to encode IPC success payload: {e}"))?;
    write_container_ipc_response(stream, true, &payload).await
}

async fn write_container_ipc_error(
    stream: &mut Async<StdTcpStream>,
    message: &str,
) -> Result<(), String> {
    let payload = rmp_serde::to_vec_named(&message.to_string())
        .map_err(|e| format!("failed to encode IPC error payload: {e}"))?;
    write_container_ipc_response(stream, false, &payload).await
}

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

    // In unsafe mode, keep tool commands usable (websearch/webfetch/ask/subagent...),
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
    /// Execution environment or mode is not configured.
    #[error("invalid terminal execution target: {0}")]
    InvalidTarget(String),
    /// Required tool runtime is unavailable or unsafe to expose.
    #[error(transparent)]
    Tools(#[from] ToolRuntimeError),
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

    /// Jobserver pool or broker error.
    #[cfg(all(unix, feature = "jobserver"))]
    #[error(transparent)]
    Jobserver(#[from] crate::jobserver::JobserverError),
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
    use heel::{DenyAll, NiceLevel};

    async fn prepared_test_registry() -> (tempfile::TempDir, ShellSessionRegistry) {
        let cache = tempfile::tempdir().unwrap();
        let tools = crate::tool_runtime::tests::bundle()
            .prepare_local(cache.path())
            .await
            .unwrap();
        let registry = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: true,
            container: false,
            ssh: false,
        })
        .with_test_tools(tools);
        (cache, registry)
    }

    /// Counts the consent prompts an SSH authorizer receives and denies each.
    #[derive(Default)]
    struct CountingSshAuthorizer {
        connects: AtomicUsize,
        installs: AtomicUsize,
    }

    impl crate::SshSessionAuthorizer for CountingSshAuthorizer {
        fn authorize_connect(
            &self,
            _target: &str,
        ) -> impl std::future::Future<Output = Result<bool, crate::SshAuthorizationError>> + Send
        {
            self.connects.fetch_add(1, AtomicOrdering::SeqCst);
            std::future::ready(Ok(false))
        }

        fn authorize_runtime_install(
            &self,
            _target: &str,
            _details: &str,
        ) -> impl std::future::Future<Output = Result<bool, crate::SshAuthorizationError>> + Send
        {
            self.installs.fetch_add(1, AtomicOrdering::SeqCst);
            std::future::ready(Ok(false))
        }
    }

    /// A container executor for registries whose container is never reached.
    struct UnreachableContainerExec;

    impl crate::ContainerExec for UnreachableContainerExec {
        fn exec(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _stdin_blocked_notice: Option<Sender<String>>,
        ) -> impl std::future::Future<Output = Result<crate::ContainerExecOutcome, String>> + Send
        {
            std::future::ready(Err("adoption never executes in the container".to_string()))
        }
    }

    fn test_container_runtime(container_id: &str) -> ContainerShellRuntime {
        ContainerShellRuntime::new(
            container_id,
            crate::ContainerIpcRoute::new(
                "host.docker.internal",
                crate::BridgeAddress::try_from(std::net::IpAddr::from([127, 0, 0, 1]))
                    .expect("loopback binds the bridge"),
            ),
            crate::AnyContainerExec::new(Arc::new(UnreachableContainerExec)),
        )
    }

    fn build_server() -> crate::SshServer {
        crate::SshServer {
            name: "build".to_string(),
            target: "builder@build.internal".to_string(),
        }
    }

    /// Local and SSH configuration gated by `authorizer`, not yet prepared.
    fn local_and_ssh_registry(authorizer: &Arc<CountingSshAuthorizer>) -> ShellSessionRegistry {
        ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: true,
            container: false,
            ssh: true,
        })
        .with_ssh_servers(vec![build_server()])
        .expect("ssh server should configure")
        .with_ssh_authorizer(Arc::clone(authorizer))
    }

    async fn fresh_terminal(
        dir: &Path,
    ) -> TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> {
        TerminalTool::new_exact(
            dir,
            TestPermissionHandler::default(),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("terminal tool should initialize")
    }

    /// A trusted manifest serving `script` for every managed tool at version 1.0.0.
    #[cfg(unix)]
    async fn script_distribution(directory: &Path, script: &[u8]) -> ToolDistribution {
        let path = directory.join("baseline-tool");
        async_fs::write(&path, script).await.unwrap();
        let artifacts = crate::ManagedTool::ALL
            .into_iter()
            .map(|tool| crate::ToolArtifact {
                tool,
                version: "1.0.0".into(),
                target: crate::ToolTarget::native().unwrap(),
                url: crate::ToolArtifactLocation::Download(
                    url::Url::from_file_path(&path).unwrap(),
                ),
                sha256: hex::encode(crate::tool_runtime::digest(script)),
                encoding: crate::ToolArtifactFormat::Executable,
            })
            .collect::<Vec<_>>();
        ToolDistribution::from_json(
            &serde_json::to_vec(&serde_json::json!({ "artifacts": artifacts })).unwrap(),
        )
        .unwrap()
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn adopted_registry_runs_baseline_tools_without_ssh_consent_or_version_probes() {
        let authorizer = Arc::new(CountingSshAuthorizer::default());

        // Full preparation asks the authorizer before connecting to every
        // server; the denial ends it.
        let artifacts = tempfile::tempdir().unwrap();
        let distribution = script_distribution(
            artifacts.path(),
            b"#!/bin/sh\nprintf '%s\\n' 'tool 1.0.0'\n",
        )
        .await;
        let first = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let denied = fresh_terminal(first.path())
            .await
            .with_shell_sessions(local_and_ssh_registry(&authorizer))
            .initialize_tools(&distribution, cache.path())
            .await;
        assert!(
            matches!(denied, Err(ToolRuntimeError::SshBootstrap(_))),
            "a denied connection must fail preparation"
        );
        assert_eq!(authorizer.connects.load(AtomicOrdering::SeqCst), 1);

        // Tools pinned to a version their commands do not report: the version
        // probe of full preparation rejects them.
        let tools_cache = tempfile::tempdir().unwrap();
        let tools = crate::tool_runtime::tests::bundle_pinned("2.0.0")
            .prepare_local(tools_cache.path())
            .await
            .unwrap();
        assert!(matches!(
            tools
                .verify_local_versions(executor_core::tokio::TokioGlobal)
                .await,
            Err(ToolRuntimeError::VersionMismatch(_))
        ));
        let prepared = local_and_ssh_registry(&authorizer)
            .with_test_tools(tools.clone())
            .with_test_ssh(build_server(), tools);

        let second = tempfile::tempdir().unwrap();
        let terminal = fresh_terminal(second.path())
            .await
            .adopt_prepared_tools(prepared)
            .await
            .expect("a prepared registry should be adopted without probing or consent");
        assert_eq!(authorizer.connects.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(authorizer.installs.load(AtomicOrdering::SeqCst), 0);

        let registry = Arc::new(ToolRegistryBuilder::new().build(terminal.outputs_dir()));
        let terminal = terminal.with_registry(registry);
        let result = terminal
            .call(
                TerminalArgs {
                    description: "run every baseline command".to_string(),
                    script: "apply_patch --version && rg --version && ast-grep --version && jq --version"
                        .to_string(),
                    mode: TerminalExecutionMode::Sandboxed,
                    ssh_server_id: None,
                    expect: OutputFormat::Text,
                    resolution: MediaResolution::Auto,
                    max_lines: 50,
                    raw: true,
                },
                ToolContext::new(),
            )
            .await
            .expect("baseline commands should run");
        let stdout = render_terminal_result(&result);
        assert!(!stdout.contains("Exit code:"), "{stdout}");
        for command in ["apply_patch", "rg", "ast-grep", "jq"] {
            assert!(
                stdout.contains(&format!("{command} 1.0.0")),
                "{command} should run from the adopted tools; stdout: {stdout}"
            );
        }
        assert_eq!(authorizer.connects.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(authorizer.installs.load(AtomicOrdering::SeqCst), 0);
    }

    #[tokio::test]
    async fn adoption_rejects_a_registry_changed_after_preparation() {
        let authorizer = Arc::new(CountingSshAuthorizer::default());
        let tools_cache = tempfile::tempdir().unwrap();
        let tools = crate::tool_runtime::tests::bundle()
            .prepare_local(tools_cache.path())
            .await
            .unwrap();
        let prepared = local_and_ssh_registry(&authorizer)
            .with_test_tools(tools.clone())
            .with_test_ssh(build_server(), tools.clone());
        let staging = crate::SshServer {
            name: "staging".to_string(),
            target: "deploy@staging.internal".to_string(),
        };
        let retargeted = crate::SshServer {
            target: "builder@build-2.internal".to_string(),
            ..build_server()
        };
        let container = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
            ssh: false,
        })
        .with_container_runtime(test_container_runtime("aither-tools-a"))
        .with_test_container_tools(
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::bundle()),
            "aither-tools-a".to_string(),
        );

        let dir = tempfile::tempdir().unwrap();
        let adopt = |sessions: ShellSessionRegistry| {
            let dir = dir.path().to_path_buf();
            async move {
                fresh_terminal(&dir)
                    .await
                    .adopt_prepared_tools(sessions)
                    .await
            }
        };

        assert!(adopt(prepared.clone()).await.is_ok());
        assert!(adopt(container.clone()).await.is_ok());
        assert!(matches!(
            adopt(local_and_ssh_registry(&authorizer)).await,
            Err(ToolRuntimeError::NotInitialized(_))
        ));
        for servers in [vec![build_server(), staging], Vec::new(), vec![retargeted]] {
            assert!(matches!(
                adopt(prepared.clone().with_ssh_servers(servers).unwrap()).await,
                Err(ToolRuntimeError::EnvironmentChanged(
                    crate::PreparedEnvironment::SshServers
                ))
            ));
        }
        assert!(matches!(
            adopt(container.with_container_runtime(test_container_runtime("aither-tools-b"))).await,
            Err(ToolRuntimeError::EnvironmentChanged(
                crate::PreparedEnvironment::Container
            ))
        ));
        assert!(matches!(
            adopt(prepared.with_availability(ShellRuntimeAvailability {
                local: true,
                container: true,
                ssh: true,
            }))
            .await,
            Err(ToolRuntimeError::EnvironmentChanged(
                crate::PreparedEnvironment::HomeBackend
            ))
        ));
        assert_eq!(authorizer.connects.load(AtomicOrdering::SeqCst), 0);
    }

    #[tokio::test]
    async fn adoption_rechecks_this_terminals_writable_paths() {
        let (cache, prepared) = prepared_test_registry().await;
        let dir = tempfile::tempdir().unwrap();
        let exposed = fresh_terminal(dir.path())
            .await
            .with_writable_paths([cache.path()])
            .adopt_prepared_tools(prepared)
            .await;
        assert!(matches!(
            exposed,
            Err(ToolRuntimeError::WritableToolDirectory(_))
        ));
    }

    fn render_terminal_result(result: &ToolResult) -> String {
        result.render_for_model()
    }

    fn started_task_id(receiver: &BackgroundTaskReceiver) -> String {
        let started = receiver.take_started();
        assert_eq!(started.len(), 1, "expected one started-task event");
        started[0].task_id.clone()
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
        let cache = tempfile::tempdir().unwrap();
        let ready = crate::tool_runtime::tests::bundle()
            .prepare_local(cache.path())
            .await
            .unwrap();
        let environment = prepare_session_runtime_environment(dir.path(), &ready)
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

    fn output_text(output: &CommandStream) -> Option<&str> {
        match output {
            CommandStream::Inline(text) => Some(text.as_str()),
            CommandStream::Stored { preview, .. } => Some(preview.as_str()),
            CommandStream::Image { .. } => None,
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
        let args: TerminalArgs =
            serde_json::from_str(r#"{"description":"print a greeting","script":"echo hello"}"#)
                .unwrap();
        assert_eq!(args.expect, OutputFormat::Text);
        assert_eq!(args.resolution, MediaResolution::Auto);
        assert_eq!(args.mode, TerminalExecutionMode::Default);
    }

    #[test]
    fn background_preview_keeps_leading_indentation() {
        assert_eq!(
            background_preview("  col a  col b\nrow\n\n"),
            "  col a  col b\nrow"
        );
        assert_eq!(background_preview(" \n\t\n"), "");
    }

    fn inline(text: &str) -> CommandStream {
        CommandStream::Inline(text.to_string())
    }

    fn completed(
        stdout: CommandStream,
        stderr: Option<CommandStream>,
        exit_code: i32,
    ) -> CommandResult {
        CompletedCommand {
            stdout,
            stderr,
            exit_code,
        }
        .into()
    }

    #[test]
    fn completed_command_exit_zero_is_the_bare_output() {
        let result = completed(inline("done"), None, 0)
            .into_tool_result()
            .unwrap();
        assert_eq!(result.render_for_model(), "done\n");
        assert_eq!(
            result.command().map(|command| &command.outcome),
            Some(&CommandOutcome::Exited { code: 0 })
        );
    }

    #[test]
    fn completed_command_without_any_output_says_so() {
        assert_eq!(
            completed(empty_stream(), None, 0).render_for_model(),
            "(no output)\n"
        );
    }

    #[test]
    fn completed_command_non_zero_exit_with_stderr() {
        let result = completed(inline("partial"), Some(inline("oops")), 2);
        assert_eq!(
            result.render_for_model(),
            "partial\nstderr:\noops\nExit code: 2.\n"
        );
    }

    #[test]
    fn completed_command_stderr_only_has_no_contradicting_placeholder() {
        assert_eq!(
            completed(empty_stream(), Some(inline("oops")), 2).render_for_model(),
            "stderr:\noops\nExit code: 2.\n"
        );
    }

    #[tokio::test]
    async fn test_terminal_background_args_parsing() {
        let args: TerminalBackgroundArgs = serde_json::from_str(
            r#"{"description":"sleep in background","script":"sleep 5","timeout":30}"#,
        )
        .unwrap();
        assert_eq!(args.timeout, 30);
        assert_eq!(args.terminal.script, "sleep 5");

        let missing_timeout = serde_json::from_str::<TerminalBackgroundArgs>(
            r#"{"description":"sleep in background","script":"sleep 5"}"#,
        )
        .unwrap_err();
        assert!(missing_timeout.to_string().contains("timeout"));

        let missing_description =
            serde_json::from_str::<TerminalArgs>(r#"{"script":"echo hello","timeout":1}"#)
                .unwrap_err();
        assert!(missing_description.to_string().contains("description"));
    }

    #[test]
    fn terminal_background_args_schema_inlines_terminal_fields() {
        let schema = serde_json::to_value(schemars::schema_for!(TerminalBackgroundArgs)).unwrap();
        let properties = schema["properties"].as_object().expect("object schema");
        for field in ["description", "script", "timeout", "max_lines"] {
            assert!(
                properties.contains_key(field),
                "{field} missing from schema {schema}"
            );
        }
        let required = schema["required"].as_array().expect("required list");
        for field in ["description", "script", "timeout"] {
            assert!(
                required.iter().any(|name| name == field),
                "{field} not required in {schema}"
            );
        }
    }

    /// The tools bin directory a container execution passes to the wrapper.
    fn container_tools_dir() -> ContainerPath {
        crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::bundle())
            .bin_dir()
            .clone()
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
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
        )
        .expect("wrap script");
        assert!(wrapped.contains("hash -r;"));
        assert!(wrapped.contains("MAY_IPC_DIR"));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));
    }

    #[test]
    fn wrap_container_script_preserves_user_script_semantics() {
        let script = "echo '$5,040' && subagent start --subagent .skills/slide/subagents/art_direction.md --prompt-file tmp/prompt.md";
        let wrapped = wrap_container_script(
            script,
            &[String::from("subagent")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
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
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
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
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
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
    async fn uninitialized_terminal_rejects_execution_and_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let tool = TerminalTool::new_exact(
            dir.path(),
            TestPermissionHandler::default(),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .unwrap();
        assert!(matches!(
            tool.prepared_tools_context(),
            Err(ToolRuntimeError::NotInitialized(_))
        ));
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        assert!(matches!(
            tool.resolve_default_target(),
            Err(TerminalError::Tools(ToolRuntimeError::NotInitialized(_)))
        ));
    }

    #[tokio::test]
    async fn child_terminal_exposes_controls_with_isolated_jobs() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted");
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
        assert_eq!(
            tool.prepared_tools_context().unwrap(),
            child.prepared_tools_context().unwrap()
        );
        assert!(child.job_registry().list().await.is_empty());
        let parent_registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let parent = tool.with_registry(parent_registry);
        let child = child.with_registry(parent.registry().clone());
        for name in ["terminal_kill", "terminal_input", "terminal_read"] {
            assert!(child.registry().is_tool_configured(name));
        }
        let args = vec!["--task-id".to_string(), "parent-task".to_string()];
        let parent_output = serde_json::to_value(
            parent
                .registry()
                .query_tool_handler("terminal_read", &args)
                .await,
        )
        .unwrap();
        let child_output = serde_json::to_value(
            child
                .registry()
                .query_tool_handler("terminal_read", &args)
                .await,
        )
        .unwrap();
        assert!(parent_output.to_string().contains("running"));
        assert!(!child_output.to_string().contains("running"));
        let names = child
            .to_dyn()
            .into_entries()
            .into_iter()
            .map(|entry| entry.definition.name().to_string())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["terminal", "terminal_background"]);
    }

    async fn query_control(
        registry: &ToolRegistry,
        name: &str,
        args: &[&str],
    ) -> serde_json::Value {
        let args = args.iter().map(ToString::to_string).collect::<Vec<_>>();
        serde_json::to_value(registry.query_tool_handler(name, &args).await)
            .expect("command envelope should serialize")
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn configured_child_terminal_controls_address_child_jobs() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let parent = tool.with_registry(registry);
        let child = parent.child();

        let result = child
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: TerminalArgs {
                        description: "echo one stdin line then sleep".to_string(),
                        script: "read line; printf 'got %s\\n' \"$line\"; sleep 30".to_string(),
                        mode: TerminalExecutionMode::Sandboxed,
                        ssh_server_id: None,
                        expect: OutputFormat::Text,
                        resolution: MediaResolution::Auto,
                        max_lines: 50,
                        raw: false,
                    },
                    timeout: 30,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&child.background_receiver());

        let input = query_control(
            child.registry(),
            "terminal_input",
            &["--task-id", &task_id, "--input", "ping"],
        )
        .await;
        assert_eq!(input["ok"], true, "child terminal_input failed: {input}");

        let echoed = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let read =
                    query_control(child.registry(), "terminal_read", &["--task-id", &task_id])
                        .await;
                assert_eq!(read["ok"], true, "child terminal_read failed: {read}");
                if read.to_string().contains("got ping") {
                    return read;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("child terminal_read should observe the echoed line");
        assert!(echoed.to_string().contains("running"), "{echoed}");

        let parent_read =
            query_control(parent.registry(), "terminal_read", &["--task-id", &task_id]).await;
        assert_eq!(parent_read["ok"], false, "{parent_read}");
        assert!(
            parent_read["error"]
                .as_str()
                .is_some_and(|error| error.contains("unknown task_id")),
            "parent terminal_read must not see the child's task: {parent_read}"
        );
        let parent_kill =
            query_control(parent.registry(), "terminal_kill", &["--task-id", &task_id]).await;
        assert!(
            parent_kill
                .to_string()
                .contains("No running background task"),
            "parent terminal_kill must not reach the child's task: {parent_kill}"
        );

        let kill = query_control(child.registry(), "terminal_kill", &["--task-id", &task_id]).await;
        assert!(
            kill.to_string().contains("terminated"),
            "child terminal_kill should stop its task: {kill}"
        );
        let stopped =
            query_control(child.registry(), "terminal_read", &["--task-id", &task_id]).await;
        assert!(
            stopped.to_string().contains("was killed"),
            "child task should be reported killed: {stopped}"
        );
    }

    // These four drive a real sandbox running `/bin/pwd`. `heel` reports
    // `UnsupportedPlatform` off Unix, and the binary they run does not exist
    // there either, so there is nothing for them to exercise on Windows.
    #[cfg(unix)]
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

    #[cfg(unix)]
    #[tokio::test]
    async fn sandboxed_terminal_executes_pwd() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted");
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
            background: false,
            script: "/bin/pwd".to_string(),
            mode: TerminalMode::Sandboxed,
            backend: ExecutionBackend::Local(tool.shell_sessions.local_tools().unwrap().clone()),
            stdin_blocked_notice: None,
            background_startup: BackgroundStartup::new(startup_tx),
            network_audit: None,
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
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

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_command_reports_nonzero_exit_code() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);

        let result = tool
            .call(
                TerminalArgs {
                    description: "exit with non-zero status".to_string(),
                    script: "exit 42".to_string(),
                    mode: TerminalExecutionMode::Sandboxed,
                    ssh_server_id: None,
                    expect: OutputFormat::Text,
                    resolution: MediaResolution::Auto,
                    max_lines: 50,
                    raw: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should not be promoted to transport error on non-zero exit");

        assert!(
            !result.is_error(),
            "non-zero exit must not produce ToolResult::Error, got: {result:?}"
        );
        assert_eq!(
            render_terminal_result(&result),
            "(no output)\nExit code: 42.\n"
        );
        assert_eq!(
            result.command().map(|command| &command.outcome),
            Some(&CommandOutcome::Exited { code: 42 })
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_terminal_accepts_terminal_input() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted")
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
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: TerminalArgs {
                        description: "wait for terminal input and echo it back".to_string(),
                        script: "printf 'name? '; read name; printf 'hello %s\\n' \"$name\""
                            .to_string(),
                        mode: TerminalExecutionMode::Sandboxed,
                        ssh_server_id: None,
                        expect: OutputFormat::Text,
                        resolution: MediaResolution::Auto,
                        max_lines: 50,
                        raw: false,
                    },
                    timeout: 30,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");

        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&tool.background_receiver());

        input_tool
            .call(
                InputTerminalArgs {
                    task_id: task_id.clone(),
                    input: "lexo".to_string(),
                    append_newline: true,
                },
                ToolContext::new(),
            )
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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        let child_pid_path = dir.path().join("child.pid");

        let result = tool
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: TerminalArgs {
                        description: "start a descendant process".to_string(),
                        script: "sleep 30 & echo $! > child.pid; wait".to_string(),
                        mode: TerminalExecutionMode::Sandboxed,
                        ssh_server_id: None,
                        expect: OutputFormat::Text,
                        resolution: MediaResolution::Auto,
                        max_lines: 50,
                        raw: false,
                    },
                    timeout: 30,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );

        let mut child_pid = None;
        for _ in 0..100 {
            // `echo $! > child.pid` creates the file before writing to it,
            // so a successful read can still yield empty or partial
            // content — poll until the pid actually parses.
            let parsed = match async_fs::read_to_string(&child_pid_path).await {
                Ok(value) => value.trim().parse::<i32>().ok(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => panic!("failed to read child pid: {error}"),
            };
            if let Some(pid) = parsed {
                child_pid = Some(pid);
                break;
            }
            async_io::Timer::after(std::time::Duration::from_millis(20)).await;
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
    async fn sandboxed_terminal_kills_stdin_blocked_command_and_reports_it() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted")
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            })
            .with_foreground_limit(Duration::from_secs(2));
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);

        let result = tool
            .call(
                TerminalArgs {
                    description: "wait for terminal input and echo it back".to_string(),
                    script: "printf 'name? '; read name; printf 'hello %s\\n' \"$name\""
                        .to_string(),
                    mode: TerminalExecutionMode::Sandboxed,
                    ssh_server_id: None,
                    expect: OutputFormat::Text,
                    resolution: MediaResolution::Auto,
                    max_lines: 50,
                    raw: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");

        let rendered = render_terminal_result(&result);
        assert!(rendered.contains("Task ID: '"), "{rendered}");
        assert!(
            rendered.contains(TERMINAL_STDIN_BLOCKED_NOTICE),
            "rendered result should report the stdin wait: {rendered}"
        );
    }

    /// The nice value `level` steps below this test process lands at, capped
    /// at the kernel's lowest priority — the same computation heel applies.
    #[cfg(unix)]
    fn expected_lowered_nice(level: u8) -> i64 {
        // SAFETY: reads the calling process's own nice value.
        let current = unsafe { libc::getpriority(libc::PRIO_PROCESS as _, 0) };
        i64::from(current + i32::from(level)).min(19)
    }

    #[cfg(unix)]
    fn test_nice_priority() -> ProcessPriority {
        ProcessPriority::new(
            CpuPriority::Nice(NiceLevel::new(10).expect("nice level 10 is in range")),
            IoPriority::Normal,
        )
    }

    /// A configured terminal tool with prepared local tools and the given
    /// shortened limits. Returns the tool and the cache owning its tools.
    #[cfg(unix)]
    async fn configured_test_terminal(
        dir: &Path,
        foreground: Duration,
        background_cap: Duration,
        alert_after: Duration,
    ) -> (
        tempfile::TempDir,
        TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal, Configured>,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir,
            TestPermissionHandler::default(),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_foreground_limit(foreground)
        .with_background_timeout_cap(background_cap)
        .with_long_job_alert_after(alert_after);
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        (cache, tool.with_registry(registry))
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_still_running_at_the_limit_is_killed_with_descendants() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(1),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let child_pid_path = dir.path().join("child.pid");

        let result = tool
            .call(
                TerminalArgs {
                    description: "start a descendant process".to_string(),
                    script: "sleep 30 & echo $! > child.pid; wait".to_string(),
                    mode: TerminalExecutionMode::Sandboxed,
                    ssh_server_id: None,
                    expect: OutputFormat::Text,
                    resolution: MediaResolution::Auto,
                    max_lines: 50,
                    raw: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(rendered.contains("Task ID: '"), "{rendered}");
        assert!(
            rendered.contains("terminal_background"),
            "rendered result should point to terminal_background: {rendered}"
        );

        let child_pid = async_fs::read_to_string(&child_pid_path)
            .await
            .expect("descendant pid file should exist")
            .trim()
            .parse::<i32>()
            .expect("descendant pid should parse");
        for _ in 0..100 {
            // SAFETY: signal 0 probes process existence.
            if unsafe { libc::kill(child_pid, 0) } != 0 {
                return;
            }
            async_io::Timer::after(std::time::Duration::from_millis(20)).await;
        }
        panic!("descendant process {child_pid} survived the foreground kill");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_background_rejects_timeout_above_cap_and_kills_at_timeout() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(2),
            Duration::from_secs(60),
        )
        .await;
        let background_receiver = tool.background_receiver();
        let background_tool = tool.background_tool();
        let args = |timeout: u64| TerminalBackgroundArgs {
            terminal: TerminalArgs {
                description: "sleep in the background".to_string(),
                script: "sleep 30".to_string(),
                mode: TerminalExecutionMode::Sandboxed,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                max_lines: 50,
                raw: false,
            },
            timeout,
        };

        assert!(
            background_tool
                .call(args(3), ToolContext::new())
                .await
                .is_err(),
            "a timeout above the cap must be rejected"
        );

        let result = background_tool
            .call(args(1), ToolContext::new())
            .await
            .expect("a timeout within the cap should start the job");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&background_receiver);

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the finished job stays in the registry");
        assert_eq!(
            info.status,
            JobStatus::Killed,
            "the job should have been killed at its timeout: {completed:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_past_the_alert_threshold_yields_one_alert() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_millis(300),
        )
        .await;
        let background_receiver = tool.background_receiver();

        let _result = tool
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: TerminalArgs {
                        description: "sleep in the background".to_string(),
                        script: "sleep 30".to_string(),
                        mode: TerminalExecutionMode::Sandboxed,
                        ssh_server_id: None,
                        expect: OutputFormat::Text,
                        resolution: MediaResolution::Auto,
                        max_lines: 50,
                        raw: false,
                    },
                    timeout: 30,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal_background should start the job");
        let task_id = started_task_id(&background_receiver);

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let alerts = loop {
            let alerts = background_receiver.take_long_running_alerts();
            if !alerts.is_empty() {
                break alerts;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "long-running alert did not fire"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].task_id, task_id);
        assert!(
            background_receiver.take_long_running_alerts().is_empty(),
            "the same job must not alert twice"
        );

        assert!(
            tool.job_registry().kill_by_task_id(&task_id).await,
            "test cleanup should kill the job"
        );
    }

    /// A terminal tool with prepared local tools and `priority` applied. The
    /// returned `TempDir` owns the prepared tools' cache; it must outlive the
    /// tool.
    #[cfg(unix)]
    async fn prioritized_terminal_tool(
        dir: &Path,
        priority: ProcessPriority,
    ) -> (
        tempfile::TempDir,
        TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal, Configured>,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir,
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted")
            .with_process_priority(priority);
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        (cache, tool.with_registry(registry))
    }

    /// A probe script made only of `sh` builtins: a strict sandbox executes
    /// nothing but the launched shell, the prepared tools and granted paths,
    /// so it reports its own `$$` into `pid` and waits for the host's `go`
    /// marker instead of running `nice`.
    #[cfg(unix)]
    fn pid_probe_args() -> TerminalArgs {
        TerminalArgs {
            description: "report the process pid and wait".to_string(),
            script: concat!(
                "printf '%s\\n' $$ > \"$PWD/pid\"; ",
                "while [ ! -f \"$PWD/go\" ]; do :; done"
            )
            .to_string(),
            mode: TerminalExecutionMode::Sandboxed,
            ssh_server_id: None,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            max_lines: 50,
            raw: false,
        }
    }

    /// Reads the pid the sandboxed probe wrote, retrying while the sandbox
    /// starts.
    #[cfg(unix)]
    async fn probe_pid(dir: &Path) -> u32 {
        let path = dir.join("pid");
        for _ in 0..150 {
            if let Ok(text) = std::fs::read_to_string(&path)
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("the sandboxed probe did not write {}", path.display());
    }

    /// The `pid`'s nice level, read from the host while the probe is alive.
    #[cfg(unix)]
    fn probe_nice(pid: u32) -> i64 {
        // SAFETY: queries the kernel for the named process's priority.
        i64::from(unsafe { libc::getpriority(libc::PRIO_PROCESS as _, pid as _) })
    }

    /// Releases the sandboxed probe by creating the `go` marker it polls for.
    #[cfg(unix)]
    fn release_probe(dir: &Path) {
        std::fs::File::create(dir.join("go")).expect("create the probe marker");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn foreground_command_runs_with_configured_nice() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_tools_cache, tool) =
            prioritized_terminal_tool(dir.path(), test_nice_priority()).await;

        let call = tool.call(pid_probe_args(), ToolContext::new());
        let probe = async {
            let pid = probe_pid(dir.path()).await;
            assert_eq!(
                probe_nice(pid),
                expected_lowered_nice(10),
                "foreground command should run at the configured nice level"
            );
            release_probe(dir.path());
        };
        let (result, ()) = futures::join!(call, probe);
        let rendered = render_terminal_result(&result.expect("terminal call should succeed"));
        assert!(!rendered.contains("Exit code:"), "{rendered}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn persistent_session_runs_with_configured_nice() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_tools_cache, tool) =
            prioritized_terminal_tool(dir.path(), test_nice_priority()).await;
        let background_receiver = tool.background_receiver();
        let input_tool = InputTerminalTool::new(tool.job_registry());

        // `exec` keeps `$$` — the pid the probe wrote — on the interactive
        // shell the session then drives.
        let result = tool
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: TerminalArgs {
                        description: "start a persistent shell".to_string(),
                        script: "printf '%s\\n' $$ > \"$PWD/pid\"; exec sh".to_string(),
                        mode: TerminalExecutionMode::Sandboxed,
                        ssh_server_id: None,
                        expect: OutputFormat::Text,
                        resolution: MediaResolution::Auto,
                        max_lines: 50,
                        raw: false,
                    },
                    timeout: 30,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&tool.background_receiver());

        let pid = probe_pid(dir.path()).await;
        assert_eq!(
            probe_nice(pid),
            expected_lowered_nice(10),
            "persistent session should run at the configured nice level"
        );

        input_tool
            .call(
                InputTerminalArgs {
                    task_id: task_id.clone(),
                    input: "exit".to_string(),
                    append_newline: true,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal_input should succeed");

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("persistent shell should exit cleanly after input");
        assert_eq!(terminal_result.exit_code, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_job_runs_with_configured_nice() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_tools_cache, tool) =
            prioritized_terminal_tool(dir.path(), test_nice_priority()).await;
        let background_receiver = tool.background_receiver();

        let result = tool
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: pid_probe_args(),
                    timeout: 30,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        let mut started = tool.background_receiver().take_started();
        assert_eq!(started.len(), 1, "expected one started-task event");
        let started = started.remove(0);
        let task_id = started.task_id;
        assert_eq!(
            rendered,
            format!(
                "Started in background as task '{task_id}'; output is being written to {} (absolute path: {}).\n",
                started.output_file.url, started.output_file.path
            )
        );
        assert_eq!(
            result.command().map(|command| &command.outcome),
            Some(&CommandOutcome::Running {
                task_id: task_id.clone(),
                output: started.output_file.clone(),
            })
        );

        let pid = probe_pid(dir.path()).await;
        assert_eq!(
            probe_nice(pid),
            expected_lowered_nice(10),
            "background job should run at the configured nice level"
        );
        release_probe(dir.path());

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("background job should complete successfully");
        assert_eq!(terminal_result.exit_code, 0);
    }

    #[test]
    fn wrap_container_script_applies_configured_priority() {
        let priority = ProcessPriority::new(
            CpuPriority::Nice(NiceLevel::new(5).expect("nice level 5 is in range")),
            IoPriority::Background,
        );
        let tools_dir = container_tools_dir();
        let executable = format!("--executable {}", shell_escape(tools_dir.as_str()));
        let wrapped = wrap_container_script(
            "echo ok",
            &[String::from("websearch")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
            priority,
            &tools_dir,
            "/workspace",
        )
        .expect("wrap script");
        assert!(wrapped.contains(
            "\"$MAY_IPC_BIN\" run --isolation permissive --working-dir '/workspace' \
             --env-passthrough PATH --env-passthrough HOME"
        ));
        assert!(wrapped.contains(&executable));
        assert!(wrapped.contains(
            "--ipc-endpoint 'tcp://host.docker.internal:9000' --grant \"$MAY_IPC_DIR=rx\" \
             --nice 5 --io-priority background -- sh -c 'echo ok'"
        ));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));

        let bare = wrap_container_script("echo ok", &[], None, priority, &tools_dir, "/workspace")
            .expect("wrap script");
        assert!(bare.contains("MAY_HEEL_BIN=\"$(command -v heel || true)\""));
        assert!(bare.contains(
            "\"$MAY_HEEL_BIN\" run --isolation permissive --working-dir '/workspace' \
             --env-passthrough PATH --env-passthrough HOME"
        ));
        assert!(bare.contains(&executable));
        assert!(bare.contains("--nice 5 --io-priority background -- sh -c 'echo ok'"));

        let normal = wrap_container_script(
            "echo ok",
            &[],
            None,
            ProcessPriority::default(),
            &tools_dir,
            "/workspace",
        )
        .expect("wrap script");
        assert_eq!(normal, "echo ok");
    }

    #[test]
    fn wrap_container_script_prioritizes_each_half_independently() {
        let io_only = wrap_container_script(
            "echo ok",
            &[],
            None,
            ProcessPriority::new(CpuPriority::Normal, IoPriority::Utility),
            &container_tools_dir(),
            "/workspace",
        )
        .expect("wrap script");
        assert!(io_only.contains(" --io-priority utility -- sh -c 'echo ok'"));
        assert!(!io_only.contains("--nice"));

        let cpu_only = wrap_container_script(
            "echo ok",
            &[],
            None,
            ProcessPriority::new(
                CpuPriority::Nice(NiceLevel::new(9).expect("nice level 9 is in range")),
                IoPriority::Normal,
            ),
            &container_tools_dir(),
            "/workspace",
        )
        .expect("wrap script");
        assert!(cpu_only.contains(" --nice 9 -- sh -c 'echo ok'"));
        assert!(!cpu_only.contains("--io-priority"));
    }

    /// Probe tool for the container acceptance test: it answers an IPC call
    /// with the marker it was invoked with, so a request that actually reached
    /// the live handler is unmistakable.
    #[derive(Debug, Deserialize, JsonSchema)]
    struct ProbeEchoArgs {
        /// Marker echoed back to the caller.
        marker: String,
    }

    struct ProbeEchoTool;

    impl Tool for ProbeEchoTool {
        fn name(&self) -> Cow<'static, str> {
            "probe_echo".into()
        }

        type Arguments = ProbeEchoArgs;
        type Res = ToolResult;

        fn call(
            &self,
            arguments: Self::Arguments,
            _cx: ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(ToolResult::text(format!(
                "marker: {}",
                arguments.marker
            ))))
        }
    }

    /// Host side of the container IPC transport on every platform: a client
    /// speaking the bridge's framing over loopback TCP reaches the registry's
    /// live handler and gets its envelope back.
    #[tokio::test(flavor = "multi_thread")]
    async fn container_ipc_bridge_round_trips_a_command_over_tcp() {
        let outputs = tempfile::tempdir().expect("outputs dir should be created");
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(ProbeEchoTool);
        let registry = Arc::new(builder.build(outputs.path()));
        let loopback = std::net::IpAddr::from([127, 0, 0, 1]);
        let bridge = start_container_ipc_bridge(
            executor_core::tokio::TokioGlobal,
            registry,
            BridgeAddress::try_from(loopback).expect("loopback binds the bridge"),
        )
        .expect("bridge should start on loopback");

        let mut stream = Async::<StdTcpStream>::connect(SocketAddr::new(loopback, bridge.port()))
            .await
            .expect("client should connect to the bridge");

        let method = b"probe_echo";
        let params = rmp_serde::to_vec_named(&serde_json::json!({ "marker": "pong" }))
            .expect("params should encode");
        let mut request = vec![u8::try_from(method.len()).expect("method name fits a byte")];
        request.extend_from_slice(method);
        request.extend_from_slice(&params);
        let request_length = u32::try_from(request.len()).expect("request fits the frame");
        stream
            .write_all(&request_length.to_be_bytes())
            .await
            .expect("request length should be written");
        stream
            .write_all(&request)
            .await
            .expect("request body should be written");

        let mut length_bytes = [0_u8; 4];
        stream
            .read_exact(&mut length_bytes)
            .await
            .expect("response length should be read");
        let mut response = vec![0_u8; u32::from_be_bytes(length_bytes) as usize];
        stream
            .read_exact(&mut response)
            .await
            .expect("response body should be read");

        let (success, payload) = response
            .split_first()
            .expect("response carries a success flag");
        assert_eq!(*success, 1, "bridge should report success");
        let rendered: String =
            rmp_serde::from_slice(payload).expect("response should decode as msgpack");
        assert_eq!(
            rendered, "marker: pong",
            "probe_echo should answer with the CLI-rendered payload, as `heel ipc` prints it"
        );

        drop(stream);
        bridge.stop().await;
    }

    /// Scratch `ubuntu:24.04` container used by the acceptance test; removed
    /// on drop.
    #[cfg(all(unix, aither_container_acceptance))]
    struct TestDockerContainer(String);

    #[cfg(all(unix, aither_container_acceptance))]
    impl TestDockerContainer {
        /// Starts a container with `heel_bin` mounted at `/usr/local/bin/heel`,
        /// the prepared tool bundle mounted read-only where production mounts
        /// it, and `host.docker.internal` resolving to the docker host gateway,
        /// and prepares the `/workspace` directory container executions run
        /// in. Returns `None` when docker cannot start the container.
        fn start(heel_bin: &Path, tools: &crate::MountSpec) -> Option<Self> {
            let name = format!("aither-test-{}", uuid::Uuid::new_v4().simple());
            let mount = format!("{}:/usr/local/bin/heel:ro", heel_bin.display());
            let tools_mount = format!("{}:{}:ro", tools.host_path.display(), tools.container_path);
            let output = std::process::Command::new("docker")
                .args([
                    "run",
                    "-d",
                    "--name",
                    &name,
                    "--add-host=host.docker.internal:host-gateway",
                    "-v",
                    &mount,
                    "-v",
                    &tools_mount,
                    "ubuntu:24.04",
                    "tail",
                    "-f",
                    "/dev/null",
                ])
                .output()
                .ok()?;
            if !output.status.success() {
                return None;
            }
            let prepared = std::process::Command::new("docker")
                .args(["exec", &name, "mkdir", "-p", CONTAINER_EXEC_WORKING_DIR])
                .output()
                .is_ok_and(|output| output.status.success());
            if !prepared {
                let _ = std::process::Command::new("docker")
                    .args(["rm", "-f", &name])
                    .output();
                return None;
            }
            Some(Self(name))
        }
    }

    #[cfg(all(unix, aither_container_acceptance))]
    impl Drop for TestDockerContainer {
        fn drop(&mut self) {
            let _ = std::process::Command::new("docker")
                .args(["rm", "-f", &self.0])
                .output();
        }
    }

    /// The docker `bridge` network's gateway as the host owns it — the
    /// address `host.docker.internal` resolves to inside the container via
    /// `--add-host …:host-gateway`, so it is the address the IPC bridge binds.
    #[cfg(all(unix, aither_container_acceptance))]
    async fn container_bridge_address(
        client: &bollard::Docker,
        container_id: &str,
    ) -> BridgeAddress {
        client
            .inspect_container(
                container_id,
                None::<bollard::query_parameters::InspectContainerOptions>,
            )
            .await
            .expect("container acceptance should inspect the container")
            .network_settings
            .and_then(|settings| settings.networks)
            .and_then(|mut networks| networks.remove("bridge"))
            .and_then(|endpoint| endpoint.gateway)
            .expect("container acceptance requires the bridge network gateway")
            .parse::<std::net::IpAddr>()
            .expect("the bridge gateway should parse as an IP address")
            .try_into()
            .expect("the bridge gateway should be a usable bridge address")
    }

    /// A Linux `heel` binary for the container under test: `HEEL_BIN` wins,
    /// then a `heel` already on `PATH`.
    #[cfg(all(unix, aither_container_acceptance))]
    fn container_test_heel_binary() -> Option<PathBuf> {
        std::env::var_os("HEEL_BIN")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                std::process::Command::new("sh")
                    .args(["-c", "command -v heel"])
                    .output()
                    .ok()
                    .filter(|output| output.status.success())
                    .and_then(|output| String::from_utf8(output.stdout).ok())
                    .map(|resolved| PathBuf::from(resolved.trim()))
            })
            .filter(|path| path.is_file())
    }

    /// End-to-end container path: `TerminalTool` execs the wrapped script in a
    /// real container through `BollardContainerExec`, and the script fires a
    /// real `heel ipc` request at the live host bridge.
    ///
    /// The wrapped `heel run` carries the configured priority through heel's
    /// own sandbox configuration, and `--ipc-endpoint` relays the nested
    /// sandbox's local IPC socket to the host bridge, so `probe_echo` answers
    /// from the live handler rather than failing at the transport.
    ///
    /// Compiled only with `--cfg aither_container_acceptance`: the test needs
    /// a real docker daemon and a Linux `heel` binary, which the dedicated CI
    /// job provides and the feature matrices do not. A missing daemon or
    /// binary fails the test — it is never skipped silently.
    /// Starts the acceptance container with the test bundle staged and
    /// mounted read-only as production mounts it, so the wrapped script runs
    /// a prepared command from that mount. The returned directory holds the
    /// staged bundle and must outlive the container.
    #[cfg(all(unix, aither_container_acceptance))]
    async fn start_acceptance_container() -> (
        TestDockerContainer,
        crate::tool_runtime::ToolBundle,
        tempfile::TempDir,
    ) {
        let docker_ready = std::process::Command::new("docker")
            .arg("info")
            .output()
            .is_ok_and(|output| output.status.success());
        assert!(
            docker_ready,
            "container acceptance requires a docker daemon"
        );
        let heel_bin = container_test_heel_binary()
            .expect("container acceptance requires a Linux heel binary (set HEEL_BIN)");
        let bundle = crate::tool_runtime::tests::bundle();
        let tools_cache = tempfile::tempdir().expect("tool cache dir should be created");
        let (tools_mount, _) = bundle
            .prepare_container_mount(tools_cache.path())
            .await
            .expect("test bundle should stage for a container mount");
        let container = TestDockerContainer::start(&heel_bin, &tools_mount)
            .expect("container acceptance could not start ubuntu:24.04");
        (container, bundle, tools_cache)
    }

    #[cfg(all(unix, aither_container_acceptance))]
    #[tokio::test]
    async fn container_command_runs_with_configured_nice_and_reaches_ipc() {
        let (container, bundle, _tools_cache) = start_acceptance_container().await;

        let client = Arc::new(
            bollard::Docker::connect_with_local_defaults().expect("docker client should connect"),
        );
        // The bridge binds the container network's gateway as the host owns
        // it (docker0), because native Linux Docker resolves
        // `host.docker.internal` (via `--add-host …:host-gateway`) to exactly
        // that address.
        let bind = container_bridge_address(&client, &container.0).await;
        println!("ipc bridge bind: {}", bind.ip());
        let exec = crate::AnyContainerExec::new(Arc::new(crate::BollardContainerExec::new(client)));
        let runtime = ContainerShellRuntime::new(
            container.0.clone(),
            crate::ContainerIpcRoute::new("host.docker.internal", bind),
            exec,
        );

        // The container backend needs its own prepared tools: the registry
        // entry names the container and its `/opt/aither/tools` bin dir, which
        // the wrapped script puts on `PATH` and heel must grant execute on.
        let container_tools = crate::tool_runtime::tests::container_tools(&bundle);
        let shell_sessions = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
            ssh: false,
        })
        .with_container_runtime(runtime)
        .with_test_container_tools(container_tools, container.0.clone());

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal> =
            TerminalTool::new_exact(
                dir.path(),
                TestPermissionHandler::default(),
                executor_core::tokio::TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize")
            .adopt_prepared_tools(shell_sessions)
            .await
            .expect("prepared tools should be adopted")
            .with_process_priority(ProcessPriority::new(
                CpuPriority::Nice(NiceLevel::new(5).expect("nice level 5 is in range")),
                IoPriority::Normal,
            ));
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(ProbeEchoTool);
        let registry = Arc::new(builder.build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);

        let result = tool
            .call(
                TerminalArgs {
                    description: "probe container priority and IPC transport".to_string(),
                    script: concat!(
                        "printf 'nice=%s\\n' \"$(nice)\"; ",
                        "jq --version; ",
                        "probe_echo --marker pong; ",
                        "printf 'probe_exit=%s\\n' \"$?\""
                    )
                    .to_string(),
                    mode: TerminalExecutionMode::Sandboxed,
                    ssh_server_id: None,
                    expect: OutputFormat::Text,
                    resolution: MediaResolution::Auto,
                    max_lines: 50,
                    raw: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("container terminal call should succeed");
        let stdout = render_terminal_result(&result);
        assert!(
            !stdout.contains("Exit code:"),
            "container command failed; stdout: {stdout}"
        );
        // The CI job runs the test with --nocapture so the container's output
        // — the applied nice level and the live probe reply — lands in the
        // job log.
        println!("container script output:\n{stdout}");
        assert!(
            stdout.contains("nice=5"),
            "container command should run at the configured nice level; stdout: {stdout}"
        );
        assert!(
            stdout.contains("jq 1.0.0"),
            "a prepared command should run from the read-only tools mount; stdout: {stdout}"
        );
        assert!(
            stdout.contains("pong"),
            "probe_echo should answer from the live handler; stdout: {stdout}"
        );
        assert!(
            stdout.contains("probe_exit=0"),
            "heel ipc should exit 0 through the relayed endpoint; stdout: {stdout}"
        );
    }
}
