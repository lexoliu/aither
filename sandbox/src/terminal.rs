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
        Arc, Mutex,
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
#[cfg(target_os = "linux")]
use heel::supervisor::{TreeControl, TreeReport};
use heel::{
    Access, AllowAll, Audited, CpuPriority, DomainRequest, Grant, IoPriority, IpcRouter,
    NetworkAuditLog, NetworkPolicy, ProcessPriority, ResolvedPath, Sandbox, SandboxConfig,
    SecurityConfig, StdioConfig, WorkingDir,
};
/// On platforms without `heel supervise` a wait produces no report.
#[cfg(not(target_os = "linux"))]
struct TreeReport;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shell_words::split as split_shell_words;
use tracing::{debug, info, warn};

#[cfg(all(unix, feature = "jobserver"))]
use crate::jobserver::{JobBroker, JobPool, job_broker_channel};
use crate::{
    ContainerPath, MountRoot, PreparedTools, ToolDistribution, ToolRuntimeError,
    builtin::builtin_router,
    command::{
        AnyCommandGate, AnyCommandObserver, CommandBlocked, CommandRole, GateDecision, ToolRegistry,
    },
    job_registry::{JobRegistry, JobStatus, job_registry_channel},
    output::{
        INLINE_OUTPUT_LIMIT, MediaResolution, OutputFormat, OutputStore, empty_stream,
        save_raw_to_file, stored_path,
    },
    permission::{PermissionError, PermissionHandler, PermissionRequest, Review, TerminalMode},
    process_tree::{JobMarker, JobProcess, JobTree},
    shell_session::{
        BridgeAddress, ContainerExecOutcome, ContainerShellRuntime, ExecLifetime, ShellBackend,
        ShellRuntimeAvailability, ShellSessionRegistry, SshRuntimeProfile,
    },
    stdin_watch::{
        EpisodeDetector, FeedCounter, LocalStdinProbe, STDIN_WATCH_INTERVAL, StdinWatchSignal,
    },
    subagent::{
        CommitLedger, CommittingWorktree, LaunchHistory, Launcher, SubagentGrant,
        SubagentGrantError, canonical as canonical_path,
    },
};

pub fn erase_terminal_tool<T>(tool: T) -> crate::command::DynTerminalToolEntry
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

#[derive(Clone)]
struct PermissionNetworkPolicy<P> {
    permission_handler: Arc<P>,
}

impl<P: PermissionHandler + 'static> NetworkPolicy for PermissionNetworkPolicy<P> {
    async fn check(&self, request: &DomainRequest) -> bool {
        self.permission_handler
            .authorize_domain(request.host(), request.port())
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

/// Turns `script` into the process to launch. A shell launch with a `report`
/// file records the shell's final directory there when it exits; a direct
/// command cannot change directory, so it reports nothing.
fn build_terminal_launch(
    script: &str,
    report: Option<&Path>,
) -> Result<TerminalLaunch, TerminalError> {
    let script = script.trim();
    if script.is_empty() {
        return Err(TerminalError::Execution(
            "command must not be empty".to_string(),
        ));
    }

    match direct_command(script)? {
        Some((program, args)) => Ok(TerminalLaunch::Direct { program, args }),
        None => shell_launch_reporting(script, report),
    }
}

/// The program and arguments of a script that runs without a shell, or
/// `None` when it needs one.
fn direct_command(script: &str) -> Result<Option<(String, Vec<String>)>, TerminalError> {
    if script_requires_shell(script) {
        return Ok(None);
    }
    let (program, args) = parse_direct_command(script)?;
    if looks_like_env_assignment(&program) || is_shell_builtin(&program) {
        return Ok(None);
    }
    Ok(Some((program, args)))
}

/// Whether a local launch of `script` runs in a shell, the only way a
/// command can change directory.
fn runs_in_shell(script: &str) -> bool {
    matches!(direct_command(script.trim()), Ok(None))
}

#[derive(Template)]
#[template(path = "carry_cwd.sh", escape = "none")]
struct CarryCwdTemplate<'a> {
    escaped_report: &'a str,
    escaped_script: &'a str,
}

fn shell_launch_reporting(
    script: &str,
    report: Option<&Path>,
) -> Result<TerminalLaunch, TerminalError> {
    let Some(report) = report else {
        let (program, args) = shell_launch(script);
        return Ok(TerminalLaunch::Shell { program, args });
    };
    let script = carry_cwd_script(script, &report.to_string_lossy())?;
    let (program, args) = shell_launch(&script);
    Ok(TerminalLaunch::Shell { program, args })
}

/// `script`, run through `eval`, followed by a trailer that writes the
/// shell's final directory to `report`, a path as the shell sees it, and
/// exits with the script's status.
///
/// The report is not written from an `EXIT` trap: setting one makes a
/// non-interactive bash catch SIGTERM, and a child it has just forked still
/// carries that handler until it resets its signals, so a `kill` sent in that
/// window is swallowed and the process it names runs on. `eval` keeps the
/// trailer out of reach of the script's own syntax, such as an unterminated
/// here-document. A script that ends through `exit` writes no report.
fn carry_cwd_script(script: &str, report: &str) -> Result<String, TerminalError> {
    let escaped_report = shell_escape(report);
    let escaped_script = shell_escape(script);
    CarryCwdTemplate {
        escaped_report: &escaped_report,
        escaped_script: &escaped_script,
    }
    .render()
    .map_err(|error| {
        TerminalError::Execution(format!("failed to render directory report: {error}"))
    })
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
/// Each call selects its own runtime mode and starts a fresh shell; only a
/// successful `cd` carries over to the next call.
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

/// A background job whose stdin pipe has a reader waiting for input.
///
/// Sent once per blocking episode: the job is not alerted again until it has
/// been fed through `terminal_input` and blocks anew.
#[derive(Debug, Clone)]
pub struct StdinBlockedJob {
    /// Task ID of the job.
    pub task_id: String,
    /// The script the job runs.
    pub script: String,
}

/// A background job whose stdin watch failed; carries the probe's error.
///
/// Sent once: after a failure the job can no longer report being blocked on
/// stdin.
#[derive(Debug, Clone)]
pub struct StdinProbeFailedJob {
    /// Task ID of the job.
    pub task_id: String,
    /// The script the job runs.
    pub script: String,
    /// Why the watch failed.
    pub error: String,
}

/// Why the next foreground command does not start where the last one left
/// the shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryResetReason {
    /// The command failed, so the directory it ended in is not carried.
    Failed,
    /// The command ended through `exit` or a signal and reported no
    /// directory.
    Unreported,
    /// The directory it ended in lies outside the paths the terminal may
    /// read.
    OutsideReadable,
    /// The directory it ended in lies outside the container's workspace
    /// mount.
    OutsideWorkspace,
    /// The directory no longer exists.
    Removed,
}

/// The working directory carried between foreground commands was reset.
///
/// Sent once per reset, at the moment it happens.
#[derive(Debug, Clone)]
pub struct DirectoryReset {
    /// The directory the next command runs in, as the command sees it.
    pub dir: PathBuf,
    /// Why the shell's directory was not carried.
    pub reason: DirectoryResetReason,
}

/// The most processes a [`DescendantsKilled`] alert names.
const KILLED_PROCESSES_NAMED: usize = 8;

/// Processes a foreground command left running when its shell exited, killed
/// with it.
///
/// Sent once per command, at the moment they were killed.
#[derive(Debug, Clone)]
pub struct DescendantsKilled {
    /// The first processes killed, at most eight.
    pub processes: Vec<JobProcess>,
    /// How many more were killed beyond those named.
    pub omitted: usize,
}

impl DescendantsKilled {
    fn new(mut killed: Vec<JobProcess>) -> Self {
        let omitted = killed.len().saturating_sub(KILLED_PROCESSES_NAMED);
        killed.truncate(KILLED_PROCESSES_NAMED);
        Self {
            processes: killed,
            omitted,
        }
    }
}

/// A system alert about a background job, or about the carried working
/// directory, delivered through the terminal's alert channel.
#[derive(Debug, Clone)]
pub enum JobAlert {
    /// A foreground command's shell exited while descendants were still
    /// running, and they were killed.
    DescendantsKilled(DescendantsKilled),
    /// The directory carried between foreground commands was reset.
    DirectoryReset(DirectoryReset),
    /// The job crossed the long-running threshold.
    LongRunning(LongRunningJob),
    /// The job is waiting for input on its stdin.
    StdinBlocked(StdinBlockedJob),
    /// Watching the job's stdin failed; no blocked-on-stdin alert can follow.
    StdinProbeFailed(StdinProbeFailedJob),
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
    alert_rx: Receiver<JobAlert>,
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
    alert_tx: Sender<JobAlert>,
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

    /// Delivers a job alert to the receiver.
    pub async fn send_alert(&self, alert: JobAlert) {
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

    /// Takes all pending job alerts without blocking.
    ///
    /// A job produces at most one [`JobAlert::LongRunning`], and one
    /// [`JobAlert::StdinBlocked`] per blocking episode.
    #[must_use]
    pub fn take_job_alerts(&self) -> Vec<JobAlert> {
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

    /// Waits for the next completed task or job alert, bounded by `duration`.
    ///
    /// Returns `None` when nothing arrives in time; a closed channel never
    /// ends the wait early.
    pub async fn recv_event_timeout(
        &self,
        duration: std::time::Duration,
    ) -> Option<BackgroundTaskEvent> {
        futures_lite::future::or(
            async {
                match self.rx.recv().await {
                    Ok(task) => Some(BackgroundTaskEvent::Completed(Box::new(task))),
                    // A closed channel never yields again; only the timer
                    // may end the wait, or callers would spin.
                    Err(_) => std::future::pending().await,
                }
            },
            futures_lite::future::or(
                async {
                    match self.alert_rx.recv().await {
                        Ok(alert) => Some(BackgroundTaskEvent::Alert(alert)),
                        Err(_) => std::future::pending().await,
                    }
                },
                async {
                    futures_lite::future::yield_now().await;
                    async_io::Timer::after(duration).await;
                    None
                },
            ),
        )
        .await
    }
}

/// What ended a [`BackgroundTaskReceiver::recv_event_timeout`] wait.
#[derive(Debug)]
pub enum BackgroundTaskEvent {
    /// A background task completed; boxed to keep the enum small.
    Completed(Box<CompletedTask>),
    /// A job alert arrived.
    Alert(JobAlert),
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

/// Launches subagents from one terminal: every subagent it creates runs with
/// at most that terminal's authority.
///
/// Each terminal that serves a registry owns one, handed to its terminal-bound
/// commands through its [`TerminalBinding`].
#[derive(Clone, Debug)]
pub struct TerminalToolFactory {
    tx: Sender<TerminalToolRequest>,
}

/// One request to a terminal's launcher: the grant the subagent launches with
/// and where the finished tool goes.
#[derive(Debug)]
struct TerminalToolRequest {
    grant: SubagentGrant,
    observer: Option<AnyCommandObserver>,
    gate: Option<AnyCommandGate>,
    reply: Sender<Result<crate::command::DynTerminalTool, SubagentGrantError>>,
}

/// Errors that can occur when requesting a subagent terminal tool.
#[derive(Debug, thiserror::Error)]
pub enum TerminalToolFactoryError {
    /// The launching terminal is gone.
    #[error("terminal tool factory is not available")]
    Unavailable,
    /// The factory failed to return a tool.
    #[error("terminal tool factory failed to return a tool")]
    NoResponse,
    /// The subagent's grant cannot be given.
    #[error(transparent)]
    Grant(#[from] SubagentGrantError),
}

impl TerminalToolFactory {
    /// Requests a new subagent terminal tool from the terminal this launcher
    /// belongs to.
    ///
    /// The tool serves the subagent registry and runs in a space of its own
    /// inside the launching terminal's working directory; `grant` is
    /// everything it may write beyond that space, and must lie inside what the
    /// launching terminal can write. The commands it calls are told to
    /// `observer`, and every shell command it runs is put to `gate`.
    ///
    /// # Errors
    /// Returns an error when the launching terminal is gone or does not
    /// respond, or when the grant cannot be given.
    pub async fn create(
        &self,
        grant: SubagentGrant,
        observer: Option<AnyCommandObserver>,
        gate: Option<AnyCommandGate>,
    ) -> Result<crate::command::DynTerminalTool, TerminalToolFactoryError> {
        let (reply, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(TerminalToolRequest {
                grant,
                observer,
                gate,
                reply,
            })
            .await
            .map_err(|_| TerminalToolFactoryError::Unavailable)?;
        Ok(reply_rx
            .recv()
            .await
            .map_err(|_| TerminalToolFactoryError::NoResponse)??)
    }
}

/// Serves a terminal's launch requests one at a time, so the grants handed
/// out from it are recorded in one place.
///
/// `source` is the terminal with its registry not yet bound: it holds no
/// handle to its own launcher, so the loop ends once every terminal-bound
/// command holding the launcher is gone.
async fn serve_launches<P, E>(
    requests: Receiver<TerminalToolRequest>,
    source: TerminalTool<P, E, Launching>,
) where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    let mut history = LaunchHistory::default();
    while let Ok(TerminalToolRequest {
        grant,
        observer,
        gate,
        reply,
    }) = requests.recv().await
    {
        let tool = source
            .subagent(&mut history, grant, observer, gate)
            .await
            .map(TerminalTool::to_dyn);
        if reply.send(tool).await.is_err() {
            warn!("terminal tool factory response channel dropped");
        }
    }
}

/// What a terminal hands the commands bound to it: where it works and the
/// launcher for the subagents it starts.
#[derive(Clone, Debug)]
pub struct TerminalBinding {
    launcher: TerminalToolFactory,
    working_dir: PathBuf,
    role: CommandRole,
    observer: Option<AnyCommandObserver>,
}

impl TerminalBinding {
    /// Launches subagents from this terminal, within its authority.
    #[must_use]
    pub const fn launcher(&self) -> &TerminalToolFactory {
        &self.launcher
    }

    /// The terminal's working directory: the main agent's sandbox, or a
    /// subagent's own space.
    #[must_use]
    pub fn working_dir(&self) -> &Path {
        &self.working_dir
    }

    /// The role the terminal's command registry serves: the main agent's, or
    /// a subagent's at any depth.
    #[must_use]
    pub const fn role(&self) -> CommandRole {
        self.role
    }

    /// The observer told about the commands this terminal's registry serves.
    #[must_use]
    pub const fn observer(&self) -> Option<&AnyCommandObserver> {
        self.observer.as_ref()
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

/// Marker type for a terminal tool with a configured registry, bound to the
/// terminal.
#[derive(Clone, Debug)]
pub struct Configured {
    registry: Arc<ToolRegistry>,
    binding: TerminalBinding,
}

/// The registry state of the copy of a terminal its launch loop holds: the
/// registry is not bound, so the copy holds no handle to its own launcher.
#[derive(Clone, Debug)]
pub struct Launching {
    registry: Arc<ToolRegistry>,
}

/// A registry state a terminal can launch subagents from.
///
/// Sealed: declared `pub` only so it may bound the `TerminalTool` impl that
/// launches subagents; it lives in this private module and is not
/// re-exported.
pub trait RegistryState: Clone + Send + Sync {
    /// The registry the terminal serves; a subagent's registry derives from it.
    fn registry(&self) -> &Arc<ToolRegistry>;
}

impl RegistryState for Configured {
    fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }
}

impl RegistryState for Launching {
    fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }
}

/// What a subagent's terminal holds that the main agent's does not.
#[derive(Clone, Debug)]
struct SubagentAuthority {
    /// The worktrees it may commit in, with their git data; it hands on no
    /// other commit grant.
    committing: Vec<CommittingWorktree>,
    /// Where its commands run when its home backend is a container: its
    /// space and granted paths, mapped through the container's workspace
    /// mount when it was launched.
    container: Option<ContainerConfinement>,
}

/// The main agent's working directory, carried from one foreground command
/// to the next. `None` inside means the terminal's own working directory.
#[derive(Clone, Default)]
struct CarriedDir(Arc<Mutex<Option<PathBuf>>>);

impl CarriedDir {
    fn get(&self) -> Option<PathBuf> {
        self.0.lock().expect("carried directory lock").clone()
    }

    fn set(&self, dir: Option<PathBuf>) {
        *self.0.lock().expect("carried directory lock") = dir;
    }
}

/// A foreground command that reports the directory it ended in.
#[derive(Clone)]
struct CwdCarry {
    /// Where the command starts.
    start: Option<PathBuf>,
    /// The file the command's shell writes its final directory to when it
    /// exits.
    report: PathBuf,
    /// Set for a container command: the workspace mount that maps the host
    /// paths above into the container and back.
    container: Option<ContainerCarry>,
    /// Whether the command runs in a shell and so writes the report; a
    /// direct command cannot change directory and writes none.
    shell: bool,
}

/// How a container command spells its directory report.
#[derive(Clone)]
struct ContainerCarry {
    mount: MountRoot,
    /// Where the container sees the report file.
    report: ContainerPath,
    /// Where the command starts inside the container.
    start: ContainerPath,
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
    alert_rx: Receiver<JobAlert>,
    /// Channel for sending long-running job alerts (cloned for each watchdog).
    alert_tx: Sender<JobAlert>,
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
    /// Set on a subagent's terminal, never on the main agent's.
    subagent: Option<SubagentAuthority>,
    /// The working directory the next foreground command starts in; held on
    /// the main agent's terminal only, a subagent's commands always start in
    /// its own space.
    carried_dir: Option<CarriedDir>,
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
            subagent: self.subagent.clone(),
            carried_dir: self.carried_dir.clone(),
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
            subagent: None,
            carried_dir: Some(CarriedDir::default()),
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
            subagent: None,
            carried_dir: Some(CarriedDir::default()),
            network_audit: None,
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            registry: Unconfigured,
        })
    }
}

impl<P, E, State> TerminalTool<P, E, State> {
    /// Moves every part into a terminal with another registry state.
    fn map_state<Next>(self, state: impl FnOnce(State) -> Next) -> TerminalTool<P, E, Next> {
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
            subagent: self.subagent,
            carried_dir: self.carried_dir,
            network_audit: self.network_audit,
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver,
            registry: state(self.registry),
        }
    }
}

impl<P, E> TerminalTool<P, E, Unconfigured>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    /// Attaches a tool registry to this terminal tool, enabling IPC command
    /// dispatch.
    ///
    /// The terminal gets its own launcher for the subagents it starts, and
    /// the registry's terminal-bound commands are built for it.
    #[must_use]
    pub fn with_registry(self, registry: Arc<ToolRegistry>) -> TerminalTool<P, E, Configured> {
        let registry = Arc::new(
            Arc::unwrap_or_clone(registry)
                .with_terminal_controls(self.job_registry(), self.outputs_dir()),
        );
        self.map_state(|Unconfigured| Launching { registry })
            .into_served()
    }
}

#[cfg(test)]
impl<P, E> TerminalTool<P, E, Configured>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    /// A subagent terminal with an empty grant, told to no observer or gate.
    async fn subagent_without_grant(&self) -> Result<Self, SubagentGrantError> {
        self.subagent(
            &mut LaunchHistory::default(),
            SubagentGrant::new(),
            None,
            None,
        )
        .await
    }
}

impl<P, E> TerminalTool<P, E, Launching>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    /// Starts this terminal's launcher and binds its registry to it.
    fn into_served(self) -> TerminalTool<P, E, Configured> {
        let (launcher, requests) = async_channel::unbounded();
        let binding = TerminalBinding {
            launcher: TerminalToolFactory { tx: launcher },
            working_dir: self.working_dir.clone(),
            role: self.registry.registry.role(),
            observer: self.registry.registry.observer().cloned(),
        };
        let source = self.clone();
        self.executor
            .spawn(async move { serve_launches(requests, source).await })
            .detach();
        self.map_state(|Launching { registry }| Configured {
            registry: Arc::new(ToolRegistry::clone(&registry).bind(&binding)),
            binding,
        })
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

/// The directory, inside a launching terminal's working directory, that holds
/// the spaces of the subagents it launches.
const SUBAGENT_SPACES_DIR: &str = "subagent-spaces";

impl<P, E, State> TerminalTool<P, E, State>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
    State: RegistryState,
{
    /// Creates the terminal tool of a subagent this terminal launches.
    ///
    /// The subagent shares the permission handler, output store and prepared
    /// tools, serves the subagent registry (so nothing reserved for the main
    /// agent is reachable, however deep subagents nest) with its
    /// terminal-bound commands built for it, and has its own launcher and
    /// background task tracking. It works in a space of its own inside this
    /// terminal's working directory; everything this terminal can write is
    /// read-only to it, apart from what `grant` hands over, which never
    /// exceeds what this terminal can write or commit. `history` records what
    /// the main agent's terminal handed out.
    ///
    /// # Errors
    /// Returns an error when `grant` names a path this terminal cannot write
    /// or a commit it cannot grant, when the grant cannot be resolved, when a
    /// container session cannot express the subagent's paths, or when the
    /// subagent's space cannot be created.
    pub(crate) async fn subagent(
        &self,
        history: &mut LaunchHistory,
        grant: SubagentGrant,
        observer: Option<AnyCommandObserver>,
        gate: Option<AnyCommandGate>,
    ) -> Result<TerminalTool<P, E, Configured>, SubagentGrantError> {
        // Grants are compared by real path, so every path of the subagent's
        // view is canonical.
        let working_dir = canonical_path(&self.working_dir).await?;
        let mut authority = vec![working_dir.clone()];
        for path in &self.writable_paths {
            authority.push(canonical_path(path).await?);
        }
        let spaces_root = working_dir.join(SUBAGENT_SPACES_DIR);
        let granted = grant
            .resolve(
                Launcher {
                    writable: &authority,
                    committing: self
                        .subagent
                        .as_ref()
                        .map(|subagent| subagent.committing.as_slice()),
                    spaces_root: &spaces_root,
                },
                history,
            )
            .await?;

        let space = spaces_root.join(crate::naming::random_word_slug(4));
        let container = self
            .container_confinement_for(&space, &granted.writable)
            .await?;
        async_fs::create_dir_all(&space)
            .await
            .map_err(|source| SubagentGrantError::Space {
                path: space.clone(),
                source,
            })?;

        let (completed_tx, completed_rx) = async_channel::unbounded();
        let (alert_tx, alert_rx) = async_channel::unbounded();
        let (started_tx, started_rx) = async_channel::unbounded();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (permission_tx, permission_rx) = async_channel::unbounded();
        let (job_registry, job_registry_service) = job_registry_channel();
        self.executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();
        let readable_paths = self
            .readable_paths
            .iter()
            .cloned()
            .chain(authority)
            .collect();
        let registry = Arc::new(
            self.registry
                .registry()
                .for_subagent(observer, gate)
                .with_terminal_controls(job_registry.clone(), self.outputs_dir()),
        );
        let launching = TerminalTool {
            working_dir: space,
            shell_sessions: self.shell_sessions.clone(),
            permission_handler: self
                .permission_handler
                .for_subagent()
                .map_or_else(|| self.permission_handler.clone(), Arc::new),
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            job_registry,
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
            writable_paths: granted.writable,
            readable_paths,
            subagent: Some(SubagentAuthority {
                committing: granted.committing,
                container,
            }),
            carried_dir: None,
            network_audit: self.network_audit.clone(),
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver.clone(),
            registry: Launching { registry },
        };
        Ok(launching.into_served())
    }

    /// The sandbox a subagent's commands run in when this terminal's home
    /// backend is a container: its space and granted paths, spelled through
    /// the container's workspace mount. `None` outside a container.
    async fn container_confinement_for(
        &self,
        space: &Path,
        writable: &[PathBuf],
    ) -> Result<Option<ContainerConfinement>, SubagentGrantError> {
        if self.shell_sessions.resolve_local_backend() != Ok(ShellBackend::Container) {
            return Ok(None);
        }
        let mount = self
            .shell_sessions
            .container_runtime()
            .and_then(ContainerShellRuntime::workspace_mount)
            .ok_or(SubagentGrantError::NoContainerMount)?;
        // The mount is compared by real path, like everything it maps.
        let mount = MountRoot::new(
            canonical_path(&mount.host_root).await?,
            mount.container_root.clone(),
        );
        let map = |path: &Path| {
            mount
                .container_path(path)
                .map_err(|source| SubagentGrantError::OutsideContainerMount {
                    path: path.to_path_buf(),
                    source,
                })
        };
        Ok(Some(ContainerConfinement {
            working_dir: map(space)?,
            writable: writable
                .iter()
                .map(|path| map(path))
                .collect::<Result<_, _>>()?,
        }))
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

    /// What this terminal hands the commands bound to it: its working
    /// directory and the launcher for the subagents it starts.
    #[must_use]
    pub const fn binding(&self) -> &TerminalBinding {
        &self.registry.binding
    }

    /// Lists the commits of the subagent this terminal serves, when it holds a
    /// commit grant.
    fn commit_ledger(&self) -> Option<CommitLedger> {
        self.subagent
            .as_ref()
            .and_then(|subagent| CommitLedger::new(&subagent.committing))
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
        let command_names = self.registry().available_tool_names();
        let commit_ledger = self.commit_ledger();
        let role = self.registry().role();
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
            commit_ledger,
            role,
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
                confinement: self
                    .subagent
                    .as_ref()
                    .and_then(|subagent| subagent.container.clone()),
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
        if mode.requires_approval() {
            match self
                .permission_handler
                .review(&PermissionRequest::Command { mode, script })
                .await
            {
                Review::Approved => return Ok(()),
                Review::Denied(reason) => {
                    return Err(anyhow::Error::new(TerminalError::from(
                        PermissionError::Denied(reason),
                    )));
                }
                Review::Passed => {}
            }
        }
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
            carry,
        } = request;
        let ExecutionTarget { mode, backend } = target;
        let (result_tx, result_rx) = async_channel::bounded(1);
        let (startup_tx, startup_rx) = async_channel::bounded(1);
        // A foreground command reads EOF from a null stdin; only a background
        // job gets a stdin pipe, fed through `terminal_input` and watched.
        let stdin_alerts = background.then(|| StdinAlerts {
            tx: self.alert_tx.clone(),
            task_id: task_id.clone(),
            script: script.clone(),
        });

        let lifetime = if background {
            JobLifetime::Background
        } else {
            JobLifetime::Foreground {
                alerts: self.alert_tx.clone(),
            }
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
                carry,
                script,
                mode,
                backend,
                stdin_alerts,
                lifetime,
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
        let carry = self.cwd_carry(&prepared, &arguments.script).await;
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
            carry: carry.clone(),
        });
        // Registration precedes `report_ready` on every backend, so once
        // startup returns the job is killable through the registry and the
        // foreground clock only measures the command's own runtime.
        wait_for_background_startup(&spawned.startup).await?;
        let outcome = wait_for_foreground_result(&spawned.result, self.limits.foreground).await;
        if let Some(carry) = &carry {
            let succeeded = matches!(&outcome, Some(Ok(result)) if result.exit_code == 0);
            self.record_carried_dir(carry, succeeded).await;
        }
        match outcome {
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
    /// The directory report of a foreground command on the main agent's
    /// terminal; `None` for a subagent, for an ssh command and for a container
    /// whose workspace mount is not declared.
    async fn cwd_carry(&self, prepared: &ExecutionPreparation, script: &str) -> Option<CwdCarry> {
        let carried = self.carried_dir.as_ref()?;
        let mount = match &prepared.target.backend {
            ExecutionBackend::Local(_) if !cfg!(windows) => None,
            ExecutionBackend::Container { runtime, .. } => Some(runtime.workspace_mount()?.clone()),
            _ => return None,
        };
        let start = carried.get();
        // A directory removed since it was recorded no longer carries.
        let start = match start {
            Some(dir)
                if async_fs::metadata(&dir)
                    .await
                    .is_ok_and(|meta| meta.is_dir()) =>
            {
                Some(dir)
            }
            Some(_) => {
                carried.set(None);
                let dir = if mount.is_some() {
                    PathBuf::from(CONTAINER_EXEC_WORKING_DIR)
                } else {
                    self.own_dir().await
                };
                self.send_directory_reset(dir, DirectoryResetReason::Removed)
                    .await;
                None
            }
            None => None,
        };
        let report = self.working_dir.join(format!(".cwd-{}", prepared.task_id));
        let container = match mount {
            None => None,
            Some(mount) => {
                // The report must lie in the mounted workspace for the host
                // to read it back.
                let container_report = mount.container_path(&report).ok()?;
                let container_start = start
                    .as_ref()
                    .and_then(|start| mount.container_path(start).ok())
                    .unwrap_or_else(|| {
                        ContainerPath::new(CONTAINER_EXEC_WORKING_DIR)
                            .expect("the exec directory is absolute")
                    });
                Some(ContainerCarry {
                    mount,
                    report: container_report,
                    start: container_start,
                })
            }
        };
        let shell = container.is_some() || runs_in_shell(script);
        Some(CwdCarry {
            start,
            report,
            container,
            shell,
        })
    }

    /// Makes the directory a finished foreground command ended in the start
    /// of the next one, when the command succeeded and the directory lies
    /// inside the paths this terminal may read. The report file is consumed
    /// either way. When the next command does not start where the shell
    /// ended, or the shell reported nothing, the model is alerted once.
    async fn record_carried_dir(&self, carry: &CwdCarry, succeeded: bool) {
        let reported = async_fs::read_to_string(&carry.report).await;
        let _ = async_fs::remove_file(&carry.report).await;
        let Some(carried) = &self.carried_dir else {
            return;
        };
        // A direct command writes no report; a shell that ended through
        // `exit` or a signal writes none either.
        let reported = reported
            .ok()
            .map(|reported| reported.trim_end_matches('\n').to_owned())
            .filter(|reported| !reported.is_empty());
        let Some(reported) = reported else {
            if carry.shell {
                self.alert_directory_reset(carry, DirectoryResetReason::Unreported)
                    .await;
            }
            return;
        };
        if !succeeded {
            if !self.left_in_start(carry, &reported).await {
                self.alert_directory_reset(carry, DirectoryResetReason::Failed)
                    .await;
            }
            return;
        }
        if let Some(container) = &carry.container {
            // A container is an unconfined main agent: what bounds its
            // directory is the workspace mount.
            match ContainerPath::new(reported.as_str())
                .map_err(drop)
                .and_then(|dir| container.mount.host_path(&dir).map_err(drop))
            {
                Ok(dir) => carried.set(Some(dir)),
                Err(()) => {
                    self.alert_directory_reset(carry, DirectoryResetReason::OutsideWorkspace)
                        .await;
                }
            }
            return;
        }
        let Ok(dir) = canonical_path(Path::new(&reported)).await else {
            self.alert_directory_reset(carry, DirectoryResetReason::Removed)
                .await;
            return;
        };
        for grant in std::iter::once(&self.working_dir)
            .chain(&self.writable_paths)
            .chain(&self.readable_paths)
        {
            if canonical_path(grant)
                .await
                .is_ok_and(|grant| dir.starts_with(grant))
            {
                carried.set(Some(dir));
                return;
            }
        }
        self.alert_directory_reset(carry, DirectoryResetReason::OutsideReadable)
            .await;
    }

    /// Whether the shell ended in the directory the command started in.
    async fn left_in_start(&self, carry: &CwdCarry, reported: &str) -> bool {
        if let Some(container) = &carry.container {
            return reported == container.start.as_str();
        }
        let start = carry.start.as_deref().unwrap_or(&self.working_dir);
        match (
            canonical_path(Path::new(reported)).await,
            canonical_path(start).await,
        ) {
            (Ok(reported), Ok(start)) => reported == start,
            _ => false,
        }
    }

    /// Alerts the model that the next command starts where `carry` started,
    /// not where the shell ended.
    async fn alert_directory_reset(&self, carry: &CwdCarry, reason: DirectoryResetReason) {
        let dir = match &carry.container {
            Some(container) => PathBuf::from(container.start.as_str()),
            None => match &carry.start {
                Some(start) => start.clone(),
                None => self.own_dir().await,
            },
        };
        self.send_directory_reset(dir, reason).await;
    }

    /// The terminal's own directory as a command sees it.
    async fn own_dir(&self) -> PathBuf {
        canonical_path(&self.working_dir)
            .await
            .unwrap_or_else(|_| self.working_dir.clone())
    }

    async fn send_directory_reset(&self, dir: PathBuf, reason: DirectoryResetReason) {
        let _ = self
            .alert_tx
            .send(JobAlert::DirectoryReset(DirectoryReset { dir, reason }))
            .await;
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
        // The one entry every foreground, background and subagent command on
        // every backend passes through, ahead of permission and execution.
        crate::kill_guard::check(&arguments.script)?;
        // Ahead of target resolution and permission, so a blocked command
        // never asks the user for anything. The decision carries no grant.
        if let Some(gate) = self.registry.registry().gate()
            && let GateDecision::Block { reason } = gate.check(&arguments.script).await
        {
            return Err(CommandBlocked { reason }.into());
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
            carry: None,
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
                        .send(JobAlert::LongRunning(LongRunningJob {
                            task_id,
                            script: job.script,
                            elapsed: job.started_at.elapsed(),
                        }))
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
        if !self.job_registry.kill_by_task_id(task_id).await
            && let Ok(result) = spawned.result.try_recv()
        {
            // The command finished between the limit expiring and the kill.
            return match result {
                Ok(result) => CommandResult::from(result).into_tool_result(),
                Err(err) => Err(anyhow::anyhow!(err)),
            };
        }
        let notice = format!(
            "Killed after running {}s in the foreground. Use terminal_background to run a command longer than that.",
            self.limits.foreground.as_secs()
        );
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
        CommandStream::Inline {
            text: redirect.preview,
        }
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
    carry: Option<CwdCarry>,
    script: String,
    mode: TerminalMode,
    backend: ExecutionBackend,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
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
    carry: Option<&'a CwdCarry>,
    script: &'a str,
    mode: TerminalMode,
    backend: &'a ExecutionBackend,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
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
    carry: Option<&'a CwdCarry>,
    script: &'a str,
    mode: TerminalMode,
    job_registry: &'a JobRegistry,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
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
    confinement: Option<&'a ContainerConfinement>,
    carry: Option<&'a ContainerCarry>,
    ipc_commands: &'a [String],
    job_registry: &'a JobRegistry,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
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
        /// The sandbox a subagent's command runs in inside the container; the
        /// main agent's commands run unconfined.
        confinement: Option<ContainerConfinement>,
    },
    /// A remote host over SSH.
    Ssh {
        target: String,
        runtime: SshRuntimeProfile,
        tools: PreparedTools<PathBuf>,
    },
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
    carry: Option<CwdCarry>,
}

struct SpawnedTerminal {
    result: Receiver<Result<CompletedCommand, String>>,
    startup: Receiver<Result<(), String>>,
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
    carry: Option<CwdCarry>,
    script: String,
    mode: TerminalMode,
    backend: ExecutionBackend,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
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
        carry,
        script,
        mode,
        backend,
        stdin_alerts,
        lifetime,
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
        carry,
        script: script.clone(),
        mode,
        backend,
        stdin_alerts,
        lifetime,
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
        carry,
        script,
        mode,
        backend,
        stdin_alerts,
        lifetime,
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
        carry: carry.as_ref(),
        script: &script,
        mode,
        backend: &backend,
        stdin_alerts,
        lifetime,
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
        ExecutionBackend::Container {
            runtime,
            tools,
            confinement,
        } => {
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
                confinement: confinement.as_ref(),
                carry: request.carry.and_then(|carry| carry.container.as_ref()),
                ipc_commands: &ipc_commands,
                job_registry: request.job_registry,
                stdin_alerts: request.stdin_alerts,
                lifetime: request.lifetime,
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
        carry: request.carry,
        script: request.script,
        mode: request.mode,
        job_registry: request.job_registry,
        stdin_alerts: request.stdin_alerts,
        lifetime: request.lifetime,
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

fn local_path_grants(
    writable: &[PathBuf],
    readable: &[PathBuf],
    tools: &PreparedTools<PathBuf>,
    job_fifo: Option<&Path>,
) -> heel::Result<Vec<Grant>> {
    writable
        .iter()
        .map(|path| Ok(Grant::new(ResolvedPath::new(path)?, Access::WRITE)))
        .chain(
            readable
                .iter()
                .map(|path| Ok(Grant::new(ResolvedPath::new(path)?, Access::READ))),
        )
        .chain(std::iter::once(Ok(Grant::new(
            ResolvedPath::new(tools.bin_dir())?,
            Access::EXEC,
        ))))
        .chain(
            job_fifo
                .iter()
                .map(|path| Ok(Grant::new(ResolvedPath::new(path)?, Access::WRITE))),
        )
        .collect()
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
    let mut runtime_environment =
        prepare_session_runtime_environment(context.working_dir, context.prepared_tools).await?;
    let marker = JobMarker::new();
    runtime_environment.push(marker.entry());
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
    let grants = local_path_grants(
        context.writable_paths,
        context.readable_paths,
        context.prepared_tools,
        job_fifo_path,
    )
    .map_err(|error| TerminalError::SandboxSetup(error.to_string()))?;
    let config_builder = SandboxConfig::builder()
        .network(policy)
        .working_dir(context.working_dir)
        .grants(grants)
        .security(SecurityConfig::interactive())
        .priority(context.priority);
    #[cfg(target_os = "linux")]
    let config_builder = config_builder.supervised_spawn(true);
    // `ipc` changes the builder's type, so both arms build where they are.
    let config = match create_ipc_router(context.registry) {
        Some(router) => config_builder.ipc(router).build(),
        None => config_builder.build(),
    };

    let sandbox = Sandbox::with_config_and_executor(config, context.executor.clone())
        .await
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let launch = build_terminal_launch(
        context.script,
        context.carry.map(|carry| carry.report.as_path()),
    )?;
    let (program, args) = match &launch {
        TerminalLaunch::Direct { program, args } | TerminalLaunch::Shell { program, args } => {
            (program, args)
        }
    };

    let mut command = sandbox.command(program);
    if let Some(start) = context.carry.and_then(|carry| carry.start.as_deref()) {
        command = command.current_dir(start);
    }
    let (child, tree) = spawn_local_command(
        command,
        args,
        runtime_environment,
        context.stdin_alerts.as_ref(),
        &context.lifetime,
        context.job_registry,
        &context.executor,
        &marker,
    )
    .await?;

    let pid = child.id();
    context
        .job_registry
        .register_with_tree(
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.mode,
            None,
            Some(tree.clone()),
        )
        .await;
    context.background_startup.report_ready().await;

    let output = collect_local_process_output(
        child,
        context.executor,
        context.job_registry,
        pid,
        LocalJob {
            tree,
            lifetime: context.lifetime,
            stdin_alerts: context.stdin_alerts,
        },
        "missing stdout pipe for sandbox process",
        "missing stderr pipe for sandbox process",
    )
    .await?;

    Ok((pid, output))
}

async fn execute_unsafe_background<E: Executor + Clone + 'static>(
    context: LocalBackgroundExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let mut runtime_environment =
        prepare_session_runtime_environment(context.working_dir, context.prepared_tools).await?;
    let marker = JobMarker::new();
    runtime_environment.push(marker.entry());
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
    let grants = local_path_grants(
        context.writable_paths,
        context.readable_paths,
        context.prepared_tools,
        job_fifo_path,
    )
    .map_err(|error| TerminalError::SandboxSetup(error.to_string()))?;
    let config_builder = SandboxConfig::builder()
        .network(AllowAll)
        .working_dir(context.working_dir)
        .grants(grants)
        .security(SecurityConfig::interactive())
        .priority(context.priority);
    #[cfg(target_os = "linux")]
    let config_builder = config_builder.supervised_spawn(true);
    // `ipc` changes the builder's type, so both arms build where they are.
    let config = match create_ipc_gateway_router(context.registry) {
        Some(router) => config_builder.ipc(router).build(),
        None => config_builder.build(),
    };

    let sandbox = Sandbox::with_config_and_executor(config, context.executor.clone())
        .await
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;

    let launch = build_terminal_launch(
        context.script,
        context.carry.map(|carry| carry.report.as_path()),
    )?;
    let (program, args) = match &launch {
        TerminalLaunch::Direct { program, args } | TerminalLaunch::Shell { program, args } => {
            (program, args)
        }
    };

    let mut command = sandbox.command(program);
    if let Some(start) = context.carry.and_then(|carry| carry.start.as_deref()) {
        command = command.current_dir(start);
    }
    let (child, tree) = spawn_local_command(
        command,
        args,
        runtime_environment,
        context.stdin_alerts.as_ref(),
        &context.lifetime,
        context.job_registry,
        &context.executor,
        &marker,
    )
    .await?;

    let pid = child.id();
    context
        .job_registry
        .register_with_tree(
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.mode,
            None,
            Some(tree.clone()),
        )
        .await;
    context.background_startup.report_ready().await;

    let output = collect_local_process_output(
        child,
        context.executor,
        context.job_registry,
        pid,
        LocalJob {
            tree,
            lifetime: context.lifetime,
            stdin_alerts: context.stdin_alerts,
        },
        "missing stdout pipe for unsafe process",
        "missing stderr pipe for unsafe process",
    )
    .await?;

    Ok((pid, output))
}

/// How long a command's descendants may outlive its top-level shell.
#[derive(Clone)]
enum JobLifetime {
    /// None: the survivors are killed and reported on `alerts`.
    Foreground { alerts: Sender<JobAlert> },
    /// All of them: the job runs until its whole tree has exited.
    Background,
}

/// Where a background job's stdin alerts go.
///
/// Its presence is what makes a command a background job: a command without
/// one reads EOF from a null stdin.
struct StdinAlerts {
    tx: Sender<JobAlert>,
    task_id: String,
    script: String,
}

impl StdinAlerts {
    async fn raise(&self) {
        debug!(task_id = %self.task_id, "background job is blocked on stdin");
        let _ = self
            .tx
            .send(JobAlert::StdinBlocked(StdinBlockedJob {
                task_id: self.task_id.clone(),
                script: self.script.clone(),
            }))
            .await;
    }

    /// Tells the agent this job's stdin watch died: no blocked-on-stdin
    /// alert can come for it anymore.
    async fn raise_failure(&self, error: String) {
        warn!(task_id = %self.task_id, error = %error, "stdin watch failed");
        let _ = self
            .tx
            .send(JobAlert::StdinProbeFailed(StdinProbeFailedJob {
                task_id: self.task_id.clone(),
                script: self.script.clone(),
                error,
            }))
            .await;
    }
}

const fn local_stdin_config(alerts: Option<&StdinAlerts>) -> StdioConfig {
    if alerts.is_some() {
        StdioConfig::Piped
    } else {
        StdioConfig::Null
    }
}

/// Watches one local background job's stdin pipe for a waiting reader.
struct LocalStdinWatch {
    probe: LocalStdinProbe,
    feed: FeedCounter,
    detector: EpisodeDetector,
    alerts: StdinAlerts,
}

impl LocalStdinWatch {
    /// Probes the pipe every [`STDIN_WATCH_INTERVAL`] and raises one alert per
    /// blocking episode, for as long as the future is polled.
    async fn run(mut self) -> std::convert::Infallible {
        loop {
            async_io::Timer::after(STDIN_WATCH_INTERVAL).await;
            let Self {
                probe,
                feed,
                mut detector,
                alerts,
            } = self;
            let fed = feed.count();
            let (probe, observed) = blocking::unblock(move || {
                let observed = probe.probe();
                (probe, observed)
            })
            .await;
            match observed {
                Ok(observed) => {
                    if detector.observe(fed, observed) {
                        alerts.raise().await;
                    }
                }
                Err(error) => {
                    // Detection cannot run: surface it once, then park — the
                    // job's exit still ends the race in
                    // `wait_for_local_process_exit`.
                    alerts.raise_failure(error.to_string()).await;
                    std::future::pending::<()>().await;
                }
            }
            self = Self {
                probe,
                feed,
                detector,
                alerts,
            };
        }
    }
}

async fn collect_local_process_output<E: Executor + Clone + 'static>(
    mut child: LocalChild,
    executor: E,
    job_registry: &JobRegistry,
    pid: u32,
    job: LocalJob,
    missing_stdout_message: &'static str,
    missing_stderr_message: &'static str,
) -> Result<std::process::Output, TerminalError> {
    let LocalJob {
        tree,
        lifetime,
        stdin_alerts,
    } = job;
    let mut stdin_watch = None;
    if let Some(stdin) = child.take_stdin() {
        let feed = FeedCounter::default();
        if let Some(alerts) = stdin_alerts {
            let probe = LocalStdinProbe::new(pid, &stdin)
                .map_err(|error| TerminalError::Execution(format!("stdin probe: {error}")))?;
            stdin_watch = Some(LocalStdinWatch {
                probe,
                feed: feed.clone(),
                detector: EpisodeDetector::default(),
                alerts,
            });
        }
        let input_tx = spawn_terminal_stdin_writer(stdin, feed);
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

    // The root stays unreaped until its tree is gone, so its pid and process
    // group stay reserved while the job's descendants are tracked by them.
    let exited = match wait_for_local_process_exit(&mut child, stdin_watch).await {
        Ok(exited) => exited,
        Err(error) => {
            tree.finish().await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            return Err(error);
        }
    };
    let descendants = end_descendants(&tree, &lifetime).await;
    // The task ends first: afterwards the root's pid may belong to anyone.
    tree.finish().await;
    let (status, report) = match exited {
        Some(status) => (Ok(status), None),
        None => match child.wait_reported().await {
            Ok((status, report)) => (Ok(status), report),
            Err(error) => (Err(TerminalError::Execution(error.to_string())), None),
        },
    };
    let _ = &report;
    #[cfg(target_os = "linux")]
    if let Some(report) = report {
        let killed: Vec<JobProcess> = report
            .killed
            .into_iter()
            .map(|process| JobProcess {
                pid: process.pid,
                name: process.name,
            })
            .collect();
        if !killed.is_empty() {
            info!(
                killed = killed.len(),
                "the supervisor killed the finished command's descendants"
            );
            if let JobLifetime::Foreground { alerts } = &lifetime {
                let _ = alerts
                    .send(JobAlert::DescendantsKilled(DescendantsKilled::new(killed)))
                    .await;
            }
        }
    }
    let status = match (descendants, status) {
        (Ok(()), Ok(status)) => status,
        (Err(error), _) | (Ok(()), Err(error)) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            job_registry.fail(pid, &error.to_string(), None).await;
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

/// A local job's process tree and how long its descendants may live.
struct LocalJob {
    tree: JobTree,
    lifetime: JobLifetime,
    stdin_alerts: Option<StdinAlerts>,
}

/// The teardown budget a job's supervisor gives its kills: the time to
/// freeze the tree, deliver `SIGKILL` and see every killed process exit.
/// Generous relative to the few milliseconds a `SIGKILL` exit takes, since a
/// too-short budget reports survivors that are merely still exiting.
#[cfg(target_os = "linux")]
const SUPERVISED_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// A spawned local job: the plain `heel` child on platforms without a
/// supervisor, or a `heel supervise`-supervised one on Linux, where the
/// supervisor is the command's parent and owns its whole tree.
enum LocalChild {
    /// Everywhere but Linux: the sandboxed process itself.
    Plain(heel::Child),
    /// Linux, foreground lifetime: `wait` yields the command's status and
    /// the teardown's report of what it killed.
    #[cfg(target_os = "linux")]
    Foreground(heel::supervisor::SupervisedChild<heel::supervisor::Foreground>),
    /// Linux, background lifetime: `wait` ends only once the last descendant
    /// of the command has exited.
    #[cfg(target_os = "linux")]
    Background(heel::supervisor::SupervisedChild<heel::supervisor::Background>),
}

impl LocalChild {
    /// The spawned process's id: the supervisor's on Linux.
    fn id(&self) -> u32 {
        match self {
            Self::Plain(child) => child.id(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.id(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.id(),
        }
    }

    fn take_stdin(&mut self) -> Option<std::fs::File> {
        match self {
            Self::Plain(child) => child.take_stdin(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.take_stdin(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.take_stdin(),
        }
    }

    fn take_stdout(&mut self) -> Option<std::fs::File> {
        match self {
            Self::Plain(child) => child.take_stdout(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.take_stdout(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.take_stdout(),
        }
    }

    fn take_stderr(&mut self) -> Option<std::fs::File> {
        match self {
            Self::Plain(child) => child.take_stderr(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.take_stderr(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.take_stderr(),
        }
    }

    /// A handle to the supervisor owning the job's tree.
    #[cfg(target_os = "linux")]
    fn tree_control(&self) -> TreeControl {
        match self {
            Self::Foreground(child) => child.tree_control(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.tree_control(),
            Self::Plain(_) => unreachable!("a supervised spawn returns no plain child"),
        }
    }

    /// Waits the job out, yielding the command's status and, for a
    /// supervised foreground job, the supervisor's teardown report: what it
    /// found still running under the command and killed.
    async fn wait_reported(
        &mut self,
    ) -> Result<(std::process::ExitStatus, Option<TreeReport>), heel::Error> {
        match self {
            Self::Plain(child) => Ok((child.wait().await?, None)),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => {
                let exit = child.wait().await?;
                Ok((exit.status, Some(exit.report)))
            }
            #[cfg(target_os = "linux")]
            Self::Background(child) => Ok((child.wait().await?, None)),
        }
    }
}

/// Spawns a local job's command and hands its tree to whoever owns it.
///
/// On Linux the sandbox was built with `supervised_spawn(true)`, so the
/// command runs under `heel supervise`: a child subreaper inside the sandbox
/// that owns the whole tree however its members change session, group or
/// environment. Everywhere else the root's process group is watched through
/// the process table, keyed by the job's marker.
#[allow(clippy::too_many_arguments)]
async fn spawn_local_command<E: Executor + Clone + 'static>(
    command: heel::Command<'_>,
    args: &[String],
    environment: Vec<(String, String)>,
    stdin_alerts: Option<&StdinAlerts>,
    lifetime: &JobLifetime,
    job_registry: &JobRegistry,
    executor: &E,
    marker: &JobMarker,
) -> Result<(LocalChild, JobTree), TerminalError> {
    let command = command
        .args(args)
        .envs(environment)
        .stdin(local_stdin_config(stdin_alerts))
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped);

    #[cfg(target_os = "linux")]
    {
        let _ = (job_registry, executor, marker);
        let child = match lifetime {
            JobLifetime::Foreground { .. } => LocalChild::Foreground(
                command
                    .spawn_supervised(heel::supervisor::Foreground {
                        reap_timeout: SUPERVISED_REAP_TIMEOUT,
                    })
                    .await
                    .map_err(|error| TerminalError::Execution(error.to_string()))?,
            ),
            JobLifetime::Background => LocalChild::Background(
                command
                    .spawn_supervised(heel::supervisor::Background {
                        reap_timeout: SUPERVISED_REAP_TIMEOUT,
                    })
                    .await
                    .map_err(|error| TerminalError::Execution(error.to_string()))?,
            ),
        };
        let tree = JobTree::supervised(child.tree_control());
        Ok((child, tree))
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = lifetime;
        let child = command
            .spawn()
            .await
            .map_err(|error| TerminalError::Execution(error.to_string()))?;
        let pid = child.id();
        let tree = job_registry.start_process_tree(executor, pid, marker);
        Ok((LocalChild::Plain(child), tree))
    }
}

/// Applies the rule that a job's descendants live exactly as long as the job,
/// after its top-level shell exited.
///
/// A foreground command's surviving descendants are killed and reported in one
/// [`JobAlert::DescendantsKilled`]; a background job stays running until every
/// descendant has exited, which its timeout, `terminal_kill` and the registry's
/// kills end by killing the tree.
async fn end_descendants(tree: &JobTree, lifetime: &JobLifetime) -> Result<(), TerminalError> {
    match lifetime {
        JobLifetime::Foreground { alerts } => {
            let killed = tree
                .kill_descendants()
                .await
                .map_err(|error| TerminalError::Execution(error.to_string()))?;
            if !killed.is_empty() {
                info!(
                    killed = killed.len(),
                    "killed descendants of a finished foreground command"
                );
                let _ = alerts
                    .send(JobAlert::DescendantsKilled(DescendantsKilled::new(killed)))
                    .await;
            }
            Ok(())
        }
        JobLifetime::Background => tree
            .wait_descendants_exit()
            .await
            .map_err(|error| TerminalError::Execution(error.to_string())),
    }
}

/// Waits for the child to exit, raising stdin alerts meanwhile.
///
/// On Unix the child is only observed to have exited and stays unreaped (the
/// answer is `None`); the caller waits for it once its tree is gone. Windows
/// has no such state and answers the exit status.
#[cfg_attr(unix, allow(clippy::needless_pass_by_ref_mut))]
async fn wait_for_local_process_exit(
    child: &mut LocalChild,
    stdin_watch: Option<LocalStdinWatch>,
) -> Result<Option<std::process::ExitStatus>, TerminalError> {
    #[cfg(unix)]
    let exit = async {
        crate::process_tree::wait_for_exit(child.id())
            .await
            .map(|()| None)
            .map_err(|error| TerminalError::Execution(error.to_string()))
    };
    #[cfg(not(unix))]
    let exit = async {
        child
            .wait_reported()
            .await
            .map(|(status, _)| Some(status))
            .map_err(|error| TerminalError::Execution(error.to_string()))
    };
    let Some(stdin_watch) = stdin_watch else {
        return exit.await;
    };
    futures_lite::future::or(exit, async { match stdin_watch.run().await {} }).await
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
    // A foreground command gets no input channel: the dropped sender is the
    // EOF its stdin reads.
    let exec_lifetime = if let Some(alerts) = context.stdin_alerts {
        context
            .job_registry
            .attach_terminal_input(pid, input_tx)
            .await;
        let (signal_tx, signal_rx) = async_channel::unbounded::<StdinWatchSignal>();
        context
            .executor
            .clone()
            .spawn(async move {
                while let Ok(signal) = signal_rx.recv().await {
                    match signal {
                        StdinWatchSignal::Blocked => alerts.raise().await,
                        StdinWatchSignal::Failed(error) => {
                            alerts.raise_failure(error).await;
                        }
                    }
                }
            })
            .detach();
        ExecLifetime::Background {
            stdin_blocked: signal_tx,
        }
    } else {
        drop(input_tx);
        ExecLifetime::Foreground
    };
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
    let script = match context.carry {
        Some(carry) => carry_cwd_script(&script, carry.report.as_str())?,
        None => script,
    };
    let working_dir = context
        .carry
        .map_or(CONTAINER_EXEC_WORKING_DIR, |carry| carry.start.as_str());
    let wrapped_script = wrap_container_script(
        &script,
        context.ipc_commands,
        ipc_bridge.as_ref().map(|bridge| ContainerIpcEndpoint {
            host: container_runtime.ipc_route().host(),
            port: bridge.port(),
        }),
        context.priority,
        context.prepared_tools.bin_dir(),
        working_dir,
        context.confinement,
    )?;

    let execution = exec
        .exec_boxed(
            container_runtime.container_id(),
            &wrapped_script,
            working_dir,
            kill_rx,
            input_rx,
            exec_lifetime,
        )
        .await;

    if let Some(bridge) = ipc_bridge {
        bridge.stop().await;
    }

    record_container_exec_outcome(context.job_registry, pid, &context.lifetime, execution).await
}

/// Records a finished container exec in the job's streams and folds it into
/// the call's result.
async fn record_container_exec_outcome(
    job_registry: &JobRegistry,
    pid: u32,
    lifetime: &JobLifetime,
    execution: Result<ContainerExecOutcome, String>,
) -> Result<(u32, std::process::Output), TerminalError> {
    match execution {
        Ok(ContainerExecOutcome::Completed { output, killed }) => {
            if let JobLifetime::Foreground { alerts } = lifetime
                && !killed.is_empty()
            {
                info!(
                    killed = killed.len(),
                    "killed processes of a finished foreground container command"
                );
                let _ = alerts
                    .send(JobAlert::DescendantsKilled(DescendantsKilled::new(killed)))
                    .await;
            }
            if !output.stdout.is_empty() {
                job_registry.append_stdout(pid, output.stdout.clone()).await;
            }
            if !output.stderr.is_empty() {
                job_registry.append_stderr(pid, output.stderr.clone()).await;
            }
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Ok((pid, output))
        }
        Ok(ContainerExecOutcome::Killed) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution("container job killed".to_string()))
        }
        Err(err) => {
            job_registry.fail(pid, &err, None).await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
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

/// The sandbox a subagent's command runs in inside a container, as container
/// paths: the main agent's commands run unconfined because the container is its
/// boundary, but a subagent's keep the per-command sandbox they have on the
/// host — write access to its own space and its granted paths only.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ContainerConfinement {
    working_dir: ContainerPath,
    writable: Vec<ContainerPath>,
}

/// Where a container command runs: the mounted tools, the exec's working
/// directory and, for a subagent, its sandbox.
#[derive(Clone, Copy)]
struct ContainerRunScope<'a> {
    tools_dir: &'a ContainerPath,
    working_dir: &'a str,
    confinement: Option<&'a ContainerConfinement>,
}

#[derive(Template)]
#[template(path = "container_heel_run.sh", escape = "none")]
struct ContainerHeelRunTemplate<'a> {
    heel_bin: &'a str,
    resolve_heel: bool,
    escaped_working_dir: &'a str,
    escaped_tools_dir: &'a str,
    /// Set for a subagent: the isolation level and grants replacing the
    /// container-wide permissive one.
    escaped_grants: Option<Vec<String>>,
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
    scope: ContainerRunScope<'_>,
) -> Result<String, TerminalError> {
    let ContainerRunScope {
        tools_dir,
        working_dir,
        confinement,
    } = scope;
    let nice = match priority.cpu() {
        CpuPriority::Nice(level) => Some(level.get()),
        CpuPriority::Normal => None,
    };
    let io_priority = match priority.io() {
        IoPriority::Utility => Some("utility"),
        IoPriority::Background => Some("background"),
        IoPriority::Normal => None,
    };
    // A subagent runs in its own space with heel's default isolation (that space
    // writable, the rest readable) plus its granted paths.
    let working_dir =
        confinement.map_or(working_dir, |confinement| confinement.working_dir.as_str());
    let escaped_grants = confinement.map(|confinement| {
        confinement
            .writable
            .iter()
            .map(|path| shell_escape(&format!("{path}=rw")))
            .collect()
    });
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
        escaped_grants,
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
    confinement: Option<&ContainerConfinement>,
) -> Result<String, TerminalError> {
    if priority.is_normal() && confinement.is_none() {
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
        ContainerRunScope {
            tools_dir,
            working_dir,
            confinement,
        },
    )
}

fn wrap_container_script(
    script: &str,
    ipc_commands: &[String],
    ipc_endpoint: Option<ContainerIpcEndpoint<'_>>,
    priority: ProcessPriority,
    tools_dir: &ContainerPath,
    working_dir: &str,
    confinement: Option<&ContainerConfinement>,
) -> Result<String, TerminalError> {
    if ipc_commands.is_empty() {
        return prioritized_container_script(script, priority, tools_dir, working_dir, confinement);
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
    let script = if priority.is_normal() && confinement.is_none() {
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
            ContainerRunScope {
                tools_dir,
                working_dir,
                confinement,
            },
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

fn spawn_terminal_stdin_writer<W>(
    mut writer: W,
    feed: FeedCounter,
) -> async_channel::Sender<Vec<u8>>
where
    W: Write + Send + 'static,
{
    let (tx, rx) = async_channel::unbounded::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(bytes) = rx.recv_blocking() {
            if bytes.is_empty() {
                continue;
            }
            feed.record();
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
    #[cfg(unix)]
    use crate::builtin::{InputTerminalArgs, InputTerminalTool};
    #[cfg(unix)]
    use heel::DenyAll;
    use heel::NiceLevel;

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
            _lifetime: ExecLifetime,
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

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_home_still_gives_the_host_runnable_tools() {
        let container = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
            ssh: false,
        })
        .with_container_runtime(test_container_runtime("aither-host-tools"))
        .with_test_container_tools(
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::bundle()),
            "aither-host-tools".to_string(),
        );
        assert!(
            matches!(
                container.local_tools(),
                Err(ToolRuntimeError::NotInitialized(_))
            ),
            "the container's own tools are not host-runnable"
        );

        let artifacts = tempfile::tempdir().unwrap();
        let distribution = script_distribution(
            artifacts.path(),
            b"#!/bin/sh\nprintf '%s\\n' 'tool 1.0.0'\n",
        )
        .await;
        let cache = tempfile::tempdir().unwrap();
        let host = container
            .host_tools(&distribution, cache.path())
            .await
            .expect("the distribution prepares the host target");
        let runtimes = std::fs::canonicalize(cache.path().join("runtimes")).unwrap();
        assert!(host.bin_dir().starts_with(runtimes));
        for command in ["apply_patch", "rg", "ast-grep", "jq"] {
            let output = std::process::Command::new(host.bin_dir().join(command))
                .output()
                .expect("the host runs the prepared tool");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                "tool 1.0.0\n",
                "{command}"
            );
        }
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
            CommandStream::Inline { text } => Some(text.as_str()),
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
        CommandStream::Inline {
            text: text.to_string(),
        }
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
            None,
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
            None,
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
            None,
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
            None,
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

    /// A handler whose review step answers `review` and whose own route
    /// waits for a person.
    struct ReviewingHandler {
        review: Review,
        waits: AtomicUsize,
        checks: AtomicUsize,
    }

    impl ReviewingHandler {
        fn new(review: Review) -> Self {
            Self {
                review,
                waits: AtomicUsize::new(0),
                checks: AtomicUsize::new(0),
            }
        }
    }

    impl PermissionHandler for ReviewingHandler {
        fn review(
            &self,
            _request: &PermissionRequest<'_>,
        ) -> impl std::future::Future<Output = Review> + Send {
            std::future::ready(self.review.clone())
        }

        fn check(
            &self,
            _mode: TerminalMode,
            _script: &str,
        ) -> impl std::future::Future<Output = Result<bool, PermissionError>> + Send {
            self.checks.fetch_add(1, AtomicOrdering::Relaxed);
            std::future::ready(Ok(true))
        }

        fn will_wait_for_approval(
            &self,
            _mode: TerminalMode,
            _script: &str,
        ) -> impl std::future::Future<Output = bool> + Send {
            self.waits.fetch_add(1, AtomicOrdering::Relaxed);
            std::future::ready(true)
        }
    }

    async fn permission_stages(
        review: Review,
    ) -> (Result<(), String>, Vec<PermissionEventStage>, [usize; 2]) {
        let dir = tempfile::tempdir().unwrap();
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir.path(),
            ReviewingHandler::new(review),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let tool = tool.with_registry(registry);
        let result = tool
            .ensure_permission_allowed(TerminalMode::Unsafe, "make install")
            .await
            .map_err(|error| error.to_string());
        let stages = tool
            .permission_receiver()
            .take_pending()
            .into_iter()
            .map(|event| event.stage)
            .collect();
        let handler = &tool.permission_handler;
        (
            result,
            stages,
            [
                handler.waits.load(AtomicOrdering::Relaxed),
                handler.checks.load(AtomicOrdering::Relaxed),
            ],
        )
    }

    #[tokio::test]
    async fn a_request_a_reviewer_approves_emits_no_permission_event() {
        let (result, stages, [waits, checks]) = permission_stages(Review::Approved).await;
        result.expect("approved");
        assert_eq!(stages, Vec::<PermissionEventStage>::new());
        assert_eq!([waits, checks], [0, 0], "the route is not asked");
    }

    #[tokio::test]
    async fn a_request_a_reviewer_denies_emits_no_permission_event() {
        let (result, stages, [waits, checks]) =
            permission_stages(Review::Denied("not today".into())).await;
        assert!(result.unwrap_err().contains("not today"));
        assert_eq!(stages, Vec::<PermissionEventStage>::new());
        assert_eq!([waits, checks], [0, 0], "the route is not asked");
    }

    #[tokio::test]
    async fn a_request_every_reviewer_passes_waits_then_resolves() {
        let (result, stages, [waits, checks]) = permission_stages(Review::Passed).await;
        result.expect("the route allows");
        assert_eq!(
            stages,
            [
                PermissionEventStage::Waiting,
                PermissionEventStage::Resolved
            ]
        );
        assert_eq!([waits, checks], [1, 1]);
    }

    #[tokio::test]
    async fn permission_network_policy_reviews_a_domain_before_asking() {
        struct Domains(Review, AtomicUsize);
        impl PermissionHandler for Domains {
            fn check(
                &self,
                _mode: TerminalMode,
                _script: &str,
            ) -> impl std::future::Future<Output = Result<bool, PermissionError>> + Send
            {
                std::future::ready(Ok(true))
            }
            fn review(
                &self,
                request: &PermissionRequest<'_>,
            ) -> impl std::future::Future<Output = Review> + Send {
                assert_eq!(
                    *request,
                    PermissionRequest::Domain {
                        domain: "example.com",
                        port: 443
                    }
                );
                std::future::ready(self.0.clone())
            }
            fn domain_needs_approval(
                &self,
                _domain: &str,
                _port: u16,
            ) -> impl std::future::Future<Output = bool> + Send {
                std::future::ready(true)
            }
            fn check_domain(
                &self,
                _domain: &str,
                _port: u16,
            ) -> impl std::future::Future<Output = bool> + Send {
                self.1.fetch_add(1, AtomicOrdering::Relaxed);
                std::future::ready(true)
            }
        }
        let request = DomainRequest::new("example.com", 443);
        for (review, allowed, asked) in [
            (Review::Approved, true, 0),
            (Review::Denied("no".into()), false, 0),
            (Review::Passed, true, 1),
        ] {
            let handler = Arc::new(Domains(review, AtomicUsize::new(0)));
            let policy = PermissionNetworkPolicy {
                permission_handler: handler.clone(),
            };
            assert_eq!(policy.check(&request).await, allowed);
            assert_eq!(handler.1.load(AtomicOrdering::Relaxed), asked);
        }
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

        let parent_registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        let parent = tool.with_registry(parent_registry);
        let child = parent
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        assert_eq!(
            parent.prepared_tools_context().unwrap(),
            child.prepared_tools_context().unwrap()
        );
        assert!(child.job_registry().list().await.is_empty());
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
        let child = parent
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");

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
            carry: None,
            script: "/bin/pwd".to_string(),
            mode: TerminalMode::Sandboxed,
            backend: ExecutionBackend::Local(tool.shell_sessions.local_tools().unwrap().clone()),
            stdin_alerts: None,
            lifetime: JobLifetime::Foreground {
                alerts: async_channel::unbounded().0,
            },
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

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_reading_stdin_gets_eof_at_once() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        let started = std::time::Instant::now();
        let result = tool
            .call(
                terminal_args("read all of stdin", "cat"),
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");

        let command = result.command().expect("terminal returns a command result");
        assert!(
            matches!(command.outcome, CommandOutcome::Exited { code: 0 }),
            "the call must exit 0, not be killed: {:?}",
            command.outcome
        );
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "cat must see EOF instead of waiting out the limit"
        );
    }

    #[cfg(unix)]
    fn terminal_args(description: &str, script: &str) -> TerminalArgs {
        TerminalArgs {
            description: description.to_string(),
            script: script.to_string(),
            mode: TerminalExecutionMode::Sandboxed,
            ssh_server_id: None,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            max_lines: 50,
            raw: false,
        }
    }

    /// Starts `script` as a background job and returns its task id.
    #[cfg(unix)]
    async fn start_background_job(
        tool: &TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal, Configured>,
        script: &str,
    ) -> String {
        let result = tool
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: terminal_args("background job", script),
                    timeout: 30,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal_background should start the job");
        match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Running { task_id, .. }) => task_id.clone(),
            other => panic!("background job needs a task id, got {other:?}"),
        }
    }

    /// Collects the alerts that arrive within `window`.
    #[cfg(unix)]
    async fn alerts_within(receiver: &BackgroundTaskReceiver, window: Duration) -> Vec<JobAlert> {
        let deadline = std::time::Instant::now() + window;
        let mut alerts = Vec::new();
        while std::time::Instant::now() < deadline {
            alerts.extend(receiver.take_job_alerts());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        alerts
    }

    /// A foreground container command must see stdin EOF at once: the input
    /// sender is dropped before the exec runs. Catches the regression where
    /// the sender lived until the call returned and `cat` waited out the
    /// limit.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn container_foreground_command_reading_stdin_gets_eof_at_once() {
        /// A container executor that asserts its stdin is already at EOF.
        struct ForegroundStdinEofExec;

        impl crate::ContainerExec for ForegroundStdinEofExec {
            async fn exec(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                stdin_rx: Receiver<Vec<u8>>,
                _lifetime: ExecLifetime,
            ) -> Result<ContainerExecOutcome, String> {
                let eof = futures_lite::future::or(async { Some(stdin_rx.recv().await) }, async {
                    async_io::Timer::after(Duration::from_secs(5)).await;
                    None
                })
                .await
                .expect("a foreground exec's stdin must be EOF at once, not left open");
                assert!(eof.is_err(), "a foreground exec's stdin must carry no data");
                Ok(ContainerExecOutcome::Completed {
                    output: std::process::Output {
                        status: std::os::unix::process::ExitStatusExt::from_raw(0),
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    },
                    killed: Vec::new(),
                })
            }
        }

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let registry = Arc::new(ToolRegistryBuilder::new().build(dir.path()));
        let (job_registry, job_registry_service) = job_registry_channel();
        tokio::spawn(async move { job_registry_service.serve().await });
        let container_tools =
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::bundle());
        let runtime = ContainerShellRuntime::new(
            "aither-test-foreground-eof",
            crate::ContainerIpcRoute::new(
                "host.docker.internal",
                crate::BridgeAddress::try_from(std::net::IpAddr::from([127, 0, 0, 1]))
                    .expect("loopback binds the bridge"),
            ),
            crate::AnyContainerExec::new(Arc::new(ForegroundStdinEofExec)),
        );
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let (_pid, output) = execute_container_background(ContainerBackgroundExecution {
            executor: executor_core::tokio::TokioGlobal,
            registry: &registry,
            task_id: "task-test",
            execution_id: "exec-test",
            script: "cat",
            mode: TerminalMode::Sandboxed,
            container_runtime: &runtime,
            prepared_tools: &container_tools,
            ipc_commands: &[],
            job_registry: &job_registry,
            stdin_alerts: None,
            lifetime: JobLifetime::Foreground {
                alerts: async_channel::unbounded().0,
            },
            background_startup: &BackgroundStartup::new(startup_tx),
            priority: ProcessPriority::default(),
            confinement: None,
            carry: None,
        })
        .await
        .expect("a foreground container exec should complete at EOF");
        assert_eq!(output.status.code(), Some(0));
    }

    /// A job whose `reader` script blocks on stdin yields exactly one alert
    /// until it is fed, and finishes once it is.
    #[cfg(unix)]
    async fn stdin_reader_alerts_once_and_resumes_when_fed(reader: &str) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let background_receiver = tool.background_receiver();
        let task_id = start_background_job(&tool, reader).await;

        let alerts = alerts_within(&background_receiver, Duration::from_secs(8)).await;
        assert!(
            matches!(
                alerts.as_slice(),
                [JobAlert::StdinBlocked(job)] if job.task_id == task_id
            ),
            "a blocked reader must yield exactly one stdin alert: {alerts:?}"
        );

        InputTerminalTool::new(tool.job_registry())
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
        let terminal_result = completed.result.expect("the fed job should complete");
        let stdout =
            output_text(&terminal_result.stdout).expect("completed task should include stdout");
        assert!(stdout.contains("hello lexo"), "got: {stdout}");
        assert!(
            background_receiver.take_job_alerts().is_empty(),
            "a fed job that finished must not alert again"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_blocked_on_stdin_yields_one_alert() {
        stdin_reader_alerts_once_and_resumes_when_fed("read name; printf 'hello %s\\n' \"$name\"")
            .await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_grandchild_blocked_on_stdin_yields_one_alert() {
        stdin_reader_alerts_once_and_resumes_when_fed(
            "( ( read name; printf 'hello %s\\n' \"$name\" ); true )",
        )
        .await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_that_never_reads_stdin_yields_no_alert() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let background_receiver = tool.background_receiver();
        let task_id = start_background_job(&tool, "sleep 3; echo finished").await;

        let alerts = alerts_within(&background_receiver, Duration::from_secs(4)).await;
        assert!(alerts.is_empty(), "no alert expected: {alerts:?}");
        wait_for_completed_task(&background_receiver, &task_id).await;
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

    /// What a test leaves running after its shell exits.
    ///
    /// `Plain` is a `sleep` in the shell's process group. `Setsid` is a helper
    /// program that calls `setsid` and stays alive, so it also leaves the
    /// group and the session. The helper is built here because macOS ships no
    /// `setsid` binary, and because the kernel withholds the environment of
    /// Apple's own binaries (`sleep`, `perl`), which is what finds a `setsid`
    /// descendant once its parent is gone.
    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum Descendant {
        Plain,
        Setsid,
    }

    /// The name of the `setsid` helper program, and so of its process.
    #[cfg(unix)]
    const SETSID_HELPER: &str = "setsid-helper";

    /// The directory the sandbox may execute programs from.
    #[cfg(unix)]
    fn exec_dir<P, E>(tool: &TerminalTool<P, E, Configured>) -> PathBuf {
        tool.shell_sessions
            .local_tools()
            .expect("the terminal has local tools")
            .bin_dir()
            .clone()
    }

    /// Builds the `setsid` helper into `dir` and returns its path. The sandbox
    /// executes only what lies under a granted directory, so `dir` is the
    /// prepared tools' [`exec_dir`].
    #[cfg(unix)]
    fn build_setsid_helper(dir: &Path) -> PathBuf {
        let source = dir.join("setsid_helper.rs");
        std::fs::write(
            &source,
            "extern \"C\" { fn setsid() -> i32; }\n\
             fn main() {\n\
                 let seconds: u64 = std::env::args().nth(1).expect(\"seconds\").parse().expect(\"seconds\");\n\
                 // SAFETY: `setsid` takes no arguments.\n\
                 if unsafe { setsid() } < 0 { std::process::exit(1); }\n\
                 std::thread::sleep(std::time::Duration::from_secs(seconds));\n\
             }\n",
        )
        .expect("helper source should be written");
        let binary = dir.join(SETSID_HELPER);
        let status = std::process::Command::new("rustc")
            .arg("--edition=2021")
            .arg("-o")
            .arg(&binary)
            .arg(&source)
            .status()
            .expect("rustc should run");
        assert!(status.success(), "the setsid helper should compile");
        binary
    }

    #[cfg(unix)]
    impl Descendant {
        /// The name the descendant's process has once it ran `exec`.
        const fn process_name(self) -> &'static str {
            match self {
                Self::Plain => "sleep",
                Self::Setsid => SETSID_HELPER,
            }
        }

        /// A command that starts the descendant for `seconds` detached from
        /// the shell's output, records its pid in `child.pid` and returns once
        /// the child had time to `exec`.
        fn script(self, exec_dir: &Path, seconds: u32) -> String {
            match self {
                Self::Plain => {
                    format!("sleep {seconds} > /dev/null 2>&1 & echo $! > child.pid; sleep 0.3")
                }
                Self::Setsid => format!(
                    "'{}' {seconds} > /dev/null 2>&1 & echo $! > child.pid; sleep 0.3",
                    build_setsid_helper(exec_dir).display()
                ),
            }
        }
    }

    /// The pid `child.pid` in `dir` holds, once the shell wrote it.
    #[cfg(unix)]
    async fn recorded_child_pid(dir: &Path) -> i32 {
        for _ in 0..250 {
            // The redirect creates the file before `echo` writes it, so an
            // empty read only means the pid is not there yet.
            if let Ok(text) = async_fs::read_to_string(dir.join("child.pid")).await
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            async_io::Timer::after(Duration::from_millis(20)).await;
        }
        panic!("the descendant never recorded its pid");
    }

    /// Whether `pid` is a live process.
    #[cfg(unix)]
    fn process_exists(pid: i32) -> bool {
        // SAFETY: signal 0 probes process existence.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    /// Fails unless `pid` is gone within a few seconds.
    #[cfg(unix)]
    async fn assert_process_gone(pid: i32, context: &str) {
        for _ in 0..250 {
            if !process_exists(pid) {
                return;
            }
            async_io::Timer::after(Duration::from_millis(20)).await;
        }
        panic!("process {pid} survived: {context}");
    }

    #[cfg(unix)]
    async fn foreground_command_leaving_a_descendant_kills_it_and_alerts_once(
        descendant: Descendant,
    ) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let receiver = tool.background_receiver();

        let result = tool
            .call(
                terminal_args(
                    "leave a descendant",
                    &descendant.script(&exec_dir(&tool), 300),
                ),
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let command = result.command().expect("terminal returns a command result");
        assert!(
            matches!(command.outcome, CommandOutcome::Exited { code: 0 }),
            "{:?}",
            command.outcome
        );

        let child = recorded_child_pid(tool.working_dir()).await;
        assert_process_gone(
            child,
            &format!("{descendant:?} descendant of a foreground command"),
        )
        .await;
        let alerts = receiver.take_job_alerts();
        match alerts.as_slice() {
            [JobAlert::DescendantsKilled(killed)] => {
                assert_eq!(killed.omitted, 0);
                // The name is read after the freeze, once the child exec'd.
                assert_eq!(
                    killed
                        .processes
                        .iter()
                        .map(|process| i64::from(process.pid))
                        .collect::<Vec<_>>(),
                    [i64::from(child)]
                );
                assert_eq!(killed.processes[0].name, descendant.process_name());
            }
            other => panic!("expected one descendants-killed alert, got {other:?}"),
        }
        assert!(
            receiver.take_job_alerts().is_empty(),
            "the same event must not alert twice"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_leaving_a_descendant_kills_it_and_alerts_once_plain() {
        foreground_command_leaving_a_descendant_kills_it_and_alerts_once(Descendant::Plain).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_leaving_a_setsid_descendant_kills_it_and_alerts_once() {
        foreground_command_leaving_a_descendant_kills_it_and_alerts_once(Descendant::Setsid).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_background_jobs_joined_by_wait_run_without_an_alert() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let receiver = tool.background_receiver();

        let result = tool
            .call(
                terminal_args(
                    "run two jobs and join them",
                    "(sleep 1; echo one > one.txt) & (sleep 1; echo two > two.txt) & wait",
                ),
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let command = result.command().expect("terminal returns a command result");
        assert!(
            matches!(command.outcome, CommandOutcome::Exited { code: 0 }),
            "{:?}",
            command.outcome
        );
        for name in ["one.txt", "two.txt"] {
            assert!(
                tool.working_dir().join(name).exists(),
                "{name} must be written before the command returns"
            );
        }
        assert!(
            receiver.take_job_alerts().is_empty(),
            "a joined job is no survivor"
        );
    }

    /// Starts the background `script`, waits until its shell has exited with the
    /// descendant still alive, and returns the task id and the descendant's pid.
    #[cfg(unix)]
    async fn background_job_whose_shell_exited(
        tool: &TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal, Configured>,
        script: &str,
        timeout: u64,
    ) -> (String, i32) {
        let result = tool
            .background_tool()
            .call(
                TerminalBackgroundArgs {
                    terminal: terminal_args("background job", script),
                    timeout,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal_background should start the job");
        let task_id = match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Running { task_id, .. }) => task_id.clone(),
            other => panic!("background job needs a task id, got {other:?}"),
        };
        let child = recorded_child_pid(tool.working_dir()).await;
        // Past the shell's exit and several of the tree's checks.
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            process_exists(child),
            "the descendant must outlive the shell"
        );
        (task_id, child)
    }

    #[cfg(unix)]
    async fn background_job_stays_running_until_its_descendant_exits(descendant: Descendant) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        let (task_id, child) =
            background_job_whose_shell_exited(&tool, &descendant.script(&exec_dir(&tool), 3), 30)
                .await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the job is registered");
        assert_eq!(
            info.status,
            JobStatus::Running,
            "the job runs while its descendant does"
        );
        assert!(
            tool.job_registry()
                .format_running_jobs()
                .await
                .contains(&task_id),
            "the running-jobs context lists the job"
        );

        assert_process_gone(child, "the descendant's own exit").await;
        let receiver = tool.background_receiver();
        let completed = wait_for_completed_task(&receiver, &task_id).await;
        assert!(completed.result.is_ok(), "{completed:?}");
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the finished job stays in the registry");
        assert_eq!(info.status, JobStatus::Completed { exit_code: 0 });
        assert!(
            tool.job_registry().format_running_jobs().await.is_empty(),
            "a finished job is no longer listed"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_stays_running_until_its_plain_descendant_exits() {
        background_job_stays_running_until_its_descendant_exits(Descendant::Plain).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_stays_running_until_its_setsid_descendant_exits() {
        background_job_stays_running_until_its_descendant_exits(Descendant::Setsid).await;
    }

    #[cfg(unix)]
    async fn killing_a_background_job_kills_its_descendant(descendant: Descendant) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        let (task_id, child) =
            background_job_whose_shell_exited(&tool, &descendant.script(&exec_dir(&tool), 300), 30)
                .await;
        assert!(
            tool.job_registry().kill_by_task_id(&task_id).await,
            "a job with a live descendant is killable"
        );
        assert_process_gone(
            child,
            &format!("terminal_kill of a job with a {descendant:?} descendant"),
        )
        .await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the killed job stays in the registry");
        assert_eq!(info.status, JobStatus::Killed);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn killing_a_background_job_kills_its_plain_descendant() {
        killing_a_background_job_kills_its_descendant(Descendant::Plain).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn killing_a_background_job_kills_its_setsid_descendant() {
        killing_a_background_job_kills_its_descendant(Descendant::Setsid).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_timeout_kills_a_descendant_that_outlived_the_shell() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        // The timeout fires while the shell is long gone: the job is only its
        // descendant by then.
        let (task_id, child) = background_job_whose_shell_exited(
            &tool,
            &Descendant::Setsid.script(&exec_dir(&tool), 300),
            2,
        )
        .await;
        assert_process_gone(child, "the timeout of a job whose shell exited").await;
        let receiver = tool.background_receiver();
        wait_for_completed_task(&receiver, &task_id).await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the finished job stays in the registry");
        assert_eq!(info.status, JobStatus::Killed);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagents_cleanup_kills_a_descendant_that_outlived_the_shell() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        let (_task_id, child) = background_job_whose_shell_exited(
            &tool,
            &Descendant::Plain.script(&exec_dir(&tool), 300),
            30,
        )
        .await;
        // What `kill_running_terminal_jobs` and the subagent run's cleanup call.
        assert_eq!(tool.job_registry().kill_running().await, 1);
        assert_process_gone(child, "the cleanup of a subagent's job").await;
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
            let alerts = background_receiver.take_job_alerts();
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
        assert!(
            matches!(&alerts[0], JobAlert::LongRunning(job) if job.task_id == task_id),
            "expected the long-running alert for the job: {alerts:?}"
        );
        assert!(
            background_receiver.take_job_alerts().is_empty(),
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
            None,
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

        let bare = wrap_container_script(
            "echo ok",
            &[],
            None,
            priority,
            &tools_dir,
            "/workspace",
            None,
        )
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
            None,
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
            None,
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
            None,
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

    /// The shell sessions of a terminal whose home is the acceptance
    /// container, reaching the host IPC bridge through the docker gateway.
    #[cfg(all(unix, aither_container_acceptance))]
    async fn acceptance_shell_sessions(
        container: &TestDockerContainer,
        bundle: &crate::tool_runtime::ToolBundle,
        workspace: &Path,
    ) -> ShellSessionRegistry {
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
        )
        .with_workspace_mount(MountRoot::new(
            workspace,
            ContainerPath::new(CONTAINER_EXEC_WORKING_DIR).expect("the workspace is absolute"),
        ));

        // The container backend needs its own prepared tools: the registry
        // entry names the container and its `/opt/aither/tools` bin dir, which
        // the wrapped script puts on `PATH` and heel must grant execute on.
        let container_tools = crate::tool_runtime::tests::container_tools(bundle);
        ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
            ssh: false,
        })
        .with_container_runtime(runtime)
        .with_test_container_tools(container_tools, container.0.clone())
    }

    /// In a real container the main agent's command runs unconfined, while a
    /// subagent's runs under its per-command sandbox: it writes its own space
    /// and nothing else of the workspace.
    #[cfg(all(unix, aither_container_acceptance))]
    #[tokio::test]
    async fn container_subagent_is_confined_while_the_main_agent_is_not() {
        let (container, bundle, _tools_cache) = start_acceptance_container().await;
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let shell_sessions = acceptance_shell_sessions(&container, &bundle, dir.path()).await;
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
        let parent = tool.with_registry(Arc::new(ToolRegistryBuilder::new().build(dir.path())));
        let child = parent
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        // The acceptance container does not really mount the host workspace,
        // so the subagent's space is created inside it too.
        let root = dunce::canonicalize(dir.path()).expect("workspace root should resolve");
        let space = child
            .working_dir()
            .strip_prefix(&root)
            .expect("the space is inside the workspace");
        let container_space = format!("{CONTAINER_EXEC_WORKING_DIR}/{}", space.display());
        let made = std::process::Command::new("docker")
            .args(["exec", &container.0, "mkdir", "-p", &container_space])
            .output()
            .expect("docker exec should run");
        assert!(made.status.success(), "{made:?}");

        let host_workspace_write =
            |name: &str| format!("touch {CONTAINER_EXEC_WORKING_DIR}/{name}");
        assert_eq!(
            exit_code_of(&parent, &host_workspace_write("main.txt")).await,
            0,
            "the main agent is unconfined in its container"
        );
        assert_ne!(
            exit_code_of(&child, &host_workspace_write("sub.txt")).await,
            0,
            "a subagent cannot write the workspace root"
        );
        assert_eq!(
            exit_code_of(&child, &format!("touch {container_space}/own.txt")).await,
            0,
            "a subagent writes its own space"
        );
    }

    #[cfg(all(unix, aither_container_acceptance))]
    #[tokio::test]
    async fn container_command_runs_with_configured_nice_and_reaches_ipc() {
        let (container, bundle, _tools_cache) = start_acceptance_container().await;

        let shell_sessions = acceptance_shell_sessions(&container, &bundle, Path::new("/")).await;

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

    // ---- subagent authority -------------------------------------------------

    type ConfiguredTestTool =
        TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal, Configured>;

    /// Runs `script` in `tool`'s sandbox and returns its exit code.
    #[cfg(unix)]
    async fn exit_code_of(tool: &ConfiguredTestTool, script: &str) -> i32 {
        let result = tool
            .call(terminal_args("authority probe", script), ToolContext::new())
            .await
            .expect("terminal call should succeed");
        match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Exited { code }) => *code,
            other => panic!("the probe should run to completion, got {other:?}"),
        }
    }

    #[cfg(unix)]
    async fn authority_terminal(dir: &Path) -> (tempfile::TempDir, ConfiguredTestTool) {
        configured_test_terminal(
            dir,
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await
    }

    /// Launches a subagent of `parent` the way its launcher does, with a
    /// launch history of its own.
    async fn launch(
        parent: &ConfiguredTestTool,
        grant: SubagentGrant,
    ) -> Result<ConfiguredTestTool, SubagentGrantError> {
        parent
            .subagent(&mut LaunchHistory::default(), grant, None, None)
            .await
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn subagent_cannot_write_the_session_root_but_writes_its_space_and_granted_worktrees() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, parent) = authority_terminal(dir.path()).await;
        let granted = dir.path().join("granted");
        let ungranted = dir.path().join("ungranted");
        std::fs::create_dir_all(&granted).unwrap();
        std::fs::create_dir_all(&ungranted).unwrap();
        let child = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .expect("a writable directory can be granted");

        let touch = |path: &Path| format!("touch '{}'", path.display());
        let root_file = dir.path().join("root.txt");
        assert_eq!(
            exit_code_of(&parent, &touch(&root_file)).await,
            0,
            "the main agent writes its own space"
        );
        std::fs::remove_file(&root_file).unwrap();

        assert_ne!(
            exit_code_of(&child, &touch(&root_file)).await,
            0,
            "the session root is read-only to a subagent"
        );
        assert!(!root_file.exists());
        assert_ne!(
            exit_code_of(&child, &touch(&ungranted.join("x"))).await,
            0,
            "what the parent can write but did not grant is read-only"
        );
        assert_eq!(
            exit_code_of(&child, &touch(&child.working_dir().join("own.txt"))).await,
            0,
            "a subagent writes its own space"
        );
        assert!(
            child
                .working_dir()
                .starts_with(dunce::canonicalize(dir.path()).unwrap())
        );
        assert_ne!(child.working_dir(), parent.working_dir());
        assert_eq!(
            exit_code_of(&child, &touch(&granted.join("g.txt"))).await,
            0,
            "a subagent writes the worktrees it was granted"
        );
    }

    /// Runs `script` in `tool` and returns its trimmed standard output.
    #[cfg(unix)]
    async fn stdout_of(tool: &ConfiguredTestTool, script: &str) -> String {
        let result = tool
            .call(terminal_args("carry probe", script), ToolContext::new())
            .await
            .expect("terminal call should succeed");
        render_terminal_result(&result).trim().to_string()
    }

    /// The directory `pwd` prints for the next command.
    #[cfg(unix)]
    async fn pwd_of(tool: &ConfiguredTestTool) -> PathBuf {
        PathBuf::from(stdout_of(tool, "pwd").await)
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_successful_cd_carries_to_the_next_foreground_command() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();
        assert_eq!(pwd_of(&tool).await, root);

        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(pwd_of(&tool).await, root.join("sub"));
        // Relative to the carried directory, and a later `cd` carries on.
        assert_eq!(exit_code_of(&tool, "mkdir -p a/b && cd a/b").await, 0);
        assert_eq!(pwd_of(&tool).await, root.join("sub/a/b"));
        // A direct command starts in the carried directory too.
        assert_eq!(
            PathBuf::from(stdout_of(&tool, "/bin/pwd").await),
            root.join("sub/a/b")
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failure_an_outside_cd_and_a_background_cd_leave_the_directory() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();
        stdout_of(&tool, "mkdir sub other && cd sub").await;
        let sub = root.join("sub");
        assert_eq!(pwd_of(&tool).await, sub);

        assert_eq!(
            exit_code_of(&tool, "cd ../other; exit 3").await,
            3,
            "the command fails"
        );
        assert_eq!(pwd_of(&tool).await, sub, "a failed command carries nothing");

        // Readable to the sandbox, yet in none of the terminal's grants.
        assert_eq!(
            exit_code_of(&tool, "cd /usr").await,
            0,
            "the cd itself succeeds"
        );
        assert_eq!(
            pwd_of(&tool).await,
            sub,
            "a directory outside the readable paths carries nothing"
        );

        start_background_job(&tool, "cd ../other && pwd").await;
        for _ in 0..500 {
            if tool.running_tasks.load(Ordering::Acquire) == 0 {
                break;
            }
            async_io::Timer::after(Duration::from_millis(20)).await;
        }
        assert_eq!(tool.running_tasks.load(Ordering::Acquire), 0);
        assert_eq!(pwd_of(&tool).await, sub, "a background cd carries nothing");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn an_exported_variable_does_not_reach_the_next_command() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        assert_eq!(
            stdout_of(
                &tool,
                "export CARRY_PROBE=set; mkdir sub; cd sub; echo $CARRY_PROBE"
            )
            .await,
            "set"
        );
        assert_eq!(
            stdout_of(&tool, "printenv CARRY_PROBE; echo rc=$?").await,
            "rc=1"
        );
        assert_eq!(
            pwd_of(&tool).await,
            dunce::canonicalize(dir.path()).unwrap().join("sub")
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagents_cd_does_not_carry_over() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, parent) = authority_terminal(dir.path()).await;
        let child = launch(&parent, SubagentGrant::new())
            .await
            .expect("a subagent without grants launches");
        let space = child.working_dir().clone();

        assert_eq!(exit_code_of(&child, "mkdir sub && cd sub").await, 0);
        assert_eq!(
            pwd_of(&child).await,
            space,
            "the subagent starts in its space"
        );
        // Nor does it disturb the main agent's directory.
        assert_eq!(
            pwd_of(&parent).await,
            dunce::canonicalize(dir.path()).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn container_directory_report_is_written_by_the_rendered_script() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let report = dir.path().join("it's report");
        let script = carry_cwd_script("mkdir sub && cd sub", &report.to_string_lossy())
            .expect("the trailer should render");
        let status = std::process::Command::new("sh")
            .args(["-c", &script])
            .current_dir(dir.path())
            .status()
            .expect("sh should run");
        assert!(status.success());
        let written = std::fs::read_to_string(&report).expect("the trailer writes the report");
        assert_eq!(
            PathBuf::from(written.trim_end()),
            dunce::canonicalize(dir.path().join("sub")).unwrap()
        );
        // The script's own exit status survives the trailer, whether the
        // script ends through `exit` or its last command fails.
        for (script, code) in [("exit 7", 7), ("cd sub; (exit 5)", 5)] {
            let status = std::process::Command::new("sh")
                .args([
                    "-c",
                    &carry_cwd_script(script, &report.to_string_lossy()).unwrap(),
                ])
                .current_dir(dir.path())
                .status()
                .expect("sh should run");
            assert_eq!(status.code(), Some(code), "{script}");
        }
        // The script's own syntax cannot swallow the trailer: an unterminated
        // here-document holding a quote still ends before the report.
        std::fs::remove_file(&report).expect("the report exists");
        let output = std::process::Command::new("sh")
            .args([
                "-c",
                &carry_cwd_script("cd sub; cat <<EOF\nit's", &report.to_string_lossy()).unwrap(),
            ])
            .current_dir(dir.path())
            .output()
            .expect("sh should run");
        assert!(output.status.success());
        // bash ends the unterminated body with a newline, dash does not.
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim_end(), "it's");
        let written = std::fs::read_to_string(&report).expect("the trailer writes the report");
        assert_eq!(
            PathBuf::from(written.trim_end()),
            dunce::canonicalize(dir.path().join("sub")).unwrap()
        );
    }

    /// Asserts `receiver` holds exactly one alert: a reset of the carried
    /// directory to `dir` for `reason`.
    #[cfg(unix)]
    fn assert_directory_reset(
        receiver: &BackgroundTaskReceiver,
        dir: impl AsRef<Path>,
        reason: DirectoryResetReason,
    ) {
        let alerts = receiver.take_job_alerts();
        match alerts.as_slice() {
            [JobAlert::DirectoryReset(reset)] => {
                assert_eq!(reset.dir, dir.as_ref());
                assert_eq!(reset.reason, reason);
            }
            other => panic!("expected one directory reset, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_carried_directory_raises_no_alert() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let receiver = tool.background_receiver();

        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(exit_code_of(&tool, "true").await, 0);
        assert_eq!(exit_code_of(&tool, "pwd").await, 0);
        // A failure that stayed in its directory, and a direct command.
        assert_ne!(exit_code_of(&tool, "false").await, 0);
        assert_ne!(exit_code_of(&tool, "ls /nonexistent-directory").await, 0);
        assert!(receiver.take_job_alerts().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_command_alerts_once_with_the_directory_commands_run_in() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let receiver = tool.background_receiver();
        let root = dunce::canonicalize(dir.path()).unwrap();

        assert_ne!(exit_code_of(&tool, "mkdir sub && cd sub && false").await, 0);
        assert_directory_reset(&receiver, &root, DirectoryResetReason::Failed);
        assert_eq!(pwd_of(&tool).await, root);
        assert!(receiver.take_job_alerts().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn exit_without_a_report_alerts_once() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let receiver = tool.background_receiver();
        let root = dunce::canonicalize(dir.path()).unwrap();

        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(exit_code_of(&tool, "cd ..; exit 0").await, 0);
        assert_directory_reset(
            &receiver,
            root.join("sub"),
            DirectoryResetReason::Unreported,
        );
        assert_eq!(pwd_of(&tool).await, root.join("sub"));
        assert!(receiver.take_job_alerts().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_directory_outside_the_readable_paths_alerts_once() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let receiver = tool.background_receiver();
        let root = dunce::canonicalize(dir.path()).unwrap();

        // Readable to the sandbox, yet in none of the terminal's grants.
        assert_eq!(exit_code_of(&tool, "cd /usr").await, 0);
        assert_directory_reset(&receiver, &root, DirectoryResetReason::OutsideReadable);
        assert_eq!(pwd_of(&tool).await, root);
        assert!(receiver.take_job_alerts().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_removed_carried_directory_alerts_once() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let receiver = tool.background_receiver();
        let root = dunce::canonicalize(dir.path()).unwrap();

        assert_eq!(exit_code_of(&tool, "mkdir gone && cd gone").await, 0);
        assert!(receiver.take_job_alerts().is_empty());
        std::fs::remove_dir(root.join("gone")).unwrap();
        assert_eq!(pwd_of(&tool).await, root);
        assert_directory_reset(&receiver, &root, DirectoryResetReason::Removed);
        assert_eq!(pwd_of(&tool).await, root);
        assert!(receiver.take_job_alerts().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_directory_report_maps_through_the_workspace_mount() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let mount = MountRoot::new(
            dir.path(),
            ContainerPath::new("/workspace").expect("absolute"),
        );
        let report = tool.working_dir().join(".cwd-container");
        let carry = CwdCarry {
            start: None,
            report: report.clone(),
            container: Some(ContainerCarry {
                mount: mount.clone(),
                report: mount
                    .container_path(&report)
                    .expect("report is in the mount"),
                start: ContainerPath::new("/workspace").unwrap(),
            }),
            shell: true,
        };
        let receiver = tool.background_receiver();
        assert_eq!(
            carry.container.as_ref().unwrap().report.as_str(),
            format!(
                "/workspace/{}",
                report.file_name().unwrap().to_string_lossy()
            )
        );
        let carried = tool.carried_dir.as_ref().unwrap();

        // A directory inside the mount, spelled as the container sees it.
        std::fs::write(&report, "/workspace/sub/dir\n").unwrap();
        tool.record_carried_dir(&carry, true).await;
        assert_eq!(carried.get(), Some(dir.path().join("sub/dir")));
        assert!(!report.exists(), "the report is consumed");

        assert!(
            receiver.take_job_alerts().is_empty(),
            "a carried directory raises no alert"
        );

        // Outside the mount, after a failure and an empty report: unchanged,
        // and each raises one alert naming where the next command starts.
        std::fs::write(&report, "/etc\n").unwrap();
        tool.record_carried_dir(&carry, true).await;
        assert_eq!(carried.get(), Some(dir.path().join("sub/dir")));
        assert_directory_reset(
            &receiver,
            "/workspace",
            DirectoryResetReason::OutsideWorkspace,
        );
        std::fs::write(&report, "/workspace/other\n").unwrap();
        tool.record_carried_dir(&carry, false).await;
        assert_eq!(carried.get(), Some(dir.path().join("sub/dir")));
        assert_directory_reset(&receiver, "/workspace", DirectoryResetReason::Failed);
        tool.record_carried_dir(&carry, true).await;
        assert_directory_reset(&receiver, "/workspace", DirectoryResetReason::Unreported);
        // A failure that stayed where it started changes nothing.
        std::fs::write(&report, "/workspace\n").unwrap();
        tool.record_carried_dir(&carry, false).await;
        assert!(receiver.take_job_alerts().is_empty());

        // The mapping back to the container for the next start.
        assert_eq!(
            mount
                .container_path(&dir.path().join("sub/dir"))
                .unwrap()
                .as_str(),
            "/workspace/sub/dir"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_grant_outside_the_launchers_authority_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let outside = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, parent) = authority_terminal(dir.path()).await;
        let error = launch(&parent, SubagentGrant::new().with_worktree(outside.path()))
            .await
            .expect_err("a launcher cannot hand on what it cannot write");
        assert!(
            matches!(error, SubagentGrantError::OutsideAuthority(_)),
            "{error}"
        );
    }

    #[cfg(unix)]
    fn commit_script(worktree: &Path, subject: &str) -> String {
        format!(
            "git -c user.name=agent -c user.email=agent@example.com -c commit.gpgSign=false -C '{}' commit --allow-empty -m '{subject}'",
            worktree.display()
        )
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn commit_in_a_linked_worktree_needs_the_commit_grant() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let worktree = crate::subagent::tests::repository_with_linked_worktree(dir.path(), "one");
        let (_cache, parent) = authority_terminal(dir.path()).await;

        let without = launch(&parent, SubagentGrant::new().with_worktree(&worktree))
            .await
            .expect("the worktree can be granted");
        assert_eq!(
            exit_code_of(
                &without,
                &format!("touch '{}'", worktree.join("f.txt").display())
            )
            .await,
            0,
            "the worktree itself is writable"
        );
        assert_ne!(
            exit_code_of(&without, &commit_script(&worktree, "denied")).await,
            0,
            "its git data lives outside the subagent's space"
        );

        let with = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .expect("the worktree can be granted with commit");
        let ledger = with.commit_ledger().expect("a commit grant has a ledger");
        let mark = ledger.mark().await.expect("the namespace can be read");
        assert_eq!(
            exit_code_of(&with, &commit_script(&worktree, "authorized")).await,
            0,
            "a commit grant lets the subagent commit"
        );
        assert_ne!(
            exit_code_of(
                &with,
                &format!(
                    "git -C '{}' update-ref refs/heads/main HEAD",
                    worktree.display()
                )
            )
            .await,
            0,
            "a branch outside the subagent's namespace stays read-only"
        );
        let commits = ledger.since(&mark).await.expect("commits can be listed");
        assert_eq!(commits.len(), 1, "{commits:?}");
        assert_eq!(commits[0].subject, "authorized");
        assert_eq!(commits[0].worktree, dunce::canonicalize(&worktree).unwrap());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_commit_grant_needs_a_linked_worktree_whose_repository_the_launcher_can_write() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let repository = crate::subagent::tests::repository_with_linked_worktree(dir.path(), "two")
            .parent()
            .unwrap()
            .join("repository");
        let plain = dir.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        let (_cache, parent) = authority_terminal(dir.path()).await;

        let checkout = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&repository),
        )
        .await
        .expect_err("a checkout holds its git data inside the grant");
        assert!(
            matches!(checkout, SubagentGrantError::RepositoryCheckout(_)),
            "{checkout}"
        );
        let not_git = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&plain),
        )
        .await
        .expect_err("a plain directory has nothing to commit in");
        assert!(
            matches!(not_git, SubagentGrantError::NotAWorktree(_)),
            "{not_git}"
        );

        // A worktree inside the launcher's space whose repository is not.
        let elsewhere = tempfile::tempdir().expect("temp dir should be created");
        let foreign =
            crate::subagent::tests::repository_with_linked_worktree(elsewhere.path(), "three");
        let inside = dir.path().join("three");
        std::fs::rename(&foreign, &inside).unwrap();
        crate::subagent::tests::git_in(
            &elsewhere.path().join("repository"),
            &["worktree", "repair", inside.to_str().unwrap()],
        );
        let outside = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&inside),
        )
        .await
        .expect_err("the launcher cannot hand on git data it cannot write");
        assert!(
            matches!(outside, SubagentGrantError::OutsideAuthority(_)),
            "{outside}"
        );
    }

    /// A subagent hands on only what it holds: commit where it was granted
    /// commit, and paths inside its own authority.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_nested_launch_never_exceeds_its_launcher() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let worktree = crate::subagent::tests::repository_with_linked_worktree(dir.path(), "four");
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&sibling).unwrap();
        let (_cache, parent) = authority_terminal(dir.path()).await;

        let writer = launch(&parent, SubagentGrant::new().with_worktree(&worktree))
            .await
            .unwrap();
        let escalated = launch(
            &writer,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .expect_err("a subagent without commit cannot grant it");
        assert!(
            matches!(escalated, SubagentGrantError::CommitOutsideAuthority(_)),
            "{escalated}"
        );
        let sideways = launch(&writer, SubagentGrant::new().with_worktree(&sibling))
            .await
            .expect_err("a subagent cannot grant what it cannot write");
        assert!(
            matches!(sideways, SubagentGrantError::OutsideAuthority(_)),
            "{sideways}"
        );

        let committer = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .unwrap();
        let handed_on = launch(
            &committer,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .expect("a subagent hands on the commit grant it holds");
        assert_eq!(handed_on.writable_paths, committer.writable_paths);
        assert!(handed_on.working_dir().starts_with(committer.working_dir()));
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct LaunchProbeArgs {
        /// A directory to grant the launched subagent.
        #[serde(default)]
        worktree: Option<String>,
    }

    /// Launches a subagent from the terminal it is bound to and reports
    /// where it works.
    struct LaunchProbe {
        binding: TerminalBinding,
    }

    impl Tool for LaunchProbe {
        fn name(&self) -> Cow<'static, str> {
            "launch_probe".into()
        }

        type Arguments = LaunchProbeArgs;
        type Res = ToolResult;

        async fn call(
            &self,
            arguments: Self::Arguments,
            _cx: ToolContext,
        ) -> aither_core::Result<Self::Res> {
            let grant = arguments
                .worktree
                .into_iter()
                .fold(SubagentGrant::new(), SubagentGrant::with_worktree);
            let launched = self.binding.launcher().create(grant, None, None).await?;
            Ok(ToolResult::Text {
                text: launched.working_dir().display().to_string(),
            })
        }
    }

    /// A subagent that runs the launching command launches from its own
    /// terminal: the child lives in its space and gets nothing it lacks.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagent_running_the_launch_command_launches_with_its_own_authority() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let granted = dir.path().join("granted");
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&granted).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        let tool = TerminalTool::new_exact(
            dir.path(),
            TestPermissionHandler::default(),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_terminal_bound_tool("launch_probe", |binding: &TerminalBinding| {
            LaunchProbe {
                binding: binding.clone(),
            }
        });
        let parent = tool.with_registry(Arc::new(builder.build(dir.path().join("outputs"))));

        // The main agent launches the subagent through the command.
        let sibling_arg = sibling.to_str().unwrap();
        let from_main = query_control(
            parent.registry(),
            "launch_probe",
            &["--worktree", sibling_arg],
        )
        .await;
        assert_eq!(from_main["ok"], true, "{from_main}");

        let child = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .unwrap();
        let refused = query_control(
            child.registry(),
            "launch_probe",
            &["--worktree", sibling_arg],
        )
        .await;
        assert_eq!(refused["ok"], false, "{refused}");
        assert!(
            refused["error"]
                .as_str()
                .is_some_and(|error| error.contains("is not writable by the agent launching")),
            "{refused}"
        );
        let nested = query_control(
            child.registry(),
            "launch_probe",
            &["--worktree", granted.to_str().unwrap()],
        )
        .await;
        assert_eq!(nested["ok"], true, "{nested}");
        let grandchild = nested["payload"].to_string();
        let child_space = child.working_dir().display().to_string();
        assert!(
            grandchild.contains(&child_space),
            "the grandchild works inside its caller's space: {grandchild} vs {child_space}"
        );
    }

    #[tokio::test]
    async fn a_subagent_of_a_subagent_gets_the_subagent_registry() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir.path(),
            TestPermissionHandler::default(),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let mut builder = ToolRegistryBuilder::new();
        builder.main_only(|reserved| reserved.configure_tool(ProbeEchoTool));
        let parent = tool.with_registry(Arc::new(builder.build(dir.path().join("outputs"))));
        assert_eq!(parent.registry().role(), crate::CommandRole::Main);

        let child = launch(&parent, SubagentGrant::new()).await.unwrap();
        let grandchild = launch(&child, SubagentGrant::new()).await.unwrap();
        for terminal in [&child, &grandchild] {
            assert_eq!(terminal.registry().role(), crate::CommandRole::Subagent);
            let refused =
                query_control(terminal.registry(), "probe_echo", &["--marker", "x"]).await;
            assert_eq!(refused["ok"], false, "{refused}");
            assert!(
                refused["error"]
                    .as_str()
                    .is_some_and(|error| error.contains("reserved for the main agent")),
                "{refused}"
            );
        }
        assert!(grandchild.working_dir().starts_with(child.working_dir()));
        let served = parent.clone().to_dyn();
        assert!(served.has_command("probe_echo"));
        assert_eq!(served.role(), crate::CommandRole::Main);
        assert_eq!(parent.binding().role(), crate::CommandRole::Main);
        assert_eq!(child.binding().role(), crate::CommandRole::Subagent);
        let child = child.to_dyn();
        assert!(!child.has_command("probe_echo"));
        assert_eq!(child.role(), crate::CommandRole::Subagent);
    }

    /// Records the script every container exec runs.
    #[cfg(unix)]
    struct RecordingExec {
        scripts: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[cfg(unix)]
    impl crate::ContainerExec for RecordingExec {
        fn exec(
            &self,
            _container_id: &str,
            script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send
        {
            self.scripts.lock().unwrap().push(script.to_string());
            std::future::ready(Ok(ContainerExecOutcome::Completed {
                output: std::process::Output {
                    status: std::os::unix::process::ExitStatusExt::from_raw(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                },
                killed: Vec::new(),
            }))
        }
    }

    /// A terminal whose home backend is a recording container, which mounts
    /// `workspace` as `/workspace` when one is given.
    #[cfg(unix)]
    async fn recording_container_terminal(
        dir: &Path,
        workspace: Option<&Path>,
    ) -> (Arc<std::sync::Mutex<Vec<String>>>, ConfiguredTestTool) {
        let scripts = Arc::new(std::sync::Mutex::new(Vec::new()));
        let runtime = ContainerShellRuntime::new(
            "aither-test-confinement",
            crate::ContainerIpcRoute::new(
                "host.docker.internal",
                crate::BridgeAddress::try_from(std::net::IpAddr::from([127, 0, 0, 1]))
                    .expect("loopback binds the bridge"),
            ),
            crate::AnyContainerExec::new(Arc::new(RecordingExec {
                scripts: scripts.clone(),
            })),
        );
        let runtime = match workspace {
            Some(workspace) => runtime.with_workspace_mount(MountRoot::new(
                workspace,
                ContainerPath::new(CONTAINER_EXEC_WORKING_DIR).expect("the workspace is absolute"),
            )),
            None => runtime,
        };
        let container_tools =
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::bundle());
        let shell_sessions = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
            ssh: false,
        })
        .with_container_runtime(runtime)
        .with_test_container_tools(container_tools, "aither-test-confinement".to_string());
        let tool = TerminalTool::new_exact(
            dir,
            TestPermissionHandler::default(),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let parent = tool.with_registry(Arc::new(ToolRegistryBuilder::new().build(dir)));
        (scripts, parent)
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn container_session_runs_the_main_agent_unconfined_and_a_subagent_confined() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let granted = dir.path().join("granted");
        std::fs::create_dir_all(&granted).unwrap();
        let (scripts, parent) = recording_container_terminal(dir.path(), Some(dir.path())).await;
        let child = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .expect("a directory inside the workspace can be granted");

        assert_eq!(exit_code_of(&parent, "echo main-command").await, 0);
        assert_eq!(exit_code_of(&child, "echo subagent-command").await, 0);

        let scripts = scripts.lock().unwrap().clone();
        let main = scripts
            .iter()
            .find(|script| script.contains("main-command"))
            .expect("the main agent's command ran");
        let sub = scripts
            .iter()
            .find(|script| script.contains("subagent-command"))
            .expect("the subagent's command ran");
        assert!(
            !main.contains(" run --"),
            "the container is the main agent's boundary: {main}"
        );
        assert!(sub.contains("\"$MAY_IPC_BIN\" run --working-dir"), "{sub}");
        assert!(!sub.contains("--isolation permissive"), "{sub}");
        let root = dunce::canonicalize(dir.path()).unwrap();
        let space = child
            .working_dir()
            .strip_prefix(&root)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            sub.contains(&format!("--working-dir '/workspace/{space}'")),
            "{sub}"
        );
        let granted_name = granted.file_name().unwrap().to_str().unwrap();
        assert!(
            sub.contains(&format!("--grant '/workspace/{granted_name}=rw'")),
            "{sub}"
        );
    }

    /// A subagent whose paths the container cannot express is refused when
    /// it is launched, not when it first runs a command.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_subagent_outside_the_workspace_mount_is_refused_at_launch() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let mounted = dir.path().join("mounted");
        let granted = dir.path().join("granted");
        std::fs::create_dir_all(&mounted).unwrap();
        std::fs::create_dir_all(&granted).unwrap();

        let (_scripts, parent) = recording_container_terminal(dir.path(), Some(&mounted)).await;
        let error = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .expect_err("the subagent's paths are not inside the mount");
        assert!(
            matches!(error, SubagentGrantError::OutsideContainerMount { .. }),
            "{error}"
        );

        let (_scripts, unmounted) = recording_container_terminal(dir.path(), None).await;
        let error = launch(&unmounted, SubagentGrant::new())
            .await
            .expect_err("a container without a declared mount cannot confine a subagent");
        assert!(
            matches!(error, SubagentGrantError::NoContainerMount),
            "{error}"
        );
    }

    /// A configured local terminal in `dir`, for the kill-guard tests.
    #[cfg(unix)]
    async fn kill_guard_tool(
        dir: &Path,
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
            .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
                local: true,
                container: false,
                ssh: false,
            });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.outputs_dir()));
        (cache, tool.with_registry(registry))
    }

    /// Name- and pattern-based kills are refused before anything of the
    /// command runs, in the foreground and the background alike.
    #[cfg(unix)]
    #[tokio::test]
    async fn name_based_kills_are_refused_before_they_run() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = kill_guard_tool(dir.path()).await;
        let marker = dir.path().join("ran");
        for killer in [
            "pkill -f anything",
            "killall anything",
            "/usr/bin/pkill -f anything",
            "/usr/bin/killall anything",
            "sudo pkill anything",
            "env FOO=1 nice timeout 5 /usr/bin/killall anything",
            "echo anything | xargs pkill",
            "sh -c 'killall anything'",
            "echo $(pkill anything)",
            "(pkill anything)",
        ] {
            let script = format!("touch {}; {killer}", marker.display());
            let error = tool
                .call(
                    terminal_args("name-based kill", &script),
                    ToolContext::new(),
                )
                .await
                .expect_err("a name-based kill must be refused");
            assert!(
                error.to_string().contains("kill that PID"),
                "`{killer}` refusal must tell the agent to kill by PID: {error}"
            );
            let error = tool
                .background_tool()
                .call(
                    TerminalBackgroundArgs {
                        terminal: terminal_args("name-based kill", &script),
                        timeout: 30,
                    },
                    ToolContext::new(),
                )
                .await
                .expect_err("a background name-based kill must be refused");
            assert!(
                error.to_string().contains("kill that PID"),
                "`{killer}` background refusal must tell the agent to kill by PID: {error}"
            );
            assert!(
                !marker.exists(),
                "`{killer}` must be refused before any part of the command runs"
            );
        }
    }

    /// Words that only mention a killer, and kills by PID, still run.
    #[cfg(unix)]
    #[tokio::test]
    async fn kill_by_pid_and_mentions_of_killers_still_run() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = kill_guard_tool(dir.path()).await;

        let result = tool
            .call(
                terminal_args("mention a killer", "echo pkill killall"),
                ToolContext::new(),
            )
            .await
            .expect("a mention of a killer must not be refused");
        assert!(
            render_terminal_result(&result).contains("pkill killall"),
            "{}",
            render_terminal_result(&result)
        );

        let result = tool
            .call(
                terminal_args(
                    "kill by pid",
                    "sleep 300 & pid=$!; kill $pid; wait $pid; echo \"status=$?\"",
                ),
                ToolContext::new(),
            )
            .await
            .expect("kill by PID must not be refused");
        assert!(
            render_terminal_result(&result).contains("status=143"),
            "kill <pid> must terminate the process it names: {}",
            render_terminal_result(&result)
        );
    }
}
