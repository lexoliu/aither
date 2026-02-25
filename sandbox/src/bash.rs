//! Bash tool implementation with leash sandboxing.
//!
//! Provides the main `bash` tool that LLMs use to execute scripts.
//! Scripts run in a sandbox with configurable permission modes.
//!
//! Each `BashTool` creates a shared working directory with four random words
//! (e.g., `amber-forest-thunder-pearl/`). All bash executions share this
//! directory, but each execution gets a fresh sandbox (new TTY/process).

#[cfg(unix)]
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::{
    borrow::Cow,
    io::{Read, Write},
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use aither_core::llm::{Tool, ToolOutput};
use async_channel::{Receiver, Sender};
#[cfg(unix)]
use async_io::Async;
use executor_core::{Executor, Task};
#[cfg(unix)]
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use leash::{
    AllowAll, DomainRequest, IpcRouter, NetworkPolicy, Sandbox, SandboxConfig, SecurityConfig,
    StdioConfig, WorkingDir,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::{
    builtin::builtin_router,
    command::ToolRegistry,
    job_registry::{JobRegistry, job_registry_channel},
    output::{
        Content, INLINE_OUTPUT_LIMIT, OutputEntry, OutputFormat, OutputStore, save_raw_to_file,
    },
    permission::{BashMode, PermissionError, PermissionHandler},
    shell_session::{ShellBackend, ShellSessionRegistry, SshRuntimeProfile, bootstrap_ssh_runtime},
};

/// Generate a random four-word ID (e.g., "amber-forest-thunder-pearl").
fn random_task_id() -> String {
    crate::naming::random_word_slug(4)
}

#[derive(Clone)]
struct PermissionNetworkPolicy<P> {
    permission_handler: Arc<P>,
}

impl<P: PermissionHandler + 'static> NetworkPolicy for PermissionNetworkPolicy<P> {
    async fn check(&self, request: &DomainRequest) -> bool {
        self.permission_handler
            .check_domain(request.target(), request.port())
            .await
    }
}

async fn ensure_mode_allowed<P: PermissionHandler>(
    permission_handler: &P,
    mode: BashMode,
    script: &str,
) -> Result<(), BashError> {
    if !mode.requires_approval() {
        return Ok(());
    }

    let allowed = permission_handler.check(mode, script).await?;
    if allowed {
        Ok(())
    } else {
        Err(BashError::PermissionDenied(mode))
    }
}

/// Execute bash scripts in a sandboxed environment.
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
/// The sandbox provides these commands without network access:
/// - `websearch "query"` - search the web, returns titles/URLs/snippets
/// - `webfetch "url"` - fetch URL content as markdown
/// - `ask "prompt"` - query a fast LLM about piped content (saves context)
/// - `subagent --subagent "<type-or-path>" --prompt "<prompt>"` - launch specialized subagents
/// - `todo` - manage task list
///
/// ## Execution Modes
///
/// `bash` is stateless. Each call selects its own runtime mode.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BashArgs {
    /// The bash script to execute.
    pub script: String,

    /// Runtime execution mode.
    /// - "default": local runtime execution with network enabled
    /// - "unsafe": direct host execution (leash profile only)
    /// - "ssh": execute on a preconfigured SSH server (requires `ssh_server_id`)
    #[serde(default)]
    pub mode: BashExecutionMode,

    /// SSH server identifier used when `mode` is `ssh`.
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BashExecutionMode {
    #[default]
    Default,
    Unsafe,
    Ssh,
}

const fn default_max_lines() -> u32 {
    200
}

const MAX_LINES_CEILING: u32 = 800;

/// Result of a bash execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashResult {
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
}

enum ForegroundDecision {
    Completed(Result<BashResult, String>),
    PromoteToBackground(Option<String>),
}

/// A completed background task result.
#[derive(Debug)]
pub struct CompletedTask {
    /// The task ID (four random words like "amber-forest-thunder-pearl").
    pub task_id: String,
    /// The original script that was executed.
    pub script: String,
    /// The result of execution.
    pub result: Result<BashResult, BashError>,
}

/// Receiver for completed background tasks.
///
/// This can be cloned and used independently of the `BashTool` to poll for
/// completed background tasks. Multiple receivers share the same channel,
/// so completed tasks are distributed among receivers.
#[derive(Clone)]
pub struct BackgroundTaskReceiver {
    rx: Receiver<CompletedTask>,
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
    /// This works by checking the sender count - `BashTool` holds one sender,
    /// and each running background task holds a cloned sender.
    #[must_use]
    pub fn has_running(&self) -> bool {
        self.rx.sender_count() > 1
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
            .finish()
    }
}

/// Factory for creating child bash tools asynchronously.
#[derive(Clone)]
pub struct BashToolFactory {
    tx: Sender<Sender<crate::command::DynBashTool>>,
}

/// Receiver that serves bash tool creation requests.
pub struct BashToolFactoryReceiver {
    rx: Receiver<Sender<crate::command::DynBashTool>>,
}

/// Errors that can occur when requesting a child bash tool.
#[derive(Debug, thiserror::Error)]
pub enum BashToolFactoryError {
    /// The factory service is not running.
    #[error("bash tool factory is not available")]
    Unavailable,
    /// The factory failed to return a tool.
    #[error("bash tool factory failed to return a tool")]
    NoResponse,
}

/// Creates a factory channel pair for spawning child bash tools.
#[must_use]
pub fn bash_tool_factory_channel() -> (BashToolFactory, BashToolFactoryReceiver) {
    let (tx, rx) = async_channel::unbounded();
    (BashToolFactory { tx }, BashToolFactoryReceiver { rx })
}

impl BashToolFactory {
    /// Requests a new child bash tool from the factory service.
    pub async fn create(&self) -> Result<crate::command::DynBashTool, BashToolFactoryError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(reply_tx)
            .await
            .map_err(|_| BashToolFactoryError::Unavailable)?;
        reply_rx
            .recv()
            .await
            .map_err(|_| BashToolFactoryError::NoResponse)
    }
}

impl BashToolFactoryReceiver {
    async fn serve<P, E>(self, bash_tool: BashTool<P, E, Configured>)
    where
        P: PermissionHandler + 'static,
        E: Executor + Clone + 'static,
    {
        while let Ok(reply_tx) = self.rx.recv().await {
            let tool = bash_tool.child().to_dyn();
            if reply_tx.send(tool).await.is_err() {
                warn!("bash tool factory response channel dropped");
            }
        }
    }
}

/// The bash tool for executing scripts in a sandbox.
///
/// Creates a shared working directory with four random words that persists
/// across all executions. Each execution creates a fresh sandbox but shares
/// the working directory.
///
/// The executor type `E` determines how async tasks are spawned for the IPC server.
/// Use `TokioExecutor` when running in a tokio runtime.
/// Marker type for a bash tool without a configured registry.
#[derive(Clone, Debug)]
pub struct Unconfigured;

/// Marker type for a bash tool with a configured registry.
#[derive(Clone)]
pub struct Configured {
    registry: Arc<ToolRegistry>,
}

/// The bash tool for executing scripts in a sandbox.
pub struct BashTool<P, E, State = Unconfigured> {
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
    /// Additional paths that should be writable in the sandbox.
    writable_paths: Vec<PathBuf>,
    /// Additional paths that should be readable (but not writable) in the sandbox.
    readable_paths: Vec<PathBuf>,
    /// Tool registry state.
    registry: State,
}

// Manual Clone impl because P doesn't need to be Clone (we use Arc<P>)
impl<P, E: Clone, State: Clone> Clone for BashTool<P, E, State> {
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
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
            registry: self.registry.clone(),
        }
    }
}

impl<P, E: Executor + Clone + 'static> BashTool<P, E, Unconfigured> {
    /// Injects the shared runtime registry used by `bash/open_ssh/list_ssh`.
    #[must_use]
    pub fn with_shell_sessions(mut self, sessions: ShellSessionRegistry) -> Self {
        self.shell_sessions = sessions;
        self
    }

    /// Sets dynamic runtime availability for bash execution.
    #[must_use]
    pub fn with_shell_runtime_availability(
        self,
        availability: crate::shell_session::ShellRuntimeAvailability,
    ) -> Self {
        let _ = self.shell_sessions.set_availability(availability);
        self
    }

    /// Creates a new bash tool with permission handler and executor.
    ///
    /// Creates a random four-word working directory under the specified parent directory.
    /// The executor is used to spawn async tasks for the IPC server.
    pub async fn new_in(
        parent_dir: impl AsRef<std::path::Path>,
        permission_handler: P,
        executor: E,
    ) -> Result<Self, BashError> {
        let parent_dir = parent_dir.as_ref();

        // Ensure parent directory exists
        async_fs::create_dir_all(parent_dir).await?;

        // Create random four-word working directory
        let working_dir = WorkingDir::random_in(parent_dir)
            .map_err(|e| BashError::SandboxSetup(format!("failed to create working dir: {e}")))?;
        let working_dir_path = working_dir.path().to_path_buf();

        info!(working_dir = %working_dir_path.display(), "created bash tool working directory");

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

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(Default::default()),
            permission_handler: Arc::new(permission_handler),
            executor,
            output_store,
            job_registry,
            completed_rx,
            completed_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            registry: Unconfigured,
        })
    }

    /// Creates a new bash tool with permission handler and executor,
    /// using the provided directory directly as the working directory.
    ///
    /// Unlike `new_in` which creates a random subdirectory, this method
    /// uses the exact path provided. Use this when you want explicit control
    /// over the working directory location.
    pub async fn new_exact(
        working_dir: impl AsRef<std::path::Path>,
        permission_handler: P,
        executor: E,
    ) -> Result<Self, BashError> {
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

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(Default::default()),
            permission_handler: Arc::new(permission_handler),
            executor,
            output_store,
            job_registry,
            completed_rx,
            completed_tx,
            writable_paths: Vec::new(),
            readable_paths: Vec::new(),
            registry: Unconfigured,
        })
    }

    /// Attaches a tool registry to this bash tool, enabling IPC command dispatch.
    #[must_use]
    pub fn with_registry(self, registry: Arc<ToolRegistry>) -> BashTool<P, E, Configured> {
        BashTool {
            working_dir: self.working_dir,
            shell_sessions: self.shell_sessions,
            permission_handler: self.permission_handler,
            executor: self.executor,
            output_store: self.output_store,
            job_registry: self.job_registry,
            completed_rx: self.completed_rx,
            completed_tx: self.completed_tx,
            writable_paths: self.writable_paths,
            readable_paths: self.readable_paths,
            registry: Configured { registry },
        }
    }
}

impl<P, E, State> BashTool<P, E, State>
where
    E: Executor + Clone + 'static,
    State: Clone,
{
    /// Adds additional writable paths to the sandbox configuration.
    ///
    /// These paths will be writable in sandboxed and network modes.
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
    pub fn with_readable_paths(
        mut self,
        paths: impl IntoIterator<Item = impl Into<PathBuf>>,
    ) -> Self {
        self.readable_paths
            .extend(paths.into_iter().map(Into::into));
        self
    }

    /// Creates a child `BashTool` that shares the same sandbox and permission handler
    /// but has independent background task tracking.
    ///
    /// Use this to create bash tools for subagents that:
    /// - Share the same working directory and output store
    /// - Share the same permission handler (security policies enforced consistently)
    /// - Have independent completion channels (no message mixup)
    pub fn child(&self) -> Self {
        let (completed_tx, completed_rx) = async_channel::unbounded();
        Self {
            working_dir: self.working_dir.clone(),
            shell_sessions: self.shell_sessions.clone(),
            permission_handler: self.permission_handler.clone(), // Arc clone - shares handler
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            job_registry: self.job_registry.clone(),
            completed_rx,
            completed_tx,
            writable_paths: self.writable_paths.clone(),
            readable_paths: self.readable_paths.clone(),
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
    /// The returned `BackgroundTaskReceiver` can be used independently of the `BashTool`
    /// to poll for completed background tasks. This is useful for integrating with
    /// the Agent's main loop.
    pub fn background_receiver(&self) -> BackgroundTaskReceiver {
        BackgroundTaskReceiver {
            rx: self.completed_rx.clone(),
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
        // If the sender is the only reference, no tasks are pending
        // But since we clone the sender for each task, we check if channel is empty
        !self.completed_rx.is_empty() || self.completed_tx.sender_count() > 1
    }
}

impl<P, E> BashTool<P, E, Configured>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    const fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry.registry
    }

    /// Starts a background service that produces child bash tools on demand.
    pub fn start_factory_service(&self, receiver: BashToolFactoryReceiver) {
        let tool = self.clone();
        self.executor
            .spawn(async move { receiver.serve(tool).await })
            .detach();
    }

    /// Converts this `BashTool` into a type-erased `DynBashTool`.
    ///
    /// This is useful for creating child bash tools for subagents where
    /// the concrete type cannot be known at compile time.
    pub fn to_dyn(self) -> crate::command::DynBashTool {
        use crate::command::{DynBashTool, DynToolHandler};
        use aither_core::llm::tool::ToolDefinition;
        use serde_json::Value;
        use std::sync::Arc;

        // Create the definition - description comes from BashArgs rustdoc
        let schema = schemars::schema_for!(BashArgs);
        let schema_value: Value = schema.to_value();
        let description = schema_value
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let definition =
            ToolDefinition::from_parts("bash".into(), description.into(), schema_value);

        // Create the handler
        let tool = Arc::new(self);
        let handler: DynToolHandler = Arc::new(move |args: &str| {
            let tool = tool.clone();
            let args_str = args.to_string();
            Box::pin(async move {
                match serde_json::from_str::<BashArgs>(&args_str) {
                    Ok(parsed) => match tool.call(parsed).await {
                        Ok(output) => {
                            let output: aither_core::llm::ToolOutput = output;
                            let text: &str = output.as_str().unwrap_or("");
                            text.to_string()
                        }
                        Err(e) => format!("{{\"error\": \"{e}\"}}"),
                    },
                    Err(e) => format!("{{\"error\": \"Parse error: {e}\"}}"),
                }
            })
        });

        DynBashTool {
            definition,
            handler,
        }
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> Tool
    for BashTool<P, E, Configured>
{
    fn name(&self) -> Cow<'static, str> {
        "bash".into()
    }

    type Arguments = BashArgs;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let task_id = random_task_id();
        let script = arguments.script.clone();
        let execution_id = format!("exec-{task_id}");
        let expect = arguments.expect;
        let max_lines = arguments.max_lines.min(MAX_LINES_CEILING) as usize;
        let raw = arguments.raw;
        let timeout = arguments.timeout;

        let (backend, mode, ssh_target, ssh_runtime, container_id) = match arguments.mode {
            BashExecutionMode::Default => {
                let backend = self
                    .shell_sessions
                    .resolve_local_backend()
                    .map_err(anyhow::Error::msg)?;
                let container_id = if matches!(backend, ShellBackend::Container) {
                    Some(self.shell_sessions.container_id().ok_or_else(|| {
                        anyhow::anyhow!("missing container_id for container backend")
                    })?)
                } else {
                    None
                };
                (backend, BashMode::Network, None, None, container_id)
            }
            BashExecutionMode::Unsafe => {
                let backend = self
                    .shell_sessions
                    .resolve_local_backend()
                    .map_err(anyhow::Error::msg)?;
                if !matches!(backend, ShellBackend::Local) {
                    return Err(anyhow::anyhow!(
                        "unsafe mode is only available in leash local runtime"
                    ));
                }
                (backend, BashMode::Unsafe, None, None, None)
            }
            BashExecutionMode::Ssh => {
                self.shell_sessions
                    .ensure_ssh_available()
                    .map_err(anyhow::Error::msg)?;
                let server_id = arguments
                    .ssh_server_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("ssh_server_id is required for ssh mode"))?;
                let server = self
                    .shell_sessions
                    .resolve_ssh_server(server_id)
                    .map_err(anyhow::Error::msg)?;
                let runtime =
                    bootstrap_ssh_runtime(&server.target, &self.shell_sessions.ssh_authorizer())
                        .await?;
                (
                    ShellBackend::Ssh,
                    BashMode::Network,
                    Some(server.target),
                    Some(runtime),
                    None,
                )
            }
        };

        ensure_mode_allowed(self.permission_handler.as_ref(), mode, &script)
            .await
            .map_err(anyhow::Error::new)?;

        if matches!(backend, ShellBackend::Container)
            && self.shell_sessions.container_exec().is_none()
        {
            return Err(anyhow::anyhow!(
                "missing container executor for container backend"
            ));
        }

        let container_exec = self.shell_sessions.container_exec();
        let working_dir = self.working_dir.clone();
        let writable_paths = self.writable_paths.clone();
        let readable_paths = self.readable_paths.clone();
        let executor = self.executor.clone();
        let registry = self.registry().clone();
        let permission_handler = self.permission_handler.clone();
        let store_dir = self.output_store.dir().to_path_buf();
        let store_dir_for_spawn = store_dir.clone();
        let completed_tx = self.completed_tx.clone();
        let job_registry = self.job_registry.clone();
        let background_mode = Arc::new(AtomicBool::new(timeout == 0));
        let (result_tx, result_rx) = async_channel::bounded(1);
        let (stdin_blocked_tx, stdin_blocked_rx) = async_channel::bounded(1);
        let stdin_blocked_notice = if matches!(backend, ShellBackend::Container) {
            Some(stdin_blocked_tx)
        } else {
            None
        };

        let task_id_for_spawn = task_id.clone();
        let background_mode_for_spawn = background_mode.clone();
        self.executor
            .spawn(async move {
                let result = execute_script_standalone(
                    &working_dir,
                    &writable_paths,
                    &readable_paths,
                    executor,
                    registry,
                    permission_handler,
                    job_registry,
                    &task_id_for_spawn,
                    &execution_id,
                    background_mode_for_spawn,
                    &script,
                    mode,
                    backend,
                    ssh_target.as_deref(),
                    ssh_runtime.clone(),
                    container_id.as_deref(),
                    container_exec.as_ref(),
                    stdin_blocked_notice,
                    expect,
                    &store_dir_for_spawn,
                    max_lines,
                    raw,
                )
                .await;

                let quick_result = match &result {
                    Ok(ok) => Ok(ok.clone()),
                    Err(err) => Err(err.to_string()),
                };
                let _ = result_tx.send(quick_result).await;

                let _ = completed_tx
                    .send(CompletedTask {
                        task_id: task_id_for_spawn,
                        script,
                        result,
                    })
                    .await;
            })
            .detach();

        if timeout == 0 {
            let stdout = start_background_output_redirect(
                &self.job_registry,
                &store_dir,
                &task_id,
                max_lines,
                None,
            )
            .await?;
            let running = BashResult {
                stdout,
                stderr: None,
                exit_code: 0,
                task_id: Some(task_id),
                status: Some("running".to_string()),
            };
            let json = serde_json::to_string(&running).map_err(|e| anyhow::anyhow!(e))?;
            return Ok(ToolOutput::text(json));
        }

        let timeout = std::time::Duration::from_secs(timeout);
        let immediate = futures_lite::future::or(
            async {
                result_rx
                    .recv()
                    .await
                    .ok()
                    .map(ForegroundDecision::Completed)
            },
            async {
                futures_lite::future::or(
                    async {
                        stdin_blocked_rx
                            .recv()
                            .await
                            .ok()
                            .map(|reason| ForegroundDecision::PromoteToBackground(Some(reason)))
                    },
                    async {
                        async_io::Timer::after(timeout).await;
                        Some(ForegroundDecision::PromoteToBackground(None))
                    },
                )
                .await
            },
        )
        .await;

        match immediate {
            Some(ForegroundDecision::Completed(Ok(mut result))) => {
                result.task_id = None;
                result.status = None;

                let failed = result.exit_code != 0;
                let json = serde_json::to_string(&result).map_err(|e| anyhow::anyhow!(e))?;

                if failed {
                    return Err(anyhow::anyhow!(format!("bash command failed: {json}")));
                }

                Ok(ToolOutput::text(json))
            }
            Some(ForegroundDecision::Completed(Err(err))) => Err(anyhow::anyhow!(err)),
            Some(ForegroundDecision::PromoteToBackground(reason)) => {
                background_mode.store(true, Ordering::Release);
                let stdout = start_background_output_redirect(
                    &self.job_registry,
                    &store_dir,
                    &task_id,
                    max_lines,
                    reason.as_deref(),
                )
                .await?;
                let running = BashResult {
                    stdout,
                    stderr: None,
                    exit_code: 0,
                    task_id: Some(task_id),
                    status: Some("running".to_string()),
                };
                let json = serde_json::to_string(&running).map_err(|e| anyhow::anyhow!(e))?;
                Ok(ToolOutput::text(json))
            }
            None => {
                background_mode.store(true, Ordering::Release);
                let stdout = start_background_output_redirect(
                    &self.job_registry,
                    &store_dir,
                    &task_id,
                    max_lines,
                    None,
                )
                .await?;
                let running = BashResult {
                    stdout,
                    stderr: None,
                    exit_code: 0,
                    task_id: Some(task_id),
                    status: Some("running".to_string()),
                };
                let json = serde_json::to_string(&running).map_err(|e| anyhow::anyhow!(e))?;
                Ok(ToolOutput::text(json))
            }
        }
    }
}

async fn start_background_output_redirect(
    job_registry: &JobRegistry,
    store_dir: &PathBuf,
    task_id: &str,
    max_lines: usize,
    promotion_reason: Option<&str>,
) -> Result<OutputEntry, anyhow::Error> {
    let url = save_raw_to_file(store_dir, &[]).await?;
    let output_path = store_dir.join(url.strip_prefix("outputs/").unwrap_or(&url));
    let snapshot = job_registry
        .start_output_redirect(task_id, output_path)
        .await
        .map_err(anyhow::Error::msg)?;
    let (preview, truncated) = preview_first_lines(&snapshot, max_lines);
    let text = match (promotion_reason, preview.is_empty()) {
        (Some(reason), true) => reason.to_string(),
        (Some(reason), false) => format!("{reason}\n{preview}"),
        (None, true) => "(no output yet)".to_string(),
        (None, false) => preview,
    };
    Ok(OutputEntry::Stored {
        url,
        content: Some(Content::Text { text, truncated }),
    })
}

/// Standalone script execution that can be spawned in a background task.
async fn execute_script_standalone<P, E>(
    working_dir: &PathBuf,
    writable_paths: &[PathBuf],
    readable_paths: &[PathBuf],
    executor: E,
    registry: Arc<ToolRegistry>,
    permission_handler: Arc<P>,
    job_registry: JobRegistry,
    task_id: &str,
    execution_id: &str,
    background_mode: Arc<AtomicBool>,
    script: &str,
    mode: BashMode,
    backend: ShellBackend,
    ssh_target: Option<&str>,
    ssh_runtime: Option<SshRuntimeProfile>,
    container_id: Option<&str>,
    container_exec: Option<&Arc<dyn crate::shell_session::ContainerExecObject>>,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
    expect: OutputFormat,
    store_dir: &PathBuf,
    max_lines: usize,
    raw: bool,
) -> Result<BashResult, BashError>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    info!(
        script_len = script.len(),
        ?mode,
        "executing background bash script"
    );
    debug!(script = %script, "script content");

    let ipc_commands = registry.registered_tool_names();

    let (pid, output) = if matches!(backend, ShellBackend::Container) {
        execute_container_background(
            executor.clone(),
            registry.clone(),
            task_id,
            execution_id,
            script,
            mode,
            container_id,
            container_exec,
            &ipc_commands,
            &job_registry,
            stdin_blocked_notice,
        )
        .await?
    } else if matches!(backend, ShellBackend::Ssh) {
        execute_ssh_background(
            task_id,
            execution_id,
            script,
            mode,
            ssh_target,
            ssh_runtime,
            &job_registry,
        )
        .await?
    } else {
        match mode {
            BashMode::Network => {
                execute_sandboxed_background(
                    working_dir,
                    writable_paths,
                    readable_paths,
                    executor.clone(),
                    registry.clone(),
                    task_id,
                    execution_id,
                    script,
                    mode,
                    PermissionNetworkPolicy { permission_handler },
                    &job_registry,
                )
                .await?
            }
            BashMode::Unsafe => {
                execute_unsafe_background(
                    working_dir,
                    writable_paths,
                    readable_paths,
                    executor,
                    registry,
                    task_id,
                    execution_id,
                    script,
                    mode,
                    &job_registry,
                )
                .await?
            }
            BashMode::Sandboxed => {
                return Err(BashError::Execution(
                    "sandboxed mode is unsupported; use mode=default or mode=unsafe".to_string(),
                ));
            }
        }
    };

    let background_output = background_mode.load(Ordering::Acquire);
    let byte_limit = Some(INLINE_OUTPUT_LIMIT);
    let stdout = if background_output {
        match crate::output::save_text_with_line_limit(
            store_dir,
            &output.stdout,
            expect,
            max_lines,
            byte_limit,
        )
        .await
        {
            Ok(entry) => entry,
            Err(err) => {
                job_registry.fail(pid, &err.to_string(), None).await;
                return Err(BashError::Io(err));
            }
        }
    } else {
        let is_text = matches!(expect, OutputFormat::Text | OutputFormat::Auto);
        let compressed = if !raw && is_text && !output.stdout.is_empty() {
            if let Ok(text) = std::str::from_utf8(&output.stdout) {
                crate::output_compress::compress_text(text)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref c) = compressed {
            if let Some(ref raw_text) = c.raw_for_file {
                if let Err(err) =
                    crate::output::save_raw_to_file(store_dir, raw_text.as_bytes()).await
                {
                    warn!(error = %err, "failed to save raw source code output");
                }
            }
        }

        let data_to_save = compressed
            .as_ref()
            .map_or(&output.stdout[..], |c| c.text.as_bytes());

        match crate::output::save_text_with_line_limit(
            store_dir,
            data_to_save,
            expect,
            max_lines,
            byte_limit,
        )
        .await
        {
            Ok(entry) => entry,
            Err(err) => {
                job_registry.fail(pid, &err.to_string(), None).await;
                return Err(BashError::Io(err));
            }
        }
    };

    // Save stderr if non-empty
    let stderr = if output.stderr.is_empty() {
        None
    } else {
        match OutputStore::save_to_dir_with_limit(
            store_dir,
            &output.stderr,
            OutputFormat::Text,
            byte_limit,
        )
        .await
        {
            Ok(entry) => Some(entry),
            Err(err) => {
                job_registry.fail(pid, &err.to_string(), None).await;
                return Err(BashError::Io(err));
            }
        }
    };

    let exit_code = output.status.code().unwrap_or(-1);
    debug!(exit_code, "background script completed");

    let output_path = stdout.stored_path(store_dir);
    job_registry.complete(pid, exit_code, output_path).await;

    Ok(BashResult {
        stdout,
        stderr,
        exit_code,
        task_id: None,
        status: None,
    })
}

async fn execute_sandboxed_background<E, N>(
    working_dir: &PathBuf,
    writable_paths: &[PathBuf],
    readable_paths: &[PathBuf],
    executor: E,
    registry: Arc<ToolRegistry>,
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: BashMode,
    policy: N,
    job_registry: &JobRegistry,
) -> Result<(u32, std::process::Output), BashError>
where
    E: Executor + Clone + 'static,
    N: NetworkPolicy + 'static,
{
    let router = create_ipc_router(registry);
    let config = SandboxConfig::builder()
        .network(policy)
        .working_dir(working_dir)
        .writable_paths(writable_paths)
        .readable_paths(readable_paths)
        .security(SecurityConfig::interactive())
        .ipc(router)
        .build()
        .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

    let sandbox = Sandbox::with_config_and_executor(config, executor.clone())
        .await
        .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

    let mut child = sandbox
        .command("bash")
        .arg("-c")
        .arg(script)
        .stdin(StdioConfig::Piped)
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped)
        .spawn()
        .await
        .map_err(|e| BashError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, task_id, execution_id, script, mode, None)
        .await;

    if let Some(stdin) = child.take_stdin() {
        let input_tx = spawn_terminal_stdin_writer(stdin);
        job_registry.attach_terminal_input(pid, input_tx).await;
    }

    let stdout = child
        .take_stdout()
        .ok_or_else(|| BashError::Execution("missing stdout pipe for sandbox process".into()))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| BashError::Execution("missing stderr pipe for sandbox process".into()))?;

    let (chunk_tx, chunk_rx) = async_channel::unbounded();
    spawn_terminal_reader(stdout, chunk_tx.clone(), false);
    spawn_terminal_reader(stderr, chunk_tx.clone(), true);
    drop(chunk_tx);
    executor
        .spawn(drain_terminal_chunks(job_registry.clone(), pid, chunk_rx))
        .detach();

    let status = match child.wait().await {
        Ok(status) => status,
        Err(error) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            return Err(BashError::Execution(error.to_string()));
        }
    };
    wait_for_terminal_stream_close(job_registry, pid).await;

    let (stdout, stderr) = job_registry
        .terminal_output(pid)
        .await
        .ok_or_else(|| BashError::Execution(format!("missing terminal output for pid {pid}")))?;
    let output = std::process::Output {
        status,
        stdout,
        stderr,
    };

    Ok((pid, output))
}

async fn execute_unsafe_background<E: Executor + Clone + 'static>(
    working_dir: &PathBuf,
    writable_paths: &[PathBuf],
    readable_paths: &[PathBuf],
    executor: E,
    registry: Arc<ToolRegistry>,
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: BashMode,
    job_registry: &JobRegistry,
) -> Result<(u32, std::process::Output), BashError> {
    let router = create_ipc_gateway_router(registry);
    let config = SandboxConfig::builder()
        .network(AllowAll)
        .working_dir(working_dir)
        .writable_paths(writable_paths)
        .readable_paths(readable_paths)
        .security(SecurityConfig::interactive())
        .ipc(router)
        .build()
        .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

    let sandbox = Sandbox::with_config_and_executor(config, executor.clone())
        .await
        .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

    let mut child = sandbox
        .command("bash")
        .arg("-c")
        .arg(script)
        .stdin(StdioConfig::Piped)
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped)
        .spawn()
        .await
        .map_err(|e| BashError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, task_id, execution_id, script, mode, None)
        .await;

    if let Some(stdin) = child.take_stdin() {
        let input_tx = spawn_terminal_stdin_writer(stdin);
        job_registry.attach_terminal_input(pid, input_tx).await;
    }

    let stdout = child
        .take_stdout()
        .ok_or_else(|| BashError::Execution("missing stdout pipe for unsafe process".into()))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| BashError::Execution("missing stderr pipe for unsafe process".into()))?;

    let (chunk_tx, chunk_rx) = async_channel::unbounded();
    spawn_terminal_reader(stdout, chunk_tx.clone(), false);
    spawn_terminal_reader(stderr, chunk_tx.clone(), true);
    drop(chunk_tx);
    executor
        .spawn(drain_terminal_chunks(job_registry.clone(), pid, chunk_rx))
        .detach();

    let status = match child.wait().await {
        Ok(status) => status,
        Err(error) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            return Err(BashError::Execution(error.to_string()));
        }
    };
    wait_for_terminal_stream_close(job_registry, pid).await;
    let (stdout, stderr) = job_registry
        .terminal_output(pid)
        .await
        .ok_or_else(|| BashError::Execution(format!("missing terminal output for pid {pid}")))?;
    let output = std::process::Output {
        status,
        stdout,
        stderr,
    };

    Ok((pid, output))
}

async fn execute_container_background<E: Executor + Clone + 'static>(
    executor: E,
    registry: Arc<ToolRegistry>,
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: BashMode,
    container_id: Option<&str>,
    container_exec: Option<&Arc<dyn crate::shell_session::ContainerExecObject>>,
    ipc_commands: &[String],
    job_registry: &JobRegistry,
    stdin_blocked_notice: Option<async_channel::Sender<String>>,
) -> Result<(u32, std::process::Output), BashError> {
    let container_id = container_id
        .ok_or_else(|| BashError::Execution("missing container_id for container backend".into()))?;
    let exec = container_exec.ok_or_else(|| {
        BashError::Execution("missing container executor for container backend".into())
    })?;

    // Use lower 32 bits of a UUID as a synthetic PID for job tracking.
    let pid = uuid::Uuid::new_v4().as_u128() as u32;
    let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
    job_registry
        .register(pid, task_id, execution_id, script, mode, None)
        .await;
    job_registry.attach_kill_switch(pid, kill_tx).await;

    let ipc_bridge = if ipc_commands.is_empty() {
        None
    } else {
        Some(start_container_ipc_bridge(executor, registry)?)
    };
    let wrapped_script = wrap_container_script(
        script,
        ipc_commands,
        ipc_bridge.as_ref().map(ContainerIpcBridge::port),
    )?;

    let execution = exec
        .exec_boxed(
            container_id,
            &wrapped_script,
            "/workspace",
            kill_rx,
            stdin_blocked_notice,
        )
        .await;

    if let Some(bridge) = ipc_bridge {
        bridge.stop().await;
    }

    match execution {
        Ok(crate::shell_session::ContainerExecOutcome::Completed(output)) => {
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
        Ok(crate::shell_session::ContainerExecOutcome::Killed) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(BashError::Execution("container job killed".to_string()))
        }
        Err(err) => {
            job_registry.fail(pid, &err, None).await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(BashError::Execution(err))
        }
    }
}

async fn execute_ssh_background(
    task_id: &str,
    execution_id: &str,
    script: &str,
    mode: BashMode,
    ssh_target: Option<&str>,
    ssh_runtime: Option<SshRuntimeProfile>,
    job_registry: &JobRegistry,
) -> Result<(u32, std::process::Output), BashError> {
    let target =
        ssh_target.ok_or_else(|| BashError::Execution("missing ssh target".to_string()))?;
    let runtime = ssh_runtime
        .ok_or_else(|| BashError::Execution("missing ssh runtime profile".to_string()))?;

    let remote_cmd = match (runtime, mode) {
        (SshRuntimeProfile::Leash { binary }, BashMode::Network) => format!(
            "{} run --network allow -- /bin/bash -lc {}",
            shell_escape(&binary),
            shell_escape(script)
        ),
        (SshRuntimeProfile::Leash { .. }, BashMode::Unsafe) => {
            return Err(BashError::Execution(
                "unsafe mode is not supported for ssh backend".to_string(),
            ));
        }
        (SshRuntimeProfile::Leash { .. }, BashMode::Sandboxed) => {
            return Err(BashError::Execution(
                "sandboxed mode is not supported for ssh backend".to_string(),
            ));
        }
    };

    let child = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg(remote_cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| BashError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, task_id, execution_id, script, mode, None)
        .await;

    match child.output().await {
        Ok(output) => {
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
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(BashError::Execution(err.to_string()))
        }
    }
}

fn shell_escape(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn wrap_container_script(
    script: &str,
    ipc_commands: &[String],
    ipc_port: Option<u16>,
) -> Result<String, BashError> {
    if ipc_commands.is_empty() {
        return Ok(script.to_string());
    }

    let port = ipc_port.ok_or_else(|| {
        BashError::Execution(
            "missing container IPC endpoint for wrapped script execution".to_string(),
        )
    })?;

    for name in ipc_commands {
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(BashError::Execution(format!(
                "invalid ipc command name for container wrapper: {name}"
            )));
        }
    }

    let escaped_commands = ipc_commands
        .iter()
        .map(|name| shell_escape(name))
        .collect::<Vec<_>>()
        .join(" ");

    let mut wrapped = String::with_capacity(script.len() + 1024);
    wrapped.push_str("MAY_IPC_BIN=\"$(command -v leash-ipc || true)\"; ");
    wrapped.push_str("if [ -z \"$MAY_IPC_BIN\" ]; then echo \"leash-ipc command not found in container\" >&2; exit 127; fi; ");
    wrapped.push_str("MAY_IPC_DIR=\"$(mktemp -d)\"; ");
    wrapped.push_str("cleanup(){ local __may_ipc_status=$?; wait >/dev/null 2>&1 || true; rm -rf \"$MAY_IPC_DIR\"; return $__may_ipc_status; }; trap cleanup EXIT; ");
    wrapped.push_str("for cmd in ");
    wrapped.push_str(&escaped_commands);
    wrapped.push_str("; do ");
    wrapped.push_str("printf '%s\\n' '#!/usr/bin/env bash' \"exec \\\"$MAY_IPC_BIN\\\" \\\"$cmd\\\" \\\"\\$@\\\"\" > \"$MAY_IPC_DIR/$cmd\"; ");
    wrapped.push_str("chmod +x \"$MAY_IPC_DIR/$cmd\"; ");
    wrapped.push_str("done; ");
    wrapped.push_str("export PATH=\"$MAY_IPC_DIR:$PATH\"; ");
    wrapped.push_str("hash -r; ");
    wrapped.push_str("export LEASH_IPC_SOCKET=\"tcp://${MAY_HOST_GATEWAY:-host.docker.internal}:");
    wrapped.push_str(&port.to_string());
    wrapped.push_str("\"; ");
    wrapped.push_str(script);

    Ok(wrapped)
}

struct ContainerIpcBridge {
    shutdown_tx: Sender<()>,
    port: u16,
}

impl ContainerIpcBridge {
    fn port(&self) -> u16 {
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
) -> Result<ContainerIpcBridge, BashError> {
    let listener = StdTcpListener::bind("127.0.0.1:0")
        .map_err(|e| BashError::Execution(format!("failed to bind container IPC tcp port: {e}")))?;
    let local_addr = listener.local_addr().map_err(|e| {
        BashError::Execution(format!("failed to resolve container IPC tcp endpoint: {e}"))
    })?;
    let endpoint = format!("tcp://host.docker.internal:{}", local_addr.port());
    tracing::debug!(
        bind = %local_addr,
        endpoint = %endpoint,
        "starting container IPC bridge"
    );
    listener.set_nonblocking(true).map_err(|e| {
        BashError::Execution(format!(
            "failed to set IPC tcp listener nonblocking mode: {e}"
        ))
    })?;
    let listener = Async::new(listener).map_err(|e| {
        BashError::Execution(format!(
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
) -> Result<ContainerIpcBridge, BashError> {
    Err(BashError::Execution(
        "container IPC bridge requires unix domain sockets".to_string(),
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
                if error.kind() == std::io::ErrorKind::Interrupted =>
            {
                continue;
            }
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

        let method = std::str::from_utf8(&body[1..1 + method_length])
            .map_err(|e| format!("invalid IPC method name utf8: {e}"))?;
        let params = &body[1 + method_length..];
        let cli_args = decode_container_ipc_args(params)?;
        let tool_output = registry.query_tool_handler(method, &cli_args).await;
        let payload_value = serde_json::from_str::<serde_json::Value>(&tool_output)
            .unwrap_or(serde_json::Value::String(tool_output));
        write_container_ipc_success(&mut stream, &payload_value).await?;
    }
}

#[cfg(unix)]
fn decode_container_ipc_args(params: &[u8]) -> Result<Vec<String>, String> {
    let parsed: serde_json::Value =
        leash::rmp_serde::from_slice(params).map_err(|e| format!("invalid IPC params: {e}"))?;
    let args = parsed
        .as_object()
        .and_then(|map| map.get("args"))
        .and_then(serde_json::Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(|value| match value {
                    serde_json::Value::String(text) => Some(text.clone()),
                    serde_json::Value::Number(number) => Some(number.to_string()),
                    serde_json::Value::Bool(flag) => Some(flag.to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(args)
}

#[cfg(unix)]
async fn write_container_ipc_success(
    stream: &mut Async<StdTcpStream>,
    response: &serde_json::Value,
) -> Result<(), String> {
    let payload = leash::rmp_serde::to_vec(response)
        .map_err(|e| format!("failed to encode IPC success payload: {e}"))?;
    write_container_ipc_response(stream, true, &payload).await
}

#[cfg(unix)]
async fn write_container_ipc_error(
    stream: &mut Async<StdTcpStream>,
    message: &str,
) -> Result<(), String> {
    let payload = leash::rmp_serde::to_vec(&message.to_string())
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
        .write_all(&[if success { 1 } else { 0 }])
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
fn create_ipc_router(registry: Arc<ToolRegistry>) -> IpcRouter {
    let mut router = builtin_router();

    // Register all configured tools as IPC commands
    let tool_names = registry.registered_tool_names();
    tracing::info!(tools = ?tool_names, "Creating IPC router with registered tools");
    for name in tool_names {
        router = crate::register_tool_command(router, registry.clone(), &name);
    }

    router
}

fn create_ipc_gateway_router(registry: Arc<ToolRegistry>) -> IpcRouter {
    let mut router = crate::register_ipc_gateway_command(IpcRouter::new(), registry.clone());

    // In unsafe mode, keep tool commands usable (websearch/webfetch/ask/task/todo...),
    // but never override native shell task/process commands like kill/jobs.
    let blocked = ["kill", "jobs"];
    let tool_names = registry.registered_tool_names();
    for name in tool_names {
        if blocked.contains(&name.as_str()) {
            continue;
        }
        router = crate::register_tool_command(router, registry.clone(), &name);
    }

    router
}

/// Errors that can occur during bash execution.
#[derive(Debug, thiserror::Error)]
pub enum BashError {
    /// Permission denied for the requested mode.
    #[error("permission denied for {0:?} mode")]
    PermissionDenied(BashMode),

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

impl<P, E, State> std::fmt::Debug for BashTool<P, E, State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashTool")
            .field("working_dir", &self.working_dir)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use leash::ConnectionDirection;

    use super::*;

    #[derive(Default)]
    struct TestPermissionHandler {
        mode_checks: AtomicUsize,
        domain_checks: AtomicUsize,
        allow_network: bool,
        allow_domain: bool,
    }

    impl PermissionHandler for TestPermissionHandler {
        async fn check(&self, mode: BashMode, _script: &str) -> Result<bool, PermissionError> {
            self.mode_checks.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(match mode {
                BashMode::Sandboxed => true,
                BashMode::Network => self.allow_network,
                BashMode::Unsafe => false,
            })
        }

        async fn check_domain(&self, _domain: &str, _port: u16) -> bool {
            self.domain_checks.fetch_add(1, AtomicOrdering::Relaxed);
            self.allow_domain
        }
    }

    #[tokio::test]
    async fn test_bash_args_defaults() {
        let args: BashArgs =
            serde_json::from_str(r#"{"script":"echo hello","timeout":1}"#).unwrap();
        assert_eq!(args.expect, OutputFormat::Text);
        assert_eq!(args.timeout, 1);
        assert_eq!(args.mode, BashExecutionMode::Default);
    }

    #[tokio::test]
    async fn test_bash_result_serialization() {
        use crate::output::Content;

        let result = BashResult {
            stdout: OutputEntry::Stored {
                url: "outputs/bold-oak-calm-river.txt".to_string(),
                content: Some(Content::Text {
                    text: "preview text".to_string(),
                    truncated: true,
                }),
            },
            stderr: None,
            exit_code: 0,
            task_id: None,
            status: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("bold-oak-calm-river.txt"));
        assert!(!json.contains("stderr")); // should be skipped when None
        assert!(!json.contains("task_id")); // should be skipped when None

        // Test inline output (no URL)
        let result_inline = BashResult {
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
        };
        let json_inline = serde_json::to_string(&result_inline).unwrap();
        assert!(!json_inline.contains("\"url\""));
        assert!(json_inline.contains("\"content\""));

        // Test background task result
        let result_background = BashResult {
            stdout: OutputEntry::Empty,
            stderr: None,
            exit_code: 0,
            task_id: Some("amber-forest-thunder-pearl".to_string()),
            status: Some("running".to_string()),
        };
        let json_bg = serde_json::to_string(&result_background).unwrap();
        assert!(json_bg.contains("\"task_id\":\"amber-forest-thunder-pearl\""));
        assert!(json_bg.contains("\"status\":\"running\""));
    }

    #[tokio::test]
    async fn test_bash_args_timeout() {
        let args: BashArgs =
            serde_json::from_str(r#"{"script":"echo hello","timeout":0}"#).unwrap();
        assert_eq!(args.timeout, 0);

        let args_default =
            serde_json::from_str::<BashArgs>(r#"{"script":"echo hello"}"#).unwrap_err();
        assert!(args_default.to_string().contains("timeout"));
    }

    #[test]
    fn wrap_container_script_refreshes_command_hash_table() {
        let wrapped = wrap_container_script(
            "websearch \"gold price\"",
            &[String::from("websearch")],
            Some(9000),
        )
        .expect("wrap script");
        assert!(wrapped.contains("hash -r;"));
        assert!(wrapped.contains("MAY_IPC_DIR"));
        assert!(wrapped.contains("LEASH_IPC_SOCKET"));
    }

    #[test]
    fn wrap_container_script_preserves_user_script_semantics() {
        let script = "echo '$5,040' && subagent --subagent .skills/slide/subagents/art_direction.md --prompt 'x'";
        let wrapped = wrap_container_script(script, &[String::from("subagent")], Some(9000))
            .expect("wrap script");
        assert!(wrapped.contains(script));
        assert!(!wrapped.contains("set -euo pipefail"));
    }

    #[tokio::test]
    async fn ensure_mode_allowed_requires_network_approval() {
        let handler = TestPermissionHandler {
            allow_network: false,
            allow_domain: true,
            ..Default::default()
        };
        let err = ensure_mode_allowed(&handler, BashMode::Network, "curl https://example.com")
            .await
            .expect_err("network mode should be denied");
        assert!(matches!(
            err,
            BashError::PermissionDenied(BashMode::Network)
        ));
        assert_eq!(handler.mode_checks.load(AtomicOrdering::Relaxed), 1);
    }

    #[tokio::test]
    async fn permission_network_policy_delegates_domain_checks() {
        let handler = Arc::new(TestPermissionHandler {
            allow_network: true,
            allow_domain: true,
            ..Default::default()
        });
        let policy = PermissionNetworkPolicy {
            permission_handler: handler.clone(),
        };
        let request = DomainRequest::new(
            "example.com".to_string(),
            443,
            ConnectionDirection::Outbound,
            1234,
        );
        assert!(policy.check(&request).await);
        assert_eq!(handler.domain_checks.load(AtomicOrdering::Relaxed), 1);
    }
}
