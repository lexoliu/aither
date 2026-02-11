//! Bash tool implementation with leash sandboxing.
//!
//! Provides the main `bash` tool that LLMs use to execute scripts.
//! Scripts run in a sandbox with configurable permission modes.
//!
//! Each `BashTool` creates a shared working directory with four random words
//! (e.g., `amber-forest-thunder-pearl/`). All bash executions share this
//! directory, but each execution gets a fresh sandbox (new TTY/process).

use std::{borrow::Cow, path::PathBuf, process::Stdio, sync::Arc};

use crate::output::INLINE_OUTPUT_LIMIT;
use aither_core::llm::{Tool, ToolOutput};
use async_channel::{Receiver, Sender};
use executor_core::{Executor, Task};
use leash::{
    AllowAll, DenyAll, IpcRouter, NetworkPolicy, Sandbox, SandboxConfig, SecurityConfig,
    StdioConfig, WorkingDir,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::{
    builtin::builtin_router,
    command::ToolRegistry,
    job_registry::{JobRegistry, job_registry_channel},
    output::{OutputEntry, OutputFormat, OutputStore},
    permission::{BashMode, PermissionError, PermissionHandler},
    shell_session::{ShellSessionRegistry, SshRuntimeProfile},
};

/// Generate a random four-word ID (e.g., "amber-forest-thunder-pearl").
fn random_task_id() -> String {
    crate::naming::random_word_slug(4)
}

/// Execute bash scripts in a sandboxed environment.
///
/// The primary interface to all system capabilities. Scripts run in a sandbox
/// with full read access to the host filesystem but writes contained to a
/// dedicated working directory.
///
/// ## Output Handling
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
/// - `task <subagent> "prompt"` - launch specialized subagents
/// - `todo` - manage task list
///
/// ## Permission Modes
///
/// Execution mode is selected when opening a shell session via `open_shell`.
/// `bash` inherits the mode from that session.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BashArgs {
    /// Active shell session id returned by `open_shell`.
    pub shell_id: String,

    /// The bash script to execute.
    pub script: String,

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
}

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
    pub fn take_completed(&self) -> Vec<CompletedTask> {
        let mut completed = Vec::new();
        while let Ok(task) = self.rx.try_recv() {
            completed.push(task);
        }
        completed
    }

    /// Checks if there are any completed tasks waiting.
    pub fn has_completed(&self) -> bool {
        !self.rx.is_empty()
    }

    /// Checks if there are any background tasks still running.
    ///
    /// This works by checking the sender count - BashTool holds one sender,
    /// and each running background task holds a cloned sender.
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
    /// Shell session registry for open_shell/bash/close_shell lifecycle.
    shell_sessions: ShellSessionRegistry,
    /// Auto-open a default session when shell_id is missing or stale.
    auto_open_default_session: bool,
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
            auto_open_default_session: self.auto_open_default_session,
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
    /// Injects the shared shell-session registry used by open_shell/bash/close_shell.
    #[must_use]
    pub fn with_shell_sessions(mut self, sessions: ShellSessionRegistry) -> Self {
        self.shell_sessions = sessions;
        self
    }

    /// Sets dynamic shell runtime availability for open_shell.
    #[must_use]
    pub fn with_shell_runtime_availability(
        self,
        availability: crate::shell_session::ShellRuntimeAvailability,
    ) -> Self {
        let _ = self.shell_sessions.set_availability(availability);
        self
    }

    /// Controls whether `bash` auto-opens a default session on missing/stale shell_id.
    #[must_use]
    pub fn with_auto_open_default_session(mut self, enabled: bool) -> Self {
        self.auto_open_default_session = enabled;
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
            auto_open_default_session: false,
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
            auto_open_default_session: false,
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
            auto_open_default_session: self.auto_open_default_session,
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

    /// Creates a child BashTool that shares the same sandbox and permission handler
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
            auto_open_default_session: self.auto_open_default_session,
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
    pub fn working_dir(&self) -> &PathBuf {
        &self.working_dir
    }

    /// Returns the outputs directory path.
    pub fn outputs_dir(&self) -> PathBuf {
        self.working_dir.join("outputs")
    }

    /// Returns the output store.
    pub fn output_store(&self) -> &Arc<OutputStore> {
        &self.output_store
    }

    /// Returns the job registry handle.
    pub fn job_registry(&self) -> JobRegistry {
        self.job_registry.clone()
    }

    /// Returns a receiver for completed background tasks.
    ///
    /// The returned `BackgroundTaskReceiver` can be used independently of the BashTool
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
    fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry.registry
    }

    /// Starts a background service that produces child bash tools on demand.
    pub fn start_factory_service(&self, receiver: BashToolFactoryReceiver) {
        let tool = self.clone();
        self.executor
            .spawn(async move { receiver.serve(tool).await })
            .detach();
    }

    /// Converts this BashTool into a type-erased DynBashTool.
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
                            let text: &str = output.as_str().unwrap_or("");
                            text.to_string()
                        }
                        Err(e) => format!("{{\"error\": \"{}\"}}", e),
                    },
                    Err(e) => format!("{{\"error\": \"Parse error: {}\"}}", e),
                }
            })
        });

        DynBashTool {
            definition,
            handler,
        }
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> BashTool<P, E, Configured> {
    /// Executes in sandboxed mode (read-only, no network).
    async fn execute_sandboxed(&self, script: &str) -> Result<std::process::Output, BashError> {
        let router = create_ipc_router(self.registry().clone());
        let config = SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(&self.working_dir)
            .writable_paths(&self.writable_paths)
            .readable_paths(&self.readable_paths)
            .security(SecurityConfig::interactive())
            .ipc(router)
            .build()
            .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

        let sandbox = Sandbox::with_config_and_executor(config, self.executor.clone())
            .await
            .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

        sandbox
            .command("bash")
            .arg("-c")
            .arg(script)
            .output()
            .await
            .map_err(|e| BashError::Execution(e.to_string()))
    }

    /// Executes with network access enabled.
    async fn execute_network(&self, script: &str) -> Result<std::process::Output, BashError> {
        let router = create_ipc_router(self.registry().clone());
        let config = SandboxConfig::builder()
            .network(AllowAll)
            .working_dir(&self.working_dir)
            .writable_paths(&self.writable_paths)
            .readable_paths(&self.readable_paths)
            .security(SecurityConfig::interactive())
            .ipc(router)
            .build()
            .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

        let sandbox = Sandbox::with_config_and_executor(config, self.executor.clone())
            .await
            .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

        sandbox
            .command("bash")
            .arg("-c")
            .arg(script)
            .output()
            .await
            .map_err(|e| BashError::Execution(e.to_string()))
    }

    /// Executes without sandbox (dangerous).
    async fn execute_unsafe(&self, script: &str) -> Result<std::process::Output, BashError> {
        async_process::Command::new("bash")
            .arg("-c")
            .arg(script)
            .current_dir(&self.working_dir)
            .output()
            .await
            .map_err(|e| BashError::Execution(e.to_string()))
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> Tool
    for BashTool<P, E, Configured>
{
    fn name(&self) -> Cow<'static, str> {
        "bash".into()
    }

    type Arguments = BashArgs;

    async fn call(&self, mut arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let session = if arguments.shell_id.trim().is_empty() {
            if !self.auto_open_default_session {
                return Err(anyhow::anyhow!(
                    "shell_id is required; open a shell first with open_shell"
                ));
            }
            let opened = self
                .shell_sessions
                .open(
                    crate::shell_session::ShellBackend::Local,
                    BashMode::Sandboxed,
                    self.working_dir.clone(),
                    None,
                    None,
                )
                .map_err(anyhow::Error::msg)?;
            arguments.shell_id = opened.id.clone();
            opened
        } else if let Some(existing) = self.shell_sessions.get(&arguments.shell_id) {
            existing
        } else {
            if !self.auto_open_default_session {
                return Err(anyhow::anyhow!(
                    "unknown shell_id; session may be closed or disconnected"
                ));
            }
            let opened = self
                .shell_sessions
                .open(
                    crate::shell_session::ShellBackend::Local,
                    BashMode::Sandboxed,
                    self.working_dir.clone(),
                    None,
                    None,
                )
                .map_err(anyhow::Error::msg)?;
            arguments.shell_id = opened.id.clone();
            opened
        };

        if !self
            .permission_handler
            .check(session.mode, &arguments.script)
            .await
            .map_err(|e| anyhow::anyhow!(e))?
        {
            return Err(anyhow::anyhow!(BashError::PermissionDenied(session.mode)));
        }

        let task_id = random_task_id();
        let script = arguments.script.clone();
        let shell_id = session.id.clone();
        let mode = session.mode;
        let backend = session.backend;
        let ssh_target = session.ssh_target.clone();
        let ssh_runtime = session.ssh_runtime.clone();
        let expect = arguments.expect;
        let working_dir = self.working_dir.clone();
        let writable_paths = self.writable_paths.clone();
        let readable_paths = self.readable_paths.clone();
        let executor = self.executor.clone();
        let registry = self.registry().clone();
        let store_dir = self.output_store.dir().to_path_buf();
        let completed_tx = self.completed_tx.clone();
        let job_registry = self.job_registry.clone();
        let (result_tx, result_rx) = async_channel::bounded(1);

        let task_id_for_spawn = task_id.clone();
        self.executor
            .spawn(async move {
                let result = execute_script_standalone(
                    &working_dir,
                    &writable_paths,
                    &readable_paths,
                    executor,
                    registry,
                    job_registry,
                    &shell_id,
                    &script,
                    mode,
                    backend,
                    ssh_target.as_deref(),
                    ssh_runtime.clone(),
                    expect,
                    &store_dir,
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

        if arguments.timeout == 0 {
            let running = BashResult {
                stdout: OutputEntry::Empty,
                stderr: None,
                exit_code: 0,
                task_id: Some(task_id),
                status: Some("running".to_string()),
            };
            let json = serde_json::to_string(&running).map_err(|e| anyhow::anyhow!(e))?;
            return Ok(ToolOutput::text(json));
        }

        let timeout = std::time::Duration::from_secs(arguments.timeout);
        let immediate = futures_lite::future::or(async { result_rx.recv().await.ok() }, async {
            async_io::Timer::after(timeout).await;
            None
        })
        .await;

        match immediate {
            Some(Ok(mut result)) => {
                result.task_id = None;
                result.status = None;
                let json = serde_json::to_string(&result).map_err(|e| anyhow::anyhow!(e))?;
                Ok(ToolOutput::text(json))
            }
            Some(Err(err)) => Err(anyhow::anyhow!(err)),
            None => {
                let running = BashResult {
                    stdout: OutputEntry::Empty,
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

/// Standalone script execution that can be spawned in a background task.
async fn execute_script_standalone<E: Executor + Clone + 'static>(
    working_dir: &PathBuf,
    writable_paths: &[PathBuf],
    readable_paths: &[PathBuf],
    executor: E,
    registry: Arc<ToolRegistry>,
    job_registry: JobRegistry,
    shell_id: &str,
    script: &str,
    mode: BashMode,
    backend: crate::shell_session::ShellBackend,
    ssh_target: Option<&str>,
    ssh_runtime: Option<SshRuntimeProfile>,
    expect: OutputFormat,
    store_dir: &PathBuf,
) -> Result<BashResult, BashError> {
    info!(
        script_len = script.len(),
        ?mode,
        "executing background bash script"
    );
    debug!(script = %script, "script content");

    let (pid, output) = if matches!(backend, crate::shell_session::ShellBackend::Ssh) {
        execute_ssh_background(
            shell_id,
            script,
            mode,
            ssh_target,
            ssh_runtime,
            &job_registry,
        )
        .await?
    } else {
        match mode {
            BashMode::Sandboxed => {
                execute_sandboxed_background(
                    working_dir,
                    writable_paths,
                    readable_paths,
                    executor.clone(),
                    registry.clone(),
                    shell_id,
                    script,
                    mode,
                    DenyAll,
                    &job_registry,
                )
                .await?
            }
            BashMode::Network => {
                execute_sandboxed_background(
                    working_dir,
                    writable_paths,
                    readable_paths,
                    executor,
                    registry,
                    shell_id,
                    script,
                    mode,
                    AllowAll,
                    &job_registry,
                )
                .await?
            }
            BashMode::Unsafe => {
                execute_unsafe_background(working_dir, shell_id, script, mode, &job_registry)
                    .await?
            }
        }
    };

    // Save stdout with system-level size limit
    let limit = Some(INLINE_OUTPUT_LIMIT);
    let stdout =
        match OutputStore::save_to_dir_with_limit(store_dir, &output.stdout, expect, limit).await {
            Ok(entry) => entry,
            Err(err) => {
                job_registry.fail(pid, &err.to_string(), None).await;
                return Err(BashError::Io(err));
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
            limit,
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
    shell_id: &str,
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

    let sandbox = Sandbox::with_config_and_executor(config, executor)
        .await
        .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

    let mut child = sandbox
        .command("bash")
        .arg("-c")
        .arg(script)
        .stdin(StdioConfig::Null)
        .stdout(StdioConfig::Piped)
        .stderr(StdioConfig::Piped)
        .spawn()
        .await
        .map_err(|e| BashError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, shell_id, script, mode, None)
        .await;

    let output_result =
        run_blocking(move || futures_lite::future::block_on(child.wait_with_output())).await;
    let output = match output_result {
        Ok(output) => output,
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            return Err(BashError::Execution(err.to_string()));
        }
    };

    Ok((pid, output))
}

async fn execute_unsafe_background(
    working_dir: &PathBuf,
    shell_id: &str,
    script: &str,
    mode: BashMode,
    job_registry: &JobRegistry,
) -> Result<(u32, std::process::Output), BashError> {
    let mut child = async_process::Command::new("bash")
        .arg("-c")
        .arg(script)
        .current_dir(working_dir)
        .stdin(async_process::Stdio::null())
        .stdout(async_process::Stdio::piped())
        .stderr(async_process::Stdio::piped())
        .spawn()
        .map_err(|e| BashError::Execution(e.to_string()))?;

    let pid = child.id();
    job_registry
        .register(pid, shell_id, script, mode, None)
        .await;

    match child.output().await {
        Ok(output) => Ok((pid, output)),
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            Err(BashError::Execution(err.to_string()))
        }
    }
}

async fn execute_ssh_background(
    shell_id: &str,
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

    let remote_cmd = match runtime {
        SshRuntimeProfile::Leash { binary } => match mode {
            BashMode::Sandboxed => format!(
                "{} run --network deny -- /bin/bash -lc {}",
                shell_escape(&binary),
                shell_escape(script)
            ),
            BashMode::Network => format!(
                "{} run --network allow -- /bin/bash -lc {}",
                shell_escape(&binary),
                shell_escape(script)
            ),
            BashMode::Unsafe => format!(
                "{} run --network allow -- /bin/bash -lc {}",
                shell_escape(&binary),
                shell_escape(script)
            ),
        },
        SshRuntimeProfile::Unsafe => {
            if mode != BashMode::Unsafe {
                return Err(BashError::Execution(
                    "remote leash unavailable: only unsafe mode is allowed".to_string(),
                ));
            }
            format!("/bin/bash -lc {}", shell_escape(script))
        }
    };

    let mut child = async_process::Command::new("ssh")
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
        .register(pid, shell_id, script, mode, None)
        .await;

    match child.output().await {
        Ok(output) => Ok((pid, output)),
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            Err(BashError::Execution(err.to_string()))
        }
    }
}

fn shell_escape(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

async fn run_blocking<T, F>(task: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = async_channel::bounded(1);
    std::thread::spawn(move || {
        let _ = tx.send_blocking(task());
    });
    rx.recv().await.expect("blocking task channel dropped")
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

/// Returns (os_name, os_version) for the current system.
pub fn get_os_info() -> (String, String) {
    #[cfg(target_os = "macos")]
    {
        let version = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        ("macOS".to_string(), version)
    }

    #[cfg(target_os = "linux")]
    {
        // Try to get pretty name from os-release
        let version = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("PRETTY_NAME="))
                    .map(|line| {
                        line.trim_start_matches("PRETTY_NAME=")
                            .trim_matches('"')
                            .to_string()
                    })
            })
            .unwrap_or_else(|| {
                // Fallback to uname -r
                std::process::Command::new("uname")
                    .arg("-r")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });
        ("Linux".to_string(), version)
    }

    #[cfg(target_os = "windows")]
    {
        let version = std::process::Command::new("cmd")
            .args(["/C", "ver"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        ("Windows".to_string(), version)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        (std::env::consts::OS.to_string(), "unknown".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bash_args_defaults() {
        let args: BashArgs =
            serde_json::from_str(r#"{"shell_id":"s1","script":"echo hello","timeout":1}"#).unwrap();
        assert_eq!(args.expect, OutputFormat::Text);
        assert_eq!(args.timeout, 1);
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
            serde_json::from_str(r#"{"shell_id":"s1","script":"echo hello","timeout":0}"#).unwrap();
        assert_eq!(args.timeout, 0);

        let args_default =
            serde_json::from_str::<BashArgs>(r#"{"shell_id":"s1","script":"echo hello"}"#)
                .unwrap_err();
        assert!(args_default.to_string().contains("timeout"));
    }
}
