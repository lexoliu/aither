//! Bash tool implementation with leash sandboxing.
//!
//! Provides the main `bash` tool that LLMs use to execute scripts.
//! Scripts run in a sandbox with configurable permission modes.
//!
//! Each `BashTool` creates a shared working directory with four random words
//! (e.g., `amber-forest-thunder-pearl/`). All bash executions share this
//! directory, but each execution gets a fresh sandbox (new TTY/process).

use std::{
    borrow::Cow,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use rand::seq::IndexedRandom;

use aither_core::llm::{Tool, ToolOutput};
use crate::output::INLINE_OUTPUT_LIMIT;
use async_channel::{Receiver, Sender};
use executor_core::{Executor, Task};
use leash::{AllowAll, DenyAll, IpcRouter, Sandbox, SandboxConfig, SecurityConfig, WorkingDir};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use crate::{
    builtin::builtin_router,
    output::{OutputEntry, OutputFormat, OutputStore},
    permission::{BashMode, PermissionError, PermissionHandler},
};

/// Word list for generating random task IDs (four words like "amber-forest-thunder-pearl").
const WORDS: &[&str] = &[
    "apple", "banana", "cherry", "dragon", "eagle", "falcon", "garden", "harbor", "island",
    "jungle", "kitten", "lemon", "mango", "night", "ocean", "planet", "queen", "river", "silver",
    "tiger", "umbrella", "violet", "winter", "yellow", "zebra", "anchor", "bridge", "castle",
    "desert", "ember", "forest", "glacier", "horizon", "ivory", "jasmine", "kingdom", "lantern",
    "meadow", "nebula", "orchid", "phoenix", "quartz", "rainbow", "shadow", "thunder", "urban",
    "velvet", "whisper", "crystal", "dolphin", "eclipse", "firefly", "granite", "hollow",
    "indigo", "journey", "karma", "lotus", "marble", "nomad", "oasis", "prism", "quest", "ripple",
    "sphinx", "temple", "unity", "vortex", "willow", "xenon", "yonder", "zenith", "amber",
    "blazer", "copper", "dusk", "ether", "flame", "golden", "haze", "iron", "jade", "kindle",
    "lunar", "mystic", "nova", "onyx", "pearl", "radiant", "storm", "tidal", "ultra", "vivid",
    "wave", "azure", "breeze",
];

/// Generate a random four-word ID (e.g., "amber-forest-thunder-pearl").
fn random_task_id() -> String {
    let mut rng = rand::rng();
    let words: Vec<&str> = WORDS.choose_multiple(&mut rng, 4).copied().collect();
    words.join("-")
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
/// - `sandboxed` (default): Read-only host access, writes to sandbox only
/// - `network`: Sandbox + network access (for curl, wget, ssh, git)
/// - `unsafe`: No sandbox, full system access (requires approval with reason)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BashArgs {
    /// The bash script to execute.
    pub script: String,

    /// Permission mode for execution.
    /// - "sandboxed" (default): read-only filesystem, no network, IPC commands work
    /// - "network": sandbox with network access enabled
    /// - "unsafe": no sandbox, full system access (requires approval)
    #[serde(default)]
    pub mode: BashMode,

    /// REQUIRED when mode is "unsafe". Explain why unsafe access is needed.
    /// Examples: "Writing config to ~/.zshrc", "Running osascript to automate Mail app"
    #[serde(default)]
    pub reason: Option<String>,

    /// Expected output format for proper handling.
    /// - "text" (default): plain text
    /// - "image": image data (auto-loaded to context)
    /// - "video": video data
    /// - "binary": binary data
    /// - "auto": auto-detect
    #[serde(default)]
    pub expect: OutputFormat,

    /// Run command in background. Returns immediately with task ID.
    /// Results are injected into context when complete.
    #[serde(default)]
    pub background: bool,
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

    /// Task ID for background execution (only present when background: true).
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
        futures_lite::future::or(
            async { self.rx.recv().await.ok() },
            async {
                futures_lite::future::yield_now().await;
                async_io::Timer::after(duration).await;
                None
            },
        )
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

/// The bash tool for executing scripts in a sandbox.
///
/// Creates a shared working directory with four random words that persists
/// across all executions. Each execution creates a fresh sandbox but shares
/// the working directory.
///
/// The executor type `E` determines how async tasks are spawned for the IPC server.
/// Use `TokioExecutor` when running in a tokio runtime.
pub struct BashTool<P, E> {
    /// Shared working directory (four random words, e.g., `amber-forest-thunder-pearl/`)
    working_dir: PathBuf,
    /// Permission handler wrapped in Arc for sharing between parent and child tools.
    permission_handler: Arc<P>,
    executor: E,
    output_store: Arc<RwLock<OutputStore>>,
    /// Channel for receiving completed background tasks.
    completed_rx: Receiver<CompletedTask>,
    /// Channel for sending completed background tasks (cloned for each background task).
    completed_tx: Sender<CompletedTask>,
}

// Manual Clone impl because P doesn't need to be Clone (we use Arc<P>)
impl<P, E: Clone> Clone for BashTool<P, E> {
    fn clone(&self) -> Self {
        Self {
            working_dir: self.working_dir.clone(),
            permission_handler: self.permission_handler.clone(),
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            completed_rx: self.completed_rx.clone(),
            completed_tx: self.completed_tx.clone(),
        }
    }
}

impl<P, E: Executor + Clone + 'static> BashTool<P, E> {
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
        let output_store = Arc::new(RwLock::new(OutputStore::new(&outputs_dir).await?));

        // Create channel for background task completion (unbounded to not block spawned tasks)
        let (completed_tx, completed_rx) = async_channel::unbounded();

        Ok(Self {
            working_dir: working_dir_path,
            permission_handler: Arc::new(permission_handler),
            executor,
            output_store,
            completed_rx,
            completed_tx,
        })
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
            permission_handler: self.permission_handler.clone(), // Arc clone - shares handler
            executor: self.executor.clone(),
            output_store: self.output_store.clone(),
            completed_rx,
            completed_tx,
        }
    }

    /// Returns the working directory path.
    pub fn working_dir(&self) -> &PathBuf {
        &self.working_dir
    }

    /// Returns the output store.
    pub fn output_store(&self) -> &Arc<RwLock<OutputStore>> {
        &self.output_store
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

    /// Converts this BashTool into a type-erased DynBashTool.
    ///
    /// This is useful for creating child bash tools for subagents where
    /// the concrete type cannot be known at compile time.
    pub fn to_dyn(self) -> crate::command::DynBashTool
    where
        P: PermissionHandler + 'static,
    {
        use std::sync::Arc;
        use crate::command::{DynBashTool, DynToolHandler};
        use aither_core::llm::tool::ToolDefinition;
        use serde_json::Value;

        // Create the definition - description comes from BashArgs rustdoc
        let schema = schemars::schema_for!(BashArgs);
        let schema_value: Value = schema.to_value();
        let description = schema_value
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let definition = ToolDefinition::from_parts(
            "bash".into(),
            description.into(),
            schema_value,
        );

        // Create the handler
        let tool = Arc::new(self);
        let handler: DynToolHandler = Arc::new(move |args: &str| {
            let tool = tool.clone();
            let args_str = args.to_string();
            Box::pin(async move {
                match serde_json::from_str::<BashArgs>(&args_str) {
                    Ok(parsed) => match tool.call(parsed).await {
                        Ok(output) => output.as_str().unwrap_or("").to_string(),
                        Err(e) => format!("{{\"error\": \"{}\"}}", e),
                    },
                    Err(e) => format!("{{\"error\": \"Parse error: {}\"}}", e),
                }
            })
        });

        DynBashTool { definition, handler }
    }
}

impl<P: PermissionHandler, E: Executor + Clone + 'static> BashTool<P, E> {
    /// Executes a bash script with the given arguments.
    #[instrument(skip(self), fields(mode = ?args.mode, expect = ?args.expect))]
    async fn execute(&self, args: BashArgs) -> Result<BashResult, BashError> {
        // Check permission
        if !self.permission_handler.check(args.mode, &args.script).await? {
            return Err(BashError::PermissionDenied(args.mode));
        }

        info!(script_len = args.script.len(), "executing bash script");
        debug!(script = %args.script, "script content");

        // Execute based on mode
        let output = match args.mode {
            BashMode::Sandboxed => self.execute_sandboxed(&args.script).await?,
            BashMode::Network => self.execute_network(&args.script).await?,
            BashMode::Unsafe => self.execute_unsafe(&args.script).await?,
        };

        // Save stdout - get store dir first, then save async
        let store_dir = {
            let store = self.output_store.read().map_err(|_| BashError::StoreLock)?;
            store.dir().to_path_buf()
        };
        let limit = Some(INLINE_OUTPUT_LIMIT);
        let stdout = OutputStore::save_to_dir_with_limit(&store_dir, &output.stdout, args.expect, limit).await?;

        // Save stderr if non-empty
        let stderr = if output.stderr.is_empty() {
            None
        } else {
            Some(OutputStore::save_to_dir_with_limit(&store_dir, &output.stderr, OutputFormat::Text, limit).await?)
        };

        let exit_code = output.status.code().unwrap_or(-1);
        debug!(exit_code, "script completed");

        Ok(BashResult {
            stdout,
            stderr,
            exit_code,
            task_id: None,
            status: None,
        })
    }

    /// Executes in sandboxed mode (read-only, no network).
    async fn execute_sandboxed(&self, script: &str) -> Result<std::process::Output, BashError> {
        let router = create_ipc_router();
        let config = SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(&self.working_dir)
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
        let router = create_ipc_router();
        let config = SandboxConfig::builder()
            .network(AllowAll)
            .working_dir(&self.working_dir)
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

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> Tool for BashTool<P, E> {
    fn name(&self) -> Cow<'static, str> {
        "bash".into()
    }

    type Arguments = BashArgs;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        // Validate: unsafe mode requires a reason
        if arguments.mode == BashMode::Unsafe && arguments.reason.is_none() {
            return Err(anyhow::anyhow!(
                "unsafe mode requires a 'reason' field explaining why unsafe access is needed"
            ));
        }

        // Check permission first (before potentially spawning background task)
        if !self.permission_handler.check(arguments.mode, &arguments.script).await
            .map_err(|e| anyhow::anyhow!(e))?
        {
            return Err(anyhow::anyhow!(BashError::PermissionDenied(arguments.mode)));
        }

        if arguments.background {
            // Background execution: spawn task and return immediately
            let task_id = random_task_id();
            let script = arguments.script.clone();
            let mode = arguments.mode;
            let expect = arguments.expect;
            let working_dir = self.working_dir.clone();
            let executor = self.executor.clone();
            let store_dir = {
                let store = self.output_store.read().map_err(|_| anyhow::anyhow!(BashError::StoreLock))?;
                store.dir().to_path_buf()
            };
            let completed_tx = self.completed_tx.clone();

            info!(task_id = %task_id, script_len = script.len(), "spawning background bash task");

            // Clone task_id for the spawned task
            let task_id_for_spawn = task_id.clone();

            // Spawn background task
            self.executor.spawn(async move {
                let result = execute_script_standalone(
                    &working_dir,
                    executor,
                    &script,
                    mode,
                    expect,
                    &store_dir,
                ).await;

                // Send completion (ignore send error if receiver dropped)
                let _ = completed_tx.send(CompletedTask {
                    task_id: task_id_for_spawn,
                    script,
                    result,
                }).await;
            }).detach();

            // Return immediately with running status
            let result = BashResult {
                stdout: OutputEntry::Empty,
                stderr: None,
                exit_code: 0,
                task_id: Some(task_id),
                status: Some("running".to_string()),
            };
            let json = serde_json::to_string(&result).map_err(|e| anyhow::anyhow!(e))?;
            Ok(ToolOutput::text(json))
        } else {
            // Normal synchronous execution
            match self.execute(arguments).await {
                Ok(result) => {
                    let json = serde_json::to_string(&result).map_err(|e| anyhow::anyhow!(e))?;
                    Ok(ToolOutput::text(json))
                }
                Err(e) => Err(anyhow::anyhow!(e)),
            }
        }
    }
}

/// Standalone script execution that can be spawned in a background task.
async fn execute_script_standalone<E: Executor + Clone + 'static>(
    working_dir: &PathBuf,
    executor: E,
    script: &str,
    mode: BashMode,
    expect: OutputFormat,
    store_dir: &PathBuf,
) -> Result<BashResult, BashError> {
    info!(script_len = script.len(), ?mode, "executing background bash script");
    debug!(script = %script, "script content");

    // Execute based on mode
    let output = match mode {
        BashMode::Sandboxed => {
            let router = create_ipc_router();
            let config = SandboxConfig::builder()
                .network(DenyAll)
                .working_dir(working_dir)
                .security(SecurityConfig::interactive())
                .ipc(router)
                .build()
                .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

            let sandbox = Sandbox::with_config_and_executor(config, executor)
                .await
                .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

            sandbox
                .command("bash")
                .arg("-c")
                .arg(script)
                .output()
                .await
                .map_err(|e| BashError::Execution(e.to_string()))?
        }
        BashMode::Network => {
            let router = create_ipc_router();
            let config = SandboxConfig::builder()
                .network(AllowAll)
                .working_dir(working_dir)
                .security(SecurityConfig::interactive())
                .ipc(router)
                .build()
                .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

            let sandbox = Sandbox::with_config_and_executor(config, executor)
                .await
                .map_err(|e| BashError::SandboxSetup(e.to_string()))?;

            sandbox
                .command("bash")
                .arg("-c")
                .arg(script)
                .output()
                .await
                .map_err(|e| BashError::Execution(e.to_string()))?
        }
        BashMode::Unsafe => {
            async_process::Command::new("bash")
                .arg("-c")
                .arg(script)
                .current_dir(working_dir)
                .output()
                .await
                .map_err(|e| BashError::Execution(e.to_string()))?
        }
    };

    // Save stdout with system-level size limit
    let limit = Some(INLINE_OUTPUT_LIMIT);
    let stdout = OutputStore::save_to_dir_with_limit(store_dir, &output.stdout, expect, limit).await?;

    // Save stderr if non-empty
    let stderr = if output.stderr.is_empty() {
        None
    } else {
        Some(OutputStore::save_to_dir_with_limit(store_dir, &output.stderr, OutputFormat::Text, limit).await?)
    };

    let exit_code = output.status.code().unwrap_or(-1);
    debug!(exit_code, "background script completed");

    Ok(BashResult {
        stdout,
        stderr,
        exit_code,
        task_id: None,
        status: None,
    })
}

/// Creates the IPC router with built-in and tool commands (standalone version).
fn create_ipc_router() -> IpcRouter {
    let mut router = builtin_router();

    // Register all configured tools as IPC commands
    for name in crate::registered_tool_names() {
        router = crate::register_tool_command(router, &name);
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

    /// Output store lock poisoned.
    #[error("output store lock poisoned")]
    StoreLock,

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl<P, E> std::fmt::Debug for BashTool<P, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashTool")
            .field("working_dir", &self.working_dir)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bash_args_defaults() {
        let args: BashArgs = serde_json::from_str(r#"{"script": "echo hello"}"#).unwrap();
        assert_eq!(args.mode, BashMode::Sandboxed);
        assert_eq!(args.expect, OutputFormat::Text);
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
    async fn test_bash_args_background() {
        let args: BashArgs = serde_json::from_str(r#"{"script": "echo hello", "background": true}"#).unwrap();
        assert!(args.background);

        let args_default: BashArgs = serde_json::from_str(r#"{"script": "echo hello"}"#).unwrap();
        assert!(!args_default.background);
    }
}
