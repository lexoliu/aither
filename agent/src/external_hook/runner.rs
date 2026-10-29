//! Runs one external hook command: typed JSON on stdin, typed JSON on stdout,
//! under a deadline, in a process group of its own.

use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use aither_sandbox::{PreparedTools, ToolRuntimeError};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The command of an external hook and how long it may run.
///
/// Every kind of external hook is configured with one; the kinds differ in
/// when they run and in the type of the answer they must print.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookCommand {
    /// The program and its arguments; nothing is passed through a shell.
    pub command: Vec<String>,
    /// How long the hook may run, in milliseconds, before it and every
    /// process it started are killed.
    pub deadline_ms: u64,
}

impl HookCommand {
    /// How long the hook may run.
    #[must_use]
    pub const fn deadline(&self) -> Duration {
        Duration::from_millis(self.deadline_ms)
    }
}

/// Which side of the authority boundary a hook's input comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookRole {
    /// The main agent.
    Main,
    /// A subagent, or a subagent of one.
    Subagent,
}

/// Why a hook did not produce an answer.
#[derive(Debug, thiserror::Error)]
pub enum HookRunError {
    /// The configured command has no program.
    #[error("the hook command is empty")]
    EmptyCommand,
    /// The prepared tool directory cannot be put on the hook's `PATH`.
    #[error("the prepared tools cannot be put on the hook's PATH: {0}")]
    ToolPath(#[source] ToolRuntimeError),
    /// The input could not be serialized.
    #[error("the hook input cannot be serialized: {0}")]
    Input(#[source] serde_json::Error),
    /// The hook could not be started.
    #[error("the hook `{program}` cannot be started: {source}")]
    Spawn {
        /// The program that was run.
        program: String,
        /// Why it failed to start.
        #[source]
        source: std::io::Error,
    },
    /// Talking to the hook's pipes failed.
    #[error("talking to the hook failed: {0}")]
    Io(#[source] std::io::Error),
    /// The hook ran past its deadline and was killed with its process group.
    #[error("the hook ran past its {deadline:?} deadline and was killed")]
    Timeout {
        /// The deadline that passed.
        deadline: Duration,
    },
    /// The hook exited unsuccessfully.
    #[error("the hook exited with {status}: {stderr}")]
    Exit {
        /// How it exited.
        status: ExitStatus,
        /// What it wrote to stderr.
        stderr: String,
    },
    /// The hook's stdout is not the answer type.
    #[error("the hook printed output that is not a valid answer: {0}")]
    Output(#[source] serde_json::Error),
}

/// Runs hook commands.
///
/// Holds nothing between runs but the directory of the terminal's prepared
/// baseline tools, so every kind of hook shares one. Every hook runs with that
/// directory on its `PATH` ahead of the host's, so `rg`, `ast-grep`, `jq` and
/// `apply_patch` are the verified ones whatever the host has installed. The
/// process work uses `async-process` and `async-io`, so it runs under any
/// executor.
#[derive(Debug, Clone)]
pub struct HookRunner {
    /// The terminal's prepared tools; absent only in a scope without hooks.
    tools: Option<PreparedTools<PathBuf>>,
}

impl HookRunner {
    /// A runner whose hooks find the baseline tools in `tools`, the same
    /// directory the terminal puts on the `PATH` of its commands.
    #[must_use]
    pub fn new(tools: &PreparedTools<PathBuf>) -> Self {
        Self {
            tools: Some(tools.clone()),
        }
    }

    /// The runner of a scope without hooks, which never runs one.
    pub(super) const fn without_hooks() -> Self {
        Self { tools: None }
    }

    /// Whether hooks run with the prepared tools on their `PATH`.
    pub(super) const fn has_tools(&self) -> bool {
        self.tools.is_some()
    }

    /// Runs `hook` with `input` serialized to JSON on its stdin and returns
    /// the answer it prints on stdout; `None` when it prints nothing.
    ///
    /// The hook runs in a process group of its own. When it passes its
    /// deadline the whole group is killed, and the run fails with
    /// [`HookRunError::Timeout`].
    ///
    /// # Errors
    ///
    /// Returns the reason the hook produced no answer: it could not run, ran
    /// past its deadline, exited unsuccessfully, or printed something that is
    /// not an `Output`.
    pub async fn run<Input, Output>(
        &self,
        hook: &HookCommand,
        input: &Input,
    ) -> Result<Option<Output>, HookRunError>
    where
        Input: Serialize + Sync,
        Output: DeserializeOwned,
    {
        let input = serde_json::to_vec(input).map_err(HookRunError::Input)?;
        let stdout = self.run_raw(hook, &input).await?;
        if stdout.trim_ascii().is_empty() {
            return Ok(None);
        }
        serde_json::from_slice(&stdout)
            .map(Some)
            .map_err(HookRunError::Output)
    }

    async fn run_raw(&self, hook: &HookCommand, input: &[u8]) -> Result<Vec<u8>, HookRunError> {
        let (program, args) = hook
            .command
            .split_first()
            .ok_or(HookRunError::EmptyCommand)?;
        let mut command = std::process::Command::new(program);
        command.args(args);
        if let Some(tools) = &self.tools {
            command.env(
                "PATH",
                tools
                    .prepend_path(std::env::var_os("PATH").as_deref())
                    .map_err(HookRunError::ToolPath)?,
            );
        }
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let mut child = async_process::Command::from(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| HookRunError::Spawn {
                program: program.clone(),
                source,
            })?;
        let pid = child.id();
        let mut stdin = child.stdin.take().expect("the hook's stdin is piped");
        let mut stdout = child.stdout.take().expect("the hook's stdout is piped");
        let mut stderr = child.stderr.take().expect("the hook's stderr is piped");

        let exchange = async {
            let write = async {
                let written = stdin.write_all(input).await;
                // A hook that never reads its input closes the pipe early.
                drop(stdin);
                match written {
                    Err(error) if error.kind() != std::io::ErrorKind::BrokenPipe => Err(error),
                    _ => Ok(()),
                }
            };
            let read = async {
                let mut output = Vec::new();
                stdout.read_to_end(&mut output).await.map(|_| output)
            };
            let diagnostics = async {
                let mut output = Vec::new();
                stderr.read_to_end(&mut output).await.map(|_| output)
            };
            let (written, output, diagnostics, status) =
                futures::join!(write, read, diagnostics, child.status());
            written.map_err(HookRunError::Io)?;
            let output = output.map_err(HookRunError::Io)?;
            let diagnostics = diagnostics.map_err(HookRunError::Io)?;
            let status = status.map_err(HookRunError::Io)?;
            if status.success() {
                Ok(output)
            } else {
                Err(HookRunError::Exit {
                    status,
                    stderr: String::from_utf8_lossy(&diagnostics).trim().to_string(),
                })
            }
        };
        let deadline = hook.deadline();
        let expiry = async {
            async_io::Timer::after(deadline).await;
            Err(HookRunError::Timeout { deadline })
        };
        let result = futures_lite::future::or(exchange, expiry).await;
        if matches!(result, Err(HookRunError::Timeout { .. }))
            && let Err(error) = aither_sandbox::kill_process_tree(pid).await
        {
            tracing::warn!(%error, "the hook's process tree could not be killed");
        }
        result
    }
}
