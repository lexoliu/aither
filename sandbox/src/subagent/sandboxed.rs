//! Running git inside a subagent's own sandbox, as the subagent.
//!
//! The host never reads what a subagent wrote. It asks git, running with the
//! subagent's authority and no more, to say what is in the subagent's
//! repository and to pack it, and treats the answer as the untrusted input it
//! is.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_channel::Sender;

/// Why git could not be run in a subagent's sandbox, or did not finish.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The sandboxed git could not be started.
    #[error("cannot start git in the subagent's sandbox: {0}")]
    Launch(String),
    /// Input or output of the sandboxed git failed.
    #[error("cannot talk to git in the subagent's sandbox: {0}")]
    Io(#[from] std::io::Error),
    /// The sandboxed git exited unsuccessfully.
    #[error("git in the subagent's sandbox {}: {stderr}", exit_description(*.code))]
    Failed {
        /// Its exit code, when it exited rather than being signalled.
        code: Option<i32>,
        /// What it printed on stderr.
        stderr: String,
    },
    /// The sandboxed git printed more than the import allows.
    #[error("git in the subagent's sandbox printed more than the {limit} bytes an import may take")]
    OverBudget {
        /// The limit.
        limit: u64,
    },
    /// The sandboxed git did not finish in time and was killed.
    #[error("git in the subagent's sandbox did not finish within {} seconds and was killed", .0.as_secs())]
    TimedOut(Duration),
    /// Nobody was reading the output of the sandboxed git any more.
    #[error("nothing was reading the output of git in the subagent's sandbox")]
    OutputClosed,
}

fn exit_description(code: Option<i32>) -> String {
    code.map_or_else(
        || "was killed by a signal".to_string(),
        |code| format!("exited with {code}"),
    )
}

/// How a sandboxed git run ended, when it ended at all.
#[derive(Debug, Clone)]
pub struct SandboxedExit {
    /// Its exit code, when it exited rather than being signalled.
    pub code: Option<i32>,
    /// What it printed on stderr, cut to a size a message can hold.
    pub stderr: String,
}

impl SandboxedExit {
    pub const fn success(&self) -> bool {
        matches!(self.code, Some(0))
    }

    /// The error for a run that did not succeed.
    pub fn into_failure(self) -> ExportError {
        ExportError::Failed {
            code: self.code,
            stderr: self.stderr,
        }
    }
}

/// One git command to run in a subagent's sandbox.
pub struct SandboxedRun<'a> {
    /// The repository, spelled for the host; the backend maps it to its own
    /// spelling. It becomes git's `--git-dir`.
    pub repository: &'a Path,
    /// The git command and its arguments.
    pub args: &'a [&'a str],
    /// What the command reads on stdin.
    pub stdin: Vec<u8>,
    /// Where its stdout goes, chunk by chunk. The run ends, and kills the
    /// command, once the receiving side is gone.
    pub stdout: Sender<Vec<u8>>,
}

/// What bounds one sandboxed git run: the bytes it may print and how long it
/// may take.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// The most bytes of stdout; `None` is unbounded.
    pub bytes: Option<u64>,
    /// How long the run may take before it is killed with everything it
    /// started.
    pub time: Duration,
}

/// `future`'s output, or `None` once `limit` has passed. The future is
/// dropped then, which kills everything it started.
pub async fn within<T>(limit: Duration, future: impl Future<Output = T>) -> Option<T> {
    futures_lite::future::or(async { Some(future.await) }, async {
        async_io::Timer::after(limit).await;
        None
    })
    .await
}

/// A backend that runs git inside a subagent's sandbox.
///
/// A run is killed with everything it started when its future is dropped.
pub trait SandboxedGitImpl: Send + Sync {
    fn run<'a>(
        &'a self,
        run: SandboxedRun<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<SandboxedExit, ExportError>> + Send + 'a>>;
}

/// Runs git in a subagent's sandbox: the same filesystem grants and network
/// denial as the subagent's commands, as the subagent.
#[derive(Clone)]
pub struct SubagentGit {
    backend: Arc<dyn SandboxedGitImpl>,
}

impl std::fmt::Debug for SubagentGit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentGit").finish_non_exhaustive()
    }
}

/// What a command printed, collected.
pub struct Collected {
    pub exit: SandboxedExit,
    pub stdout: Vec<u8>,
}

/// How many output chunks a run may have in flight before it is held back.
pub const PIPE_DEPTH: usize = 8;

impl SubagentGit {
    pub fn new(backend: Arc<dyn SandboxedGitImpl>) -> Self {
        Self { backend }
    }

    /// Runs the command, streaming its stdout to `stdout`.
    pub async fn run(&self, run: SandboxedRun<'_>) -> Result<SandboxedExit, ExportError> {
        self.backend.run(run).await
    }

    /// Runs the command and collects its stdout, failing when it prints more
    /// than the byte bound or runs past the time bound, in which case it is
    /// killed with everything it started. Everything the host asks a
    /// subagent's repository goes through here, so none of it can be made to
    /// hang.
    pub async fn collect(
        &self,
        repository: &Path,
        args: &[&str],
        bounds: Bounds,
    ) -> Result<Collected, ExportError> {
        let (tx, rx) = async_channel::bounded(PIPE_DEPTH);
        let run = self.run(SandboxedRun {
            repository,
            args,
            stdin: Vec::new(),
            stdout: tx,
        });
        let gather = async {
            let mut stdout = Vec::new();
            while let Ok(chunk) = rx.recv().await {
                if let Some(limit) = bounds.bytes
                    && u64::try_from(stdout.len() + chunk.len()).map_or(true, |len| len > limit)
                {
                    return Err(ExportError::OverBudget { limit });
                }
                stdout.extend_from_slice(&chunk);
            }
            Ok(stdout)
        };
        let (exit, stdout) = within(bounds.time, futures::future::join(run, gather))
            .await
            .ok_or(ExportError::TimedOut(bounds.time))?;
        match (exit, stdout) {
            (_, Err(error)) | (Err(error), _) => Err(error),
            (Ok(exit), Ok(stdout)) => Ok(Collected { exit, stdout }),
        }
    }

    /// Runs the command like [`Self::collect`], failing unless it exits
    /// successfully, and returns its stdout.
    pub async fn succeed(
        &self,
        repository: &Path,
        args: &[&str],
        bounds: Bounds,
    ) -> Result<Vec<u8>, ExportError> {
        let Collected { exit, stdout } = self.collect(repository, args, bounds).await?;
        if exit.success() {
            Ok(stdout)
        } else {
            Err(exit.into_failure())
        }
    }
}
