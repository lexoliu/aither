//! Git inside a subagent's sandbox: the same filesystem grants and network
//! denial as the subagent's commands, as the subagent, with stdout streamed.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use async_channel::Sender;
use blocking::Unblock;
use executor_core::{Executor, Task};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::{
    Access, DenyAll, Grant, ProcessPriority, Sandbox, SandboxConfig, SecurityConfig, StdioConfig,
};

use super::{ContainerConfinement, prioritized_container_script};
use crate::MountRoot;
use crate::shell_session::{ContainerExecOutcome, ShellSessionRegistry};
use crate::subagent::{ExportError, SandboxedExit, SandboxedGitImpl, SandboxedRun};

/// How much of what a command printed on stderr is kept for a message.
const STDERR_KEPT: usize = 8 * 1024;

/// How much of a command's stdout travels in one chunk.
const CHUNK: usize = 64 * 1024;

fn stderr_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(STDERR_KEPT)])
        .trim()
        .to_string()
}

/// Kills the process, with everything it started, when dropped before it was
/// waited for.
struct KillOnDrop {
    child: heel::Child,
    finished: bool,
}

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
        }
    }
}

/// Git in the local sandbox of a subagent.
pub(super) struct LocalGit<E> {
    pub executor: E,
    /// The subagent's space.
    pub working_dir: PathBuf,
    pub writable: Vec<PathBuf>,
    pub readable: Vec<PathBuf>,
    pub priority: ProcessPriority,
}

impl<E: Executor + Clone + 'static> LocalGit<E> {
    async fn run_in_sandbox(&self, run: SandboxedRun<'_>) -> Result<SandboxedExit, ExportError> {
        let config = SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(&self.working_dir)
            .grants(
                self.writable
                    .iter()
                    .map(|path| Grant::new(path, Access::WRITE))
                    .chain(
                        self.readable
                            .iter()
                            .map(|path| Grant::new(path, Access::READ)),
                    ),
            )
            .security(SecurityConfig::interactive())
            .priority(self.priority)
            .build();
        let sandbox = Sandbox::with_config_and_executor(config, self.executor.clone())
            .await
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        let child = sandbox
            .command("git")
            .arg("--git-dir")
            .arg(run.repository.to_string_lossy())
            .args(run.args)
            .env("HOME", self.working_dir.to_string_lossy())
            .stdin(StdioConfig::Piped)
            .stdout(StdioConfig::Piped)
            .stderr(StdioConfig::Piped)
            .spawn()
            .await
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        let mut child = KillOnDrop {
            child,
            finished: false,
        };

        let stdin = child.child.take_stdin();
        let stdout = child.child.take_stdout();
        let stderr = child.child.take_stderr();
        let feed = async {
            let Some(stdin) = stdin else {
                return Ok(());
            };
            let mut stdin = Unblock::new(stdin);
            // Git may stop reading early; its exit status says why.
            match stdin.write_all(&run.stdin).await {
                Ok(()) => stdin.flush().await.or_else(ignore_broken_pipe)?,
                Err(error) => ignore_broken_pipe(error)?,
            }
            // Closing the pipe is what tells git its input is complete.
            drop(stdin.into_inner().await);
            Ok::<_, ExportError>(())
        };
        let forward = async {
            let Some(stdout) = stdout else {
                return Ok(());
            };
            let mut stdout = Unblock::new(stdout);
            let mut buffer = vec![0_u8; CHUNK];
            loop {
                let read = stdout.read(&mut buffer).await?;
                if read == 0 {
                    return Ok::<_, ExportError>(());
                }
                run.stdout
                    .send(buffer[..read].to_vec())
                    .await
                    .map_err(|_| ExportError::OutputClosed)?;
            }
        };
        let errors = async {
            let Some(stderr) = stderr else {
                return Ok(Vec::new());
            };
            let mut stderr = Unblock::new(stderr);
            let mut kept = Vec::new();
            let mut buffer = vec![0_u8; 4096];
            loop {
                let read = stderr.read(&mut buffer).await?;
                if read == 0 {
                    return Ok::<_, ExportError>(kept);
                }
                // The rest is read and dropped so git is never held back.
                let room = STDERR_KEPT.saturating_sub(kept.len());
                kept.extend_from_slice(&buffer[..read.min(room)]);
            }
        };
        let ((), (), stderr) = futures::future::try_join3(feed, forward, errors).await?;
        let status = child
            .child
            .wait()
            .await
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        child.finished = true;
        Ok(SandboxedExit {
            code: status.code(),
            stderr: stderr_text(&stderr),
        })
    }
}

fn ignore_broken_pipe(error: std::io::Error) -> Result<(), ExportError> {
    if error.kind() == std::io::ErrorKind::BrokenPipe {
        Ok(())
    } else {
        Err(error.into())
    }
}

impl<E: Executor + Clone + 'static> SandboxedGitImpl for LocalGit<E> {
    fn run<'a>(
        &'a self,
        run: SandboxedRun<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<SandboxedExit, ExportError>> + Send + 'a>> {
        Box::pin(self.run_in_sandbox(run))
    }
}

/// Git in the per-command sandbox of a subagent inside a container.
pub(super) struct ContainerGit<E> {
    pub executor: E,
    pub shell_sessions: ShellSessionRegistry,
    pub confinement: ContainerConfinement,
    pub mount: MountRoot,
    pub priority: ProcessPriority,
}

impl<E: Executor + Clone + 'static> ContainerGit<E> {
    async fn run_in_container(&self, run: SandboxedRun<'_>) -> Result<SandboxedExit, ExportError> {
        let runtime = self
            .shell_sessions
            .container_runtime()
            .cloned()
            .ok_or_else(|| ExportError::Launch("no container runtime".to_string()))?;
        let tools = self
            .shell_sessions
            .container_tools()
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        let repository = self
            .mount
            .container_path(run.repository)
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        let script = shell_words::join(
            ["git", "--git-dir", repository.as_str()]
                .into_iter()
                .chain(run.args.iter().copied()),
        );
        let wrapped = prioritized_container_script(
            &script,
            self.priority,
            tools.bin_dir(),
            self.confinement.working_dir.as_str(),
            Some(&self.confinement),
        )
        .map_err(|error| ExportError::Launch(error.to_string()))?;

        // Dropping the kill switch, as this future's being dropped does, is
        // what stops the command inside the container.
        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        let (stdin_tx, stdin_rx) = async_channel::bounded::<Vec<u8>>(1);
        if !run.stdin.is_empty() {
            stdin_tx
                .try_send(run.stdin)
                .expect("an empty channel with room for one chunk takes it");
        }
        drop(stdin_tx);
        let (done_tx, done_rx) = async_channel::bounded(1);
        let exec = runtime.exec();
        let container_id = runtime.container_id().to_string();
        let working_dir = self.confinement.working_dir.as_str().to_string();
        let stdout: Sender<Vec<u8>> = run.stdout;
        self.executor
            .spawn(async move {
                let outcome = exec
                    .exec_streamed_boxed(
                        &container_id,
                        &wrapped,
                        &working_dir,
                        kill_rx,
                        stdin_rx,
                        stdout,
                    )
                    .await;
                let _ = done_tx.send(outcome).await;
            })
            .detach();
        let outcome = done_rx
            .recv()
            .await
            .map_err(|_| ExportError::Launch("the container exec ended without a result".into()))?;
        drop(kill_tx);
        match outcome {
            Ok(ContainerExecOutcome::Completed(output)) => Ok(SandboxedExit {
                code: output.status.code(),
                stderr: stderr_text(&output.stderr),
            }),
            // Only this run's own kill switch, or a gone reader, kills it.
            Ok(ContainerExecOutcome::Killed) => Err(ExportError::OutputClosed),
            Err(error) => Err(ExportError::Launch(error)),
        }
    }
}

impl<E: Executor + Clone + 'static> SandboxedGitImpl for ContainerGit<E> {
    fn run<'a>(
        &'a self,
        run: SandboxedRun<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<SandboxedExit, ExportError>> + Send + 'a>> {
        Box::pin(self.run_in_container(run))
    }
}
