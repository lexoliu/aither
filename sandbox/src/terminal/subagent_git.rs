//! Git inside a subagent's sandbox: the same filesystem grants and network
//! denial as the subagent's commands, as the subagent, with stdout streamed.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use async_channel::Sender;
use blocking::Unblock;
use executor_core::{Executor, Task};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::{
    Access, DenyAll, Grant, ProcessPriority, ResolvedPath, Sandbox, SandboxConfig, SecurityConfig,
    StdioConfig,
};

use super::{ContainerConfinement, prioritized_container_script};
use crate::MountRoot;
use crate::shell_session::{ContainerExecOutcome, ShellSessionRegistry};
use crate::subagent::{
    CommittingWorktree, ExportError, SandboxedExit, SandboxedGitImpl, SandboxedRun,
};

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
    /// The worktrees the subagent may commit in, to find the worktree a
    /// repository answers to.
    pub committing: Vec<CommittingWorktree>,
    pub priority: ProcessPriority,
}

/// The host's own `git` as an absolute path, so a sandboxed git does not
/// depend on the sandbox's `PATH` — which holds only system directories on
/// Windows, where `git` never lives. The directory that holds it is granted
/// beside it: the binary's sibling libraries live there.
fn host_git_binary() -> Option<(PathBuf, PathBuf)> {
    // `/usr/bin/git` on macOS is the xcode-select stub: it execs the real
    // toolchain git, which a grant on `/usr/bin` does not cover. `xcrun`
    // names the resolved binary.
    #[cfg(target_os = "macos")]
    if let Ok(resolved) = std::process::Command::new("xcrun")
        .args(["-f", "git"])
        .output()
        && resolved.status.success()
    {
        let path = PathBuf::from(String::from_utf8_lossy(&resolved.stdout).trim());
        if let (true, Some(dir)) = (path.is_file(), path.parent().map(Path::to_path_buf)) {
            return Some((path, dir));
        }
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in ["git", "git.exe"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some((candidate, dir));
            }
        }
    }
    None
}

/// A path spelled the way the sandbox resolves grants: canonicalized, minus
/// the `\\?\` prefix canonicalization puts on Windows paths. A grant and a
/// spelled path must name the object the same way — an 8.3 alias and its
/// long name do not match.
fn canonical(path: &std::path::Path) -> PathBuf {
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    resolved
        .to_str()
        .and_then(|spelled| spelled.strip_prefix(r"\\?\"))
        .map_or_else(|| resolved.clone(), PathBuf::from)
}

/// The object a spelled path names, in the spelling a resolved grant
/// checks.
fn resolved(path: &std::path::Path) -> heel::Result<ResolvedPath> {
    ResolvedPath::new(canonical(path))
}

/// The object stores the repository borrows through its
/// `objects/info/alternates`, as absolute paths. Each line there is relative
/// to the repository's object directory.
fn borrowed_object_stores(repository: &Path) -> Vec<PathBuf> {
    let objects = repository.join("objects");
    std::fs::read_to_string(objects.join("info/alternates"))
        .map(|alternates| alternates.lines().map(|line| objects.join(line)).collect())
        .unwrap_or_default()
}

impl<E: Executor + Clone + 'static> LocalGit<E> {
    /// The worktree the repository answers to: the committing grant knows
    /// which worktree holds the `.git` file that names it. `git` enters the
    /// worktree — the subagent's authorized space — and discovers the
    /// repository through that file, opening only paths the grants cover.
    fn enter<'a>(&'a self, repository: &'a Path) -> &'a Path {
        self.committing
            .iter()
            .find(|granted| canonical(repository).parent() == Some(&*canonical(granted.git_dir())))
            .map_or(repository, CommittingWorktree::worktree)
    }

    /// The sandbox the export's git runs in, and the program it launches:
    /// the host's resolved `git`, falling back to a bare `git` when the host
    /// itself cannot find one — the error then is the launch's to report.
    /// The working directory is the worktree the run enters: it lives inside
    /// the subagent's authorized space, and a sandbox denies exec beneath a
    /// working directory, so the git binary's own directory could never be
    /// chosen. The repository is granted read beside the subagent's grants:
    /// it lives inside the main worktree's git directory, outside them. So
    /// are the pieces of the main repository the export's reads resolve to —
    /// the object store its alternates name and the configuration its own
    /// includes — spelled exactly, because a directory grant would open
    /// everything beneath it.
    async fn sandbox(
        &self,
        repository: &Path,
        enter: &Path,
    ) -> Result<(Sandbox, String), ExportError> {
        let git = host_git_binary();
        // On Windows the working directory is the git binary's own: git's
        // resolution of the directory it runs in asks more of ancestors than
        // the traverse the sandbox grants them — reading a directory's entry
        // needs its parent's list — where the binary's system ancestors are
        // legible to every package.
        #[cfg(windows)]
        let working_dir = git
            .as_ref()
            .map_or_else(|| enter.to_path_buf(), |(_, dir)| dir.clone());
        #[cfg(not(windows))]
        let working_dir = enter.to_path_buf();
        let grants: Vec<Grant> =
            self.writable
                .iter()
                .map(|path| resolved(path).map(|p| Grant::new(p, Access::WRITE)))
                .chain([resolved(&self.working_dir).map(|p| Grant::new(p, Access::READ))])
                .chain(
                    self.readable
                        .iter()
                        .map(|path| resolved(path).map(|p| Grant::new(p, Access::READ))),
                )
                .chain(git.as_ref().map(|(_, dir)| {
                    resolved(dir).map(|p| Grant::new(p, Access::READ | Access::EXEC))
                }))
                .chain(
                    // The repository `git -C` enters: it lives inside the main
                    // worktree's git directory, outside the subagent's grants.
                    [resolved(repository).map(|p| Grant::new(p, Access::READ))],
                )
                .chain(
                    // The worktrees `git -C` may enter: the repository lives
                    // inside the main worktree's git directory, and the worktree
                    // is outside the subagent's grants — its `.git` file names
                    // the repository.
                    self.committing.iter().map(|granted| {
                        resolved(granted.worktree()).map(|p| Grant::new(p, Access::READ))
                    }),
                )
                .chain(
                    // The alternates the repository's object store names: the
                    // main object store, as a subtree.
                    borrowed_object_stores(repository)
                        .into_iter()
                        .map(|store| resolved(&store).map(|p| Grant::new(p, Access::READ))),
                )
                .chain(
                    // The configuration `include.path` reaches: one file, never
                    // its directory.
                    borrowed_object_stores(repository)
                        .into_iter()
                        .filter_map(|store| store.parent().map(Path::to_path_buf))
                        .map(|common| common.join("config"))
                        .filter(|config| config.is_file())
                        .map(|config| resolved(&config).map(|p| Grant::new(p, Access::READ))),
                )
                .collect::<heel::Result<_>>()
                .map_err(|error| ExportError::Launch(error.to_string()))?;
        let config = SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(canonical(&working_dir))
            .grants(grants)
            .security(SecurityConfig::interactive())
            .priority(self.priority)
            .build();
        let sandbox = Sandbox::with_config_and_executor(config, self.executor.clone())
            .await
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        let program = git.map_or_else(
            || "git".to_string(),
            // The binary is spelled canonically the same way its directory's
            // grant is: a firmlinked `/Applications` or an 8.3 alias resolves
            // to the object the grant names, where the spelled path does not.
            |(path, _)| canonical(&path).to_string_lossy().into_owned(),
        );
        Ok((sandbox, program))
    }

    async fn run_in_sandbox(&self, run: SandboxedRun<'_>) -> Result<SandboxedExit, ExportError> {
        let enter = self.enter(run.repository);
        let (sandbox, program) = self.sandbox(run.repository, enter).await?;
        let child = sandbox
            .command(program)
            .args(run.args)
            .env("HOME", self.working_dir.to_string_lossy())
            // An absolute `GIT_DIR` is used as it stands — `git` discovers a
            // relative one by resolving it, a read on every ancestor that a
            // traverse grant does not carry.
            .env("GIT_DIR", canonical(run.repository).to_string_lossy())
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
            Ok(ContainerExecOutcome::Completed { output, .. }) => Ok(SandboxedExit {
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
