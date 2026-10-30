//! The git the host runs, on the host and outside any sandbox.
//!
//! It is pointed at a repository directly, never through a worktree, so no
//! `.git` file, `commondir` or per-worktree configuration a subagent wrote is
//! consulted. The only configuration read is the repository's own, which no
//! grant makes writable; system and global configuration and every inherited
//! `GIT_*` variable are dropped, and hooks, signature display, replace objects,
//! lazy fetching and the file-system monitor, the paths by which a query could
//! start another program or read another repository's objects, are off. The
//! hooks are off on the command line, which wins over every configuration file,
//! the include of the main repository's among them: a relative hooks path in a
//! repository the host creates would otherwise be resolved against the
//! directory git runs in, which can lie where a subagent writes.
//! Nothing here is ever pointed at a repository a subagent can write.

use std::path::{Path, PathBuf};

use async_channel::Receiver;
use futures_lite::io::AsyncWriteExt as _;

use super::{CommitLedgerError, NULL_DEVICE, SubagentGrantError};

/// The oldest git the subagent commit grant works with: its snapshot refresh
/// needs negative refspecs, which arrived in git 2.29, and the
/// import moves symbolic HEADs with `symref-update`, which arrived in
/// git 2.46.
const MINIMUM_VERSION: (u32, u32) = (2, 46);

/// The object directories a host git reads and writes while a pack is
/// quarantined.
#[derive(Debug, Clone, Copy)]
pub(super) struct Quarantine<'a> {
    /// Where new objects are written: the quarantine, host-created and
    /// host-written only.
    pub objects: &'a Path,
    /// The main object store, read through as an alternate.
    pub alternate: &'a Path,
}

/// One host git invocation.
pub(super) struct HostGit<'a> {
    /// The repository, or `None` for a command that needs none.
    git_dir: Option<&'a Path>,
    quarantine: Option<Quarantine<'a>>,
    args: &'a [&'a str],
    /// Whether dropping the invocation leaves git running. A command that
    /// writes refs holds lock files while it works, which killing it would
    /// leave behind.
    keep_running: bool,
}

impl<'a> HostGit<'a> {
    /// A git invocation against the repository at `git_dir`.
    pub(super) const fn in_repository(git_dir: &'a Path, args: &'a [&'a str]) -> Self {
        Self {
            git_dir: Some(git_dir),
            quarantine: None,
            args,
            keep_running: false,
        }
    }

    /// Lets git finish what it started when this invocation is dropped. Its
    /// input then ends, so it stops at the end of what it was sent.
    pub(super) const fn finish_when_dropped(mut self) -> Self {
        self.keep_running = true;
        self
    }

    /// Reads and writes objects through the quarantine and the main store.
    pub(super) const fn through(mut self, quarantine: Quarantine<'a>) -> Self {
        self.quarantine = Some(quarantine);
        self
    }

    fn command(&self) -> async_process::Command {
        let mut command = async_process::Command::new("git");
        for (key, _) in std::env::vars_os() {
            if key.to_str().is_some_and(|key| key.starts_with("GIT_")) {
                command.env_remove(key);
            }
        }
        command
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", NULL_DEVICE)
            .env("GIT_NO_LAZY_FETCH", "1")
            .env("GIT_NO_REPLACE_OBJECTS", "1")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env("GIT_TERMINAL_PROMPT", "0");
        if let Some(quarantine) = self.quarantine {
            command
                .env("GIT_OBJECT_DIRECTORY", quarantine.objects)
                .env("GIT_ALTERNATE_OBJECT_DIRECTORIES", quarantine.alternate);
        }
        if let Some(git_dir) = self.git_dir {
            command.current_dir(git_dir).arg("--git-dir").arg(git_dir);
        }
        command
            .args([
                "-c",
                "log.showSignature=false",
                "-c",
                "core.fsmonitor=false",
                "-c",
            ])
            .arg(format!("core.hooksPath={NULL_DEVICE}"))
            .args(self.args)
            .stdout(async_process::Stdio::piped())
            .stderr(async_process::Stdio::piped())
            .kill_on_drop(!self.keep_running);
        command
    }

    fn repository(&self) -> PathBuf {
        self.git_dir.map(Path::to_path_buf).unwrap_or_default()
    }

    /// Runs git with its input fed from `stdin` until that closes, and returns
    /// what it printed whatever its exit status.
    pub(super) async fn output(
        &self,
        stdin: Receiver<Vec<u8>>,
    ) -> Result<std::process::Output, CommitLedgerError> {
        let spawn_error = |source| CommitLedgerError::Spawn {
            repository: self.repository(),
            source,
        };
        let mut command = self.command();
        command.stdin(async_process::Stdio::piped());
        let mut child = command.spawn().map_err(spawn_error)?;
        let mut input = child.stdin.take();
        let feed = async move {
            let Some(input) = input.as_mut() else {
                return Ok(());
            };
            while let Ok(chunk) = stdin.recv().await {
                input.write_all(&chunk).await?;
            }
            input.flush().await
        };
        // The input is closed once everything is written, never earlier: the
        // future owning it ends, and drops it, only then.
        let (written, output) = futures::join!(feed, child.output());
        let output = output.map_err(spawn_error)?;
        // Git may stop reading early, for instance when it rejects what it
        // was sent: its exit status says why.
        match written {
            Err(error) if error.kind() != std::io::ErrorKind::BrokenPipe => Err(spawn_error(error)),
            _ => Ok(output),
        }
    }

    /// Runs git with `stdin` as its input and returns its output, failing
    /// when it exits unsuccessfully.
    pub(super) async fn succeed(
        &self,
        stdin: Vec<u8>,
    ) -> Result<std::process::Output, CommitLedgerError> {
        let output = self.output(input(stdin)).await?;
        if output.status.success() {
            Ok(output)
        } else {
            Err(self.failed(&output))
        }
    }

    /// The error for a git invocation that exited unsuccessfully.
    pub(super) fn failed(&self, output: &std::process::Output) -> CommitLedgerError {
        CommitLedgerError::Git {
            repository: self.repository(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        }
    }
}

/// `data` as a closed input channel, for a command that reads it as stdin.
pub(super) fn input(data: Vec<u8>) -> Receiver<Vec<u8>> {
    let (tx, rx) = async_channel::bounded(1);
    if !data.is_empty() {
        tx.try_send(data)
            .expect("an empty channel with room for one chunk takes it");
    }
    rx
}

/// Runs a read-only git query against the repository at `git_dir` and
/// returns its output as text.
pub(super) async fn git(git_dir: &Path, args: &[&str]) -> Result<String, CommitLedgerError> {
    let output = HostGit::in_repository(git_dir, args)
        .succeed(Vec::new())
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Checks the git on the host against what the commit grant needs.
pub(super) async fn require_supported_version() -> Result<(), SubagentGrantError> {
    let output = HostGit {
        git_dir: None,
        quarantine: None,
        args: &["--version"],
        keep_running: false,
    }
    .succeed(Vec::new())
    .await?;
    check_version(String::from_utf8_lossy(&output.stdout).trim())
}

/// Accepts the output of `git --version` when it names git 2.46 or later.
pub(super) fn check_version(printed: &str) -> Result<(), SubagentGrantError> {
    let version = printed
        .strip_prefix("git version ")
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|version| {
            let mut parts = version.split('.');
            let major = parts.next()?.parse::<u32>().ok()?;
            let minor = parts.next()?.parse::<u32>().ok()?;
            Some((major, minor))
        });
    match version {
        Some(version) if version >= MINIMUM_VERSION => Ok(()),
        _ => Err(SubagentGrantError::GitTooOld {
            found: printed.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_before_2_46_is_refused_naming_the_version_found() {
        for old in [
            "git version 2.45.9",
            "git version 1.9.5",
            "git version 2.7.4",
        ] {
            let error = check_version(old).expect_err("symref-update needs git 2.46");
            assert!(
                matches!(&error, SubagentGrantError::GitTooOld { found } if found == old),
                "{error}"
            );
            assert!(error.to_string().contains(old), "{error}");
        }
        for fine in [
            "git version 2.46.0",
            "git version 2.54.0 (Apple Git-157)",
            "git version 3.0.1",
        ] {
            check_version(fine).expect("a supported git");
        }
        assert!(matches!(
            check_version("not git"),
            Err(SubagentGrantError::GitTooOld { .. })
        ));
    }
}
