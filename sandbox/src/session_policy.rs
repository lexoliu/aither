//! The sandbox policy a session gives its commands, for hosts that start
//! their own processes inside the session sandbox.
//!
//! [`TerminalTool`](crate::TerminalTool) launches every sandboxed command
//! under a [`SessionSandboxPolicy`]; a host that runs other long-lived
//! processes in the same session (an ACP agent, say) builds the policy from
//! the same writable and readable sets and hands it to
//! [`heel::SandboxConfig`] and the process, so both run under exactly the
//! same grants and environment.

use std::path::{Path, PathBuf};

use heel::Grant;

#[cfg(all(unix, feature = "jobserver"))]
use crate::terminal::job_fifo_grant;
use crate::terminal::{
    linked_worktree_grants, local_path_grants, path_directory_grants,
    prepare_session_runtime_environment, session_write_denylist,
};
use crate::{GitDirAccess, PreparedTools, TerminalError};

/// The filesystem grants and process environment a session sandbox gives a
/// command.
///
/// The grants are: every writable path with write and execute access, every
/// readable path with read access, the managed tools directory with execute
/// access, the shared `uv` directory with write and execute access, the
/// command's jobserver fifo (see [`Self::with_job_fifo`]) and the host
/// toolchain homes (`CARGO_HOME`, `RUSTUP_HOME`) that exist, with write and
/// execute access. A granted linked git worktree also brings its repository's
/// common git directory, with the worktree's access, once git's back-links
/// confirm it. Every existing directory of the session's `PATH` is granted
/// read and execute access, never write. The policy also carries the write denylist
/// ([`Self::write_denylist`]). The environment is the session's `HOME`, temporary and
/// cache directories, `PATH` led by the managed tools directory, `uv` and
/// toolchain variables, and the jobserver variables.
///
/// Per-command additions stay with the terminal: the git shim's worktree
/// map, the embedder's environment source and the job marker.
#[derive(Debug, Clone)]
pub struct SessionSandboxPolicy {
    grants: Vec<Grant>,
    environment: Vec<(String, String)>,
    write_denylist: heel::WriteDenylist,
}

impl SessionSandboxPolicy {
    /// Builds the policy of a session whose directory is `working_dir`, for
    /// the given writable and readable sets and prepared tools.
    ///
    /// `git_dir` is the terminal's access to the common git directory of a
    /// linked worktree it may write.
    ///
    /// Creates the session's `tmp`, `.cache` and `.config` directories,
    /// which its environment points at.
    ///
    /// # Errors
    /// Returns an error when a grant path cannot be resolved, a directory
    /// cannot be created, or an environment path is not valid UTF-8.
    pub async fn new(
        working_dir: &Path,
        writable: &[PathBuf],
        readable: &[PathBuf],
        git_dir: GitDirAccess,
        tools: &PreparedTools<PathBuf>,
    ) -> Result<Self, TerminalError> {
        let environment = prepare_session_runtime_environment(working_dir, tools).await?;
        let setup = |error: heel::Error| TerminalError::SandboxSetup(error.to_string());
        // The `PATH` directories lead, so a writable grant over one of them
        // is never narrowed by them.
        let mut grants = path_directory_grants(&environment).map_err(setup)?;
        grants.extend(local_path_grants(writable, readable, tools).map_err(setup)?);
        grants.extend(
            linked_worktree_grants(writable, readable, git_dir)
                .await
                .map_err(setup)?,
        );
        Ok(Self {
            grants,
            environment,
            write_denylist: session_write_denylist(working_dir, tools),
        })
    }

    /// Adds a command's jobserver fifo: write access to it and the
    /// variables that point the command's build tools at it.
    ///
    /// # Errors
    /// Returns an error when the fifo path cannot be resolved.
    #[cfg(all(unix, feature = "jobserver"))]
    pub fn with_job_fifo(
        mut self,
        fifo: &crate::jobserver::CommandFifo,
    ) -> Result<Self, TerminalError> {
        self.grants.push(
            job_fifo_grant(fifo.path())
                .map_err(|error| TerminalError::SandboxSetup(error.to_string()))?,
        );
        self.environment.extend(fifo.environment());
        Ok(self)
    }

    /// The filesystem grants, in the order the sandbox applies them.
    #[must_use]
    pub fn grants(&self) -> &[Grant] {
        &self.grants
    }

    /// The paths the sandbox must not write, however they are granted: heel's
    /// persistence table, less cargo's git cache, the session directory and the `uv` cache.
    /// A host building its own
    /// [`heel::SandboxConfig`] passes it to `write_denylist` (not on
    /// Windows, whose backend refuses one).
    #[must_use]
    pub const fn write_denylist(&self) -> &heel::WriteDenylist {
        &self.write_denylist
    }

    /// The process environment, as name and value pairs.
    #[must_use]
    pub fn environment(&self) -> &[(String, String)] {
        &self.environment
    }

    /// Splits the policy into its grants and environment.
    #[must_use]
    pub fn into_parts(self) -> (Vec<Grant>, Vec<(String, String)>) {
        (self.grants, self.environment)
    }
}
