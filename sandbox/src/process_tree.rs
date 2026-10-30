//! Killing a process together with everything it started.

/// Why a process tree could not be killed.
#[derive(Debug, thiserror::Error)]
pub enum KillTreeError {
    /// The signal to the process group failed.
    #[cfg(unix)]
    #[error("failed to signal the process group: {0}")]
    Signal(#[source] std::io::Error),
    /// `taskkill` could not be started.
    #[cfg(windows)]
    #[error("failed to run taskkill: {0}")]
    Taskkill(#[source] std::io::Error),
    /// `taskkill` ran and reported failure.
    #[cfg(windows)]
    #[error("taskkill reported failure ({status}): {stderr}")]
    TaskkillFailed {
        /// The exit status `taskkill` returned.
        status: std::process::ExitStatus,
        /// What `taskkill` wrote to its standard error.
        stderr: String,
    },
}

/// Kills the process `pid` and every process it started.
///
/// On Unix `pid` must lead its own process group (spawned with
/// `process_group(0)`), and the group receives `SIGKILL`. On Windows
/// `taskkill /F /T` ends the tree.
///
/// # Errors
///
/// Returns [`KillTreeError`] when the kill could not be delivered.
pub async fn kill_process_tree(pid: u32) -> Result<(), KillTreeError> {
    #[cfg(unix)]
    {
        // SAFETY: `kill` takes plain integers and touches no memory.
        let result = unsafe { libc::kill(-pid.cast_signed(), libc::SIGKILL) };
        if result == 0 {
            Ok(())
        } else {
            Err(KillTreeError::Signal(std::io::Error::last_os_error()))
        }
    }

    #[cfg(windows)]
    {
        let output = async_process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output()
            .await
            .map_err(KillTreeError::Taskkill)?;
        if output.status.success() {
            Ok(())
        } else {
            Err(KillTreeError::TaskkillFailed {
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })
        }
    }
}
