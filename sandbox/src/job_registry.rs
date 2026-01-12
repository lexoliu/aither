//! Background job registry for tracking and managing background bash tasks.
//!
//! Provides a thread-safe registry for tracking background jobs spawned by the bash tool.
//! Jobs are keyed by PID for simplicity and standard Unix conventions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::permission::BashMode;

/// Information about a running or completed background job.
#[derive(Debug, Clone)]
pub struct JobInfo {
    /// Process ID.
    pub pid: u32,
    /// The script that was executed.
    pub script: String,
    /// The permission mode used.
    pub mode: BashMode,
    /// When the job was started.
    pub started_at: Instant,
    /// Current status of the job.
    pub status: JobStatus,
    /// Path to output file.
    pub output_path: Option<PathBuf>,
}

/// Status of a background job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Job is currently running.
    Running,
    /// Job completed successfully.
    Completed {
        /// Exit code of the process.
        exit_code: i32,
    },
    /// Job failed with an error.
    Failed {
        /// Error message.
        error: String,
    },
    /// Job was killed by user.
    Killed,
}

/// Registry for tracking background jobs.
///
/// Thread-safe via internal `Arc<RwLock>`. Can be cloned to share across components.
/// Jobs are keyed by PID for simplicity.
#[derive(Debug, Clone, Default)]
pub struct JobRegistry {
    jobs: Arc<RwLock<HashMap<u32, JobInfo>>>,
}

impl JobRegistry {
    /// Creates a new empty job registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new running job.
    pub fn register(&self, pid: u32, script: &str, mode: BashMode, output_path: Option<PathBuf>) {
        let info = JobInfo {
            pid,
            script: script.to_string(),
            mode,
            started_at: Instant::now(),
            status: JobStatus::Running,
            output_path,
        };

        self.jobs
            .write()
            .expect("job registry lock poisoned")
            .insert(pid, info);

        tracing::debug!(pid = %pid, "registered background job");
    }

    /// Marks a job as completed.
    pub fn complete(&self, pid: u32, exit_code: i32) {
        if let Some(job) = self
            .jobs
            .write()
            .expect("job registry lock poisoned")
            .get_mut(&pid)
        {
            job.status = JobStatus::Completed { exit_code };
            tracing::debug!(pid = %pid, exit_code = %exit_code, "job completed");
        }
    }

    /// Marks a job as failed.
    pub fn fail(&self, pid: u32, error: &str) {
        if let Some(job) = self
            .jobs
            .write()
            .expect("job registry lock poisoned")
            .get_mut(&pid)
        {
            job.status = JobStatus::Failed {
                error: error.to_string(),
            };
            tracing::debug!(pid = %pid, error = %error, "job failed");
        }
    }

    /// Lists all jobs.
    pub fn list(&self) -> Vec<JobInfo> {
        self.jobs
            .read()
            .expect("job registry lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Gets information about a specific job by PID.
    pub fn get(&self, pid: u32) -> Option<JobInfo> {
        self.jobs
            .read()
            .expect("job registry lock poisoned")
            .get(&pid)
            .cloned()
    }

    /// Kills a running job by its PID.
    ///
    /// Returns `true` if a job with that PID was found and kill signal was sent.
    pub fn kill(&self, pid: u32) -> bool {
        let mut jobs = self.jobs.write().expect("job registry lock poisoned");

        if let Some(job) = jobs.get_mut(&pid) {
            if !matches!(job.status, JobStatus::Running) {
                tracing::warn!(pid = %pid, "cannot kill non-running job");
                return false;
            }

            // Send SIGKILL to the process
            #[cfg(unix)]
            {
                let result = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
                if result == 0 {
                    job.status = JobStatus::Killed;
                    tracing::info!(pid = %pid, "killed job");
                    return true;
                }
                tracing::warn!(pid = %pid, "failed to kill job");
            }

            #[cfg(windows)]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/PID", &pid.to_string()])
                    .output();
                job.status = JobStatus::Killed;
                tracing::info!(pid = %pid, "killed job (Windows)");
                return true;
            }
        }

        false
    }

    /// Formats only RUNNING jobs for compression preservation.
    ///
    /// Returns a human-readable summary of running jobs only.
    pub fn format_running_jobs(&self) -> String {
        let jobs = self.jobs.read().expect("job registry lock poisoned");

        let running: Vec<_> = jobs
            .values()
            .filter(|job| matches!(job.status, JobStatus::Running))
            .collect();

        if running.is_empty() {
            return String::new();
        }

        let mut output = String::new();
        for job in running {
            let script_preview = if job.script.len() > 50 {
                format!("{}...", &job.script[..50])
            } else {
                job.script.clone()
            };

            let output_path = job
                .output_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(no output)".to_string());

            output.push_str(&format!(
                "  - PID {}: `{}` → {}\n",
                job.pid, script_preview, output_path
            ));
        }

        output
    }

    /// Returns true if there are any running jobs.
    #[must_use]
    pub fn has_running(&self) -> bool {
        self.jobs
            .read()
            .expect("job registry lock poisoned")
            .values()
            .any(|job| matches!(job.status, JobStatus::Running))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_registry_lifecycle() {
        let registry = JobRegistry::new();

        registry.register(
            12345,
            "sleep 10",
            BashMode::Sandboxed,
            Some(PathBuf::from("outputs/12345.txt")),
        );

        let jobs = registry.list();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].pid, 12345);
        assert!(matches!(jobs[0].status, JobStatus::Running));

        registry.complete(12345, 0);
        let job = registry.get(12345).unwrap();
        assert!(matches!(job.status, JobStatus::Completed { exit_code: 0 }));
    }

    #[test]
    fn test_format_running_jobs() {
        let registry = JobRegistry::new();

        registry.register(
            11111,
            "sleep 100",
            BashMode::Sandboxed,
            Some(PathBuf::from("outputs/11111.txt")),
        );

        registry.register(
            22222,
            "echo done",
            BashMode::Sandboxed,
            Some(PathBuf::from("outputs/22222.txt")),
        );

        registry.complete(22222, 0);

        let running = registry.format_running_jobs();
        assert!(running.contains("11111"));
        assert!(!running.contains("22222"));
    }
}
