//! Background job registry for tracking and managing background bash tasks.
//!
//! Uses an internal command channel to avoid shared locks.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use async_channel::{Receiver, Sender};

use crate::permission::BashMode;

/// Information about a running or completed background job.
#[derive(Debug, Clone)]
pub struct JobInfo {
    /// Process ID.
    pub pid: u32,
    /// Shell session id that owns this job.
    pub shell_id: String,
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

enum JobCommand {
    Register {
        pid: u32,
        shell_id: String,
        script: String,
        mode: BashMode,
        output_path: Option<PathBuf>,
    },
    Complete {
        pid: u32,
        exit_code: i32,
        output_path: Option<PathBuf>,
    },
    Fail {
        pid: u32,
        error: String,
        output_path: Option<PathBuf>,
    },
    List {
        reply: Sender<Vec<JobInfo>>,
    },
    Get {
        pid: u32,
        reply: Sender<Option<JobInfo>>,
    },
    Kill {
        pid: u32,
        reply: Sender<bool>,
    },
    KillBySession {
        shell_id: String,
        reply: Sender<usize>,
    },
    FormatRunning {
        reply: Sender<String>,
    },
    HasRunning {
        reply: Sender<bool>,
    },
}

/// Registry handle for tracking background jobs.
///
/// Cloning keeps a sender to the same registry service.
#[derive(Clone, Debug)]
pub struct JobRegistry {
    tx: Sender<JobCommand>,
}

/// Background service that owns the job registry state.
pub struct JobRegistryService {
    rx: Receiver<JobCommand>,
}

/// Creates a registry handle and its background service.
#[must_use]
pub fn job_registry_channel() -> (JobRegistry, JobRegistryService) {
    let (tx, rx) = async_channel::unbounded();
    (JobRegistry { tx }, JobRegistryService { rx })
}

impl JobRegistry {
    /// Registers a new running job.
    pub async fn register(
        &self,
        pid: u32,
        shell_id: &str,
        script: &str,
        mode: BashMode,
        output_path: Option<PathBuf>,
    ) {
        self.tx
            .send(JobCommand::Register {
                pid,
                shell_id: shell_id.to_string(),
                script: script.to_string(),
                mode,
                output_path,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Marks a job as completed.
    pub async fn complete(&self, pid: u32, exit_code: i32, output_path: Option<PathBuf>) {
        self.tx
            .send(JobCommand::Complete {
                pid,
                exit_code,
                output_path,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Marks a job as failed.
    pub async fn fail(&self, pid: u32, error: &str, output_path: Option<PathBuf>) {
        self.tx
            .send(JobCommand::Fail {
                pid,
                error: error.to_string(),
                output_path,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Lists all jobs.
    pub async fn list(&self) -> Vec<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::List { reply: reply_tx })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Gets information about a specific job by PID.
    pub async fn get(&self, pid: u32) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::Get {
                pid,
                reply: reply_tx,
            })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Kills a running job by its PID.
    ///
    /// Returns `true` if a job with that PID was found and kill signal was sent.
    pub async fn kill(&self, pid: u32) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::Kill {
                pid,
                reply: reply_tx,
            })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Kills all running jobs owned by a shell session.
    ///
    /// Returns number of jobs that were successfully killed.
    pub async fn kill_by_session(&self, shell_id: &str) -> usize {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::KillBySession {
                shell_id: shell_id.to_string(),
                reply: reply_tx,
            })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Blocking variant of `kill_by_session`, intended for drop-time cleanup.
    #[must_use]
    pub fn kill_by_session_blocking(&self, shell_id: &str) -> usize {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        if self
            .tx
            .send_blocking(JobCommand::KillBySession {
                shell_id: shell_id.to_string(),
                reply: reply_tx,
            })
            .is_err()
        {
            return 0;
        }
        reply_rx.recv_blocking().unwrap_or(0)
    }

    /// Formats only RUNNING jobs for compression preservation.
    ///
    /// Returns a human-readable summary of running jobs only.
    pub async fn format_running_jobs(&self) -> String {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::FormatRunning { reply: reply_tx })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Returns true if there are any running jobs.
    #[must_use]
    pub async fn has_running(&self) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::HasRunning { reply: reply_tx })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }
}

impl JobRegistryService {
    /// Runs the registry service loop until all senders are dropped.
    pub async fn serve(self) {
        let mut jobs: HashMap<u32, JobInfo> = HashMap::new();

        while let Ok(cmd) = self.rx.recv().await {
            match cmd {
                JobCommand::Register {
                    pid,
                    shell_id,
                    script,
                    mode,
                    output_path,
                } => {
                    let info = JobInfo {
                        pid,
                        shell_id,
                        script,
                        mode,
                        started_at: Instant::now(),
                        status: JobStatus::Running,
                        output_path,
                    };
                    jobs.insert(pid, info);
                    tracing::debug!(pid = %pid, "registered background job");
                }
                JobCommand::Complete {
                    pid,
                    exit_code,
                    output_path,
                } => {
                    if let Some(job) = jobs.get_mut(&pid) {
                        job.status = JobStatus::Completed { exit_code };
                        if output_path.is_some() {
                            job.output_path = output_path;
                        }
                        tracing::debug!(pid = %pid, exit_code = %exit_code, "job completed");
                    }
                }
                JobCommand::Fail {
                    pid,
                    error,
                    output_path,
                } => {
                    if let Some(job) = jobs.get_mut(&pid) {
                        job.status = JobStatus::Failed { error };
                        if output_path.is_some() {
                            job.output_path = output_path;
                        }
                        tracing::debug!(pid = %pid, "job failed");
                    }
                }
                JobCommand::List { reply } => {
                    let items = jobs.values().cloned().collect();
                    let _ = reply.send(items).await;
                }
                JobCommand::Get { pid, reply } => {
                    let item = jobs.get(&pid).cloned();
                    let _ = reply.send(item).await;
                }
                JobCommand::Kill { pid, reply } => {
                    let result = kill_job(&mut jobs, pid).await;
                    let _ = reply.send(result).await;
                }
                JobCommand::KillBySession { shell_id, reply } => {
                    let pids: Vec<u32> = jobs
                        .iter()
                        .filter_map(|(pid, job)| {
                            if job.shell_id == shell_id && matches!(job.status, JobStatus::Running)
                            {
                                Some(*pid)
                            } else {
                                None
                            }
                        })
                        .collect();

                    let mut killed = 0usize;
                    for pid in pids {
                        if kill_job(&mut jobs, pid).await {
                            killed += 1;
                        }
                    }
                    let _ = reply.send(killed).await;
                }
                JobCommand::FormatRunning { reply } => {
                    let summary = format_running_jobs(&jobs);
                    let _ = reply.send(summary).await;
                }
                JobCommand::HasRunning { reply } => {
                    let has_running = jobs
                        .values()
                        .any(|job| matches!(job.status, JobStatus::Running));
                    let _ = reply.send(has_running).await;
                }
            }
        }
    }
}

async fn kill_job(jobs: &mut HashMap<u32, JobInfo>, pid: u32) -> bool {
    if let Some(job) = jobs.get_mut(&pid) {
        if !matches!(job.status, JobStatus::Running) {
            tracing::warn!(pid = %pid, "cannot kill non-running job");
            return false;
        }

        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
            if result == 0 {
                job.status = JobStatus::Killed;
                tracing::info!(pid = %pid, "killed job");
                return true;
            }
            tracing::warn!(pid = %pid, "failed to kill job");
            return false;
        }

        #[cfg(windows)]
        {
            let output = async_process::Command::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .output()
                .await;
            match output {
                Ok(_) => {
                    job.status = JobStatus::Killed;
                    tracing::info!(pid = %pid, "killed job (Windows)");
                    return true;
                }
                Err(err) => {
                    tracing::warn!(pid = %pid, error = %err, "failed to kill job (Windows)");
                    return false;
                }
            }
        }
    }

    false
}

fn format_running_jobs(jobs: &HashMap<u32, JobInfo>) -> String {
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
            .map_or_else(|| "(no output)".to_string(), |p| p.display().to_string());

        output.push_str(&format!(
            "  - PID {}: `{}` → {}\n",
            job.pid, script_preview, output_path
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use executor_core::Executor;
    use executor_core::Task;
    use executor_core::tokio::TokioGlobal;

    #[tokio::test]
    async fn test_job_registry_lifecycle() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(
                12345,
                "shell-test",
                "sleep 10",
                BashMode::Sandboxed,
                Some(PathBuf::from("outputs/12345.txt")),
            )
            .await;

        let jobs = registry.list().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].pid, 12345);
        assert_eq!(jobs[0].shell_id, "shell-test");
        assert!(matches!(jobs[0].status, JobStatus::Running));

        registry.complete(12345, 0, None).await;
        let job = registry.get(12345).await.unwrap();
        assert!(matches!(job.status, JobStatus::Completed { exit_code: 0 }));
    }

    #[tokio::test]
    async fn test_format_running_jobs() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(1, "shell-a", "echo hello", BashMode::Sandboxed, None)
            .await;
        registry
            .register(2, "shell-b", "sleep 5", BashMode::Sandboxed, None)
            .await;
        registry.complete(2, 0, None).await;

        let running = registry.format_running_jobs().await;
        assert!(running.contains("PID 1"));
        assert!(!running.contains("PID 2"));
    }

    #[tokio::test]
    async fn test_kill_by_session_no_match() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(1, "shell-a", "echo hello", BashMode::Sandboxed, None)
            .await;
        registry.complete(1, 0, None).await;

        let killed = registry.kill_by_session("shell-b").await;
        assert_eq!(killed, 0);
    }
}
