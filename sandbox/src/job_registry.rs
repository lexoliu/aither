//! Background job registry for tracking and managing background bash tasks.
//!
//! Uses an internal command channel to avoid shared locks.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use async_channel::{Receiver, Sender};
use futures_lite::io::AsyncWriteExt;

use crate::permission::BashMode;

/// Information about a running or completed background job.
#[derive(Debug, Clone)]
pub struct JobInfo {
    /// Process ID.
    pub pid: u32,
    /// Stable task identifier exposed to the model.
    pub task_id: String,
    /// Stateless execution key that owns this job.
    pub execution_key: String,
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

/// Sender used by the registry to push bytes into a running process stdin.
pub type TerminalInputSender = Sender<Vec<u8>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalStream {
    Stdout,
    Stderr,
}

#[derive(Debug)]
struct JobState {
    info: JobInfo,
    terminal_buffer: Vec<u8>,
    stdout_buffer: Vec<u8>,
    stderr_buffer: Vec<u8>,
    stdout_closed: bool,
    stderr_closed: bool,
    input_tx: Option<TerminalInputSender>,
    kill_switch: Option<Sender<()>>,
    output_redirect: Option<async_fs::File>,
    finished_at: Option<Instant>,
}

enum JobCommand {
    Register {
        pid: u32,
        task_id: String,
        execution_key: String,
        script: String,
        mode: BashMode,
        output_path: Option<PathBuf>,
    },
    AttachTerminalInput {
        pid: u32,
        input_tx: TerminalInputSender,
    },
    AttachKillSwitch {
        pid: u32,
        kill_tx: Sender<()>,
    },
    AppendTerminalOutput {
        pid: u32,
        stream: TerminalStream,
        chunk: Vec<u8>,
    },
    CloseTerminalStream {
        pid: u32,
        stream: TerminalStream,
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
    KillByTaskId {
        task_id: String,
        reply: Sender<bool>,
    },
    KillByExecutionKey {
        execution_key: String,
        reply: Sender<usize>,
    },
    InputTerminal {
        task_id: String,
        bytes: Vec<u8>,
        reply: Sender<Result<(), String>>,
    },
    StartOutputRedirect {
        task_id: String,
        output_path: PathBuf,
        reply: Sender<Result<Vec<u8>, String>>,
    },
    TerminalOutput {
        pid: u32,
        reply: Sender<Option<(Vec<u8>, Vec<u8>)>>,
    },
    TerminalStreamsClosed {
        pid: u32,
        reply: Sender<bool>,
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
        task_id: &str,
        execution_key: &str,
        script: &str,
        mode: BashMode,
        output_path: Option<PathBuf>,
    ) {
        self.tx
            .send(JobCommand::Register {
                pid,
                task_id: task_id.to_string(),
                execution_key: execution_key.to_string(),
                script: script.to_string(),
                mode,
                output_path,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Attaches a stdin input channel to a running job.
    pub async fn attach_terminal_input(&self, pid: u32, input_tx: TerminalInputSender) {
        self.tx
            .send(JobCommand::AttachTerminalInput { pid, input_tx })
            .await
            .expect("job registry service unavailable");
    }

    /// Attaches a kill switch used by runtimes without host-visible PIDs.
    pub async fn attach_kill_switch(&self, pid: u32, kill_tx: Sender<()>) {
        self.tx
            .send(JobCommand::AttachKillSwitch { pid, kill_tx })
            .await
            .expect("job registry service unavailable");
    }

    /// Appends stdout bytes for a running job.
    pub async fn append_stdout(&self, pid: u32, chunk: Vec<u8>) {
        self.tx
            .send(JobCommand::AppendTerminalOutput {
                pid,
                stream: TerminalStream::Stdout,
                chunk,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Appends stderr bytes for a running job.
    pub async fn append_stderr(&self, pid: u32, chunk: Vec<u8>) {
        self.tx
            .send(JobCommand::AppendTerminalOutput {
                pid,
                stream: TerminalStream::Stderr,
                chunk,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Marks stdout stream closed.
    pub async fn close_stdout(&self, pid: u32) {
        self.tx
            .send(JobCommand::CloseTerminalStream {
                pid,
                stream: TerminalStream::Stdout,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Marks stderr stream closed.
    pub async fn close_stderr(&self, pid: u32) {
        self.tx
            .send(JobCommand::CloseTerminalStream {
                pid,
                stream: TerminalStream::Stderr,
            })
            .await
            .expect("job registry service unavailable");
    }

    /// Sends bytes to a running task terminal stdin.
    pub async fn input_terminal(&self, task_id: &str, bytes: Vec<u8>) -> Result<(), String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::InputTerminal {
                task_id: task_id.to_string(),
                bytes,
                reply: reply_tx,
            })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Starts continuous output redirection for a task and returns current snapshot bytes.
    pub async fn start_output_redirect(
        &self,
        task_id: &str,
        output_path: PathBuf,
    ) -> Result<Vec<u8>, String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::StartOutputRedirect {
                task_id: task_id.to_string(),
                output_path,
                reply: reply_tx,
            })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Returns true when both stdout/stderr streams are closed for the PID.
    pub async fn terminal_streams_closed(&self, pid: u32) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::TerminalStreamsClosed {
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

    /// Returns collected stdout/stderr buffers for a PID.
    pub async fn terminal_output(&self, pid: u32) -> Option<(Vec<u8>, Vec<u8>)> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::TerminalOutput {
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

    /// Kills a running job by its task ID.
    pub async fn kill_by_task_id(&self, task_id: &str) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::KillByTaskId {
                task_id: task_id.to_string(),
                reply: reply_tx,
            })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Kills all running jobs owned by an execution key.
    pub async fn kill_by_execution_key(&self, execution_key: &str) -> usize {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(JobCommand::KillByExecutionKey {
                execution_key: execution_key.to_string(),
                reply: reply_tx,
            })
            .await
            .expect("job registry service unavailable");
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Blocking variant of `kill_by_execution_key`, intended for drop-time cleanup.
    #[must_use]
    pub fn kill_by_execution_key_blocking(&self, execution_key: &str) -> usize {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        if self
            .tx
            .send_blocking(JobCommand::KillByExecutionKey {
                execution_key: execution_key.to_string(),
                reply: reply_tx,
            })
            .is_err()
        {
            return 0;
        }
        reply_rx.recv_blocking().unwrap_or(0)
    }

    /// Formats only RUNNING jobs for compression preservation.
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
        let mut jobs: HashMap<u32, JobState> = HashMap::new();

        while let Ok(cmd) = self.rx.recv().await {
            match cmd {
                JobCommand::Register {
                    pid,
                    task_id,
                    execution_key,
                    script,
                    mode,
                    output_path,
                } => {
                    let info = JobInfo {
                        pid,
                        task_id,
                        execution_key,
                        script,
                        mode,
                        started_at: Instant::now(),
                        status: JobStatus::Running,
                        output_path,
                    };
                    jobs.insert(
                        pid,
                        JobState {
                            info,
                            terminal_buffer: Vec::new(),
                            stdout_buffer: Vec::new(),
                            stderr_buffer: Vec::new(),
                            stdout_closed: false,
                            stderr_closed: false,
                            input_tx: None,
                            kill_switch: None,
                            output_redirect: None,
                            finished_at: None,
                        },
                    );
                    tracing::debug!(pid = %pid, "registered background job");
                }
                JobCommand::AttachTerminalInput { pid, input_tx } => {
                    if let Some(job) = jobs.get_mut(&pid) {
                        job.input_tx = Some(input_tx);
                    }
                }
                JobCommand::AttachKillSwitch { pid, kill_tx } => {
                    if let Some(job) = jobs.get_mut(&pid) {
                        job.kill_switch = Some(kill_tx);
                    }
                }
                JobCommand::AppendTerminalOutput { pid, stream, chunk } => {
                    if let Some(job) = jobs.get_mut(&pid) {
                        if chunk.is_empty() {
                            continue;
                        }
                        job.terminal_buffer.extend_from_slice(&chunk);
                        if let Some(file) = job.output_redirect.as_mut()
                            && let Err(error) = file.write_all(&chunk).await
                        {
                            mark_job_failed(
                                job,
                                format!(
                                    "failed to append redirected output for task {}: {error}",
                                    job.info.task_id
                                ),
                            );
                            tracing::error!(
                                pid = %pid,
                                error = %error,
                                "failed to append redirected output"
                            );
                        }
                        match stream {
                            TerminalStream::Stdout => job.stdout_buffer.extend_from_slice(&chunk),
                            TerminalStream::Stderr => job.stderr_buffer.extend_from_slice(&chunk),
                        }
                    }
                }
                JobCommand::CloseTerminalStream { pid, stream } => {
                    if let Some(job) = jobs.get_mut(&pid) {
                        match stream {
                            TerminalStream::Stdout => job.stdout_closed = true,
                            TerminalStream::Stderr => job.stderr_closed = true,
                        }
                    }
                }
                JobCommand::Complete {
                    pid,
                    exit_code,
                    output_path,
                } => {
                    if let Some(job) = jobs.get_mut(&pid) {
                        if !matches!(job.info.status, JobStatus::Running) {
                            tracing::debug!(
                                pid = %pid,
                                status = ?job.info.status,
                                "ignoring complete for non-running job"
                            );
                            continue;
                        }
                        if let Err(error) = finalize_output_redirect(job).await {
                            tracing::error!(
                                pid = %pid,
                                error = %error,
                                "failed to flush redirected output on complete"
                            );
                            mark_job_failed(job, error);
                            continue;
                        }
                        job.info.status = JobStatus::Completed { exit_code };
                        job.finished_at = Some(Instant::now());
                        job.kill_switch = None;
                        if output_path.is_some() {
                            job.info.output_path = output_path;
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
                        if !matches!(job.info.status, JobStatus::Running) {
                            tracing::debug!(
                                pid = %pid,
                                status = ?job.info.status,
                                "ignoring fail for non-running job"
                            );
                            continue;
                        }
                        let status_error = if let Err(finalize_error) =
                            finalize_output_redirect(job).await
                        {
                            format!(
                                "{error}; additionally failed to flush redirected output: {finalize_error}"
                            )
                        } else {
                            error
                        };
                        mark_job_failed(job, status_error);
                        job.kill_switch = None;
                        if output_path.is_some() {
                            job.info.output_path = output_path;
                        }
                        tracing::debug!(pid = %pid, "job failed");
                    }
                }
                JobCommand::List { reply } => {
                    let items = jobs.values().map(|state| state.info.clone()).collect();
                    let _ = reply.send(items).await;
                }
                JobCommand::Get { pid, reply } => {
                    let item = jobs.get(&pid).map(|state| state.info.clone());
                    let _ = reply.send(item).await;
                }
                JobCommand::Kill { pid, reply } => {
                    let result = kill_job(&mut jobs, pid).await;
                    let _ = reply.send(result).await;
                }
                JobCommand::KillByTaskId { task_id, reply } => {
                    let result = if let Some(pid) = find_pid_by_task_id(&jobs, &task_id) {
                        kill_job(&mut jobs, pid).await
                    } else {
                        false
                    };
                    let _ = reply.send(result).await;
                }
                JobCommand::KillByExecutionKey {
                    execution_key,
                    reply,
                } => {
                    let pids: Vec<u32> = jobs
                        .iter()
                        .filter_map(|(pid, state)| {
                            if state.info.execution_key == execution_key
                                && matches!(state.info.status, JobStatus::Running)
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
                JobCommand::InputTerminal {
                    task_id,
                    bytes,
                    reply,
                } => {
                    let result = if let Some(pid) = find_pid_by_task_id(&jobs, &task_id) {
                        if let Some(job) = jobs.get(&pid) {
                            if !matches!(job.info.status, JobStatus::Running) {
                                Err(format!(
                                    "task {} is not running; current status: {:?}",
                                    task_id, job.info.status
                                ))
                            } else if let Some(tx) = &job.input_tx {
                                tx.send(bytes).await.map_err(|_| {
                                    format!("terminal input channel closed for task {task_id}")
                                })
                            } else {
                                Err(format!(
                                    "task {} does not support terminal input in this runtime",
                                    task_id
                                ))
                            }
                        } else {
                            Err(format!("unknown task_id: {task_id}"))
                        }
                    } else {
                        Err(format!("unknown task_id: {task_id}"))
                    };
                    let _ = reply.send(result).await;
                }
                JobCommand::StartOutputRedirect {
                    task_id,
                    output_path,
                    reply,
                } => {
                    let result = if let Some(pid) = find_pid_by_task_id(&jobs, &task_id) {
                        if let Some(job) = jobs.get_mut(&pid) {
                            start_output_redirect(job, output_path).await
                        } else {
                            Err(format!("unknown task_id: {task_id}"))
                        }
                    } else {
                        Err(format!("unknown task_id: {task_id}"))
                    };
                    let _ = reply.send(result).await;
                }
                JobCommand::TerminalOutput { pid, reply } => {
                    let output = jobs
                        .get(&pid)
                        .map(|job| (job.stdout_buffer.clone(), job.stderr_buffer.clone()));
                    let _ = reply.send(output).await;
                }
                JobCommand::TerminalStreamsClosed { pid, reply } => {
                    let closed = jobs
                        .get(&pid)
                        .is_some_and(|job| job.stdout_closed && job.stderr_closed);
                    let _ = reply.send(closed).await;
                }
                JobCommand::FormatRunning { reply } => {
                    let summary = format_running_jobs(&jobs);
                    let _ = reply.send(summary).await;
                }
                JobCommand::HasRunning { reply } => {
                    let has_running = jobs
                        .values()
                        .any(|job| matches!(job.info.status, JobStatus::Running));
                    let _ = reply.send(has_running).await;
                }
            }
        }
    }
}

async fn start_output_redirect(
    job: &mut JobState,
    output_path: PathBuf,
) -> Result<Vec<u8>, String> {
    if let Some(existing) = job.info.output_path.as_ref() {
        if existing != &output_path {
            return Err(format!(
                "task {} already redirects output to {}",
                job.info.task_id,
                existing.display()
            ));
        }
        return Ok(job.terminal_buffer.clone());
    }

    let mut file = async_fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&output_path)
        .await
        .map_err(|error| {
            format!(
                "failed to open output redirect file {}: {error}",
                output_path.display()
            )
        })?;

    file.write_all(&job.terminal_buffer)
        .await
        .map_err(|error| {
            format!(
                "failed to write initial output redirect snapshot for task {}: {error}",
                job.info.task_id
            )
        })?;
    file.flush().await.map_err(|error| {
        format!(
            "failed to flush initial output redirect snapshot for task {}: {error}",
            job.info.task_id
        )
    })?;

    job.info.output_path = Some(output_path);
    job.output_redirect = Some(file);
    Ok(job.terminal_buffer.clone())
}

fn mark_job_failed(job: &mut JobState, error: String) {
    job.info.status = JobStatus::Failed { error };
    job.finished_at = Some(Instant::now());
    job.kill_switch = None;
}

async fn finalize_output_redirect(job: &mut JobState) -> Result<(), String> {
    if let Some(mut file) = job.output_redirect.take()
        && let Err(error) = file.flush().await
    {
        return Err(format!(
            "failed to flush redirected output for task {}: {error}",
            job.info.task_id
        ));
    }
    Ok(())
}

fn find_pid_by_task_id(jobs: &HashMap<u32, JobState>, task_id: &str) -> Option<u32> {
    jobs.iter().find_map(|(pid, state)| {
        if state.info.task_id == task_id {
            Some(*pid)
        } else {
            None
        }
    })
}

async fn kill_job(jobs: &mut HashMap<u32, JobState>, pid: u32) -> bool {
    if let Some(job) = jobs.get_mut(&pid) {
        if !matches!(job.info.status, JobStatus::Running) {
            tracing::warn!(pid = %pid, "cannot kill non-running job");
            return false;
        }

        if let Some(kill_switch) = job.kill_switch.take() {
            if kill_switch.send(()).await.is_err() {
                tracing::warn!(pid = %pid, "failed to signal runtime kill switch");
                return false;
            }
            if let Err(error) = finalize_output_redirect(job).await {
                tracing::warn!(
                    pid = %pid,
                    error = %error,
                    "failed to flush redirected output after kill switch signal"
                );
            }
            job.info.status = JobStatus::Killed;
            job.finished_at = Some(Instant::now());
            job.stdout_closed = true;
            job.stderr_closed = true;
            tracing::info!(pid = %pid, "killed job via runtime kill switch");
            return true;
        }

        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
            if result == 0 {
                if let Err(error) = finalize_output_redirect(job).await {
                    tracing::warn!(
                        pid = %pid,
                        error = %error,
                        "failed to flush redirected output after kill"
                    );
                }
                job.info.status = JobStatus::Killed;
                job.finished_at = Some(Instant::now());
                job.stdout_closed = true;
                job.stderr_closed = true;
                tracing::info!(pid = %pid, "killed job");
                return true;
            }
            let error = std::io::Error::last_os_error();
            tracing::warn!(pid = %pid, error = %error, "failed to kill job");
            return false;
        }

        #[cfg(windows)]
        {
            let output = async_process::Command::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .output()
                .await;
            match output {
                Ok(output) if output.status.success() => {
                    if let Err(error) = finalize_output_redirect(job).await {
                        tracing::warn!(
                            pid = %pid,
                            error = %error,
                            "failed to flush redirected output after kill"
                        );
                    }
                    job.info.status = JobStatus::Killed;
                    job.finished_at = Some(Instant::now());
                    job.stdout_closed = true;
                    job.stderr_closed = true;
                    tracing::info!(pid = %pid, "killed job (Windows)");
                    return true;
                }
                Ok(output) => {
                    tracing::warn!(
                        pid = %pid,
                        status = %output.status,
                        stderr = %String::from_utf8_lossy(&output.stderr),
                        "taskkill reported failure"
                    );
                    return false;
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

fn format_running_jobs(jobs: &HashMap<u32, JobState>) -> String {
    let running: Vec<_> = jobs
        .values()
        .filter(|job| matches!(job.info.status, JobStatus::Running))
        .collect();

    if running.is_empty() {
        return String::new();
    }

    let mut output = String::new();
    for job in running {
        let script_preview = if job.info.script.len() > 50 {
            format!("{}...", &job.info.script[..50])
        } else {
            job.info.script.clone()
        };

        let output_path = job
            .info
            .output_path
            .as_ref()
            .map_or_else(|| "(no output)".to_string(), |p| p.display().to_string());

        output.push_str(&format!(
            "  - task {} (pid {}): `{}` -> {}\n",
            job.info.task_id, job.info.pid, script_preview, output_path
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
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_job_registry_lifecycle() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(
                12345,
                "amber-forest-thunder-pearl",
                "exec-test",
                "sleep 10",
                BashMode::Network,
                Some(PathBuf::from("outputs/12345.txt")),
            )
            .await;

        let jobs = registry.list().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].pid, 12345);
        assert_eq!(jobs[0].task_id, "amber-forest-thunder-pearl");
        assert_eq!(jobs[0].execution_key, "exec-test");
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
            .register(1, "task-a", "exec-a", "echo hello", BashMode::Network, None)
            .await;
        registry
            .register(2, "task-b", "exec-b", "sleep 5", BashMode::Network, None)
            .await;
        registry.complete(2, 0, None).await;

        let running = registry.format_running_jobs().await;
        assert!(running.contains("task task-a"));
        assert!(!running.contains("task task-b"));
    }

    #[tokio::test]
    async fn test_output_redirect_streams_full_output() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        let temp = tempdir().unwrap();
        let redirect_path = temp.path().join("task-stream.log");

        registry
            .register(
                101,
                "task-stream",
                "exec-stream",
                "printf one",
                BashMode::Network,
                None,
            )
            .await;

        registry
            .append_stdout(101, b"line1\nline2\n".to_vec())
            .await;
        let snapshot = registry
            .start_output_redirect("task-stream", redirect_path.clone())
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&snapshot), "line1\nline2\n");

        registry.append_stderr(101, b"line3\n".to_vec()).await;
        registry.complete(101, 0, None).await;
        let job = registry.get(101).await.unwrap();
        assert!(matches!(job.status, JobStatus::Completed { exit_code: 0 }));

        let redirected = async_fs::read(&redirect_path).await.unwrap();
        assert_eq!(
            String::from_utf8_lossy(&redirected),
            "line1\nline2\nline3\n"
        );
    }

    #[tokio::test]
    async fn test_kill_by_execution_key_no_match() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(1, "task-a", "exec-a", "echo hello", BashMode::Network, None)
            .await;
        registry.complete(1, 0, None).await;

        let killed = registry.kill_by_execution_key("exec-b").await;
        assert_eq!(killed, 0);
    }

    #[tokio::test]
    async fn test_kill_switch_path_marks_job_killed() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(
                555,
                "task-kill-switch",
                "exec-kill-switch",
                "sleep 10",
                BashMode::Network,
                None,
            )
            .await;
        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        registry.attach_kill_switch(555, kill_tx).await;

        assert!(registry.kill_by_task_id("task-kill-switch").await);
        assert!(kill_rx.recv().await.is_ok());
        let job = registry.get(555).await.expect("job should exist");
        assert!(matches!(job.status, JobStatus::Killed));
    }

    #[tokio::test]
    async fn test_complete_does_not_override_killed_status() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(
                556,
                "task-killed-complete",
                "exec-killed-complete",
                "sleep 10",
                BashMode::Network,
                None,
            )
            .await;
        let (kill_tx, _kill_rx) = async_channel::bounded::<()>(1);
        registry.attach_kill_switch(556, kill_tx).await;
        assert!(registry.kill_by_task_id("task-killed-complete").await);

        registry.complete(556, 0, None).await;
        let job = registry.get(556).await.expect("job should exist");
        assert!(matches!(job.status, JobStatus::Killed));
    }
}
