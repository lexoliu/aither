//! Background job registry for tracking and managing background terminal tasks.
//!
//! Uses an internal command channel to avoid shared locks.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use async_channel::{Receiver, Sender};
use futures_lite::io::AsyncWriteExt;

use crate::permission::TerminalMode;
use crate::process_tree::{JobMarker, JobTree, KillTreeError, ProcessScanner};

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
    pub mode: TerminalMode,
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

/// Incremental read result for a background terminal buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalDelta {
    /// Stable task identifier exposed to the model.
    pub task_id: String,
    /// Cursor the caller should pass on the next read.
    pub cursor: usize,
    /// Total bytes currently buffered for this task.
    pub total_bytes: usize,
    /// Newly read bytes from the terminal buffer.
    pub bytes: Vec<u8>,
    /// Current task status at the time of the read.
    pub status: JobStatus,
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
    process_tree: Option<JobTree>,
    /// How many kills of the job run outside the registry; an ending the
    /// job's own flow reports meanwhile waits in `deferred_end`.
    kills_in_flight: usize,
    deferred_end: Option<Ending>,
    output_redirect: Option<async_fs::File>,
    finished_at: Option<Instant>,
}

/// State mutations sent to the registry service; none expect a reply.
enum StateCommand {
    Register {
        pid: u32,
        task_id: String,
        execution_key: String,
        script: String,
        mode: TerminalMode,
        output_path: Option<PathBuf>,
        process_tree: Option<JobTree>,
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
}

/// Registry queries and job-control requests, each answered on `reply`.
enum JobRequest {
    List {
        reply: Sender<Vec<JobInfo>>,
    },
    Get {
        pid: u32,
        reply: Sender<Option<JobInfo>>,
    },
    GetByTaskId {
        task_id: String,
        reply: Sender<Option<JobInfo>>,
    },
    BeginKill {
        pid: u32,
        reply: Sender<KillPlan>,
    },
    BeginKillByTaskId {
        task_id: String,
        reply: Sender<Option<(u32, KillPlan)>>,
    },
    BeginKillByExecutionKey {
        execution_key: String,
        reply: Sender<Vec<(u32, KillPlan)>>,
    },
    EndKill {
        pid: u32,
        killed: bool,
        reply: Sender<bool>,
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
    ReadTerminalDelta {
        task_id: String,
        cursor: usize,
        max_bytes: usize,
        reply: Sender<Result<TerminalDelta, String>>,
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

/// How the registry's caller kills a job: the kill itself runs outside the
/// registry's state actor, which would otherwise stall every other job's
/// output, listing and kills meanwhile.
enum KillPlan {
    /// The registry decided at once.
    Done(bool),
    /// Kill the job's whole process tree.
    Tree(JobTree),
    /// Kill the process group led by this pid.
    Group(u32),
}

/// How a job's own flow ended it while a kill of it was running.
#[derive(Debug)]
enum Ending {
    Complete {
        exit_code: i32,
        output_path: Option<PathBuf>,
    },
    Fail {
        error: String,
        output_path: Option<PathBuf>,
    },
}

/// Message delivered to the registry service.
enum JobCommand {
    State(StateCommand),
    Request(JobRequest),
}

impl From<StateCommand> for JobCommand {
    fn from(command: StateCommand) -> Self {
        Self::State(command)
    }
}

impl From<JobRequest> for JobCommand {
    fn from(request: JobRequest) -> Self {
        Self::Request(request)
    }
}

/// Registry handle for tracking background jobs.
///
/// Cloning keeps a sender to the same registry service.
#[derive(Clone, Debug)]
pub struct JobRegistry {
    tx: Sender<JobCommand>,
    /// The tree scanner — kept for the other unixes and the tests; on Linux
    /// the supervisor owns each tree.
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    scanner: ProcessScanner,
}

/// Background service that owns the job registry state.
#[derive(Debug)]
pub struct JobRegistryService {
    rx: Receiver<JobCommand>,
}

/// Creates a registry handle and its background service.
#[must_use]
pub fn job_registry_channel() -> (JobRegistry, JobRegistryService) {
    let (tx, rx) = async_channel::unbounded();
    (
        JobRegistry {
            tx,
            scanner: ProcessScanner::default(),
        },
        JobRegistryService { rx },
    )
}

impl JobRegistry {
    async fn send_command(&self, command: impl Into<JobCommand>) {
        self.tx
            .send(command.into())
            .await
            .expect("job registry service unavailable");
    }

    async fn recv_reply<T>(reply_rx: Receiver<T>) -> T {
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Registers a new running job.
    pub async fn register(
        &self,
        pid: u32,
        task_id: &str,
        execution_key: &str,
        script: &str,
        mode: TerminalMode,
        output_path: Option<PathBuf>,
    ) {
        self.register_with_tree(pid, task_id, execution_key, script, mode, output_path, None)
            .await;
    }

    /// Registers a new running job whose process tree was started with
    /// [`Self::start_process_tree`]: the tree is part of the job from its
    /// first moment, so a kill never finds a job without it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn register_with_tree(
        &self,
        pid: u32,
        task_id: &str,
        execution_key: &str,
        script: &str,
        mode: TerminalMode,
        output_path: Option<PathBuf>,
        process_tree: Option<JobTree>,
    ) {
        self.send_command(StateCommand::Register {
            pid,
            task_id: task_id.to_string(),
            execution_key: execution_key.to_string(),
            script: script.to_string(),
            mode,
            output_path,
            process_tree,
        })
        .await;
    }

    /// Attaches a stdin input channel to a running job.
    pub async fn attach_terminal_input(&self, pid: u32, input_tx: TerminalInputSender) {
        self.send_command(StateCommand::AttachTerminalInput { pid, input_tx })
            .await;
    }

    /// Attaches a kill switch used by runtimes without host-visible PIDs.
    pub async fn attach_kill_switch(&self, pid: u32, kill_tx: Sender<()>) {
        self.send_command(StateCommand::AttachKillSwitch { pid, kill_tx })
            .await;
    }

    /// Starts watching the process tree of a local job whose root shell is
    /// `pid`. Called right after the spawn and before [`Self::register`]:
    /// the tree follows the root from its first moments, while its children's
    /// lineage can still be read. The caller keeps `pid` unreaped until it
    /// calls [`JobTree::finish`].
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    pub(crate) fn start_process_tree<E: executor_core::Executor>(
        &self,
        executor: &E,
        pid: u32,
        marker: &JobMarker,
    ) -> JobTree {
        JobTree::spawn(executor, self.scanner.clone(), pid, marker)
    }

    /// Appends stdout bytes for a running job.
    pub async fn append_stdout(&self, pid: u32, chunk: Vec<u8>) {
        self.send_command(StateCommand::AppendTerminalOutput {
            pid,
            stream: TerminalStream::Stdout,
            chunk,
        })
        .await;
    }

    /// Appends stderr bytes for a running job.
    pub async fn append_stderr(&self, pid: u32, chunk: Vec<u8>) {
        self.send_command(StateCommand::AppendTerminalOutput {
            pid,
            stream: TerminalStream::Stderr,
            chunk,
        })
        .await;
    }

    /// Marks stdout stream closed.
    pub async fn close_stdout(&self, pid: u32) {
        self.send_command(StateCommand::CloseTerminalStream {
            pid,
            stream: TerminalStream::Stdout,
        })
        .await;
    }

    /// Marks stderr stream closed.
    pub async fn close_stderr(&self, pid: u32) {
        self.send_command(StateCommand::CloseTerminalStream {
            pid,
            stream: TerminalStream::Stderr,
        })
        .await;
    }

    /// Sends bytes to a running task terminal stdin.
    ///
    /// # Errors
    /// Returns an error when the task does not exist or no stdin channel is attached.
    pub async fn terminal_input(&self, task_id: &str, bytes: Vec<u8>) -> Result<(), String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::InputTerminal {
            task_id: task_id.to_string(),
            bytes,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Starts continuous output redirection for a task and returns current snapshot bytes.
    ///
    /// # Errors
    /// Returns an error when the task does not exist or output cannot be redirected.
    pub async fn start_output_redirect(
        &self,
        task_id: &str,
        output_path: PathBuf,
    ) -> Result<Vec<u8>, String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::StartOutputRedirect {
            task_id: task_id.to_string(),
            output_path,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns true when both stdout/stderr streams are closed for the PID.
    pub async fn terminal_streams_closed(&self, pid: u32) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::TerminalStreamsClosed {
            pid,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns collected stdout/stderr buffers for a PID.
    pub async fn terminal_output(&self, pid: u32) -> Option<(Vec<u8>, Vec<u8>)> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::TerminalOutput {
            pid,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns terminal output added since the provided cursor.
    ///
    /// # Errors
    /// Returns an error when the task id is unknown.
    pub async fn terminal_read(
        &self,
        task_id: &str,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<TerminalDelta, String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::ReadTerminalDelta {
            task_id: task_id.to_string(),
            cursor,
            max_bytes,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Marks a job as completed.
    pub async fn complete(&self, pid: u32, exit_code: i32, output_path: Option<PathBuf>) {
        self.send_command(StateCommand::Complete {
            pid,
            exit_code,
            output_path,
        })
        .await;
    }

    /// Marks a job as failed.
    pub async fn fail(&self, pid: u32, error: &str, output_path: Option<PathBuf>) {
        self.send_command(StateCommand::Fail {
            pid,
            error: error.to_string(),
            output_path,
        })
        .await;
    }

    /// Lists all jobs.
    pub async fn list(&self) -> Vec<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::List { reply: reply_tx })
            .await;
        Self::recv_reply(reply_rx).await
    }

    /// Gets information about a specific job by PID.
    pub async fn get(&self, pid: u32) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::Get {
            pid,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Gets information about a specific job by its task ID.
    pub async fn get_by_task_id(&self, task_id: &str) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::GetByTaskId {
            task_id: task_id.to_string(),
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Kills a running job by its PID.
    pub async fn kill(&self, pid: u32) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKill {
            pid,
            reply: reply_tx,
        })
        .await;
        let plan = Self::recv_reply(reply_rx).await;
        self.finish_kill(pid, plan).await
    }

    /// Kills a running job by its task ID.
    pub async fn kill_by_task_id(&self, task_id: &str) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKillByTaskId {
            task_id: task_id.to_string(),
            reply: reply_tx,
        })
        .await;
        match Self::recv_reply(reply_rx).await {
            Some((pid, plan)) => self.finish_kill(pid, plan).await,
            None => false,
        }
    }

    /// Kills every job that is still running in this registry.
    pub async fn kill_running(&self) -> usize {
        let jobs = self.list().await;
        futures::future::join_all(
            jobs.into_iter()
                .filter(|job| job.status == JobStatus::Running)
                .map(|job| self.kill(job.pid)),
        )
        .await
        .into_iter()
        .filter(|killed| *killed)
        .count()
    }

    /// Kills all running jobs owned by an execution key.
    pub async fn kill_by_execution_key(&self, execution_key: &str) -> usize {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKillByExecutionKey {
            execution_key: execution_key.to_string(),
            reply: reply_tx,
        })
        .await;
        let plans = Self::recv_reply(reply_rx).await;
        futures::future::join_all(
            plans
                .into_iter()
                .map(|(pid, plan)| self.finish_kill(pid, plan)),
        )
        .await
        .into_iter()
        .filter(|killed| *killed)
        .count()
    }

    /// Runs the kill a [`KillPlan`] describes and records its outcome.
    async fn finish_kill(&self, pid: u32, plan: KillPlan) -> bool {
        let killed = match plan {
            KillPlan::Done(killed) => return killed,
            KillPlan::Tree(tree) => match tree.kill().await {
                Ok(()) => true,
                #[cfg(unix)]
                Err(KillTreeError::Ended) => {
                    tracing::debug!(pid = %pid, "the job ended before the kill reached it");
                    false
                }
                Err(error) => {
                    tracing::warn!(pid = %pid, error = %error, "failed to kill job");
                    false
                }
            },
            KillPlan::Group(group) => match crate::process_tree::kill_process_tree(group).await {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(pid = %pid, error = %error, "failed to kill job");
                    false
                }
            },
        };
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::EndKill {
            pid,
            killed,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Formats only RUNNING jobs for compression preservation.
    pub async fn format_running_jobs(&self) -> String {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::FormatRunning { reply: reply_tx })
            .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns true if there are any running jobs.
    pub async fn has_running(&self) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::HasRunning { reply: reply_tx })
            .await;
        Self::recv_reply(reply_rx).await
    }
}

impl JobRegistryService {
    /// Runs the registry service loop until all senders are dropped.
    pub async fn serve(self) {
        let mut jobs: HashMap<u32, JobState> = HashMap::new();

        while let Ok(command) = self.rx.recv().await {
            match command {
                JobCommand::State(command) => handle_state_command(&mut jobs, command).await,
                JobCommand::Request(request) => handle_job_request(&mut jobs, request).await,
            }
        }
    }
}

async fn handle_state_command(jobs: &mut HashMap<u32, JobState>, command: StateCommand) {
    match command {
        StateCommand::Register {
            pid,
            task_id,
            execution_key,
            script,
            mode,
            output_path,
            process_tree,
        } => {
            register_job(jobs, pid, task_id, execution_key, script, mode, output_path);
            if let Some(job) = jobs.get_mut(&pid) {
                job.process_tree = process_tree;
            }
        }
        StateCommand::AttachTerminalInput { pid, input_tx } => {
            if let Some(job) = jobs.get_mut(&pid) {
                job.input_tx = Some(input_tx);
            }
        }
        StateCommand::AttachKillSwitch { pid, kill_tx } => {
            if let Some(job) = jobs.get_mut(&pid) {
                job.kill_switch = Some(kill_tx);
            }
        }
        StateCommand::AppendTerminalOutput { pid, stream, chunk } => {
            append_terminal_output(jobs, pid, stream, chunk).await;
        }
        StateCommand::CloseTerminalStream { pid, stream } => {
            close_terminal_stream(jobs, pid, stream);
        }
        StateCommand::Complete {
            pid,
            exit_code,
            output_path,
        } => complete_job(jobs, pid, exit_code, output_path).await,
        StateCommand::Fail {
            pid,
            error,
            output_path,
        } => fail_job(jobs, pid, error, output_path).await,
    }
}

async fn handle_job_request(jobs: &mut HashMap<u32, JobState>, request: JobRequest) {
    match request {
        JobRequest::List { reply } => {
            let items = jobs.values().map(|state| state.info.clone()).collect();
            let _ = reply.send(items).await;
        }
        JobRequest::Get { pid, reply } => {
            let item = jobs.get(&pid).map(|state| state.info.clone());
            let _ = reply.send(item).await;
        }
        JobRequest::GetByTaskId { task_id, reply } => {
            let item = find_pid_by_task_id(jobs, &task_id)
                .and_then(|pid| jobs.get(&pid))
                .map(|state| state.info.clone());
            let _ = reply.send(item).await;
        }
        JobRequest::BeginKill { pid, reply } => {
            let plan = begin_kill(jobs, pid).await;
            let _ = reply.send(plan).await;
        }
        JobRequest::BeginKillByTaskId { task_id, reply } => {
            let plan = match find_pid_by_task_id(jobs, &task_id) {
                Some(pid) => Some((pid, begin_kill(jobs, pid).await)),
                None => None,
            };
            let _ = reply.send(plan).await;
        }
        JobRequest::BeginKillByExecutionKey {
            execution_key,
            reply,
        } => {
            let plans = begin_kill_by_execution_key(jobs, &execution_key).await;
            let _ = reply.send(plans).await;
        }
        JobRequest::EndKill { pid, killed, reply } => {
            let result = end_kill(jobs, pid, killed).await;
            let _ = reply.send(result).await;
        }
        JobRequest::InputTerminal {
            task_id,
            bytes,
            reply,
        } => {
            let result = input_terminal(jobs, &task_id, bytes).await;
            let _ = reply.send(result).await;
        }
        JobRequest::StartOutputRedirect {
            task_id,
            output_path,
            reply,
        } => {
            let result = redirect_task_output(jobs, &task_id, output_path).await;
            let _ = reply.send(result).await;
        }
        JobRequest::TerminalOutput { pid, reply } => {
            send_terminal_output(jobs, pid, reply).await;
        }
        JobRequest::ReadTerminalDelta {
            task_id,
            cursor,
            max_bytes,
            reply,
        } => {
            let result = read_terminal_delta(jobs, &task_id, cursor, max_bytes);
            let _ = reply.send(result).await;
        }
        JobRequest::TerminalStreamsClosed { pid, reply } => {
            let closed = jobs
                .get(&pid)
                .is_some_and(|job| job.stdout_closed && job.stderr_closed);
            let _ = reply.send(closed).await;
        }
        JobRequest::FormatRunning { reply } => {
            let _ = reply.send(format_running_jobs(jobs)).await;
        }
        JobRequest::HasRunning { reply } => {
            let _ = reply.send(has_running_job(jobs)).await;
        }
    }
}

async fn send_terminal_output(
    jobs: &HashMap<u32, JobState>,
    pid: u32,
    reply: Sender<Option<(Vec<u8>, Vec<u8>)>>,
) {
    let output = jobs
        .get(&pid)
        .map(|job| (job.stdout_buffer.clone(), job.stderr_buffer.clone()));
    let _ = reply.send(output).await;
}

fn has_running_job(jobs: &HashMap<u32, JobState>) -> bool {
    jobs.values()
        .any(|job| matches!(job.info.status, JobStatus::Running))
}

fn register_job(
    jobs: &mut HashMap<u32, JobState>,
    pid: u32,
    task_id: String,
    execution_key: String,
    script: String,
    mode: TerminalMode,
    output_path: Option<PathBuf>,
) {
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
    jobs.insert(pid, JobState::new(info));
    tracing::debug!(pid = %pid, "registered background job");
}

impl JobState {
    const fn new(info: JobInfo) -> Self {
        Self {
            info,
            terminal_buffer: Vec::new(),
            stdout_buffer: Vec::new(),
            stderr_buffer: Vec::new(),
            stdout_closed: false,
            stderr_closed: false,
            input_tx: None,
            kill_switch: None,
            process_tree: None,
            kills_in_flight: 0,
            deferred_end: None,
            output_redirect: None,
            finished_at: None,
        }
    }
}

async fn append_terminal_output(
    jobs: &mut HashMap<u32, JobState>,
    pid: u32,
    stream: TerminalStream,
    chunk: Vec<u8>,
) {
    if chunk.is_empty() {
        return;
    }
    if let Some(job) = jobs.get_mut(&pid) {
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

fn close_terminal_stream(jobs: &mut HashMap<u32, JobState>, pid: u32, stream: TerminalStream) {
    if let Some(job) = jobs.get_mut(&pid) {
        match stream {
            TerminalStream::Stdout => job.stdout_closed = true,
            TerminalStream::Stderr => job.stderr_closed = true,
        }
    }
}

async fn complete_job(
    jobs: &mut HashMap<u32, JobState>,
    pid: u32,
    exit_code: i32,
    output_path: Option<PathBuf>,
) {
    if let Some(job) = jobs.get_mut(&pid) {
        if !matches!(job.info.status, JobStatus::Running) {
            tracing::debug!(
                pid = %pid,
                status = ?job.info.status,
                "ignoring complete for non-running job"
            );
            return;
        }
        if job.kills_in_flight > 0 {
            job.deferred_end = Some(Ending::Complete {
                exit_code,
                output_path,
            });
            return;
        }
        if let Err(error) = finalize_output_redirect(job).await {
            tracing::error!(
                pid = %pid,
                error = %error,
                "failed to flush redirected output on complete"
            );
            mark_job_failed(job, error);
            return;
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

async fn fail_job(
    jobs: &mut HashMap<u32, JobState>,
    pid: u32,
    error: String,
    output_path: Option<PathBuf>,
) {
    if let Some(job) = jobs.get_mut(&pid) {
        if !matches!(job.info.status, JobStatus::Running) {
            tracing::debug!(
                pid = %pid,
                status = ?job.info.status,
                "ignoring fail for non-running job"
            );
            return;
        }
        if job.kills_in_flight > 0 {
            job.deferred_end = Some(Ending::Fail { error, output_path });
            return;
        }
        let status_error = if let Err(finalize_error) = finalize_output_redirect(job).await {
            format!("{error}; additionally failed to flush redirected output: {finalize_error}")
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

async fn begin_kill_by_execution_key(
    jobs: &mut HashMap<u32, JobState>,
    execution_key: &str,
) -> Vec<(u32, KillPlan)> {
    let pids: Vec<u32> = jobs
        .iter()
        .filter_map(|(pid, state)| {
            (state.info.execution_key == execution_key
                && matches!(state.info.status, JobStatus::Running))
            .then_some(*pid)
        })
        .collect();

    let mut plans = Vec::with_capacity(pids.len());
    for pid in pids {
        plans.push((pid, begin_kill(jobs, pid).await));
    }
    plans
}

async fn input_terminal(
    jobs: &HashMap<u32, JobState>,
    task_id: &str,
    bytes: Vec<u8>,
) -> Result<(), String> {
    let Some(pid) = find_pid_by_task_id(jobs, task_id) else {
        return Err(format!("unknown task_id: {task_id}"));
    };
    let Some(job) = jobs.get(&pid) else {
        return Err(format!("unknown task_id: {task_id}"));
    };
    if !matches!(job.info.status, JobStatus::Running) {
        return Err(format!(
            "task {} is not running; current status: {:?}",
            task_id, job.info.status
        ));
    }
    let Some(tx) = &job.input_tx else {
        return Err(format!(
            "task {task_id} does not support terminal input in this runtime"
        ));
    };
    tx.send(bytes)
        .await
        .map_err(|_| format!("terminal input channel closed for task {task_id}"))
}

async fn redirect_task_output(
    jobs: &mut HashMap<u32, JobState>,
    task_id: &str,
    output_path: PathBuf,
) -> Result<Vec<u8>, String> {
    let Some(pid) = find_pid_by_task_id(jobs, task_id) else {
        return Err(format!("unknown task_id: {task_id}"));
    };
    let Some(job) = jobs.get_mut(&pid) else {
        return Err(format!("unknown task_id: {task_id}"));
    };
    start_output_redirect(job, output_path).await
}

fn read_terminal_delta(
    jobs: &HashMap<u32, JobState>,
    task_id: &str,
    cursor: usize,
    max_bytes: usize,
) -> Result<TerminalDelta, String> {
    find_pid_by_task_id(jobs, task_id).map_or_else(
        || Err(format!("unknown task_id: {task_id}")),
        |pid| {
            jobs.get(&pid).map_or_else(
                || Err(format!("unknown task_id: {task_id}")),
                |job| terminal_read(job, cursor, max_bytes),
            )
        },
    )
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

/// Starts killing a job: decides at once what needs no process signal, and
/// otherwise hands back what the caller must kill outside the registry.
async fn begin_kill(jobs: &mut HashMap<u32, JobState>, pid: u32) -> KillPlan {
    let Some(job) = jobs.get_mut(&pid) else {
        return KillPlan::Done(false);
    };
    if !matches!(job.info.status, JobStatus::Running) {
        tracing::warn!(pid = %pid, "cannot kill non-running job");
        return KillPlan::Done(false);
    }

    if let Some(kill_switch) = job.kill_switch.take() {
        if kill_switch.send(()).await.is_err() {
            tracing::warn!(pid = %pid, "failed to signal runtime kill switch");
            return KillPlan::Done(false);
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
        return KillPlan::Done(true);
    }

    job.kills_in_flight += 1;
    job.process_tree
        .as_ref()
        .map_or(KillPlan::Group(pid), |tree| KillPlan::Tree(tree.clone()))
}

/// Records the outcome of a kill that ran outside the registry. A job whose
/// own flow ended it meanwhile takes that ending only when the kill failed.
async fn end_kill(jobs: &mut HashMap<u32, JobState>, pid: u32, killed: bool) -> bool {
    let Some(job) = jobs.get_mut(&pid) else {
        return false;
    };
    job.kills_in_flight = job.kills_in_flight.saturating_sub(1);
    if killed {
        job.deferred_end = None;
        if matches!(job.info.status, JobStatus::Running) {
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
        }
        return true;
    }
    // While another kill of the job still runs, `complete_job` and `fail_job`
    // defer the ending again.
    match job.deferred_end.take() {
        Some(Ending::Complete {
            exit_code,
            output_path,
        }) => complete_job(jobs, pid, exit_code, output_path).await,
        Some(Ending::Fail { error, output_path }) => {
            fail_job(jobs, pid, error, output_path).await;
        }
        None => {}
    }
    false
}

/// The bytes of a script a running-jobs line shows.
const SCRIPT_PREVIEW_BYTES: usize = 50;

/// The start of `script`, cut at a character boundary at or below
/// [`SCRIPT_PREVIEW_BYTES`] bytes.
fn script_preview(script: &str) -> String {
    if script.len() <= SCRIPT_PREVIEW_BYTES {
        return script.to_string();
    }
    let end = (0..=SCRIPT_PREVIEW_BYTES)
        .rev()
        .find(|index| script.is_char_boundary(*index))
        .unwrap_or(0);
    format!("{}...", &script[..end])
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
        let script_preview = script_preview(&job.info.script);

        let output_path = job
            .info
            .output_path
            .as_ref()
            .map_or_else(|| "(no output)".to_string(), |p| p.display().to_string());

        let _ = writeln!(
            output,
            "  - task {} (pid {}): `{}` -> {}",
            job.info.task_id, job.info.pid, script_preview, output_path
        );
    }

    output
}

fn terminal_read(job: &JobState, cursor: usize, max_bytes: usize) -> Result<TerminalDelta, String> {
    let total_bytes = job.terminal_buffer.len();
    if cursor > total_bytes {
        return Err(format!(
            "cursor {cursor} exceeds terminal buffer size {total_bytes} for task {}",
            job.info.task_id
        ));
    }
    let end = total_bytes.min(cursor.saturating_add(max_bytes));
    Ok(TerminalDelta {
        task_id: job.info.task_id.clone(),
        cursor: end,
        total_bytes,
        bytes: job.terminal_buffer[cursor..end].to_vec(),
        status: job.info.status.clone(),
    })
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
                TerminalMode::Sandboxed,
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
            .register(
                1,
                "task-a",
                "exec-a",
                "echo hello",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        registry
            .register(
                2,
                "task-b",
                "exec-b",
                "sleep 5",
                TerminalMode::Sandboxed,
                None,
            )
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
                TerminalMode::Sandboxed,
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
    async fn test_read_terminal_delta_reads_incrementally() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(
                202,
                "task-delta",
                "exec-delta",
                "tail -f log",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        registry.append_stdout(202, b"hello ".to_vec()).await;
        registry.append_stderr(202, b"world".to_vec()).await;

        let first = registry.terminal_read("task-delta", 0, 6).await.unwrap();
        assert_eq!(String::from_utf8_lossy(&first.bytes), "hello ");
        assert_eq!(first.cursor, 6);
        assert_eq!(first.total_bytes, 11);
        assert!(matches!(first.status, JobStatus::Running));

        let second = registry
            .terminal_read("task-delta", first.cursor, 32)
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&second.bytes), "world");
        assert_eq!(second.cursor, 11);
        assert_eq!(second.total_bytes, 11);
    }

    #[tokio::test]
    async fn test_kill_by_execution_key_no_match() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(
                1,
                "task-a",
                "exec-a",
                "echo hello",
                TerminalMode::Sandboxed,
                None,
            )
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
                TerminalMode::Sandboxed,
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
                TerminalMode::Sandboxed,
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

    /// A script longer than the preview, with a multibyte character across
    /// the cut, is listed instead of panicking inside the registry.
    #[tokio::test]
    async fn test_format_running_jobs_cuts_a_multibyte_script_at_a_boundary() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        // Three bytes per character: byte 50 lies inside the 17th.
        let script = "日".repeat(30);
        assert!(!script.is_char_boundary(SCRIPT_PREVIEW_BYTES));
        registry
            .register(
                1,
                "task-cjk",
                "exec-cjk",
                &script,
                TerminalMode::Sandboxed,
                None,
            )
            .await;

        let running = registry.format_running_jobs().await;
        assert!(
            running.contains(&format!("`{}...`", "日".repeat(16))),
            "{running}"
        );
        assert_eq!(script_preview("short"), "short");
        assert_eq!(script_preview(&"x".repeat(50)), "x".repeat(50));
        assert_eq!(
            script_preview(&"x".repeat(51)),
            format!("{}...", "x".repeat(50))
        );
    }

    fn running_job(pid: u32) -> HashMap<u32, JobState> {
        let mut jobs = HashMap::new();
        register_job(
            &mut jobs,
            pid,
            "task".to_string(),
            "exec".to_string(),
            "true".to_string(),
            TerminalMode::Sandboxed,
            None,
        );
        jobs
    }

    /// Two kills of one job overlap: the first to end must not release an
    /// ending the job's own flow reported while the second still runs.
    #[tokio::test]
    async fn overlapping_kills_hold_the_deferred_ending_until_the_last_ends() {
        let mut jobs = running_job(7);
        assert!(matches!(begin_kill(&mut jobs, 7).await, KillPlan::Group(7)));
        assert!(matches!(begin_kill(&mut jobs, 7).await, KillPlan::Group(7)));
        complete_job(&mut jobs, 7, 0, None).await;
        assert_eq!(jobs[&7].info.status, JobStatus::Running, "the ending waits");

        assert!(!end_kill(&mut jobs, 7, false).await);
        assert_eq!(
            jobs[&7].info.status,
            JobStatus::Running,
            "the other kill still runs"
        );
        assert!(!end_kill(&mut jobs, 7, false).await);
        assert_eq!(
            jobs[&7].info.status,
            JobStatus::Completed { exit_code: 0 },
            "the last kill to fail releases the ending"
        );
    }

    /// A kill that succeeds wins over an ending reported while it ran.
    #[tokio::test]
    async fn a_successful_kill_wins_over_a_deferred_ending() {
        let mut jobs = running_job(8);
        assert!(matches!(begin_kill(&mut jobs, 8).await, KillPlan::Group(8)));
        complete_job(&mut jobs, 8, 0, None).await;
        assert!(end_kill(&mut jobs, 8, true).await);
        assert_eq!(jobs[&8].info.status, JobStatus::Killed);
    }
}
