//! Background job registry for tracking and managing background terminal tasks.
//!
//! Uses an internal command channel to avoid shared locks.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use aither_core::{IdLease, WeakIdLease};
use async_channel::{Receiver, Sender, WeakSender};
use futures_lite::io::AsyncWriteExt;

use crate::memory::MemoryDomain;
use crate::permission::TerminalMode;
use crate::process_tree::{JobMarker, JobTree, KillTreeError, ProcessScanner};
use crate::script_language::ScriptLanguage;

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
    /// The language of `script`.
    pub language: ScriptLanguage,
    /// The permission mode used.
    pub mode: TerminalMode,
    /// When the job was started.
    pub started_at: Instant,
    /// Current status of the job.
    pub status: JobStatus,
    /// Path to output file.
    pub output_path: Option<PathBuf>,
    /// Where the job runs.
    pub place: JobPlace,
}

/// Status of a background job.
///
/// Every finished status carries its [`Delivery`]: the transition that
/// finishes a job also records that its completion is owed to the agent.
/// The registry forgets a job once its completion was taken and nothing
/// watches it, so a [`Delivery::Taken`] job reads only while it is watched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Job is currently running.
    Running,
    /// Job completed successfully.
    Completed {
        /// Exit code of the process.
        exit_code: i32,
        /// Whether the agent has taken the job's completion.
        delivery: Delivery,
    },
    /// Job failed with an error.
    Failed {
        /// Error message.
        error: String,
        /// Whether the agent has taken the job's completion.
        delivery: Delivery,
    },
    /// Job was killed.
    Killed {
        /// Why it was killed.
        cause: KillCause,
        /// Whether the agent has taken the job's completion.
        delivery: Delivery,
    },
}

/// Why a job was killed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillCause {
    /// A request ended it: `terminal_kill`, its `timeout` bound, or the
    /// shutdown of its terminal.
    User,
    /// Its home environment's memory ran out of room for the agent's jobs
    /// (see [`MemoryDomain`]), and the agent left it running past the
    /// response window that followed the final warning.
    MemoryRunaway {
        /// The tree's physical footprint when it was killed.
        footprint_bytes: u64,
        /// The memory of the domain it ran in: the machine's physical memory,
        /// or the container's limit.
        capacity_bytes: u64,
        /// The domain it ran in.
        domain: MemoryDomain,
    },
}

/// Where a job runs, which decides what is watched of it.
///
/// Only the home environment is watched: the machine the embedder runs on,
/// and the container the agent works in when it has one. The memory and the
/// disk of a remote execution environment are never judged and its jobs are
/// never killed for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobPlace {
    /// A process of the machine the embedder runs on.
    Host,
    /// A process inside the container the agent works in.
    Container,
    /// A process on a remote host over SSH.
    Remote,
}

impl JobStatus {
    /// How far a finished job's completion got; `None` while it runs.
    #[must_use]
    pub const fn delivery(&self) -> Option<Delivery> {
        match self {
            Self::Running => None,
            Self::Completed { delivery, .. }
            | Self::Failed { delivery, .. }
            | Self::Killed { delivery, .. } => Some(*delivery),
        }
    }

    /// Why a killed job was killed; `None` for a job that was not.
    #[must_use]
    pub const fn kill_cause(&self) -> Option<KillCause> {
        match self {
            Self::Killed { cause, .. } => Some(*cause),
            Self::Running | Self::Completed { .. } | Self::Failed { .. } => None,
        }
    }

    /// Whether the job finished and its agent has yet to take the
    /// completion: the agent still owes its session a turn for it.
    #[must_use]
    pub const fn completion_pending(&self) -> bool {
        matches!(self.delivery(), Some(Delivery::Pending))
    }
}

/// Whether a finished job's completion has reached its agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// The completion is on its way to the agent, which has not taken it.
    Pending,
    /// The agent took the completion: its [`PendingCompletion`] was dropped.
    Taken,
}

/// A finished job's completion on its way to the agent.
///
/// The job reads [`Delivery::Pending`] in its registry for as long as this
/// lives; dropping it is the agent taking the completion, which marks the
/// job [`Delivery::Taken`] and lets the registry forget it, so its task id
/// goes back to the allocator. The registry hands out one per job, through
/// [`JobRegistry::owe_completion`].
#[must_use = "dropping it marks the completion taken"]
pub struct PendingCompletion {
    tx: Sender<JobCommand>,
    /// Keeps the word naming this job until the registry hears of the take.
    task_id: Option<IdLease>,
}

impl std::fmt::Debug for PendingCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingCompletion")
            .field("task_id", &self.task_id)
            .finish_non_exhaustive()
    }
}

impl PendingCompletion {
    /// Takes the completion as dropping it does, returning once the registry
    /// recorded the take: a job it then forgets reads as unknown to every
    /// later request.
    pub async fn take(mut self) {
        // Only `take` and `drop` empty it, each consuming the completion.
        let Some(task_id) = self.task_id.take() else {
            return;
        };
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        // A closed registry has no reader left to tell.
        if self
            .tx
            .send(
                JobRequest::TakeCompletion {
                    task_id,
                    reply: reply_tx,
                }
                .into(),
            )
            .await
            .is_ok()
        {
            let _ = reply_rx.recv().await;
        }
    }
}

impl Drop for PendingCompletion {
    fn drop(&mut self) {
        if let Some(task_id) = self.task_id.take() {
            // A closed registry has no reader left to tell.
            let _ = self
                .tx
                .try_send(StateCommand::TakeCompletion { task_id }.into());
        }
    }
}

/// Keeps a job in its registry, finished or not, while something reads it
/// by its task id; the registry forgets a finished job only once every
/// watch of it dropped. [`JobRegistry::watch`] hands them out.
pub(crate) struct JobWatch {
    tx: Sender<JobCommand>,
    /// Keeps the word naming this job; the unwatch it sends keeps a clone
    /// until the registry hears of it.
    task_id: IdLease,
}

impl std::fmt::Debug for JobWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobWatch")
            .field("task_id", &self.task_id)
            .finish_non_exhaustive()
    }
}

impl Drop for JobWatch {
    fn drop(&mut self) {
        let task_id = self.task_id.clone();
        let _ = self.tx.try_send(StateCommand::Unwatch { task_id }.into());
    }
}

/// What [`JobRegistry::owe_completion`] settled for a job.
#[derive(Debug)]
pub struct OwedCompletion {
    /// Why the job was killed; `None` when it ended on its own.
    pub kill: Option<KillCause>,
    /// Keeps the completion pending until the agent takes it.
    pub pending: PendingCompletion,
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

/// A `terminal_read` refused because it would return what the task's last
/// read returned: only more output, or the task's end, can tell the model
/// something new.
///
/// [`SandboxNotices::unchanged_task_read`](crate::SandboxNotices::unchanged_task_read)
/// renders its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnchangedTaskRead {
    /// The background task whose output did not change.
    pub task_id: String,
}

/// Why [`JobRegistry::terminal_read`] returned no delta.
#[derive(Debug, thiserror::Error)]
pub enum TerminalReadError {
    /// The read would return what the task's last read returned; the
    /// caller renders it through its
    /// [`SandboxNotices`](crate::SandboxNotices).
    #[error("refused as a poll: {0:?}")]
    Refused(UnchangedTaskRead),
    /// The read could not be served, such as an unknown task id.
    #[error("{0}")]
    Failed(String),
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
    /// Keeps the job's task id leased while the registry keeps the job.
    task_id: IdLease,
    terminal_buffer: Vec<u8>,
    /// Complete lines only, each from one stream, in the order they
    /// completed: what the `monitor` pump reads.
    line_buffer: Vec<u8>,
    stdout_partial: Vec<u8>,
    stderr_partial: Vec<u8>,
    stdout_buffer: Vec<u8>,
    stderr_buffer: Vec<u8>,
    stdout_closed: bool,
    stderr_closed: bool,
    input_tx: Option<TerminalInputSender>,
    kill_switch: Option<Sender<()>>,
    process_tree: Option<JobTree>,
    /// Why the kill in flight was asked for: the first of overlapping kills.
    kill_cause: Option<KillCause>,
    /// How many kills of the job run outside the registry; an ending the
    /// job's own flow reports meanwhile waits in `deferred_end`.
    kills_in_flight: usize,
    deferred_end: Option<Ending>,
    /// Callers parked until the last kill in flight records its outcome.
    settled_waiters: Vec<SettledWaiter>,
    output_redirect: Option<async_fs::File>,
    finished_at: Option<Instant>,
    /// Whether the job's [`PendingCompletion`] was handed out.
    completion_owed: bool,
    /// How many [`JobWatch`]es of the job live.
    watchers: usize,
}

/// A caller waiting for a job's kills in flight to record their outcome.
#[derive(Debug)]
enum SettledWaiter {
    /// `get_by_task_id_settled`.
    Read(Sender<Option<JobInfo>>),
    /// `owe_completion`.
    Owe {
        ending: Result<i32, String>,
        reply: Sender<Option<OwedCompletion>>,
    },
}

/// State mutations sent to the registry service; none expect a reply.
enum StateCommand {
    Register {
        pid: u32,
        task_id: IdLease,
        execution_key: String,
        script: String,
        language: ScriptLanguage,
        mode: TerminalMode,
        output_path: Option<PathBuf>,
        process_tree: Option<JobTree>,
        place: JobPlace,
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
    /// The agent took the completion of task `task_id`.
    TakeCompletion {
        task_id: IdLease,
    },
    /// A [`JobWatch`] of task `task_id` dropped.
    Unwatch {
        task_id: IdLease,
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
    GetByTaskIdSettled {
        task_id: String,
        reply: Sender<Option<JobInfo>>,
    },
    /// Appends `chunk` to the stdout of task `task_id`, replying once it is
    /// in; a task not registered yet gets it when it registers.
    AppendTaskStdout {
        task_id: IdLease,
        chunk: Vec<u8>,
        reply: Sender<()>,
    },
    OweCompletion {
        task_id: String,
        ending: Result<i32, String>,
        reply: Sender<Option<OwedCompletion>>,
    },
    BeginKill {
        pid: u32,
        cause: KillCause,
        reply: Sender<KillPlan>,
    },
    BeginKillByTaskId {
        task_id: String,
        cause: KillCause,
        reply: Sender<Option<(u32, KillPlan)>>,
    },
    BeginKillByExecutionKey {
        execution_key: String,
        cause: KillCause,
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
        /// Whether this read is a model read of the merged terminal buffer,
        /// tracked by the unchanged-read refusal; a monitor pump's reads are
        /// not, and read the complete-line buffer instead.
        tracked: bool,
        reply: Sender<Result<TerminalDelta, TerminalReadError>>,
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
    Watch {
        task_id: String,
        reply: Sender<Option<JobWatch>>,
    },
    /// [`PendingCompletion::take`].
    TakeCompletion {
        task_id: IdLease,
        reply: Sender<()>,
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
    /// What each [`PendingCompletion`] reports back on; weak, so the service
    /// still ends with the last handle.
    tx: WeakSender<JobCommand>,
}

/// Creates a registry handle and its background service.
#[must_use]
pub fn job_registry_channel() -> (JobRegistry, JobRegistryService) {
    let (tx, rx) = async_channel::unbounded();
    let weak = tx.downgrade();
    (
        JobRegistry {
            tx,
            scanner: ProcessScanner::default(),
        },
        JobRegistryService { rx, tx: weak },
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

    /// Registers a new running job under `task_id`, the word its model
    /// addresses it by.
    ///
    /// The registry holds the lease for as long as it keeps the job: until
    /// the job finished, its agent took the completion
    /// ([`Self::owe_completion`]) and nothing watches it, until a job with the
    /// same pid replaces it, or until the registry drops.
    pub async fn register(
        &self,
        pid: u32,
        task_id: &IdLease,
        execution_key: &str,
        script: &str,
        mode: TerminalMode,
        output_path: Option<PathBuf>,
    ) {
        self.register_with_tree(
            pid,
            task_id,
            execution_key,
            script,
            ScriptLanguage::Shell,
            mode,
            output_path,
            None,
            JobPlace::Host,
        )
        .await;
    }

    /// Registers a new running job like [`Self::register`], for a job that
    /// runs at `place` instead of on this machine, with `script` in
    /// `language`.
    #[allow(clippy::too_many_arguments)]
    pub async fn register_at(
        &self,
        place: JobPlace,
        pid: u32,
        task_id: &IdLease,
        execution_key: &str,
        script: &str,
        language: ScriptLanguage,
        mode: TerminalMode,
    ) {
        self.register_with_tree(
            pid,
            task_id,
            execution_key,
            script,
            language,
            mode,
            None,
            None,
            place,
        )
        .await;
    }

    /// Registers a new running job whose process tree was started with
    /// [`Self::start_process_tree`]: the tree is part of the job from its
    /// first moment, so a kill never finds a job without it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn register_with_tree(
        &self,
        pid: u32,
        task_id: &IdLease,
        execution_key: &str,
        script: &str,
        language: ScriptLanguage,
        mode: TerminalMode,
        output_path: Option<PathBuf>,
        process_tree: Option<JobTree>,
        place: JobPlace,
    ) {
        self.send_command(StateCommand::Register {
            place,
            pid,
            task_id: task_id.clone(),
            execution_key: execution_key.to_string(),
            script: script.to_string(),
            language,
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

    /// Appends stdout bytes to task `task_id`, returning once they are in its
    /// output. A task whose job is not registered yet (its process may already
    /// be running a command) gets them as it registers, ahead of anything the
    /// process wrote later; one the registry already forgot drops them, since
    /// nothing can read them.
    ///
    /// `task_id` is the job's own lease, which the bytes hold until they are
    /// in: its word cannot name a later job while they are on their way.
    pub async fn append_task_stdout(&self, task_id: &IdLease, chunk: Vec<u8>) {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::AppendTaskStdout {
            task_id: task_id.clone(),
            chunk,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await;
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
    /// [`TerminalReadError::Refused`] when the read would return what the
    /// task's last read returned; [`TerminalReadError::Failed`] when the
    /// task id is unknown.
    pub async fn terminal_read(
        &self,
        task_id: &str,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<TerminalDelta, TerminalReadError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::ReadTerminalDelta {
            task_id: task_id.to_string(),
            cursor,
            max_bytes,
            tracked: true,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns terminal output added since the provided cursor, without
    /// touching the unchanged-read tracker: the caller is the `monitor`
    /// pump, whose own cadence is not the model polling.
    ///
    /// # Errors
    /// Returns an error when the task id is unknown.
    pub async fn monitor_read(
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
            tracked: false,
            reply: reply_tx,
        })
        .await;
        match Self::recv_reply(reply_rx).await {
            Ok(delta) => Ok(delta),
            Err(TerminalReadError::Failed(error)) => Err(error),
            Err(TerminalReadError::Refused(refusal)) => {
                unreachable!("an untracked read is never refused: {refusal:?}")
            }
        }
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

    /// Gets information about a job by its task ID once no kill of it is in
    /// flight. A job whose process has exited from a kill's signal still reads
    /// `Running` until the kill records its outcome; this waits for that
    /// outcome, so the status it returns is the job's real ending.
    pub async fn get_by_task_id_settled(&self, task_id: &str) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::GetByTaskIdSettled {
            task_id: task_id.to_string(),
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Settles how task `task_id` ended and hands out the completion it owes
    /// its agent; its flow calls this once, after the job's process is done,
    /// and sends the returned [`PendingCompletion`] with the completed task.
    ///
    /// Waits for any kill of the job in flight, as
    /// [`Self::get_by_task_id_settled`] does. A job the registry still reads
    /// as running ends with `ending` — `Ok(exit_code)` or `Err(error)` — so
    /// no completion leaves a job running. `None` for a task the registry
    /// never registered.
    ///
    /// # Panics
    ///
    /// The registry panics when the job's completion was already handed out.
    pub async fn owe_completion(
        &self,
        task_id: &str,
        ending: Result<i32, String>,
    ) -> Option<OwedCompletion> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::OweCompletion {
            task_id: task_id.to_string(),
            ending,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Kills a running job by its PID.
    pub async fn kill(&self, pid: u32) -> bool {
        self.kill_for(pid, KillCause::User).await
    }

    /// Kills a running job whose process tree holds `footprint_bytes` of a
    /// `domain` with `capacity_bytes`; the job reads
    /// [`KillCause::MemoryRunaway`].
    pub async fn kill_runaway(
        &self,
        pid: u32,
        footprint_bytes: u64,
        capacity_bytes: u64,
        domain: MemoryDomain,
    ) -> bool {
        self.kill_for(
            pid,
            KillCause::MemoryRunaway {
                footprint_bytes,
                capacity_bytes,
                domain,
            },
        )
        .await
    }

    async fn kill_for(&self, pid: u32, cause: KillCause) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKill {
            pid,
            cause,
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
            cause: KillCause::User,
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
            cause: KillCause::User,
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

    /// Keeps task `task_id` in the registry, and its word naming it, until
    /// the returned watch drops; `None` for a task the registry does not
    /// hold.
    pub(crate) async fn watch(&self, task_id: &str) -> Option<JobWatch> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::Watch {
            task_id: task_id.to_string(),
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }
}

/// Stdout appends for tasks the registry holds no job for: those whose job is
/// not registered yet wait for it, those whose job it already forgot are
/// dropped.
///
/// Both refer to their job's lease weakly. The sender of an append holds the
/// lease until it is answered, so while one waits its word names that job
/// alone; once the lease ends, what waited for it goes with it, and a later
/// job leasing the same word never receives it.
#[derive(Default)]
struct ParkedOutput {
    /// Appends waiting for their job to register, by its task id.
    by_task: HashMap<String, Parked>,
    /// Forgotten jobs whose lease something still holds, such as a command
    /// that may still report progress.
    forgotten: Vec<WeakIdLease>,
}

/// The appends waiting for one job, in arrival order.
struct Parked {
    lease: WeakIdLease,
    appends: Vec<ParkedAppend>,
}

/// One parked append: its bytes and the reply that releases its sender.
type ParkedAppend = (Vec<u8>, Sender<()>);

impl ParkedOutput {
    /// Parks an append to task `task_id`, whose job is not registered. Hands
    /// back the reply when the registry already forgot that job, for the
    /// sender to be released without its bytes going anywhere.
    fn park(&mut self, task_id: &IdLease, chunk: Vec<u8>, reply: Sender<()>) -> Option<Sender<()>> {
        self.prune();
        if self
            .forgotten
            .iter()
            .any(|forgotten| forgotten.refers_to(task_id))
        {
            return Some(reply);
        }
        let parked = self
            .by_task
            .entry(task_id.to_string())
            .or_insert_with(|| Parked {
                lease: task_id.downgrade(),
                appends: Vec::new(),
            });
        assert!(
            parked.lease.refers_to(task_id),
            "two live leases share the task id '{task_id}'"
        );
        parked.appends.push((chunk, reply));
        None
    }

    /// Takes the appends waiting for task `task_id`, which is registering, in
    /// arrival order.
    fn take(&mut self, task_id: &IdLease) -> Vec<ParkedAppend> {
        self.prune();
        self.by_task
            .remove(task_id.as_str())
            .map(|parked| {
                assert!(
                    parked.lease.refers_to(task_id),
                    "two live leases share the task id '{task_id}'"
                );
                parked.appends
            })
            .unwrap_or_default()
    }

    /// Records that the registry forgot the job of task `task_id`, so what is
    /// still sent to it is dropped instead of waiting for it.
    fn forgot(&mut self, task_id: &IdLease) {
        self.prune();
        self.forgotten.push(task_id.downgrade());
    }

    /// Drops what refers to a lease that ended: no sender of an append to it
    /// is left, and its word may name another job.
    fn prune(&mut self) {
        self.by_task
            .retain(|_, parked| parked.lease.upgrade().is_some());
        self.forgotten
            .retain(|forgotten| forgotten.upgrade().is_some());
    }
}

/// Per-session read tracking behind the unchanged-read refusal: what each
/// task's last `terminal_read` returned with.
#[derive(Default)]
struct ReadTracker {
    /// Task id → the `(cursor, total_bytes, status)` its last read returned.
    delta_reads: HashMap<String, (usize, usize, JobStatus)>,
}

impl JobRegistryService {
    /// Runs the registry service loop until all senders are dropped.
    pub async fn serve(self) {
        let mut jobs: HashMap<u32, JobState> = HashMap::new();
        let mut reads = ReadTracker::default();
        let mut parked = ParkedOutput::default();

        while let Ok(command) = self.rx.recv().await {
            match command {
                JobCommand::State(command) => {
                    handle_state_command(&mut jobs, &mut reads, &mut parked, command).await;
                }
                JobCommand::Request(request) => {
                    handle_job_request(&mut jobs, &mut reads, &mut parked, &self.tx, request).await;
                }
            }
        }
    }
}

async fn handle_state_command(
    jobs: &mut HashMap<u32, JobState>,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    command: StateCommand,
) {
    match command {
        StateCommand::Register {
            pid,
            task_id,
            execution_key,
            script,
            language,
            mode,
            output_path,
            process_tree,
            place,
        } => {
            // A job this one replaces under the same pid is forgotten with it.
            if let Some(replaced) = jobs.get(&pid) {
                parked.forgot(&replaced.task_id);
            }
            let waiting = parked.take(&task_id);
            register_job(
                jobs,
                pid,
                task_id,
                execution_key,
                script,
                language,
                mode,
                output_path,
                place,
            );
            if let Some(job) = jobs.get_mut(&pid) {
                job.process_tree = process_tree;
            }
            for (chunk, reply) in waiting {
                append_terminal_output(jobs, pid, TerminalStream::Stdout, chunk).await;
                let _ = reply.send(()).await;
            }
        }
        StateCommand::AttachTerminalInput { pid, input_tx } => {
            // A job that already finished takes no input: the sender drops here.
            if let Some(job) = jobs
                .get_mut(&pid)
                .filter(|job| matches!(job.info.status, JobStatus::Running))
            {
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
        // The command's lease kept the word from naming another job, so the
        // job found by it is the one the command is about. A job whose pid a
        // later job reused left the registry with it; nothing reads it.
        StateCommand::TakeCompletion { task_id } => {
            take_completion(jobs, reads, parked, &task_id);
        }
        StateCommand::Unwatch { task_id } => {
            if let Some(pid) = find_pid_by_task_id(jobs, &task_id) {
                let job = jobs.get_mut(&pid).expect("the pid was just found");
                job.watchers -= 1;
                forget_if_done(jobs, reads, parked, pid);
            }
        }
    }
}

/// Forgets job `pid` once nothing can address it any more: it finished, its
/// agent took the completion and nothing watches it. Its task id then goes
/// back to the allocator once nothing else holds it, a later read of it
/// answers as for any unknown task, and progress still reported to it is
/// dropped.
fn forget_if_done(
    jobs: &mut HashMap<u32, JobState>,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    pid: u32,
) {
    let job = &jobs[&pid];
    if job.info.status.delivery() == Some(Delivery::Taken) && job.watchers == 0 {
        let job = jobs.remove(&pid).expect("the job is registered");
        reads.delta_reads.remove(job.task_id.as_str());
        parked.forgot(&job.task_id);
        tracing::debug!(pid = %pid, task_id = %job.task_id, "forgot finished job");
    }
}

#[allow(clippy::too_many_lines)]
async fn handle_job_request(
    jobs: &mut HashMap<u32, JobState>,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    tx: &WeakSender<JobCommand>,
    request: JobRequest,
) {
    match request {
        JobRequest::AppendTaskStdout {
            task_id,
            chunk,
            reply,
        } => {
            if let Some(pid) = find_pid_by_task_id(jobs, &task_id) {
                assert!(
                    jobs[&pid].task_id.same_lease(&task_id),
                    "two live leases share the task id '{task_id}'"
                );
                append_terminal_output(jobs, pid, TerminalStream::Stdout, chunk).await;
                let _ = reply.send(()).await;
            } else if let Some(reply) = parked.park(&task_id, chunk, reply) {
                let _ = reply.send(()).await;
            }
        }
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
        JobRequest::GetByTaskIdSettled { task_id, reply } => {
            get_settled_by_task_id(jobs, &task_id, reply).await;
        }
        JobRequest::OweCompletion {
            task_id,
            ending,
            reply,
        } => owe_settled_completion(jobs, &task_id, ending, reply, tx).await,
        JobRequest::BeginKill { pid, cause, reply } => {
            let plan = begin_kill(jobs, pid, cause).await;
            let _ = reply.send(plan).await;
        }
        JobRequest::BeginKillByTaskId {
            task_id,
            cause,
            reply,
        } => {
            let plan = match find_pid_by_task_id(jobs, &task_id) {
                Some(pid) => Some((pid, begin_kill(jobs, pid, cause).await)),
                None => None,
            };
            let _ = reply.send(plan).await;
        }
        JobRequest::BeginKillByExecutionKey {
            execution_key,
            cause,
            reply,
        } => {
            let plans = begin_kill_by_execution_key(jobs, &execution_key, cause).await;
            let _ = reply.send(plans).await;
        }
        JobRequest::EndKill { pid, killed, reply } => {
            let result = end_kill(jobs, pid, killed).await;
            release_settled_waiters(jobs, pid, tx).await;
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
            tracked,
            reply,
        } => {
            let result = read_terminal_delta(jobs, &task_id, cursor, max_bytes, !tracked)
                .map_err(TerminalReadError::Failed)
                .and_then(|delta| tracked_delta(reads, &task_id, delta, tracked));
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
        JobRequest::TakeCompletion { task_id, reply } => {
            take_completion(jobs, reads, parked, &task_id);
            let _ = reply.send(()).await;
        }
        JobRequest::Watch { task_id, reply } => {
            let watch = find_pid_by_task_id(jobs, &task_id).map(|pid| {
                let job = jobs.get_mut(&pid).expect("the pid was just found");
                job.watchers += 1;
                JobWatch {
                    tx: tx
                        .upgrade()
                        .expect("a request arrives through a live registry handle"),
                    task_id: job.task_id.clone(),
                }
            });
            let _ = reply.send(watch).await;
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

#[allow(clippy::too_many_arguments)]
fn register_job(
    jobs: &mut HashMap<u32, JobState>,
    pid: u32,
    task_id: IdLease,
    execution_key: String,
    script: String,
    language: ScriptLanguage,
    mode: TerminalMode,
    output_path: Option<PathBuf>,
    place: JobPlace,
) {
    let info = JobInfo {
        place,
        pid,
        task_id: task_id.to_string(),
        execution_key,
        script,
        language,
        mode,
        started_at: Instant::now(),
        status: JobStatus::Running,
        output_path,
    };
    jobs.insert(pid, JobState::new(info, task_id));
    tracing::debug!(pid = %pid, "registered background job");
}

impl JobState {
    const fn new(info: JobInfo, task_id: IdLease) -> Self {
        Self {
            info,
            task_id,
            terminal_buffer: Vec::new(),
            line_buffer: Vec::new(),
            stdout_partial: Vec::new(),
            stderr_partial: Vec::new(),
            stdout_buffer: Vec::new(),
            stderr_buffer: Vec::new(),
            stdout_closed: false,
            stderr_closed: false,
            input_tx: None,
            kill_switch: None,
            process_tree: None,
            kill_cause: None,
            kills_in_flight: 0,
            deferred_end: None,
            settled_waiters: Vec::new(),
            output_redirect: None,
            finished_at: None,
            completion_owed: false,
            watchers: 0,
        }
    }
}

/// How a job ended, before the registry records it.
enum JobEnd {
    Completed { exit_code: i32 },
    Failed { error: String },
    Killed(KillCause),
}

/// Records how `job` ended: the one transition out of
/// [`JobStatus::Running`]. The same step marks its completion
/// [`Delivery::Pending`], so no reader ever sees the job finished with
/// nothing owed to its agent.
fn finish(job: &mut JobState, end: JobEnd) {
    assert!(
        matches!(job.info.status, JobStatus::Running),
        "job {} ended twice: it already reads {:?}",
        job.info.pid,
        job.info.status
    );
    let delivery = Delivery::Pending;
    job.info.status = match end {
        JobEnd::Completed { exit_code } => JobStatus::Completed {
            exit_code,
            delivery,
        },
        JobEnd::Failed { error } => JobStatus::Failed { error, delivery },
        JobEnd::Killed(cause) => JobStatus::Killed { cause, delivery },
    };
    job.finished_at = Some(Instant::now());
    job.kill_switch = None;
    // Dropping the last sender ends the stdin writer thread blocked on it.
    job.input_tx = None;
}

/// Marks the completion of task `task_id` taken, the agent having dropped
/// its [`PendingCompletion`], and forgets the job when nothing watches it.
fn take_completion(
    jobs: &mut HashMap<u32, JobState>,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    task_id: &str,
) {
    if let Some(pid) = find_pid_by_task_id(jobs, task_id) {
        mark_taken(jobs.get_mut(&pid).expect("the pid was just found"));
        forget_if_done(jobs, reads, parked, pid);
    }
}

fn mark_taken(job: &mut JobState) {
    let pid = job.info.pid;
    let delivery = match &mut job.info.status {
        JobStatus::Running => panic!("job {pid}'s completion was taken while it runs"),
        JobStatus::Completed { delivery, .. }
        | JobStatus::Failed { delivery, .. }
        | JobStatus::Killed { delivery, .. } => delivery,
    };
    assert_eq!(
        *delivery,
        Delivery::Pending,
        "job {pid}'s completion was taken twice"
    );
    *delivery = Delivery::Taken;
}

/// Hands out the completion of task `task_id`, or parks the request while a
/// kill of it is in flight.
async fn owe_settled_completion(
    jobs: &mut HashMap<u32, JobState>,
    task_id: &str,
    ending: Result<i32, String>,
    reply: Sender<Option<OwedCompletion>>,
    tx: &WeakSender<JobCommand>,
) {
    let Some(pid) = find_pid_by_task_id(jobs, task_id) else {
        let _ = reply.send(None).await;
        return;
    };
    let job = jobs.get_mut(&pid).expect("the pid was just found");
    if job.kills_in_flight > 0 {
        job.settled_waiters
            .push(SettledWaiter::Owe { ending, reply });
        return;
    }
    let owed = owe_completion(jobs, pid, ending, tx).await;
    let _ = reply.send(Some(owed)).await;
}

/// Hands out the completion of job `pid`, no kill of which is in flight; a
/// job still running ends with `ending` first.
async fn owe_completion(
    jobs: &mut HashMap<u32, JobState>,
    pid: u32,
    ending: Result<i32, String>,
    tx: &WeakSender<JobCommand>,
) -> OwedCompletion {
    if matches!(jobs[&pid].info.status, JobStatus::Running) {
        match ending {
            Ok(exit_code) => complete_job(jobs, pid, exit_code, None).await,
            Err(error) => fail_job(jobs, pid, error, None).await,
        }
    }
    let job = jobs.get_mut(&pid).expect("the job is registered");
    assert!(
        !job.completion_owed,
        "job {pid}'s completion was handed out twice"
    );
    assert!(
        job.info.status.completion_pending(),
        "job {pid} owes a completion it never recorded: {:?}",
        job.info.status
    );
    job.completion_owed = true;
    OwedCompletion {
        kill: job.info.status.kill_cause(),
        pending: PendingCompletion {
            tx: tx
                .upgrade()
                .expect("a request arrives through a live registry handle"),
            task_id: Some(job.task_id.clone()),
        },
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
            // A redirect the model started after the job ended fails the
            // write alone; the job keeps the ending it already recorded.
            if matches!(job.info.status, JobStatus::Running) {
                mark_job_failed(
                    job,
                    format!(
                        "failed to append redirected output for task {}: {error}",
                        job.info.task_id
                    ),
                );
            }
            tracing::error!(
                pid = %pid,
                error = %error,
                "failed to append redirected output"
            );
        }
        let (buffer, partial) = match stream {
            TerminalStream::Stdout => (&mut job.stdout_buffer, &mut job.stdout_partial),
            TerminalStream::Stderr => (&mut job.stderr_buffer, &mut job.stderr_partial),
        };
        buffer.extend_from_slice(&chunk);
        partial.extend_from_slice(&chunk);
        // Each stream completes its own lines; a line reaches the line
        // buffer whole, so one stream never interrupts the other's.
        if let Some(last_newline) = partial.iter().rposition(|byte| *byte == b'\n') {
            job.line_buffer.extend(partial.drain(..=last_newline));
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
        finish(job, JobEnd::Completed { exit_code });
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
        if output_path.is_some() {
            job.info.output_path = output_path;
        }
        tracing::debug!(pid = %pid, "job failed");
    }
}

async fn begin_kill_by_execution_key(
    jobs: &mut HashMap<u32, JobState>,
    execution_key: &str,
    cause: KillCause,
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
        plans.push((pid, begin_kill(jobs, pid, cause).await));
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

/// A read returning what the last read of this task returned is a poll:
/// nothing new arrived. Refuse it so the agent waits for the completion
/// event instead. Untracked reads — the `monitor` pump's own cadence —
/// neither refuse nor update the tracker.
fn tracked_delta(
    reads: &mut ReadTracker,
    task_id: &str,
    delta: TerminalDelta,
    tracked: bool,
) -> Result<TerminalDelta, TerminalReadError> {
    let returned = (delta.cursor, delta.total_bytes, delta.status.clone());
    if !tracked {
        return Ok(delta);
    }
    if reads.delta_reads.get(task_id) == Some(&returned) {
        Err(TerminalReadError::Refused(UnchangedTaskRead {
            task_id: task_id.to_string(),
        }))
    } else {
        reads.delta_reads.insert(task_id.to_string(), returned);
        Ok(delta)
    }
}

fn read_terminal_delta(
    jobs: &HashMap<u32, JobState>,
    task_id: &str,
    cursor: usize,
    max_bytes: usize,
    lines: bool,
) -> Result<TerminalDelta, String> {
    find_pid_by_task_id(jobs, task_id).map_or_else(
        || Err(format!("unknown task_id: {task_id}")),
        |pid| {
            jobs.get(&pid).map_or_else(
                || Err(format!("unknown task_id: {task_id}")),
                |job| terminal_read(job, cursor, max_bytes, lines),
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
    finish(job, JobEnd::Failed { error });
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
async fn begin_kill(jobs: &mut HashMap<u32, JobState>, pid: u32, cause: KillCause) -> KillPlan {
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
        finish(job, JobEnd::Killed(cause));
        job.stdout_closed = true;
        job.stderr_closed = true;
        tracing::info!(pid = %pid, "killed job via runtime kill switch");
        return KillPlan::Done(true);
    }

    if job.kills_in_flight == 0 {
        job.kill_cause = Some(cause);
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
            let cause = job
                .kill_cause
                .expect("a kill in flight was asked for with a cause");
            finish(job, JobEnd::Killed(cause));
            job.stdout_closed = true;
            job.stderr_closed = true;
            tracing::info!(pid = %pid, ?cause, "killed job");
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

/// Answers a settled read at once, or parks it while a kill of the job runs.
async fn get_settled_by_task_id(
    jobs: &mut HashMap<u32, JobState>,
    task_id: &str,
    reply: Sender<Option<JobInfo>>,
) {
    match find_pid_by_task_id(jobs, task_id).and_then(|pid| jobs.get_mut(&pid)) {
        Some(job) if job.kills_in_flight > 0 => {
            job.settled_waiters.push(SettledWaiter::Read(reply));
        }
        Some(job) => {
            let _ = reply.send(Some(job.info.clone())).await;
        }
        None => {
            let _ = reply.send(None).await;
        }
    }
}

/// Answers the callers parked on a job once its last kill in flight ended.
async fn release_settled_waiters(
    jobs: &mut HashMap<u32, JobState>,
    pid: u32,
    tx: &WeakSender<JobCommand>,
) {
    let Some(job) = jobs.get_mut(&pid) else {
        return;
    };
    if job.kills_in_flight > 0 {
        return;
    }
    for waiter in std::mem::take(&mut job.settled_waiters) {
        match waiter {
            SettledWaiter::Read(reply) => {
                let _ = reply.send(Some(jobs[&pid].info.clone())).await;
            }
            SettledWaiter::Owe { ending, reply } => {
                let owed = owe_completion(jobs, pid, ending, tx).await;
                let _ = reply.send(Some(owed)).await;
            }
        }
    }
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

fn terminal_read(
    job: &JobState,
    cursor: usize,
    max_bytes: usize,
    lines: bool,
) -> Result<TerminalDelta, String> {
    let buffer = if lines {
        &job.line_buffer
    } else {
        &job.terminal_buffer
    };
    let total_bytes = buffer.len();
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
        bytes: buffer[cursor..end].to_vec(),
        status: job.info.status.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ids::lease;
    use executor_core::Executor;
    use executor_core::Task;
    use executor_core::tokio::TokioGlobal;
    use tempfile::tempdir;

    fn spawn_registry() -> JobRegistry {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();
        registry
    }

    /// A job's task id stays leased while the registry keeps the job, past
    /// its end, and is released once the registry forgets it: when a job
    /// with the same pid replaces it, or with the registry.
    #[tokio::test]
    async fn a_job_keeps_its_task_id_leased_until_the_registry_forgets_it() {
        let ids = crate::test_ids::RecordingIds::default();
        let allocator = ids.allocator();
        let (registry, service) = job_registry_channel();
        let served = tokio::spawn(service.serve());

        let first = allocator.lease().unwrap();
        registry
            .register(1, &first, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        let second = allocator.lease().unwrap();
        registry
            .register(2, &second, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        drop((first, second));
        registry.complete(1, 0, None).await;
        assert!(registry.get_by_task_id("alfa").await.is_some());
        assert!(ids.released().is_empty(), "the registry holds both");

        let reused_pid = allocator.lease().unwrap();
        registry
            .register(1, &reused_pid, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        drop(reused_pid);
        // The registry serves commands in order, so this list sees the
        // replacement.
        assert_eq!(registry.list().await.len(), 2);
        assert_eq!(ids.released(), ["alfa"]);

        drop(registry);
        served.await.expect("the service ends with its last handle");
        let mut released = ids.released();
        released.sort();
        assert_eq!(released, ["alfa", "bravo", "charlie"]);
    }

    /// Once its completion is taken and nothing watches it, a finished job
    /// leaves the registry: its id goes back to the allocator, every read of
    /// it answers as for an unknown task, and a job that leases the word
    /// again starts with no read history.
    #[tokio::test]
    async fn a_taken_completion_forgets_the_job_and_releases_its_id() {
        let ids = aither_core::AnyIdAllocator::new(aither_core::WordAllocator::new(["river"]));
        let registry = spawn_registry();
        let river = ids.lease().unwrap();
        registry
            .register(1, &river, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        drop(river);
        registry.append_stdout(1, b"done\n".to_vec()).await;
        registry.complete(1, 0, None).await;
        registry
            .terminal_read("river", 0, 1024)
            .await
            .expect("a finished job reads until delivered");
        assert!(ids.lease().is_err(), "the registry holds river");

        // A watch keeps the job past the take, reading it taken.
        let watch = registry.watch("river").await.expect("river is held");
        let owed = registry.owe_completion("river", Ok(0)).await.unwrap();
        drop(owed);
        assert_eq!(
            registry.get_by_task_id("river").await.unwrap().status,
            JobStatus::Completed {
                exit_code: 0,
                delivery: Delivery::Taken
            }
        );
        assert!(ids.lease().is_err(), "the watch holds river");

        drop(watch);
        assert!(registry.list().await.is_empty(), "the job is forgotten");
        assert!(registry.get_by_task_id("river").await.is_none());
        assert!(matches!(
            registry.terminal_read("river", 0, 1024).await,
            Err(TerminalReadError::Failed(_))
        ));
        assert!(!registry.kill_by_task_id("river").await);
        assert!(
            registry
                .terminal_input("river", b"x".to_vec())
                .await
                .is_err()
        );

        // The word names a new job, whose first read is not refused as a
        // repeat of the old job's last one.
        let again = ids.lease().expect("river went back to the allocator");
        registry
            .register(2, &again, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        registry.append_stdout(2, b"done\n".to_vec()).await;
        registry.complete(2, 0, None).await;
        registry
            .terminal_read("river", 0, 1024)
            .await
            .expect("a new job under a reused word reads afresh");
    }

    #[tokio::test]
    async fn task_output_appended_before_registration_lands_first_and_in_order() {
        let registry = spawn_registry();
        let early = lease("early");
        let appends = {
            let registry = registry.clone();
            let early = early.clone();
            tokio::spawn(async move {
                registry.append_task_stdout(&early, b"one\n".to_vec()).await;
                registry.append_task_stdout(&early, b"two\n".to_vec()).await;
            })
        };
        // The appends are parked: the job does not exist, and their sender
        // is still waiting for it.
        tokio::task::yield_now().await;
        assert!(!appends.is_finished());
        assert!(registry.get_by_task_id("early").await.is_none());

        registry
            .register(9, &early, "exec-e", "run", TerminalMode::Sandboxed, None)
            .await;
        appends.await.expect("appends are released by registration");
        registry.append_stdout(9, b"three\n".to_vec()).await;

        let (stdout, _) = registry.terminal_output(9).await.expect("job exists");
        assert_eq!(stdout, b"one\ntwo\nthree\n");
    }

    /// Parked progress refers to its job's lease weakly: it never keeps the
    /// word leased, and once the lease ends it goes with it, so a later job
    /// leasing the same word starts with none of it.
    #[tokio::test]
    async fn parked_output_never_reaches_a_later_job_under_the_same_word() {
        let ids = aither_core::AnyIdAllocator::new(aither_core::WordAllocator::new(["river"]));
        let registry = spawn_registry();
        let first = ids.lease().unwrap();
        let reporter = {
            let registry = registry.clone();
            let first = first.clone();
            tokio::spawn(async move {
                registry
                    .append_task_stdout(&first, b"stale\n".to_vec())
                    .await;
            })
        };
        tokio::task::yield_now().await;
        assert!(!reporter.is_finished(), "the report is parked");
        drop(first);
        assert!(ids.lease().is_err(), "the waiting reporter holds river");

        // The first job never registers; its reporter goes away.
        reporter.abort();
        assert!(reporter.await.unwrap_err().is_cancelled());
        let second = ids.lease().expect("parked output holds no lease");
        registry
            .register(2, &second, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        registry
            .append_task_stdout(&second, b"fresh\n".to_vec())
            .await;

        let (stdout, _) = registry.terminal_output(2).await.expect("job exists");
        assert_eq!(stdout, b"fresh\n");
    }

    /// Progress a command reports after its job was forgotten is dropped:
    /// its sender is released at once, and the word goes back to the
    /// allocator when the command lets go of it.
    #[tokio::test]
    async fn output_for_a_forgotten_job_is_dropped() {
        let ids = aither_core::AnyIdAllocator::new(aither_core::WordAllocator::new(["river"]));
        let registry = spawn_registry();
        let river = ids.lease().unwrap();
        registry
            .register(1, &river, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        registry.complete(1, 0, None).await;
        drop(registry.owe_completion("river", Ok(0)).await.unwrap());
        assert!(registry.list().await.is_empty(), "the job is forgotten");

        // Answered at once, not parked for a registration that never comes.
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            registry.append_task_stdout(&river, b"late\n".to_vec()),
        )
        .await
        .expect("a report to a forgotten job is answered");
        assert!(registry.get_by_task_id("river").await.is_none());
        drop(river);
        let again = ids.lease().expect("nothing kept river");
        registry
            .register(2, &again, "exec", "run", TerminalMode::Sandboxed, None)
            .await;
        let (stdout, _) = registry.terminal_output(2).await.expect("job exists");
        assert!(stdout.is_empty(), "the late report reached no job");
    }

    #[tokio::test]
    async fn unchanged_terminal_read_is_refused_then_succeeds_after_new_output() {
        let registry = spawn_registry();
        registry
            .register(
                7,
                &lease("alpha"),
                "exec-a",
                "run",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        registry.append_stdout(7, b"one".to_vec()).await;

        let first = registry
            .terminal_read("alpha", 0, 1024)
            .await
            .expect("first read succeeds");
        let second = registry.terminal_read("alpha", first.cursor, 1024).await;
        assert!(
            matches!(
                &second,
                Err(TerminalReadError::Refused(UnchangedTaskRead { task_id }))
                    if task_id == "alpha"
            ),
            "an unchanged repeat read must be refused as a poll: {second:?}"
        );

        registry.append_stdout(7, b"two".to_vec()).await;
        let third = registry
            .terminal_read("alpha", first.cursor, 1024)
            .await
            .expect("read succeeds once output changed");
        assert_eq!(third.bytes, b"two");
    }

    #[tokio::test]
    async fn test_job_registry_lifecycle() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        registry
            .register(
                12345,
                &lease("amber"),
                "exec-test",
                "sleep 10",
                TerminalMode::Sandboxed,
                Some(PathBuf::from("outputs/12345.txt")),
            )
            .await;

        let jobs = registry.list().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].pid, 12345);
        assert_eq!(jobs[0].task_id, "amber");
        assert_eq!(jobs[0].execution_key, "exec-test");
        assert!(matches!(jobs[0].status, JobStatus::Running));

        registry.complete(12345, 0, None).await;
        let job = registry.get(12345).await.unwrap();
        assert_eq!(
            job.status,
            JobStatus::Completed {
                exit_code: 0,
                delivery: Delivery::Pending
            }
        );
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
                &lease("alpha"),
                "exec-a",
                "echo hello",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        registry
            .register(
                2,
                &lease("beta"),
                "exec-b",
                "sleep 5",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        registry.complete(2, 0, None).await;

        let running = registry.format_running_jobs().await;
        assert!(running.contains("task alpha"));
        assert!(!running.contains("task beta"));
    }

    #[tokio::test]
    async fn test_output_redirect_streams_full_output() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        let temp = tempdir().unwrap();
        let redirect_path = temp.path().join("stream.log");

        registry
            .register(
                101,
                &lease("stream"),
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
            .start_output_redirect("stream", redirect_path.clone())
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&snapshot), "line1\nline2\n");

        registry.append_stderr(101, b"line3\n".to_vec()).await;
        registry.complete(101, 0, None).await;
        let job = registry.get(101).await.unwrap();
        assert!(matches!(
            job.status,
            JobStatus::Completed { exit_code: 0, .. }
        ));

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
                &lease("delta"),
                "exec-delta",
                "tail -f log",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        registry.append_stdout(202, b"hello ".to_vec()).await;
        registry.append_stderr(202, b"world".to_vec()).await;

        let first = registry.terminal_read("delta", 0, 6).await.unwrap();
        assert_eq!(String::from_utf8_lossy(&first.bytes), "hello ");
        assert_eq!(first.cursor, 6);
        assert_eq!(first.total_bytes, 11);
        assert!(matches!(first.status, JobStatus::Running));

        let second = registry
            .terminal_read("delta", first.cursor, 32)
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&second.bytes), "world");
        assert_eq!(second.cursor, 11);
        assert_eq!(second.total_bytes, 11);
    }

    /// The rotation case: tail writes `tail: ` on stderr, the new file's line
    /// arrives on stdout, then the rest of the notice on stderr. The monitor
    /// reads whole lines, while the merged terminal buffer stays in arrival
    /// order.
    #[tokio::test]
    async fn monitor_reads_whole_lines_when_partial_writes_interleave() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();
        registry
            .register(
                203,
                &lease("lines"),
                "exec-lines",
                "tail -F f",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        registry.append_stderr(203, b"tail: ".to_vec()).await;
        registry.append_stdout(203, b"rotated\n".to_vec()).await;
        registry
            .append_stderr(203, b"'f' has been replaced\n".to_vec())
            .await;
        registry.append_stdout(203, b"unfinished".to_vec()).await;

        let lines = registry.monitor_read("lines", 0, usize::MAX).await.unwrap();
        assert_eq!(
            String::from_utf8_lossy(&lines.bytes),
            "rotated\ntail: 'f' has been replaced\n"
        );
        let merged = registry
            .terminal_read("lines", 0, usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&merged.bytes),
            "tail: rotated\n'f' has been replaced\nunfinished"
        );
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
                &lease("alpha"),
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
                &lease("switch"),
                "exec-kill-switch",
                "sleep 10",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        registry.attach_kill_switch(555, kill_tx).await;

        assert!(registry.kill_by_task_id("switch").await);
        assert!(kill_rx.recv().await.is_ok());
        let job = registry.get(555).await.expect("job should exist");
        assert!(matches!(job.status, JobStatus::Killed { .. }));
    }

    #[tokio::test]
    async fn a_finished_job_releases_its_input_sender_and_refuses_input() {
        let registry = spawn_registry();
        registry
            .register(
                557,
                &lease("input"),
                "exec-input",
                "cat",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        let (input_tx, input_rx) = async_channel::unbounded::<Vec<u8>>();
        registry.attach_terminal_input(557, input_tx).await;
        registry
            .terminal_input("input", b"x".to_vec())
            .await
            .expect("a running job takes input");
        assert_eq!(input_rx.recv().await.expect("input arrives"), b"x");

        registry.complete(557, 0, None).await;

        // The registry handles commands in order, so this reply follows the
        // completion.
        let error = registry
            .terminal_input("input", b"y".to_vec())
            .await
            .expect_err("a finished job refuses input");
        assert!(error.contains("is not running"), "{error}");
        assert!(input_rx.is_closed(), "the registry must drop its sender");
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
                &lease("finished"),
                "exec-killed-complete",
                "sleep 10",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        let (kill_tx, _kill_rx) = async_channel::bounded::<()>(1);
        registry.attach_kill_switch(556, kill_tx).await;
        assert!(registry.kill_by_task_id("finished").await);

        registry.complete(556, 0, None).await;
        let job = registry.get(556).await.expect("job should exist");
        assert!(matches!(job.status, JobStatus::Killed { .. }));
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
                &lease("cjk"),
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
            lease("task"),
            "exec".to_string(),
            "true".to_string(),
            ScriptLanguage::Shell,
            TerminalMode::Sandboxed,
            None,
            JobPlace::Host,
        );
        jobs
    }

    /// Two kills of one job overlap: the first to end must not release an
    /// ending the job's own flow reported while the second still runs.
    #[tokio::test]
    async fn overlapping_kills_hold_the_deferred_ending_until_the_last_ends() {
        let mut jobs = running_job(7);
        let user = KillCause::User;
        assert!(matches!(
            begin_kill(&mut jobs, 7, user).await,
            KillPlan::Group(7)
        ));
        assert!(matches!(
            begin_kill(&mut jobs, 7, user).await,
            KillPlan::Group(7)
        ));
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
            JobStatus::Completed {
                exit_code: 0,
                delivery: Delivery::Pending
            },
            "the last kill to fail releases the ending"
        );
    }

    /// A kill that succeeds wins over an ending reported while it ran.
    #[tokio::test]
    async fn a_successful_kill_wins_over_a_deferred_ending() {
        let mut jobs = running_job(8);
        assert!(matches!(
            begin_kill(&mut jobs, 8, KillCause::User).await,
            KillPlan::Group(8)
        ));
        complete_job(&mut jobs, 8, 0, None).await;
        assert!(end_kill(&mut jobs, 8, true).await);
        assert_eq!(
            jobs[&8].info.status,
            JobStatus::Killed {
                cause: KillCause::User,
                delivery: Delivery::Pending
            }
        );
    }

    /// A job whose process exited from a kill's signal reads as running until
    /// the kill records its outcome; the settled read waits for it.
    #[tokio::test]
    async fn the_settled_read_waits_for_a_kill_in_flight() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();
        registry
            .register(
                557,
                &lease("settled"),
                "exec-settled",
                "sleep 10",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        registry
            .send_command(JobRequest::BeginKill {
                pid: 557,
                cause: KillCause::User,
                reply: reply_tx,
            })
            .await;
        let plan = JobRegistry::recv_reply(reply_rx).await;
        assert!(matches!(plan, KillPlan::Group(557)));
        // The killed process exits and its flow reports completion while the
        // kill has not returned.
        registry.complete(557, 137, None).await;
        let running = registry.get_by_task_id("settled").await;
        assert!(matches!(
            running.expect("job should exist").status,
            JobStatus::Running
        ));

        let waiting = {
            let registry = registry.clone();
            tokio::spawn(async move { registry.get_by_task_id_settled("settled").await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(!waiting.is_finished(), "the kill is still in flight");

        let (reply_tx, reply_rx) = async_channel::bounded(1);
        registry
            .send_command(JobRequest::EndKill {
                pid: 557,
                killed: true,
                reply: reply_tx,
            })
            .await;
        assert!(JobRegistry::recv_reply(reply_rx).await);
        let job = waiting.await.unwrap().expect("job should exist");
        assert!(matches!(job.status, JobStatus::Killed { .. }));
    }

    /// What a host reading the registry can tell: whether the session still
    /// owes its agent a turn.
    fn host_sees_owed_turn(jobs: &[JobInfo]) -> bool {
        jobs.iter()
            .any(|job| job.status == JobStatus::Running || job.status.completion_pending())
    }

    #[tokio::test]
    async fn a_job_reads_its_completion_pending_from_the_transition_that_finishes_it() {
        let registry = spawn_registry();
        registry
            .register(
                9,
                &lease("owed"),
                "exec",
                "run",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        // The registry serves its commands in order, so this list sees the
        // state the finishing transition left, before anything else runs.
        registry.complete(9, 0, None).await;
        let jobs = registry.list().await;
        assert!(
            host_sees_owed_turn(&jobs),
            "a finished job whose completion was never delivered reads as done: {jobs:?}"
        );

        // The job's flow hands the completion out; the agent holds it.
        let owed = registry
            .owe_completion("owed", Ok(0))
            .await
            .expect("the job is registered");
        assert_eq!(owed.kill, None);
        let jobs = registry.list().await;
        assert!(host_sees_owed_turn(&jobs), "{jobs:?}");

        // The agent takes it, and the registry forgets the job.
        drop(owed);
        let jobs = registry.list().await;
        assert!(jobs.is_empty(), "{jobs:?}");
        assert!(!host_sees_owed_turn(&jobs));
    }

    #[tokio::test]
    async fn a_completion_owed_by_a_job_still_running_ends_it_pending() {
        let registry = spawn_registry();
        registry
            .register(
                10,
                &lease("lost"),
                "exec",
                "run",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        // The flow failed without the registry recording how.
        let owed = registry
            .owe_completion("lost", Err("output could not be saved".to_string()))
            .await
            .expect("the job is registered");
        assert_eq!(owed.kill, None);
        let job = registry.get(10).await.expect("job should exist");
        assert_eq!(
            job.status,
            JobStatus::Failed {
                error: "output could not be saved".to_string(),
                delivery: Delivery::Pending
            }
        );
        assert!(
            registry
                .owe_completion("task-never-registered", Ok(0))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_completion_owed_during_a_kill_waits_for_its_outcome() {
        let registry = spawn_registry();
        registry
            .register(
                11,
                &lease("kill"),
                "exec",
                "run",
                TerminalMode::Sandboxed,
                None,
            )
            .await;
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        registry
            .send_command(JobRequest::BeginKill {
                pid: 11,
                cause: KillCause::User,
                reply: reply_tx,
            })
            .await;
        assert!(matches!(
            JobRegistry::recv_reply(reply_rx).await,
            KillPlan::Group(11)
        ));
        let (reply_tx, owed) = async_channel::bounded(1);
        registry
            .send_command(JobRequest::OweCompletion {
                task_id: "kill".to_string(),
                ending: Ok(137),
                reply: reply_tx,
            })
            .await;
        // Served after the request above: the owed completion is parked.
        assert_eq!(
            registry.get(11).await.expect("job should exist").status,
            JobStatus::Running
        );
        assert!(owed.is_empty(), "the kill is still in flight");

        let (reply_tx, reply_rx) = async_channel::bounded(1);
        registry
            .send_command(JobRequest::EndKill {
                pid: 11,
                killed: true,
                reply: reply_tx,
            })
            .await;
        assert!(JobRegistry::recv_reply(reply_rx).await);
        let owed = JobRegistry::recv_reply(owed)
            .await
            .expect("the job is registered");
        assert_eq!(owed.kill, Some(KillCause::User));
        assert_eq!(
            registry.get(11).await.expect("job should exist").status,
            JobStatus::Killed {
                cause: KillCause::User,
                delivery: Delivery::Pending
            }
        );
    }

    #[tokio::test]
    #[should_panic(expected = "handed out twice")]
    async fn a_completion_is_handed_out_once() {
        let (tx, _rx) = async_channel::unbounded();
        let weak = tx.downgrade();
        let mut jobs = HashMap::new();
        register_job(
            &mut jobs,
            12,
            lease("twice"),
            "exec".to_string(),
            "run".to_string(),
            ScriptLanguage::Shell,
            TerminalMode::Sandboxed,
            None,
            JobPlace::Host,
        );
        let _first = owe_completion(&mut jobs, 12, Ok(0), &weak).await;
        let _second = owe_completion(&mut jobs, 12, Ok(0), &weak).await;
    }
}
