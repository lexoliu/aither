//! `monitor` — watch a command or a file and deliver its new output as it
//! arrives.
//!
//! Following a running command or a growing log used to mean polling: sleep,
//! read, sleep, read. `monitor` runs the source as an ordinary background
//! job — same sandbox, same PID-only kills, `terminal_kill` reaches it — and
//! a pump feeds its new output to the agent through [`JobAlert`], which
//! merges increments into the next turn boundary and wakes an idle agent.

use std::borrow::Cow;
use std::fmt;
use std::time::{Duration, Instant};

use aither_core::llm::{CommandOutcome, Tool, ToolContext, ToolResult};
use executor_core::{Executor, Task};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::job_registry::{JobRegistry, JobStatus};
use crate::notices::{MonitorStarted, Refused};
use crate::output::{MediaResolution, OutputFormat};
use crate::permission::PermissionHandler;
use crate::script_language::ScriptLanguage;
use crate::terminal::{Configured, JobAlert, TerminalArgs, TerminalExecutionMode, TerminalTool};

/// The longest a monitor may run: two hours, the same cap `terminal` has.
pub(crate) const MAX_TIMEOUT_SECS: u64 = 7200;

const fn default_interval_secs() -> u64 {
    10
}

/// `monitor` arguments.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MonitorArgs {
    /// What to watch: `{"command": "<source>", "language": "shell"|"python"}`
    /// runs the source inside the sandbox; `{"file": "<path>"}` follows a
    /// file's appended lines inside the sandbox, across rotation and
    /// truncation.
    pub source: MonitorSource,
    /// Seconds between output collections; only non-empty increments are
    /// delivered. Default 10.
    #[serde(default = "default_interval_secs")]
    pub interval_secs: u64,
    /// Optional regex — only matching new lines are delivered.
    #[serde(default)]
    pub filter: Option<String>,
    /// Optional regex — the first matching new line ends the monitor.
    #[serde(default)]
    pub stop_on: Option<String>,
    /// With `stop_on`, also kill the watched process on the match. Default
    /// false: the command keeps running detached.
    #[serde(default)]
    pub kill_on_stop: bool,
    /// Report the run as stuck once if no new output arrives for this many
    /// seconds.
    #[serde(default)]
    pub stall_secs: Option<u64>,
    /// Maximum lifetime in seconds. Required, at most 7200: at the deadline
    /// the process tree is killed and the end reported as a timeout.
    pub timeout_secs: u64,
}

/// The `monitor` source: a command's source text or a file path.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum MonitorSource {
    /// Source text run inside the sandbox.
    Command {
        /// The source to run.
        command: String,
        /// Its language: `shell` or `python`. Default `shell`.
        #[serde(default)]
        language: ScriptLanguage,
    },
    /// A file whose appended lines are followed inside the sandbox.
    File {
        /// The path, resolved against the working directory when relative.
        file: String,
    },
}

/// New output a monitor collected since the last delivery.
#[derive(Debug, Clone)]
pub struct MonitorIncrement {
    /// Task id of the watched job.
    pub task_id: String,
    /// The new lines (filtered when the monitor has a `filter`).
    pub lines: Vec<String>,
}

/// A monitor saw no new output for `stall_secs`.
#[derive(Debug, Clone)]
pub struct MonitorStalled {
    /// Task id of the watched job.
    pub task_id: String,
    /// How long it has produced nothing.
    pub stall_secs: u64,
}

/// A monitor ended, and why.
#[derive(Debug, Clone)]
pub struct MonitorEnded {
    /// Task id of the watched job.
    pub task_id: String,
    /// What ended it.
    pub end: MonitorEnd,
}

/// What ended a monitor.
#[derive(Debug, Clone)]
pub enum MonitorEnd {
    /// The watched command exited.
    Exited {
        /// Its exit code.
        exit_code: i32,
    },
    /// The first line matching `stop_on` arrived.
    StopMatched {
        /// The matching line.
        line: String,
    },
    /// The monitor's `timeout_secs` passed.
    TimedOut,
    /// The watched job was killed (`terminal_kill` or the watchdog).
    Killed,
    /// The watched job failed.
    Failed {
        /// Its error.
        error: String,
    },
    /// The job left the registry with no status.
    Lost,
}

/// Why a `monitor` call was refused before anything started.
#[derive(Debug)]
pub enum MonitorRefusal {
    /// `timeout_secs` was 0 or above the 120-minute cap.
    TimeoutOutOfRange,
    /// `interval_secs` was 0.
    IntervalTooSmall,
    /// `filter` or `stop_on` is not a valid regex.
    BadRegex {
        /// Which argument failed to compile.
        argument: &'static str,
        /// The regex error.
        reason: String,
    },
    /// A file source the sandbox may not read.
    UnreadableFile {
        /// The path as given.
        file: String,
    },
    /// The monitored job would not start.
    StartFailed {
        /// The spawn error.
        reason: String,
    },
}

/// Everything a running monitor needs to know to watch and report.
struct Watch {
    task_id: String,
    /// Keeps the job, and its word, until the pump reports its end, so a
    /// completion the agent took first still leaves the pump its status.
    held: Option<crate::job_registry::JobWatch>,
    registry: JobRegistry,
    alerts: async_channel::Sender<JobAlert>,
    interval: Duration,
    deadline: Instant,
    stall: Option<Duration>,
    filter: Option<regex::Regex>,
    stop_on: Option<regex::Regex>,
    kill_on_stop: bool,
}

/// The native `monitor` tool: starts the watched source as a background job
/// and a pump that delivers its increments through [`JobAlert`].
pub struct MonitorTool<P, E> {
    terminal: TerminalTool<P, E, Configured>,
}

impl<P, E> MonitorTool<P, E> {
    /// Creates the tool bound to `terminal`, the configured terminal whose
    /// jobs it watches.
    #[must_use]
    pub const fn new(terminal: TerminalTool<P, E, Configured>) -> Self {
        Self { terminal }
    }
}

impl<P, E> std::fmt::Debug for MonitorTool<P, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MonitorTool")
    }
}

/// The literal `'` inside a shell single-quoted string.
fn single_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> MonitorTool<P, E> {
    /// The script the monitored job runs for `source` and its language; an
    /// unreadable file is refused before anything spawns.
    fn job_script(
        &self,
        source: &MonitorSource,
    ) -> Result<(String, ScriptLanguage), MonitorRefusal> {
        match source {
            MonitorSource::Command { command, language } => {
                if command.trim().is_empty() {
                    return Err(MonitorRefusal::StartFailed {
                        reason: "command must not be empty".to_string(),
                    });
                }
                Ok((command.clone(), *language))
            }
            MonitorSource::File { file } => {
                let mut path = std::path::PathBuf::from(file);
                if path.is_relative() {
                    path = self.terminal.working_dir().join(path);
                }
                let readable = std::fs::metadata(&path).is_ok_and(|meta| meta.is_file())
                    && (path.starts_with(self.terminal.working_dir())
                        || self
                            .terminal
                            .grants()
                            .snapshot()
                            .readable
                            .iter()
                            .any(|dir| path.starts_with(dir)));
                if !readable {
                    return Err(MonitorRefusal::UnreadableFile { file: file.clone() });
                }
                // `tail -F` follows appended lines and survives rotation
                // (the name is reopened) and truncation.
                Ok((
                    format!("tail -Fn0 {}", single_quote(&path.to_string_lossy())),
                    ScriptLanguage::Shell,
                ))
            }
        }
    }
}

impl<P, E> Tool for MonitorTool<P, E>
where
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        "monitor".into()
    }

    fn description(&self) -> Cow<'static, str> {
        "Watch a command or a file and receive its new output as it arrives. \
         The source runs as a background job under the same sandbox as `terminal`; \
         non-empty increments are delivered at turn boundaries or wake an idle agent. \
         `source` is `{\"command\": <source>, \"language\": \"shell\"|\"python\"}` or \
         `{\"file\": <path>}`. `timeout_secs` is required (max 7200). Options: \
         `interval_secs` (default 10), `filter` regex, `stop_on` regex, `kill_on_stop`, \
         `stall_secs`."
            .into()
    }

    type Arguments = MonitorArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let notices = self.terminal.notices();
        let interval_secs = args.interval_secs;
        match self.start(args).await {
            Ok(task_id) => Ok(ToolResult::text(notices.monitor_started(&MonitorStarted {
                task_id: &task_id,
                interval_secs,
            }))),
            Err(refusal) => {
                Err(Refused::rendered(refusal, |refusal| notices.monitor_refusal(refusal)).into())
            }
        }
    }
}

impl<P: PermissionHandler + 'static, E: Executor + Clone + 'static> MonitorTool<P, E> {
    /// Starts the watched job and its pump; returns the job's task id.
    async fn start(&self, args: MonitorArgs) -> Result<String, MonitorRefusal> {
        if args.timeout_secs == 0 || args.timeout_secs > MAX_TIMEOUT_SECS {
            return Err(MonitorRefusal::TimeoutOutOfRange);
        }
        if args.interval_secs == 0 {
            return Err(MonitorRefusal::IntervalTooSmall);
        }
        let compile = |argument: &'static str, pattern: &Option<String>| {
            pattern
                .as_deref()
                .map(|pattern| {
                    regex::Regex::new(pattern).map_err(|error| MonitorRefusal::BadRegex {
                        argument,
                        reason: error.to_string(),
                    })
                })
                .transpose()
        };
        let filter = compile("filter", &args.filter)?;
        let stop_on = compile("stop_on", &args.stop_on)?;
        let (script, language) = self.job_script(&args.source)?;
        let description = match &args.source {
            MonitorSource::Command { .. } => "monitored command".to_string(),
            MonitorSource::File { file } => format!("monitor file {file}"),
        };
        let result = self
            .terminal
            .run_background(TerminalArgs {
                description,
                script,
                language,
                mode: TerminalExecutionMode::Default,
                ssh_server_id: None,
                expect: OutputFormat::Text,
                resolution: MediaResolution::Auto,
                max_lines: 1,
                raw: true,
                // The job's own watchdog stays a minute past the monitor's
                // deadline, so the pump's kill — not the watchdog's — ends a
                // timed-out run and the end reports `TimedOut`.
                timeout: (args.timeout_secs + 60).min(MAX_TIMEOUT_SECS),
                background: true,
            })
            .await
            .map_err(|error| MonitorRefusal::StartFailed {
                reason: error.to_string(),
            })?;
        let task_id = match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Running { task_id, .. }) => task_id.clone(),
            _ => {
                return Err(MonitorRefusal::StartFailed {
                    reason: "the monitored job did not start detached".to_string(),
                });
            }
        };
        let registry = self.terminal.job_registry();
        let watch = Watch {
            held: registry.watch(&task_id).await,
            task_id: task_id.clone(),
            registry,
            alerts: self.terminal.job_alert_sender(),
            interval: Duration::from_secs(args.interval_secs),
            deadline: Instant::now() + Duration::from_secs(args.timeout_secs),
            stall: args.stall_secs.map(Duration::from_secs),
            filter,
            stop_on,
            kill_on_stop: args.kill_on_stop,
        };
        self.terminal
            .executor()
            .spawn(async move { watch_task(watch).await })
            .detach();
        Ok(task_id)
    }
}

/// Trailing bytes of a line without its `\n` yet; kept across polls.
fn drain_lines(buffer: &mut Vec<u8>) -> Vec<String> {
    let Some(last_nl) = buffer.iter().rposition(|byte| *byte == b'\n') else {
        return Vec::new();
    };
    let chunk: Vec<u8> = buffer.drain(..=last_nl).collect();
    String::from_utf8_lossy(&chunk)
        .lines()
        .map(ToString::to_string)
        .collect()
}

async fn send(alerts: &async_channel::Sender<JobAlert>, alert: JobAlert) {
    let _ = alerts.send(alert).await;
}

async fn watch_task(watch: Watch) {
    let Watch {
        task_id,
        held: _held,
        registry,
        alerts,
        interval,
        deadline,
        stall,
        filter,
        stop_on,
        kill_on_stop,
    } = watch;
    let mut cursor = 0usize;
    let mut remainder = Vec::new();
    let mut last_output = Instant::now();
    let mut stalled = false;
    loop {
        async_io::Timer::after(
            interval
                .min(deadline.saturating_duration_since(Instant::now()) + Duration::from_secs(1)),
        )
        .await;
        let now = Instant::now();
        let mut end = match registry.monitor_read(&task_id, cursor, usize::MAX).await {
            Ok(delta) => {
                cursor = delta.cursor;
                remainder.extend_from_slice(&delta.bytes);
                let mut stop = None;
                let mut lines = Vec::new();
                for line in drain_lines(&mut remainder) {
                    if stop_on.as_ref().is_some_and(|re| re.is_match(&line)) {
                        stop = Some(line);
                        break;
                    }
                    if filter.as_ref().is_none_or(|re| re.is_match(&line)) {
                        lines.push(line);
                    }
                }
                if !lines.is_empty() {
                    last_output = now;
                    send(
                        &alerts,
                        JobAlert::MonitorIncrement(MonitorIncrement {
                            task_id: task_id.clone(),
                            lines,
                        }),
                    )
                    .await;
                }
                if stop.is_some() {
                    last_output = now;
                }
                if let Some(line) = stop {
                    if kill_on_stop {
                        registry.kill_by_task_id(&task_id).await;
                    }
                    Some(MonitorEnd::StopMatched { line })
                } else {
                    match delta.status {
                        JobStatus::Running => None,
                        JobStatus::Completed { exit_code, .. } => {
                            Some(MonitorEnd::Exited { exit_code })
                        }
                        JobStatus::Failed { error, .. } => Some(MonitorEnd::Failed { error }),
                        JobStatus::Killed { .. } => Some(MonitorEnd::Killed),
                    }
                }
            }
            Err(_) => Some(MonitorEnd::Lost),
        };
        if end.is_none() && now >= deadline {
            registry.kill_by_task_id(&task_id).await;
            end = Some(MonitorEnd::TimedOut);
        }
        if let Some(end) = end {
            send(
                &alerts,
                JobAlert::MonitorEnded(MonitorEnded { task_id, end }),
            )
            .await;
            return;
        }
        if !stalled && stall.is_some_and(|stall| now.duration_since(last_output) >= stall) {
            stalled = true;
            send(
                &alerts,
                JobAlert::MonitorStalled(MonitorStalled {
                    task_id: task_id.clone(),
                    stall_secs: stall.map_or(0, |s| s.as_secs()),
                }),
            )
            .await;
        }
    }
}
