//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages the context window, forks compaction in the background, and
//! handles tool execution in an agent-controlled loop.

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

#[cfg(feature = "skills")]
use crate::context::{ActiveSkills, ContextSource as _, SkillCatalogEntry};
use crate::{
    checkpoint::AgentCheckpoint,
    compaction_state::CompactionState,
    config::{AgentConfig, AgentKind, BackgroundJobPolicy},
    context::{
        AnyContextSource, BackgroundJobs, Block, CacheLifetime, CompactionFork,
        CompletedCompaction, Context, ContextBudget, HandoffDocument, InstructionSlot,
        PendingCompaction, PrefixChain, PrefixRewrites, PromptCache, PromptPrefix,
        recovery::{SpillLimits, clean_for_recovery},
        restore_context,
    },
    context_window::{ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot},
    error::AgentError,
    event::{AgentEvent, RunPauseReason, TimedEvent},
    external_hook,
    hook::{
        CheckpointContext, CheckpointReason, Hook, PostToolAction, PreToolAction, StopContext,
        StopReason, ToolResultContext, ToolUseContext, TurnBoundaryAction, TurnBoundaryContext,
    },
    model_group::ModelTier,
    request_parameters::{TierParameters, WithParameters},
    tools::AgentTools,
};
use aither_core::{
    LanguageModel,
    llm::{
        Attachment, CommandStream, Event, LLMRequest, Message, ReasoningState, RetryPolicy, Role,
        StoredFile, ToolCall,
        model::{Ability, Parameters, Profile as ModelProfile},
    },
};
use askama::Template;
use futures_core::Stream;
use futures_lite::StreamExt;
use sha2::{Digest, Sha256};

use aither_sandbox::{
    BackgroundTaskEvent, BackgroundTaskReceiver, DirectoryResetReason, JobAlert, JobProcess,
    JobRegistry, MonitorEnd, PermissionEvent, PermissionEventReceiver, PermissionEventStage,
    TerminalArgs, TerminalExecutionMode, TerminalMode,
};
use std::collections::{HashMap, HashSet, VecDeque};

const BACKGROUND_TASK_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Which of the two futures raced during a turn finished first.
enum TurnStep {
    /// The model turn completed.
    Turn(Result<ModelTurn, AgentError>),
    /// The forked compaction landed while the turn was still running.
    Compacted(Result<CompletedCompaction, crate::context::CompactionError>),
}

/// The number of characters an assembled request renders to, tool calls
/// and tool definitions included — the provider counts all of them.
fn request_chars(
    messages: &[Message],
    definitions: &[aither_core::llm::tool::ToolDefinition],
) -> usize {
    let messages: usize = messages
        .iter()
        .map(|message| {
            let calls: usize = message
                .tool_calls()
                .iter()
                .map(|call| call.name.len() + call.arguments.to_string().len())
                .sum();
            message.content().len() + calls
        })
        .sum();
    let definitions: usize = definitions
        .iter()
        .map(|definition| {
            definition.name().len()
                + definition.description().len()
                + definition.arguments_openai_schema().to_string().len()
        })
        .sum();
    messages + definitions
}

enum ExecutionSignal<T> {
    Tool(T),
    Permission(PermissionEvent),
}

struct ModelTurn {
    response_text: String,
    tool_calls: Vec<ToolCall>,
    /// Opaque provider reasoning produced this turn, in the order it arrived.
    ///
    /// Replayed on the next request: providers that verify their own reasoning
    /// reject a turn whose blocks were reordered or partially dropped.
    reasoning: Vec<ReasoningState>,
    malformed_function_call: bool,
}

/// Streams agent events to the consumer as they happen.
///
/// Wraps an unbounded channel: emitting never blocks, and events emitted
/// after the consumer dropped the stream are discarded.
#[derive(Clone)]
struct EventSink {
    sender: async_channel::Sender<TimedEvent>,
}

impl EventSink {
    /// A sink whose events nobody reads, for tests driving one step of the
    /// loop.
    #[cfg(test)]
    fn discarding() -> Self {
        Self {
            sender: async_channel::unbounded().0,
        }
    }

    fn emit(&self, event: AgentEvent) {
        let _ = self.sender.try_send(TimedEvent {
            at: jiff::Timestamp::now(),
            event,
        });
    }
}

/// What [`Agent::wait_background`] took from the background channel.
#[derive(Debug)]
pub struct BackgroundWake {
    /// Events the background work produced, such as a completed command's
    /// result, for the host to show.
    pub events: Vec<TimedEvent>,
    /// Whether the model has something to act on: a command it started in
    /// the background completed, or one is blocked on stdin. The host then
    /// calls [`Agent::resume`].
    pub needs_model: bool,
    /// The completions this wake recorded. Each job reads
    /// [`Delivery::Pending`](aither_sandbox::Delivery) in its registry until
    /// this is dropped, so a host that drops the wake only once the turn it
    /// starts is visible never reads a finished job with nothing owed.
    pub pending: Vec<aither_sandbox::PendingCompletion>,
}

/// One run of an [`Agent`], from [`Agent::run`] or [`Agent::resume`].
///
/// A run does nothing until [`drive`](Self::drive) consumes it. Driving hands
/// each event to the caller's handler while the run itself keeps going, so a
/// handler that waits on something the run holds lets the run release it.
#[must_use = "a run does nothing until it is driven"]
pub struct Run<F> {
    /// The run's state machine spans every turn and is tens of kilobytes;
    /// boxed so a `Run` moves into the caller's future without copying it.
    run: Pin<Box<F>>,
    events: async_channel::Receiver<TimedEvent>,
}

impl<F> std::fmt::Debug for Run<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Run")
            .field("queued_events", &self.events.len())
            .finish_non_exhaustive()
    }
}

impl<F> Run<F>
where
    F: Future<Output = Result<(), AgentError>>,
{
    fn new(start: impl FnOnce(EventSink) -> F) -> Self {
        let (sender, events) = async_channel::unbounded();
        // The run owns the sink and drops it as it finishes.
        Self {
            run: Box::pin(start(EventSink { sender })),
            events,
        }
    }

    /// Runs to the end, handing each event to `handler` in the order the run
    /// emitted it, one call at a time.
    ///
    /// The run is polled together with every handler call, so it keeps
    /// making progress while the handler awaits; events it emits meanwhile
    /// queue for the next call. The run stays on the caller's task.
    ///
    /// `state` is what the handler works on, typically `&mut` borrows of the
    /// caller's locals: each call takes it and returns it in
    /// [`ControlFlow::Continue`] for the next. Threading it through, rather
    /// than capturing it in an async closure, keeps the driven future `Send`
    /// whenever the state is. A handler returns [`ControlFlow::Break`] to
    /// stop: the run is cancelled, its queued events are discarded, and
    /// `drive` returns that `Break`. Otherwise `drive` returns the state in
    /// `Continue` once the run has finished and its every event was handled.
    ///
    /// `stop` cancels the run from outside, such as on a user's request: it
    /// is polled, ahead of the run and its next event, only while no handler
    /// call is in progress, so a call already running always finishes. Once
    /// it completes, the run is dropped with its queued events and `drive`
    /// returns its value in `Break`. A caller with nothing to stop on passes
    /// [`std::future::pending`].
    ///
    /// # Errors
    ///
    /// The run's error, after the handler has seen every event emitted
    /// before it.
    pub async fn drive<S, B, H>(
        self,
        mut state: S,
        stop: impl Future<Output = B>,
        mut handler: impl FnMut(S, TimedEvent) -> H,
    ) -> Result<ControlFlow<B, S>, AgentError>
    where
        H: Future<Output = ControlFlow<B, S>>,
    {
        let Self { mut run, events } = self;
        futures_lite::pin!(events);
        futures_lite::pin!(stop);
        let mut outcome = None;
        loop {
            let next = std::future::poll_fn(|cx| {
                if let Poll::Ready(value) = stop.as_mut().poll(cx) {
                    return Poll::Ready(RunStep::Stopped(value));
                }
                poll_run(run.as_mut(), &mut outcome, cx);
                if let Some(result) = outcome.take() {
                    // Finished: what it emitted is all queued.
                    return Poll::Ready(match events.try_recv() {
                        Ok(event) => {
                            outcome = Some(result);
                            RunStep::Event(event)
                        }
                        Err(_) => RunStep::Finished(result),
                    });
                }
                match Stream::poll_next(events.as_mut(), cx) {
                    Poll::Ready(Some(event)) => Poll::Ready(RunStep::Event(event)),
                    // Every sink is gone but the run has not returned; its
                    // own wake-up reports the end.
                    Poll::Ready(None) | Poll::Pending => Poll::Pending,
                }
            })
            .await;
            let event = match next {
                RunStep::Event(event) => event,
                RunStep::Finished(result) => {
                    result?;
                    return Ok(ControlFlow::Continue(state));
                }
                RunStep::Stopped(value) => return Ok(ControlFlow::Break(value)),
            };
            let call = handler(state, event);
            futures_lite::pin!(call);
            let flow = std::future::poll_fn(|cx| {
                poll_run(run.as_mut(), &mut outcome, cx);
                call.as_mut().poll(cx)
            })
            .await;
            state = match flow {
                ControlFlow::Continue(state) => state,
                ControlFlow::Break(value) => return Ok(ControlFlow::Break(value)),
            };
        }
    }
}

/// What [`Run::drive`] waits for between handler calls.
enum RunStep<B> {
    Event(TimedEvent),
    Finished(Result<(), AgentError>),
    Stopped(B),
}

/// Polls `run` unless it already finished, recording its result.
fn poll_run<F>(
    run: Pin<&mut F>,
    outcome: &mut Option<Result<(), AgentError>>,
    cx: &mut std::task::Context<'_>,
) where
    F: Future<Output = Result<(), AgentError>>,
{
    if outcome.is_none()
        && let Poll::Ready(result) = run.poll(cx)
    {
        *outcome = Some(result);
    }
}

struct RunLoopOutcome {
    final_text: String,
    stop_reason: StopReason,
    iteration: usize,
}

/// What a tool turn's results gave the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolTurn {
    /// Some call returned something, or a background event arrived with the
    /// results.
    Progress,
    /// Every call was refused for re-reading unchanged output, and no
    /// background event arrived with them.
    OnlyUnchangedReads,
}

enum TurnControl {
    Continue,
    Break {
        final_text: String,
        stop_reason: StopReason,
    },
}

enum ModelEventAction {
    Emit(AgentEvent),
    Continue,
    RetryMalformed,
}

/// A response stream that ended with an error instead of finishing.
struct BrokenStream {
    error: AgentError,
    /// Whether an event arrived before the error: the stream had opened.
    opened: bool,
}

/// A finished tool call, waiting to be paired with its call in the context.
#[derive(Debug)]
pub struct FinishedToolCall {
    id: String,
    name: String,
    result: aither_core::llm::ToolResult,
    /// The call was refused for re-reading output that had not changed
    /// ([`aither_sandbox::PollRefusal::is_unchanged_read`]).
    unchanged_read: bool,
}

type ToolExecutionResult = Result<FinishedToolCall, AgentError>;
type ToolExecutionSignal = ExecutionSignal<ToolExecutionResult>;

async fn handle_model_event<H: Hook>(
    hooks: &H,
    cache_stats: &mut crate::CacheStats,
    budget: &mut ContextBudget,
    event: Result<Event, AgentError>,
    text_chunks: &mut Vec<String>,
    tool_calls: &mut Vec<ToolCall>,
    reasoning: &mut Vec<ReasoningState>,
) -> Result<ModelEventAction, AgentError> {
    match event {
        Ok(Event::Text(text)) => {
            hooks.on_text(&text).await;
            text_chunks.push(text.clone());
            Ok(ModelEventAction::Emit(AgentEvent::Text(text)))
        }
        Ok(Event::Reasoning(reasoning)) => {
            Ok(ModelEventAction::Emit(AgentEvent::Reasoning(reasoning)))
        }
        Ok(Event::ToolCallDelta {
            id,
            name,
            arguments_fragment,
        }) => Ok(ModelEventAction::Emit(AgentEvent::ToolCallDelta {
            id,
            name,
            arguments_fragment,
        })),
        Ok(Event::ToolCall(call)) => {
            tool_calls.push(call);
            Ok(ModelEventAction::Continue)
        }
        // State, unlike Reasoning, is not for the reader: it is collected so
        // the next request can hand it back to the provider.
        Ok(Event::ReasoningState(state)) => {
            reasoning.push(state);
            Ok(ModelEventAction::Continue)
        }
        Ok(Event::BuiltInToolCall { id, tool, detail }) => {
            if !detail.is_empty() {
                text_chunks.push(format_builtin_tool_result(&tool, &detail));
            }
            Ok(ModelEventAction::Emit(AgentEvent::BuiltInToolCall {
                id,
                tool,
                detail,
            }))
        }
        Ok(Event::BuiltInToolResult { id, tool, result }) => {
            text_chunks.push(format_builtin_tool_result(&tool, &result));
            Ok(ModelEventAction::Emit(AgentEvent::BuiltInToolResult {
                id,
                tool,
                result,
            }))
        }
        Ok(Event::Usage(usage)) => {
            cache_stats.record(&usage);
            // The provider just counted the prompt it received; that is the
            // only honest measure of how much window the window costs.
            budget.calibrate(&usage);
            Ok(ModelEventAction::Emit(AgentEvent::Usage(usage)))
        }
        Err(AgentError::Llm { message, .. }) if message.contains("malformed function call") => {
            tracing::warn!("Model generated malformed function call, retrying...");
            Ok(ModelEventAction::RetryMalformed)
        }
        Err(error) => Err(error),
    }
}

fn ensure_non_empty_tool_call_ids(
    tool_calls: &[ToolCall],
    response_text: &str,
) -> Result<(), AgentError> {
    if tool_calls.iter().all(|call| !call.id.trim().is_empty()) {
        return Ok(());
    }
    tracing::error!(
        response_text,
        ?tool_calls,
        "provider emitted tool call with empty id"
    );
    let tool_calls_text = tool_calls
        .iter()
        .map(|call| format!("{} ({})", call.name, call.id))
        .collect::<Vec<_>>()
        .join(", ");
    Err(AgentError::llm_local(format!(
        "provider emitted tool call with empty id; response_text={response_text:?}; tool_calls=[{tool_calls_text}]"
    )))
}

/// The blocks rendered into a request's tail, each with the context source
/// that rendered it, so a commit reaches only that source.
#[derive(Debug, Default)]
struct RenderedTail {
    blocks: Vec<Block>,
    /// The index into the agent's context sources of the source that rendered
    /// the block at the same position; `None` for a block no source owns.
    sources: Vec<Option<usize>>,
}

impl std::ops::Deref for RenderedTail {
    type Target = [Block];

    fn deref(&self) -> &[Block] {
        &self.blocks
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactResult {
    /// Number of messages that were compacted.
    pub messages_compacted: usize,
    /// Number of messages remaining (preserved).
    pub messages_remaining: usize,
    /// The generated summary.
    pub summary: String,
}

/// Whether the pre-compaction alert may still be raised.
///
/// The alert is one-shot per approach: it is raised when usage crosses the
/// warning point and not again until usage has fallen back below it, which
/// only a compaction does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionWarning {
    /// Usage has not crossed the warning point since it last fell below it.
    Armed,
    /// The crossing already happened; nothing raises the alert again.
    Raised,
}

/// Prompt-cache keep-alive bookkeeping for one run.
///
/// The entry's lifetime is the conversation's [`CacheLifetime`], counted
/// from the start of the last request; each keep-alive send refreshes it.
/// `sent` counts sends in the current run so the owner can spot a run that
/// idles frequently.
///
/// [`CacheLifetime`]: crate::context::CacheLifetime
#[derive(Debug, Default)]
pub struct CacheKeepAliveState {
    /// Keep-alives sent so far this run.
    pub(crate) sent: usize,
    /// Whether the frequent-idle threshold was already crossed this run.
    pub(crate) alerted: bool,
    /// Why the run was waiting when the last keep-alive went out — the
    /// background command it was parked on.
    pub(crate) last_reason: Option<String>,
}

impl CacheKeepAliveState {
    /// Starts the per-run counters afresh; a continued agent reports only
    /// the idling of its current run.
    fn begin_run(&mut self) {
        self.sent = 0;
        self.alerted = false;
        self.last_reason = None;
    }
}

#[derive(Debug, Clone, Copy)]
struct EmittedCheckpoint {
    phase: ContextWindowPhase,
    message_count: usize,
}

#[derive(Template)]
#[template(path = "background_started_reminder.txt", escape = "none")]
struct BackgroundStartedReminderTemplate<'a> {
    task_id: &'a str,
    output_preview: &'a str,
    output_file: &'a StoredFile,
}

/// `knowledge_and_time` system block. `websearch_command` selects whether the
/// verification hint names the `websearch` CLI command or a provider-native
/// web search capability.
#[derive(Template)]
#[template(path = "knowledge_and_time.txt", escape = "none")]
pub struct KnowledgeAndTimeTemplate {
    pub websearch_command: bool,
}

/// A user message with the UTC time it was sent at appended.
#[derive(Template)]
#[template(path = "user_message_time.txt", escape = "none")]
struct UserMessageTimeTemplate<'a> {
    prompt: &'a str,
    utc: String,
}

/// The system prompt's statement of the communication and working languages.
#[cfg(feature = "languages")]
#[derive(Template)]
#[template(path = "language_system.txt", escape = "none")]
struct LanguagesTemplate {
    communication: crate::Language,
    working: crate::Language,
}

#[derive(Template)]
#[template(path = "background_terminal_result.txt", escape = "none")]
struct BackgroundTerminalResultTemplate<'a> {
    task_id: &'a str,
    script: &'a str,
    outcome: BackgroundOutcome<'a>,
    output: Option<&'a str>,
    stderr: Option<&'a str>,
    interruptions: &'a str,
}

/// Whether the session's first run has begun.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    /// No run has begun.
    NotStarted,
    /// A run has begun.
    Started,
}

/// The alert text for a background job event.
#[derive(Template)]
#[template(path = "job_alert.txt", escape = "none")]
struct JobAlertTemplate<'a> {
    alert: &'a JobAlertText<'a>,
}

/// What a background job alert reports.
enum JobAlertText<'a> {
    /// A foreground command left processes running; they were killed.
    DescendantsKilled {
        processes: &'a [JobProcess],
        omitted: usize,
    },
    /// The job crossed the long-running threshold.
    LongRunning { task_id: &'a str, elapsed: String },
    /// The job is waiting for input on its stdin.
    StdinBlocked { task_id: &'a str },
    /// Watching the job's stdin failed.
    StdinProbeFailed { task_id: &'a str, error: &'a str },
    /// The carried working directory was reset.
    DirectoryReset {
        dir: String,
        reason: DirectoryResetReason,
    },
    /// A monitor delivered new output lines.
    MonitorIncrement {
        task_id: &'a str,
        lines: &'a [String],
    },
    /// A monitor saw no output for its stall window.
    MonitorStalled { task_id: &'a str, stall_secs: u64 },
    /// A monitor ended.
    MonitorEnded {
        task_id: &'a str,
        end: MonitorEndText<'a>,
    },
}

/// How a monitor ended, as the alert text reports it.
enum MonitorEndText<'a> {
    /// The watched command exited.
    Exited { exit_code: i32 },
    /// A `stop_on` line matched.
    StopMatched { line: &'a str },
    /// The monitor's `timeout_secs` passed.
    TimedOut,
    /// The watched job was killed.
    Killed,
    /// The watched job failed.
    Failed { error: &'a str },
    /// The job left the registry with no status.
    Lost,
}

fn job_alert_text(alert: &JobAlertText<'_>) -> String {
    crate::notices::rendered(&JobAlertTemplate { alert })
}

/// aither's wording of a background job alert.
pub fn default_job_alert(alert: &JobAlert) -> String {
    let text = match alert {
        JobAlert::DescendantsKilled(killed) => JobAlertText::DescendantsKilled {
            processes: &killed.processes,
            omitted: killed.omitted,
        },
        JobAlert::DirectoryReset(reset) => JobAlertText::DirectoryReset {
            dir: reset.dir.display().to_string(),
            reason: reset.reason,
        },
        JobAlert::LongRunning(job) => JobAlertText::LongRunning {
            task_id: &job.task_id,
            elapsed: format_job_elapsed(job.elapsed),
        },
        JobAlert::StdinBlocked(job) => JobAlertText::StdinBlocked {
            task_id: &job.task_id,
        },
        JobAlert::StdinProbeFailed(job) => JobAlertText::StdinProbeFailed {
            task_id: &job.task_id,
            error: &job.error,
        },
        JobAlert::MonitorIncrement(increment) => JobAlertText::MonitorIncrement {
            task_id: &increment.task_id,
            lines: &increment.lines,
        },
        JobAlert::MonitorStalled(stalled) => JobAlertText::MonitorStalled {
            task_id: &stalled.task_id,
            stall_secs: stalled.stall_secs,
        },
        JobAlert::MonitorEnded(ended) => JobAlertText::MonitorEnded {
            task_id: &ended.task_id,
            end: match &ended.end {
                MonitorEnd::Exited { exit_code } => MonitorEndText::Exited {
                    exit_code: *exit_code,
                },
                MonitorEnd::StopMatched { line } => MonitorEndText::StopMatched { line },
                MonitorEnd::TimedOut => MonitorEndText::TimedOut,
                MonitorEnd::Killed => MonitorEndText::Killed,
                MonitorEnd::Failed { error } => MonitorEndText::Failed { error },
                MonitorEnd::Lost => MonitorEndText::Lost,
            },
        },
    };
    job_alert_text(&text)
}

/// aither's wording of the reminder after a command moved to the
/// background.
pub fn default_background_started(task: &aither_sandbox::StartedBackgroundTask) -> String {
    crate::notices::rendered(&BackgroundStartedReminderTemplate {
        task_id: &task.task_id,
        output_preview: &task.output_preview,
        output_file: &task.output_file,
    })
}

/// What a killed memory runaway held, as the text reports it.
struct RunawayText {
    footprint: aither_sandbox::MemorySize,
    capacity: aither_sandbox::MemorySize,
    domain: aither_sandbox::MemoryDomain,
}

/// How a finished background task ended.
enum BackgroundOutcome<'a> {
    /// The script ran to completion.
    Finished { code: i32 },
    /// The job was killed — its `timeout` bound or `terminal_kill` — and
    /// whatever `code` the harness saw means nothing.
    Killed {
        timeout: Option<&'a str>,
        /// Set when a memory runaway was the cause.
        runaway: Option<RunawayText>,
    },
    /// The task could not be run or tracked to completion.
    Failed { error: &'a str },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PermissionEventKey {
    mode: TerminalMode,
    script: String,
}

#[derive(Debug, Default)]
struct PermissionPauseTracker {
    pending: HashMap<PermissionEventKey, VecDeque<String>>,
    active: HashMap<PermissionEventKey, VecDeque<String>>,
}

impl PermissionPauseTracker {
    fn from_tool_calls(tool_calls: &[aither_core::llm::ToolCall]) -> Self {
        let mut tracker = Self::default();
        for call in tool_calls {
            let Some(key) = permission_event_key_for_call(call.name.as_str(), &call.arguments)
            else {
                continue;
            };
            tracker
                .pending
                .entry(key)
                .or_default()
                .push_back(call.id.clone());
        }
        tracker
    }

    fn event_for(&mut self, event: PermissionEvent) -> Option<(RunPauseReason, String)> {
        let key = PermissionEventKey {
            mode: event.mode,
            script: event.script,
        };
        match event.stage {
            PermissionEventStage::Waiting => {
                let tool_call_id = self.pending.get_mut(&key)?.pop_front()?;
                self.active
                    .entry(key)
                    .or_default()
                    .push_back(tool_call_id.clone());
                Some((RunPauseReason::PermissionRequest, tool_call_id))
            }
            PermissionEventStage::Resolved => {
                let tool_call_id = self.active.get_mut(&key)?.pop_front()?;
                Some((RunPauseReason::PermissionRequest, tool_call_id))
            }
        }
    }
}

/// An autonomous agent that processes tasks using tiered language models.
///
/// The agent manages conversation context, handles tool execution in a loop,
/// and applies context compression when needed. Unlike the raw `LanguageModel`
/// trait which only emits events, the agent handles the full tool execution
/// cycle.
///
/// # Type Parameters
///
/// - `Advanced`: The primary LLM for main reasoning (most capable)
/// - `Balanced`: LLM for moderate tasks like subagents (defaults to Advanced)
/// - `Fast`: LLM for quick tasks like translating a reply (defaults to Balanced)
/// - `H`: Composed hooks for customizing behavior (defaults to `()`)
/// - `Compactor`: The optional compaction model, see
///   [`AgentBuilder::compaction_model`](crate::builder::AgentBuilder::compaction_model)
///   (defaults to `Advanced`; unused while none is set)
///
/// # Model Tier Selection
///
/// Each agent uses one tier for its main reasoning loop (set via `tier()`):
/// - Main agent: typically `ModelTier::Advanced`
/// - Explore subagent: `ModelTier::Balanced` (cheaper but capable)
/// - Quick tasks: `ModelTier::Fast`
///
/// Compaction runs on the session's own tier, so it reads the conversation's
/// prompt cache, unless a compaction model is set.
///
/// # Example
///
/// ```rust,ignore
/// // Simple usage (all tiers use same model)
/// let agent = Agent::new(claude);
/// let response = agent.query("What is 2+2?").await?;
///
/// // With tiered models, each with the parameters of its requests
/// let agent = Agent::builder(opus)
///     .request_parameters(opus_parameters)
///     .balanced_model(sonnet, sonnet_parameters)
///     .fast_model(haiku, haiku_parameters)
///     .build();
///
/// // Subagent using balanced tier
/// let explore_agent = Agent::builder(opus)
///     .balanced_model(sonnet, sonnet_parameters)
///     .fast_model(haiku, haiku_parameters)
///     .tier(ModelTier::Balanced)  // Use sonnet for reasoning
///     .build();
/// ```
///
/// `Schema` is the [`HandoffSchema`](crate::HandoffSchema) every compaction
/// asks the model for; [`HandoffDocument`] unless the builder's
/// [`handoff`](crate::AgentBuilder::handoff) sets another.
#[derive(Debug)]
pub struct Agent<
    Advanced,
    Balanced = Advanced,
    Fast = Balanced,
    H = (),
    Compactor = Advanced,
    Schema = HandoffDocument,
> {
    /// Primary LLM for main reasoning (most capable).
    /// The advanced reasoning model, behind an Arc so a forked compaction
    /// can own it while the run continues.
    pub(crate) advanced: Arc<Advanced>,

    /// Balanced LLM for moderate tasks (subagents).
    /// The balanced model, behind an Arc for the same reason.
    pub(crate) balanced: Arc<Balanced>,

    /// Fast LLM for quick tasks (compaction, ask command).
    ///
    /// Shared rather than owned because compaction runs as a future that
    /// outlives the borrow of `self` taken by the turn it races against.
    pub(crate) fast: Arc<Fast>,

    /// The model compaction runs on, when it is not the session's own.
    ///
    /// A different model cannot read the conversation's prompt cache, so with
    /// one set compaction is never run early to catch the cache warm (see
    /// [`Agent::compact_when_idle`]), and an idle gap past the cache lifetime
    /// reassembles the window instead. Its requests carry its own parameters.
    pub(crate) compaction_model: Option<WithParameters<Arc<Compactor>>>,

    /// The parameters of the requests sent on each tier's model.
    pub(crate) parameters: TierParameters,

    /// The request-level reasoning effort the agent's own requests carry.
    pub(crate) effort: crate::effort::EffortAnchor,

    /// Whether the idle compaction already ran since the last request, so one
    /// idle gap compacts once.
    pub(crate) idle_compacted: bool,

    /// Which model tier to use for main reasoning.
    pub(crate) tier: ModelTier,

    /// Registered tools.
    pub(crate) tools: AgentTools,

    /// Composed hooks.
    pub(crate) hooks: H,

    /// Agent configuration.
    pub(crate) config: AgentConfig,

    /// Unified context manager.
    pub(crate) context: Context,

    /// Window accounting, calibrated against the provider's token counts.
    pub(crate) budget: ContextBudget,

    /// Prompt cache controls derived from the current window layout.
    pub(crate) prompt_cache: PromptCache,

    /// Sources rendered into the volatile tail of every request.
    pub(crate) context_sources: Vec<AnyContextSource>,

    /// Whether the window was rewritten since the sources last heard a
    /// request begin, which wipes every durable alert they pushed before.
    pub(crate) reassembled: bool,

    /// What rewrote the prompt prefix since the previous request, reported
    /// with the next one's [`AgentEvent::PromptPrefix`].
    pub(crate) prefix_rewrites: PrefixRewrites,

    /// How many requests this agent has sent.
    pub(crate) requests_sent: u64,

    /// The one compaction in the making: running alongside the turn, or
    /// finished and waiting for a request boundary.
    pub(crate) compaction: CompactionState,

    /// Why the compaction forked alongside the run failed, until one lands:
    /// the next compaction the window forces recovers first.
    pub(crate) compaction_failure: Option<crate::context::CompactionError>,

    /// Results of the tool calls of the turn in flight, in completion order,
    /// until they are moved into the context.
    pub(crate) tool_results: VecDeque<ToolExecutionResult>,

    /// Cached model profile (for the selected tier).
    pub(crate) profile: Option<ModelProfile>,

    /// The compaction model's profile, when one is set. It tells the cache
    /// controls of a compaction request what that model's client will do with
    /// the messages.
    pub(crate) compaction_profile: Option<ModelProfile>,

    /// Whether tools have been bootstrapped.
    pub(crate) initialized: bool,

    /// Whether the session-start lifecycle event has fired.
    pub(crate) session: SessionPhase,

    /// Receiver for completed background terminal tasks.
    pub(crate) background_receiver: Option<BackgroundTaskReceiver>,
    /// Receiver for permission wait/resume events emitted by terminal execution.
    pub(crate) permission_receiver: Option<PermissionEventReceiver>,
    /// The terminal calls in which a kept command succeeded.
    pub(crate) kept_receiver: Option<aither_sandbox::KeptExchangeReceiver>,
    /// Calls reported by `kept_receiver` whose results are not committed yet.
    pub(crate) kept_exchanges: HashSet<String>,
    /// The commands `terminal` runs, kept current in the command table
    /// block; `None` for an agent without a terminal agent's command table.
    pub(crate) commands: Option<crate::command_catalog::CommandCatalog>,
    /// Receiver for the host memory pressure and machine load events.
    pub(crate) host_load_events: Option<async_channel::Receiver<crate::HostLoadEvent>>,
    /// Registry for running background terminal tasks.
    pub(crate) job_registry: Option<JobRegistry>,

    /// Task ids of jobs the model knows as background tasks.
    ///
    /// Every spawned job reports a `CompletedTask` when it ends, foreground
    /// ones included, but only a job that detached — started `background:
    /// true` or promoted past the foreground window — is announced to the
    /// model. An id enters the set when `take_started` reports it; a
    /// completion whose id is absent already returned through the tool call
    /// and is dropped.
    pub(crate) detached_task_ids: HashSet<String>,

    /// Optional sandbox directory for working-doc supervision (`tasks.md`).
    pub(crate) sandbox_dir: Option<PathBuf>,

    /// What the host gives the `tasks.md` supervision; the agent records its
    /// finished background jobs on its events.
    pub(crate) tasks: crate::tasks::TasksWiring,

    /// The source supervising `tasks.md` in the sandbox, owned apart from
    /// [`context_sources`](Self::context_sources) because what it showed
    /// travels in the agent's checkpoint.
    pub(crate) tasks_document: Option<crate::tasks::TasksDocument>,

    /// Directory files are restored from after a forked compaction.
    pub(crate) restore_dir: Option<PathBuf>,

    /// Whether the provider's prompt-cache entry is warm; stamped on every
    /// request and keep-alive, shared with hosts through
    /// [`Agent::cache_warmth`].
    pub(crate) warmth: crate::CacheWarmth,

    /// Ends a provider stream in flight when the machine wakes or goes
    /// offline.
    pub(crate) connectivity: crate::connectivity::Watch,

    /// The most recent LLM request issued by this agent, retained so a
    /// prompt-cache keep-alive can re-send it verbatim.
    pub(crate) last_request: Option<LLMRequest>,

    /// Prompt-cache keep-alive bookkeeping for the current run.
    pub(crate) cache_keep_alive: CacheKeepAliveState,

    /// Where every request this agent sends is published for the `fork`
    /// subagent kind to start from; `None` when no fork reads this agent.
    pub(crate) fork_source: Option<crate::fork::ForkSource>,

    /// What the `compact` tool shares with this agent; `None` without the
    /// tool, which also means no pre-compaction alert is worth raising.
    pub(crate) compact_signal: Option<Arc<crate::compact_tool::CompactSignal>>,

    /// Whether the pre-compaction alert has been raised for the current
    /// approach to the auto-compact point.
    pub(crate) compaction_warning: CompactionWarning,

    /// The agent every tool call of this loop is made by.
    pub(crate) identity: aither_core::llm::AgentIdentity,
    /// Renders every alert, notice, reminder and refusal the model reads.
    pub(crate) notices: crate::AnyNotices,

    /// Rolling KV-cache statistics accumulated from emitted `Usage` events.
    ///
    /// Hosts can read [`Agent::cache_stats`] at any time to observe the
    /// cumulative prompt-caching hit rate for this agent session.
    pub(crate) cache_stats: crate::CacheStats,

    /// Runtime skill registry loaded by host integrations.
    /// Automatic prompt-triggered activation is intentionally disabled.
    #[cfg(feature = "skills")]
    pub(crate) skill_registry: Option<Arc<aither_skills::SkillRegistry>>,

    /// Skills activated for the current run.
    ///
    /// Held as the context source that renders them, so the instructions the
    /// model sees and the allowlist the tools enforce cannot drift apart.
    #[cfg(feature = "skills")]
    pub(crate) active_skills: ActiveSkills,

    /// Exact tool-name allowlist derived from activated skills.
    #[cfg(feature = "skills")]
    pub(crate) active_allowed_tools: Option<HashSet<String>>,

    /// The schema compactions ask for; no value is held.
    pub(crate) handoff_schema: std::marker::PhantomData<fn() -> Schema>,
}

impl<LLM: LanguageModel + Clone> Agent<LLM, LLM, LLM, (), LLM> {
    /// Creates a new agent with default configuration.
    ///
    /// All model tiers (advanced/balanced/fast) use the same model.
    #[must_use]
    pub fn new(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::default())
    }

    /// Creates a new agent with the specified configuration.
    ///
    /// All model tiers (advanced/balanced/fast) use the same model.
    #[must_use]
    pub fn with_config(llm: LLM, config: AgentConfig) -> Self {
        let prompt_cache = PromptCache::new(uuid::Uuid::new_v4().to_string());
        Self {
            warmth: crate::CacheWarmth::new(prompt_cache.lifetime(), crate::AnyClock::default()),
            advanced: Arc::new(llm.clone()),
            balanced: Arc::new(llm.clone()),
            fast: Arc::new(llm),
            compaction_model: None,
            parameters: TierParameters::default(),
            effort: crate::effort::EffortAnchor::default(),
            idle_compacted: false,
            tier: ModelTier::default(),
            tools: AgentTools::new(),
            hooks: (),
            config,
            context: Context::default(),
            budget: ContextBudget::default(),
            prompt_cache,
            context_sources: Vec::new(),
            reassembled: false,
            prefix_rewrites: PrefixRewrites::default(),
            requests_sent: 0,
            compaction: CompactionState::default(),
            compaction_failure: None,
            tool_results: VecDeque::new(),
            profile: None,
            compaction_profile: None,
            initialized: false,
            session: SessionPhase::NotStarted,
            connectivity: crate::connectivity::Watch::default(),
            background_receiver: None,
            permission_receiver: None,
            kept_receiver: None,
            kept_exchanges: HashSet::new(),
            commands: None,
            host_load_events: None,
            job_registry: None,
            detached_task_ids: HashSet::new(),
            sandbox_dir: None,
            tasks: crate::tasks::TasksWiring::default(),
            tasks_document: None,
            restore_dir: None,
            last_request: None,
            cache_keep_alive: CacheKeepAliveState::default(),
            fork_source: None,
            compact_signal: None,
            compaction_warning: CompactionWarning::Armed,
            identity: aither_core::llm::AgentIdentity::Main,
            notices: crate::AnyNotices::default(),
            cache_stats: crate::CacheStats::new(),
            #[cfg(feature = "skills")]
            skill_registry: None,
            #[cfg(feature = "skills")]
            active_skills: ActiveSkills::new(),
            #[cfg(feature = "skills")]
            active_allowed_tools: None,
            handoff_schema: std::marker::PhantomData,
        }
    }
}

impl<LLM: LanguageModel + Clone> Agent<LLM, LLM, LLM, ()> {
    /// Returns a builder for more complex agent construction.
    pub fn builder(llm: LLM) -> crate::builder::AgentBuilder<LLM, LLM, LLM, ()> {
        crate::builder::AgentBuilder::new(llm)
    }
}

impl<Advanced, Balanced, Fast, H, Compactor, Schema>
    Agent<Advanced, Balanced, Fast, H, Compactor, Schema>
where
    Schema: crate::HandoffSchema,
    Advanced: LanguageModel + 'static,
    Balanced: LanguageModel + 'static,
    // Compaction runs as an owned future that outlives the turn it races
    // against, so the fast model it borrows must outlive the agent.
    Fast: LanguageModel + 'static,
    // Owned by the compaction future for the same reason.
    Compactor: LanguageModel + 'static,
    H: Hook,
{
    /// Response stream of the active tier, boxed so the provider's concrete
    /// stream type does not inflate this async chain's poll frames in debug
    /// builds (aither#143).
    fn respond_stream<'a>(
        tier: ModelTier,
        advanced: &'a Arc<Advanced>,
        balanced: &'a Arc<Balanced>,
        fast: &'a Arc<Fast>,
        connectivity: &crate::connectivity::Watch,
        request: LLMRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, AgentError>> + Send + 'a>> {
        let stream: crate::connectivity::EventStream<'a> = match tier {
            ModelTier::Advanced => Box::pin(
                advanced
                    .respond(request)
                    .map(|event| event.map_err(|error| AgentError::from_model_error(&error))),
            ),
            ModelTier::Balanced => Box::pin(
                balanced
                    .respond(request)
                    .map(|event| event.map_err(|error| AgentError::from_model_error(&error))),
            ),
            ModelTier::Fast => Box::pin(
                fast.respond(request)
                    .map(|event| event.map_err(|error| AgentError::from_model_error(&error))),
            ),
        };
        connectivity.guard(stream)
    }

    /// The retry policy of the active tier's model.
    fn retry_policy(&self) -> RetryPolicy {
        match self.tier {
            ModelTier::Advanced => self.advanced.retry_policy(),
            ModelTier::Balanced => self.balanced.retry_policy(),
            ModelTier::Fast => self.fast.retry_policy(),
        }
    }

    /// Streams the model's response to `request`, sending the request again
    /// when the stream breaks after it opened.
    ///
    /// A failure before the first event is the provider's to retry, since it
    /// retries establishing the stream itself; a break after it is retried
    /// here, under the model's [`RetryPolicy`], unless the machine is
    /// offline. The broken attempt's text, calls and reasoning are dropped
    /// with it, and the request sent again is the one recorded as the last
    /// request, so the provider reads the same cached prefix.
    async fn collect_model_turn(
        &mut self,
        mut request: LLMRequest,
        events: &EventSink,
    ) -> Result<ModelTurn, AgentError> {
        let policy = self.retry_policy();
        let mut retries = 0;
        loop {
            let broken = match self.stream_model_turn(request, events).await {
                Ok(turn) => return Ok(turn),
                Err(broken) => broken,
            };
            let failure = match broken.error.failure() {
                Some(failure)
                    if broken.opened
                        && policy.allows(retries, failure)
                        && !self.connectivity.is_offline() =>
                {
                    *failure
                }
                _ => return Err(broken.error),
            };
            let delay = policy.delay(retries, &failure);
            retries += 1;
            tracing::warn!(
                retry = retries,
                max_retries = policy.max_retries,
                delay_ms = delay.as_millis(),
                error = %broken.error,
                "model response stream broke mid-response; sending the request again"
            );
            events.emit(AgentEvent::ResponseRetry {
                error: broken.error,
                retry: retries,
                max_retries: policy.max_retries,
                delay,
            });
            async_io::Timer::after(delay).await;
            request = self
                .last_request
                .clone()
                .expect("a request that reached the model is recorded as the last request");
        }
    }

    /// One attempt at streaming the model's response to `request`.
    async fn stream_model_turn(
        &mut self,
        request: LLMRequest,
        events: &EventSink,
    ) -> Result<ModelTurn, BrokenStream> {
        let mut text_chunks = Vec::new();
        let mut tool_calls = Vec::new();
        let mut reasoning = Vec::new();
        let mut malformed_function_call = false;
        let mut opened = false;

        let stream = Self::respond_stream(
            self.tier,
            &self.advanced,
            &self.balanced,
            &self.fast,
            &self.connectivity,
            request,
        );
        futures_lite::pin!(stream);
        while let Some(event) = stream.next().await {
            let received = event.is_ok();
            let action = handle_model_event(
                &self.hooks,
                &mut self.cache_stats,
                &mut self.budget,
                event,
                &mut text_chunks,
                &mut tool_calls,
                &mut reasoning,
            )
            .await
            .map_err(|error| BrokenStream { error, opened })?;
            opened |= received;
            match action {
                ModelEventAction::Emit(event) => events.emit(event),
                ModelEventAction::Continue => {}
                ModelEventAction::RetryMalformed => {
                    malformed_function_call = true;
                    break;
                }
            }
        }

        Ok(ModelTurn {
            response_text: text_chunks.join(""),
            tool_calls,
            reasoning,
            malformed_function_call,
        })
    }

    async fn run_streaming(
        &mut self,
        prompt: String,
        attachments: Vec<Attachment>,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let run_id = uuid::Uuid::new_v4().to_string();
        self.cache_keep_alive.begin_run();
        events.emit(AgentEvent::run_start(
            run_id.clone(),
            self.config.max_iterations,
        ));
        self.ensure_initialized().await?;
        if std::mem::replace(&mut self.session, SessionPhase::Started) == SessionPhase::NotStarted {
            self.hooks
                .on_lifecycle(&external_hook::SessionStart {})
                .await;
        }
        #[cfg(feature = "languages")]
        if let Some(guard) = &self.config.languages {
            guard.record_user_message(&prompt);
        }
        self.hooks
            .on_lifecycle(&external_hook::UserMessage {
                message: prompt.clone(),
                attachments: attachments.len(),
            })
            .await;
        #[cfg(feature = "skills")]
        for event in self.activate_skills_for_prompt(&prompt) {
            events.emit(event);
        }
        self.settle_interrupted_tool_calls();
        // An effort change lands before the user message it applies to.
        self.settle_effort()?;
        self.context.push(
            self.timestamped_user_message(&prompt)
                .with_attachments(attachments),
        );

        let outcome = self.run_agent_loop(run_id.as_str(), 0, events).await?;
        self.finish_run(run_id.as_str(), outcome, events).await
    }

    /// The user message for `prompt` with the current UTC time appended.
    ///
    /// The time is part of the message from the moment it joins the context
    /// and is never rewritten, so the cached prefix a later request replays
    /// is unchanged.
    fn timestamped_user_message(&self, prompt: &str) -> Message {
        let utc = self
            .warmth
            .utc_now()
            .strftime("%Y-%m-%d %H:%M:%S")
            .to_string();
        Message::user(
            UserMessageTimeTemplate { prompt, utc }
                .render()
                .expect("failed to render user_message_time template"),
        )
    }

    /// A run without a new user message: the model acts on what
    /// [`wait_background`](Self::wait_background) recorded.
    async fn resume_streaming(&mut self, events: &EventSink) -> Result<(), AgentError> {
        let run_id = uuid::Uuid::new_v4().to_string();
        self.cache_keep_alive.begin_run();
        events.emit(AgentEvent::run_start(
            run_id.clone(),
            self.config.max_iterations,
        ));
        self.ensure_initialized().await?;
        self.settle_interrupted_tool_calls();
        let outcome = self.run_agent_loop(run_id.as_str(), 0, events).await?;
        self.finish_run(run_id.as_str(), outcome, events).await
    }

    /// Ends a run whose loop stopped: collects its background events as the
    /// [`BackgroundJobPolicy`] says, then emits the stop events.
    async fn finish_run(
        &mut self,
        run_id: &str,
        outcome: RunLoopOutcome,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        if outcome.stop_reason != StopReason::EndTurn
            && self
                .collect_final_background_events(run_id, outcome.iteration, events)
                .await?
        {
            return Ok(());
        }
        self.push_stop_events(run_id, outcome, events).await
    }

    async fn run_agent_loop(
        &mut self,
        run_id: &str,
        turn_offset: usize,
        events: &EventSink,
    ) -> Result<RunLoopOutcome, AgentError> {
        let mut iteration = 0;
        loop {
            iteration += 1;
            let turn = turn_offset + iteration;
            if turn > self.config.max_iterations {
                return Err(AgentError::MaxIterations {
                    limit: self.config.max_iterations,
                });
            }
            match self.run_agent_turn(run_id, turn, events).await? {
                TurnControl::Continue => {}
                TurnControl::Break {
                    final_text,
                    stop_reason,
                } => {
                    return Ok(RunLoopOutcome {
                        final_text,
                        stop_reason,
                        iteration: turn,
                    });
                }
            }
        }
    }

    async fn run_agent_turn(
        &mut self,
        run_id: &str,
        turn: usize,
        events: &EventSink,
    ) -> Result<TurnControl, AgentError> {
        let Some(request) = self.prepare_request(events).await? else {
            events.emit(AgentEvent::run_paused(
                RunPauseReason::ContextExhausted,
                String::new(),
            ));
            return Ok(TurnControl::Break {
                final_text: String::new(),
                stop_reason: StopReason::NoToolCalls,
            });
        };
        let model_turn = self.run_turn_racing_compaction(request, events).await?;

        if model_turn.malformed_function_call {
            return Ok(TurnControl::Continue);
        }
        let response_text = model_turn.response_text;
        let tool_calls = model_turn.tool_calls;
        ensure_non_empty_tool_call_ids(&tool_calls, &response_text)?;

        self.commit_assistant_turn(&response_text, &tool_calls, model_turn.reasoning);
        if tool_calls.is_empty() {
            return self.finish_no_tool_turn(response_text, turn, events).await;
        }

        Self::push_tool_start_events(&tool_calls, events);
        self.execute_tool_calls(&tool_calls, turn, events).await;
        let polled = self.apply_tool_results(events)?;
        self.push_tool_turn_alerts().await;
        let control = self
            .push_turn_boundary_events(run_id, turn, &response_text, events)
            .await?;
        // Another request would only re-read the same output: park until the
        // background work reports instead. Every call already has its result,
        // so the transcript stays valid while the run waits.
        if polled == ToolTurn::OnlyUnchangedReads
            && matches!(control, TurnControl::Continue)
            && self
                .background_receiver
                .as_ref()
                .is_some_and(BackgroundTaskReceiver::has_running)
        {
            tracing::info!(
                turn,
                "every call re-read unchanged output; parking the run until background work reports"
            );
            return Ok(TurnControl::Break {
                final_text: response_text,
                stop_reason: StopReason::AwaitingBackground,
            });
        }
        Ok(control)
    }

    /// Runs one model turn while a forked compaction keeps running.
    ///
    /// The two are polled together instead of sequenced: compaction is a full
    /// request against the fast model, and waiting for it would stall the run
    /// at exactly the moment the window is tightest. Whichever finishes
    /// first is handled — a landed handoff is parked for the next request
    /// boundary, where replacing the conversation is safe.
    async fn run_turn_racing_compaction(
        &mut self,
        request: LLMRequest,
        events: &EventSink,
    ) -> Result<ModelTurn, AgentError> {
        let mut pending = self.compaction.take_running();
        let mut landed = None;

        let turn = {
            let turn = self.collect_model_turn(request, events);
            futures_lite::pin!(turn);
            loop {
                let step = match pending.as_mut() {
                    Some(compaction) => {
                        futures_lite::future::or(
                            async { TurnStep::Turn((&mut turn).await) },
                            async { TurnStep::Compacted(compaction.wait().await) },
                        )
                        .await
                    }
                    None => TurnStep::Turn((&mut turn).await),
                };
                match step {
                    TurnStep::Turn(turn) => break turn,
                    TurnStep::Compacted(result) => {
                        landed = Some(result);
                        pending = None;
                    }
                }
            }
        };

        match landed {
            Some(Ok(completed)) => self.compaction.land(completed),
            Some(Err(error)) => {
                tracing::warn!(%error, "forked compaction failed");
                self.compaction_failure = Some(error);
            }
            None => {}
        }
        self.compaction.resume(pending);
        turn
    }

    /// Stores what the model produced this turn: its text, calls and reasoning.
    ///
    /// The reasoning travels with the turn that produced it, or later requests
    /// carry the calls and the answer without the thinking behind them. A turn
    /// that produced nothing at all stores nothing.
    fn commit_assistant_turn(
        &mut self,
        response_text: &str,
        tool_calls: &[ToolCall],
        reasoning: Vec<ReasoningState>,
    ) {
        if response_text.is_empty() && tool_calls.is_empty() && reasoning.is_empty() {
            return;
        }
        self.context.push(Message::assistant_with_reasoning(
            response_text,
            tool_calls.to_vec(),
            reasoning,
        ));
    }

    async fn finish_no_tool_turn(
        &mut self,
        response_text: String,
        turn: usize,
        events: &EventSink,
    ) -> Result<TurnControl, AgentError> {
        if !response_text.is_empty() {
            #[cfg(feature = "languages")]
            self.enforce_reply_language(&response_text, events).await;
        }
        events.emit(AgentEvent::turn_complete(turn, false));
        if self.push_final_turn_alerts().await {
            // The alerts stay in the conversation either way; a host that has
            // a message waiting ends the run here so the message travels with
            // them in the next request instead of one request later.
            let boundary_ctx = TurnBoundaryContext {
                assistant_text: &response_text,
                turn,
                message_count: self.context.len_recent(),
            };
            if self.hooks.on_turn_boundary(&boundary_ctx).await == TurnBoundaryAction::EndTurn {
                return Ok(TurnControl::Break {
                    final_text: response_text,
                    stop_reason: StopReason::EndTurn,
                });
            }
            return Ok(TurnControl::Continue);
        }
        Ok(TurnControl::Break {
            final_text: response_text,
            stop_reason: StopReason::NoToolCalls,
        })
    }

    /// Holds a reply to the communication language.
    ///
    /// A reply in another language raises one alert telling the model to
    /// return to it, and the user is shown a translation by the fast model,
    /// marked by [`AgentEvent::ReplyTranslated`]. The history keeps the
    /// original, so the cached prefix is untouched. A translation that fails
    /// is logged and leaves the original on display, which the user has
    /// already been streamed.
    #[cfg(feature = "languages")]
    async fn enforce_reply_language(&mut self, reply: &str, events: &EventSink) {
        let Some(guard) = self.config.languages.clone() else {
            return;
        };
        let Some(language) = guard
            .languages()
            .communication()
            .filter(|_| guard.languages().enforced() && !guard.reply_in_language(reply))
        else {
            return;
        };
        let alert = self
            .notices
            .language_alerts(&[crate::language::LanguageAlert::Reply { language }]);
        self.push_system_alert(alert);
        match crate::language::translate(
            self.fast.as_ref(),
            self.parameters.get(ModelTier::Fast),
            reply,
            language,
        )
        .await
        {
            Ok(text) => events.emit(AgentEvent::ReplyTranslated {
                text,
                language: language.code().to_string(),
            }),
            Err(error) => {
                tracing::warn!(%error, %language, "failed to translate a reply for display");
            }
        }
    }

    /// Gives every tool call in the context a result.
    ///
    /// A run stream dropped while tools execute leaves the assistant message
    /// carrying the calls with no results behind it, and providers reject a
    /// request holding such calls. Runs at the start of the next run (and of a
    /// manual compaction), so the repair needs nothing from the host and only
    /// appends: the prefix, thinking blocks included, stays byte-identical.
    /// Calls whose results had already landed keep them; the rest get a
    /// cancelled result.
    fn settle_interrupted_tool_calls(&mut self) {
        let mut parked = std::mem::take(&mut self.tool_results);
        let mut unresolved: Vec<ToolCall> = Vec::new();
        for message in self.context.recent() {
            unresolved.extend(message.tool_calls().iter().cloned());
            if let Some(id) = message.tool_call_id() {
                unresolved.retain(|call| call.id != id);
            }
        }
        for call in unresolved {
            let finished = parked
                .iter()
                .position(|result| matches!(result, Ok(finished) if finished.id == call.id))
                .and_then(|index| parked.remove(index));
            let content = match finished {
                Some(Ok(finished)) => finished.result.render_for_model(),
                _ => self
                    .notices
                    .tool_call_cancelled(&crate::CancelledToolCall { tool: &call.name }),
            };
            self.context.push(Message::tool(&call.id, content));
        }
    }

    fn push_tool_start_events(tool_calls: &[ToolCall], events: &EventSink) {
        for call in tool_calls {
            let args = call.arguments.to_string();
            events.emit(AgentEvent::ToolCallStart {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: args,
            });
            if let Some(reason) = pause_reason_for_tool(call.name.as_str()) {
                events.emit(AgentEvent::run_paused(reason, call.id.clone()));
            }
        }
    }

    /// Runs `tool_calls` concurrently, parking each result in
    /// `self.tool_results` the moment it lands.
    ///
    /// The results live on the agent rather than in this future so a run
    /// stream dropped mid-execution still leaves the finished ones for
    /// [`Self::settle_interrupted_tool_calls`] to pair.
    async fn execute_tool_calls(
        &mut self,
        tool_calls: &[ToolCall],
        turn: usize,
        events: &EventSink,
    ) {
        if let Some(signal) = &self.compact_signal
            && tool_calls
                .iter()
                .any(|call| call.name != crate::compact_tool::COMPACT_TOOL_NAME)
        {
            signal.acted();
        }
        let permission_receiver = self.permission_receiver.clone();
        let mut permission_pause_tracker = PermissionPauseTracker::from_tool_calls(tool_calls);
        let message_count = self.context.len_recent();
        let (hooks, tools, agent) = (&self.hooks, &self.tools, &self.identity);
        let results = &mut self.tool_results;
        let tool_futures = tool_calls.iter().map(|call| {
            let args_json = call.arguments.to_string();
            let identity = aither_core::llm::CallIdentity::new(
                aither_core::llm::ToolCallId::new(call.id.clone()),
                agent.clone(),
            );
            async move {
                let tool_ctx = ToolUseContext {
                    call: &identity,
                    tool_name: &call.name,
                    arguments: &args_json,
                    turn,
                    message_count,
                };
                let (result, duration, unchanged_read) =
                    Self::call_tool_with_hooks(hooks, tools, call, &args_json, &tool_ctx).await?;
                let result_ctx = ToolResultContext {
                    tool_name: &call.name,
                    arguments: &args_json,
                    result: &result,
                    duration,
                };
                let tool_result = match hooks.post_tool_use(&result_ctx).await {
                    PostToolAction::Abort(reason) => {
                        return Err(AgentError::HookRejected {
                            hook: "post_tool_use",
                            reason,
                        });
                    }
                    PostToolAction::Replace(replacement) => replacement,
                    PostToolAction::Keep => result,
                };
                Ok(FinishedToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    result: tool_result,
                    unchanged_read,
                })
            }
        });

        let mut pending = tool_futures.collect::<futures::stream::FuturesUnordered<_>>();
        while !pending.is_empty() {
            let next = match &permission_receiver {
                Some(receiver) => {
                    futures_lite::future::or(
                        async { pending.next().await.map(ExecutionSignal::Tool) },
                        async { receiver.recv().await.map(ExecutionSignal::Permission) },
                    )
                    .await
                }
                None => pending.next().await.map(ExecutionSignal::Tool),
            };
            Self::handle_execution_signal(next, &mut permission_pause_tracker, results, events);
        }
        if let Some(receiver) = &permission_receiver {
            for event in receiver.take_pending() {
                emit_permission_event(&mut permission_pause_tracker, &event, events);
            }
        }
    }

    async fn call_tool_with_hooks(
        hooks: &H,
        tools: &AgentTools,
        call: &ToolCall,
        args_json: &str,
        tool_ctx: &ToolUseContext<'_>,
    ) -> Result<(aither_core::llm::ToolResult, Duration, bool), AgentError> {
        match hooks.pre_tool_use(tool_ctx).await {
            PreToolAction::Abort(reason) => Err(AgentError::HookRejected {
                hook: "pre_tool_use",
                reason,
            }),
            PreToolAction::Deny(reason) => Ok((
                aither_core::llm::ToolResult::error(reason),
                Duration::ZERO,
                false,
            )),
            PreToolAction::Allow => {
                let start = Instant::now();
                // The agent loop has nowhere to show a tool's progress, but
                // the tool learns which call, of which agent, it serves.
                let cx = aither_core::llm::ToolContext::new().with_identity(tool_ctx.call.clone());
                let (result, unchanged_read) = match tools.call(&call.name, args_json, cx).await {
                    Ok(result) => (result, false),
                    Err(error) => {
                        let unchanged_read = error
                            .downcast_ref::<aither_sandbox::Refused<aither_sandbox::PollRefusal>>()
                            .is_some_and(|refused| refused.reason().is_unchanged_read());
                        let mut message = String::from("Error: ");
                        message.push_str(error.to_string().as_str());
                        (aither_core::llm::ToolResult::error(message), unchanged_read)
                    }
                };
                Ok((result, start.elapsed(), unchanged_read))
            }
        }
    }

    fn handle_execution_signal(
        signal: Option<ToolExecutionSignal>,
        tracker: &mut PermissionPauseTracker,
        results: &mut VecDeque<ToolExecutionResult>,
        events: &EventSink,
    ) {
        match signal {
            Some(ExecutionSignal::Tool(result)) => results.push_back(result),
            Some(ExecutionSignal::Permission(event)) => {
                emit_permission_event(tracker, &event, events);
            }
            None => {}
        }
    }

    /// Moves the parked tool results into the context.
    ///
    /// Results not reached because one failed stay parked for
    /// [`Self::settle_interrupted_tool_calls`]. Returns whether the turn gave
    /// the model anything new.
    fn apply_tool_results(&mut self, events: &EventSink) -> Result<ToolTurn, AgentError> {
        if let Some(receiver) = &self.kept_receiver {
            self.kept_exchanges.extend(
                receiver
                    .take_pending()
                    .into_iter()
                    .map(|call| call.as_str().to_owned()),
            );
        }
        let mut has_tool_error = false;
        let mut unknown_tools: Vec<String> = Vec::new();
        let known_tools: Vec<String> = self
            .tools
            .active_definitions()
            .into_iter()
            .map(|definition| definition.name().to_string())
            .collect();
        let mut only_unchanged_reads = !self.tool_results.is_empty();
        while let Some(result) = self.tool_results.pop_front() {
            let FinishedToolCall {
                id: call_id,
                name: call_name,
                result: tool_result,
                unchanged_read,
            } = result?;
            only_unchanged_reads &= unchanged_read;
            let is_terminal_call = call_name == "terminal";
            if !known_tools.contains(&call_name) && !unknown_tools.contains(&call_name) {
                unknown_tools.push(call_name.clone());
            }
            events.emit(AgentEvent::ToolCallEnd {
                id: call_id.clone(),
                name: call_name.clone(),
                result: tool_result.clone(),
            });
            if let Some(reason) = pause_reason_for_tool(call_name.as_str()) {
                events.emit(AgentEvent::run_resumed(reason, call_id.clone()));
            }
            let content = tool_result.render_for_model();
            // A refused re-read is no failure to re-assess: its refusal
            // already says how to wait.
            has_tool_error |= !unchanged_read && tool_result_is_error(&tool_result, &content);
            self.push_terminal_followup_events(is_terminal_call, &tool_result, events);
            let message = Message::tool(&call_id, content);
            if call_name == crate::ask_user::TOOL_NAME || self.kept_exchanges.remove(&call_id) {
                self.context.push_kept(message);
            } else {
                self.context.push(message);
            }
        }
        let has_terminal_tool = known_tools.iter().any(|name| name == "terminal");
        if has_terminal_tool && !unknown_tools.is_empty() {
            // Terminal-first architecture: non-native capabilities are CLI
            // commands, and models trained on native tool calling routinely
            // reach for them as tools. Redirect immediately instead of letting
            // the model retry the same wrong call.
            for name in &unknown_tools {
                self.push_system_alert(format!(
                    "There is no native tool named '{name}'. This runtime is terminal-first: the only native tools are terminal, terminal_kill, terminal_input, and terminal_read; every other capability is a CLI command executed through the `terminal` tool. If '{name}' is a capability, run it as a command (check with `{name} --help`)."
                ));
            }
        } else if has_tool_error {
            self.push_system_alert(
                "A tool call failed. Re-assess the current state, inspect the latest tool result carefully, and choose the next action deliberately. Native tools remain terminal, terminal_kill, terminal_input, and terminal_read.",
            );
        }
        let background_news = self.push_completed_background_events(events);
        Ok(if only_unchanged_reads && !background_news {
            ToolTurn::OnlyUnchangedReads
        } else {
            ToolTurn::Progress
        })
    }

    fn push_terminal_followup_events(
        &mut self,
        is_terminal_call: bool,
        tool_result: &aither_core::llm::ToolResult,
        events: &EventSink,
    ) {
        if !is_terminal_call || tool_result.is_error() {
            return;
        }
        let Some(receiver) = self.background_receiver.clone() else {
            return;
        };
        for started in receiver.take_started() {
            self.detached_task_ids.insert(started.task_id.clone());
            let reminder = self.notices.background_started(&started);
            self.context.push(Message::system(&reminder));
            events.emit(AgentEvent::background_task_started(
                started.task_id,
                started.output_preview,
                started.output_file.url,
            ));
        }
    }

    /// Records queued background completions and alerts; returns whether
    /// any of them is news the model must act on — an announced completion
    /// or an alert that would wake a parked wait.
    fn push_completed_background_events(&mut self, events: &EventSink) -> bool {
        let Some(receiver) = self.background_receiver.clone() else {
            return false;
        };
        let mut news = self.push_job_alerts();
        for task in receiver.take_completed() {
            tracing::info!(task_id = %task.task_id, "background task completed");
            news |= self.push_completed_task_event(task, events);
        }
        news
    }

    async fn push_turn_boundary_events(
        &self,
        run_id: &str,
        turn: usize,
        response_text: &str,
        events: &EventSink,
    ) -> Result<TurnControl, AgentError> {
        let boundary_ctx = TurnBoundaryContext {
            assistant_text: response_text,
            turn,
            message_count: self.context.len_recent(),
        };
        let checkpoint = self
            .emit_checkpoint(CheckpointReason::TurnBoundary, response_text, turn)
            .await?;
        events.emit(AgentEvent::checkpoint(
            run_id.to_string(),
            CheckpointReason::TurnBoundary,
            turn,
            checkpoint.phase,
            checkpoint.message_count,
        ));
        events.emit(AgentEvent::turn_complete(turn, true));
        if self.hooks.on_turn_boundary(&boundary_ctx).await == TurnBoundaryAction::EndTurn {
            Ok(TurnControl::Break {
                final_text: response_text.to_string(),
                stop_reason: StopReason::EndTurn,
            })
        } else {
            Ok(TurnControl::Continue)
        }
    }

    async fn collect_final_background_events(
        &mut self,
        run_id: &str,
        iteration: usize,
        events: &EventSink,
    ) -> Result<bool, AgentError> {
        let Some(receiver) = self.background_receiver.clone() else {
            return Ok(false);
        };
        let mut iteration = iteration;
        // The outcome of the last loop this wait resumed; the run ends with
        // its stop events once the wait is over.
        let mut resumed: Option<RunLoopOutcome> = None;
        loop {
            if !self.wait_for_background_events(&receiver, events).await? {
                break;
            }
            let outcome = self.run_agent_loop(run_id, iteration, events).await?;
            iteration = outcome.iteration;
            let hook_ended = outcome.stop_reason == StopReason::EndTurn;
            resumed = Some(outcome);
            // A model turn that answered an alert goes back to waiting: a job
            // may still run, or may have finished while the model replied and
            // left its completion queued, which the wait collects first. A
            // turn a hook ended stays ended.
            if hook_ended {
                break;
            }
        }
        let Some(outcome) = resumed else {
            return Ok(false);
        };
        self.push_stop_events(run_id, outcome, events).await?;
        Ok(true)
    }

    /// Collects the run's background events; returns whether the model has
    /// something new to see.
    ///
    /// Under [`BackgroundJobPolicy::Await`] it parks until the background jobs
    /// are done or a job alert that needs the model arrives; each job's own
    /// timeout bounds the wait. Under [`BackgroundJobPolicy::Detach`] it only
    /// takes what is already queued.
    async fn wait_for_background_events(
        &mut self,
        receiver: &BackgroundTaskReceiver,
        events: &EventSink,
    ) -> Result<bool, AgentError> {
        let (mut needs_model, taken) = self.drain_background_receiver(receiver, events);
        // The run itself recorded them: the agent has taken them.
        drop(taken);
        loop {
            // A job blocked on stdin can never complete, so parking on it
            // would hide its alert from the model for the whole wait. Other
            // alerts become durable context without ending the wait.
            if self.push_job_alerts() {
                return Ok(true);
            }
            if self.config.background_jobs == BackgroundJobPolicy::Detach {
                break;
            }
            self.maybe_send_cache_keep_alive().await?;
            match receiver
                .recv_event_timeout(BACKGROUND_TASK_POLL_INTERVAL)
                .await
            {
                Some(BackgroundTaskEvent::Completed(task)) => {
                    tracing::info!(task_id = %task.task_id, "background task completed (waiting)");
                    needs_model |= self.push_completed_task_event(*task, events);
                }
                Some(BackgroundTaskEvent::Alert(alert)) => {
                    if self.push_job_alert(&alert) {
                        return Ok(true);
                    }
                }
                None if !receiver.has_running() => break,
                None => {}
            }
        }
        Ok(needs_model)
    }

    /// Sends one prompt-cache keep-alive when the wait on background tasks
    /// is about to outlive the conversation's cache entry.
    ///
    /// Fires only while the run is parked in
    /// [`collect_final_background_events`](Self::collect_final_background_events),
    /// and only when [`AgentConfig::cache_keep_alive`](crate::config::AgentConfig::cache_keep_alive)
    /// is set; the caller sets it only for providers whose cache expires.
    /// The keep-alive re-sends the run's last request with `max_tokens: 0`
    /// and non-streaming, keeping the same thinking and effort settings,
    /// shortly before the entry's [`CacheLifetime`] — counted from the
    /// start of the last request, or of the previous keep-alive — expires.
    /// A [`CacheLifetime::Permanent`] cache never gets one, and there is no
    /// cap on sends: they repeat for as long as the run waits.
    ///
    /// [`CacheLifetime`]: crate::context::CacheLifetime
    ///
    /// # Errors
    ///
    /// Returns the provider error when the keep-alive request fails.
    async fn maybe_send_cache_keep_alive(&mut self) -> Result<(), AgentError> {
        let Some(keep_alive) = self.config.cache_keep_alive.clone() else {
            return Ok(());
        };
        // A cold entry is past saving by a keep-alive; a permanent one needs none.
        let Some(cold_in) = self.warmth.cold_in() else {
            return Ok(());
        };
        if cold_in > keep_alive.send_before_expiry {
            return Ok(());
        }
        let Some(last_request) = self.last_request.clone() else {
            return Ok(());
        };
        let (messages, parameters, tool_definitions) = last_request.into_parts();
        let request = LLMRequest::new(messages)
            .with_tool_definitions(tool_definitions)
            .with_parameters(parameters.max_tokens(0).stream(false));
        let stream = Self::respond_stream(
            self.tier,
            &self.advanced,
            &self.balanced,
            &self.fast,
            &self.connectivity,
            request,
        );
        futures_lite::pin!(stream);
        while let Some(event) = stream.next().await {
            match event {
                Ok(Event::Usage(usage)) => self.cache_stats.record(&usage),
                Ok(_) => {}
                Err(error) => return Err(error),
            }
        }
        self.warmth.start_entry_now();
        self.cache_keep_alive.sent += 1;
        self.cache_keep_alive.last_reason = self.format_wait_reason().await;
        if self.cache_keep_alive.sent >= keep_alive.frequent_idle_threshold {
            self.cache_keep_alive.alerted = true;
        }
        Ok(())
    }

    /// Describes the background commands this run is currently parked on.
    async fn format_wait_reason(&self) -> Option<String> {
        let job_registry = self.job_registry.as_ref()?;
        let running: Vec<String> = job_registry
            .list()
            .await
            .into_iter()
            .filter(|job| job.status == aither_sandbox::JobStatus::Running)
            .map(|job| format!("'{}' ({})", job.script, job.task_id))
            .collect();
        if running.is_empty() {
            return None;
        }
        Some(format!(
            "waiting on background command {}",
            running.join(", ")
        ))
    }

    /// A shared handle on whether the provider's prompt-cache entry for this
    /// agent's context is warm and how long until it goes cold. It keeps
    /// tracking this agent's later requests and keep-alives.
    #[must_use]
    pub fn cache_warmth(&self) -> crate::CacheWarmth {
        self.warmth.clone()
    }

    /// Pretends the cache entry was started `age` ago.
    ///
    /// The system clock counts from boot, so a freshly booted machine cannot
    /// stand `age` back; the entry moves to a manual clock standing `age`
    /// past its origin, with the entry at the origin.
    #[cfg(test)]
    pub(crate) fn backdate_cache_entry(&mut self, age: Duration) {
        let clock = crate::ManualClock::new();
        clock.advance(age);
        self.warmth =
            crate::CacheWarmth::new(self.prompt_cache.lifetime(), crate::AnyClock::new(clock));
        self.warmth
            .start_entry(crate::clock::Instant::from_nanos(0));
    }

    /// Swaps the cache lifetime, forgetting any started entry.
    #[cfg(test)]
    pub(crate) fn use_cache_lifetime(&mut self, lifetime: CacheLifetime) {
        self.prompt_cache = self.prompt_cache.clone().with_lifetime(lifetime);
        self.warmth = self.warmth.with_lifetime(lifetime);
    }

    /// How long the provider's prompt-cache entry for this agent's context
    /// stays warm without a hit.
    #[must_use]
    pub const fn cache_lifetime(&self) -> CacheLifetime {
        self.prompt_cache.lifetime()
    }

    /// The session's communication and working languages, when the agent
    /// enforces any. An application that writes to the user on the agent's
    /// behalf, such as a recap or a prompt suggestion, reads the
    /// communication language here.
    #[cfg(feature = "languages")]
    #[must_use]
    pub const fn languages(&self) -> Option<&crate::LanguageGuard> {
        self.config.languages.as_ref()
    }

    /// Keep-alive bookkeeping for the current run.
    pub const fn cache_keep_alive_state(&self) -> &CacheKeepAliveState {
        &self.cache_keep_alive
    }

    /// Pushes one durable alert per pending background job alert: a job past
    /// the long-running threshold, one blocked reading its stdin, or one
    /// whose stdin watch failed. Returns whether any of them is a job blocked
    /// on stdin, which the model must act on.
    fn push_job_alerts(&mut self) -> bool {
        self.push_host_load_alerts();
        let Some(receiver) = self.background_receiver.clone() else {
            return false;
        };
        let mut wake = false;
        for alert in receiver.take_job_alerts() {
            wake |= self.push_job_alert(&alert);
        }
        wake
    }

    /// Pushes one durable alert per host load event raised since the last
    /// call: a condition starting or ending. An event leaves the channel as
    /// it is pushed, so the model sees it once.
    fn push_host_load_alerts(&mut self) {
        let Some(receiver) = self.host_load_events.clone() else {
            return;
        };
        while let Ok(event) = receiver.try_recv() {
            let alert = self.notices.host_load(&event);
            self.push_system_alert(alert);
        }
    }

    /// Pushes one durable alert for a background job alert; returns whether
    /// it is a job blocked on stdin, which wakes a parked wait.
    fn push_job_alert(&mut self, alert: &JobAlert) -> bool {
        let wakes = match alert {
            JobAlert::DescendantsKilled(killed) => {
                tracing::info!(
                    killed = killed.processes.len() + killed.omitted,
                    "foreground command left processes running; killed"
                );
                false
            }
            JobAlert::DirectoryReset(reset) => {
                tracing::info!(dir = %reset.dir.display(), reason = ?reset.reason, "carried working directory reset");
                false
            }
            JobAlert::LongRunning(job) => {
                tracing::info!(task_id = %job.task_id, "background job crossed long-running threshold");
                false
            }
            JobAlert::StdinBlocked(job) => {
                tracing::info!(task_id = %job.task_id, "background job is blocked on stdin");
                true
            }
            JobAlert::StdinProbeFailed(job) => {
                tracing::warn!(task_id = %job.task_id, error = %job.error, "stdin watch failed");
                false
            }
            JobAlert::MonitorIncrement(increment) => {
                tracing::info!(task_id = %increment.task_id, lines = increment.lines.len(), "monitor increment delivered");
                true
            }
            JobAlert::MonitorStalled(stalled) => {
                tracing::info!(task_id = %stalled.task_id, "monitor reports a stall");
                true
            }
            JobAlert::MonitorEnded(ended) => {
                tracing::info!(task_id = %ended.task_id, "monitor ended");
                true
            }
        };
        let text = self.notices.job_alert(alert);
        self.push_system_alert(text);
        wakes
    }

    /// Records every queued completion; returns whether any was announced,
    /// with the completions still pending in their registries. Dropping
    /// those is the agent taking them.
    fn drain_background_receiver(
        &mut self,
        receiver: &BackgroundTaskReceiver,
        events: &EventSink,
    ) -> (bool, Vec<aither_sandbox::PendingCompletion>) {
        let mut announced = false;
        let mut pending = Vec::new();
        for mut task in receiver.take_completed() {
            tracing::info!(task_id = %task.task_id, "background task completed (final check)");
            pending.extend(task.pending.take());
            announced |= self.push_completed_task_event(task, events);
        }
        (announced, pending)
    }

    /// Records a completed job in the context; returns whether it was
    /// announced, which a job the model knows as a background task is.
    fn push_completed_task_event(
        &mut self,
        task: aither_sandbox::CompletedTask,
        events: &EventSink,
    ) -> bool {
        // Every spawned job reports its completion, foreground ones
        // included; only a job the model knows as a background task is
        // announced, since a foreground result already returned through the
        // tool call.
        if !self.detached_task_ids.contains(&task.task_id) {
            return false;
        }
        let result_msg = self.notices.background_finished(&task);
        self.context.push(Message::system(&result_msg));
        self.tasks
            .events
            .record(crate::tasks::TaskEvent::JobFinished(task.task_id.clone()));
        events.emit(AgentEvent::background_task_completed(
            task.task_id,
            result_msg,
        ));
        true
    }

    async fn push_stop_events(
        &self,
        run_id: &str,
        outcome: RunLoopOutcome,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let checkpoint = self
            .emit_checkpoint(
                CheckpointReason::Stop,
                &outcome.final_text,
                outcome.iteration,
            )
            .await?;
        events.emit(AgentEvent::checkpoint(
            run_id.to_string(),
            CheckpointReason::Stop,
            outcome.iteration,
            checkpoint.phase,
            checkpoint.message_count,
        ));
        let stop_ctx = StopContext {
            final_text: &outcome.final_text,
            turns: outcome.iteration,
            reason: outcome.stop_reason,
        };
        self.hooks
            .on_lifecycle(&external_hook::Stop {
                final_text: outcome.final_text.clone(),
                turns: outcome.iteration,
                reason: outcome.stop_reason,
            })
            .await;
        if let Some(reason) = self.hooks.on_stop(&stop_ctx).await {
            return Err(AgentError::HookRejected {
                hook: "on_stop",
                reason,
            });
        }
        events.emit(AgentEvent::Complete {
            final_text: outcome.final_text,
            turns: outcome.iteration,
        });
        Ok(())
    }

    /// Performs a one-shot query and returns the final response.
    ///
    /// This is the simplest way to use the agent. The agent handles tool
    /// execution internally and returns the final text response.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLM returns an error
    /// - A hook aborts the operation
    /// - Tool execution fails
    pub async fn query(&mut self, prompt: &str) -> Result<String, AgentError> {
        let flow = self
            .run(prompt, std::iter::empty())
            .drive(
                String::new(),
                std::future::pending(),
                |mut text, timed| async move {
                    match timed.event {
                        AgentEvent::Text(chunk) => text.push_str(&chunk),
                        AgentEvent::Complete { final_text, .. } => {
                            return ControlFlow::Break(final_text);
                        }
                        _ => {}
                    }
                    ControlFlow::Continue(text)
                },
            )
            .await?;
        Ok(match flow {
            ControlFlow::Break(text) | ControlFlow::Continue(text) => text,
        })
    }

    /// Starts a run on `prompt`; [`Run::drive`] runs it and hands each event
    /// to the caller as it happens, text chunks as they arrive from the LLM.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::ops::ControlFlow;
    ///
    /// let mut shown = String::new();
    /// agent
    ///     .run("Implement the feature", std::iter::empty())
    ///     .drive(&mut shown, std::future::pending(), |shown, timed| async move {
    ///         if let AgentEvent::Text(text) = timed.event {
    ///             shown.push_str(&text);
    ///         }
    ///         ControlFlow::<(), _>::Continue(shown)
    ///     })
    ///     .await?;
    /// ```
    pub fn run(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> Run<impl Future<Output = Result<(), AgentError>> + '_> {
        let prompt = prompt.to_string();
        let attachments: Vec<Attachment> = attachments.into_iter().collect();
        Run::new(move |sink| async move { self.run_streaming(prompt, attachments, &sink).await })
    }

    /// Continues the conversation without a new user message, so the model
    /// acts on the background events [`wait_background`](Self::wait_background)
    /// recorded. Driven like [`run`](Self::run).
    pub fn resume(&mut self) -> Run<impl Future<Output = Result<(), AgentError>> + '_> {
        Run::new(move |sink| async move { self.resume_streaming(&sink).await })
    }

    /// Waits, between runs, for a background command of an earlier run to
    /// complete or raise an alert, and records it in the context.
    ///
    /// For an agent built with [`BackgroundJobPolicy::Detach`]: its runs end
    /// with the model's turn, and the host races this against its own work
    /// while the agent is idle. When the returned
    /// [`BackgroundWake::needs_model`] is set, the host continues with
    /// [`resume`](Self::resume). The wait is cancel-safe: nothing is taken
    /// until it completes. Without a terminal it never completes.
    ///
    /// The completions it recorded stay pending in their job registries
    /// until the host drops the wake's [`BackgroundWake::pending`]: the host
    /// takes them on the agent's behalf, once its own state shows the turn
    /// they start.
    pub async fn wait_background(&mut self) -> BackgroundWake {
        let Some(receiver) = self.background_receiver.clone() else {
            return std::future::pending().await;
        };
        let event = receiver.recv_event().await;
        let (sender, emitted) = async_channel::unbounded();
        let sink = EventSink { sender };
        let mut pending = Vec::new();
        let mut needs_model = match event {
            BackgroundTaskEvent::Completed(mut task) => {
                tracing::info!(task_id = %task.task_id, "background task completed (idle)");
                pending.extend(task.pending.take());
                self.push_completed_task_event(*task, &sink)
            }
            BackgroundTaskEvent::Alert(alert) => self.push_job_alert(&alert),
        };
        let (announced, drained) = self.drain_background_receiver(&receiver, &sink);
        needs_model |= announced;
        pending.extend(drained);
        needs_model |= self.push_job_alerts();
        drop(sink);
        let mut events = Vec::new();
        while let Ok(event) = emitted.try_recv() {
            events.push(event);
        }
        BackgroundWake {
            events,
            needs_model,
            pending,
        }
    }

    /// Registers a tool for the agent to use.
    ///
    /// # Errors
    ///
    /// Returns [`RegisterError`](aither_core::llm::tool::RegisterError) if the
    /// name collides with a registered tool, or the tool has no description.
    pub fn register_tool<T: aither_core::llm::Tool + 'static>(
        &mut self,
        tool: T,
    ) -> Result<(), aither_core::llm::tool::RegisterError> {
        self.tools.register(tool)
    }

    /// Returns a reference to the unified context manager.
    #[must_use]
    pub const fn context(&self) -> &Context {
        &self.context
    }

    /// Returns a mutable reference to the unified context manager.
    ///
    /// Use this to insert/update system blocks, push messages, etc.
    pub const fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Adds a message to the conversation history.
    pub fn push_message(&mut self, message: Message) {
        self.context.push(message);
    }

    /// Returns the rolling KV-cache statistics for this agent session.
    ///
    /// The stats are updated every time the underlying provider emits a
    /// `Usage` event (which, for providers like Claude with prompt
    /// caching, includes `cache_read_tokens` and `cache_write_tokens`).
    /// Hosts can read the hit rate, reset the accumulator, or snapshot
    /// it for per-turn diffs.
    #[must_use]
    pub const fn cache_stats(&self) -> &crate::CacheStats {
        &self.cache_stats
    }

    /// Returns a mutable reference to the KV-cache statistics accumulator.
    ///
    /// Hosts typically only use this to [`CacheStats::reset`] between
    /// distinct sessions, since the agent itself records new observations.
    ///
    /// [`CacheStats::reset`]: crate::CacheStats::reset
    pub const fn cache_stats_mut(&mut self) -> &mut crate::CacheStats {
        &mut self.cache_stats
    }

    /// Stops every terminal task that is still running for this agent.
    ///
    /// Returns the number of tasks that received a stop request.
    pub async fn kill_running_terminal_jobs(&self) -> usize {
        match &self.job_registry {
            Some(job_registry) => job_registry.kill_running().await,
            None => 0,
        }
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.compaction.retire();
        self.context.clear_history();
    }

    /// Compacts the conversation by generating a structured handoff and starting fresh.
    ///
    /// # Errors
    ///
    /// Returns an error if handoff generation fails.
    pub async fn compact(
        &mut self,
        focus: Option<&str>,
    ) -> Result<Option<CompactResult>, AgentError> {
        self.ensure_initialized().await?;
        self.settle_interrupted_tool_calls();

        let messages = self.context.conversation_messages();
        if messages.is_empty() {
            return Ok(None);
        }

        let messages_compacted = messages.len();
        // This handoff describes the whole conversation as it stands, so a
        // compaction forked earlier, running or landed, is stale from here.
        self.compaction.retire();
        self.hooks
            .on_lifecycle(&external_hook::BeforeCompaction {
                messages: messages_compacted,
                focus: focus.map(str::to_string),
            })
            .await;
        let tail = self.render_tail().await;
        let definitions = self.tools.active_definitions();
        let messages = self.request_messages(&tail);
        let request = self.compaction_request(messages, definitions);
        #[cfg(feature = "skills")]
        let skills = self.skill_catalog();
        #[cfg(not(feature = "skills"))]
        let skills = Vec::new();
        let fork = CompactionFork::<Schema>::new(
            &request,
            self.context.fork_point(),
            self.context.handoff_file_index(),
            focus,
            &skills,
        )
        .map_err(AgentError::from)?
        .with_outputs_dir(self.outputs_dir());
        let completed = self
            .compaction_future(fork)
            .await
            .map_err(AgentError::from)?;
        let summary = completed.handoff.text().to_string();

        self.compaction.land(completed);
        self.apply_completed_compaction().await;

        Ok(Some(CompactResult {
            messages_compacted,
            messages_remaining: self.context.len_recent(),
            summary,
        }))
    }

    /// The parameters of the requests the agent sends on its own tier's model.
    fn own_parameters(&self) -> &Parameters {
        self.parameters.get(self.tier)
    }

    /// The parameters of the requests sent on `tier`'s model.
    #[must_use]
    pub fn parameters(&self, tier: ModelTier) -> &Parameters {
        self.parameters.get(tier)
    }

    /// Changes the parameters of the requests sent on `tier`'s model, from
    /// the next request on.
    ///
    /// The advanced tier's parameters are also those of every tier still on
    /// the model the agent was created with, as
    /// [`AgentOptions::request_parameters`](crate::AgentOptions::request_parameters)
    /// sets them; a balanced or fast tier gets parameters of its own.
    ///
    /// A reasoning effort change on the agent's own tier keeps the prompt
    /// cache warm when the model takes per-message effort updates
    /// ([`Ability::PerMessageEffort`](aither_core::llm::model::Ability::PerMessageEffort)):
    /// the next request carries the update inside the conversation, before
    /// the next user message when one starts the turn, and the request-level
    /// effort stays as the first request sent it. Clearing the effort there
    /// is an update to the model's documented default
    /// ([`Profile::default_reasoning_effort`](aither_core::llm::model::Profile::default_reasoning_effort));
    /// with none recorded, the next turn fails with [`AgentError::Config`]
    /// rather than leave the earlier level in force. Any other model, and any
    /// other parameter, takes the new value at the request level, which
    /// restarts the cache.
    pub fn set_parameters(&mut self, tier: ModelTier, parameters: Parameters) {
        match tier {
            ModelTier::Advanced => self.parameters.set_shared(parameters),
            ModelTier::Balanced => self.parameters.set_balanced(parameters),
            ModelTier::Fast => self.parameters.set_fast(parameters),
        }
    }

    /// Replaces the advanced tier's model.
    ///
    /// The next request on the tier goes to `model`; history, context and
    /// background jobs stay. A new model reads none of the old one's prompt
    /// cache, so the change applies at once. A compaction already running
    /// finishes on the model it started on.
    pub fn set_advanced_model(&mut self, model: Advanced) {
        self.advanced = Arc::new(model);
        self.replaced_model(ModelTier::Advanced);
    }

    /// Replaces the balanced tier's model, as
    /// [`set_advanced_model`](Self::set_advanced_model) does the advanced
    /// tier's.
    pub fn set_balanced_model(&mut self, model: Balanced) {
        self.balanced = Arc::new(model);
        self.replaced_model(ModelTier::Balanced);
    }

    /// Replaces the fast tier's model, as
    /// [`set_advanced_model`](Self::set_advanced_model) does the advanced
    /// tier's.
    pub fn set_fast_model(&mut self, model: Fast) {
        self.fast = Arc::new(model);
        self.replaced_model(ModelTier::Fast);
    }

    /// Forgets what belonged to the replaced model of `tier` when it is the
    /// agent's own: its profile, fetched again before the next request, and
    /// the last request, which a keep-alive would otherwise send it.
    fn replaced_model(&mut self, tier: ModelTier) {
        if tier == self.tier {
            self.profile = None;
            self.last_request = None;
        }
    }

    /// Whether the agent's own model takes reasoning effort changes inside
    /// the conversation.
    fn takes_per_message_effort(&self) -> bool {
        self.profile
            .as_ref()
            .is_some_and(|profile| profile.abilities.contains(&Ability::PerMessageEffort))
    }

    /// Whether the agent's own model takes a system message inside the
    /// conversation, which decides whether a provider that cannot keeps the
    /// live tail out of its system prompt.
    fn takes_mid_conversation_system(&self) -> bool {
        Self::profile_takes_mid_conversation_system(self.profile.as_ref())
    }

    /// Whether the model a compaction request goes to takes a system message
    /// inside the conversation: the compaction model when one is set, else the
    /// agent's own.
    ///
    /// A model's client decides this from the same ability its profile
    /// reports, so the cache controls and the client cannot disagree.
    ///
    /// # Panics
    ///
    /// Panics when a compaction model is set and its profile is not loaded,
    /// which initialization does before any request.
    fn compaction_takes_mid_conversation_system(&self) -> bool {
        if self.compaction_model.is_none() {
            return self.takes_mid_conversation_system();
        }
        let profile = self
            .compaction_profile
            .as_ref()
            .expect("an initialized agent has its compaction model's profile");
        Self::profile_takes_mid_conversation_system(Some(profile))
    }

    fn profile_takes_mid_conversation_system(profile: Option<&ModelProfile>) -> bool {
        profile.is_some_and(|profile| profile.abilities.contains(&Ability::MidConversationSystem))
    }

    /// Brings the effort in force to the one the agent's own tier wants:
    /// appends a per-message update when its model takes one, clearing the
    /// effort to the model's default level, and otherwise settles the
    /// request-level effort.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Config`] when the effort is cleared on a model
    /// that takes per-message updates but whose default effort is not
    /// recorded, while another level is in force: no update can undo it.
    ///
    /// # Panics
    ///
    /// Panics before the agent is initialized, since only the model's profile
    /// says how an effort change is sent.
    fn settle_effort(&mut self) -> Result<(), AgentError> {
        let profile = self
            .profile
            .as_ref()
            .expect("the effort is settled only by an initialized agent, which has its profile");
        let changes = if profile.abilities.contains(&Ability::PerMessageEffort) {
            crate::effort::EffortChanges::PerMessage {
                default: profile.default_reasoning_effort,
            }
        } else {
            crate::effort::EffortChanges::RequestLevel
        };
        let update = self
            .effort
            .plan(
                self.own_parameters().reasoning_effort,
                changes,
                self.context.configured_effort(),
            )
            .map_err(|crate::effort::UnknownDefaultEffort| {
                AgentError::Config(format!(
                    "the reasoning effort was cleared, but model '{}' takes effort changes \
                     inside the conversation and its default effort is not recorded, so no \
                     update can undo the level in force; set an explicit effort",
                    profile.name
                ))
            })?;
        if let Some(effort) = update {
            self.context.push(Message::configuration(effort));
        }
        Ok(())
    }

    /// The parameters of the agent's own requests as they are sent: its
    /// tier's, with the request-level effort [`settle_effort`](Self::settle_effort)
    /// settled on.
    fn request_parameters(&self) -> Parameters {
        let mut parameters = self.own_parameters().clone();
        parameters.reasoning_effort = self.effort.request_effort(parameters.reasoning_effort);
        parameters
    }

    /// The window as a request to the agent's own model carries it, with the
    /// live `tail`: its effort updates are left out when the model cannot
    /// read them.
    fn request_messages(&self, tail: &[Block]) -> Vec<Message> {
        let mut messages = self.context.build_request(tail);
        if !self.takes_per_message_effort() {
            messages.retain(|message| message.role() != Role::Configuration);
        }
        messages
    }

    /// The session's outputs directory, where saved command outputs are
    /// written, when the agent has a sandbox.
    fn outputs_dir(&self) -> Option<PathBuf> {
        self.sandbox_dir.as_ref().map(|dir| dir.join("outputs"))
    }

    /// A compaction request over `messages` and `definitions`: the
    /// compaction model's parameters when one is set, else the session
    /// model's, under the cache controls the requests that wrote the prefix
    /// carry, so a compaction on the session's own model is served from that
    /// cache.
    ///
    /// `messages` is the window as the session model's requests carry it. A
    /// compaction model runs at its own effort, so the session's effort
    /// updates are left out of its request.
    fn compaction_request(
        &self,
        mut messages: Vec<Message>,
        definitions: Vec<aither_core::llm::tool::ToolDefinition>,
    ) -> LLMRequest {
        if self.compaction_model.is_some() {
            messages.retain(|message| message.role() != Role::Configuration);
        }
        let parameters = self.compaction_model.as_ref().map_or_else(
            || self.request_parameters(),
            |compactor| (*compactor.parameters).clone(),
        );
        let parameters = self.prompt_cache.apply(
            parameters,
            &messages,
            !definitions.is_empty(),
            self.compaction_takes_mid_conversation_system(),
        );
        LLMRequest::new(messages)
            .with_parameters(parameters)
            .with_tool_definitions(definitions)
    }

    /// Runs `fork` to completion on the compaction model: the configured one,
    /// or else the session's own tier.
    async fn compaction_future(
        &self,
        fork: CompactionFork<Schema>,
    ) -> Result<CompletedCompaction, crate::context::CompactionError> {
        if let Some(compactor) = &self.compaction_model {
            return fork.run(Arc::clone(&compactor.model)).await;
        }
        match self.tier {
            ModelTier::Advanced => fork.run(Arc::clone(&self.advanced)).await,
            ModelTier::Balanced => fork.run(Arc::clone(&self.balanced)).await,
            ModelTier::Fast => fork.run(Arc::clone(&self.fast)).await,
        }
    }

    /// Starts `fork` in the background on the compaction model: the
    /// configured one, or else the session's own tier.
    fn start_pending_compaction(&self, fork: CompactionFork<Schema>) -> PendingCompaction {
        if let Some(compactor) = &self.compaction_model {
            return PendingCompaction::start(fork, Arc::clone(&compactor.model));
        }
        match self.tier {
            ModelTier::Advanced => PendingCompaction::start(fork, Arc::clone(&self.advanced)),
            ModelTier::Balanced => PendingCompaction::start(fork, Arc::clone(&self.balanced)),
            ModelTier::Fast => PendingCompaction::start(fork, Arc::clone(&self.fast)),
        }
    }

    /// How long from now the idle compaction is due: `Some(Duration::ZERO)`
    /// when it is due now, `None` when none is pending.
    ///
    /// The window is compacted once, shortly before the conversation's cache
    /// lifetime ends ([`ContextPolicy::idle_compact_before_expiry`] ahead of
    /// it), counted from the entry's last use, so the compaction reads the
    /// still-warm cache. None is pending when:
    ///
    /// - a compaction model is set, which cannot read that cache;
    /// - the lifetime is [`CacheLifetime::Permanent`];
    /// - nothing has been sent yet, or the conversation is empty;
    /// - the context holds fewer tokens than the policy's
    ///   [`compaction_floor`](crate::ContextPolicy::compaction_floor);
    /// - it already ran since the last request;
    /// - the cache has expired already, when compacting would only pay to
    ///   write the whole context again.
    ///
    /// The agent has no timer of its own: the host sleeps this long while the
    /// session is idle, then calls [`compact_when_idle`](Self::compact_when_idle).
    ///
    /// [`ContextPolicy::idle_compact_before_expiry`]: crate::ContextPolicy::idle_compact_before_expiry
    #[must_use]
    pub fn idle_compaction_due_in(&self) -> Option<Duration> {
        if self.compaction_model.is_some()
            || self.idle_compacted
            || self.compaction.is_active()
            || self.context.is_conversation_empty()
            || self
                .config
                .context_policy
                .below_floor(self.budget.estimated_tokens())
        {
            return None;
        }
        let cold_in = self.warmth.cold_in()?;
        Some(cold_in.saturating_sub(self.config.context_policy.idle_compact_before_expiry))
    }

    /// Compacts the idle session once its cache is about to expire.
    ///
    /// Runs one compaction through the normal fork and apply path when
    /// [`idle_compaction_due_in`](Self::idle_compaction_due_in) says it is due
    /// now, and returns `None` otherwise, so a host may call it whenever its
    /// idle timer fires. One idle gap compacts once: the next request re-arms
    /// it. Nothing is kept across the handoff, so no thinking produced under
    /// the old prefix is replayed after it.
    ///
    /// # Errors
    ///
    /// Returns the compaction failure; the window is left as it was.
    pub async fn compact_when_idle(&mut self) -> Result<Option<CompactResult>, AgentError> {
        if self.idle_compaction_due_in() != Some(Duration::ZERO) {
            return Ok(None);
        }
        let result = self.compact(None).await?;
        self.idle_compacted = true;
        if let Some(result) = &result {
            tracing::info!(
                compacted = result.messages_compacted,
                "compacted the idle context shortly before its cache expires"
            );
        }
        Ok(result)
    }

    /// Forks a compaction off `request` to run alongside the turns that follow.
    ///
    /// The compaction replays the session's current model unless a compaction
    /// model is set: the handoff has to carry the same understanding of the
    /// conversation the next requests are made of, and only that model reads
    /// the cache the run just paid for.
    ///
    /// A failure to even start — an empty conversation — is not an error for
    /// the run: there is simply nothing to compact yet.
    fn start_compaction(&mut self, request: &LLMRequest, focus: Option<&str>) {
        #[cfg(feature = "skills")]
        let skills = self.skill_catalog();
        #[cfg(not(feature = "skills"))]
        let skills = Vec::new();
        match CompactionFork::<Schema>::new(
            request,
            self.context.fork_point(),
            self.context.handoff_file_index(),
            focus,
            &skills,
        )
        .map(|fork| fork.with_outputs_dir(self.outputs_dir()))
        {
            Ok(fork) => {
                tracing::debug!(
                    fork_point = fork.fork_point(),
                    usage = self.budget.usage(),
                    "forking compaction alongside the run"
                );
                self.compaction.start(self.start_pending_compaction(fork));
            }
            Err(error) => {
                tracing::debug!(%error, "compaction not forked");
            }
        }
    }

    /// The session's skill catalog, offered to compaction prompts and used to
    /// resolve skill names during restoration.
    #[cfg(feature = "skills")]
    fn skill_catalog(&self) -> Vec<SkillCatalogEntry> {
        if let Some(registry) = &self.skill_registry {
            return registry
                .all()
                .into_iter()
                .map(|skill| SkillCatalogEntry {
                    name: skill.name.clone(),
                    description: skill.description.clone(),
                    instructions: skill.instructions.clone(),
                })
                .collect();
        }
        Vec::new()
    }

    /// Folds a landed handoff into the window and restores what it named.
    ///
    /// Called only at request boundaries: the conversation is whole there, so
    /// replacing everything below the fork point cannot strand a tool call
    /// from its result.
    async fn apply_completed_compaction(&mut self) {
        // The files are read while the completed compaction is still stored:
        // a caller that drops this future during the read leaves it pending
        // for the next boundary, with nothing half applied.
        let Some(completed) = self.compaction.landed() else {
            return;
        };
        let handoff = completed.handoff.clone();
        let restoration = completed.restoration.clone();
        let restored = match self.restore_dir.clone() {
            Some(restore_dir) => {
                let tokens = self
                    .budget
                    .fraction_tokens(self.config.context_policy.restore_budget);
                let budget_chars = self.budget.chars_for_tokens(tokens);
                #[cfg(feature = "skills")]
                let skills = self.skill_catalog();
                #[cfg(not(feature = "skills"))]
                let skills = Vec::new();
                restore_context(
                    &restoration,
                    handoff.file_index(),
                    &[restore_dir],
                    &skills,
                    budget_chars,
                )
                .await
            }
            None => None,
        };

        // From here to the hook there is no await: the handoff, the cleanup
        // and the restored block commit together.
        let Some(CompletedCompaction { fork_point, .. }) = self.compaction.take_landed() else {
            return;
        };
        // Compaction rewrites the cacheable prefix anyway, so this is where
        // staged system-block updates and superseded tool results fold in.
        self.context.reassemble_below(fork_point);
        let user_message_chars = self
            .budget
            .chars_for_tokens(self.config.context_policy.user_message_budget);
        self.context
            .apply_handoff(handoff, fork_point, user_message_chars);
        if let Some(block) = restored {
            self.context.push_block_alert(&block);
        }
        self.reassembled = true;
        self.prefix_rewrites.compacted = true;
        self.compaction_failure = None;
        if let Some(signal) = &self.compact_signal {
            signal.landed();
        }
        tracing::info!(
            fork_point,
            kept = self.context.len_recent(),
            "applied handoff from forked compaction"
        );
        self.hooks
            .on_lifecycle(&external_hook::AfterCompaction {
                messages_remaining: self.context.len_recent(),
            })
            .await;
    }

    /// Makes room at the pause point: waits for the compaction in flight and
    /// applies it, or runs one now when none is.
    ///
    /// A compaction that fails — the one in flight, one that failed earlier
    /// alongside the run, or the one run here — sends the context through
    /// recovery: alerts and reminders are removed, long tool outputs move to
    /// files, [`AgentEvent::CompactionRecovery`] tells the host, and the
    /// compaction reruns synchronously.
    ///
    /// Returns `false` when there was nothing to compact, so the window is
    /// still over the pause point and the run cannot issue another request.
    ///
    /// # Errors
    ///
    /// Returns the error of the rerun when it fails too, or of the file
    /// write recovery needed.
    async fn make_room(&mut self, events: &EventSink) -> Result<bool, AgentError> {
        let mut failure = self.compaction_failure.take().map(AgentError::from);
        if let Some(mut pending) = self.compaction.take_running() {
            match pending.wait().await {
                Ok(completed) => {
                    self.compaction.land(completed);
                    self.apply_completed_compaction().await;
                    return Ok(true);
                }
                Err(error) => {
                    tracing::warn!(%error, "compaction failed with the window exhausted");
                    failure = Some(error.into());
                }
            }
        }
        let failure = match failure {
            Some(failure) => failure,
            None => match Box::pin(self.compact(None)).await {
                Ok(result) => return Ok(result.is_some()),
                Err(error) => {
                    tracing::warn!(%error, "compaction failed with the window exhausted");
                    error
                }
            },
        };

        let policy = self.config.context_policy;
        let limits = SpillLimits {
            threshold: self.budget.chars_for_tokens(policy.spill_threshold),
            head: self.budget.chars_for_tokens(policy.spill_head),
        };
        let fork_point = self.context.fork_point();
        let report = clean_for_recovery(
            &mut self.context,
            fork_point,
            self.restore_dir.as_deref(),
            limits,
        )
        .await
        .map_err(|error| {
            AgentError::llm_local(format!(
                "compaction recovery could not move a long tool output to a file: {error}"
            ))
        })?;
        tracing::warn!(
            alerts_removed = report.alerts_removed,
            outputs_spilled = report.outputs_spilled,
            "rerunning the failed compaction on the cleaned context"
        );
        events.emit(AgentEvent::CompactionRecovery {
            error: failure,
            alerts_removed: report.alerts_removed,
            outputs_spilled: report.outputs_spilled,
        });
        self.reassembled = true;
        self.prefix_rewrites.reassembled = true;
        let result = Box::pin(self.compact(None)).await?;
        Ok(result.is_some())
    }

    /// Renders every context source into the volatile tail of a request.
    ///
    /// Pure: sources are only read, so a preview or a measurement can use
    /// this without consuming a diff the next real request should carry.
    /// Rendering is concurrent because the sources are independent — one
    /// reading `tasks.md` from disk must not wait on another polling the job
    /// registry.
    async fn render_tail(&self) -> RenderedTail {
        #[cfg(feature = "skills")]
        let skills = self.active_skills.render();
        #[cfg(not(feature = "skills"))]
        let skills = std::future::ready(None);
        let (rendered, skills) = futures_lite::future::zip(
            futures::future::join_all(self.context_sources.iter().map(AnyContextSource::render)),
            skills,
        )
        .await;
        let mut tail = RenderedTail::default();
        for (source, block) in rendered.into_iter().enumerate() {
            if let Some(block) = block {
                tail.blocks.push(block);
                tail.sources.push(Some(source));
            }
        }
        if let Some(block) = skills {
            tail.blocks.push(block);
            tail.sources.push(None);
        }
        tail
    }

    /// Tells each source that the block it produced has been sent.
    ///
    /// Separate from rendering so that only a request that actually goes out
    /// advances a source's baseline.
    async fn commit_tail(&mut self, tail: &RenderedTail) {
        for (block, source) in tail.blocks.iter().zip(&tail.sources) {
            if let Some(source) = *source {
                self.context_sources[source].commit(block).await;
            }
        }
    }

    /// Pushes a durable system alert into the conversation.
    ///
    /// Unlike a context source's block, which is re-rendered for every
    /// request, an alert pushed here is an item in the conversation: it
    /// survives until reassembly or a handoff prunes it, which is what makes
    /// it the right channel for a one-off correction the model must see.
    fn push_system_alert(&mut self, content: impl Into<String>) {
        let block = Block::new("system-reminder", content);
        self.budget.append(block.len());
        self.context.push_block_alert(&block);
    }

    /// Brings the command table block to the commands the terminal runs now.
    ///
    /// The table is live content: a command a set gained or lost since the
    /// model last learned the table is announced once, as a diff, and the
    /// next reassembly folds the clean table into the prefix.
    fn sync_command_table(&mut self) {
        let Some(commands) = &self.commands else {
            return;
        };
        let table = commands.render();
        let before = self.context.conversation_chars();
        if self
            .context
            .update_live(&crate::context::LiveContent::COMMANDS, |update| {
                update.instruction(&crate::context::InstructionSlot::COMMANDS, table);
            })
        {
            self.budget
                .append(self.context.conversation_chars().saturating_sub(before));
        }
    }

    /// Delivers the context lifecycle hooks injected since the last call, one
    /// system alert per piece. A piece leaves the hooks' inbox as it is
    /// delivered, so it reaches the model once.
    fn push_injected_context(&mut self) {
        for injected in self.hooks.take_injected_context() {
            let alert = self.notices.lifecycle_context(&injected);
            self.push_system_alert(alert);
        }
    }

    /// Collects the durable alerts sources raise as a request begins.
    ///
    /// Tells the sources whether the window was reassembled since the last
    /// time, which is when their earlier alerts stopped being in it.
    async fn push_request_alerts(&mut self) {
        self.push_injected_context();
        self.push_host_load_alerts();
        self.sync_command_table();
        let reassembled = std::mem::take(&mut self.reassembled);
        for index in 0..self.context_sources.len() {
            if let Some(block) = self.context_sources[index].on_request(reassembled).await {
                self.push_source_alert(&block);
            }
        }
        if let Some(document) = self.tasks_document.as_mut()
            && let Some(block) =
                crate::context::ContextSource::on_request(document, reassembled).await
        {
            self.push_source_alert(&block);
        }
    }

    /// Appends a source's durable alert to the conversation.
    fn push_source_alert(&mut self, block: &Block) {
        self.budget.append(block.len());
        self.context.push_block_alert(block);
    }

    /// Collects the durable alerts sources raise after a tool turn.
    async fn push_tool_turn_alerts(&mut self) {
        self.push_injected_context();
        for index in 0..self.context_sources.len() {
            if let Some(block) = self.context_sources[index].on_tool_turn_end().await {
                self.push_source_alert(&block);
            }
        }
        if let Some(document) = self.tasks_document.as_mut()
            && let Some(block) = crate::context::ContextSource::on_tool_turn_end(document).await
        {
            self.push_source_alert(&block);
        }
    }

    /// Collects the alerts sources raise when the model wants to stop.
    ///
    /// Returns `true` when at least one source objected, which keeps the loop
    /// running for another turn so the model can act on the objection.
    async fn push_final_turn_alerts(&mut self) -> bool {
        let mut pushed = false;
        for index in 0..self.context_sources.len() {
            if let Some(block) = self.context_sources[index].on_final_turn().await {
                self.push_source_alert(&block);
                pushed = true;
            }
        }
        if let Some(document) = self.tasks_document.as_mut()
            && let Some(block) = crate::context::ContextSource::on_final_turn(document).await
        {
            self.push_source_alert(&block);
            pushed = true;
        }
        pushed
    }

    /// Returns the current conversation history.
    #[must_use]
    pub fn history(&self) -> Vec<Message> {
        self.context.conversation_messages()
    }

    fn tool_surface_hash(&self) -> String {
        let mut names = self
            .tools
            .active_definitions()
            .into_iter()
            .map(|definition| definition.name().to_string())
            .collect::<Vec<_>>();
        names.sort();
        let mut hasher = Sha256::new();
        for name in names {
            hasher.update(name.as_bytes());
            hasher.update(b"\n");
        }
        hex_encode(hasher.finalize().as_slice())
    }

    /// Export a checkpoint bundle for restart-safe persistence.
    ///
    /// # Errors
    ///
    /// Returns an error when checkpoint construction or checkpoint hooks fail.
    pub async fn export_checkpoint(&mut self) -> Result<AgentCheckpoint, AgentError> {
        self.ensure_initialized().await?;
        let context_window = self.snapshot_context_window().await?;
        Ok(AgentCheckpoint {
            context: self.context.checkpoint(),
            tool_surface_hash: self.tool_surface_hash(),
            context_window,
            has_background_tasks: self
                .background_receiver
                .as_ref()
                .is_some_and(aither_sandbox::BackgroundTaskReceiver::has_running),
            tasks: self
                .tasks_document
                .as_ref()
                .and_then(crate::tasks::TasksDocument::checkpoint),
        })
    }

    /// Restore a previously exported checkpoint bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when the checkpoint cannot be restored into this agent.
    pub fn restore_checkpoint(&mut self, checkpoint: AgentCheckpoint) -> Result<(), AgentError> {
        self.context.restore(checkpoint.context);
        // The restored conversation holds what its tasks document showed.
        if let Some(document) = self.tasks_document.as_mut() {
            document.restore(checkpoint.tasks);
        }
        // Everything below belongs to the window just replaced: its fork
        // point, its failure, its parked tool results and its last request
        // mean nothing in the restored one.
        self.compaction.retire();
        self.compaction_failure = None;
        self.idle_compacted = false;
        self.tool_results.clear();
        self.kept_exchanges.clear();
        self.last_request = None;
        #[cfg(feature = "skills")]
        {
            self.active_skills.set(Vec::new());
            self.active_allowed_tools = None;
        }
        Ok(())
    }

    /// Returns a structured snapshot of the current context window.
    ///
    /// This is a pure measurement: no source baseline is advanced and no
    /// alert is consumed, so a host may poll it as often as it likes.
    ///
    /// # Errors
    ///
    /// Returns an error when the model profiles cannot be fetched during
    /// initialization.
    pub async fn snapshot_context_window(&mut self) -> Result<ContextWindowSnapshot, AgentError> {
        self.ensure_initialized().await?;
        let tail = self.render_tail().await;
        Ok(self.context_window_snapshot(&tail))
    }

    /// Returns the model profile if available.
    #[must_use]
    pub const fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    /// Ensures the agent is initialized (profiles fetched, static blocks set up).
    ///
    /// The profile is fetched again after the agent's own model was
    /// replaced.
    async fn ensure_initialized(&mut self) -> Result<(), AgentError> {
        if self.profile.is_none() {
            self.load_profile().await?;
        }
        if self.compaction_profile.is_none()
            && let Some(compactor) = &self.compaction_model
        {
            self.compaction_profile = Some(
                compactor
                    .model
                    .profile()
                    .await
                    .map_err(|e| AgentError::from_model_error(&e))?,
            );
        }
        if self.initialized {
            return Ok(());
        }

        self.register_builtin_context_sources();

        // Populate static system blocks in Context from AgentConfig.
        // These form the stable, cacheable prefix.
        self.populate_system_blocks();

        self.initialized = true;
        Ok(())
    }

    /// Fetches the profile of the agent's own model and sizes the window to
    /// it.
    async fn load_profile(&mut self) -> Result<(), AgentError> {
        // The window the current tier sees is the only one the budget may
        // spend: compaction replays the main request's prefix through the
        // same model, so no second profile is involved.
        let profile = match self.tier {
            ModelTier::Advanced => self
                .advanced
                .profile()
                .await
                .map_err(|e| AgentError::from_model_error(&e)),
            ModelTier::Balanced => self
                .balanced
                .profile()
                .await
                .map_err(|e| AgentError::from_model_error(&e)),
            ModelTier::Fast => self
                .fast
                .profile()
                .await
                .map_err(|e| AgentError::from_model_error(&e)),
        }?;
        let policy = self.config.context_policy;
        if !policy.admits(profile.context_length) {
            return Err(AgentError::Config(format!(
                "the {}-token context window leaves no room below its {}-token auto-compact point \
                 once the {}-token safe buffer is kept; lower `ContextPolicy::safe_buffer`",
                profile.context_length,
                policy.compact_point(profile.context_length),
                policy.safe_buffer,
            )));
        }
        self.budget.set_window(profile.context_length);
        self.profile = Some(profile);
        Ok(())
    }

    /// Populates the Context's system blocks from `AgentConfig`.
    ///
    /// Called once during initialization. These blocks form the stable
    /// cacheable prefix of the system message; each renders at its slot in
    /// the host's layout. A built-in block the host already set keeps the
    /// host's text.
    fn populate_system_blocks(&mut self) {
        let instructions = self.context.instructions_mut();
        if let Some(ref system_prompt) = self.config.system_prompt {
            instructions.insert_tagged(&InstructionSlot::BASE_SYSTEM, system_prompt);
        }

        if let Some(ref persona_prompt) = self.config.persona_prompt {
            instructions.insert_tagged(&InstructionSlot::PERSONA, persona_prompt);
        }

        if self.config.agent_kind == AgentKind::Coding
            && !instructions.contains(&InstructionSlot::WORKSPACE_FACTS)
        {
            instructions.insert_tagged(
                &InstructionSlot::WORKSPACE_FACTS,
                include_str!("prompts/workspace_facts.txt"),
            );
        }

        // TerminalAgentBuilder pre-seeds this block rendered against the
        // command registry; keep that version when present.
        if !instructions.contains(&InstructionSlot::KNOWLEDGE_AND_TIME) {
            instructions.insert_tagged(
                &InstructionSlot::KNOWLEDGE_AND_TIME,
                &KnowledgeAndTimeTemplate {
                    websearch_command: true,
                }
                .render()
                .expect("failed to render knowledge_and_time template"),
            );
        }

        if !instructions.contains(&InstructionSlot::PERMISSIONS) {
            instructions.insert_tagged(
                &InstructionSlot::PERMISSIONS,
                include_str!("prompts/permissions.txt"),
            );
        }

        #[cfg(feature = "languages")]
        if let Some(languages) = self
            .config
            .languages
            .as_ref()
            .map(|guard| *guard.languages())
            && languages.enforced()
            && let Some(communication) = languages.communication()
        {
            instructions.insert_tagged(
                &InstructionSlot::LANGUAGES,
                &LanguagesTemplate {
                    communication,
                    working: languages.working(),
                }
                .render()
                .expect("language system template must render"),
            );
        }
    }

    /// Registers the context sources the agent itself owns.
    ///
    /// A host's own sources are registered through the builder and kept, so
    /// this only adds what the agent has the state for.
    fn register_builtin_context_sources(&mut self) {
        #[cfg(feature = "languages")]
        if let Some(guard) = self
            .config
            .languages
            .clone()
            .filter(|guard| guard.languages().enforced())
        {
            self.context_sources
                .push(AnyContextSource::new(crate::language::LanguageAlerts::new(
                    guard,
                    self.sandbox_dir.clone(),
                    self.notices.clone(),
                )));
        }
        if let Some(job_registry) = self.job_registry.clone() {
            self.context_sources
                .push(AnyContextSource::new(BackgroundJobs::new(job_registry)));
        }
    }

    /// Assembles the next request, handling the window's lifecycle first.
    ///
    /// This is the only request boundary there is, and everything that must
    /// not happen mid-turn happens here, in order: a landed handoff is
    /// folded in, the window is measured against the provider's own token
    /// counts, a compaction is forked if usage crossed
    /// [`ContextPolicy::start_point`], and the compaction is awaited (or run,
    /// with recovery when it fails) once the room left before the
    /// auto-compact point is down to [`ContextPolicy::safe_buffer`].
    ///
    /// With a compaction model set, a cache that went cold since the last
    /// request is rebuilt first (`rebuild_if_cache_is_cold`).
    ///
    /// Returns `None` when the window is exhausted and no compaction can
    /// relieve it: the caller pauses the run rather than sending a request
    /// the provider would reject for length.
    ///
    /// # Errors
    ///
    /// Returns the compaction failure when the cold-cache compaction fails;
    /// nothing is sent on an uncompacted context.
    ///
    /// [`ContextPolicy::start_point`]: crate::ContextPolicy::start_point
    /// [`ContextPolicy::safe_buffer`]: crate::ContextPolicy::safe_buffer
    async fn prepare_request(
        &mut self,
        events: &EventSink,
    ) -> Result<Option<LLMRequest>, AgentError> {
        self.apply_completed_compaction().await;
        self.rebuild_if_cache_is_cold().await?;
        self.settle_effort()?;
        self.push_request_alerts().await;

        let definitions = self.tools.active_definitions();
        let mut tail = self.render_tail().await;
        let mut messages = self.request_messages(&tail);
        self.budget
            .observe_assembled(request_chars(&messages, &definitions));

        self.sync_compact_signal();
        if self.raise_precompaction_alert() {
            messages = self.request_messages(&tail);
            self.budget
                .observe_assembled(request_chars(&messages, &definitions));
        }
        let requested = self
            .compact_signal
            .as_ref()
            .is_some_and(|signal| signal.take_request());
        if (requested
            || self
                .config
                .context_policy
                .should_compact(self.budget.estimated_tokens(), self.budget.window()))
            && !self.compaction.is_active()
        {
            let fork_request = self.compaction_request(messages.clone(), definitions.clone());
            self.hooks
                .on_lifecycle(&external_hook::BeforeCompaction {
                    messages: self.context.len_recent(),
                    focus: None,
                })
                .await;
            self.start_compaction(&fork_request, None);
        }

        if self
            .config
            .context_policy
            .should_pause(self.budget.estimated_tokens(), self.budget.window())
        {
            tracing::warn!(
                tokens = self.budget.estimated_tokens(),
                "context window at the safe buffer; waiting for compaction"
            );
            if !self.make_room(events).await? {
                return Ok(None);
            }
            self.push_request_alerts().await;
            tail = self.render_tail().await;
            messages = self.request_messages(&tail);
            self.budget
                .observe_assembled(request_chars(&messages, &definitions));
            // A landed compaction is still no guarantee the rebuilt request
            // fits; an unshrinkable request is paused rather than sent.
            if self
                .config
                .context_policy
                .should_pause(self.budget.estimated_tokens(), self.budget.window())
            {
                return Ok(None);
            }
        }

        self.sync_compact_signal();
        self.commit_tail(&tail).await;
        self.context.mark_prefix_sent();
        self.warmth.start_entry_now();
        self.idle_compacted = false;
        self.publish_fork_snapshot(&tail, &definitions);

        let has_tools = !definitions.is_empty();
        events.emit(AgentEvent::PromptPrefix(PromptPrefix {
            chain: PrefixChain::of(&definitions, &messages),
            rewrites: std::mem::take(&mut self.prefix_rewrites),
            first_request: self.requests_sent == 0,
        }));
        self.requests_sent += 1;
        self.prompt_cache
            .observe_prefix(self.context.stable_prefix_fingerprint());
        let parameters = self.prompt_cache.apply(
            self.request_parameters(),
            &messages,
            has_tools,
            self.takes_mid_conversation_system(),
        );
        let request = LLMRequest::new(messages)
            .with_parameters(parameters)
            .with_tool_definitions(definitions);
        self.last_request = Some(request.clone());
        Ok(Some(request))
    }

    /// Whether a compaction is running or waiting to be applied.
    const fn compaction_in_flight(&self) -> bool {
        self.compaction.is_active()
    }

    /// Tells the `compact` tool whether a compaction is in flight and how
    /// much the context holds against the policy's floor, so a call that is
    /// not worth a compaction starts nothing.
    fn sync_compact_signal(&self) {
        if let Some(signal) = &self.compact_signal {
            signal.sync(
                self.compaction_in_flight(),
                self.budget.estimated_tokens(),
                self.config.context_policy.compaction_floor,
            );
        }
    }

    /// Pushes the pre-compaction alert when usage first comes within
    /// [`ContextPolicy::warn_distance`] of the auto-compact point, and
    /// returns whether it did.
    ///
    /// The alert is one-shot: it is raised on the crossing and not again
    /// until usage has fallen back below the warning point, which only a
    /// compaction does. It is a durable alert appended to the conversation,
    /// so the cached prefix is untouched, and it is not raised when a
    /// compaction is already in flight or the agent has no `compact` tool to
    /// point the model at.
    ///
    /// [`ContextPolicy::warn_distance`]: crate::ContextPolicy::warn_distance
    fn raise_precompaction_alert(&mut self) -> bool {
        if self.compact_signal.is_none() {
            return false;
        }
        let policy = self.config.context_policy;
        let tokens = self.budget.estimated_tokens();
        let window = self.budget.window();
        if !policy.should_warn(tokens, window) {
            self.compaction_warning = CompactionWarning::Armed;
            return false;
        }
        if self.compaction_warning == CompactionWarning::Raised {
            return false;
        }
        self.compaction_warning = CompactionWarning::Raised;
        if self.compaction_in_flight() {
            return false;
        }
        let alert = self.notices.precompaction(&crate::PreCompaction {
            remaining_tokens: policy.compact_point(window).saturating_sub(tokens),
        });
        self.push_system_alert(alert);
        true
    }

    /// Publishes the request about to be sent for a fork to start from.
    fn publish_fork_snapshot(
        &self,
        tail: &[Block],
        definitions: &[aither_core::llm::tool::ToolDefinition],
    ) {
        let Some(source) = &self.fork_source else {
            return;
        };
        let mut context = self.context.clone();
        context.push_tail(tail);
        source.publish(crate::fork::ForkSnapshot {
            tier: self.tier,
            context,
            prompt_cache: self.prompt_cache.clone(),
            budget: self.budget,
            profile: self
                .profile
                .clone()
                .expect("a request is only sent by an initialized agent, which has its profile"),
            request_parameters: self.own_parameters().clone(),
            effort: self.effort,
            context_policy: self.config.context_policy,
            definitions: definitions.to_vec(),
            usage: self.budget.usage(),
        });
    }

    /// Rebuilds the prefix once, before a send, when the provider's cache has
    /// gone cold.
    ///
    /// Only with a compaction model set (otherwise the session is compacted
    /// shortly before the cache expires, see [`Agent::compact_when_idle`], and
    /// a send never rebuilds). The cache is cold when the conversation's
    /// [`CacheLifetime`] has passed since the entry was last used; a permanent
    /// lifetime never is, and nothing runs while the session is idle.
    ///
    /// The compaction model reads the context uncached whatever the timing, so
    /// compacting now costs no more than later, and the send then pays its
    /// cache write on the compacted context instead of the whole one. The
    /// compaction's apply already reassembles the window, so the prefix is
    /// rewritten once: reassembly alone runs only when there is no
    /// conversation to compact. Nothing is kept across the handoff, so no
    /// thinking produced under the old prefix is replayed after it.
    ///
    /// A forked compaction already in flight lands and rebuilds the prefix
    /// itself, so it is left alone.
    ///
    /// [`CacheLifetime`]: crate::context::CacheLifetime
    async fn rebuild_if_cache_is_cold(&mut self) -> Result<(), AgentError> {
        if self.compaction_model.is_none() || self.compaction.is_running() {
            return Ok(());
        }
        if !self.warmth.has_expired() {
            return Ok(());
        }
        if self.context.is_conversation_empty() {
            self.reassemble_if_cache_is_cold();
            return Ok(());
        }
        // Boxed: the compaction future is large and sits on every run's path
        // (see `run_futures_stay_small_with_a_real_provider_type`).
        if let Some(result) = Box::pin(self.compact(None)).await? {
            tracing::info!(
                compacted = result.messages_compacted,
                "compacted the context before sending on a cold cache"
            );
        }
        Ok(())
    }

    /// Rebuilds the window once the provider's cache has gone cold anyway.
    ///
    /// Reached only for a session with a compaction model and nothing to
    /// compact (`rebuild_if_cache_is_cold`).
    ///
    /// Reassembly rewrites the cacheable prefix, so it is only free once the
    /// cached copy has expired — an idle gap at least as long as the
    /// conversation's [`CacheLifetime`]. A staged system-block rewrite and
    /// superseded tool results wait for exactly that moment; a permanent
    /// cache lifetime means no idle gap ever justifies the rewrite.
    ///
    /// [`CacheLifetime`]: crate::context::CacheLifetime
    fn reassemble_if_cache_is_cold(&mut self) -> crate::context::ReassemblyReport {
        let unchanged = crate::context::ReassemblyReport::default();
        if !self.warmth.has_expired() {
            return unchanged;
        }
        let report = self.context.reassemble();
        self.reassembled |= report.system_messages_pruned > 0;
        self.prefix_rewrites.reassembled |= report.changed();
        if report.changed() {
            tracing::debug!(
                system_messages = report.system_messages_pruned,
                deduped = report.tool_results_deduped,
                bytes_freed = report.bytes_freed,
                "rebuilt the context window on a cold cache"
            );
        }
        report
    }

    #[cfg(feature = "skills")]
    pub(crate) fn activate_skills_for_prompt(&mut self, _prompt: &str) -> Vec<AgentEvent> {
        let _ = self.skill_registry.as_ref();
        self.active_skills.set(Vec::new());
        self.active_allowed_tools = None;
        Vec::new()
    }

    /// Measures the window exactly as it stands, against a rendered `tail`.
    ///
    /// Pure, so checkpoints and status displays can call it at any point in
    /// a turn without disturbing what the next request will carry.
    fn context_window_snapshot(&self, tail: &[Block]) -> ContextWindowSnapshot {
        let messages = self.context.build_request(tail);
        let metrics = self.context_window_metrics(&messages);
        ContextWindowSnapshot {
            phase: self.classify_context_window_phase(&metrics),
            metrics,
            messages,
        }
    }

    fn context_window_metrics(&self, messages: &[Message]) -> ContextWindowMetrics {
        let mut budget = self.budget;
        budget.observe_assembled(request_chars(messages, &self.tools.active_definitions()));
        ContextWindowMetrics {
            usage_fraction: budget.usage(),
            context_window: budget.window(),
            estimated_tokens: budget.estimated_tokens(),
            chars_per_token: budget.chars_per_token(),
            system_block_count: self.context.system_block_count(),
            recent_message_count: self.context.len_recent(),
            recent_system_message_count: self.context.count_recent_system_messages(),
            has_handoff: self.context.handoff().is_some(),
            has_staged_system_updates: self.context.has_staged_system_updates(),
            prompt_cache_epoch: self.prompt_cache.epoch(),
        }
    }

    fn classify_context_window_phase(&self, metrics: &ContextWindowMetrics) -> ContextWindowPhase {
        let policy = &self.config.context_policy;
        if policy.should_pause(metrics.estimated_tokens, metrics.context_window) {
            return ContextWindowPhase::Exhausted;
        }
        if self.compaction.is_active() {
            return ContextWindowPhase::Compacting;
        }
        if metrics.has_staged_system_updates
            || (self.compaction_model.is_some() && self.warmth.has_expired())
        {
            return ContextWindowPhase::ReassemblyDue;
        }
        ContextWindowPhase::Stable
    }

    /// Returns the exact request message sequence that would be sent for the next turn.
    ///
    /// Intended for observability and debug UIs: the prompt is added to a
    /// forked context and the sources are rendered without committing, so a
    /// preview cannot consume a change the real request should surface.
    ///
    /// # Errors
    ///
    /// Returns an error when the model profiles cannot be fetched during
    /// initialization.
    pub async fn preview_request_messages(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> Result<Vec<Message>, AgentError> {
        self.preview_request_messages_with(prompt, std::iter::empty(), attachments)
            .await
    }

    /// Returns the exact request message sequence that would be sent for the
    /// next turn, with `extra_messages` appended at the conversation tail
    /// before the prompt.
    ///
    /// This is the non-mutating counterpart of pushing those messages onto the
    /// live context and then running the turn — the shape a host gets for
    /// per-turn hidden context injected ahead of the user's message. The
    /// extras are not persisted: the context and the sources' committed state
    /// are left exactly as a normal first assembly would, because sources are
    /// only told about a request once it is actually sent.
    ///
    /// # Errors
    ///
    /// Returns an error when the model profiles cannot be fetched during
    /// initialization.
    pub async fn preview_request_messages_with(
        &mut self,
        prompt: &str,
        extra_messages: impl IntoIterator<Item = Message>,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> Result<Vec<Message>, AgentError> {
        self.ensure_initialized().await?;
        let checkpoint = self.context.checkpoint();
        self.context.extend(extra_messages);
        self.context.push(
            self.timestamped_user_message(prompt)
                .with_attachments(attachments),
        );
        let tail = self.render_tail().await;
        let messages = self.context.build_request(&tail);
        self.context.restore(checkpoint);
        Ok(messages)
    }

    async fn emit_checkpoint(
        &self,
        reason: CheckpointReason,
        assistant_text: &str,
        turn: usize,
    ) -> Result<EmittedCheckpoint, AgentError> {
        let context = self.context.checkpoint();
        let tool_surface_hash = self.tool_surface_hash();
        let tail = self.render_tail().await;
        let window = self.context_window_snapshot(&tail);
        let tasks = self
            .tasks_document
            .as_ref()
            .and_then(crate::tasks::TasksDocument::checkpoint);
        let checkpoint_ctx = CheckpointContext {
            reason,
            assistant_text,
            turn,
            message_count: self.context.len_recent(),
            context: &context,
            tool_surface_hash: &tool_surface_hash,
            has_background_tasks: self
                .background_receiver
                .as_ref()
                .is_some_and(aither_sandbox::BackgroundTaskReceiver::has_running),
            window: &window,
            tasks: tasks.as_ref(),
        };
        if let Some(reason) = self.hooks.on_checkpoint(&checkpoint_ctx).await {
            return Err(AgentError::HookRejected {
                hook: "on_checkpoint",
                reason,
            });
        }
        Ok(EmittedCheckpoint {
            phase: window.phase,
            message_count: self.context.len_recent(),
        })
    }
}

/// Formats a completed background task result as a system message.
/// aither's wording of a finished background task; `timeout` states the
/// timeout kill, when there was one.
pub fn default_background_finished(
    task: &aither_sandbox::CompletedTask,
    timeout: Option<&str>,
) -> String {
    let error_text;
    let (outcome, output, stderr) = match &task.result {
        Ok(command) if task.kill.is_some() => (
            BackgroundOutcome::Killed {
                timeout,
                runaway: match task.kill {
                    Some(aither_sandbox::KillCause::MemoryRunaway {
                        footprint_bytes,
                        capacity_bytes,
                        domain,
                    }) => Some(RunawayText {
                        footprint: aither_sandbox::MemorySize(footprint_bytes),
                        capacity: aither_sandbox::MemorySize(capacity_bytes),
                        domain,
                    }),
                    Some(aither_sandbox::KillCause::User) | None => None,
                },
            },
            output_block(&command.stdout),
            command.stderr.as_ref().and_then(output_block),
        ),
        Ok(command) => (
            BackgroundOutcome::Finished {
                code: command.exit_code,
            },
            output_block(&command.stdout),
            command.stderr.as_ref().and_then(output_block),
        ),
        Err(error) => {
            error_text = error.to_string();
            (BackgroundOutcome::Failed { error: &error_text }, None, None)
        }
    };
    let interruptions = task.result.as_ref().map_or_else(
        |_| String::new(),
        |command| aither_core::llm::render_interruptions(&command.interruptions),
    );
    BackgroundTerminalResultTemplate {
        task_id: &task.task_id,
        script: truncate_script(&task.script, 100),
        outcome,
        output: output.as_deref(),
        stderr: stderr.as_deref(),
        interruptions: &interruptions,
    }
    .render()
    .expect("background terminal result template must render")
}

/// A command's output as a block of lines, or `None` when it printed nothing.
fn output_block(stream: &CommandStream) -> Option<String> {
    Some(stream.render_for_model()).filter(|block| !block.is_empty())
}

fn format_builtin_tool_result(tool: &str, result: &str) -> String {
    let mut text = String::from("[");
    text.push_str(tool);
    text.push_str("] ");
    text.push_str(result);
    text
}

fn permission_event_key_for_call(
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Option<PermissionEventKey> {
    let (mode, script) = match tool_name {
        "terminal" => {
            let args: TerminalArgs = serde_json::from_value(arguments.clone()).ok()?;
            (args.mode, args.script)
        }
        _ => return None,
    };
    let mode = match mode {
        TerminalExecutionMode::Sandboxed
        | TerminalExecutionMode::Default
        | TerminalExecutionMode::Ssh => return None,
        TerminalExecutionMode::Unsafe => TerminalMode::Unsafe,
    };
    Some(PermissionEventKey { mode, script })
}

/// Renders a background job's runtime the way the alert text states it.
fn format_job_elapsed(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds >= 60 {
        format!("{} minutes", seconds / 60)
    } else {
        format!("{seconds} seconds")
    }
}

fn pause_reason_for_tool(tool_name: &str) -> Option<RunPauseReason> {
    match tool_name {
        "ask_user" => Some(RunPauseReason::AskUser),
        "request_workspace" => Some(RunPauseReason::WorkspaceRequest),
        _ => None,
    }
}

fn emit_permission_event(
    tracker: &mut PermissionPauseTracker,
    event: &PermissionEvent,
    events: &EventSink,
) {
    if let Some((reason, tool_call_id)) = tracker.event_for(event.clone()) {
        match event.stage {
            PermissionEventStage::Waiting => {
                events.emit(AgentEvent::run_paused(reason, tool_call_id));
            }
            PermissionEventStage::Resolved => {
                events.emit(AgentEvent::run_resumed(reason, tool_call_id));
            }
        }
    }
}

fn tool_result_is_error(tool_result: &aither_core::llm::ToolResult, content: &str) -> bool {
    tool_result.is_error()
        || content.contains("ssh_server_id is required")
        || content.contains("unknown ssh_server_id")
        || content.contains("not found")
        || content.contains("Invalid arguments")
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Truncates a script for display in messages.
fn truncate_script(script: &str, max_chars: usize) -> &str {
    let script = script.trim();
    // Find byte index at max_chars boundary
    match script.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &script[..byte_idx],
        None => script, // String is shorter than max_chars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentOptions;

    use crate::Handoff;
    use aither_sandbox::{CompletedCommand, CompletedTask, TerminalError};
    use futures_core::Stream;

    /// Lands `document` as a compaction forked at `fork_point` finished.
    fn land_document(
        compaction: &mut crate::compaction_state::CompactionState,
        fork_point: u64,
        document: HandoffDocument,
    ) {
        compaction.land(CompletedCompaction::new(fork_point, document, Vec::new()));
    }

    /// aither's wording of a finished background task.
    fn format_background_task_result(task: &CompletedTask) -> String {
        crate::Notices::background_finished(&crate::DefaultNotices, task)
    }

    fn inline(text: &str) -> CommandStream {
        CommandStream::Inline {
            text: text.to_string(),
        }
    }

    fn completed(result: Result<CompletedCommand, TerminalError>) -> CompletedTask {
        CompletedTask {
            task_id: "amber-forest-thunder-pearl".to_string(),
            script: "make test".to_string(),
            result,
            kill: None,
            timeout_kill: None,
            pending: None,
        }
    }

    #[test]
    fn background_result_states_the_outages_it_ran_through() {
        use aither_core::llm::{Interruption, InterruptionCause};
        let task = completed(Ok(CompletedCommand {
            stdout: inline("all green"),
            stderr: None,
            exit_code: 0,
            interruptions: vec![Interruption {
                cause: InterruptionCause::Offline,
                from: "14:02:11".to_string(),
                to: "14:31:40".to_string(),
            }],
        }));
        assert_eq!(
            format_background_task_result(&task),
            "Background task 'amber-forest-thunder-pearl' (`make test`) finished with exit code 0.\nall green\nThe network was offline from 14:02:11 to 14:31:40 while this ran.\n"
        );
    }

    #[test]
    fn background_result_exit_zero_with_output() {
        let task = completed(Ok(CompletedCommand {
            stdout: inline("all green"),
            stderr: None,
            exit_code: 0,
            interruptions: Vec::new(),
        }));
        assert_eq!(
            format_background_task_result(&task),
            "Background task 'amber-forest-thunder-pearl' (`make test`) finished with exit code 0.\nall green\n"
        );
    }

    #[test]
    fn background_result_non_zero_exit_with_stderr() {
        let task = completed(Ok(CompletedCommand {
            stdout: inline("1 passed"),
            stderr: Some(inline("boom")),
            exit_code: 101,
            interruptions: Vec::new(),
        }));
        assert_eq!(
            format_background_task_result(&task),
            "Background task 'amber-forest-thunder-pearl' (`make test`) finished with exit code 101.\n1 passed\nstderr:\nboom\n"
        );
    }

    #[test]
    fn background_result_killed_reports_the_kill_not_the_exit_code() {
        let mut task = completed(Ok(CompletedCommand {
            stdout: inline("partial output"),
            stderr: None,
            exit_code: -1,
            interruptions: Vec::new(),
        }));
        task.kill = Some(aither_sandbox::KillCause::User);
        assert_eq!(
            format_background_task_result(&task),
            "Background task 'amber-forest-thunder-pearl' (`make test`) was killed.\npartial output\n"
        );
    }

    #[test]
    fn background_result_killed_at_its_timeout_states_the_utc_times() {
        let mut task = completed(Ok(CompletedCommand {
            stdout: inline("partial output"),
            stderr: None,
            exit_code: -1,
            interruptions: Vec::new(),
        }));
        task.kill = Some(aither_sandbox::KillCause::User);
        task.timeout_kill = Some(aither_sandbox::TimeoutKill {
            timeout: Duration::from_secs(90),
            started: "2026-11-04T12:00:00Z".parse().expect("timestamp"),
            killed: "2026-11-04T12:01:31Z".parse().expect("timestamp"),
        });
        assert_eq!(
            format_background_task_result(&task),
            "Background task 'amber-forest-thunder-pearl' (`make test`) was killed. Its 90s timeout bound passed; it started at 2026-11-04 12:00:00 UTC and was killed at 2026-11-04 12:01:31 UTC.\npartial output\n"
        );
    }

    #[test]
    fn job_alerts_render_as_prose() {
        assert_eq!(
            job_alert_text(&JobAlertText::StdinBlocked { task_id: "amber" }),
            "Background job 'amber' is waiting for input on stdin — send it with terminal_input, or terminal_kill it if it needs none."
        );
        assert_eq!(
            job_alert_text(&JobAlertText::StdinProbeFailed {
                task_id: "amber",
                error: "no /proc"
            }),
            "Monitoring background job 'amber' for stdin reads failed: no /proc — it cannot report being blocked on stdin."
        );
        for (reason, expected) in [
            (DirectoryResetReason::Failed, "the last command failed"),
            (DirectoryResetReason::Unreported, "exit or a signal"),
            (
                DirectoryResetReason::OutsideReadable,
                "paths the terminal can read",
            ),
            (DirectoryResetReason::OutsideWorkspace, "workspace mount"),
            (DirectoryResetReason::Removed, "no longer exists"),
        ] {
            let text = job_alert_text(&JobAlertText::DirectoryReset {
                dir: "/work/space".to_string(),
                reason,
            });
            assert!(
                text.starts_with("Commands now run in /work/space: "),
                "{text}"
            );
            assert!(text.contains(expected), "{text}");
        }
        let killed = job_alert_text(&JobAlertText::DescendantsKilled {
            processes: &[
                JobProcess {
                    pid: 41,
                    name: "sleep".to_string(),
                },
                JobProcess {
                    pid: 42,
                    name: "node".to_string(),
                },
            ],
            omitted: 3,
        });
        assert_eq!(
            killed,
            "The command ended while sleep (pid 41), node (pid 42) and 3 more kept running, so they were killed with it. A process meant to keep running is started with `background: true`."
        );
        let one = job_alert_text(&JobAlertText::DescendantsKilled {
            processes: &[JobProcess {
                pid: 7,
                name: "sleep".to_string(),
            }],
            omitted: 0,
        });
        assert!(one.contains("sleep (pid 7) kept running"), "{one}");
        let long = job_alert_text(&JobAlertText::LongRunning {
            task_id: "amber",
            elapsed: "5m".to_string(),
        });
        assert!(
            long.starts_with("Background job 'amber' has been running for 5m"),
            "{long}"
        );
    }

    #[test]
    fn background_result_error() {
        let task = completed(Err(TerminalError::Execution("spawn failed".to_string())));
        assert_eq!(
            format_background_task_result(&task),
            "Background task 'amber-forest-thunder-pearl' (`make test`) failed: execution failed: spawn failed\n"
        );
    }

    #[test]
    fn background_started_reminder_names_the_output_file() {
        let file = StoredFile {
            url: "outputs/calm-river-bold-oak.txt".to_string(),
            path: "/sandbox/outputs/calm-river-bold-oak.txt".to_string(),
        };
        let rendered = BackgroundStartedReminderTemplate {
            task_id: "amber-forest-thunder-pearl",
            output_preview: "",
            output_file: &file,
        }
        .render()
        .unwrap();
        assert_eq!(
            rendered,
            "<system-reminder>\nA terminal command is running in background (task_id=amber-forest-thunder-pearl).\nCurrent output snapshot (first max_lines):\n(no output yet)\nFull redirected output file: outputs/calm-river-bold-oak.txt (absolute path: /sandbox/outputs/calm-river-bold-oak.txt)\nUse terminal_read for incremental reads, read the stored file via terminal commands when needed, use terminal_input for stdin, and terminal_kill to stop it. The task announces itself when it finishes — waiting is an option; you do not need to poll it.\n</system-reminder>"
        );
    }

    use crate::ContextWindowPhase;
    #[cfg(feature = "skills")]
    use crate::{AgentCheckpoint, ContextWindowMetrics, ContextWindowSnapshot};
    #[cfg(feature = "skills")]
    use aither_skills::{Skill, SkillRegistry};
    #[cfg(feature = "skills")]
    use std::collections::HashMap;
    #[cfg(feature = "skills")]
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockError;

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("mock error")
        }
    }

    impl std::error::Error for MockError {}

    impl aither_core::llm::Classify for MockError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    #[derive(Clone)]
    struct MockLlm {
        context_length: u32,
    }

    impl LanguageModel for MockLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::stream::empty()
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                self.context_length,
            )))
        }
    }

    /// Answers every request with two text chunks.
    #[derive(Clone)]
    struct TwoChunkLlm;

    impl LanguageModel for TwoChunkLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::stream::iter([Ok(Event::text("first")), Ok(Event::text("second"))])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "two-chunk",
                "two chunks",
                200_000,
            )))
        }
    }

    /// Fails every request.
    #[derive(Clone)]
    struct FailingLlm;

    impl LanguageModel for FailingLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::stream::once(Err(MockError))
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock", "test", "failing", "failing", 200_000,
            )))
        }
    }

    /// Takes `lock` on the first chunk and keeps it across that chunk's
    /// emit, releasing it on the next chunk after a suspension: the shape of
    /// a run that holds a resource (a database writer) across an emit.
    struct HoldAcrossEmit {
        lock: Arc<futures::lock::Mutex<()>>,
        held: std::sync::Mutex<Option<futures::lock::OwnedMutexGuard<()>>>,
    }

    impl Hook for HoldAcrossEmit {
        async fn on_text(&self, text: &str) {
            if text == "first" {
                let guard = Arc::clone(&self.lock).lock_owned().await;
                *self.held.lock().expect("held lock") = Some(guard);
            } else {
                futures_lite::future::yield_now().await;
                self.held.lock().expect("held lock").take();
            }
        }
    }

    /// aither#501: the run keeps making progress while the caller handles an
    /// event, so a handler waiting on a resource the run holds lets the run
    /// release it instead of waiting forever.
    #[tokio::test]
    async fn the_run_progresses_while_the_caller_handles_an_event() {
        let lock = Arc::new(futures::lock::Mutex::new(()));
        let mut agent = Agent::builder(TwoChunkLlm)
            .hook(HoldAcrossEmit {
                lock: Arc::clone(&lock),
                held: std::sync::Mutex::default(),
            })
            .build();
        let lock = &lock;
        let drive = agent.run("hello", []).drive(
            Vec::new(),
            std::future::pending(),
            |mut texts, timed| async move {
                if let AgentEvent::Text(text) = timed.event {
                    if text == "first" {
                        drop(lock.lock().await);
                    }
                    texts.push(text);
                }
                ControlFlow::<(), _>::Continue(texts)
            },
        );
        let flow = tokio::time::timeout(Duration::from_secs(5), drive)
            .await
            .expect("the handler waiting on the run's resource must not hang")
            .expect("the run succeeds");
        drop(agent);
        assert_eq!(
            flow,
            ControlFlow::Continue(vec!["first".to_string(), "second".to_string()])
        );
    }

    /// A handler that breaks stops the run and gets its value back.
    #[tokio::test]
    async fn a_breaking_handler_stops_the_run() {
        let mut agent = Agent::new(TwoChunkLlm);
        let mut handled = 0;
        let flow = agent
            .run("hello", [])
            .drive(
                &mut handled,
                std::future::pending(),
                |handled, timed| async move {
                    *handled += 1;
                    match timed.event {
                        AgentEvent::Text(text) => ControlFlow::Break(text),
                        _ => ControlFlow::Continue(handled),
                    }
                },
            )
            .await
            .expect("the run succeeds");
        assert_eq!(flow, ControlFlow::Break("first".to_string()));
        assert_eq!(
            handled, 3,
            "the run start, the request's prompt prefix and the first chunk"
        );
        assert_eq!(
            agent.query("again").await.expect("the agent runs again"),
            "firstsecond"
        );
    }

    /// The run's error reaches the caller after every event emitted before
    /// it was handled.
    #[tokio::test]
    async fn a_failed_run_returns_its_error_after_its_events() {
        let mut agent = Agent::new(FailingLlm);
        let mut kinds = Vec::new();
        let result = agent
            .run("hello", [])
            .drive(
                &mut kinds,
                std::future::pending(),
                |kinds, timed| async move {
                    kinds.push(std::mem::discriminant(&timed.event));
                    ControlFlow::<(), _>::Continue(kinds)
                },
            )
            .await;
        assert!(result.is_err(), "the model's failure ends the run");
        assert!(!kinds.is_empty(), "the run start was handled first");
    }

    /// Starts each response and never sends anything: a run busy with no
    /// events, like one waiting on a long tool call.
    #[derive(Clone)]
    struct SilentLlm;

    impl LanguageModel for SilentLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::stream::pending()
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock", "test", "silent", "silent", 200_000,
            )))
        }
    }

    /// A stop that fires while the run works without emitting ends it at
    /// once.
    #[tokio::test]
    async fn a_stop_ends_a_run_busy_without_events() {
        let mut agent = Agent::new(SilentLlm);
        let (cancel, cancelled) = async_channel::bounded::<&str>(1);
        let stopper = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel.send("cancelled").await.expect("the run listens");
        });
        let drive = agent.run("hello", []).drive(
            0,
            async move { cancelled.recv().await.expect("the stopper sends") },
            |handled, _| async move { ControlFlow::Continue(handled + 1) },
        );
        let flow = tokio::time::timeout(Duration::from_secs(5), drive)
            .await
            .expect("the stop must end the run promptly")
            .expect("a stopped run is no error");
        assert_eq!(flow, ControlFlow::Break("cancelled"));
        stopper.await.expect("the stopper finishes");
    }

    /// A stop that fires during a handler call takes effect once that call
    /// returned, before the next event is handled.
    #[tokio::test]
    async fn a_stop_never_interrupts_a_handler_call() {
        let mut agent = Agent::new(TwoChunkLlm);
        let (cancel, cancelled) = async_channel::bounded::<()>(1);
        let cancel = &cancel;
        let mut completed = Vec::new();
        let flow = agent
            .run("hello", [])
            .drive(
                &mut completed,
                async move { cancelled.recv().await.expect("the handler cancels") },
                |completed, timed| async move {
                    cancel.send(()).await.expect("the run listens");
                    // Suspends with the stop ready: polling it here would
                    // drop this call before it records its completion.
                    futures_lite::future::yield_now().await;
                    completed.push(timed.event);
                    ControlFlow::Continue(completed)
                },
            )
            .await
            .expect("a stopped run is no error");
        assert!(flow.is_break(), "the stop ended the run");
        assert_eq!(
            completed.len(),
            1,
            "the running call finished, no other began"
        );
        assert!(matches!(completed[0], AgentEvent::RunStart { .. }));
    }

    /// A host spawns its turn loop onto a multi-threaded executor, so the
    /// driven future must be `Send` while its handler borrows the host's
    /// locals, including inside an outer future.
    #[test]
    fn a_driven_run_is_send() {
        fn assert_send(_: &impl Send) {}
        let mut agent = Agent::new(TwoChunkLlm);
        let turn = async move {
            let mut texts = Vec::new();
            let mut stopped = None;
            let (_cancel, cancelled) = async_channel::bounded::<()>(1);
            let outcome = agent
                .run("hello", [])
                .drive(
                    (&mut texts, &mut stopped),
                    async move { cancelled.recv().await.ok() },
                    |(texts, stopped), timed| async move {
                        if let AgentEvent::Text(text) = timed.event {
                            texts.push(text);
                        }
                        futures_lite::future::yield_now().await;
                        *stopped = Some(());
                        ControlFlow::Continue((texts, stopped))
                    },
                )
                .await
                .map(|_| ());
            (outcome, texts, stopped)
        };
        assert_send(&turn);
    }

    #[tokio::test]
    async fn events_carry_their_emission_time() {
        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 200_000,
            },
            AgentConfig::default(),
        );
        let before = jiff::Timestamp::now();
        let ControlFlow::Continue(events) = agent
            .run("hello", std::iter::empty())
            .drive(
                Vec::new(),
                std::future::pending(),
                |mut events, timed| async move {
                    events.push(timed);
                    ControlFlow::<std::convert::Infallible, _>::Continue(events)
                },
            )
            .await
            .expect("the run succeeds");
        let after = jiff::Timestamp::now();

        assert!(matches!(
            events.first().map(|timed| &timed.event),
            Some(AgentEvent::RunStart { .. })
        ));
        for timed in &events {
            assert!(before <= timed.at && timed.at <= after);
        }
        assert!(events.windows(2).all(|pair| pair[0].at <= pair[1].at));
    }

    /// aither#143: the provider `respond` stream's multi-KB state machine must
    /// stay boxed at the leaf (`respond_stream`); if the box is removed these
    /// futures embed it by value and each poll frame pays the ~32 KB temp.
    #[test]
    fn run_futures_stay_small_with_a_real_provider_type() {
        let provider = aither_openai::OpenAI::new("sk-test").with_base_url("http://127.0.0.1:9");
        let mut agent = Agent::new(provider);
        let (tx, _rx) = async_channel::unbounded();
        let sink = EventSink { sender: tx };

        let request = LLMRequest::new(vec![Message::user("hi")]);
        {
            let fut = agent.collect_model_turn(request, &sink);
            assert!(
                std::mem::size_of_val(&fut) < 8 * 1024,
                "collect_model_turn future is {} B",
                std::mem::size_of_val(&fut)
            );
        }
        {
            let fut = agent.run_streaming("hi".to_string(), Vec::new(), &sink);
            assert!(
                std::mem::size_of_val(&fut) < 8 * 1024,
                "run_streaming future is {} B",
                std::mem::size_of_val(&fut)
            );
        }
    }

    #[tokio::test]
    async fn a_window_with_no_room_below_the_safe_buffer_is_refused() {
        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 4_096,
            },
            AgentConfig::default(),
        );
        let error = agent
            .ensure_initialized()
            .await
            .expect_err("a 4k window under the 5k buffer cannot hold a request");
        assert!(
            matches!(&error, AgentError::Config(message) if message.contains("safe_buffer")),
            "unexpected error: {error:?}"
        );
    }

    #[tokio::test]
    async fn snapshot_phase_tracks_usage_against_the_policy() {
        // A buffer small enough for the 100-token window to admit requests.
        let config = AgentConfig {
            system_prompt: Some("x".repeat(400)),
            context_policy: crate::ContextPolicy {
                safe_buffer: 10,
                ..crate::ContextPolicy::default()
            },
            ..AgentConfig::default()
        };

        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 100,
            },
            config,
        );
        agent.push_message(Message::user("hello"));

        let snapshot = agent
            .snapshot_context_window()
            .await
            .expect("snapshot should initialize");

        assert_eq!(snapshot.phase, ContextWindowPhase::Exhausted);
        assert!(snapshot.metrics.system_block_count > 0);
        assert!(
            snapshot.metrics.estimated_tokens
                >= agent
                    .config
                    .context_policy
                    .pause_point(snapshot.metrics.context_window)
        );
        assert_eq!(snapshot.metrics.recent_message_count, 1);
        assert_eq!(snapshot.metrics.context_window, 100);
    }

    #[tokio::test]
    async fn snapshot_reports_a_handoff_without_calling_it_a_phase() {
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.context_mut().set_handoff(Handoff::new("resume here"));

        let snapshot = agent
            .snapshot_context_window()
            .await
            .expect("snapshot should initialize");

        assert_eq!(snapshot.phase, ContextWindowPhase::Stable);
        assert!(snapshot.metrics.has_handoff);
    }

    #[tokio::test]
    async fn a_second_compaction_failure_stops_the_run_with_the_error() {
        let config = AgentConfig {
            system_prompt: Some("x".repeat(4_000)),
            context_policy: crate::ContextPolicy {
                safe_buffer: 10,
                ..crate::ContextPolicy::default()
            },
            ..AgentConfig::default()
        };
        // The window is far over the pause point. The model never answers
        // with a handoff: the forked compaction fails, and so does the
        // rerun on the cleaned context.
        let llm = ScriptedLlm::new(100, []);
        let mut agent = Agent::with_config(llm.clone(), config);
        agent.push_message(Message::user("hello"));
        agent.ensure_initialized().await.expect("agent initializes");
        let (sender, events) = async_channel::unbounded();

        let error = agent
            .prepare_request(&EventSink { sender })
            .await
            .expect_err("no request is sent on a window this far over the pause point");

        assert!(
            error.to_string().contains("compaction"),
            "the compaction's own error stops the run: {error}"
        );
        let recoveries = std::iter::from_fn(|| events.try_recv().ok())
            .filter(|timed| matches!(timed.event, AgentEvent::CompactionRecovery { .. }))
            .count();
        assert_eq!(
            recoveries, 1,
            "the host hears of the first failure and the rerun"
        );
        // The forked attempt and the rerun each get a request and a
        // corrective retry.
        assert_eq!(llm.recorded().len(), 4);
    }

    /// Fills the window to `tokens` with one long tool output, behind an
    /// alert, and aims the agent's cleanup at a temporary working directory.
    async fn agent_with_long_output(
        llm: &ScriptedLlm,
        tokens: usize,
        dir: &std::path::Path,
    ) -> Agent<ScriptedLlm> {
        let mut agent = Agent::new(llm.clone());
        agent.restore_dir = Some(dir.to_path_buf());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("run it"));
        agent.push_message(Message::assistant_with_tool_calls(
            "",
            vec![aither_core::llm::ToolCall::new(
                "call_1",
                "terminal",
                serde_json::json!({"description": "build"}),
            )],
        ));
        agent.push_message(Message::tool("call_1", "o".repeat(tokens * 4)));
        agent.push_message(Message::system(
            "<system-reminder>stale alert</system-reminder>",
        ));
        agent
    }

    /// A 1M window starts compacting 50k tokens before 400k, and the turn's
    /// request still goes out.
    #[tokio::test]
    async fn crossing_the_start_distance_forks_without_blocking() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut agent = agent_with_long_output(&llm, 360_000, dir.path()).await;

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");

        assert!(request.is_some(), "the turn is not held back");
        assert!(agent.compaction.is_running(), "the fork started");
        assert!(llm.recorded().is_empty(), "nothing waited for it");
        assert!(agent.context().handoff().is_none());
    }

    /// Below the start point nothing forks, on a 200k window as on a 1M one.
    #[tokio::test]
    async fn below_the_start_distance_nothing_forks() {
        let llm = ScriptedLlm::new(200_000, [HANDOFF_JSON.to_string()]);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut agent = agent_with_long_output(&llm, 100_000, dir.path()).await;

        assert!(
            agent
                .prepare_request(&EventSink::discarding())
                .await
                .expect("request boundary")
                .is_some()
        );
        assert!(!agent.compaction.is_active());
    }

    /// Down to the safe buffer with no compaction in flight, the loop runs
    /// one synchronously before the request goes out.
    #[tokio::test]
    async fn reaching_the_buffer_runs_a_compaction_before_the_request() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut agent = agent_with_long_output(&llm, 396_000, dir.path()).await;
        // The fork the boundary starts is the compaction it then waits for.
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");

        assert!(request.is_some());
        assert!(!agent.compaction.is_active());
        assert!(
            agent.context().handoff().is_some(),
            "the request waited for the handoff and went out on the compacted window"
        );
        assert_eq!(llm.recorded().len(), 1);
    }

    /// The recovery of a failed compaction reaches the host on the run's
    /// stream, ahead of the turn that goes out on the compacted window.
    #[tokio::test]
    async fn a_run_streams_the_recovery_of_a_failed_compaction() {
        let llm = ScriptedLlm::new(
            1_000_000,
            [
                "bad".to_string(),
                "bad".to_string(),
                HANDOFF_JSON.to_string(),
                "done".to_string(),
            ],
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let mut agent = agent_with_long_output(&llm, 396_000, dir.path()).await;

        let events = drain_run(agent.run("keep going", [])).await;

        let recovery = events
            .iter()
            .position(|event| matches!(event, AgentEvent::CompactionRecovery { .. }))
            .expect("the host is told of the failure and the rerun");
        let reply = events
            .iter()
            .position(|event| matches!(event, AgentEvent::Text(text) if text == "done"))
            .expect("the turn goes out after the rerun");
        assert!(recovery < reply);
        assert!(events.iter().any(AgentEvent::is_complete));
    }

    /// A failed compaction strips the alerts and reminders, moves the long
    /// tool output to a file, and the synchronous rerun succeeds.
    #[tokio::test]
    async fn a_failed_compaction_recovers_on_the_cleaned_context() {
        let llm = ScriptedLlm::new(
            1_000_000,
            [
                "bad".to_string(),
                "bad".to_string(),
                HANDOFF_JSON.to_string(),
            ],
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let mut agent = agent_with_long_output(&llm, 396_000, dir.path()).await;

        let (sender, events) = async_channel::unbounded();
        let request = agent
            .prepare_request(&EventSink { sender })
            .await
            .expect("recovery succeeds");

        assert!(request.is_some());
        assert!(agent.context().handoff().is_some());
        let recoveries = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|timed| match timed.event {
                AgentEvent::CompactionRecovery {
                    error,
                    alerts_removed,
                    outputs_spilled,
                } => Some((error.to_string(), alerts_removed, outputs_spilled)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            recoveries.len(),
            1,
            "one failure, one rerun: {recoveries:?}"
        );
        let (error, alerts_removed, outputs_spilled) = &recoveries[0];
        assert!(
            error.contains("compaction"),
            "the event names the failure: {error}"
        );
        assert_eq!(
            (*alerts_removed, *outputs_spilled),
            (1, 1),
            "the event reports the cleanup"
        );
        let recorded = llm.recorded();
        assert_eq!(
            recorded.len(),
            3,
            "the failed attempt, its retry, the rerun"
        );
        let holds = |request: &LLMRequest, needle: &str| {
            request
                .messages()
                .iter()
                .any(|message| message.content().contains(needle))
        };
        assert!(holds(&recorded[0], "stale alert"));
        assert!(holds(&recorded[0], &"o".repeat(1_000)));
        let rerun = &recorded[2];
        assert!(!holds(rerun, "stale alert"), "alerts are stripped");
        assert!(
            !holds(rerun, &"o".repeat(10_000)),
            "the long output is gone from the window"
        );
        assert!(holds(rerun, "outputs/compaction-call_1.txt"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("outputs/compaction-call_1.txt"))
                .expect("the spilled output is readable"),
            "o".repeat(396_000 * 4)
        );
    }

    #[tokio::test]
    async fn stable_prefix_survives_an_appended_turn() {
        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 1_000_000,
            },
            AgentConfig::default(),
        );
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));

        let first = agent.context().stable_prefix_fingerprint();
        let before = agent.context().build_messages()[0].content().to_string();

        agent.push_message(Message::assistant("hi"));
        agent.push_message(Message::user("again"));

        assert_eq!(agent.context().stable_prefix_fingerprint(), first);
        assert_eq!(agent.context().build_messages()[0].content(), before);
    }

    #[tokio::test]
    async fn a_user_message_carries_its_utc_time_and_the_prefix_stays_put() {
        let clock = crate::ManualClock::at_utc("2026-11-04T12:00:00Z".parse().expect("timestamp"));
        let mut agent = Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .clock(clock.clone())
        .build();
        agent.ensure_initialized().await.expect("agent initializes");
        let prefix = agent.context().system_prefix().expect("prefix").clone();
        assert!(!prefix.contains("run `date`"), "{prefix}");

        let first = agent
            .preview_request_messages_with("hello", [], [])
            .await
            .expect("preview");
        assert_eq!(
            first.last().expect("user message").content(),
            "hello\n\nSent at 2026-11-04 12:00:00 UTC."
        );

        let message = agent.timestamped_user_message("hello");
        agent.push_message(message);
        clock.advance(Duration::from_secs(90));
        let message = agent.timestamped_user_message("again");
        agent.push_message(message);

        let sent = agent.context().build_messages();
        let contents: Vec<_> = sent
            .iter()
            .map(aither_core::llm::Message::content)
            .collect();
        assert!(
            contents.contains(&"hello\n\nSent at 2026-11-04 12:00:00 UTC."),
            "{contents:?}"
        );
        assert!(
            contents.contains(&"again\n\nSent at 2026-11-04 12:01:30 UTC."),
            "{contents:?}"
        );
        assert_eq!(agent.context().system_prefix().expect("prefix"), prefix);
    }

    #[tokio::test]
    async fn the_host_layout_places_and_replaces_built_in_blocks() {
        use crate::context::{ExternalContent, ExternalSlot, SystemLayout};
        const MEMORY: ExternalSlot = ExternalSlot::new("memory");

        let mut agent = Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .system_layout(SystemLayout::new(
            [
                InstructionSlot::PERMISSIONS,
                InstructionSlot::KNOWLEDGE_AND_TIME,
                InstructionSlot::WORKSPACE_FACTS,
                InstructionSlot::BASE_SYSTEM,
            ],
            [MEMORY],
        ))
        .external_content(&MEMORY, ExternalContent::data("memory", "remembered"))
        .system_named(&InstructionSlot::PERMISSIONS, "host permissions")
        .system_prompt("base")
        .build();
        agent.ensure_initialized().await.expect("agent initializes");

        let prefix = agent.context().system_prefix().expect("prefix");
        assert!(
            prefix.starts_with("<permissions>host permissions</permissions>\n<knowledge_and_time>")
        );
        let base = prefix
            .find("<base_system>base</base_system>")
            .expect("base");
        assert!(prefix.find("<workspace_facts>").expect("workspace facts") < base);
        assert!(base < prefix.find("<external-content-notice>").expect("notice"));
        assert!(!prefix.contains(include_str!("prompts/permissions.txt")));
    }

    #[test]
    fn permission_event_key_parses_terminal_modes() {
        let default_args = serde_json::to_value(TerminalArgs {
            description: "list the working directory".to_string(),
            script: "ls".to_string(),
            language: aither_sandbox::ScriptLanguage::Shell,
            mode: TerminalExecutionMode::Default,
            ssh_server_id: None,
            expect: aither_sandbox::OutputFormat::Text,
            resolution: aither_sandbox::MediaResolution::Auto,
            max_lines: 200,
            raw: false,
            timeout: 60,
            background: false,
        })
        .expect("default args should serialize");
        let unsafe_args = serde_json::to_value(TerminalArgs {
            description: "remove the demo directory".to_string(),
            script: "rm -rf /tmp/demo".to_string(),
            language: aither_sandbox::ScriptLanguage::Shell,
            mode: TerminalExecutionMode::Unsafe,
            ssh_server_id: None,
            expect: aither_sandbox::OutputFormat::Text,
            resolution: aither_sandbox::MediaResolution::Auto,
            max_lines: 200,
            raw: false,
            timeout: 60,
            background: false,
        })
        .expect("unsafe args should serialize");
        let sandboxed_args = serde_json::to_value(TerminalArgs {
            description: "print the current working directory".to_string(),
            script: "pwd".to_string(),
            language: aither_sandbox::ScriptLanguage::Shell,
            mode: TerminalExecutionMode::Sandboxed,
            ssh_server_id: None,
            expect: aither_sandbox::OutputFormat::Text,
            resolution: aither_sandbox::MediaResolution::Auto,
            max_lines: 200,
            raw: false,
            timeout: 60,
            background: false,
        })
        .expect("sandboxed args should serialize");

        let unsafe_key = permission_event_key_for_call("terminal", &unsafe_args)
            .expect("unsafe terminal mode should require permission routing");

        // Default and sandboxed no longer require approval: network is
        // available by default and only unsafe escalation is gated.
        assert!(permission_event_key_for_call("terminal", &default_args).is_none());
        assert_eq!(unsafe_key.mode, TerminalMode::Unsafe);
        assert_eq!(unsafe_key.script, "rm -rf /tmp/demo");
        assert!(permission_event_key_for_call("terminal", &sandboxed_args).is_none());
    }

    #[test]
    fn permission_pause_tracker_routes_waiting_and_resolved_events() {
        let tool_calls = vec![aither_core::llm::ToolCall::new(
            "tool-1",
            "terminal",
            serde_json::to_value(TerminalArgs {
                description: "remove the demo directory".to_string(),
                script: "rm -rf /tmp/demo".to_string(),
                language: aither_sandbox::ScriptLanguage::Shell,
                mode: TerminalExecutionMode::Unsafe,
                ssh_server_id: None,
                expect: aither_sandbox::OutputFormat::Text,
                resolution: aither_sandbox::MediaResolution::Auto,
                max_lines: 200,
                raw: false,
                timeout: 60,
                background: false,
            })
            .expect("tool args should serialize"),
        )];

        let mut tracker = PermissionPauseTracker::from_tool_calls(&tool_calls);
        let waiting = tracker
            .event_for(PermissionEvent {
                mode: TerminalMode::Unsafe,
                script: "rm -rf /tmp/demo".to_string(),
                language: aither_sandbox::ScriptLanguage::Shell,
                stage: PermissionEventStage::Waiting,
            })
            .expect("waiting event should map to tool call");
        let resolved = tracker
            .event_for(PermissionEvent {
                mode: TerminalMode::Unsafe,
                script: "rm -rf /tmp/demo".to_string(),
                language: aither_sandbox::ScriptLanguage::Shell,
                stage: PermissionEventStage::Resolved,
            })
            .expect("resolved event should map to tool call");

        assert_eq!(
            waiting,
            (RunPauseReason::PermissionRequest, "tool-1".to_string())
        );
        assert_eq!(
            resolved,
            (RunPauseReason::PermissionRequest, "tool-1".to_string())
        );
    }

    #[cfg(feature = "skills")]
    fn empty_checkpoint() -> AgentCheckpoint {
        AgentCheckpoint {
            context: crate::Context::default().checkpoint(),
            tool_surface_hash: "tool-surface".to_string(),
            context_window: ContextWindowSnapshot {
                phase: ContextWindowPhase::Stable,
                metrics: ContextWindowMetrics {
                    usage_fraction: 0.0,
                    context_window: 0,
                    estimated_tokens: 0,
                    chars_per_token: 4.0,
                    system_block_count: 0,
                    recent_message_count: 0,
                    recent_system_message_count: 0,
                    has_handoff: false,
                    has_staged_system_updates: false,
                    prompt_cache_epoch: 0,
                },
                messages: Vec::new(),
            },
            has_background_tasks: false,
            tasks: None,
        }
    }

    #[cfg(feature = "skills")]
    fn review_skill_registry() -> Arc<SkillRegistry> {
        let mut registry = SkillRegistry::new();
        registry.register(Skill {
            name: "code-review".to_string(),
            description: "Review code carefully".to_string(),
            instructions: "Use a review checklist.".to_string(),
            allowed_tools: Some(vec!["mock_tool".to_string()]),
            resources: HashMap::new(),
        });
        Arc::new(registry)
    }

    /// A source that renders one named block and records what it is told was
    /// sent.
    struct Recording {
        name: &'static str,
        committed: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl crate::context::ContextSource for Recording {
        fn render(&self) -> impl std::future::Future<Output = Option<Block>> + Send {
            std::future::ready(Some(Block::new(self.name, self.name)))
        }

        async fn commit(&mut self, block: &Block) {
            self.committed
                .lock()
                .expect("commit log")
                .push(block.content().to_string());
        }
    }

    #[tokio::test]
    async fn each_source_is_committed_only_the_block_it_rendered() {
        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 200_000,
            },
            AgentConfig::default(),
        );
        let logs: Vec<_> = ["alpha", "beta"]
            .into_iter()
            .map(|name| {
                let committed = Arc::new(std::sync::Mutex::new(Vec::new()));
                agent
                    .context_sources
                    .push(crate::context::AnyContextSource::new(Recording {
                        name,
                        committed: Arc::clone(&committed),
                    }));
                committed
            })
            .collect();

        let tail = agent.render_tail().await;
        assert_eq!(tail.len(), 2);
        agent.commit_tail(&tail).await;

        assert_eq!(*logs[0].lock().unwrap(), ["alpha"]);
        assert_eq!(*logs[1].lock().unwrap(), ["beta"]);
    }

    #[cfg(feature = "skills")]
    #[test]
    fn restore_checkpoint_succeeds_without_legacy_skill_state() {
        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 1000,
            },
            AgentConfig::default(),
        );
        agent.skill_registry = Some(review_skill_registry());

        agent
            .restore_checkpoint(empty_checkpoint())
            .expect("restore checkpoint must succeed");
    }

    /// A compaction forked from the window being replaced describes a
    /// conversation the restored window never had.
    #[tokio::test]
    async fn restoring_a_checkpoint_retires_a_landed_compaction() {
        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 200_000,
            },
            AgentConfig::default(),
        );
        agent.push_message(Message::user("before"));
        let checkpoint = agent.export_checkpoint().await.expect("checkpoint");
        agent.push_message(Message::user("after"));
        land_document(
            &mut agent.compaction,
            agent.context.fork_point(),
            HandoffDocument {
                current_task: "stale".to_string(),
                ..HandoffDocument::default()
            },
        );

        agent.restore_checkpoint(checkpoint).expect("restore");

        assert!(!agent.compaction.is_active(), "the landed handoff is gone");
    }

    #[tokio::test]
    async fn restoring_a_checkpoint_retires_a_running_compaction() {
        let llm = ScriptedLlm::new(200_000, [HANDOFF_JSON.to_string()]);
        let mut agent = Agent::with_config(llm.clone(), AgentConfig::default());
        agent.push_message(Message::user("before"));
        let checkpoint = agent.export_checkpoint().await.expect("checkpoint");
        agent.push_message(Message::user("after"));
        let tail = agent.render_tail().await;
        let request = LLMRequest::new(agent.context.build_request(&tail));
        let fork = CompactionFork::<HandoffDocument>::new(
            &request,
            agent.context.fork_point(),
            &[],
            None,
            &[],
        )
        .expect("fork");
        agent.compaction.start(PendingCompaction::start(fork, llm));
        agent.compaction_failure = Some(crate::context::CompactionError::Empty);
        agent.idle_compacted = true;

        agent.restore_checkpoint(checkpoint).expect("restore");

        assert!(!agent.compaction.is_active(), "the running fork is retired");
        assert!(agent.compaction_failure.is_none());
        assert!(!agent.idle_compacted);
    }

    #[cfg(feature = "skills")]
    #[tokio::test]
    async fn activate_skills_for_prompt_is_disabled() {
        let mut registry = SkillRegistry::new();
        registry.register(Skill {
            name: "code-review".to_string(),
            description: "Review code carefully".to_string(),
            instructions: "Use a review checklist.".to_string(),
            allowed_tools: Some(vec!["mock_tool".to_string()]),
            resources: HashMap::new(),
        });

        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 1000,
            },
            AgentConfig::default(),
        );
        agent.skill_registry = Some(Arc::new(registry));
        let events = agent.activate_skills_for_prompt("please review this patch");
        assert!(events.is_empty());
        assert!(agent.active_skills.is_empty());
        assert!(agent.active_allowed_tools.is_none());
    }

    #[cfg(feature = "skills")]
    #[tokio::test]
    async fn requests_omit_the_skill_resource_catalog() {
        let mut registry = SkillRegistry::new();
        registry.register(Skill {
            name: "code-review".to_string(),
            description: "Review code carefully".to_string(),
            instructions: "Use a review checklist.".to_string(),
            allowed_tools: Some(vec!["mock_tool".to_string()]),
            resources: HashMap::from([(
                "templates/review.md".to_string(),
                "# Review template".to_string(),
            )]),
        });

        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 200_000,
            },
            AgentConfig::default(),
        );
        agent.skill_registry = Some(Arc::new(registry));
        let _ = agent.activate_skills_for_prompt("please review this patch");
        agent.ensure_initialized().await.expect("agent initializes");

        let messages = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("a fresh window must accept a request")
            .messages()
            .to_vec();
        assert!(
            messages
                .iter()
                .all(|message| !message.content().contains("templates/review.md")),
            "skill resource catalog must not be injected automatically"
        );
    }

    /// A `terminal` call that ran the kept IPC command `ask_user` survives a
    /// handoff with its result; one that ran an ordinary command does not,
    /// and the native `ask_user` tool survives by its name at commit.
    #[tokio::test]
    async fn exchanges_of_kept_commands_and_native_ask_user_survive_a_handoff() {
        use aither_core::llm::{AgentIdentity, CallIdentity, ToolCallId, ToolResult};
        use aither_sandbox::ToolRegistryBuilder;

        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct NoArgs {}
        struct Command(&'static str);
        impl aither_core::llm::Tool for Command {
            fn name(&self) -> std::borrow::Cow<'static, str> {
                self.0.into()
            }
            type Arguments = NoArgs;
            type Res = ToolResult;
            fn call(
                &self,
                _: Self::Arguments,
                _: aither_core::llm::ToolContext,
            ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send
            {
                std::future::ready(Ok(ToolResult::text("done")))
            }
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let mut builder = ToolRegistryBuilder::new();
        builder.kept_through_compaction(|kept| kept.configure_tool(Command("ask_user")));
        builder.configure_tool(Command("ls"));
        let registry = builder.build(dir.path());
        let in_call = |id: &str| {
            registry.for_call(CallIdentity::new(ToolCallId::new(id), AgentIdentity::Main))
        };
        assert!(in_call("kept").query_tool_handler("ask_user", &[]).await.ok);
        assert!(in_call("plain").query_tool_handler("ls", &[]).await.ok);

        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.kept_receiver = Some(registry.kept_receiver());
        let call = |id: &str, name: &str| ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::json!({}),
            reasoning_state: None,
        };
        agent.push_message(Message::assistant_with_tool_calls(
            "",
            vec![
                call("kept", "terminal"),
                call("plain", "terminal"),
                call("native", "ask_user"),
            ],
        ));
        for (id, name) in [
            ("kept", "terminal"),
            ("plain", "terminal"),
            ("native", "ask_user"),
        ] {
            agent.tool_results.push_back(Ok(FinishedToolCall {
                id: id.into(),
                name: name.into(),
                result: ToolResult::text(id),
                unchanged_read: false,
            }));
        }
        let (sender, _events) = async_channel::unbounded();
        agent
            .apply_tool_results(&EventSink { sender })
            .expect("results apply");

        let fork = agent.context.fork_point();
        agent.context.apply_handoff(Handoff::new("task"), fork, 0);

        let kept = agent.context.recent();
        let ids: Vec<&str> = kept[0].tool_calls().iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["kept", "native"]);
        let results: Vec<&str> = kept[1..].iter().map(|m| m.content()).collect();
        assert_eq!(results, ["kept", "native"]);
    }

    fn tasks_agent(dir: &std::path::Path) -> Agent<MockLlm, MockLlm, MockLlm, ()> {
        tasks_agent_on(dir, crate::tasks::TaskEvents::new())
    }

    fn tasks_agent_on(
        dir: &std::path::Path,
        events: crate::tasks::TaskEvents,
    ) -> Agent<MockLlm, MockLlm, MockLlm, ()> {
        Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .sandbox_dir(dir)
        .task_events(events)
        .build()
    }

    /// An event the host records on the inbox it gave the builder reaches
    /// the agent's tasks document, and its alert the conversation, once.
    #[tokio::test]
    async fn tool_turn_alerts_from_sources_reach_the_conversation() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "## Doing\n- #3 PR #12 open\n")
            .expect("write tasks.md");
        let events = crate::tasks::TaskEvents::new();
        let mut agent = tasks_agent_on(dir.path(), events.clone());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_request_alerts().await;

        events.record(crate::tasks::TaskEvent::PullRequestMerged(
            crate::tasks::WorkRef::pull_request(12),
        ));
        for _ in 0..3 {
            agent.push_request_alerts().await;
            agent.push_tool_turn_alerts().await;
        }

        let alerts = agent
            .context()
            .conversation_messages()
            .iter()
            .filter(|message| message.content().contains("PR #12 was merged"))
            .count();
        assert_eq!(alerts, 1, "an unrecorded event alerts exactly once");
    }

    #[tokio::test]
    async fn final_turn_alerts_keep_the_loop_running() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "## Doing\n- pending\n")
            .expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");

        assert!(
            agent.push_final_turn_alerts().await,
            "an actionable row must object to the model stopping"
        );
    }

    /// A host with a message waiting ends the run a stop-gate alert would
    /// keep going, so the message reaches the next request together with the
    /// alert, which stays in the conversation.
    #[tokio::test]
    async fn a_waiting_message_ends_the_run_a_final_turn_alert_keeps_going() {
        use tempfile::tempdir;

        struct MessageWaiting;
        impl Hook for MessageWaiting {
            fn on_turn_boundary(
                &self,
                _ctx: &TurnBoundaryContext<'_>,
            ) -> impl std::future::Future<Output = TurnBoundaryAction> + Send {
                std::future::ready(TurnBoundaryAction::EndTurn)
            }
        }

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "## Doing\n- pending\n")
            .expect("write tasks.md");
        let mut agent = Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .sandbox_dir(dir.path())
        .hook(MessageWaiting)
        .build();
        agent.ensure_initialized().await.expect("agent initializes");

        let control = agent
            .finish_no_tool_turn("done".to_string(), 1, &EventSink::discarding())
            .await
            .expect("the turn finishes");

        assert!(
            matches!(
                control,
                TurnControl::Break {
                    stop_reason: StopReason::EndTurn,
                    ..
                }
            ),
            "a waiting message must end the run"
        );
        let messages = agent.context().conversation_messages();
        assert!(
            messages
                .iter()
                .any(|message| message.content().contains("tasks.md still has rows")),
            "the alert stays for the next request: {messages:?}"
        );
    }

    #[tokio::test]
    async fn system_alerts_reach_the_next_request() {
        use aither_core::llm::Role;

        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_system_alert("no native tool named foo");

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("a fresh window must accept a request");

        assert!(
            request.messages().iter().any(|message| {
                message.role() == Role::System
                    && message.content().contains("no native tool named foo")
            }),
            "ad-hoc alerts must reach the next request"
        );
    }

    /// A host load event is injected once, at the next request, and nothing
    /// restates it on the requests that follow.
    #[tokio::test]
    async fn host_load_events_are_injected_once() {
        use aither_core::llm::Role;

        async fn mentions(agent: &mut Agent<MockLlm, MockLlm, MockLlm, ()>, text: &str) -> usize {
            let request = agent
                .prepare_request(&EventSink::discarding())
                .await
                .expect("request boundary")
                .expect("a fresh window must accept a request");
            request
                .messages()
                .iter()
                .filter(|message| {
                    message.role() == Role::System && message.content().contains(text)
                })
                .count()
        }

        let (events, receiver) = async_channel::unbounded();
        let mut agent = Agent::new(MockLlm {
            context_length: 200_000,
        });
        agent.host_load_events = Some(receiver);
        agent.ensure_initialized().await.expect("agent initializes");

        assert_eq!(mentions(&mut agent, "load is high").await, 0);
        events
            .send(crate::HostLoadEvent::LoadHigh {
                load: crate::MachineLoad(1.8),
                foreground_wait: Duration::from_secs(90),
            })
            .await
            .expect("agent listens");
        assert_eq!(mentions(&mut agent, "load is high").await, 1);
        assert_eq!(mentions(&mut agent, "load is high").await, 1);
        assert_eq!(mentions(&mut agent, "load is high").await, 1);

        events
            .send(crate::HostLoadEvent::LoadEnded {
                load: crate::MachineLoad(0.2),
                foreground_wait: Duration::from_secs(10),
            })
            .await
            .expect("agent listens");
        assert_eq!(mentions(&mut agent, "load has ended").await, 1);
        assert_eq!(mentions(&mut agent, "load has ended").await, 1);
        assert_eq!(mentions(&mut agent, "load is high").await, 1);
    }

    /// Notices that reword the host load and job alert families and keep
    /// aither's wording for the rest.
    struct Reworded;

    impl aither_sandbox::SandboxNotices for Reworded {}

    impl crate::Notices for Reworded {
        fn host_load(&self, event: &crate::HostLoadEvent) -> String {
            match event {
                crate::HostLoadEvent::LoadHigh { load, .. } => {
                    format!("Reworded: the host runs at {}%.", load.percent())
                }
                _ => crate::DefaultNotices.host_load(event),
            }
        }

        fn job_alert(&self, alert: &JobAlert) -> String {
            match alert {
                JobAlert::StdinBlocked(job) => format!("Reworded: {} waits on stdin.", job.task_id),
                _ => crate::DefaultNotices.job_alert(alert),
            }
        }
    }

    /// The system messages of the next request that contain `text`.
    async fn system_mentions(agent: &mut Agent<MockLlm>, text: &str) -> usize {
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("a fresh window must accept a request");
        request
            .messages()
            .iter()
            .filter(|message| {
                message.role() == aither_core::llm::Role::System && message.content().contains(text)
            })
            .count()
    }

    #[tokio::test]
    async fn a_host_load_alert_reaches_the_conversation_in_the_embedders_words() {
        let (events, receiver) = async_channel::unbounded();
        let mut agent = Agent::builder(MockLlm {
            context_length: 200_000,
        })
        .notices(crate::AnyNotices::new(Reworded))
        .build();
        agent.host_load_events = Some(receiver);
        agent.ensure_initialized().await.expect("agent initializes");

        events
            .send(crate::HostLoadEvent::LoadHigh {
                load: crate::MachineLoad(1.8),
                foreground_wait: Duration::from_secs(90),
            })
            .await
            .expect("agent listens");
        assert_eq!(
            system_mentions(&mut agent, "Reworded: the host runs at 180%.").await,
            1
        );
        assert_eq!(system_mentions(&mut agent, "load is high").await, 0);
    }

    #[tokio::test]
    async fn a_job_alert_reaches_the_conversation_in_the_embedders_words() {
        let (sender, receiver) = aither_sandbox::background_task_channel();
        let mut agent = Agent::builder(MockLlm {
            context_length: 200_000,
        })
        .notices(crate::AnyNotices::new(Reworded))
        .build();
        agent.background_receiver = Some(receiver);
        agent.ensure_initialized().await.expect("agent initializes");

        sender
            .send_alert(JobAlert::StdinBlocked(aither_sandbox::StdinBlockedJob {
                task_id: "amber-forest-thunder-pearl".to_string(),
                script: "cat".to_string(),
            }))
            .await;
        agent.push_completed_background_events(&EventSink::discarding());
        assert_eq!(
            system_mentions(
                &mut agent,
                "Reworded: amber-forest-thunder-pearl waits on stdin."
            )
            .await,
            1
        );
        assert_eq!(system_mentions(&mut agent, "terminal_input").await, 0);
    }

    /// A job blocked on stdin can never complete, so its alert must end the
    /// post-turn wait and resume the loop — otherwise the model never sees
    /// it until the job's own timeout kills it.
    #[tokio::test]
    async fn stdin_blocked_alert_wakes_the_parked_background_wait() {
        let (sender, receiver) = aither_sandbox::background_task_channel();
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.background_receiver = Some(receiver);
        agent.ensure_initialized().await.expect("agent initializes");
        sender.task_started();

        let (tx, _rx) = async_channel::unbounded();
        let sink = EventSink { sender: tx };

        let alerter = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            sender
                .send_alert(JobAlert::StdinBlocked(aither_sandbox::StdinBlockedJob {
                    task_id: "task-test".to_string(),
                    script: "cat".to_string(),
                }))
                .await;
            // The job finishes a while after the model turn the alert resumed.
            tokio::time::sleep(Duration::from_millis(1_500)).await;
            sender.task_finished();
        });

        let started = Instant::now();
        let woke = tokio::time::timeout(
            Duration::from_secs(20),
            agent.collect_final_background_events("run-test", 1, &sink),
        )
        .await
        .expect("the wait must end with the job")
        .expect("collecting final events should succeed");
        assert!(woke, "a stdin-blocked alert must resume the run loop");
        assert!(
            started.elapsed() >= Duration::from_millis(1_400),
            "the model turn the alert woke must go back to waiting for the job, not end the run"
        );
        alerter.await.expect("the alerter should send");

        assert!(
            agent
                .context()
                .recent()
                .iter()
                .any(|message| message.content().contains("waiting for input on stdin")),
            "the alert must be durable context for the resumed turn"
        );
    }

    /// A model that finishes the background job while it answers its first
    /// request, as a job fed through `terminal_input` does a moment later.
    #[derive(Clone)]
    struct FinishesJobLlm {
        job: async_channel::Receiver<aither_sandbox::BackgroundTaskSender>,
    }

    impl LanguageModel for FinishesJobLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let job = self.job.try_recv().ok();
            futures_lite::StreamExt::filter_map(
                futures_lite::stream::once_future(async move {
                    if let Some(sender) = job {
                        sender
                            .send_completed(aither_sandbox::CompletedTask {
                                task_id: "task-test".to_string(),
                                script: "cat".to_string(),
                                kill: None,
                                timeout_kill: None,
                                pending: None,
                                result: Err(aither_sandbox::TerminalError::Execution(
                                    "fed and finished".to_string(),
                                )),
                            })
                            .await;
                        sender.task_finished();
                    }
                }),
                |()| None,
            )
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                1_000_000,
            )))
        }
    }

    /// The job the alert woke the model for finishes while the model replies:
    /// the wait goes back to the receiver and collects that completion before
    /// the run ends, instead of leaving it for the next user message.
    #[tokio::test]
    async fn job_finishing_during_the_alerted_turn_is_collected() {
        let (sender, receiver) = aither_sandbox::background_task_channel();
        let (job_tx, job_rx) = async_channel::bounded(1);
        let mut agent = Agent::new(FinishesJobLlm { job: job_rx });
        agent.background_receiver = Some(receiver);
        agent.ensure_initialized().await.expect("agent initializes");
        // The model already knows this job: `terminal` detached it earlier
        // and its started entry was drained into `detached_task_ids`.
        agent.detached_task_ids.insert("task-test".to_string());
        sender.task_started();
        sender
            .send_alert(JobAlert::StdinBlocked(aither_sandbox::StdinBlockedJob {
                task_id: "task-test".to_string(),
                script: "cat".to_string(),
            }))
            .await;
        job_tx.try_send(sender).expect("hand the job to the model");

        let (tx, _rx) = async_channel::unbounded();
        let sink = EventSink { sender: tx };
        let woke = tokio::time::timeout(
            Duration::from_secs(20),
            agent.collect_final_background_events("run-test", 1, &sink),
        )
        .await
        .expect("the wait must end once the job is collected")
        .expect("collecting final events should succeed");
        assert!(woke, "a stdin-blocked alert must resume the run loop");
        assert!(
            agent
                .context()
                .recent()
                .iter()
                .any(|message| message.content().contains("fed and finished")),
            "the completion that landed during the resumed turn must be in context"
        );
    }

    #[tokio::test]
    async fn a_preview_does_not_change_the_window() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] pending\n").expect("write tasks.md");
        let mut agent = Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .sandbox_dir(dir.path())
        .build();
        agent.ensure_initialized().await.expect("agent initializes");

        let before = agent.context().checkpoint();
        let preview = agent
            .preview_request_messages("hi", std::iter::empty())
            .await
            .expect("preview should initialize");

        assert!(
            preview
                .iter()
                .any(|message| message.content().contains("hi")),
            "the preview must contain the prompt it previews"
        );
        assert_eq!(
            agent.context().checkpoint(),
            before,
            "a preview must leave the window untouched"
        );
    }

    #[tokio::test]
    async fn preview_with_extras_leaves_uninitialized_agent_in_first_turn_state() {
        use aither_core::llm::Role;

        let mut previewed = Agent::builder(MockLlm {
            context_length: 200_000,
        })
        .build();
        let mut control = Agent::builder(MockLlm {
            context_length: 200_000,
        })
        .build();
        // A normal first assembly initializes the agent and populates the
        // static system blocks into its context.
        control
            .snapshot_context_window()
            .await
            .expect("control agent initializes");

        let messages = previewed
            .preview_request_messages_with(
                "hi",
                [Message::system("hidden note")],
                std::iter::empty(),
            )
            .await
            .expect("preview should initialize");

        // The extra message lands at the conversation tail ahead of the
        // prompt, matching the run ordering when a host pushes it first.
        let hidden_pos = messages
            .iter()
            .position(|message| {
                message.role() == Role::System && message.content() == "hidden note"
            })
            .expect("extra message must appear in the previewed request");
        assert_eq!(messages[hidden_pos + 1].role(), Role::User);
        assert!(
            messages[hidden_pos + 1]
                .content()
                .starts_with("hi\n\nSent at ")
        );

        // Post-preview state is exactly what a normal first turn leaves:
        // initialized with the static blocks populated, and neither the
        // prompt nor the extra message persisted.
        assert!(previewed.initialized);
        assert_eq!(
            previewed.context().instructions(),
            control.context().instructions()
        );
        assert_eq!(previewed.context().external(), control.context().external());
        assert!(previewed.context().is_conversation_empty());
    }

    /// A mock whose `respond` replays scripted text and records requests,
    /// so tests can drive compaction and inspect what was sent.
    #[derive(Clone)]
    struct ScriptedLlm {
        context_length: u32,
        abilities: Vec<Ability>,
        default_effort: Option<aither_core::llm::model::ReasoningEffort>,
        requests: Arc<std::sync::Mutex<Vec<LLMRequest>>>,
        responses: Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    }

    impl ScriptedLlm {
        fn new(context_length: u32, responses: impl IntoIterator<Item = String>) -> Self {
            Self {
                context_length,
                abilities: Vec::new(),
                default_effort: None,
                requests: Arc::new(std::sync::Mutex::new(Vec::new())),
                responses: Arc::new(std::sync::Mutex::new(responses.into_iter().collect())),
            }
        }

        /// The same mock, reporting `ability` in its profile.
        fn with_ability(mut self, ability: Ability) -> Self {
            self.abilities.push(ability);
            self
        }

        /// The same mock, reporting `effort` as its default in its profile.
        const fn with_default_effort(
            mut self,
            effort: aither_core::llm::model::ReasoningEffort,
        ) -> Self {
            self.default_effort = Some(effort);
            self
        }

        fn recorded(&self) -> Vec<LLMRequest> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    impl LanguageModel for ScriptedLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests.lock().expect("requests lock").push(request);
            let text = self
                .responses
                .lock()
                .expect("responses lock")
                .pop_front()
                .unwrap_or_default();
            futures_lite::stream::iter([Ok(Event::Text(text))])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            let context_length = self.context_length;
            let mut profile =
                ModelProfile::new("mock", "test", "mock-model", "mock model", context_length)
                    .with_abilities(self.abilities.clone());
            profile.default_reasoning_effort = self.default_effort;
            std::future::ready(Ok(profile))
        }
    }

    use aither_core::llm::model::ReasoningEffort;

    /// Parameters asking for `effort` and nothing else.
    fn effort(effort: ReasoningEffort) -> Parameters {
        Parameters::default().reasoning_effort(effort)
    }

    /// The index of the user message whose text starts with `prompt`.
    fn user_message_index(request: &LLMRequest, prompt: &str) -> usize {
        request
            .messages()
            .iter()
            .position(|message| {
                message.role() == Role::User && message.content().starts_with(prompt)
            })
            .unwrap_or_else(|| panic!("the request carries the user message {prompt:?}"))
    }

    /// An effort change on a model that takes per-message updates leaves the
    /// request-level effort and every earlier message as the first request
    /// sent them, and carries the change as an update before the next user
    /// message.
    #[tokio::test]
    async fn an_effort_change_on_a_per_message_model_keeps_the_prefix() {
        let llm = ScriptedLlm::new(1_000_000, ["one".to_string(), "two".to_string()])
            .with_ability(Ability::PerMessageEffort);
        let mut agent = Agent::builder(llm.clone())
            .request_parameters(effort(ReasoningEffort::High))
            .build();
        agent.query("first").await.expect("the first turn runs");
        agent.set_parameters(ModelTier::Advanced, effort(ReasoningEffort::Low));
        agent.query("second").await.expect("the second turn runs");

        let [first, second]: [LLMRequest; 2] =
            llm.recorded().try_into().expect("two turns, two requests");
        assert_eq!(
            first.parameters().reasoning_effort,
            Some(ReasoningEffort::High)
        );
        assert_eq!(
            second.parameters().reasoning_effort,
            Some(ReasoningEffort::High),
            "the request-level effort stays as first sent"
        );
        let end = user_message_index(&first, "first");
        assert_eq!(
            first.messages()[..=end],
            second.messages()[..=end],
            "the system prefix and earlier messages are unchanged"
        );
        let user = user_message_index(&second, "second");
        assert_eq!(
            second.messages()[user - 1],
            Message::configuration(ReasoningEffort::Low),
            "the update precedes the next user message"
        );
        assert_eq!(
            second
                .messages()
                .iter()
                .filter(|message| message.role() == Role::Configuration)
                .count(),
            1
        );
    }

    /// Clearing the effort after a Low update, on a model that takes
    /// per-message updates, sends an update to the model's default that
    /// overrides Low, with the request-level effort and every earlier message
    /// unchanged. The defaults are the documented ones of Claude Opus 5.5 and
    /// GPT-6 Sol, as their clients report them.
    #[tokio::test]
    async fn clearing_the_effort_after_an_update_restores_the_models_default() {
        for (model, default) in [
            ("claude-opus-5-5", ReasoningEffort::Medium),
            ("gpt-6-sol", ReasoningEffort::Medium),
        ] {
            let llm = ScriptedLlm::new(
                1_000_000,
                ["one".to_string(), "two".to_string(), "three".to_string()],
            )
            .with_ability(Ability::PerMessageEffort)
            .with_default_effort(default);
            let mut agent = Agent::builder(llm.clone())
                .request_parameters(effort(ReasoningEffort::High))
                .build();
            agent.query("first").await.expect("the first turn runs");
            agent.set_parameters(ModelTier::Advanced, effort(ReasoningEffort::Low));
            agent.query("second").await.expect("the second turn runs");
            agent.set_parameters(ModelTier::Advanced, Parameters::default());
            agent.query("third").await.expect("the third turn runs");

            let [_, low, cleared]: [LLMRequest; 3] = llm
                .recorded()
                .try_into()
                .expect("three turns, three requests");
            assert_eq!(
                cleared.parameters().reasoning_effort,
                Some(ReasoningEffort::High),
                "{model}: the request-level effort stays as first sent"
            );
            let end = user_message_index(&low, "second");
            assert_eq!(
                low.messages()[..=end],
                cleared.messages()[..=end],
                "{model}: earlier messages are unchanged"
            );
            let user = user_message_index(&cleared, "third");
            assert_eq!(
                cleared.messages()[user - 1],
                Message::configuration(default),
                "{model}: the update to the default precedes the next user message"
            );
            let updates: Vec<_> = cleared
                .messages()
                .iter()
                .filter_map(Message::configured_effort)
                .collect();
            assert_eq!(
                updates.last(),
                Some(&default),
                "{model}: the default overrides Low"
            );
        }
    }

    /// Clearing the effort on a per-message model whose default is not
    /// recorded fails rather than leaving the earlier update in force.
    #[tokio::test]
    async fn clearing_the_effort_without_a_known_default_fails() {
        let llm = ScriptedLlm::new(1_000_000, ["one".to_string(), "two".to_string()])
            .with_ability(Ability::PerMessageEffort);
        let mut agent = Agent::builder(llm.clone())
            .request_parameters(effort(ReasoningEffort::High))
            .build();
        agent.query("first").await.expect("the first turn runs");
        agent.set_parameters(ModelTier::Advanced, effort(ReasoningEffort::Low));
        agent.query("second").await.expect("the second turn runs");
        agent.set_parameters(ModelTier::Advanced, Parameters::default());

        let error = agent
            .query("third")
            .await
            .expect_err("no update can undo Low");
        assert!(matches!(error, AgentError::Config(_)), "{error}");
        assert_eq!(llm.recorded().len(), 2, "nothing more is sent");
    }

    /// Without per-message updates the new effort goes at the request level.
    #[tokio::test]
    async fn an_effort_change_elsewhere_moves_the_request_level_effort() {
        let llm = ScriptedLlm::new(1_000_000, ["one".to_string(), "two".to_string()]);
        let mut agent = Agent::builder(llm.clone())
            .request_parameters(effort(ReasoningEffort::High))
            .build();
        agent.query("first").await.expect("the first turn runs");
        agent.set_parameters(ModelTier::Advanced, effort(ReasoningEffort::Low));
        agent.query("second").await.expect("the second turn runs");

        let [_, second]: [LLMRequest; 2] =
            llm.recorded().try_into().expect("two turns, two requests");
        assert_eq!(
            second.parameters().reasoning_effort,
            Some(ReasoningEffort::Low)
        );
        assert!(
            second
                .messages()
                .iter()
                .all(|message| message.role() != Role::Configuration),
            "no update reaches a model that cannot read it"
        );
    }

    /// A compaction restates the effort in force: the update it would drop
    /// is kept after the handoff.
    #[tokio::test]
    async fn a_compaction_restates_the_effort_in_force() {
        let llm = ScriptedLlm::new(
            1_000_000,
            [
                "one".to_string(),
                "two".to_string(),
                HANDOFF_JSON.to_string(),
                "three".to_string(),
            ],
        )
        .with_ability(Ability::PerMessageEffort);
        let mut agent = Agent::builder(llm.clone())
            .request_parameters(effort(ReasoningEffort::High))
            .build();
        agent.query("first").await.expect("the first turn runs");
        agent.set_parameters(ModelTier::Advanced, effort(ReasoningEffort::Low));
        agent.query("second").await.expect("the second turn runs");
        agent.compact(None).await.expect("compaction succeeds");
        agent.query("third").await.expect("the third turn runs");

        let after = llm.recorded().pop().expect("the turn after compaction");
        assert_eq!(
            after.parameters().reasoning_effort,
            Some(ReasoningEffort::High)
        );
        let updates: Vec<_> = after
            .messages()
            .iter()
            .filter_map(Message::configured_effort)
            .collect();
        assert_eq!(updates, [ReasoningEffort::Low], "the update survives once");
    }

    /// An embedder's handoff: decisions quoted verbatim, files named by path.
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct QuotedHandoff {
        goal: String,
        quoted_decisions: Vec<String>,
        files: Vec<String>,
    }

    impl crate::HandoffSchema for QuotedHandoff {
        fn instruction(prompt: &crate::HandoffPrompt<'_>) -> String {
            assert!(prompt.focus.is_none(), "no focus was given");
            "Write the quoted handoff.".to_string()
        }

        fn retry(reason: &str) -> String {
            reason.to_owned()
        }

        fn validate(&self) -> Result<(), crate::InvalidHandoff> {
            if self.quoted_decisions.is_empty() {
                return Err(crate::InvalidHandoff::new("no decision was quoted"));
            }
            Ok(())
        }

        fn render(&self, file_index: &[String]) -> String {
            std::iter::once(self.goal.as_str())
                .chain(self.quoted_decisions.iter().map(String::as_str))
                .chain(file_index.iter().map(String::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        }

        fn into_restoration(self) -> crate::Restoration {
            crate::Restoration {
                context_files: self
                    .files
                    .into_iter()
                    .map(|path| crate::ContextFile {
                        path,
                        lines: None,
                        why: "named by the handoff".to_string(),
                    })
                    .collect(),
                skills: Vec::new(),
            }
        }
    }

    /// A compaction with the embedder's own handoff asks with its schema and
    /// instruction, retries with its prompt after its validation fails,
    /// renders its text into the window, restores the files it named, and
    /// survives a checkpoint.
    #[tokio::test]
    async fn a_custom_handoff_round_trips_through_compaction_and_restore() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("notes.txt"), "restored marker text").expect("write");
        let llm = ScriptedLlm::new(
            1_000_000,
            [
                r#"{"goal":"ship the parser","quoted_decisions":[],"files":[]}"#.to_string(),
                r#"{"goal":"ship the parser","quoted_decisions":["\"use the PEG grammar\""],"files":["notes.txt"]}"#
                    .to_string(),
            ],
        );
        let mut agent = Agent::builder(llm.clone())
            .handoff::<QuotedHandoff>()
            .build();
        agent.restore_dir = Some(dir.path().to_path_buf());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("build the parser"));

        let result = agent
            .compact(None)
            .await
            .expect("compaction succeeds")
            .expect("there was a conversation to compact");

        let requests = llm.recorded();
        assert_eq!(requests.len(), 2, "one attempt and one retry");
        let asked: Vec<&str> = requests[0]
            .messages()
            .iter()
            .map(Message::content)
            .collect();
        assert!(asked.contains(&"Write the quoted handoff."), "{asked:?}");
        assert!(
            asked
                .iter()
                .any(|content| content.contains("quoted_decisions")),
            "the structured output carries the embedder's schema: {asked:?}"
        );
        assert_eq!(
            requests[1].messages().last().map(Message::content),
            Some("no decision was quoted"),
            "the retry is the embedder's, quoting its validation"
        );

        assert_eq!(result.summary, "ship the parser\n\"use the PEG grammar\"");
        let handoff = agent
            .context()
            .handoff()
            .cloned()
            .expect("the handoff is in the window");
        assert_eq!(handoff.text(), result.summary);
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after compaction");
        let sent: Vec<&str> = request.messages().iter().map(Message::content).collect();
        assert!(
            sent.iter()
                .any(|content| content.contains("\"use the PEG grammar\"")),
            "{sent:?}"
        );
        assert!(
            sent.iter()
                .any(|content| content.contains("restored marker text")),
            "the file the handoff named is restored: {sent:?}"
        );

        let checkpoint = agent.export_checkpoint().await.expect("checkpoint");
        let json = serde_json::to_string(&checkpoint).expect("checkpoint serializes");
        let checkpoint: crate::AgentCheckpoint =
            serde_json::from_str(&json).expect("checkpoint deserializes");
        let mut restored = Agent::builder(llm).handoff::<QuotedHandoff>().build();
        restored
            .restore_checkpoint(checkpoint)
            .expect("checkpoint restores");
        assert_eq!(restored.context().handoff(), Some(&handoff));
    }

    /// A replaced model takes the next request, with the history the old one
    /// built; a model that reads no updates gets the effort at the request
    /// level and no update.
    #[tokio::test]
    async fn a_replaced_model_takes_the_next_request_with_the_history() {
        let old = ScriptedLlm::new(1_000_000, ["one".to_string(), "two".to_string()])
            .with_ability(Ability::PerMessageEffort);
        let new = ScriptedLlm::new(500_000, ["three".to_string()]);
        let mut agent = Agent::builder(old.clone())
            .request_parameters(effort(ReasoningEffort::High))
            .build();
        agent.query("first").await.expect("the first turn runs");
        agent.set_parameters(ModelTier::Advanced, effort(ReasoningEffort::Low));
        agent.query("second").await.expect("the second turn runs");

        agent.set_advanced_model(new.clone());
        agent
            .query("third")
            .await
            .expect("the turn on the new model runs");

        assert_eq!(old.recorded().len(), 2, "the old model is asked no more");
        let [request]: [LLMRequest; 1] = new.recorded().try_into().expect("one request");
        for prompt in ["first", "second", "third"] {
            user_message_index(&request, prompt);
        }
        assert_eq!(
            request.parameters().reasoning_effort,
            Some(ReasoningEffort::Low)
        );
        assert!(
            request
                .messages()
                .iter()
                .all(|message| message.role() != Role::Configuration)
        );
        assert_eq!(
            agent.profile().map(|profile| profile.context_length),
            Some(500_000),
            "the window is the new model's"
        );
    }

    const HANDOFF_JSON: &str = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#;

    /// The prompt prefix the agent reports for its next request.
    async fn next_prompt_prefix(agent: &mut Agent<ScriptedLlm>) -> PromptPrefix {
        let (sender, events) = async_channel::unbounded();
        agent
            .prepare_request(&EventSink { sender })
            .await
            .expect("request boundary")
            .expect("a request goes out");
        std::iter::from_fn(|| events.try_recv().ok())
            .find_map(|timed| match timed.event {
                AgentEvent::PromptPrefix(prefix) => Some(prefix),
                _ => None,
            })
            .expect("every request reports its prompt prefix")
    }

    /// A conversation that only grows keeps the previous request's cached
    /// prefix, and only the agent's first request says it is the first.
    #[tokio::test]
    async fn a_growing_conversation_keeps_its_prompt_prefix() {
        let llm = ScriptedLlm::new(1_000_000, []);
        let mut agent = Agent::new(llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        let first = next_prompt_prefix(&mut agent).await;
        agent.push_message(Message::assistant("hi"));
        agent.push_message(Message::user("and again"));
        let second = next_prompt_prefix(&mut agent).await;

        assert!(first.first_request);
        assert!(!second.first_request);
        assert_eq!(second.chain.first_divergence(&first.chain), None);
        assert_eq!(second.rewrites, PrefixRewrites::default());
    }

    /// A compaction is reported once, with the next request, and that
    /// request no longer repeats the conversation it replaced.
    #[tokio::test]
    async fn a_compaction_is_reported_with_the_next_request() {
        let llm = ScriptedLlm::new(200_000, [HANDOFF_JSON.to_string()]);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut agent = agent_with_long_output(&llm, 100_000, dir.path()).await;
        let before = next_prompt_prefix(&mut agent).await;
        agent
            .compact(None)
            .await
            .expect("compaction runs")
            .expect("there is something to compact");
        let after = next_prompt_prefix(&mut agent).await;
        let later = next_prompt_prefix(&mut agent).await;

        assert!(after.rewrites.compacted);
        assert!(matches!(
            after.chain.first_divergence(&before.chain),
            Some(crate::context::PrefixBlock::Message { .. })
        ));
        assert!(!later.rewrites.compacted, "reported once");
    }

    /// A tool registered between two requests changes the prefix at the
    /// tool definitions, ahead of every message.
    #[tokio::test]
    async fn a_changed_tool_set_diverges_at_the_tool_definitions() {
        let llm = ScriptedLlm::new(1_000_000, []);
        let mut agent = Agent::new(llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent.push_message(Message::assistant("hi"));
        agent.push_message(Message::user("go on"));
        let before = next_prompt_prefix(&mut agent).await;
        agent.register_tool(QuickTool).expect("registers");
        let after = next_prompt_prefix(&mut agent).await;

        assert_eq!(
            after.chain.first_divergence(&before.chain),
            Some(crate::context::PrefixBlock::Tools)
        );
    }

    /// A config whose compaction distances fit the few-hundred-token windows
    /// [`window_for_usage`] sizes: the fork starts 50 tokens before the
    /// auto-compact point (80% of the window) and the pause one token before.
    fn scaled_config() -> AgentConfig {
        AgentConfig {
            system_prompt: Some("x".repeat(400)),
            context_policy: crate::ContextPolicy {
                start_distance: 50,
                safe_buffer: 1,
                // These windows are a few hundred tokens: the conversation
                // must shrink to the handoff for a compaction to relieve them.
                user_message_budget: 0,
                ..crate::ContextPolicy::default()
            },
            ..AgentConfig::default()
        }
    }

    /// Sizes a mock's context window so the agent's next request lands at
    /// `usage_percent` percent of it — the only honest way to aim between the
    /// start and pause points, since tool definitions and system blocks
    /// weigh as much as the conversation does.
    async fn window_for_usage(
        config: &AgentConfig,
        messages: &[Message],
        usage_percent: usize,
    ) -> (Agent<MockLlm, MockLlm, MockLlm, ()>, u32) {
        // Probe with a huge window to measure the request the real agent
        // will assemble, then pick the window that lands the usage exactly.
        let mut probe = Agent::with_config(
            MockLlm {
                context_length: u32::MAX / 4,
            },
            config.clone(),
        );
        probe.ensure_initialized().await.expect("probe initializes");
        for message in messages {
            probe.context.push(message.clone());
        }
        let tail = probe.render_tail().await;
        let assembled = probe.context.build_request(&tail);
        let chars = request_chars(&assembled, &probe.tools.active_definitions());
        let tokens = chars.div_ceil(4);
        let window = u32::try_from((tokens * 100).div_ceil(usage_percent))
            .expect("the aimed window fits in u32");
        (
            Agent::with_config(
                MockLlm {
                    context_length: window,
                },
                config.clone(),
            ),
            window,
        )
    }

    /// The handoff exists but the window grew past `compact_at` again: a new
    /// compaction must fork, not rest on the previous one.
    #[tokio::test]
    async fn a_landed_handoff_compacts_again_when_the_window_grows() {
        let config = scaled_config();
        let (mut agent, _window) = window_for_usage(&config, &[Message::user("hello")], 75).await;
        agent.ensure_initialized().await.expect("agent initializes");
        agent.context_mut().set_handoff(Handoff::new("resume here"));
        agent.push_message(Message::user("hello"));

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");

        assert!(
            request.is_some(),
            "usage between compact_at and pause_at must not withhold the request"
        );
        assert!(
            agent.compaction.is_running(),
            "growth past compact_at must fork a compaction even after a handoff exists"
        );
    }

    /// Crossing `compact_at` during a tool round starts the compaction but
    /// still lets that turn's request out: forking must not block the turn.
    #[tokio::test]
    async fn a_tool_round_crossing_compact_at_forks_without_blocking() {
        let config = scaled_config();
        let messages = vec![
            Message::user("run it"),
            Message::assistant_with_tool_calls(
                "",
                vec![aither_core::llm::ToolCall::new(
                    "call_1",
                    "terminal",
                    serde_json::json!({"description": "build"}),
                )],
            ),
            Message::tool("call_1", "ok"),
        ];
        let (mut agent, _window) = window_for_usage(&config, &messages, 75).await;
        agent.ensure_initialized().await.expect("agent initializes");
        for message in messages {
            agent.push_message(message);
        }

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");

        assert!(
            request.is_some(),
            "crossing compact_at alone must not withhold the turn's request"
        );
        assert!(
            agent.compaction.is_running(),
            "crossing compact_at must fork the compaction"
        );
    }

    /// At `pause_at` the boundary waits for the in-flight compaction and
    /// proceeds with the rebuilt request once it lands.
    #[tokio::test]
    async fn the_pause_waits_for_an_in_flight_compaction() {
        let config = scaled_config();
        // Aim at 95% of the window: a handoff a fraction of the conversation
        // size lands the rebuilt request back under compact_at.
        let probe_window = {
            let mut probe = Agent::with_config(
                ScriptedLlm::new(u32::MAX / 4, [HANDOFF_JSON.to_string()]),
                config.clone(),
            );
            probe.ensure_initialized().await.expect("probe initializes");
            probe.context.push(Message::user("u".repeat(3_400)));
            let tail = probe.render_tail().await;
            let assembled = probe.context.build_request(&tail);
            let chars = request_chars(&assembled, &probe.tools.active_definitions());
            u32::try_from((chars.div_ceil(4) * 100).div_ceil(95)).expect("window fits in u32")
        };
        let llm = ScriptedLlm::new(probe_window, [HANDOFF_JSON.to_string()]);
        let mut agent = Agent::with_config(llm.clone(), config);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("u".repeat(3_400)));

        // Seed an in-flight compaction the way prepare_request would.
        let tail = agent.render_tail().await;
        let request = LLMRequest::new(agent.context.build_request(&tail));
        let fork = CompactionFork::<HandoffDocument>::new(
            &request,
            agent.context.fork_point(),
            &[],
            None,
            &[],
        )
        .expect("fork");
        agent
            .compaction
            .start(PendingCompaction::start(fork, llm.clone()));

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");

        assert!(
            request.is_some(),
            "a landed compaction relieves the pause and the request goes out"
        );
        assert!(
            agent.context().handoff().is_some(),
            "the landed handoff is folded into the window"
        );
        let compaction_request = &llm.recorded()[0];
        assert!(
            compaction_request
                .messages()
                .iter()
                .any(|m| m.content().contains("handoff")),
            "the compaction request carries the instruction"
        );
    }

    /// The first request after a compaction carries the current tasks.md —
    /// sources render fresh after every apply.
    #[tokio::test]
    async fn the_request_after_a_compaction_carries_tasks_md() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(
            dir.path().join("tasks.md"),
            "- [ ] fix the flaky test
",
        )
        .expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        land_document(
            &mut agent.compaction,
            agent.context.fork_point(),
            HandoffDocument {
                current_task: "keep going".to_string(),
                ..HandoffDocument::default()
            },
        );

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after compaction");

        assert!(
            request
                .messages()
                .iter()
                .any(|m| m.content().contains("fix the flaky test")),
            "the post-compaction request must carry the live tasks.md"
        );
    }

    /// Items appended while a compaction was in flight are newer than the
    /// fork and survive the apply, in order.
    #[tokio::test]
    async fn items_appended_during_a_compaction_survive_in_order() {
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("before the fork"));
        let fork_point = agent.context.fork_point();
        // Two more items land while the hypothetical compaction runs.
        agent.push_message(Message::assistant("during one"));
        agent.push_message(Message::user("during two"));
        land_document(
            &mut agent.compaction,
            fork_point,
            HandoffDocument {
                current_task: "keep going".to_string(),
                ..HandoffDocument::default()
            },
        );

        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after apply");

        let conversation: Vec<String> = agent
            .context()
            .conversation_messages()
            .iter()
            .map(|m| m.content().to_string())
            .collect();
        // The user's own message outlives the handoff beside the suffix.
        assert_eq!(
            conversation,
            ["before the fork", "during one", "during two"]
        );
    }

    /// Alerts that arrive after the fork — a background job's completion, a
    /// hook's injected context — are not described by the handoff and must
    /// reach the model; alerts from before the fork go with the rest.
    #[tokio::test]
    async fn alerts_after_the_fork_survive_the_apply() {
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("before the fork"));
        agent.push_system_alert("stale alert before the fork");
        let fork_point = agent.context.fork_point();
        agent.push_system_alert("background job 7 completed");
        agent.push_message(Message::user("during"));
        agent.push_system_alert("hook context injected mid-turn");
        land_document(
            &mut agent.compaction,
            fork_point,
            HandoffDocument {
                current_task: "keep going".to_string(),
                ..HandoffDocument::default()
            },
        );

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after apply");

        let text: Vec<&str> = request.messages().iter().map(Message::content).collect();
        let seen = |needle: &str| text.iter().any(|content| content.contains(needle));
        assert!(seen("background job 7 completed"), "{text:?}");
        assert!(seen("hook context injected mid-turn"), "{text:?}");
        assert!(!seen("stale alert before the fork"), "{text:?}");
    }

    /// Dropping the apply while it reads the files to restore commits
    /// nothing: the compaction stays landed and the next boundary applies it
    /// in full, restored files included.
    ///
    /// The read runs on the blocking pool, which may finish it within the
    /// first poll; the apply must then have committed in full. Either way
    /// the apply is never seen half done.
    #[tokio::test]
    async fn dropping_the_apply_during_restoration_leaves_the_compaction_pending() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("notes.txt"), "restored marker text").expect("write");
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.restore_dir = Some(dir.path().to_path_buf());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("before the fork"));
        let fork_point = agent.context.fork_point();
        land_document(
            &mut agent.compaction,
            fork_point,
            HandoffDocument {
                current_task: "keep going".to_string(),
                context_files: vec![crate::context::ContextFile {
                    path: "notes.txt".to_string(),
                    lines: None,
                    why: "needed".to_string(),
                }],
                ..HandoffDocument::default()
            },
        );

        let first_poll = futures_lite::future::poll_once(agent.apply_completed_compaction()).await;
        if first_poll.is_none() {
            assert!(agent.context().handoff().is_none(), "nothing was committed");
            assert!(agent.compaction.landed().is_some(), "still pending");
        } else {
            assert!(agent.context().handoff().is_some(), "committed in full");
            assert!(agent.compaction.landed().is_none(), "nothing left pending");
        }

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after apply");
        assert!(agent.context().handoff().is_some());
        assert!(
            request
                .messages()
                .iter()
                .any(|m| m.content().contains("restored marker text")),
            "the restored file reached the request"
        );
    }

    /// The compaction request must be the main request's prefix verbatim,
    /// or the provider's prompt cache is useless for it.
    #[tokio::test]
    async fn the_compaction_request_shares_the_main_prefix() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = Agent::new(llm.clone());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("do the thing"));

        // Fork the way prepare_request does, around a real assembled request.
        let tail = agent.render_tail().await;
        let main = LLMRequest::new(agent.context.build_request(&tail));
        let fork = CompactionFork::<HandoffDocument>::new(
            &main,
            agent.context.fork_point(),
            &[],
            None,
            &[],
        )
        .expect("fork");
        agent
            .compaction
            .start(PendingCompaction::start(fork, llm.clone()));
        assert!(
            agent
                .make_room(&EventSink::discarding())
                .await
                .expect("compaction lands")
        );

        let recorded = llm.recorded();
        let compaction = &recorded[0];
        // The instruction request generate() sends starts from the fork's
        // messages; the first len-1 are the main request verbatim and the
        // last is the instruction.
        for (index, message) in main.messages().iter().enumerate() {
            assert_eq!(
                message.content(),
                compaction.messages()[index].content(),
                "compaction perturbed main-request message {index}"
            );
        }
        assert!(compaction.messages().len() > main.messages().len());
    }

    /// A preview does not consume the session start: the real request still
    /// carries the document.
    #[tokio::test]
    async fn a_preview_leaves_source_committed_state_untouched() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(
            dir.path().join("tasks.md"),
            "- [ ] pending work
",
        )
        .expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");

        agent
            .preview_request_messages("hi", std::iter::empty())
            .await
            .expect("preview");

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after preview");
        assert!(
            request
                .messages()
                .iter()
                .any(|m| m.content().contains("pending work")),
            "the real request must still carry the tasks diff the preview saw"
        );
    }

    /// Idle reassembly fires only once the conversation's cache lifetime has
    /// passed — never earlier, and never for a permanent cache. It runs only
    /// with a compaction model: otherwise idle compaction takes the idle gap.
    #[tokio::test]
    async fn an_idle_gap_reassembles_only_past_the_cache_lifetime() {
        let mut agent = Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .compaction_model(
            ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]),
            Parameters::default(),
        )
        .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));

        // The default lifetime is one hour: five minutes is nothing.
        agent.backdate_cache_entry(Duration::from_secs(300));
        assert!(
            agent
                .prepare_request(&EventSink::discarding())
                .await
                .expect("request boundary")
                .is_some()
        );
        assert_eq!(agent.context().count_recent_system_messages(), 1);

        // Fifty-nine minutes is still inside the hour.
        agent.backdate_cache_entry(Duration::from_secs(3540));
        assert!(
            agent
                .prepare_request(&EventSink::discarding())
                .await
                .expect("request boundary")
                .is_some()
        );
        assert_eq!(agent.context().count_recent_system_messages(), 1);

        // Sixty-one minutes is past it: the cached prefix is gone anyway,
        // so the compromises the append-only log forced get cleaned up.
        agent.backdate_cache_entry(Duration::from_secs(3660));
        assert!(
            agent
                .prepare_request(&EventSink::discarding())
                .await
                .expect("request boundary")
                .is_some()
        );
        assert_eq!(agent.context().count_recent_system_messages(), 0);
    }

    #[tokio::test]
    async fn a_permanent_cache_lifetime_never_reassembles() {
        use crate::context::CacheLifetime;

        let mut agent = Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .compaction_model(
            MockLlm {
                context_length: 1_000_000,
            },
            Parameters::default(),
        )
        .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.use_cache_lifetime(CacheLifetime::Permanent);
        agent.push_message(Message::user("hello"));
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));

        // Rewriting the prefix always costs a permanent cache, so no idle
        // gap ever justifies it — even a day later.
        agent.backdate_cache_entry(Duration::from_secs(86_400));
        assert!(
            agent
                .prepare_request(&EventSink::discarding())
                .await
                .expect("request boundary")
                .is_some()
        );
        assert_eq!(agent.context().count_recent_system_messages(), 1);
    }

    /// An agent that has sent one request, with `llm` as its model.
    async fn agent_after_one_request(
        llm: ScriptedLlm,
    ) -> Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, (), ScriptedLlm> {
        agent_after_one_message(
            llm,
            "hello",
            crate::ContextPolicy {
                compaction_floor: 0,
                ..crate::ContextPolicy::default()
            },
        )
        .await
    }

    /// An agent under `policy` that has sent one request carrying `text`.
    async fn agent_after_one_message(
        llm: ScriptedLlm,
        text: &str,
        policy: crate::ContextPolicy,
    ) -> Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, (), ScriptedLlm> {
        let mut agent = Agent::with_config(llm, AgentConfig::default().with_context_policy(policy));
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user(text));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("the first request goes out");
        agent
    }

    /// A message of roughly `tokens` tokens at the uncalibrated ratio.
    fn text_of_tokens(tokens: usize) -> String {
        "word ".repeat(tokens)
    }

    /// Idle compaction leaves a context under the floor alone, and runs on
    /// one over it, reading the floor from the agent's own policy.
    #[tokio::test]
    async fn idle_compaction_respects_the_policy_floor() {
        let policy = crate::ContextPolicy::default();
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut small = agent_after_one_message(llm.clone(), "hello", policy).await;
        small.backdate_cache_entry(Duration::from_secs(3240));
        assert_eq!(small.idle_compaction_due_in(), None);
        assert!(small.compact_when_idle().await.expect("no-op").is_none());
        assert!(llm.recorded().is_empty());

        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let big_text = text_of_tokens(2 * policy.compaction_floor);
        let mut big = agent_after_one_message(llm.clone(), &big_text, policy).await;
        big.backdate_cache_entry(Duration::from_secs(3240));
        assert_eq!(big.idle_compaction_due_in(), Some(Duration::ZERO));
        big.compact_when_idle()
            .await
            .expect("the compaction succeeds")
            .expect("the compaction ran");
        assert_eq!(llm.recorded().len(), 1);

        // A lowered floor in the given policy admits the small context.
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let lowered = crate::ContextPolicy {
            compaction_floor: 1,
            ..policy
        };
        let mut lowered = agent_after_one_message(llm, "hello", lowered).await;
        lowered.backdate_cache_entry(Duration::from_secs(3240));
        assert_eq!(lowered.idle_compaction_due_in(), Some(Duration::ZERO));
    }

    /// The idle compaction fires once, shortly before the cache lifetime ends,
    /// and not again while the session stays idle.
    #[tokio::test]
    async fn an_idle_session_compacts_once_before_its_cache_expires() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_after_one_request(llm.clone()).await;

        // Fifty-three minutes of a one-hour lifetime: not yet due.
        agent.backdate_cache_entry(Duration::from_secs(3180));
        let due_in = agent
            .idle_compaction_due_in()
            .expect("a compaction is pending");
        assert!(due_in > Duration::from_secs(30) && due_in <= Duration::from_secs(60));
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert!(llm.recorded().is_empty());

        // Fifty-four minutes: due now.
        agent.backdate_cache_entry(Duration::from_secs(3240));
        assert_eq!(agent.idle_compaction_due_in(), Some(Duration::ZERO));
        let result = agent
            .compact_when_idle()
            .await
            .expect("the compaction succeeds")
            .expect("the compaction ran");
        assert_eq!(result.messages_compacted, 1);
        assert_eq!(llm.recorded().len(), 1);
        assert!(agent.context().handoff().is_some());

        // Still idle: nothing more, however long the gap grows inside the hour.
        agent.backdate_cache_entry(Duration::from_secs(3300));
        assert_eq!(agent.idle_compaction_due_in(), None);
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert_eq!(llm.recorded().len(), 1, "one compaction per idle gap");

        // The next request re-arms it for the next gap.
        agent.push_message(Message::user("again"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after idle");
        agent.backdate_cache_entry(Duration::from_secs(3240));
        assert_eq!(agent.idle_compaction_due_in(), Some(Duration::ZERO));
    }

    fn handoff_json(task: &str) -> String {
        HANDOFF_JSON.replace("keep going", task)
    }

    /// A model that yields to the executor once before it answers, so a
    /// compaction raced against its turn gets polled.
    #[derive(Clone)]
    struct YieldingLlm(ScriptedLlm);

    impl LanguageModel for YieldingLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::StreamExt::filter_map(
                futures_lite::stream::once_future(futures_lite::future::yield_now()),
                |()| None::<Result<Event, MockError>>,
            )
            .chain(self.0.respond(request))
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            self.0.profile()
        }
    }

    /// A compaction still in flight when a manual one runs is retired by it:
    /// the runs after it keep the manual handoff and never poll the stale fork.
    #[tokio::test]
    async fn a_manual_compaction_retires_a_pending_one() {
        let manual = ScriptedLlm::new(1_000_000, [handoff_json("manual")]);
        let stale = ScriptedLlm::new(1_000_000, [handoff_json("stale")]);
        let mut agent = Agent::with_config(
            YieldingLlm(manual.clone()),
            AgentConfig::default().with_context_policy(crate::ContextPolicy {
                compaction_floor: 0,
                ..crate::ContextPolicy::default()
            }),
        );
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));

        let tail = agent.render_tail().await;
        let request = LLMRequest::new(agent.context.build_request(&tail));
        let fork = CompactionFork::<HandoffDocument>::new(
            &request,
            agent.context.fork_point(),
            &[],
            None,
            &[],
        )
        .expect("fork");
        agent
            .compaction
            .start(PendingCompaction::start(fork, stale.clone()));
        assert!(agent.compaction.is_running());

        agent
            .compact(Some("the manual focus"))
            .await
            .expect("manual compaction succeeds")
            .expect("the manual compaction ran");
        assert!(!agent.compaction.is_active(), "the pending one was retired");

        // A stale fork would land during the first run and apply at the
        // boundary of the second.
        for prompt in ["go on", "and again"] {
            let _ = crate::test_support::finish_run(agent.run(prompt, std::iter::empty())).await;
        }
        let handoff = agent
            .context()
            .handoff()
            .expect("the manual handoff stays in the window");
        assert!(handoff.text().contains("manual"), "{}", handoff.text());
        assert!(!handoff.text().contains("stale"), "{}", handoff.text());
        assert!(stale.recorded().is_empty(), "the stale fork never ran");
    }

    /// An expired cache is not worth compacting for: the context is written
    /// again either way.
    #[tokio::test]
    async fn an_expired_cache_is_not_idle_compacted() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_after_one_request(llm.clone()).await;
        agent.backdate_cache_entry(Duration::from_secs(3660));
        assert_eq!(agent.idle_compaction_due_in(), None);
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert!(llm.recorded().is_empty());
    }

    #[tokio::test]
    async fn a_permanent_cache_lifetime_never_idle_compacts() {
        use crate::context::CacheLifetime;

        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_after_one_request(llm.clone()).await;
        agent.use_cache_lifetime(CacheLifetime::Permanent);
        agent.backdate_cache_entry(Duration::from_secs(86_400));
        assert_eq!(agent.idle_compaction_due_in(), None);
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert!(llm.recorded().is_empty());
    }

    /// With a compaction model, compaction requests go to it and idle
    /// compaction never runs, since it cannot read the conversation's cache.
    #[tokio::test]
    async fn a_compaction_model_takes_the_compaction_requests_and_disables_idle_compaction() {
        let main = ScriptedLlm::new(1_000_000, []);
        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = Agent::builder(main.clone())
            .compaction_model(compactor.clone(), Parameters::default())
            .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("first request");

        agent.backdate_cache_entry(Duration::from_secs(3240));
        assert_eq!(agent.idle_compaction_due_in(), None);
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert!(main.recorded().is_empty() && compactor.recorded().is_empty());

        agent.compact(None).await.expect("compaction succeeds");
        assert_eq!(compactor.recorded().len(), 1, "the compaction model ran it");
        assert!(main.recorded().is_empty(), "the session's model did not");

        // The forked compaction goes to the compaction model as well.
        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = Agent::builder(ScriptedLlm::new(1_000_000, []))
            .compaction_model(compactor.clone(), Parameters::default())
            .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        let tail = agent.render_tail().await;
        let request = LLMRequest::new(agent.context.build_request(&tail));
        agent.start_compaction(&request, None);
        assert!(
            agent
                .make_room(&EventSink::discarding())
                .await
                .expect("compaction lands")
        );
        assert_eq!(compactor.recorded().len(), 1);
    }

    /// The Claude message breakpoint of a compaction request lands on the
    /// message the Claude request marks, from the ability of the model the
    /// request goes to: the compaction model's, not the agent's.
    #[tokio::test]
    async fn a_compaction_request_places_its_breakpoint_by_the_compaction_models_ability() {
        use aither_core::llm::model::{ClaudeCacheBreakpointTarget, ClaudePromptCacheStrategy};

        for (main_takes, compactor_takes) in [(false, true), (true, false)] {
            let model = |takes: bool| {
                let llm = ScriptedLlm::new(1_000_000, []);
                if takes {
                    llm.with_ability(Ability::MidConversationSystem)
                } else {
                    llm
                }
            };
            let mut agent = Agent::builder(model(main_takes))
                .compaction_model(model(compactor_takes), Parameters::default())
                .build();
            agent.ensure_initialized().await.expect("agent initializes");
            let messages = vec![
                Message::system("prefix"),
                Message::user("hello"),
                Message::assistant("hi"),
                Message::system("live tail"),
            ];
            let request = agent.compaction_request(messages.clone(), Vec::new());
            let Some(claude) = request.parameters().cache.claude else {
                panic!("the request carries Claude cache controls");
            };
            let ClaudePromptCacheStrategy::Explicit(breakpoints) = claude.strategy else {
                panic!("expected explicit breakpoints");
            };
            let Some(ClaudeCacheBreakpointTarget::Message { message_index, .. }) =
                breakpoints.second.map(|point| point.target)
            else {
                panic!("expected a message breakpoint");
            };

            // The list the compaction model's client sends: the leading
            // system message always leaves it, a later one only when the
            // model cannot take it in place.
            let sent: Vec<&Message> = messages
                .iter()
                .enumerate()
                .filter(|(position, message)| {
                    message.role() != Role::System || (compactor_takes && *position > 0)
                })
                .map(|(_, message)| message)
                .collect();
            assert_eq!(
                sent[message_index].content(),
                "hi",
                "the breakpoint marks the newest message before the live tail \
                 (agent {main_takes}, compaction model {compactor_takes})"
            );
        }
    }

    /// The host's parameters reach the provider on the turns and the
    /// compaction the agent sends its model, beside the cache controls aither
    /// derives and, on the compaction, structured output.
    #[tokio::test]
    async fn the_hosts_parameters_arrive_with_the_cache_controls_and_structured_output() {
        use crate::request_parameters::test_support::{
            assert_cache_controls, host_parameters, host_part,
        };
        let host = host_parameters(aither_core::llm::model::ReasoningEffort::High, 0.3);
        let llm = ScriptedLlm::new(
            1_000_000,
            ["the answer".to_string(), HANDOFF_JSON.to_string()],
        );
        let mut agent = Agent::builder(llm.clone())
            .request_parameters(host.clone())
            .prompt_cache(PromptCache::new("session-1"))
            .build();
        agent.query("hello").await.expect("the turn runs");
        agent.compact(None).await.expect("compaction succeeds");

        let [turn, compaction]: [LLMRequest; 2] = llm
            .recorded()
            .try_into()
            .expect("a turn, then the compaction");
        for request in [&turn, &compaction] {
            assert_eq!(host_part(request), host, "the host's parameters arrive");
            assert_cache_controls(request);
        }
        assert_eq!(
            turn.parameters()
                .cache
                .openai
                .as_ref()
                .and_then(|openai| openai.key.as_deref()),
            Some("session-1:0"),
            "the turn carries the session's cache key"
        );
        assert!(!turn.parameters().structured_outputs);
        assert!(
            compaction.parameters().structured_outputs,
            "the compaction asks for structured output"
        );
        assert_eq!(
            compaction.parameters().cache.openai,
            turn.parameters().cache.openai,
            "the compaction reads the cache the turn wrote"
        );
    }

    /// A compaction model's requests carry its own parameters, not the
    /// session model's, under the cache controls and structured output.
    #[tokio::test]
    async fn a_compaction_model_gets_its_own_parameters_under_aithers_fields() {
        use crate::request_parameters::test_support::{
            assert_cache_controls, host_parameters, host_part,
        };
        use aither_core::llm::model::ReasoningEffort;
        let session = host_parameters(ReasoningEffort::High, 0.3);
        let compacting = host_parameters(ReasoningEffort::Low, 0.9);
        let main = ScriptedLlm::new(1_000_000, ["the answer".to_string()]);
        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = Agent::builder(main.clone())
            .request_parameters(session.clone())
            .compaction_model(compactor.clone(), compacting.clone())
            .build();
        agent.query("hello").await.expect("the turn runs");
        agent.compact(None).await.expect("compaction succeeds");

        let [turn]: [LLMRequest; 1] = main
            .recorded()
            .try_into()
            .expect("one turn on the session's model");
        let [compaction]: [LLMRequest; 1] = compactor
            .recorded()
            .try_into()
            .expect("the compaction on the compaction model");
        assert_eq!(host_part(&turn), session);
        assert_eq!(
            host_part(&compaction),
            compacting,
            "the compaction model's own parameters arrive"
        );
        assert_cache_controls(&compaction);
        assert!(compaction.parameters().structured_outputs);
    }

    /// Idle reassembly and idle compaction exclude each other: an idle gap
    /// past the cache lifetime reassembles only when idle compaction is off.
    #[tokio::test]
    async fn idle_reassembly_runs_only_when_idle_compaction_is_off() {
        async fn system_reminders_after_a_gap(compaction_model: Option<ScriptedLlm>) -> usize {
            let llm = ScriptedLlm::new(1_000_000, []);
            let builder = Agent::builder(llm);
            let mut agent = if let Some(model) = compaction_model {
                builder
                    .compaction_model(model, Parameters::default())
                    .build()
            } else {
                builder.build()
            };
            agent.ensure_initialized().await.expect("agent initializes");
            agent.push_message(Message::user("hello"));
            agent.push_message(Message::system(
                "<system-reminder>ephemeral</system-reminder>",
            ));
            agent.backdate_cache_entry(Duration::from_secs(3660));
            assert!(
                agent
                    .prepare_request(&EventSink::discarding())
                    .await
                    .expect("request boundary")
                    .is_some()
            );
            agent.context().count_recent_system_messages()
        }

        assert_eq!(system_reminders_after_a_gap(None).await, 1);
        assert_eq!(
            system_reminders_after_a_gap(Some(ScriptedLlm::new(
                1_000_000,
                [HANDOFF_JSON.to_string()]
            )))
            .await,
            0
        );
    }

    /// An agent with `compactor` as its compaction model that has sent one
    /// request carrying two conversation messages and a stale reminder.
    async fn agent_with_compactor_after_one_request(
        main: ScriptedLlm,
        compactor: Option<ScriptedLlm>,
    ) -> Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, (), ScriptedLlm> {
        let builder = Agent::builder(main);
        let mut agent = match compactor {
            Some(compactor) => builder
                .compaction_model(compactor, Parameters::default())
                .build(),
            None => builder.build(),
        };
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("an old question"));
        agent.push_message(Message::assistant("an old answer"));
        agent.push_message(Message::user("the next question"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("the first request goes out");
        agent
    }

    fn mentions(request: &LLMRequest, text: &str) -> bool {
        request
            .messages()
            .iter()
            .any(|m| m.content().contains(text))
    }

    /// A send after the cache lifetime is preceded by exactly one compaction
    /// on the compaction model, and the request carries the compacted context.
    #[tokio::test]
    async fn a_send_on_a_cold_cache_compacts_first() {
        let main = ScriptedLlm::new(1_000_000, []);
        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent =
            agent_with_compactor_after_one_request(main.clone(), Some(compactor.clone())).await;
        assert!(compactor.recorded().is_empty());

        agent.backdate_cache_entry(Duration::from_secs(3660));
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after the gap");

        assert_eq!(compactor.recorded().len(), 1, "one compaction preceded it");
        assert!(main.recorded().is_empty(), "on the compaction model only");
        assert!(mentions(&request, "keep going"), "the handoff is sent");
        assert!(
            !mentions(&request, "an old answer"),
            "the old context is gone"
        );
        assert!(!agent.compaction.is_active());

        // The send restarts the cache entry: the next one is warm.
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("next request");
        assert_eq!(compactor.recorded().len(), 1);
    }

    /// The compaction's apply is the one rebuild of the prefix: the window is
    /// reassembled once, not reassembled and then compacted.
    #[tokio::test]
    async fn a_cold_send_rebuilds_the_prefix_once() {
        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_with_compactor_after_one_request(
            ScriptedLlm::new(1_000_000, []),
            Some(compactor.clone()),
        )
        .await;
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));

        agent.backdate_cache_entry(Duration::from_secs(3660));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after the gap");

        assert_eq!(agent.context().count_recent_system_messages(), 0);
        assert_eq!(
            compactor.recorded().len(),
            1,
            "compacted once, not also rebuilt on its own"
        );
    }

    #[tokio::test]
    async fn a_send_on_a_warm_cache_does_not_compact() {
        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_with_compactor_after_one_request(
            ScriptedLlm::new(1_000_000, []),
            Some(compactor.clone()),
        )
        .await;

        agent.backdate_cache_entry(Duration::from_secs(3540));
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request inside the lifetime");

        assert!(compactor.recorded().is_empty());
        assert!(mentions(&request, "an old answer"));
    }

    #[tokio::test]
    async fn without_a_compaction_model_a_cold_send_does_not_compact() {
        let main = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_with_compactor_after_one_request(main.clone(), None).await;

        agent.backdate_cache_entry(Duration::from_secs(86_400));
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after the gap");

        assert!(main.recorded().is_empty());
        assert!(mentions(&request, "an old answer"));
    }

    #[tokio::test]
    async fn a_permanent_cache_lifetime_never_compacts_on_send() {
        use crate::context::CacheLifetime;

        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_with_compactor_after_one_request(
            ScriptedLlm::new(1_000_000, []),
            Some(compactor.clone()),
        )
        .await;
        agent.use_cache_lifetime(CacheLifetime::Permanent);

        agent.backdate_cache_entry(Duration::from_secs(86_400));
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after the gap");

        assert!(compactor.recorded().is_empty());
        assert!(mentions(&request, "an old answer"));
    }

    /// A failing compaction model fails the send instead of sending the
    /// whole uncompacted context.
    #[tokio::test]
    async fn a_failed_cold_compaction_fails_the_send() {
        let compactor = ScriptedLlm::new(1_000_000, ["not a handoff".to_string()]);
        let mut agent = agent_with_compactor_after_one_request(
            ScriptedLlm::new(1_000_000, []),
            Some(compactor),
        )
        .await;

        agent.backdate_cache_entry(Duration::from_secs(3660));
        assert!(
            agent
                .prepare_request(&EventSink::discarding())
                .await
                .is_err()
        );
    }

    /// Messages of `request` that carry the whole tasks.md.
    fn full_document_messages(request: &LLMRequest) -> usize {
        request
            .messages()
            .iter()
            .filter(|m| m.content().contains("Contents of tasks.md"))
            .count()
    }

    /// Messages of `request` that carry a tasks.md diff.
    fn document_diff_messages(request: &LLMRequest) -> usize {
        request
            .messages()
            .iter()
            .filter(|m| m.content().contains("Changes to tasks.md"))
            .count()
    }

    #[tokio::test]
    async fn the_first_request_carries_the_whole_tasks_document() {
        let dir = tempfile::tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] first\n").expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("first request");

        assert_eq!(full_document_messages(&request), 1);
        assert_eq!(document_diff_messages(&request), 0);
    }

    #[tokio::test]
    async fn an_unchanged_tasks_document_adds_nothing_to_a_later_request() {
        let dir = tempfile::tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] first\n").expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        let first = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("first request");

        agent.push_message(Message::assistant("working"));
        agent.push_message(Message::user("go on"));
        let later = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("later request");

        // The earlier request is an exact prefix of the later one apart from
        // the volatile tail, and no new tasks.md text joined the conversation.
        assert_eq!(full_document_messages(&later), 1);
        assert_eq!(document_diff_messages(&later), 0);
        assert_eq!(
            later.messages().len(),
            first.messages().len() + 2,
            "only the two pushed messages may be new"
        );
        let tail = agent.render_tail().await;
        assert!(
            tail.iter().all(|block| block.tag() != "tasks_document"),
            "the tail must never carry the document"
        );
    }

    #[tokio::test]
    async fn a_tasks_change_adds_its_diff_exactly_once() {
        let dir = tempfile::tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] first\n").expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("first request");

        std::fs::write(dir.path().join("tasks.md"), "- [x] first\n- [ ] second\n")
            .expect("rewrite tasks.md");
        let changed = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after change");
        assert_eq!(document_diff_messages(&changed), 1);
        assert_eq!(
            full_document_messages(&changed),
            1,
            "only the session start's"
        );

        let after = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after that");
        assert_eq!(
            document_diff_messages(&after),
            1,
            "the diff stays in the conversation once and is not repeated"
        );
        assert_eq!(after.messages().len(), changed.messages().len());
    }

    #[tokio::test]
    async fn a_cold_cache_reassembly_resends_the_whole_document() {
        let dir = tempfile::tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] first\n").expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("first request");

        agent.backdate_cache_entry(Duration::from_secs(3660));
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after idle");

        assert_eq!(full_document_messages(&request), 1);
    }

    #[tokio::test]
    async fn the_request_after_a_compaction_carries_the_whole_document_once() {
        let dir = tempfile::tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] first\n").expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("first request");

        agent.push_message(Message::assistant("done"));
        land_document(
            &mut agent.compaction,
            agent.context.fork_point(),
            HandoffDocument {
                current_task: "keep going".to_string(),
                ..HandoffDocument::default()
            },
        );
        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after compaction");

        assert_eq!(
            full_document_messages(&request),
            1,
            "the compaction dropped the earlier copy; exactly one is sent again"
        );
        let next = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request after that");
        assert_eq!(full_document_messages(&next), 1);
        assert_eq!(next.messages().len(), request.messages().len());
    }

    /// A model whose first request calls every tool in `calls`; later requests
    /// answer with text. Every request is recorded.
    #[derive(Clone)]
    struct ToolCallingLlm {
        calls: Vec<(&'static str, &'static str)>,
        requests: Arc<std::sync::Mutex<Vec<LLMRequest>>>,
    }

    impl LanguageModel for ToolCallingLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let mut requests = self.requests.lock().expect("requests lock");
            let events: Vec<_> = if requests.is_empty() {
                self.calls
                    .iter()
                    .map(|(id, name)| {
                        Ok(Event::ToolCall(aither_core::llm::ToolCall::new(
                            *id,
                            *name,
                            serde_json::json!({}),
                        )))
                    })
                    .collect()
            } else {
                vec![Ok(Event::Text("done".to_string()))]
            };
            requests.push(request);
            drop(requests);
            futures_lite::stream::iter(events)
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                100_000,
            )))
        }
    }

    /// Arguments for the tools of the cancellation tests.
    #[derive(Debug, schemars::JsonSchema, serde::Deserialize)]
    struct NoArgs {}

    /// Returns at once.
    struct QuickTool;

    impl aither_core::llm::Tool for QuickTool {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "quick".into()
        }

        fn description(&self) -> std::borrow::Cow<'static, str> {
            "Returns at once".into()
        }

        type Arguments = NoArgs;
        type Res = aither_core::llm::ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(aither_core::llm::ToolResult::text("quick finished")))
        }
    }

    /// Announces it started, then blocks until released.
    struct BlockingTool {
        started: async_channel::Sender<()>,
        release: async_channel::Receiver<()>,
    }

    impl aither_core::llm::Tool for BlockingTool {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "blocking".into()
        }

        fn description(&self) -> std::borrow::Cow<'static, str> {
            "Blocks until released".into()
        }

        type Arguments = NoArgs;
        type Res = aither_core::llm::ToolResult;

        async fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::ToolContext,
        ) -> aither_core::Result<Self::Res> {
            self.started.send(()).await.expect("test listens");
            let _ = self.release.recv().await;
            Ok(aither_core::llm::ToolResult::text("blocking finished"))
        }
    }

    /// Runs `calls` on a fresh agent, drops the run stream once the blocking
    /// tool is executing, then sends a second prompt and returns the second
    /// request's messages.
    async fn messages_after_cancelled_run(
        calls: Vec<(&'static str, &'static str)>,
    ) -> Vec<Message> {
        let llm = ToolCallingLlm {
            calls,
            requests: Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let mut agent = Agent::new(llm.clone());
        let (started_tx, started_rx) = async_channel::bounded(1);
        let (_release_tx, release_rx) = async_channel::bounded(1);
        agent
            .register_tool(BlockingTool {
                started: started_tx,
                release: release_rx,
            })
            .expect("registers");
        agent.register_tool(QuickTool).expect("registers");

        // Dropping the driven run once the blocking tool started cancels it.
        tokio::select! {
            started = started_rx.recv() => started.expect("the blocking tool starts"),
            outcome = drain_run(agent.run("go", std::iter::empty())) => {
                panic!("the run ended before its blocking tool started: {outcome:?}");
            }
        }

        drain_run(agent.run("again", std::iter::empty())).await;
        let requests = llm.requests.lock().expect("requests lock");
        assert_eq!(requests.len(), 2, "one request per run");
        requests[1].messages().to_vec()
    }

    fn assert_every_call_has_a_result(messages: &[Message]) {
        for (index, message) in messages.iter().enumerate() {
            for call in message.tool_calls() {
                assert!(
                    messages[index + 1..]
                        .iter()
                        .any(|later| later.tool_call_id() == Some(call.id.as_str())),
                    "tool call {} has no result in the request",
                    call.id
                );
            }
        }
    }

    #[tokio::test]
    async fn a_dropped_run_leaves_cancelled_results_for_running_tools() {
        let messages = messages_after_cancelled_run(vec![("call_slow", "blocking")]).await;

        assert_every_call_has_a_result(&messages);
        let result = messages
            .iter()
            .position(|message| message.tool_call_id() == Some("call_slow"))
            .expect("the cancelled call has a result");
        assert!(
            messages[result].content().contains("cancelled"),
            "{}",
            messages[result].content()
        );
        let prompt = messages
            .iter()
            .position(|message| message.content().starts_with("again\n\nSent at "))
            .expect("the new prompt is sent");
        assert!(result < prompt, "the result precedes the new prompt");
    }

    #[tokio::test]
    async fn a_dropped_run_keeps_results_of_tools_that_finished() {
        let messages =
            messages_after_cancelled_run(vec![("call_slow", "blocking"), ("call_quick", "quick")])
                .await;

        assert_every_call_has_a_result(&messages);
        let quick = messages
            .iter()
            .find(|message| message.tool_call_id() == Some("call_quick"))
            .expect("the finished call has a result");
        assert_eq!(quick.content(), "quick finished");
        let slow = messages
            .iter()
            .find(|message| message.tool_call_id() == Some("call_slow"))
            .expect("the cancelled call has a result");
        assert!(slow.content().contains("cancelled"), "{}", slow.content());
    }

    async fn drain_run(run: Run<impl Future<Output = Result<(), AgentError>>>) -> Vec<AgentEvent> {
        let ControlFlow::Continue(events) = run
            .drive(
                Vec::new(),
                std::future::pending(),
                |mut events, timed| async move {
                    events.push(timed.event);
                    ControlFlow::<std::convert::Infallible, _>::Continue(events)
                },
            )
            .await
            .expect("the run should succeed");
        events
    }

    fn detached_agent(
        responses: impl IntoIterator<Item = String>,
    ) -> (
        Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, ()>,
        aither_sandbox::BackgroundTaskSender,
        ScriptedLlm,
    ) {
        let llm = ScriptedLlm::new(100_000, responses);
        let (sender, receiver) = aither_sandbox::background_task_channel();
        let mut agent = Agent::with_config(
            llm.clone(),
            AgentConfig {
                background_jobs: BackgroundJobPolicy::Detach,
                ..AgentConfig::default()
            },
        );
        agent.background_receiver = Some(receiver);
        (agent, sender, llm)
    }

    fn finished(task_id: &str, output: &str) -> aither_sandbox::CompletedTask {
        aither_sandbox::CompletedTask {
            task_id: task_id.to_string(),
            script: "make".to_string(),
            kill: None,
            timeout_kill: None,
            result: Err(aither_sandbox::TerminalError::Execution(output.to_string())),
            pending: None,
        }
    }

    #[cfg(feature = "languages")]
    fn language_agent(
        languages: crate::Languages,
        responses: impl IntoIterator<Item = String>,
    ) -> (
        Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, ()>,
        ScriptedLlm,
    ) {
        let llm = ScriptedLlm::new(1_000_000, responses);
        let agent = Agent::with_config(
            llm.clone(),
            AgentConfig::default().with_languages(crate::LanguageGuard::new(languages)),
        );
        (agent, llm)
    }

    #[cfg(feature = "languages")]
    fn chinese() -> crate::Languages {
        crate::Languages::new("Chinese".parse().unwrap())
    }

    #[cfg(feature = "languages")]
    fn language_alerts(agent: &Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, ()>) -> usize {
        agent
            .history()
            .iter()
            .filter(|message| message.content().contains("was not in Chinese"))
            .count()
    }

    #[cfg(feature = "languages")]
    #[tokio::test]
    async fn an_english_reply_yields_one_alert_and_a_translated_display_copy() {
        let original = "I have finished the database migration and every test passes.";
        let (mut agent, llm) = language_agent(
            chinese(),
            [
                original.to_string(),
                "数据库迁移已经完成，所有测试都通过了。".to_string(),
            ],
        );
        let events = drain_run(agent.run("请迁移数据库", std::iter::empty())).await;

        let translated: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ReplyTranslated { text, language } => Some((text, language.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            translated,
            [(&"数据库迁移已经完成，所有测试都通过了。".to_string(), "cmn")]
        );
        assert_eq!(language_alerts(&agent), 1, "one alert per lapse");
        assert!(
            agent
                .history()
                .iter()
                .any(|message| message.content() == original),
            "the history keeps the original reply"
        );
        assert!(
            agent
                .history()
                .iter()
                .all(|message| !message.content().contains("数据库迁移已经完成")),
            "the translation is for display only"
        );
        let requests = llm.recorded();
        assert_eq!(requests.len(), 2, "one turn and one translation");
    }

    #[cfg(feature = "languages")]
    #[tokio::test]
    async fn a_chinese_reply_needs_no_alert_or_translation() {
        let (mut agent, llm) = language_agent(chinese(), ["数据库迁移已经完成。".to_string()]);
        let events = drain_run(agent.run("请迁移数据库", std::iter::empty())).await;
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::ReplyTranslated { .. }))
        );
        assert_eq!(language_alerts(&agent), 0);
        assert_eq!(llm.recorded().len(), 1);
    }

    #[cfg(feature = "languages")]
    #[tokio::test]
    async fn languages_unset_or_english_change_nothing() {
        for languages in [
            crate::Languages::unset(),
            crate::Languages::new(crate::Language::ENGLISH),
        ] {
            let (mut agent, llm) =
                language_agent(languages, ["All done, the migration passes.".to_string()]);
            let events = drain_run(agent.run("go", std::iter::empty())).await;
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, AgentEvent::ReplyTranslated { .. }))
            );
            assert_eq!(llm.recorded().len(), 1);
            assert!(
                !agent
                    .context()
                    .instructions()
                    .contains(&InstructionSlot::LANGUAGES),
                "no language section in the system prompt"
            );
        }
    }

    #[cfg(feature = "languages")]
    #[tokio::test]
    async fn the_system_prompt_states_both_languages() {
        let (mut agent, _llm) =
            language_agent(chinese().with_working("German".parse().unwrap()), []);
        agent.ensure_initialized().await.expect("agent initializes");
        let section = agent
            .context()
            .instructions()
            .get(&InstructionSlot::LANGUAGES)
            .expect("the languages section is present")
            .clone();
        assert!(
            section.contains("Chinese") && section.contains("German"),
            "{section}"
        );
        assert_eq!(
            agent
                .languages()
                .map(|guard| guard.languages().communication()),
            Some(Some("Chinese".parse().unwrap())),
            "an application reads the language from the agent"
        );
    }

    /// Under `Detach` the run ends with the model's turn even while a
    /// background command it started still runs.
    #[tokio::test]
    async fn a_detached_run_ends_without_waiting_for_its_jobs() {
        let (mut agent, sender, _llm) = detached_agent(["started the build".to_string()]);
        sender.task_started();
        let events = tokio::time::timeout(
            Duration::from_secs(5),
            drain_run(agent.run("build it", std::iter::empty())),
        )
        .await
        .expect("the run must not park on the running job");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Complete { .. })),
            "the run should complete: {events:?}"
        );
    }

    /// Between runs, a finished background command lands in the context and
    /// asks for the model; `resume` then runs it without a user message.
    #[tokio::test]
    async fn a_command_finishing_between_runs_wakes_the_model() {
        use aither_core::llm::Role;

        let (mut agent, sender, llm) = detached_agent([
            "started the build".to_string(),
            "the build failed".to_string(),
        ]);
        drain_run(agent.run("build it", std::iter::empty())).await;
        agent.detached_task_ids.insert("task-build".to_string());

        let waiting = tokio::spawn(async move {
            let wake = agent.wait_background().await;
            (agent, wake)
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        sender
            .send_completed(finished("task-build", "linker error"))
            .await;
        let (mut agent, wake) = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("the completion must end the wait")
            .expect("the wait task should not panic");

        assert!(
            wake.needs_model,
            "a background command's result needs the model"
        );
        assert!(
            wake.events
                .iter()
                .any(|timed| matches!(timed.event, AgentEvent::BackgroundTaskCompleted { .. })),
            "the host should see the completion: {:?}",
            wake.events
        );
        assert_eq!(
            agent.tasks.events.take(),
            vec![crate::tasks::TaskEvent::JobFinished(
                "task-build".to_string()
            )],
            "the finished job reaches the tasks inbox"
        );

        let events = drain_run(agent.resume()).await;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Complete { .. })),
            "the resumed run should complete: {events:?}"
        );
        let requests = llm.recorded();
        let resumed = requests.last().expect("the resumed run asked the model");
        assert!(
            resumed
                .messages()
                .iter()
                .any(|message| message.content().contains("linker error")),
            "the resumed request must carry the command's result"
        );
        assert!(
            !resumed
                .messages()
                .iter()
                .rev()
                .take_while(|message| message.role() != Role::Assistant)
                .any(|message| message.role() == Role::User),
            "resume must not add a user message"
        );
    }

    /// A foreground command's completion is not news to the model: its
    /// result already returned through the tool call.
    #[tokio::test]
    async fn a_foreground_completion_does_not_wake_the_model() {
        let (mut agent, sender, _llm) = detached_agent(Vec::new());
        sender
            .send_completed(finished("task-foreground", "ok"))
            .await;
        let wake = tokio::time::timeout(Duration::from_secs(5), agent.wait_background())
            .await
            .expect("the queued completion ends the wait");
        assert!(!wake.needs_model);
        assert!(wake.events.is_empty(), "{:?}", wake.events);
    }

    /// The idle agent's wake keeps a finished job's completion pending in
    /// its registry until the host drops it: a host reading the registry in
    /// between always sees a turn owed.
    #[tokio::test]
    async fn a_wake_keeps_its_completions_pending_until_the_host_drops_it() {
        let (mut agent, sender, _llm) = detached_agent(Vec::new());
        agent.detached_task_ids.insert("task-build".to_string());
        let (jobs, service) = aither_sandbox::job_registry_channel();
        tokio::spawn(service.serve());
        jobs.register(
            41,
            "task-build",
            "exec",
            "make",
            aither_sandbox::TerminalMode::Sandboxed,
            None,
        )
        .await;
        let owed = jobs
            .owe_completion("task-build", Err("linker error".to_string()))
            .await
            .expect("the job is registered");
        let mut task = finished("task-build", "linker error");
        task.pending = Some(owed.pending);
        sender.send_completed(task).await;

        let wake = tokio::time::timeout(Duration::from_secs(5), agent.wait_background())
            .await
            .expect("the queued completion ends the wait");
        assert!(wake.needs_model);
        assert_eq!(wake.pending.len(), 1);
        let status = jobs.get(41).await.expect("job should exist").status;
        assert!(status.completion_pending(), "{status:?}");

        drop(wake);
        let status = jobs.get(41).await.expect("job should exist").status;
        assert_eq!(status.delivery(), Some(aither_sandbox::Delivery::Taken));
    }

    /// The idle wait gives the event up when it is dropped before anything
    /// arrived, so a host can race it against its own work.
    #[tokio::test]
    async fn an_abandoned_idle_wait_takes_nothing() {
        let (mut agent, sender, _llm) = detached_agent(Vec::new());
        agent.detached_task_ids.insert("task-late".to_string());
        assert!(
            tokio::time::timeout(Duration::from_millis(50), agent.wait_background())
                .await
                .is_err(),
            "nothing is queued yet"
        );
        sender
            .send_completed(finished("task-late", "done late"))
            .await;
        let wake = tokio::time::timeout(Duration::from_secs(5), agent.wait_background())
            .await
            .expect("the completion is still there for the next wait");
        assert!(wake.needs_model);
    }

    /// An agent with the `compact` tool on a 1M window, and the tool itself.
    fn agent_with_compact_tool(
        llm: &ScriptedLlm,
    ) -> (Agent<ScriptedLlm>, crate::compact_tool::CompactTool) {
        agent_with_compact_tool_under(
            llm,
            crate::ContextPolicy {
                compaction_floor: 0,
                ..crate::ContextPolicy::default()
            },
        )
    }

    /// The same under `policy`.
    fn agent_with_compact_tool_under(
        llm: &ScriptedLlm,
        policy: crate::ContextPolicy,
    ) -> (Agent<ScriptedLlm>, crate::compact_tool::CompactTool) {
        let agent = Agent::builder(llm.clone())
            .context_policy(policy)
            .compact_tool()
            .build();
        let signal = Arc::clone(
            agent
                .compact_signal
                .as_ref()
                .expect("compact_tool arms the signal"),
        );
        let notices = agent.notices.clone();
        (
            agent,
            crate::compact_tool::CompactTool::new(signal, notices),
        )
    }

    async fn call_compact(tool: &crate::compact_tool::CompactTool) -> String {
        let result = aither_core::llm::Tool::call(
            tool,
            crate::CompactArgs::default(),
            aither_core::llm::ToolContext::new(),
        )
        .await
        .expect("compact returns at once");
        let aither_core::llm::ToolResult::Text { text } = result else {
            panic!("compact answers in text: {result:?}");
        };
        text
    }

    /// Calling `compact` starts the compaction at the next request boundary
    /// without blocking the turn, and the handoff applies at a later one.
    #[tokio::test]
    async fn the_compact_tool_starts_a_compaction_and_it_applies_at_a_later_boundary() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let (mut agent, tool) = agent_with_compact_tool(&llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("work on it"));

        let answer = call_compact(&tool).await;
        assert!(answer.contains("Compaction started"), "{answer}");
        assert!(
            !agent.compaction.is_active(),
            "the tool itself starts nothing"
        );

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");
        assert!(request.is_some(), "the turn is not held back");
        assert!(agent.compaction.is_running(), "the fork started");
        assert!(llm.recorded().is_empty(), "nothing waited for it");

        // The compaction lands while turns run; the next boundary applies it.
        let mut pending = agent.compaction.take_running().expect("in flight");
        let completed = pending.wait().await.expect("handoff");
        agent.compaction.land(completed);
        assert!(agent.context().handoff().is_none());
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("later boundary");
        assert!(agent.context().handoff().is_some());
    }

    /// A call after a compaction completed with no new action since starts
    /// nothing and says so; any other tool call re-opens the valve.
    #[tokio::test]
    async fn a_compact_call_after_a_completed_compaction_with_no_action_starts_nothing() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let (mut agent, tool) = agent_with_compact_tool(&llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("work on it"));
        assert!(call_compact(&tool).await.contains("Compaction started"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");
        let mut pending = agent.compaction.take_running().expect("in flight");
        let completed = pending.wait().await.expect("handoff");
        agent.compaction.land(completed);
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("later boundary");
        assert!(agent.context().handoff().is_some(), "the compaction landed");

        let answer = call_compact(&tool).await;
        assert!(answer.contains("nothing has been done since"), "{answer}");
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("next boundary");
        assert!(!agent.compaction.is_active(), "nothing was started");

        // The model acts: the valve opens again.
        let (sender, _events) = async_channel::unbounded();
        let sink = EventSink { sender };
        let call = ToolCall::new("call-1", "terminal", serde_json::json!({}));
        agent.execute_tool_calls(&[call], 1, &sink).await;
        assert!(call_compact(&tool).await.contains("Compaction started"));
    }

    /// A context under the policy floor starts nothing and names the floor;
    /// one at the floor starts a compaction. The floor is the agent's own.
    #[tokio::test]
    async fn a_compact_call_under_the_floor_starts_nothing() {
        let policy = crate::ContextPolicy {
            compaction_floor: 30_000,
            ..crate::ContextPolicy::default()
        };
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let (mut agent, tool) = agent_with_compact_tool_under(&llm, policy);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("work on it"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");

        let answer = call_compact(&tool).await;
        assert!(answer.contains("fewer than 30000 tokens"), "{answer}");
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("next boundary");
        assert!(!agent.compaction.is_active(), "nothing was started");

        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let (mut agent, tool) = agent_with_compact_tool_under(&llm, policy);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("u".repeat(4 * 60_000)));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");
        assert!(call_compact(&tool).await.contains("Compaction started"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("next boundary");
        assert!(agent.compaction.is_running(), "the fork started");
    }

    /// A second call while a compaction is requested or running starts
    /// nothing.
    #[tokio::test]
    async fn a_second_compact_call_starts_nothing() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let (mut agent, tool) = agent_with_compact_tool(&llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("work on it"));

        assert!(call_compact(&tool).await.contains("Compaction started"));
        assert!(
            call_compact(&tool).await.contains("already in flight"),
            "requested twice before the boundary"
        );
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");
        let fork_point = agent
            .compaction
            .take_running()
            .map(|pending| {
                let fork_point = pending.fork_point();
                agent.compaction.resume(Some(pending));
                fork_point
            })
            .expect("one compaction");

        agent.push_message(Message::user("more work"));
        assert!(call_compact(&tool).await.contains("already in flight"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("next boundary");
        let still = agent.compaction.take_running().expect("still the first");
        assert_eq!(
            still.fork_point(),
            fork_point,
            "no second compaction replaced it"
        );
    }

    /// A compaction the harness forked on its own counts as in flight too.
    #[tokio::test]
    async fn a_compact_call_during_an_automatic_compaction_starts_nothing() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let (mut agent, tool) = agent_with_compact_tool(&llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("u".repeat(4 * 360_000)));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");
        assert!(agent.compaction.is_running());

        assert!(call_compact(&tool).await.contains("already in flight"));
    }

    fn count_alerts(agent: &Agent<ScriptedLlm>) -> usize {
        agent
            .context()
            .recent()
            .iter()
            .filter(|message| {
                message.role() == aither_core::llm::Role::System
                    && message.content().contains("`compact`")
                    && message.content().contains("`tasks.md`")
                    && message.content().contains("notes")
            })
            .count()
    }

    /// Crossing the warning distance yields exactly one alert naming
    /// `compact` and `tasks.md`; crossing again after a compaction brought
    /// usage back down yields one more.
    #[tokio::test]
    async fn crossing_the_warning_distance_raises_one_alert() {
        let llm = ScriptedLlm::new(1_000_000, []);
        let (mut agent, _tool) = agent_with_compact_tool(&llm);
        agent.ensure_initialized().await.expect("agent initializes");

        // 330k tokens: past the warning point (320k), short of the fork (350k).
        agent.push_message(Message::user("u".repeat(4 * 330_000)));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");
        assert_eq!(count_alerts(&agent), 1);
        assert!(!agent.compaction.is_active(), "the alert only asks");

        agent.push_message(Message::user("more"));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("next boundary");
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("and again");
        assert_eq!(count_alerts(&agent), 1, "one-shot");

        // A compaction shrinks the window; the next approach warns again.
        agent.context.clear_history();
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("below the point");
        assert_eq!(count_alerts(&agent), 0);
        agent.push_message(Message::user("u".repeat(4 * 330_000)));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("second approach");
        assert_eq!(count_alerts(&agent), 1);
    }

    /// The alert is raised before the request is built, so it is part of the
    /// request that crossed the point, and it appends to the window.
    #[tokio::test]
    async fn the_alert_reaches_the_request_that_crossed_the_point() {
        let llm = ScriptedLlm::new(1_000_000, []);
        let (mut agent, _tool) = agent_with_compact_tool(&llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("u".repeat(4 * 330_000)));

        let request = agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary")
            .expect("request");

        assert!(
            request
                .messages()
                .last()
                .expect("messages")
                .content()
                .contains("`compact`")
                || request
                    .messages()
                    .iter()
                    .any(|message| message.content().contains("`compact`")),
        );
    }

    /// Without the tool there is nothing to point the model at.
    #[tokio::test]
    async fn no_alert_without_the_compact_tool() {
        let llm = ScriptedLlm::new(1_000_000, []);
        let mut agent = Agent::new(llm.clone());
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("u".repeat(4 * 330_000)));
        agent
            .prepare_request(&EventSink::discarding())
            .await
            .expect("request boundary");
        assert!(
            agent
                .context()
                .recent()
                .iter()
                .all(|message| !message.content().contains("`compact`"))
        );
    }

    /// A model that replays one scripted event list per request and records
    /// the requests it receives.
    #[derive(Clone)]
    struct EventScriptLlm {
        requests: Arc<std::sync::Mutex<Vec<LLMRequest>>>,
        turns: Arc<std::sync::Mutex<std::collections::VecDeque<Vec<Event>>>>,
    }

    impl EventScriptLlm {
        fn new(turns: impl IntoIterator<Item = Vec<Event>>) -> Self {
            Self {
                requests: Arc::new(std::sync::Mutex::new(Vec::new())),
                turns: Arc::new(std::sync::Mutex::new(turns.into_iter().collect())),
            }
        }
    }

    impl LanguageModel for EventScriptLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests.lock().expect("requests lock").push(request);
            let events = self
                .turns
                .lock()
                .expect("turns lock")
                .pop_front()
                .unwrap_or_default();
            futures_lite::stream::iter(events.into_iter().map(Ok))
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                1_000_000,
            )))
        }
    }

    /// The assistant messages of the last request the model received.
    fn replayed_assistants(llm: &EventScriptLlm) -> Vec<Message> {
        llm.requests
            .lock()
            .expect("requests lock")
            .last()
            .expect("a request was sent")
            .messages()
            .iter()
            .filter(|message| matches!(message, Message::Assistant { .. }))
            .cloned()
            .collect()
    }

    #[tokio::test]
    async fn a_final_answer_keeps_its_reasoning_in_the_next_request() {
        let thinking = ReasoningState::new("claude", "signed-thinking");
        let llm = EventScriptLlm::new([
            vec![
                Event::ReasoningState(thinking.clone()),
                Event::Text("the answer".to_string()),
            ],
            vec![Event::Text("done".to_string())],
        ]);
        let mut agent = Agent::new(llm.clone());

        agent.query("first").await.expect("first run");
        agent.query("second").await.expect("second run");

        let assistants = replayed_assistants(&llm);
        assert_eq!(assistants.len(), 1);
        assert_eq!(assistants[0].content(), "the answer");
        assert_eq!(assistants[0].reasoning(), [thinking]);
    }

    #[tokio::test]
    async fn a_reasoning_only_turn_is_kept() {
        let thinking = ReasoningState::new("openai", "encrypted-reasoning");
        let llm = EventScriptLlm::new([
            vec![Event::ReasoningState(thinking.clone())],
            vec![Event::Text("done".to_string())],
        ]);
        let mut agent = Agent::new(llm.clone());

        agent.query("first").await.expect("first run");
        agent.query("second").await.expect("second run");

        let assistants = replayed_assistants(&llm);
        assert_eq!(assistants.len(), 1);
        assert_eq!(assistants[0].content(), "");
        assert_eq!(assistants[0].reasoning(), [thinking]);
    }
}
