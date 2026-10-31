//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages the context window, forks compaction in the background, and
//! handles tool execution in an agent-controlled loop.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "skills")]
use crate::context::{ActiveSkills, ContextSource as _, SkillCatalogEntry};
use crate::{
    checkpoint::AgentCheckpoint,
    config::{AgentConfig, AgentKind, BackgroundJobPolicy},
    context::{
        AnyContextSource, BackgroundJobs, Block, CacheLifetime, CompactionFork,
        CompletedCompaction, Context, ContextBudget, HandoffDocument, PendingCompaction,
        PromptCache, TasksDocument, restore_context,
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
    tools::AgentTools,
};
use aither_core::{
    LanguageModel,
    llm::{
        Attachment, CommandStream, Event, LLMRequest, Message, ReasoningState, StoredFile,
        ToolCall, model::Profile as ModelProfile,
    },
};
use askama::Template;
use futures_core::Stream;
use futures_lite::StreamExt;
use sha2::{Digest, Sha256};

use aither_sandbox::{
    BackgroundTaskEvent, BackgroundTaskReceiver, DirectoryResetReason, JobAlert, JobProcess,
    JobRegistry, PermissionEvent, PermissionEventReceiver, PermissionEventStage, TerminalArgs,
    TerminalExecutionMode, TerminalMode,
};
use std::collections::{HashMap, HashSet, VecDeque};

const BACKGROUND_TASK_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Which of the two futures raced during a turn finished first.
enum TurnStep {
    /// The model turn completed.
    Turn(Result<ModelTurn, AgentError>),
    /// The forked compaction landed while the turn was still running.
    Compacted(Result<HandoffDocument, crate::context::CompactionError>),
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
    fn emit(&self, event: AgentEvent) {
        let _ = self.sender.try_send(TimedEvent {
            at: jiff::Timestamp::now(),
            event,
        });
    }
}

/// One step of the merged run/event stream in [`Agent::run`].
enum RunStep {
    Event(Option<TimedEvent>),
    Finished(Result<(), AgentError>),
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
}

/// Streams the events of one run, then its error if it failed.
fn drive_run<'a, F, Fut>(run: F) -> impl Stream<Item = Result<TimedEvent, AgentError>> + 'a
where
    F: FnOnce(EventSink) -> Fut + 'a,
    Fut: std::future::Future<Output = Result<(), AgentError>> + 'a,
{
    async_stream::stream! {
        let (sender, receiver) = async_channel::unbounded();
        // The run owns the sink and drops it as it finishes.
        let run = run(EventSink { sender });
        futures_lite::pin!(run);

        let mut outcome = None;
        while outcome.is_none() {
            let step = futures_lite::future::or(
                async { RunStep::Event(receiver.recv().await.ok()) },
                async { RunStep::Finished((&mut run).await) },
            )
            .await;
            match step {
                RunStep::Event(Some(event)) => yield Ok(event),
                // Sink dropped: the run is wrapping up — await its outcome.
                RunStep::Event(None) => outcome = Some((&mut run).await),
                RunStep::Finished(result) => outcome = Some(result),
            }
        }
        // Deliver events that raced with run completion.
        while let Ok(event) = receiver.try_recv() {
            yield Ok(event);
        }
        if let Some(Err(error)) = outcome {
            yield Err(error);
        }
    }
}

struct RunLoopOutcome {
    final_text: String,
    stop_reason: StopReason,
    iteration: usize,
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

type ToolExecutionResult = Result<(String, String, aither_core::llm::ToolResult), AgentError>;
type ToolExecutionSignal = ExecutionSignal<ToolExecutionResult>;

async fn handle_model_event<H: Hook>(
    hooks: &H,
    cache_stats: &mut crate::CacheStats,
    budget: &mut ContextBudget,
    event: Result<Event, String>,
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
        Err(error) if error.contains("malformed function call") => {
            tracing::warn!("Model generated malformed function call, retrying...");
            Ok(ModelEventAction::RetryMalformed)
        }
        Err(error) => Err(AgentError::Llm(error)),
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
    Err(AgentError::Llm(format!(
        "provider emitted tool call with empty id; response_text={response_text:?}; tool_calls=[{tool_calls_text}]"
    )))
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
    /// When the most recent keep-alive was sent.
    last_sent_at: Option<Instant>,
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

/// The result given to a tool call whose run was cancelled before it finished.
#[derive(Template)]
#[template(path = "tool_call_cancelled.txt", escape = "none")]
struct ToolCallCancelledTemplate<'a> {
    tool: &'a str,
}

#[derive(Template)]
#[template(path = "background_terminal_result.txt", escape = "none")]
struct BackgroundTerminalResultTemplate<'a> {
    task_id: &'a str,
    script: &'a str,
    outcome: BackgroundOutcome<'a>,
    output: Option<&'a str>,
    stderr: Option<&'a str>,
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
}

fn job_alert_text(alert: &JobAlertText<'_>) -> String {
    JobAlertTemplate { alert }
        .render()
        .expect("job alert template must render")
}

/// How a finished background task ended.
enum BackgroundOutcome<'a> {
    /// The script ran to completion.
    Finished { code: i32 },
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
/// - `Fast`: LLM for quick tasks like compaction (defaults to Balanced)
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
/// // With tiered models
/// let agent = Agent::builder(opus)
///     .balanced_model(sonnet)
///     .fast_model(haiku)
///     .build();
///
/// // Subagent using balanced tier
/// let explore_agent = Agent::builder(opus)
///     .balanced_model(sonnet)
///     .fast_model(haiku)
///     .tier(ModelTier::Balanced)  // Use sonnet for reasoning
///     .build();
/// ```
#[derive(Debug)]
pub struct Agent<Advanced, Balanced = Advanced, Fast = Balanced, H = (), Compactor = Advanced> {
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
    /// reassembles the window instead.
    pub(crate) compaction_model: Option<Arc<Compactor>>,

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

    /// Compaction running alongside the current turn, if any.
    pub(crate) pending_compaction: Option<PendingCompaction>,

    /// Compaction that finished and is waiting for a request boundary.
    pub(crate) completed_compaction: Option<CompletedCompaction>,

    /// Results of the tool calls of the turn in flight, in completion order,
    /// until they are moved into the context.
    pub(crate) tool_results: VecDeque<ToolExecutionResult>,

    /// Cached model profile (for the selected tier).
    pub(crate) profile: Option<ModelProfile>,

    /// Whether tools have been bootstrapped.
    pub(crate) initialized: bool,

    /// Whether the session-start lifecycle event has fired.
    pub(crate) session: SessionPhase,

    /// Receiver for completed background terminal tasks.
    pub(crate) background_receiver: Option<BackgroundTaskReceiver>,
    /// Receiver for permission wait/resume events emitted by terminal execution.
    pub(crate) permission_receiver: Option<PermissionEventReceiver>,
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

    /// Directory files are restored from after a forked compaction.
    pub(crate) restore_dir: Option<PathBuf>,

    /// Start time of the most recent LLM request issued by this agent.
    pub(crate) last_request_started_at: Option<Instant>,

    /// Time source of every idle-timing read.
    pub(crate) clock: crate::AnyClock,

    /// The most recent LLM request issued by this agent, retained so a
    /// prompt-cache keep-alive can re-send it verbatim.
    pub(crate) last_request: Option<LLMRequest>,

    /// Prompt-cache keep-alive bookkeeping for the current run.
    pub(crate) cache_keep_alive: CacheKeepAliveState,

    /// Where every request this agent sends is published for the `fork`
    /// subagent kind to start from; `None` when no fork reads this agent.
    pub(crate) fork_source: Option<crate::fork::ForkSource>,

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
        Self {
            advanced: Arc::new(llm.clone()),
            balanced: Arc::new(llm.clone()),
            fast: Arc::new(llm),
            compaction_model: None,
            idle_compacted: false,
            tier: ModelTier::default(),
            tools: AgentTools::new(),
            hooks: (),
            config,
            context: Context::default(),
            budget: ContextBudget::default(),
            prompt_cache: PromptCache::new(uuid::Uuid::new_v4().to_string()),
            context_sources: Vec::new(),
            reassembled: false,
            pending_compaction: None,
            completed_compaction: None,
            tool_results: VecDeque::new(),
            profile: None,
            initialized: false,
            session: SessionPhase::NotStarted,
            last_request_started_at: None,
            clock: crate::AnyClock::default(),
            background_receiver: None,
            permission_receiver: None,
            job_registry: None,
            detached_task_ids: HashSet::new(),
            sandbox_dir: None,
            restore_dir: None,
            last_request: None,
            cache_keep_alive: CacheKeepAliveState::default(),
            fork_source: None,
            cache_stats: crate::CacheStats::new(),
            #[cfg(feature = "skills")]
            skill_registry: None,
            #[cfg(feature = "skills")]
            active_skills: ActiveSkills::new(),
            #[cfg(feature = "skills")]
            active_allowed_tools: None,
        }
    }
}

impl<LLM: LanguageModel + Clone> Agent<LLM, LLM, LLM, ()> {
    /// Returns a builder for more complex agent construction.
    pub fn builder(llm: LLM) -> crate::builder::AgentBuilder<LLM, LLM, LLM, ()> {
        crate::builder::AgentBuilder::new(llm)
    }
}

impl<Advanced, Balanced, Fast, H, Compactor> Agent<Advanced, Balanced, Fast, H, Compactor>
where
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
        request: LLMRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, String>> + Send + 'a>> {
        match tier {
            ModelTier::Advanced => Box::pin(
                advanced
                    .respond(request)
                    .map(|event| event.map_err(|error| error.to_string())),
            ),
            ModelTier::Balanced => Box::pin(
                balanced
                    .respond(request)
                    .map(|event| event.map_err(|error| error.to_string())),
            ),
            ModelTier::Fast => Box::pin(
                fast.respond(request)
                    .map(|event| event.map_err(|error| error.to_string())),
            ),
        }
    }

    async fn collect_model_turn(
        &mut self,
        request: LLMRequest,
        events: &EventSink,
    ) -> Result<ModelTurn, AgentError> {
        let mut text_chunks = Vec::new();
        let mut tool_calls = Vec::new();
        let mut reasoning = Vec::new();
        let mut malformed_function_call = false;

        let stream = Self::respond_stream(
            self.tier,
            &self.advanced,
            &self.balanced,
            &self.fast,
            request,
        );
        futures_lite::pin!(stream);
        while let Some(event) = stream.next().await {
            match handle_model_event(
                &self.hooks,
                &mut self.cache_stats,
                &mut self.budget,
                event,
                &mut text_chunks,
                &mut tool_calls,
                &mut reasoning,
            )
            .await?
            {
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
        self.context
            .push(Message::user(&prompt).with_attachments(attachments));

        let outcome = self.run_agent_loop(run_id.as_str(), 0, events).await?;
        self.finish_run(run_id.as_str(), outcome, events).await
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
        let Some(request) = self.prepare_request().await? else {
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

        if tool_calls.is_empty() {
            return self.finish_no_tool_turn(response_text, turn, events).await;
        }

        // The reasoning has to travel with the turn that produced these calls,
        // or the model receives their results without the thinking behind them.
        self.context.push(Message::assistant_with_reasoning(
            &response_text,
            tool_calls.clone(),
            model_turn.reasoning,
        ));
        Self::push_tool_start_events(&tool_calls, events);
        self.execute_tool_calls(&tool_calls, turn, events).await;
        self.apply_tool_results(events)?;
        self.push_tool_turn_alerts().await;
        self.push_turn_boundary_events(run_id, turn, &response_text, events)
            .await
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
        let mut pending = self.pending_compaction.take();
        let fork_point = pending.as_ref().map(PendingCompaction::fork_point);
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

        match (landed, fork_point) {
            (Some(Ok(handoff)), Some(fork_point)) => {
                self.completed_compaction = Some(CompletedCompaction {
                    fork_point,
                    handoff,
                });
            }
            (Some(Err(error)), _) => {
                tracing::warn!(%error, "forked compaction failed");
            }
            (Some(Ok(_)), None) | (None, _) => {}
        }
        self.pending_compaction = pending;
        turn
    }

    async fn finish_no_tool_turn(
        &mut self,
        response_text: String,
        turn: usize,
        events: &EventSink,
    ) -> Result<TurnControl, AgentError> {
        if !response_text.is_empty() {
            self.context.push(Message::assistant(&response_text));
        }
        events.emit(AgentEvent::turn_complete(turn, false));
        if self.push_final_turn_alerts().await {
            return Ok(TurnControl::Continue);
        }
        Ok(TurnControl::Break {
            final_text: response_text,
            stop_reason: StopReason::NoToolCalls,
        })
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
                .position(|result| matches!(result, Ok((id, _, _)) if *id == call.id))
                .and_then(|index| parked.remove(index));
            let content = match finished {
                Some(Ok((_, _, result))) => result.render_for_model(),
                _ => ToolCallCancelledTemplate { tool: &call.name }
                    .render()
                    .expect("tool call cancelled template must render"),
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
        let permission_receiver = self.permission_receiver.clone();
        let mut permission_pause_tracker = PermissionPauseTracker::from_tool_calls(tool_calls);
        let message_count = self.context.len_recent();
        let (hooks, tools) = (&self.hooks, &self.tools);
        let results = &mut self.tool_results;
        let tool_futures = tool_calls.iter().map(|call| {
            let args_json = call.arguments.to_string();
            async move {
                let tool_ctx = ToolUseContext {
                    tool_name: &call.name,
                    arguments: &args_json,
                    turn,
                    message_count,
                };
                let (result, duration) =
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
                Ok((call.id.clone(), call.name.clone(), tool_result))
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
    ) -> Result<(aither_core::llm::ToolResult, Duration), AgentError> {
        match hooks.pre_tool_use(tool_ctx).await {
            PreToolAction::Abort(reason) => Err(AgentError::HookRejected {
                hook: "pre_tool_use",
                reason,
            }),
            PreToolAction::Deny(reason) => {
                Ok((aither_core::llm::ToolResult::error(reason), Duration::ZERO))
            }
            PreToolAction::Allow => {
                let start = Instant::now();
                // The agent loop has nowhere to show a tool's progress.
                let result = tools
                    .call(&call.name, args_json, aither_core::llm::ToolContext::new())
                    .await
                    .unwrap_or_else(|error| {
                        let mut message = String::from("Error: ");
                        message.push_str(error.to_string().as_str());
                        aither_core::llm::ToolResult::error(message)
                    });
                Ok((result, start.elapsed()))
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
    /// [`Self::settle_interrupted_tool_calls`].
    fn apply_tool_results(&mut self, events: &EventSink) -> Result<(), AgentError> {
        let mut has_tool_error = false;
        let mut unknown_tools: Vec<String> = Vec::new();
        let known_tools: Vec<String> = self
            .tools
            .active_definitions()
            .into_iter()
            .map(|definition| definition.name().to_string())
            .collect();
        while let Some(result) = self.tool_results.pop_front() {
            let (call_id, call_name, tool_result) = result?;
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
            has_tool_error |= tool_result_is_error(&tool_result, &content);
            self.push_terminal_followup_events(is_terminal_call, &tool_result, events);
            self.context.push(Message::tool(&call_id, content));
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
        self.push_completed_background_events(events);
        Ok(())
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
            let reminder = BackgroundStartedReminderTemplate {
                task_id: &started.task_id,
                output_preview: &started.output_preview,
                output_file: &started.output_file,
            }
            .render()
            .expect("background started reminder template must render");
            self.context.push(Message::system(&reminder));
            events.emit(AgentEvent::background_task_started(
                started.task_id,
                started.output_preview,
                started.output_file.url,
            ));
        }
    }

    fn push_completed_background_events(&mut self, events: &EventSink) {
        if let Some(receiver) = self.background_receiver.clone() {
            self.push_job_alerts();
            for task in receiver.take_completed() {
                tracing::info!(task_id = %task.task_id, "background task completed");
                self.push_completed_task_event(task, events);
            }
        }
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
        let mut needs_model = self.drain_background_receiver(receiver, events);
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
                    if self.push_job_alert(alert) {
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
        let CacheLifetime::Limited(lifetime) = self.prompt_cache.lifetime() else {
            return Ok(());
        };
        let Some(entry_started_at) = self.cache_entry_started_at() else {
            return Ok(());
        };
        if self.clock.since(entry_started_at) + keep_alive.send_before_expiry < lifetime {
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
            request,
        );
        futures_lite::pin!(stream);
        while let Some(event) = stream.next().await {
            match event {
                Ok(Event::Usage(usage)) => self.cache_stats.record(&usage),
                Ok(_) => {}
                Err(error) => return Err(AgentError::Llm(error)),
            }
        }
        self.cache_keep_alive.last_sent_at = Some(self.clock.now());
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

    /// When the provider's prompt-cache entry for this agent's context was
    /// last started or refreshed: the later of the last request's start and
    /// the last keep-alive send. `None` before the first request.
    #[must_use]
    pub fn cache_entry_started_at(&self) -> Option<Instant> {
        self.cache_keep_alive
            .last_sent_at
            .into_iter()
            .chain(self.last_request_started_at)
            .max()
    }

    /// How long the provider's prompt-cache entry for this agent's context
    /// stays warm without a hit.
    #[must_use]
    pub const fn cache_lifetime(&self) -> CacheLifetime {
        self.prompt_cache.lifetime()
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
        let Some(receiver) = self.background_receiver.clone() else {
            return false;
        };
        let mut wake = false;
        for alert in receiver.take_job_alerts() {
            wake |= self.push_job_alert(alert);
        }
        wake
    }

    /// Pushes one durable alert for a background job alert; returns whether
    /// it is a job blocked on stdin, which wakes a parked wait.
    fn push_job_alert(&mut self, alert: JobAlert) -> bool {
        match alert {
            JobAlert::DescendantsKilled(killed) => {
                tracing::info!(
                    killed = killed.processes.len() + killed.omitted,
                    "foreground command left processes running; killed"
                );
                self.push_system_alert(job_alert_text(&JobAlertText::DescendantsKilled {
                    processes: &killed.processes,
                    omitted: killed.omitted,
                }));
                false
            }
            JobAlert::DirectoryReset(reset) => {
                tracing::info!(dir = %reset.dir.display(), reason = ?reset.reason, "carried working directory reset");
                self.push_system_alert(job_alert_text(&JobAlertText::DirectoryReset {
                    dir: reset.dir.display().to_string(),
                    reason: reset.reason,
                }));
                false
            }
            JobAlert::LongRunning(job) => {
                tracing::info!(task_id = %job.task_id, "background job crossed long-running threshold");
                self.push_system_alert(job_alert_text(&JobAlertText::LongRunning {
                    task_id: &job.task_id,
                    elapsed: format_job_elapsed(job.elapsed),
                }));
                false
            }
            JobAlert::StdinBlocked(job) => {
                tracing::info!(task_id = %job.task_id, "background job is blocked on stdin");
                self.push_system_alert(job_alert_text(&JobAlertText::StdinBlocked {
                    task_id: &job.task_id,
                }));
                true
            }
            JobAlert::StdinProbeFailed(job) => {
                tracing::warn!(task_id = %job.task_id, error = %job.error, "stdin watch failed");
                self.push_system_alert(job_alert_text(&JobAlertText::StdinProbeFailed {
                    task_id: &job.task_id,
                    error: &job.error,
                }));
                false
            }
        }
    }

    fn drain_background_receiver(
        &mut self,
        receiver: &BackgroundTaskReceiver,
        events: &EventSink,
    ) -> bool {
        let mut announced = false;
        for task in receiver.take_completed() {
            tracing::info!(task_id = %task.task_id, "background task completed (final check)");
            announced |= self.push_completed_task_event(task, events);
        }
        announced
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
        let result_msg = format_background_task_result(&task);
        self.context.push(Message::system(&result_msg));
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
        use futures_lite::StreamExt;

        let stream = self.run(prompt, std::iter::empty());
        futures_lite::pin!(stream);

        let mut final_text = String::new();
        while let Some(event) = stream.next().await {
            match event?.event {
                AgentEvent::Text(chunk) => final_text.push_str(&chunk),
                AgentEvent::Complete {
                    final_text: text, ..
                } => {
                    return Ok(text);
                }
                _ => {}
            }
        }
        Ok(final_text)
    }

    /// Runs the agent with streaming events.
    ///
    /// Returns a stream of `AgentEvent`s that can be consumed to observe
    /// the agent's progress in real-time. Text chunks are yielded as they
    /// arrive from the LLM for true streaming display.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = agent.run("Implement the feature", std::iter::empty());
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         AgentEvent::Text(t) => print!("{}", t),
    ///         AgentEvent::Complete { .. } => break,
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn run(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> impl Stream<Item = Result<TimedEvent, AgentError>> + '_ {
        let prompt = prompt.to_string();
        let attachments: Vec<Attachment> = attachments.into_iter().collect();
        drive_run(move |sink| async move { self.run_streaming(prompt, attachments, &sink).await })
    }

    /// Continues the conversation without a new user message, so the model
    /// acts on the background events [`wait_background`](Self::wait_background)
    /// recorded. Streams like [`run`](Self::run).
    pub fn resume(&mut self) -> impl Stream<Item = Result<TimedEvent, AgentError>> + '_ {
        drive_run(move |sink| async move { self.resume_streaming(&sink).await })
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
    pub async fn wait_background(&mut self) -> BackgroundWake {
        let Some(receiver) = self.background_receiver.clone() else {
            return std::future::pending().await;
        };
        let event = receiver.recv_event().await;
        let (sender, emitted) = async_channel::unbounded();
        let sink = EventSink { sender };
        let mut needs_model = match event {
            BackgroundTaskEvent::Completed(task) => {
                tracing::info!(task_id = %task.task_id, "background task completed (idle)");
                self.push_completed_task_event(*task, &sink)
            }
            BackgroundTaskEvent::Alert(alert) => self.push_job_alert(alert),
        };
        needs_model |= self.drain_background_receiver(&receiver, &sink);
        needs_model |= self.push_job_alerts();
        drop(sink);
        let mut events = Vec::new();
        while let Ok(event) = emitted.try_recv() {
            events.push(event);
        }
        BackgroundWake {
            events,
            needs_model,
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
        self.hooks
            .on_lifecycle(&external_hook::BeforeCompaction {
                messages: messages_compacted,
                focus: focus.map(str::to_string),
            })
            .await;
        let tail = self.render_tail().await;
        let definitions = self.tools.active_definitions();
        let messages = self.context.build_request(&tail);
        // The same cache controls as the requests that wrote the prefix, so a
        // compaction on the session's own model is served from that cache.
        let request = LLMRequest::new(messages.clone())
            .with_parameters(self.prompt_cache.apply(
                self.config.request_parameters.clone(),
                &messages,
                !definitions.is_empty(),
            ))
            .with_tool_definitions(definitions);
        #[cfg(feature = "skills")]
        let skills = self.skill_catalog();
        #[cfg(not(feature = "skills"))]
        let skills = Vec::new();
        let fork = CompactionFork::new(
            &request,
            self.context.fork_point(),
            self.context.handoff_file_index(),
            focus,
            &skills,
        )
        .map_err(|error| AgentError::Llm(error.to_string()))?;
        let fork_point = fork.fork_point();
        let handoff = self
            .compaction_future(fork)
            .await
            .map_err(|error| AgentError::Llm(error.to_string()))?;
        let summary = handoff.render_markdown();

        self.completed_compaction = Some(CompletedCompaction {
            fork_point,
            handoff,
        });
        self.apply_completed_compaction().await;

        Ok(Some(CompactResult {
            messages_compacted,
            messages_remaining: self.context.len_recent(),
            summary,
        }))
    }

    /// Runs `fork` to completion on the compaction model: the configured one,
    /// or else the session's own tier.
    async fn compaction_future(
        &self,
        fork: CompactionFork,
    ) -> Result<HandoffDocument, crate::context::CompactionError> {
        if let Some(model) = &self.compaction_model {
            return fork.run(Arc::clone(model)).await;
        }
        match self.tier {
            ModelTier::Advanced => fork.run(Arc::clone(&self.advanced)).await,
            ModelTier::Balanced => fork.run(Arc::clone(&self.balanced)).await,
            ModelTier::Fast => fork.run(Arc::clone(&self.fast)).await,
        }
    }

    /// Starts `fork` in the background on the compaction model: the
    /// configured one, or else the session's own tier.
    fn start_pending_compaction(&self, fork: CompactionFork) -> PendingCompaction {
        if let Some(model) = &self.compaction_model {
            return PendingCompaction::start(fork, Arc::clone(model));
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
            || self.pending_compaction.is_some()
            || self.completed_compaction.is_some()
            || self.context.is_conversation_empty()
        {
            return None;
        }
        let CacheLifetime::Limited(lifetime) = self.prompt_cache.lifetime() else {
            return None;
        };
        let idle = self.clock.since(self.cache_entry_started_at()?);
        if idle >= lifetime {
            return None;
        }
        let due_at = lifetime.saturating_sub(self.config.context_policy.idle_compact_before_expiry);
        Some(due_at.saturating_sub(idle))
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
        match CompactionFork::new(
            request,
            self.context.fork_point(),
            self.context.handoff_file_index(),
            focus,
            &skills,
        ) {
            Ok(fork) => {
                tracing::debug!(
                    fork_point = fork.fork_point(),
                    usage = self.budget.usage(),
                    "forking compaction alongside the run"
                );
                self.pending_compaction = Some(self.start_pending_compaction(fork));
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
        let Some(CompletedCompaction {
            fork_point,
            handoff,
        }) = self.completed_compaction.take()
        else {
            return;
        };

        // Compaction rewrites the cacheable prefix anyway, so this is where
        // staged system-block updates and superseded tool results fold in.
        self.context.reassemble();
        self.context.apply_handoff(&handoff, fork_point);
        self.reassembled = true;
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

        if let Some(restore_dir) = self.restore_dir.clone() {
            let tokens = self
                .budget
                .fraction_tokens(self.config.context_policy.restore_budget);
            let budget_chars = self.budget.chars_for_tokens(tokens);
            #[cfg(feature = "skills")]
            let skills = self.skill_catalog();
            #[cfg(not(feature = "skills"))]
            let skills = Vec::new();
            if let Some(block) =
                restore_context(&handoff, &[restore_dir], &skills, budget_chars).await
            {
                self.context.push_block_alert(&block);
            }
        }
    }

    /// Waits for a forked compaction to land, then applies it.
    ///
    /// Returns `false` when there is nothing to wait for or the compaction
    /// failed — in both cases the window is still over the pause threshold
    /// and the run cannot issue another request.
    async fn await_pending_compaction(&mut self) -> bool {
        let Some(mut pending) = self.pending_compaction.take() else {
            return false;
        };
        let fork_point = pending.fork_point();
        match pending.wait().await {
            Ok(handoff) => {
                self.completed_compaction = Some(CompletedCompaction {
                    fork_point,
                    handoff,
                });
                self.apply_completed_compaction().await;
                true
            }
            Err(error) => {
                tracing::error!(%error, "compaction failed with the window exhausted");
                false
            }
        }
    }

    /// Renders every context source into the volatile tail of a request.
    ///
    /// Pure: sources are only read, so a preview or a measurement can use
    /// this without consuming a diff the next real request should carry.
    /// Rendering is concurrent because the sources are independent — one
    /// reading `tasks.md` from disk must not wait on another polling the job
    /// registry.
    async fn render_tail(&self) -> Vec<Block> {
        #[cfg(feature = "skills")]
        let skills = self.active_skills.render();
        #[cfg(not(feature = "skills"))]
        let skills = std::future::ready(None);
        let (rendered, skills) = futures_lite::future::zip(
            futures::future::join_all(self.context_sources.iter().map(AnyContextSource::render)),
            skills,
        )
        .await;
        rendered
            .into_iter()
            .chain(std::iter::once(skills))
            .flatten()
            .collect()
    }

    /// Tells each source that the tail it produced has been sent.
    ///
    /// Separate from rendering so that only a request that actually goes out
    /// advances a source's baseline.
    async fn commit_tail(&mut self, tail: &[Block]) {
        for block in tail {
            for source in &mut self.context_sources {
                source.commit(block).await;
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

    /// Delivers the context lifecycle hooks injected since the last call, one
    /// system alert per piece. A piece leaves the hooks' inbox as it is
    /// delivered, so it reaches the model once.
    fn push_injected_context(&mut self) {
        for injected in self.hooks.take_injected_context() {
            self.push_system_alert(injected.alert_text());
        }
    }

    /// Collects the durable alerts sources raise as a request begins.
    ///
    /// Tells the sources whether the window was reassembled since the last
    /// time, which is when their earlier alerts stopped being in it.
    async fn push_request_alerts(&mut self) {
        self.push_injected_context();
        let reassembled = std::mem::take(&mut self.reassembled);
        for index in 0..self.context_sources.len() {
            if let Some(block) = self.context_sources[index].on_request(reassembled).await {
                self.budget.append(block.len());
                self.context.push_block_alert(&block);
            }
        }
    }

    /// Collects the durable alerts sources raise after a tool turn.
    async fn push_tool_turn_alerts(&mut self) {
        self.push_injected_context();
        for index in 0..self.context_sources.len() {
            if let Some(block) = self.context_sources[index].on_tool_turn_end().await {
                self.budget.append(block.len());
                self.context.push_block_alert(&block);
            }
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
                self.budget.append(block.len());
                self.context.push_block_alert(&block);
                pushed = true;
            }
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
        })
    }

    /// Restore a previously exported checkpoint bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when the checkpoint cannot be restored into this agent.
    pub fn restore_checkpoint(&mut self, checkpoint: AgentCheckpoint) -> Result<(), AgentError> {
        self.context.restore(checkpoint.context);
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
    async fn ensure_initialized(&mut self) -> Result<(), AgentError> {
        if self.initialized {
            return Ok(());
        }

        // The window the current tier sees is the only one the budget may
        // spend: compaction replays the main request's prefix through the
        // same model, so no second profile is involved.
        let profile = match self.tier {
            ModelTier::Advanced => self.advanced.profile().await.map_err(|e| e.to_string()),
            ModelTier::Balanced => self.balanced.profile().await.map_err(|e| e.to_string()),
            ModelTier::Fast => self.fast.profile().await.map_err(|e| e.to_string()),
        }
        .map_err(AgentError::Llm)?;
        self.budget.set_window(profile.context_length);
        self.profile = Some(profile);

        self.register_builtin_context_sources();

        // Populate static system blocks in Context from AgentConfig.
        // These form the stable, cacheable prefix.
        self.populate_system_blocks();

        self.initialized = true;
        Ok(())
    }

    /// Populates the Context's system blocks from `AgentConfig`.
    ///
    /// Called once during initialization. These blocks form the stable
    /// cacheable prefix of the system message.
    fn populate_system_blocks(&mut self) {
        if let Some(ref system_prompt) = self.config.system_prompt {
            self.context
                .insert_system_named("base_system", system_prompt);
        }

        if let Some(ref persona_prompt) = self.config.persona_prompt {
            self.context.insert_system_named("persona", persona_prompt);
        }

        if self.config.agent_kind == AgentKind::Coding {
            self.context.insert_system_named(
                "workspace_facts",
                "When discovering workspace guidance, load AGENT.md first. If AGENT.md is missing, load CLAUDE.md. Treat these files as repository policy for coding tasks; this behavior is not required for chatbot-style sessions.",
            );
        }

        // TerminalAgentBuilder pre-seeds this block rendered against the
        // command registry; keep that version when present.
        if !self.context.has_system_block("knowledge_and_time") {
            self.context.insert_system_named(
                "knowledge_and_time",
                KnowledgeAndTimeTemplate {
                    websearch_command: true,
                }
                .render()
                .expect("failed to render knowledge_and_time template"),
            );
        }

        self.context
            .insert_system_named("permissions", include_str!("prompts/permissions.txt"));

        let tool_hints = self.format_tool_hints_block();
        if !tool_hints.is_empty() {
            self.context.insert_system_named("tool_hints", &tool_hints);
        }
    }

    /// Registers the context sources the agent itself owns.
    ///
    /// A host's own sources are registered through the builder and kept, so
    /// this only adds what the agent has the state for.
    fn register_builtin_context_sources(&mut self) {
        if let Some(sandbox_dir) = self.sandbox_dir.clone() {
            self.context_sources
                .push(AnyContextSource::new(TasksDocument::new(
                    sandbox_dir,
                    self.config.tasks_stale_after_turns,
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
    /// [`ContextPolicy::compact_at`], and one that was already running is
    /// awaited if usage reached [`ContextPolicy::pause_at`].
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
    /// [`ContextPolicy::compact_at`]: crate::ContextPolicy::compact_at
    /// [`ContextPolicy::pause_at`]: crate::ContextPolicy::pause_at
    async fn prepare_request(&mut self) -> Result<Option<LLMRequest>, AgentError> {
        self.apply_completed_compaction().await;
        self.rebuild_if_cache_is_cold().await?;
        self.push_request_alerts().await;

        let definitions = self.tools.active_definitions();
        let mut tail = self.render_tail().await;
        let mut messages = self.context.build_request(&tail);
        self.budget
            .observe_assembled(request_chars(&messages, &definitions));

        let usage = self.budget.usage();
        if self.config.context_policy.should_compact(usage)
            && self.pending_compaction.is_none()
            && self.completed_compaction.is_none()
        {
            let fork_request = LLMRequest::new(messages.clone())
                .with_parameters(self.prompt_cache.apply(
                    self.config.request_parameters.clone(),
                    &messages,
                    !definitions.is_empty(),
                ))
                .with_tool_definitions(definitions.clone());
            self.hooks
                .on_lifecycle(&external_hook::BeforeCompaction {
                    messages: self.context.len_recent(),
                    focus: None,
                })
                .await;
            self.start_compaction(&fork_request, None);
        }

        if self.config.context_policy.should_pause(usage) {
            tracing::warn!(usage, "context window exhausted; waiting for compaction");
            if !self.await_pending_compaction().await {
                return Ok(None);
            }
            self.push_request_alerts().await;
            tail = self.render_tail().await;
            messages = self.context.build_request(&tail);
            self.budget
                .observe_assembled(request_chars(&messages, &definitions));
            // A landed compaction is still no guarantee the rebuilt request
            // fits; an unshrinkable request is paused rather than sent.
            if self.config.context_policy.should_pause(self.budget.usage()) {
                return Ok(None);
            }
        }

        self.commit_tail(&tail).await;
        self.last_request_started_at = Some(self.clock.now());
        self.idle_compacted = false;
        self.publish_fork_snapshot(&tail, &definitions);

        let has_tools = !definitions.is_empty();
        self.prompt_cache
            .observe_prefix(self.context.stable_prefix_fingerprint());
        let parameters =
            self.prompt_cache
                .apply(self.config.request_parameters.clone(), &messages, has_tools);
        let request = LLMRequest::new(messages)
            .with_parameters(parameters)
            .with_tool_definitions(definitions);
        self.last_request = Some(request.clone());
        Ok(Some(request))
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
            request_parameters: self.config.request_parameters.clone(),
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
        if self.compaction_model.is_none() || self.pending_compaction.is_some() {
            return Ok(());
        }
        let Some(entry_started_at) = self.cache_entry_started_at() else {
            return Ok(());
        };
        if !self
            .prompt_cache
            .lifetime()
            .is_cold(self.clock.since(entry_started_at))
        {
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
        let Some(entry_started_at) = self.cache_entry_started_at() else {
            return unchanged;
        };
        if !self
            .prompt_cache
            .lifetime()
            .is_cold(self.clock.since(entry_started_at))
        {
            return unchanged;
        }
        let report = self.context.reassemble();
        self.reassembled |= report.system_messages_pruned > 0;
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
        if policy.should_pause(metrics.usage_fraction) {
            return ContextWindowPhase::Exhausted;
        }
        if self.pending_compaction.is_some() || self.completed_compaction.is_some() {
            return ContextWindowPhase::Compacting;
        }
        if metrics.has_staged_system_updates
            || (self.compaction_model.is_some()
                && self
                    .cache_entry_started_at()
                    .is_some_and(|at| self.prompt_cache.lifetime().is_cold(self.clock.since(at))))
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
        self.context
            .push(Message::user(prompt).with_attachments(attachments));
        let tail = self.render_tail().await;
        let messages = self.context.build_request(&tail);
        self.context.restore(checkpoint);
        Ok(messages)
    }

    fn format_tool_hints_block(&self) -> String {
        let defs = self.tools.active_definitions();
        let mut lines = Vec::new();

        for def in defs {
            let desc = first_paragraph(def.description());
            if desc.is_empty() {
                continue;
            }
            let mut line = def.name().to_string();
            line.push_str(": ");
            line.push_str(desc.as_str());
            lines.push(line);
        }

        lines.join("\n")
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
fn format_background_task_result(task: &aither_sandbox::CompletedTask) -> String {
    let error_text;
    let (outcome, output, stderr) = match &task.result {
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
    BackgroundTerminalResultTemplate {
        task_id: &task.task_id,
        script: truncate_script(&task.script, 100),
        outcome,
        output: output.as_deref(),
        stderr: stderr.as_deref(),
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

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn first_paragraph(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if let Some((first, _)) = trimmed.split_once("\n\n") {
        return first.split_whitespace().collect::<Vec<_>>().join(" ");
    }

    trimmed.split_whitespace().collect::<Vec<_>>().join(" ")
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

    use aither_sandbox::{CompletedCommand, CompletedTask, TerminalError};
    use futures_core::Stream;

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
        }
    }

    #[test]
    fn background_result_exit_zero_with_output() {
        let task = completed(Ok(CompletedCommand {
            stdout: inline("all green"),
            stderr: None,
            exit_code: 0,
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
        }));
        assert_eq!(
            format_background_task_result(&task),
            "Background task 'amber-forest-thunder-pearl' (`make test`) finished with exit code 101.\n1 passed\nstderr:\nboom\n"
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
            "The command ended while sleep (pid 41), node (pid 42) and 3 more kept running, so they were killed with it. A process meant to keep running belongs in terminal_background."
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
            "<system-reminder>\nA terminal command is running in background (task_id=amber-forest-thunder-pearl).\nCurrent output snapshot (first max_lines):\n(no output yet)\nFull redirected output file: outputs/calm-river-bold-oak.txt (absolute path: /sandbox/outputs/calm-river-bold-oak.txt)\nUse terminal_read for incremental reads, read the stored file via terminal commands when needed, use terminal_input for stdin, and terminal_kill to stop it.\n</system-reminder>"
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

    #[tokio::test]
    async fn events_carry_their_emission_time() {
        use futures_lite::StreamExt;

        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 1000,
            },
            AgentConfig::default(),
        );
        let before = jiff::Timestamp::now();
        let events: Vec<TimedEvent> = {
            let stream = agent.run("hello", std::iter::empty());
            futures_lite::pin!(stream);
            let mut events = Vec::new();
            while let Some(item) = stream.next().await {
                if let Ok(event) = item {
                    events.push(event);
                }
            }
            events
        };
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
    async fn snapshot_phase_tracks_usage_against_the_policy() {
        let config = AgentConfig {
            system_prompt: Some("x".repeat(400)),
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
        assert!(snapshot.metrics.usage_fraction >= agent.config.context_policy.pause_at);
        assert_eq!(snapshot.metrics.recent_message_count, 1);
        assert_eq!(snapshot.metrics.context_window, 100);
    }

    #[tokio::test]
    async fn snapshot_reports_a_handoff_without_calling_it_a_phase() {
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.context_mut().set_handoff("resume here");

        let snapshot = agent
            .snapshot_context_window()
            .await
            .expect("snapshot should initialize");

        assert_eq!(snapshot.phase, ContextWindowPhase::Stable);
        assert!(snapshot.metrics.has_handoff);
    }

    #[tokio::test]
    async fn exhausted_window_pauses_the_run_instead_of_requesting() {
        let config = AgentConfig {
            system_prompt: Some("x".repeat(4_000)),
            ..AgentConfig::default()
        };
        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 100,
            },
            config,
        );
        agent.push_message(Message::user("hello"));
        agent.ensure_initialized().await.expect("agent initializes");

        // The conversation is non-empty, so a compaction is forked; the mock
        // model answers with an empty stream, so it cannot land, and the
        // request must not be sent on a window this far over the threshold.
        assert!(
            agent
                .prepare_request()
                .await
                .expect("request boundary")
                .is_none()
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

    #[test]
    fn permission_event_key_parses_terminal_modes() {
        let default_args = serde_json::to_value(TerminalArgs {
            description: "list the working directory".to_string(),
            script: "ls".to_string(),
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
                stage: PermissionEventStage::Waiting,
            })
            .expect("waiting event should map to tool call");
        let resolved = tracker
            .event_for(PermissionEvent {
                mode: TerminalMode::Unsafe,
                script: "rm -rf /tmp/demo".to_string(),
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
                context_length: 1000,
            },
            AgentConfig::default(),
        );
        agent.skill_registry = Some(Arc::new(registry));
        let _ = agent.activate_skills_for_prompt("please review this patch");
        agent.ensure_initialized().await.expect("agent initializes");

        let messages = agent
            .prepare_request()
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

    fn tasks_agent(dir: &std::path::Path) -> Agent<MockLlm, MockLlm, MockLlm, ()> {
        Agent::builder(MockLlm {
            context_length: 1_000_000,
        })
        .sandbox_dir(dir)
        .config(AgentConfig {
            tasks_stale_after_turns: 2,
            ..AgentConfig::default()
        })
        .build()
    }

    #[tokio::test]
    async fn tool_turn_alerts_from_sources_reach_the_conversation() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] pending\n").expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");

        agent.push_tool_turn_alerts().await; // baseline
        agent.push_tool_turn_alerts().await; // stale turn 1
        agent.push_tool_turn_alerts().await; // stale turn 2 -> alert

        let alerts = agent
            .context()
            .conversation_messages()
            .iter()
            .filter(|message| message.content().contains("<tasks_stale>"))
            .count();
        assert_eq!(alerts, 1, "a stale episode must alert exactly once");
    }

    #[tokio::test]
    async fn final_turn_alerts_keep_the_loop_running() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir should exist");
        std::fs::write(dir.path().join("tasks.md"), "- [ ] pending\n").expect("write tasks.md");
        let mut agent = tasks_agent(dir.path());
        agent.ensure_initialized().await.expect("agent initializes");

        assert!(
            agent.push_final_turn_alerts().await,
            "unchecked tasks must object to the model stopping"
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
            .prepare_request()
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

    /// A job blocked on stdin can never complete, so its alert must end the
    /// post-turn wait and resume the loop — otherwise the model never sees
    /// it until the job's own timeout kills it.
    #[tokio::test]
    async fn stdin_blocked_alert_wakes_the_parked_background_wait() {
        let (sender, receiver) = aither_sandbox::background_task_channel();
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000,
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
                1_000,
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
            context_length: 1_000,
        })
        .build();
        let mut control = Agent::builder(MockLlm {
            context_length: 1_000,
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
        assert_eq!(messages[hidden_pos + 1].content(), "hi");

        // Post-preview state is exactly what a normal first turn leaves:
        // initialized with the static blocks populated, and neither the
        // prompt nor the extra message persisted.
        assert!(previewed.initialized);
        assert_eq!(
            previewed.context().system_blocks(),
            control.context().system_blocks()
        );
        assert!(previewed.context().is_conversation_empty());
    }

    /// A mock whose `respond` replays scripted text and records requests,
    /// so tests can drive compaction and inspect what was sent.
    #[derive(Clone)]
    struct ScriptedLlm {
        context_length: u32,
        requests: Arc<std::sync::Mutex<Vec<LLMRequest>>>,
        responses: Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    }

    impl ScriptedLlm {
        fn new(context_length: u32, responses: impl IntoIterator<Item = String>) -> Self {
            Self {
                context_length,
                requests: Arc::new(std::sync::Mutex::new(Vec::new())),
                responses: Arc::new(std::sync::Mutex::new(responses.into_iter().collect())),
            }
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
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                context_length,
            )))
        }
    }

    const HANDOFF_JSON: &str = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#;

    /// Sizes a mock's context window so the agent's next request lands at
    /// `usage_percent` percent of it — the only honest way to aim between the
    /// `compact_at` and `pause_at` thresholds, since tool definitions and
    /// system blocks weigh as much as the conversation does.
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
        let config = AgentConfig {
            system_prompt: Some("x".repeat(400)),
            ..AgentConfig::default()
        };
        let (mut agent, _window) = window_for_usage(&config, &[Message::user("hello")], 75).await;
        agent.ensure_initialized().await.expect("agent initializes");
        agent.context_mut().set_handoff("resume here");
        agent.push_message(Message::user("hello"));

        let request = agent.prepare_request().await.expect("request boundary");

        assert!(
            request.is_some(),
            "usage between compact_at and pause_at must not withhold the request"
        );
        assert!(
            agent.pending_compaction.is_some(),
            "growth past compact_at must fork a compaction even after a handoff exists"
        );
    }

    /// Crossing `compact_at` during a tool round starts the compaction but
    /// still lets that turn's request out: forking must not block the turn.
    #[tokio::test]
    async fn a_tool_round_crossing_compact_at_forks_without_blocking() {
        let config = AgentConfig {
            system_prompt: Some("x".repeat(400)),
            ..AgentConfig::default()
        };
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

        let request = agent.prepare_request().await.expect("request boundary");

        assert!(
            request.is_some(),
            "crossing compact_at alone must not withhold the turn's request"
        );
        assert!(
            agent.pending_compaction.is_some(),
            "crossing compact_at must fork the compaction"
        );
    }

    /// At `pause_at` the boundary waits for the in-flight compaction and
    /// proceeds with the rebuilt request once it lands.
    #[tokio::test]
    async fn the_pause_waits_for_an_in_flight_compaction() {
        let config = AgentConfig {
            system_prompt: Some("x".repeat(400)),
            ..AgentConfig::default()
        };
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
        let fork = CompactionFork::new(&request, agent.context.fork_point(), &[], None, &[])
            .expect("fork");
        agent.pending_compaction = Some(PendingCompaction::start(fork, llm.clone()));

        let request = agent.prepare_request().await.expect("request boundary");

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
        agent.completed_compaction = Some(CompletedCompaction {
            fork_point: agent.context.fork_point(),
            handoff: HandoffDocument {
                current_task: "keep going".to_string(),
                ..HandoffDocument::default()
            },
        });

        let request = agent
            .prepare_request()
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
        agent.completed_compaction = Some(CompletedCompaction {
            fork_point,
            handoff: HandoffDocument {
                current_task: "keep going".to_string(),
                ..HandoffDocument::default()
            },
        });

        agent
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("request after apply");

        let conversation: Vec<String> = agent
            .context()
            .conversation_messages()
            .iter()
            .map(|m| m.content().to_string())
            .collect();
        assert_eq!(conversation, ["during one", "during two"]);
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
        let fork =
            CompactionFork::new(&main, agent.context.fork_point(), &[], None, &[]).expect("fork");
        agent.pending_compaction = Some(PendingCompaction::start(fork, llm.clone()));
        assert!(agent.await_pending_compaction().await);

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
            .prepare_request()
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
        .compaction_model(ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]))
        .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));

        // The default lifetime is one hour: five minutes is nothing.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(300));
        assert!(
            agent
                .prepare_request()
                .await
                .expect("request boundary")
                .is_some()
        );
        assert_eq!(agent.context().count_recent_system_messages(), 1);

        // Fifty-nine minutes is still inside the hour.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3540));
        assert!(
            agent
                .prepare_request()
                .await
                .expect("request boundary")
                .is_some()
        );
        assert_eq!(agent.context().count_recent_system_messages(), 1);

        // Sixty-one minutes is past it: the cached prefix is gone anyway,
        // so the compromises the append-only log forced get cleaned up.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
        assert!(
            agent
                .prepare_request()
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
        .compaction_model(MockLlm {
            context_length: 1_000_000,
        })
        .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.prompt_cache = agent
            .prompt_cache
            .clone()
            .with_lifetime(CacheLifetime::Permanent);
        agent.push_message(Message::user("hello"));
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));

        // Rewriting the prefix always costs a permanent cache, so no idle
        // gap ever justifies it — even a day later.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(86_400));
        assert!(
            agent
                .prepare_request()
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
        let mut agent = Agent::new(llm);
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("the first request goes out");
        agent
    }

    /// The idle compaction fires once, shortly before the cache lifetime ends,
    /// and not again while the session stays idle.
    #[tokio::test]
    async fn an_idle_session_compacts_once_before_its_cache_expires() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_after_one_request(llm.clone()).await;

        // Fifty-three minutes of a one-hour lifetime: not yet due.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3180));
        let due_in = agent
            .idle_compaction_due_in()
            .expect("a compaction is pending");
        assert!(due_in > Duration::from_secs(30) && due_in <= Duration::from_secs(60));
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert!(llm.recorded().is_empty());

        // Fifty-four minutes: due now.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3240));
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
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3300));
        assert_eq!(agent.idle_compaction_due_in(), None);
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert_eq!(llm.recorded().len(), 1, "one compaction per idle gap");

        // The next request re-arms it for the next gap.
        agent.push_message(Message::user("again"));
        agent
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("request after idle");
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3240));
        assert_eq!(agent.idle_compaction_due_in(), Some(Duration::ZERO));
    }

    /// An expired cache is not worth compacting for: the context is written
    /// again either way.
    #[tokio::test]
    async fn an_expired_cache_is_not_idle_compacted() {
        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_after_one_request(llm.clone()).await;
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
        assert_eq!(agent.idle_compaction_due_in(), None);
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert!(llm.recorded().is_empty());
    }

    #[tokio::test]
    async fn a_permanent_cache_lifetime_never_idle_compacts() {
        use crate::context::CacheLifetime;

        let llm = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = agent_after_one_request(llm.clone()).await;
        agent.prompt_cache = agent
            .prompt_cache
            .clone()
            .with_lifetime(CacheLifetime::Permanent);
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(86_400));
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
            .compaction_model(compactor.clone())
            .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("first request");

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3240));
        assert_eq!(agent.idle_compaction_due_in(), None);
        assert!(agent.compact_when_idle().await.expect("no-op").is_none());
        assert!(main.recorded().is_empty() && compactor.recorded().is_empty());

        agent.compact(None).await.expect("compaction succeeds");
        assert_eq!(compactor.recorded().len(), 1, "the compaction model ran it");
        assert!(main.recorded().is_empty(), "the session's model did not");

        // The forked compaction goes to the compaction model as well.
        let compactor = ScriptedLlm::new(1_000_000, [HANDOFF_JSON.to_string()]);
        let mut agent = Agent::builder(ScriptedLlm::new(1_000_000, []))
            .compaction_model(compactor.clone())
            .build();
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        let tail = agent.render_tail().await;
        let request = LLMRequest::new(agent.context.build_request(&tail));
        agent.start_compaction(&request, None);
        assert!(agent.await_pending_compaction().await);
        assert_eq!(compactor.recorded().len(), 1);
    }

    /// Idle reassembly and idle compaction exclude each other: an idle gap
    /// past the cache lifetime reassembles only when idle compaction is off.
    #[tokio::test]
    async fn idle_reassembly_runs_only_when_idle_compaction_is_off() {
        async fn system_reminders_after_a_gap(compaction_model: Option<ScriptedLlm>) -> usize {
            let llm = ScriptedLlm::new(1_000_000, []);
            let builder = Agent::builder(llm);
            let mut agent = if let Some(model) = compaction_model {
                builder.compaction_model(model).build()
            } else {
                builder.build()
            };
            agent.ensure_initialized().await.expect("agent initializes");
            agent.push_message(Message::user("hello"));
            agent.push_message(Message::system(
                "<system-reminder>ephemeral</system-reminder>",
            ));
            agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
            assert!(
                agent
                    .prepare_request()
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
            Some(compactor) => builder.compaction_model(compactor).build(),
            None => builder.build(),
        };
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("an old question"));
        agent.push_message(Message::assistant("an old answer"));
        agent.push_message(Message::user("the next question"));
        agent
            .prepare_request()
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

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
        let request = agent
            .prepare_request()
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
        assert!(agent.completed_compaction.is_none());

        // The send restarts the cache entry: the next one is warm.
        agent
            .prepare_request()
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

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
        agent
            .prepare_request()
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

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3540));
        let request = agent
            .prepare_request()
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

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(86_400));
        let request = agent
            .prepare_request()
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
        agent.prompt_cache = agent
            .prompt_cache
            .clone()
            .with_lifetime(CacheLifetime::Permanent);

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(86_400));
        let request = agent
            .prepare_request()
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

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
        assert!(agent.prepare_request().await.is_err());
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
            .prepare_request()
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
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("first request");

        agent.push_message(Message::assistant("working"));
        agent.push_message(Message::user("go on"));
        let later = agent
            .prepare_request()
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
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("first request");

        std::fs::write(dir.path().join("tasks.md"), "- [x] first\n- [ ] second\n")
            .expect("rewrite tasks.md");
        let changed = agent
            .prepare_request()
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
            .prepare_request()
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
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("first request");

        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
        let request = agent
            .prepare_request()
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
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("first request");

        agent.push_message(Message::assistant("done"));
        agent.completed_compaction = Some(CompletedCompaction {
            fork_point: agent.context.fork_point(),
            handoff: HandoffDocument {
                current_task: "keep going".to_string(),
                ..HandoffDocument::default()
            },
        });
        let request = agent
            .prepare_request()
            .await
            .expect("request boundary")
            .expect("request after compaction");

        assert_eq!(
            full_document_messages(&request),
            1,
            "the compaction dropped the earlier copy; exactly one is sent again"
        );
        let next = agent
            .prepare_request()
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

        {
            let stream = agent.run("go", std::iter::empty());
            futures_lite::pin!(stream);
            let mut running = false;
            while !running {
                tokio::select! {
                    _ = started_rx.recv() => running = true,
                    event = stream.next() => {
                        event.expect("run is alive").expect("no error");
                    }
                }
            }
        }

        {
            let stream = agent.run("again", std::iter::empty());
            futures_lite::pin!(stream);
            while let Some(event) = stream.next().await {
                event.expect("second run succeeds");
            }
        }
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
            .position(|message| message.content() == "again")
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

    async fn drain_run(
        stream: impl Stream<Item = Result<TimedEvent, AgentError>>,
    ) -> Vec<AgentEvent> {
        futures_lite::pin!(stream);
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.expect("the run should succeed").event);
        }
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
            result: Err(aither_sandbox::TerminalError::Execution(output.to_string())),
        }
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
}
