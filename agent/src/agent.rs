//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages the context window, forks compaction in the background, and
//! handles tool execution in an agent-controlled loop.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aither_core::{
    LanguageModel,
    llm::{
        Attachment, Event, LLMRequest, Message, ReasoningState, ToolCall,
        model::Profile as ModelProfile,
    },
};
use askama::Template;
use futures_core::Stream;
use futures_lite::StreamExt;
use sha2::{Digest, Sha256};
#[cfg(feature = "skills")]
use std::collections::HashSet;

#[cfg(feature = "skills")]
use crate::context::{ActiveSkills, ContextSource as _, SkillCatalogEntry};
use crate::{
    checkpoint::AgentCheckpoint,
    config::{AgentConfig, AgentKind},
    context::{
        AnyContextSource, BackgroundJobs, Block, CompactionFork, CompletedCompaction, Context,
        ContextBudget, HandoffDocument, PendingCompaction, PromptCache, TasksDocument,
        restore_context, serialize_xml,
    },
    context_window::{ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot},
    error::AgentError,
    event::{AgentEvent, RunPauseReason},
    hook::{
        CheckpointContext, CheckpointReason, Hook, PostToolAction, PreToolAction, StopContext,
        StopReason, ToolResultContext, ToolUseContext, TurnBoundaryAction, TurnBoundaryContext,
    },
    model_group::ModelTier,
    tools::AgentTools,
};

use aither_sandbox::{
    BackgroundReason, BackgroundTaskReceiver, JobRegistry, PermissionEvent,
    PermissionEventReceiver, PermissionEventStage, TERMINAL_STDIN_BLOCKED_NOTICE, TerminalArgs,
    TerminalExecutionMode, TerminalMode,
};
use std::collections::{HashMap, VecDeque};

const BACKGROUND_TASK_MAX_WAIT: Duration = Duration::from_secs(300);
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
    sender: async_channel::Sender<AgentEvent>,
}

impl EventSink {
    fn emit(&self, event: AgentEvent) {
        let _ = self.sender.try_send(event);
    }
}

/// One step of the merged run/event stream in [`Agent::run`].
enum RunStep {
    Event(Option<AgentEvent>),
    Finished(Result<(), AgentError>),
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
        Ok(Event::BuiltInToolResult { tool, result }) => {
            let formatted = format_builtin_tool_result(&tool, &result);
            text_chunks.push(formatted.clone());
            Ok(ModelEventAction::Emit(AgentEvent::Text(formatted)))
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
    let tool_calls_json = serde_json::to_string(tool_calls)
        .unwrap_or_else(|error| format!("failed to serialize tool calls: {error}"));
    Err(AgentError::Llm(format!(
        "provider emitted tool call with empty id; response_text={response_text:?}; tool_calls={tool_calls_json}"
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
    output_file: &'a str,
}

/// `knowledge_and_time` system block. `websearch_command` selects whether the
/// verification hint names the `websearch` CLI command or a provider-native
/// web search capability.
#[derive(Template)]
#[template(path = "knowledge_and_time.txt", escape = "none")]
pub struct KnowledgeAndTimeTemplate {
    pub websearch_command: bool,
}

#[derive(serde::Serialize)]
struct BackgroundTerminalResultXml {
    #[serde(rename = "@task_id")]
    task_id: String,
    script: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stderr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

struct BackgroundTaskStartedEvent {
    task_id: String,
    output_preview: String,
    output_file: String,
    reminder: String,
    reason: BackgroundReason,
}

struct TerminalInputNeededEvent {
    task_id: Option<String>,
    notice: String,
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
///
/// # Model Tier Selection
///
/// Each agent uses one tier for its main reasoning loop (set via `tier()`):
/// - Main agent: typically `ModelTier::Advanced`
/// - Explore subagent: `ModelTier::Balanced` (cheaper but capable)
/// - Quick tasks: `ModelTier::Fast`
///
/// The `Fast` tier is always used for compaction, regardless of main tier.
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
pub struct Agent<Advanced, Balanced = Advanced, Fast = Balanced, H = ()> {
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

    /// Compaction running alongside the current turn, if any.
    pub(crate) pending_compaction: Option<PendingCompaction>,

    /// Compaction that finished and is waiting for a request boundary.
    pub(crate) completed_compaction: Option<CompletedCompaction>,

    /// Cached model profile (for the selected tier).
    pub(crate) profile: Option<ModelProfile>,

    /// Whether tools have been bootstrapped.
    pub(crate) initialized: bool,

    /// Receiver for completed background terminal tasks.
    pub(crate) background_receiver: Option<BackgroundTaskReceiver>,
    /// Receiver for permission wait/resume events emitted by terminal execution.
    pub(crate) permission_receiver: Option<PermissionEventReceiver>,
    /// Registry for running background terminal tasks.
    pub(crate) job_registry: Option<JobRegistry>,

    /// Optional sandbox directory for working-doc supervision (`tasks.md`).
    pub(crate) sandbox_dir: Option<PathBuf>,

    /// Start time of the most recent LLM request issued by this agent.
    pub(crate) last_request_started_at: Option<Instant>,

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

impl<LLM: LanguageModel + Clone> Agent<LLM, LLM, LLM, ()> {
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
            tier: ModelTier::default(),
            tools: AgentTools::new(),
            hooks: (),
            config,
            context: Context::default(),
            budget: ContextBudget::default(),
            prompt_cache: PromptCache::new(uuid::Uuid::new_v4().to_string()),
            context_sources: Vec::new(),
            pending_compaction: None,
            completed_compaction: None,
            profile: None,
            initialized: false,
            last_request_started_at: None,
            background_receiver: None,
            permission_receiver: None,
            job_registry: None,
            sandbox_dir: None,
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

impl<Advanced, Balanced, Fast, H> Agent<Advanced, Balanced, Fast, H>
where
    Advanced: LanguageModel + 'static,
    Balanced: LanguageModel + 'static,
    // Compaction runs as an owned future that outlives the turn it races
    // against, so the fast model it borrows must outlive the agent.
    Fast: LanguageModel + 'static,
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
        events.emit(AgentEvent::run_start(
            run_id.clone(),
            self.config.max_iterations,
        ));
        self.ensure_initialized().await?;
        #[cfg(feature = "skills")]
        for event in self.activate_skills_for_prompt(&prompt) {
            events.emit(event);
        }
        self.context
            .push(Message::user(&prompt).with_attachments(attachments));

        let outcome = self.run_agent_loop(run_id.as_str(), 0, events).await?;
        if outcome.stop_reason != StopReason::EndTurn
            && self
                .collect_final_background_events(run_id.as_str(), outcome.iteration, events)
                .await?
        {
            return Ok(());
        }
        self.push_stop_events(run_id.as_str(), outcome, events)
            .await?;
        Ok(())
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
        let Some(request) = self.prepare_request().await else {
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
        let results = self.execute_tool_calls(&tool_calls, turn, events).await;
        self.apply_tool_results(results, events)?;
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

    async fn execute_tool_calls(
        &self,
        tool_calls: &[ToolCall],
        turn: usize,
        events: &EventSink,
    ) -> Vec<Result<(String, String, aither_core::llm::ToolResult), AgentError>> {
        let permission_receiver = self.permission_receiver.clone();
        let mut permission_pause_tracker = PermissionPauseTracker::from_tool_calls(tool_calls);
        let tool_futures = tool_calls.iter().map(|call| {
            let args_json = call.arguments.to_string();
            let message_count = self.context.len_recent();
            async move {
                let tool_ctx = ToolUseContext {
                    tool_name: &call.name,
                    arguments: &args_json,
                    turn,
                    message_count,
                };
                let (result, duration) = self
                    .call_tool_with_hooks(call, &args_json, &tool_ctx)
                    .await?;
                let result_ctx = ToolResultContext {
                    tool_name: &call.name,
                    arguments: &args_json,
                    result: &result,
                    duration,
                };
                let tool_result = match self.hooks.post_tool_use(&result_ctx).await {
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
        let mut results = Vec::new();
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
            Self::handle_execution_signal(
                next,
                &mut permission_pause_tracker,
                &mut results,
                events,
            );
        }
        if let Some(receiver) = &permission_receiver {
            for event in receiver.take_pending() {
                emit_permission_event(&mut permission_pause_tracker, &event, events);
            }
        }
        results
    }

    async fn call_tool_with_hooks(
        &self,
        call: &ToolCall,
        args_json: &str,
        tool_ctx: &ToolUseContext<'_>,
    ) -> Result<(aither_core::llm::ToolResult, Duration), AgentError> {
        match self.hooks.pre_tool_use(tool_ctx).await {
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
                let result = self
                    .tools
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
        results: &mut Vec<ToolExecutionResult>,
        events: &EventSink,
    ) {
        match signal {
            Some(ExecutionSignal::Tool(result)) => results.push(result),
            Some(ExecutionSignal::Permission(event)) => {
                emit_permission_event(tracker, &event, events);
            }
            None => {}
        }
    }

    fn apply_tool_results(
        &mut self,
        results: Vec<Result<(String, String, aither_core::llm::ToolResult), AgentError>>,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let mut has_tool_error = false;
        let mut unknown_tools: Vec<String> = Vec::new();
        let known_tools: Vec<String> = self
            .tools
            .active_definitions()
            .into_iter()
            .map(|definition| definition.name().to_string())
            .collect();
        for result in results {
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
            let content = tool_result.render_for_model()?;
            has_tool_error |= tool_result_is_error(&tool_result, &content);
            self.push_terminal_followup_events(is_terminal_call, &tool_result, &content, events);
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
        content: &str,
        events: &EventSink,
    ) {
        if !is_terminal_call || tool_result.is_error() {
            return;
        }
        if let Some(started) = Self::format_background_started_event(content) {
            self.context.push(Message::system(&started.reminder));
            events.emit(AgentEvent::background_task_started(
                started.task_id,
                started.output_preview,
                started.output_file,
                started.reason,
            ));
        }
        if let Some(waiting) = Self::detect_terminal_input_needed(content) {
            events.emit(AgentEvent::terminal_input_needed(
                waiting.task_id,
                waiting.notice,
            ));
        }
    }

    fn push_completed_background_events(&mut self, events: &EventSink) {
        if let Some(ref receiver) = self.background_receiver {
            for task in receiver.take_completed() {
                tracing::info!(task_id = %task.task_id, "background task completed");
                let result_msg = Self::format_background_task_result(&task);
                self.context.push(Message::system(&result_msg));
                events.emit(AgentEvent::background_task_completed(
                    task.task_id.clone(),
                    result_msg,
                ));
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
        let mut had_completed = self.drain_background_receiver(&receiver, events);
        let start = Instant::now();
        while start.elapsed() < BACKGROUND_TASK_MAX_WAIT {
            if let Some(task) = receiver.recv_timeout(BACKGROUND_TASK_POLL_INTERVAL).await {
                tracing::info!(task_id = %task.task_id, "background task completed (waiting)");
                self.push_completed_task_event(task, events);
                had_completed = true;
            } else if !self
                .background_receiver
                .as_ref()
                .is_some_and(aither_sandbox::BackgroundTaskReceiver::has_running)
            {
                break;
            }
        }
        if !had_completed {
            return Ok(false);
        }
        let outcome = self.run_agent_loop(run_id, iteration, events).await?;
        self.push_stop_events(run_id, outcome, events).await?;
        Ok(true)
    }

    fn drain_background_receiver(
        &mut self,
        receiver: &BackgroundTaskReceiver,
        events: &EventSink,
    ) -> bool {
        let completed_tasks = receiver.take_completed();
        let had_completed = !completed_tasks.is_empty();
        for task in completed_tasks {
            tracing::info!(task_id = %task.task_id, "background task completed (final check)");
            self.push_completed_task_event(task, events);
        }
        had_completed
    }

    fn push_completed_task_event(
        &mut self,
        task: aither_sandbox::CompletedTask,
        events: &EventSink,
    ) {
        let result_msg = Self::format_background_task_result(&task);
        self.context.push(Message::system(&result_msg));
        events.emit(AgentEvent::background_task_completed(
            task.task_id,
            result_msg,
        ));
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
            match event? {
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
    ) -> impl Stream<Item = Result<AgentEvent, AgentError>> + '_ {
        let prompt = prompt.to_string();
        let attachments: Vec<Attachment> = attachments.into_iter().collect();

        async_stream::stream! {
            let (sender, receiver) = async_channel::unbounded();
            let sink = EventSink { sender };
            let run = async {
                let result = self.run_streaming(prompt, attachments, &sink).await;
                drop(sink);
                result
            };
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

        let messages = self.context.conversation_messages();
        if messages.is_empty() {
            return Ok(None);
        }

        let messages_compacted = messages.len();
        let tail = self.render_tail().await;
        let request = LLMRequest::new(self.context.build_request(&tail))
            .with_parameters(self.config.request_parameters.clone())
            .with_tool_definitions(self.tools.active_definitions());
        #[cfg(feature = "skills")]
        let skills = self.skill_catalog();
        #[cfg(not(feature = "skills"))]
        let skills = Vec::new();
        let fork = CompactionFork::new(&request, self.context.fork_point(), focus, &skills)
            .map_err(|error| AgentError::Llm(error.to_string()))?;
        let fork_point = fork.fork_point();
        let handoff = match self.tier {
            ModelTier::Advanced => fork.run(Arc::clone(&self.advanced)).await,
            ModelTier::Balanced => fork.run(Arc::clone(&self.balanced)).await,
            ModelTier::Fast => fork.run(Arc::clone(&self.fast)).await,
        }
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

    /// Forks a compaction off `request` to run alongside the turns that follow.
    ///
    /// The compaction replays the session's current model, not a cheaper one:
    /// the handoff has to carry the same understanding of the conversation the
    /// next requests are made of.
    ///
    /// A failure to even start — an empty conversation — is not an error for
    /// the run: there is simply nothing to compact yet.
    fn start_compaction(&mut self, request: &LLMRequest, focus: Option<&str>) {
        #[cfg(feature = "skills")]
        let skills = self.skill_catalog();
        #[cfg(not(feature = "skills"))]
        let skills = Vec::new();
        match CompactionFork::new(request, self.context.fork_point(), focus, &skills) {
            Ok(fork) => {
                tracing::debug!(
                    fork_point = fork.fork_point(),
                    usage = self.budget.usage(),
                    "forking compaction alongside the run"
                );
                self.pending_compaction = Some(match self.tier {
                    ModelTier::Advanced => {
                        PendingCompaction::start(fork, Arc::clone(&self.advanced))
                    }
                    ModelTier::Balanced => {
                        PendingCompaction::start(fork, Arc::clone(&self.balanced))
                    }
                    ModelTier::Fast => PendingCompaction::start(fork, Arc::clone(&self.fast)),
                });
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
        tracing::info!(
            fork_point,
            kept = self.context.len_recent(),
            "applied handoff from forked compaction"
        );

        if let Some(sandbox_dir) = self.sandbox_dir.clone() {
            let tokens = self
                .budget
                .fraction_tokens(self.config.context_policy.restore_budget);
            let budget_chars = self.budget.chars_for_tokens(tokens);
            #[cfg(feature = "skills")]
            let skills = self.skill_catalog();
            #[cfg(not(feature = "skills"))]
            let skills = Vec::new();
            if let Some(block) =
                restore_context(&handoff, &[sandbox_dir], &skills, budget_chars).await
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

    /// Collects the durable alerts sources raise after a tool turn.
    async fn push_tool_turn_alerts(&mut self) {
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
    /// Returns `None` when the window is exhausted and no compaction can
    /// relieve it: the caller pauses the run rather than sending a request
    /// the provider would reject for length.
    ///
    /// [`ContextPolicy::compact_at`]: crate::ContextPolicy::compact_at
    /// [`ContextPolicy::pause_at`]: crate::ContextPolicy::pause_at
    async fn prepare_request(&mut self) -> Option<LLMRequest> {
        self.apply_completed_compaction().await;
        self.reassemble_if_cache_is_cold();

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
            self.start_compaction(&fork_request, None);
        }

        if self.config.context_policy.should_pause(usage) {
            tracing::warn!(usage, "context window exhausted; waiting for compaction");
            if !self.await_pending_compaction().await {
                return None;
            }
            tail = self.render_tail().await;
            messages = self.context.build_request(&tail);
            self.budget
                .observe_assembled(request_chars(&messages, &definitions));
            // A landed compaction is still no guarantee the rebuilt request
            // fits; an unshrinkable request is paused rather than sent.
            if self.config.context_policy.should_pause(self.budget.usage()) {
                return None;
            }
        }

        self.commit_tail(&tail).await;
        self.last_request_started_at = Some(Instant::now());

        let has_tools = !definitions.is_empty();
        self.prompt_cache
            .observe_prefix(self.context.stable_prefix_fingerprint());
        let parameters =
            self.prompt_cache
                .apply(self.config.request_parameters.clone(), &messages, has_tools);
        Some(
            LLMRequest::new(messages)
                .with_parameters(parameters)
                .with_tool_definitions(definitions),
        )
    }

    /// Rebuilds the window once the provider's cache has gone cold anyway.
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
        let Some(last_request_started_at) = self.last_request_started_at else {
            return unchanged;
        };
        if !self
            .prompt_cache
            .lifetime()
            .is_cold(last_request_started_at.elapsed())
        {
            return unchanged;
        }
        let report = self.context.reassemble();
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
            || self
                .last_request_started_at
                .is_some_and(|at| self.prompt_cache.lifetime().is_cold(at.elapsed()))
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

    /// Formats a reminder and event payload when `terminal` has been auto-promoted to background.
    fn format_background_started_event(tool_content: &str) -> Option<BackgroundTaskStartedEvent> {
        let payload: serde_json::Value = serde_json::from_str(tool_content).ok()?;
        let status = payload.get("status")?.as_str()?;
        if status != "running" {
            return None;
        }

        let task_id = payload.get("task_id")?.as_str()?.trim();
        if task_id.is_empty() {
            return None;
        }

        let output_preview = payload
            .get("stdout")
            .and_then(|stdout| stdout.get("content"))
            .and_then(|content| content.get("text"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or("(no output yet)");

        let output_file = payload
            .get("stdout")
            .and_then(|stdout| stdout.get("url"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(missing output file)");

        let reason = payload
            .get("background_reason")
            .cloned()
            .and_then(|raw| serde_json::from_value::<BackgroundReason>(raw).ok())
            .unwrap_or(BackgroundReason::Explicit);

        let reminder = BackgroundStartedReminderTemplate {
            task_id,
            output_preview,
            output_file,
        }
        .render()
        .ok()?;

        Some(BackgroundTaskStartedEvent {
            task_id: task_id.to_string(),
            output_preview: output_preview.to_string(),
            output_file: output_file.to_string(),
            reminder,
            reason,
        })
    }

    fn detect_terminal_input_needed(tool_content: &str) -> Option<TerminalInputNeededEvent> {
        let payload: aither_sandbox::TerminalResult = serde_json::from_str(tool_content).ok()?;
        if payload.status.as_deref() != Some("running") {
            return None;
        }
        let notice = match &payload.stdout {
            aither_sandbox::OutputEntry::Inline { content }
            | aither_sandbox::OutputEntry::Loaded { content, .. } => match content {
                aither_sandbox::Content::Text { text, .. } => text.as_str(),
                aither_sandbox::Content::Image { .. } => return None,
            },
            aither_sandbox::OutputEntry::Stored { content, .. } => match content {
                Some(aither_sandbox::Content::Text { text, .. }) => text.as_str(),
                Some(aither_sandbox::Content::Image { .. }) | None => return None,
            },
            aither_sandbox::OutputEntry::Empty => return None,
        };
        if !notice.starts_with(TERMINAL_STDIN_BLOCKED_NOTICE) {
            return None;
        }
        Some(TerminalInputNeededEvent {
            task_id: payload.task_id,
            notice: notice.to_string(),
        })
    }

    /// Formats a completed background task result as a system message.
    fn format_background_task_result(task: &aither_sandbox::CompletedTask) -> String {
        let mut xml = BackgroundTerminalResultXml {
            task_id: task.task_id.clone(),
            script: truncate_script(&task.script, 100).to_string(),
            exit_code: None,
            output: None,
            stderr: None,
            error: None,
        };

        match &task.result {
            Ok(result) => {
                xml.exit_code = Some(result.exit_code);
                let stdout = result.stdout.to_string();
                if !stdout.is_empty() {
                    xml.output = Some(stdout);
                }
                if let Some(stderr) = &result.stderr {
                    let stderr = stderr.to_string();
                    if !stderr.is_empty() {
                        xml.stderr = Some(stderr);
                    }
                }
            }
            Err(error) => {
                xml.error = Some(error.to_string());
            }
        }

        serialize_xml("background-terminal-result", &xml)
    }
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
    if tool_name != "terminal" {
        return None;
    }
    let args: TerminalArgs = serde_json::from_value(arguments.clone()).ok()?;
    let mode = match args.mode {
        TerminalExecutionMode::Sandboxed
        | TerminalExecutionMode::Default
        | TerminalExecutionMode::Ssh => return None,
        TerminalExecutionMode::Unsafe => TerminalMode::Unsafe,
    };
    Some(PermissionEventKey {
        mode,
        script: args.script,
    })
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

    use futures_core::Stream;

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
        assert!(agent.prepare_request().await.is_none());
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
            timeout: 30,
            max_lines: 200,
            raw: false,
        })
        .expect("default args should serialize");
        let unsafe_args = serde_json::to_value(TerminalArgs {
            description: "remove the demo directory".to_string(),
            script: "rm -rf /tmp/demo".to_string(),
            mode: TerminalExecutionMode::Unsafe,
            ssh_server_id: None,
            expect: aither_sandbox::OutputFormat::Text,
            resolution: aither_sandbox::MediaResolution::Auto,
            timeout: 30,
            max_lines: 200,
            raw: false,
        })
        .expect("unsafe args should serialize");
        let sandboxed_args = serde_json::to_value(TerminalArgs {
            description: "print the current working directory".to_string(),
            script: "pwd".to_string(),
            mode: TerminalExecutionMode::Sandboxed,
            ssh_server_id: None,
            expect: aither_sandbox::OutputFormat::Text,
            resolution: aither_sandbox::MediaResolution::Auto,
            timeout: 30,
            max_lines: 200,
            raw: false,
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
                timeout: 30,
                max_lines: 200,
                raw: false,
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
            .expect("a fresh window must accept a request");

        assert!(
            request.messages().iter().any(|message| {
                message.role() == Role::System
                    && message.content().contains("no native tool named foo")
            }),
            "ad-hoc alerts must reach the next request"
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

        let request = agent.prepare_request().await;

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

        let request = agent.prepare_request().await;

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
        let fork =
            CompactionFork::new(&request, agent.context.fork_point(), None, &[]).expect("fork");
        agent.pending_compaction = Some(PendingCompaction::start(fork, llm.clone()));

        let request = agent.prepare_request().await;

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

        agent.prepare_request().await.expect("request after apply");

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
        let fork = CompactionFork::new(&main, agent.context.fork_point(), None, &[]).expect("fork");
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

    /// A preview renders sources but does not commit them: the real request
    /// still carries the same tasks diff.
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
    /// passed — never earlier, and never for a permanent cache.
    #[tokio::test]
    async fn an_idle_gap_reassembles_only_past_the_cache_lifetime() {
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
        agent.ensure_initialized().await.expect("agent initializes");
        agent.push_message(Message::user("hello"));
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));

        // The default lifetime is one hour: five minutes is nothing.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(300));
        assert!(agent.prepare_request().await.is_some());
        assert_eq!(agent.context().count_recent_system_messages(), 1);

        // Fifty-nine minutes is still inside the hour.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3540));
        assert!(agent.prepare_request().await.is_some());
        assert_eq!(agent.context().count_recent_system_messages(), 1);

        // Sixty-one minutes is past it: the cached prefix is gone anyway,
        // so the compromises the append-only log forced get cleaned up.
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(3660));
        assert!(agent.prepare_request().await.is_some());
        assert_eq!(agent.context().count_recent_system_messages(), 0);
    }

    #[tokio::test]
    async fn a_permanent_cache_lifetime_never_reassembles() {
        use crate::context::CacheLifetime;

        let mut agent = Agent::new(MockLlm {
            context_length: 1_000_000,
        });
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
        assert!(agent.prepare_request().await.is_some());
        assert_eq!(agent.context().count_recent_system_messages(), 1);
    }
}
