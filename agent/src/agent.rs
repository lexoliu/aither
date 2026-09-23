//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages conversation memory, applies context compression, and
//! handles tool execution in an agent-controlled loop.

use num_traits::ToPrimitive as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use aither_core::{
    LanguageModel,
    llm::{
        Attachment, Event, LLMRequest, Message, ReasoningState, ToolCall,
        model::Profile as ModelProfile,
    },
};
#[cfg(feature = "skills")]
use aither_skills::Skill;
use askama::Template;
use futures_core::Stream;
use futures_lite::StreamExt;
use sha2::{Digest, Sha256};
#[cfg(feature = "skills")]
use std::collections::HashSet;

use crate::{
    checkpoint::AgentCheckpoint,
    compression::{ContextStrategy, estimate_context_usage},
    config::{AgentConfig, AgentKind},
    context::{Context, serialize_xml},
    context_window::{ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot},
    error::AgentError,
    event::{AgentEvent, RunPauseReason},
    handoff::HandoffDocument,
    hook::{
        CheckpointContext, CheckpointReason, Hook, PostToolAction, PreToolAction, StopContext,
        StopReason, ToolResultContext, ToolUseContext, TurnBoundaryAction, TurnBoundaryContext,
    },
    model_group::ModelTier,
    todo::{TodoItem, TodoList, TodoStatus},
    tools::AgentTools,
    transcript::Transcript,
    working_docs,
};

use aither_sandbox::{
    BackgroundReason, BackgroundTaskReceiver, JobRegistry, PermissionEvent,
    PermissionEventReceiver, PermissionEventStage, TERMINAL_STDIN_BLOCKED_NOTICE, TerminalArgs,
    TerminalExecutionMode, TerminalMode,
};
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "skills")]
use std::sync::Arc;

const BACKGROUND_TASK_MAX_WAIT: Duration = Duration::from_secs(300);
const BACKGROUND_TASK_POLL_INTERVAL: Duration = Duration::from_millis(100);
const FOCUS_INSTRUCTION_PLACEHOLDER: &str = "{focus_instruction}";

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

macro_rules! drain_model_stream {
    ($stream:expr, $hooks:expr, $cache_stats:expr, $events:expr, $text_chunks:expr, $tool_calls:expr, $reasoning:expr, $malformed:expr) => {{
        let stream = $stream;
        futures_lite::pin!(stream);
        while let Some(event) = stream.next().await {
            match handle_model_event(
                $hooks,
                $cache_stats,
                event.map_err(|error| error.to_string()),
                $text_chunks,
                $tool_calls,
                $reasoning,
            )
            .await?
            {
                ModelEventAction::Emit(event) => $events.emit(event),
                ModelEventAction::Continue => {}
                ModelEventAction::RetryMalformed => {
                    $malformed = true;
                    break;
                }
            }
        }
    }};
}

async fn handle_model_event<H: Hook>(
    hooks: &H,
    cache_stats: &mut crate::CacheStats,
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

#[derive(serde::Serialize)]
struct SystemReminder {
    #[serde(rename = "$text")]
    content: String,
}

#[derive(serde::Serialize)]
struct Tasks {
    #[serde(rename = "$text")]
    content: String,
}

#[derive(serde::Serialize)]
struct TasksDiffReminder {
    #[serde(rename = "$text")]
    diff: String,
}

#[derive(Template)]
#[template(path = "todo_reminder.txt", escape = "none")]
struct TodoReminderTemplate<'a> {
    items_json: &'a str,
}

#[derive(Template)]
#[template(path = "todo_context.txt", escape = "none")]
struct TodoContextTemplate<'a> {
    items_json: &'a str,
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

#[derive(Template)]
#[template(path = "next_task_reminder.txt", escape = "none")]
struct NextTaskReminderTemplate<'a> {
    completed_task: &'a str,
    next_task: &'a str,
    active_form: &'a str,
}

#[derive(Template)]
#[template(path = "all_tasks_complete_reminder.txt", escape = "none")]
struct AllTasksCompleteReminderTemplate<'a> {
    completed_task: &'a str,
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

#[cfg(feature = "skills")]
#[derive(serde::Serialize)]
struct SkillInstructionXml {
    name: String,
    description: String,
    instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resource_paths: Option<Vec<String>>,
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
    pub(crate) advanced: Advanced,

    /// Balanced LLM for moderate tasks (subagents).
    pub(crate) balanced: Balanced,

    /// Fast LLM for quick tasks (compaction, ask command).
    pub(crate) fast: Fast,

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

    /// Cached model profile (for the selected tier).
    pub(crate) profile: Option<ModelProfile>,

    /// Cached fast model profile (for compression decisions).
    pub(crate) fast_profile: Option<ModelProfile>,

    /// Whether tools have been bootstrapped.
    pub(crate) initialized: bool,

    /// Todo list for tracking long tasks.
    pub(crate) todo_list: Option<TodoList>,

    /// Receiver for completed background terminal tasks.
    pub(crate) background_receiver: Option<BackgroundTaskReceiver>,
    /// Receiver for permission wait/resume events emitted by terminal execution.
    pub(crate) permission_receiver: Option<PermissionEventReceiver>,
    /// Registry for running background terminal tasks.
    pub(crate) job_registry: Option<JobRegistry>,

    /// Optional readable transcript for long-context recovery.
    pub(crate) transcript: Option<Transcript>,

    /// Optional sandbox directory for working-doc supervision (`tasks.md`).
    pub(crate) sandbox_dir: Option<PathBuf>,

    /// Last observed working-doc snapshot for diff reminders.
    pub(crate) last_working_docs: Option<working_docs::WorkingDocsSnapshot>,

    /// Start time of the most recent LLM request issued by this agent.
    pub(crate) last_request_started_at: Option<Instant>,

    /// Transient per-turn system messages injected by the host application.
    ///
    /// These participate in prompt assembly for the current turn but are not
    /// stored in the persistent context or checkpoints.
    pub(crate) transient_system_messages: Vec<String>,

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
    #[cfg(feature = "skills")]
    pub(crate) active_skills: Vec<Skill>,

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
            advanced: llm.clone(),
            balanced: llm.clone(),
            fast: llm,
            tier: ModelTier::default(),
            tools: AgentTools::new(),
            hooks: (),
            config,
            context: Context::default(),
            profile: None,
            fast_profile: None,
            initialized: false,
            todo_list: None,
            background_receiver: None,
            permission_receiver: None,
            job_registry: None,
            transcript: None,
            sandbox_dir: None,
            last_working_docs: None,
            last_request_started_at: None,
            transient_system_messages: Vec::new(),
            cache_stats: crate::CacheStats::new(),
            #[cfg(feature = "skills")]
            skill_registry: None,
            #[cfg(feature = "skills")]
            active_skills: Vec::new(),
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
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    H: Hook,
{
    async fn collect_model_turn(
        &mut self,
        request: LLMRequest,
        events: &EventSink,
    ) -> Result<ModelTurn, AgentError> {
        let mut text_chunks = Vec::new();
        let mut tool_calls = Vec::new();
        let mut reasoning = Vec::new();
        let mut malformed_function_call = false;

        match self.tier {
            ModelTier::Advanced => {
                drain_model_stream!(
                    self.advanced.respond(request),
                    &self.hooks,
                    &mut self.cache_stats,
                    events,
                    &mut text_chunks,
                    &mut tool_calls,
                    &mut reasoning,
                    malformed_function_call
                );
            }
            ModelTier::Balanced => {
                drain_model_stream!(
                    self.balanced.respond(request),
                    &self.hooks,
                    &mut self.cache_stats,
                    events,
                    &mut text_chunks,
                    &mut tool_calls,
                    &mut reasoning,
                    malformed_function_call
                );
            }
            ModelTier::Fast => {
                drain_model_stream!(
                    self.fast.respond(request),
                    &self.hooks,
                    &mut self.cache_stats,
                    events,
                    &mut text_chunks,
                    &mut tool_calls,
                    &mut reasoning,
                    malformed_function_call
                );
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
        self.ensure_initialized().await;
        #[cfg(feature = "skills")]
        for event in self.activate_skills_for_prompt(&prompt) {
            events.emit(event);
        }
        self.maybe_compress().await?;

        self.context
            .push(Message::user(&prompt).with_attachments(attachments));
        if let Some(transcript) = &self.transcript {
            transcript.write_user_message(&prompt).await;
        }

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
        let messages = self.build_live_request_messages().await;
        let request =
            LLMRequest::new(messages).with_tool_definitions(self.tools.active_definitions());
        let model_turn = self.collect_model_turn(request, events).await?;

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
        let old_todo_items = self
            .todo_list
            .as_ref()
            .map(super::todo::TodoList::items)
            .unwrap_or_default();
        let tool_names = tool_calls
            .iter()
            .map(|call| call.name.clone())
            .collect::<Vec<_>>();

        self.push_tool_start_events(&tool_calls, events).await;
        let results = self.execute_tool_calls(&tool_calls, turn, events).await;
        self.apply_tool_results(results, &tool_names, &old_todo_items, events)
            .await?;
        self.push_turn_boundary_events(run_id, turn, &response_text, events)
            .await
    }

    async fn finish_no_tool_turn(
        &mut self,
        response_text: String,
        turn: usize,
        events: &EventSink,
    ) -> Result<TurnControl, AgentError> {
        if !response_text.is_empty() {
            self.context.push(Message::assistant(&response_text));
            if let Some(transcript) = &self.transcript {
                transcript.write_assistant_text(&response_text).await;
            }
        }
        events.emit(AgentEvent::turn_complete(turn, false));
        if self.inject_working_doc_continue_reminder().await {
            return Ok(TurnControl::Continue);
        }
        Ok(TurnControl::Break {
            final_text: response_text,
            stop_reason: StopReason::NoToolCalls,
        })
    }

    async fn push_tool_start_events(&self, tool_calls: &[ToolCall], events: &EventSink) {
        for call in tool_calls {
            let args = call.arguments.to_string();
            if let Some(transcript) = &self.transcript {
                transcript.write_tool_call(&call.name, &args).await;
            }
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
                let result = self
                    .tools
                    .call(&call.name, args_json)
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

    async fn apply_tool_results(
        &mut self,
        results: Vec<Result<(String, String, aither_core::llm::ToolResult), AgentError>>,
        tool_names: &[String],
        old_todo_items: &[TodoItem],
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
            if let Some(transcript) = &self.transcript {
                transcript.write_tool_result(&call_name, &tool_result).await;
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
                self.context.insert_reminder(&SystemReminder {
                    content: format!(
                        "There is no native tool named '{name}'. This runtime is terminal-first: the only native tools are terminal, terminal_kill, terminal_input, and terminal_read; every other capability is a CLI command executed through the `terminal` tool. If '{name}' is a capability, run it as a command (check with `{name} --help`)."
                    ),
                });
            }
        } else if has_tool_error {
            self.context.insert_reminder(&SystemReminder {
                content: "A tool call failed. Re-assess the current state, inspect the latest tool result carefully, and choose the next action deliberately. Native tools remain terminal, terminal_kill, terminal_input, and terminal_read.".to_string(),
            });
        }
        self.push_todo_followup(tool_names, old_todo_items);
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

    fn push_todo_followup(&mut self, tool_names: &[String], old_todo_items: &[TodoItem]) {
        if !tool_names.iter().any(|name| name == "todo") {
            return;
        }
        let new_items = self
            .todo_list
            .as_ref()
            .map(super::todo::TodoList::items)
            .unwrap_or_default();
        let newly_completed = new_items.iter().find(|new_item| {
            new_item.status == TodoStatus::Completed
                && old_todo_items.iter().any(|old| {
                    old.content == new_item.content && old.status != TodoStatus::Completed
                })
        });
        let reminder = newly_completed
            .and_then(|completed| self.format_next_task_reminder(&completed.content))
            .or_else(|| self.format_todo_reminder());
        if let Some(reminder) = reminder {
            self.context.push(Message::system(&reminder));
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
        &mut self,
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
        &mut self,
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

    /// Replaces transient per-turn system messages.
    pub fn set_transient_system_messages(&mut self, messages: impl IntoIterator<Item = String>) {
        self.transient_system_messages = messages
            .into_iter()
            .map(|message| message.trim().to_string())
            .filter(|message| !message.is_empty())
            .collect();
    }

    /// Clears transient per-turn system messages.
    pub fn clear_transient_system_messages(&mut self) {
        self.transient_system_messages.clear();
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
        self.ensure_initialized().await;

        let messages = self.context.conversation_messages();
        if messages.is_empty() {
            return Ok(None);
        }

        let messages_compacted = messages.len();
        let mut handoff = self.generate_handoff_document(focus).await?;
        // The file index is built mechanically from the conversation: prompting
        // the model for it is unreliable, and missing entries directly cause
        // post-compaction re-reads.
        handoff.file_index = crate::handoff::build_file_index(&messages);
        let summary = handoff.render_markdown();

        if let Some(transcript) = &self.transcript {
            transcript.write_compact_marker().await;
        }

        self.context.clear_recent();
        self.context.clear_reminders();
        self.context.set_handoff_document(&handoff);

        Ok(Some(CompactResult {
            messages_compacted,
            messages_remaining: 0,
            summary,
        }))
    }

    /// Generates a structured handoff document using the current tier model.
    async fn generate_handoff_document(
        &self,
        focus: Option<&str>,
    ) -> Result<HandoffDocument, AgentError> {
        let focus_instruction = focus.map(str::trim).filter(|s| !s.is_empty()).map_or_else(
            || "No additional focus hint was provided.".to_string(),
            |f| {
                let mut instruction = String::with_capacity(f.len() + 22);
                instruction.push_str("Focus the handoff on: ");
                instruction.push_str(f);
                instruction
            },
        );
        let handoff_prompt = include_str!("prompts/compact_handoff.txt")
            .replace(FOCUS_INSTRUCTION_PLACEHOLDER, &focus_instruction);

        let mut messages = self.context.conversation_messages();
        messages.push(Message::user(handoff_prompt));

        let mut chunks = Vec::new();
        match self.tier {
            ModelTier::Advanced => {
                let stream = self.advanced.respond(LLMRequest::new(messages.clone()));
                futures_lite::pin!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(Event::Text(text)) => chunks.push(text),
                        Ok(Event::BuiltInToolResult { tool, result }) => {
                            chunks.push(format_builtin_tool_result(&tool, &result));
                        }
                        Ok(_) => {}
                        Err(e) => return Err(AgentError::Llm(e.to_string())),
                    }
                }
            }
            ModelTier::Balanced => {
                let stream = self.balanced.respond(LLMRequest::new(messages.clone()));
                futures_lite::pin!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(Event::Text(text)) => chunks.push(text),
                        Ok(Event::BuiltInToolResult { tool, result }) => {
                            chunks.push(format_builtin_tool_result(&tool, &result));
                        }
                        Ok(_) => {}
                        Err(e) => return Err(AgentError::Llm(e.to_string())),
                    }
                }
            }
            ModelTier::Fast => {
                let stream = self.fast.respond(LLMRequest::new(messages));
                futures_lite::pin!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(Event::Text(text)) => chunks.push(text),
                        Ok(Event::BuiltInToolResult { tool, result }) => {
                            chunks.push(format_builtin_tool_result(&tool, &result));
                        }
                        Ok(_) => {}
                        Err(e) => return Err(AgentError::Llm(e.to_string())),
                    }
                }
            }
        }

        let response = chunks.join("").trim().to_string();
        if response.is_empty() {
            return Err(AgentError::Llm(
                "Compaction failed to generate handoff summary".to_string(),
            ));
        }
        serde_json::from_str::<HandoffDocument>(&response)
            .map_err(|error| AgentError::Llm(error.to_string()))?
            .validate()
            .map_err(AgentError::Llm)
    }

    /// Injects a continuation reminder when working documents still have pending tasks.
    async fn inject_working_doc_continue_reminder(&mut self) -> bool {
        let Some(sandbox_dir) = self.sandbox_dir.as_deref() else {
            return false;
        };

        let docs = working_docs::read_snapshot(sandbox_dir).await;
        if !docs.has_unchecked_items() {
            return false;
        }

        self.context.insert_reminder(&SystemReminder {
            content: "tasks.md still has unchecked items. Continue working through the checklist. If user input is required, call ask_user and then proceed.".to_string(),
        });
        true
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
        self.ensure_initialized().await;
        let context_window = self.snapshot_context_window().await;
        let todo_items = self
            .todo_list
            .as_ref()
            .map(super::todo::TodoList::items)
            .unwrap_or_default();
        Ok(AgentCheckpoint {
            context: self.context.checkpoint(),
            todo_items,
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
        if !checkpoint.todo_items.is_empty() {
            let list = self.todo_list.get_or_insert_with(TodoList::new);
            list.write(checkpoint.todo_items);
        } else if let Some(list) = &self.todo_list {
            list.clear();
        }
        #[cfg(feature = "skills")]
        {
            self.active_skills.clear();
            self.active_allowed_tools = None;
        }
        Ok(())
    }

    /// Returns a structured snapshot of the currently assembled context window.
    pub async fn snapshot_context_window(&mut self) -> ContextWindowSnapshot {
        self.ensure_initialized().await;
        self.assemble_context_window().await
    }

    /// Returns the model profile if available.
    #[must_use]
    pub const fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    /// Ensures the agent is initialized (profiles fetched, static blocks set up).
    async fn ensure_initialized(&mut self) {
        if self.initialized {
            return;
        }

        // Fetch profile for the selected tier (for context window decisions)
        self.profile = Some(match self.tier {
            ModelTier::Advanced => self.advanced.profile().await,
            ModelTier::Balanced => self.balanced.profile().await,
            ModelTier::Fast => self.fast.profile().await,
        });

        // Fetch fast model profile (for compression decisions)
        // We always need this because compression uses the fast model
        self.fast_profile = Some(self.fast.profile().await);

        // Populate static system blocks in Context from AgentConfig.
        // These form the stable, cacheable prefix.
        self.populate_system_blocks();

        self.initialized = true;
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

        self.context.insert_system_named(
            "knowledge_and_time",
            include_str!("prompts/knowledge_and_time.txt"),
        );

        self.context
            .insert_system_named("permissions", include_str!("prompts/permissions.txt"));

        let tool_hints = self.format_tool_hints_block();
        if !tool_hints.is_empty() {
            self.context.insert_system_named("tool_hints", &tool_hints);
        }
    }

    async fn populate_dynamic_reminders(&mut self) {
        self.context.clear_reminders();

        #[cfg(feature = "skills")]
        self.populate_skill_reminders();

        if let Some(todo_ctx) = self.format_todo_context() {
            self.context
                .insert_reminder(&SystemReminder { content: todo_ctx });
        }

        if let Some(sandbox_dir) = self.sandbox_dir.as_deref() {
            let docs = working_docs::read_snapshot(sandbox_dir).await;
            if let Some(previous) = self.last_working_docs.as_ref()
                && let Some(diff) = working_docs::tasks_md_diff(previous, &docs)
            {
                self.context.insert_reminder(&TasksDiffReminder { diff });
            }
            self.last_working_docs = Some(docs.clone());
            if let Some(tasks_md) = docs.tasks_md {
                self.context.insert_reminder(&Tasks { content: tasks_md });
            }
        }

        if let Some(job_registry) = &self.job_registry {
            let running = job_registry.format_running_jobs().await;
            if !running.is_empty() {
                self.context.insert_reminder(&SystemReminder {
                    content: format!(
                        "Running background terminals:\n{running}Use terminal_read for incremental output, read redirected output files via terminal commands like head/tail/grep/cat when needed, terminal_input for stdin, and terminal_kill to stop tasks."
                    ),
                });
            }
        }
    }

    /// Builds the message list for an LLM request.
    ///
    /// Uses `context.build_messages()` for the stable system prefix + conversation,
    /// then prepends per-turn ephemeral context (todo, working docs, background
    /// jobs, context usage) as system messages inserted before conversation messages.
    async fn build_request_messages(&mut self) -> Vec<Message> {
        self.assemble_context_window().await.messages
    }

    async fn build_live_request_messages(&mut self) -> Vec<Message> {
        let _ = self.maybe_reassemble_after_idle_gap();
        let messages = self.build_request_messages().await;
        self.last_request_started_at = Some(Instant::now());
        messages
    }

    #[cfg(feature = "skills")]
    pub(crate) fn activate_skills_for_prompt(&mut self, _prompt: &str) -> Vec<AgentEvent> {
        let _ = self.skill_registry.as_ref();
        self.active_skills.clear();
        self.active_allowed_tools = None;
        Vec::new()
    }

    #[cfg(feature = "skills")]
    fn populate_skill_reminders(&mut self) {
        for skill in &self.active_skills {
            let mut resource_paths = skill.resources.keys().cloned().collect::<Vec<_>>();
            resource_paths.sort();
            let payload = serialize_xml(
                "skill_instruction",
                &SkillInstructionXml {
                    name: skill.name.clone(),
                    description: skill.description.clone(),
                    instructions: skill.instructions.clone(),
                    allowed_tools: skill.allowed_tools.clone(),
                    resource_paths: (!resource_paths.is_empty()).then_some(resource_paths),
                },
            );
            self.context
                .insert_reminder(&SystemReminder { content: payload });
        }
    }

    async fn assemble_context_window(&mut self) -> ContextWindowSnapshot {
        self.populate_dynamic_reminders().await;

        let mut messages = self
            .context
            .build_messages_with_transient_system(&self.transient_system_messages);
        let mut metrics = self.estimate_context_window_metrics(&messages);

        if !metrics.has_handoff
            && metrics.usage_fraction >= self.config.context_assembler.handoff_threshold
            && let Some(handoff_ctx) = self.format_handoff_context(metrics.usage_fraction)
        {
            self.context.insert_reminder(&SystemReminder {
                content: handoff_ctx,
            });
            messages = self.context.build_messages();
            metrics = self.estimate_context_window_metrics(&messages);
        }

        ContextWindowSnapshot {
            phase: self.classify_context_window_phase(&metrics),
            metrics,
            messages,
        }
    }

    fn maybe_reassemble_after_idle_gap(&mut self) -> crate::context::ReassemblyReport {
        let unchanged = crate::context::ReassemblyReport::default();
        if self.context.handoff().is_some() {
            return unchanged;
        }
        let Some(last_request_started_at) = self.last_request_started_at else {
            return unchanged;
        };
        if last_request_started_at.elapsed() < self.config.context_assembler.idle_reassemble_after {
            return unchanged;
        }
        self.reassemble_context()
    }

    fn estimate_context_window_metrics(&self, messages: &[Message]) -> ContextWindowMetrics {
        ContextWindowMetrics {
            usage_fraction: estimate_context_usage(messages, self.effective_context_window()),
            selected_model_context_window: self
                .profile
                .as_ref()
                .map(|profile| profile.context_length),
            fast_model_context_window: self
                .fast_profile
                .as_ref()
                .map(|profile| profile.context_length),
            effective_context_window: self.effective_context_window(),
            system_block_count: self.context.system_block_count(),
            reminder_count: self.context.reminders().len(),
            recent_message_count: self.context.len_recent(),
            recent_system_message_count: self.context.count_recent_system_messages(),
            has_handoff: self.context.handoff().is_some(),
        }
    }

    fn classify_context_window_phase(&self, metrics: &ContextWindowMetrics) -> ContextWindowPhase {
        if metrics.has_handoff {
            return ContextWindowPhase::HandoffActive;
        }

        if metrics.usage_fraction >= self.config.context_assembler.handoff_threshold {
            return ContextWindowPhase::HandoffDue;
        }

        match &self.config.context {
            ContextStrategy::Smart(config)
                if metrics.usage_fraction >= config.effective_trigger()
                    && metrics.recent_system_message_count > 0 =>
            {
                ContextWindowPhase::ReassemblyDue
            }
            ContextStrategy::Smart(config)
                if metrics.usage_fraction >= config.effective_trigger() =>
            {
                ContextWindowPhase::CompressionDue
            }
            ContextStrategy::Unlimited | ContextStrategy::Smart(_) => ContextWindowPhase::Stable,
        }
    }

    fn effective_context_window(&self) -> usize {
        let tier_context = self
            .profile
            .as_ref()
            .map_or(100_000, |profile| profile.context_length as usize);
        let fast_context = self
            .fast_profile
            .as_ref()
            .map_or(100_000, |profile| profile.context_length as usize);
        tier_context.min(fast_context)
    }

    /// Returns the exact request message sequence that would be sent for the next turn.
    ///
    /// This is intended for observability/debug UIs and does not mutate agent memory.
    /// Temporarily adds the prompt to a forked context, builds the messages, then discards.
    pub async fn preview_request_messages(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> Vec<Message> {
        self.ensure_initialized().await;
        // Fork context, add the user message, build messages
        let checkpoint = self.context.checkpoint();
        self.context
            .push(Message::user(prompt).with_attachments(attachments));
        let messages = self.build_request_messages().await;
        self.context.restore(checkpoint);
        messages
    }

    /// Injects a handoff instruction when context usage approaches the threshold.
    fn format_handoff_context(&self, usage: f32) -> Option<String> {
        if usage < self.config.context_assembler.handoff_threshold {
            return None;
        }

        let mut note = String::new();
        note.push_str(&self.config.context_assembler.handoff_instruction);
        if let Some(path) = &self.config.transcript_path {
            note.push_str(" Transcript source: ");
            note.push_str(path);
            note.push('.');
        }
        Some(note)
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

    /// Compresses context if needed.
    ///
    /// Considers BOTH the selected tier's context window AND the fast model's
    /// context window. Compression is triggered based on the most constrained
    /// of the two windows, ensuring the fast LLM can actually see the content
    /// during compaction.
    ///
    /// Uses `effective_trigger()` which accounts for a 20% context reservation
    /// to ensure there's room for the fast LLM during compaction.
    ///
    async fn maybe_compress(&mut self) -> Result<(), AgentError> {
        self.ensure_initialized().await;
        let snapshot = self.assemble_context_window().await;
        let context_strategy = self.config.context.clone();

        match context_strategy {
            ContextStrategy::Unlimited => Ok(()),
            ContextStrategy::Smart(config) => {
                if !matches!(
                    snapshot.phase,
                    ContextWindowPhase::ReassemblyDue | ContextWindowPhase::CompressionDue
                ) {
                    return Ok(());
                }

                // Reassembly alone is preferred only when it frees enough of
                // the window to leave the danger zone; a marginal saving is
                // not worth the cache invalidation on top of a compaction
                // that would follow anyway.
                let window = self.effective_context_window();
                let savings_tokens = self.context.estimate_reassembly_savings() / 4;
                let savings_fraction =
                    savings_tokens.to_f32().unwrap_or(0.0) / window.to_f32().unwrap_or(f32::MAX);
                let usage_after_reassembly = snapshot.metrics.usage_fraction - savings_fraction;

                if usage_after_reassembly < config.effective_trigger() {
                    let report = self.reassemble_context();
                    tracing::debug!(
                        reminders = report.reminders_cleared,
                        system_messages = report.system_messages_pruned,
                        deduped = report.tool_results_deduped,
                        bytes_freed = report.bytes_freed,
                        "context reassembly freed enough window; compaction skipped"
                    );
                    return Ok(());
                }

                if self.context.len_recent() > config.preserve_recent {
                    // Compress directly; if compaction itself fails (for
                    // example the compaction request overflows the window),
                    // fall back to reassembly and retry once.
                    if let Err(error) = self.compact(None).await {
                        let report = self.reassemble_context();
                        if !report.changed() {
                            return Err(error);
                        }
                        tracing::warn!(
                            %error,
                            bytes_freed = report.bytes_freed,
                            "compaction failed; retrying after context reassembly"
                        );
                        self.compact(None).await?;
                    }
                } else {
                    let _ = self.reassemble_context();
                }
                Ok(())
            }
        }
    }

    fn reassemble_context(&mut self) -> crate::context::ReassemblyReport {
        self.context.reassemble()
    }

    async fn emit_checkpoint(
        &mut self,
        reason: CheckpointReason,
        assistant_text: &str,
        turn: usize,
    ) -> Result<EmittedCheckpoint, AgentError> {
        let context = self.context.checkpoint();
        let todo_items = self
            .todo_list
            .as_ref()
            .map(super::todo::TodoList::items)
            .unwrap_or_default();
        let tool_surface_hash = self.tool_surface_hash();
        let window = self.assemble_context_window().await;
        let checkpoint_ctx = CheckpointContext {
            reason,
            assistant_text,
            turn,
            message_count: self.context.len_recent(),
            context: &context,
            todo_items: &todo_items,
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

    /// Formats the todo list as a system reminder.
    ///
    /// Returns None if there's no todo list or it's empty.
    fn format_todo_reminder(&self) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();
        if items.is_empty() {
            return None;
        }
        let items_json = format_todo_items_json(&items);
        TodoReminderTemplate {
            items_json: &items_json,
        }
        .render()
        .ok()
    }

    /// Formats the current todo list for context injection before each request.
    fn format_todo_context(&self) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();
        if items.is_empty() {
            return None;
        }

        let items_json = format_todo_items_json(&items);
        TodoContextTemplate {
            items_json: &items_json,
        }
        .render()
        .ok()
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

    /// Generates a reminder about the next task after a task was completed.
    fn format_next_task_reminder(&self, completed_task: &str) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();

        // Find the next pending or in_progress task
        let next_task = items
            .iter()
            .find(|item| matches!(item.status, TodoStatus::Pending | TodoStatus::InProgress));

        next_task.map_or_else(
            || {
                items
                    .iter()
                    .all(|i| i.status == TodoStatus::Completed)
                    .then(|| {
                        AllTasksCompleteReminderTemplate { completed_task }
                            .render()
                            .ok()
                    })
                    .flatten()
            },
            |task| {
                NextTaskReminderTemplate {
                    completed_task,
                    next_task: &task.content,
                    active_form: &task.active_form,
                }
                .render()
                .ok()
            },
        )
    }
}

/// Formats todo items into the JSON-ish list used in system reminders.
fn format_todo_items_json(items: &[TodoItem]) -> String {
    serde_json::to_string(items).unwrap_or_else(|_| "[]".to_string())
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

    use crate::compression::{CompressionLevel, PreserveConfig, SmartCompressionConfig};
    use crate::config::ContextAssemblerConfig;
    #[cfg(feature = "skills")]
    use crate::{AgentCheckpoint, ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot};
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

        fn profile(&self) -> impl std::future::Future<Output = ModelProfile> + Send {
            std::future::ready(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                self.context_length,
            ))
        }
    }

    #[tokio::test]
    async fn snapshot_phase_accounts_for_system_blocks() {
        let config = AgentConfig {
            system_prompt: Some("x".repeat(120)),
            context: ContextStrategy::Smart(SmartCompressionConfig {
                trigger_threshold: 0.4,
                emergency_threshold: 0.9,
                preserve_recent: 0,
                preserve: PreserveConfig::default(),
                level: CompressionLevel::Standard,
            }),
            context_assembler: ContextAssemblerConfig {
                handoff_threshold: 10.0,
                ..ContextAssemblerConfig::default()
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

        let snapshot = agent.snapshot_context_window().await;

        assert_eq!(snapshot.phase, ContextWindowPhase::CompressionDue);
        assert!(snapshot.metrics.system_block_count > 0);
        assert!(snapshot.metrics.usage_fraction >= 0.4);
        assert_eq!(snapshot.metrics.recent_message_count, 1);
    }

    #[tokio::test]
    async fn snapshot_phase_reports_handoff_active() {
        let mut agent = Agent::new(MockLlm {
            context_length: 1_000,
        });
        agent.context_mut().set_handoff("resume here");

        let snapshot = agent.snapshot_context_window().await;

        assert_eq!(snapshot.phase, ContextWindowPhase::HandoffActive);
        assert!(snapshot.metrics.has_handoff);
    }

    #[tokio::test]
    async fn handoff_due_does_not_trigger_automatic_compaction() {
        let config = AgentConfig {
            system_prompt: Some("x".repeat(120)),
            context: ContextStrategy::Smart(SmartCompressionConfig {
                trigger_threshold: 0.2,
                emergency_threshold: 0.9,
                preserve_recent: 0,
                preserve: PreserveConfig::default(),
                level: CompressionLevel::Standard,
            }),
            context_assembler: ContextAssemblerConfig {
                handoff_threshold: 0.1,
                ..ContextAssemblerConfig::default()
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

        let before_recent = agent.context().len_recent();
        agent
            .maybe_compress()
            .await
            .unwrap_or_else(|error| panic!("handoff_due must not compact automatically: {error}"));

        assert_eq!(agent.context().len_recent(), before_recent);
        assert!(agent.context().handoff().is_none());
    }

    #[tokio::test]
    async fn reassembly_due_prunes_recent_system_messages_before_compaction() {
        let config = AgentConfig {
            context: ContextStrategy::Smart(SmartCompressionConfig {
                trigger_threshold: 0.05,
                emergency_threshold: 0.9,
                preserve_recent: 100,
                preserve: PreserveConfig::default(),
                level: CompressionLevel::Standard,
            }),
            context_assembler: ContextAssemblerConfig {
                handoff_threshold: 10.0,
                ..ContextAssemblerConfig::default()
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
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));

        let snapshot = agent.snapshot_context_window().await;
        assert_eq!(snapshot.phase, ContextWindowPhase::ReassemblyDue);

        agent
            .maybe_compress()
            .await
            .unwrap_or_else(|error| panic!("reassembly_due must not fail: {error}"));

        assert_eq!(agent.context().count_recent_system_messages(), 0);
        assert_eq!(agent.context().len_recent(), 1);
        assert_eq!(agent.context().recent()[0].content(), "hello");
    }

    #[tokio::test]
    async fn idle_gap_reassembles_before_next_live_request() {
        let mut config = AgentConfig::default();
        config.context_assembler.idle_reassemble_after = Duration::from_secs(1);

        let mut agent = Agent::with_config(
            MockLlm {
                context_length: 1_000,
            },
            config,
        );
        agent.push_message(Message::user("hello"));
        agent.push_message(Message::system(
            "<system-reminder>ephemeral</system-reminder>",
        ));
        agent.last_request_started_at = Instant::now().checked_sub(Duration::from_secs(5));

        let _ = agent.build_live_request_messages().await;

        assert_eq!(agent.context().count_recent_system_messages(), 0);
        assert_eq!(agent.context().len_recent(), 1);
        assert_eq!(agent.context().recent()[0].content(), "hello");
        assert!(agent.last_request_started_at.is_some());
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
            todo_items: Vec::new(),
            tool_surface_hash: "tool-surface".to_string(),
            context_window: ContextWindowSnapshot {
                phase: ContextWindowPhase::Stable,
                metrics: ContextWindowMetrics {
                    usage_fraction: 0.0,
                    selected_model_context_window: None,
                    fast_model_context_window: None,
                    effective_context_window: 0,
                    system_block_count: 0,
                    reminder_count: 0,
                    recent_message_count: 0,
                    recent_system_message_count: 0,
                    has_handoff: false,
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
    async fn build_live_request_messages_omit_skill_resource_catalog() {
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

        let messages = agent.build_live_request_messages().await;
        assert!(
            messages
                .iter()
                .all(|message| !message.content().contains("templates/review.md")),
            "skill resource catalog must not be injected automatically"
        );
    }
}
