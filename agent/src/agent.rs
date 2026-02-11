//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages conversation memory, applies context compression, and
//! handles tool execution in an agent-controlled loop.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, Message, model::Profile as ModelProfile},
};
use futures_core::Stream;
use futures_lite::StreamExt;

use crate::{
    compression::{ContextStrategy, estimate_context_usage},
    config::{AgentConfig, AgentKind, ContextBlock},
    context::ConversationMemory,
    error::AgentError,
    event::AgentEvent,
    hook::{
        Hook, PostToolAction, PreToolAction, StopContext, StopReason, ToolResultContext,
        ToolUseContext,
    },
    todo::{TodoItem, TodoList, TodoStatus},
    tools::AgentTools,
    transcript::Transcript,
    working_docs,
};

use aither_sandbox::{BackgroundTaskReceiver, OutputStore};
use std::sync::Arc;

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

/// Which model tier to use for the agent's main reasoning loop.
///
/// This allows creating agents that use different capability levels:
/// - Main agent: typically uses `Advanced`
/// - Explore subagent: uses `Balanced` (cheaper, still capable)
/// - Quick tasks: uses `Fast`
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelTier {
    /// Use the most capable model (default for main agent).
    #[default]
    Advanced,
    /// Use the balanced model (good for subagents).
    Balanced,
    /// Use the fast model (for quick, simple tasks).
    Fast,
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

    /// Conversation memory.
    pub(crate) memory: ConversationMemory,

    /// Cached model profile (for the selected tier).
    pub(crate) profile: Option<ModelProfile>,

    /// Cached fast model profile (for compression decisions).
    pub(crate) fast_profile: Option<ModelProfile>,

    /// Whether tools have been bootstrapped.
    pub(crate) initialized: bool,

    /// Todo list for tracking long tasks.
    pub(crate) todo_list: Option<TodoList>,

    /// Output store for lazy URL allocation during compression.
    pub(crate) output_store: Option<Arc<OutputStore>>,

    /// Receiver for completed background bash tasks.
    pub(crate) background_receiver: Option<BackgroundTaskReceiver>,

    /// Optional readable transcript for long-context recovery.
    pub(crate) transcript: Option<Transcript>,

    /// Optional sandbox directory for working-doc supervision (TODO.md/PLAN.md).
    pub(crate) sandbox_dir: Option<PathBuf>,
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
            memory: ConversationMemory::default(),
            profile: None,
            fast_profile: None,
            initialized: false,
            todo_list: None,
            output_store: None,
            background_receiver: None,
            transcript: None,
            sandbox_dir: None,
        }
    }
}

impl<LLM: LanguageModel + Clone> Agent<LLM, LLM, LLM, ()> {
    /// Returns a builder for more complex agent construction.
    #[must_use]
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
    #[must_use]
    pub fn run(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = url::Url>,
    ) -> impl Stream<Item = Result<AgentEvent, AgentError>> + '_ {
        let prompt = prompt.to_string();
        let attachments: Vec<url::Url> = attachments.into_iter().collect();

        async_stream::try_stream! {
            self.ensure_initialized().await;

            // Apply context compression if needed
            self.maybe_compress().await?;

            // Add user message with attachments
            let user_msg = Message::user(&prompt).with_attachments(attachments);
            self.memory.push(user_msg);
            if let Some(transcript) = &self.transcript {
                transcript.write_user_message(&prompt).await;
            }

            // Run the tool loop
            let mut iteration = 0;
            let mut all_text_chunks: Vec<String> = Vec::new();

            let final_text = loop {
                iteration += 1;
                if iteration > self.config.max_iterations {
                    Err(AgentError::MaxIterations {
                        limit: self.config.max_iterations,
                    })?;
                }

                // Build messages
                let messages = self.build_request_messages().await;

                // Create request with tool definitions
                let tool_defs = self.tools.active_definitions();
                let request = LLMRequest::new(messages).with_tool_definitions(tool_defs);

                // Stream the response and yield text events as they arrive
                let mut text_chunks: Vec<String> = Vec::new();
                let mut tool_calls = Vec::new();
                let mut malformed_function_call = false;
                let mut error: Option<String> = None;

                // Process stream based on tier
                match self.tier {
                    ModelTier::Advanced => {
                        let stream = self.advanced.respond(request);
                        futures_lite::pin!(stream);

                        while let Some(event) = stream.next().await {
                            match event {
                                Ok(Event::Text(text)) => {
                                    self.hooks.on_text(&text).await;
                                    // Yield text event for streaming display
                                    yield AgentEvent::Text(text.clone());
                                    text_chunks.push(text);
                                }
                                Ok(Event::Reasoning(r)) => {
                                    yield AgentEvent::Reasoning(r);
                                }
                                Ok(Event::ToolCall(call)) => tool_calls.push(call),
                                Ok(Event::BuiltInToolResult { tool, result }) => {
                                    let formatted = format!("[{tool}] {result}");
                                    yield AgentEvent::Text(formatted.clone());
                                    text_chunks.push(formatted);
                                }
                                Ok(Event::Usage(u)) => {
                                    yield AgentEvent::Usage(u);
                                }
                                Err(e) => {
                                    let error_msg = e.to_string();
                                    if error_msg.contains("malformed function call") {
                                        tracing::warn!("Model generated malformed function call, retrying...");
                                        malformed_function_call = true;
                                        break;
                                    }
                                    error = Some(error_msg);
                                    break;
                                }
                            }
                        }
                    }
                    ModelTier::Balanced => {
                        let stream = self.balanced.respond(request);
                        futures_lite::pin!(stream);

                        while let Some(event) = stream.next().await {
                            match event {
                                Ok(Event::Text(text)) => {
                                    self.hooks.on_text(&text).await;
                                    yield AgentEvent::Text(text.clone());
                                    text_chunks.push(text);
                                }
                                Ok(Event::Reasoning(r)) => {
                                    yield AgentEvent::Reasoning(r);
                                }
                                Ok(Event::ToolCall(call)) => tool_calls.push(call),
                                Ok(Event::BuiltInToolResult { tool, result }) => {
                                    let formatted = format!("[{tool}] {result}");
                                    yield AgentEvent::Text(formatted.clone());
                                    text_chunks.push(formatted);
                                }
                                Ok(Event::Usage(u)) => {
                                    yield AgentEvent::Usage(u);
                                }
                                Err(e) => {
                                    let error_msg = e.to_string();
                                    if error_msg.contains("malformed function call") {
                                        tracing::warn!("Model generated malformed function call, retrying...");
                                        malformed_function_call = true;
                                        break;
                                    }
                                    error = Some(error_msg);
                                    break;
                                }
                            }
                        }
                    }
                    ModelTier::Fast => {
                        let stream = self.fast.respond(request);
                        futures_lite::pin!(stream);

                        while let Some(event) = stream.next().await {
                            match event {
                                Ok(Event::Text(text)) => {
                                    self.hooks.on_text(&text).await;
                                    yield AgentEvent::Text(text.clone());
                                    text_chunks.push(text);
                                }
                                Ok(Event::Reasoning(r)) => {
                                    yield AgentEvent::Reasoning(r);
                                }
                                Ok(Event::ToolCall(call)) => tool_calls.push(call),
                                Ok(Event::BuiltInToolResult { tool, result }) => {
                                    let formatted = format!("[{tool}] {result}");
                                    yield AgentEvent::Text(formatted.clone());
                                    text_chunks.push(formatted);
                                }
                                Ok(Event::Usage(u)) => {
                                    yield AgentEvent::Usage(u);
                                }
                                Err(e) => {
                                    let error_msg = e.to_string();
                                    if error_msg.contains("malformed function call") {
                                        tracing::warn!("Model generated malformed function call, retrying...");
                                        malformed_function_call = true;
                                        break;
                                    }
                                    error = Some(error_msg);
                                    break;
                                }
                            }
                        }
                    }
                }

                if let Some(e) = error {
                    Err(AgentError::Llm(e))?;
                }

                // If malformed function call, retry this iteration
                if malformed_function_call {
                    continue;
                }

                let response_text = text_chunks.join("");
                all_text_chunks.extend(text_chunks);

                // If no tool calls, we're done unless working-doc supervision requires continuation.
                if tool_calls.is_empty() {
                    if !response_text.is_empty() {
                        self.memory.push(Message::assistant(&response_text));
                        if let Some(transcript) = &self.transcript {
                            transcript.write_assistant_text(&response_text).await;
                        }
                    }
                    if self.inject_working_doc_continue_reminder().await {
                        continue;
                    }
                    break response_text;
                }

                // Store assistant response with tool calls in memory
                self.memory.push(Message::assistant_with_tool_calls(
                    &response_text,
                    tool_calls.clone(),
                ));

                // Snapshot todo state BEFORE executing tool calls
                let old_todo_items: Vec<TodoItem> = self
                    .todo_list
                    .as_ref()
                    .map(|l| l.items())
                    .unwrap_or_default();

                // Track tool names for later todo detection
                let tool_names: Vec<_> = tool_calls.iter().map(|c| c.name.clone()).collect();

                // Yield tool call start events
                for call in &tool_calls {
                    let args = call.arguments.to_string();
                    if let Some(transcript) = &self.transcript {
                        transcript.write_tool_call(&call.name, &args).await;
                    }
                    yield AgentEvent::ToolCallStart {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: args,
                    };
                }

                // Execute tool calls in parallel
                let tools = &self.tools;
                let hooks = &self.hooks;
                let tool_futures = tool_calls.iter().map(|call| {
                    let args_json = call.arguments.to_string();
                    let message_count = self.memory.len();

                    async move {
                        let tool_ctx = ToolUseContext {
                            tool_name: &call.name,
                            arguments: &args_json,
                            turn: iteration,
                            message_count,
                        };

                        let (result, duration) = match hooks.pre_tool_use(&tool_ctx).await {
                            PreToolAction::Abort(reason) => {
                                return Err(AgentError::HookRejected {
                                    hook: "pre_tool_use",
                                    reason,
                                });
                            }
                            PreToolAction::Deny(reason) => {
                                (Err(anyhow::anyhow!(reason)), Duration::ZERO)
                            }
                            PreToolAction::Allow => {
                                let start = Instant::now();
                                let result = tools.call(&call.name, &args_json).await;
                                let result = result.map(|output| output.as_str().unwrap_or("").to_string());
                                (result, start.elapsed())
                            }
                        };

                        let result_str = match &result {
                            Ok(s) => s.clone(),
                            Err(e) => format!("Error: {e}"),
                        };
                        let result_ref = result
                            .as_ref()
                            .map(|s| s.as_str())
                            .map_err(|e| e.to_string());
                        let result_ctx = ToolResultContext {
                            tool_name: &call.name,
                            arguments: &args_json,
                            result: result_ref.as_ref().map(|s| *s).map_err(|s| s.as_str()),
                            duration,
                        };

                        let tool_content = match hooks.post_tool_use(&result_ctx).await {
                            PostToolAction::Abort(reason) => {
                                return Err(AgentError::HookRejected {
                                    hook: "post_tool_use",
                                    reason,
                                });
                            }
                            PostToolAction::Replace(replacement) => replacement,
                            PostToolAction::Keep => result_str,
                        };

                        Ok((call.id.clone(), call.name.clone(), tool_content))
                    }
                });

                // Wait for all tool calls to complete
                let results: Vec<Result<(String, String, String), AgentError>> =
                    futures::future::join_all(tool_futures).await;

                // Check if todo tool was called
                let todo_tool_called = tool_names.iter().any(|name| name == "todo");

                // Add results to memory and yield tool end events
                let mut has_tool_error = false;
                for result in results {
                    let (call_id, call_name, content) = result?;

                    if let Some(transcript) = &self.transcript {
                        transcript.write_tool_result(&call_name, &Ok(content.clone())).await;
                    }

                    // Yield tool call end event
                    yield AgentEvent::ToolCallEnd {
                        id: call_id.clone(),
                        name: call_name,
                        result: Ok(content.clone()),
                    };

                    if content.contains("unknown shell_id")
                        || content.contains("shell_id is required")
                        || content.contains("not found")
                        || content.contains("Invalid arguments")
                    {
                        has_tool_error = true;
                    }
                    let processed_content = self.process_reload_marker(&content);
                    self.memory.push(Message::tool(&call_id, processed_content));
                }

                // If there was a tool error, inject a reminder
                if has_tool_error {
                    let reminder = concat!(
                        "<system-reminder>\n",
                        "IMPORTANT: You can ONLY act through tool calls: `open_shell`, `bash`, `close_shell`.\n",
                        "All commands (websearch, webfetch, ask_user, todo, task, etc.) are CLI commands that MUST be executed via `bash` tool calls.\n",
                        "You cannot execute commands by writing them in text -- always use tool calls.\n\n",
                        "Correct usage:\n",
                        "- open_shell first, then call bash with shell_id + timeout + script\n",
                        "- Example: bash with script `websearch \"query\"` or `ask_user \"question\" --options A --options B`\n",
                        "- IPC commands (websearch, webfetch, ask_user) work in sandboxed mode and only on local shells.\n",
                        "</system-reminder>"
                    );
                    self.memory.push(Message::system(reminder));
                }

                // If todo tool was called, inject updated todo list
                if todo_tool_called {
                    let new_items = self
                        .todo_list
                        .as_ref()
                        .map(|l| l.items())
                        .unwrap_or_default();

                    let newly_completed: Vec<_> = new_items
                        .iter()
                        .filter(|new_item| {
                            new_item.status == TodoStatus::Completed
                                && old_todo_items.iter().any(|old| {
                                    old.content == new_item.content
                                        && old.status != TodoStatus::Completed
                                })
                        })
                        .collect();

                    if let Some(completed) = newly_completed.first() {
                        if let Some(reminder) = self.format_next_task_reminder(&completed.content) {
                            self.memory.push(Message::system(&reminder));
                        }
                    } else if let Some(reminder) = self.format_todo_reminder() {
                        self.memory.push(Message::system(&reminder));
                    }
                }

                // Poll for completed background tasks
                if let Some(ref receiver) = self.background_receiver {
                    let completed_tasks = receiver.take_completed();
                    for task in completed_tasks {
                        tracing::info!(task_id = %task.task_id, "background task completed");
                        let result_msg = self.format_background_task_result(&task);
                        self.memory.push(Message::system(&result_msg));
                    }
                }
            };

            // Handle background tasks before completing
            if let Some(ref receiver) = self.background_receiver {
                let completed_tasks = receiver.take_completed();
                let mut had_completed = !completed_tasks.is_empty();
                for task in completed_tasks {
                    tracing::info!(task_id = %task.task_id, "background task completed (final check)");
                    let result_msg = self.format_background_task_result(&task);
                    self.memory.push(Message::system(&result_msg));
                }

                const MAX_WAIT: Duration = Duration::from_secs(300);
                const POLL_INTERVAL: Duration = Duration::from_millis(100);
                let start = Instant::now();

                while start.elapsed() < MAX_WAIT {
                    if let Some(task) = receiver.recv_timeout(POLL_INTERVAL).await {
                        tracing::info!(task_id = %task.task_id, "background task completed (waiting)");
                        let result_msg = self.format_background_task_result(&task);
                        self.memory.push(Message::system(&result_msg));
                        had_completed = true;
                    } else {
                        let has_running = self
                            .background_receiver
                            .as_ref()
                            .is_some_and(|receiver| receiver.has_running());
                        if !has_running {
                            break;
                        }
                    }
                }

                if had_completed {
                    // Continue processing with background results
                    let continuation = self.continue_after_background_streaming().await;
                    for event in continuation {
                        yield event?;
                    }
                    return;
                }
            }

            // Notify hooks
            let stop_ctx = StopContext {
                final_text: &final_text,
                turns: iteration,
                reason: StopReason::Complete,
            };

            if let Some(reason) = self.hooks.on_stop(&stop_ctx).await {
                Err(AgentError::HookRejected {
                    hook: "on_stop",
                    reason,
                })?;
            }

            // Yield completion event
            yield AgentEvent::Complete {
                final_text,
                turns: iteration,
            };
        }
    }

    /// Registers a tool for the agent to use.
    pub fn register_tool<T: aither_core::llm::Tool + 'static>(&mut self, tool: T) {
        self.tools.register(tool);
    }

    /// Registers a tool (alias for register_tool).
    /// Adds a message to the conversation history.
    pub fn push_message(&mut self, message: Message) {
        self.memory.push(message);
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.memory.clear();
    }

    /// Force-closes active shell sessions and their running jobs.
    pub async fn close_shell_sessions(&self) {
        self.tools.close_shell_sessions().await;
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

        let messages = self.memory.all();
        if messages.is_empty() {
            return Ok(None);
        }

        let messages_compacted = messages.len();
        let summary = self.generate_handoff_summary(focus).await?;

        if let Some(transcript) = &self.transcript {
            transcript.write_compact_marker().await;
        }

        self.memory.clear();
        self.memory.push_summary(Message::system(summary.clone()));
        self.memory.push(Message::system(
            "Session continues from compacted context. Continue without asking the user to repeat details. Recover missing details from files, TODO.md/PLAN.md, or transcript when needed.",
        ));

        Ok(Some(CompactResult {
            messages_compacted,
            messages_remaining: 0,
            summary,
        }))
    }

    /// Generates a structured handoff summary using the current tier model.
    async fn generate_handoff_summary(&self, focus: Option<&str>) -> Result<String, AgentError> {
        let focus_instruction = match focus.map(str::trim).filter(|s| !s.is_empty()) {
            Some(f) => format!("Focus the handoff on: {f}"),
            None => "No additional focus hint was provided.".to_string(),
        };

        let transcript_path = self
            .transcript
            .as_ref()
            .map(|t| t.path().display().to_string())
            .or_else(|| self.config.transcript_path.clone())
            .unwrap_or_else(|| "transcript.md".to_string());

        let handoff_prompt = include_str!("prompts/compact_handoff.txt")
            .replace("{focus_instruction}", &focus_instruction)
            .replace("{transcript_path}", &transcript_path);

        let mut messages = self.memory.all();
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
                            chunks.push(format!("[{tool}] {result}"));
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
                            chunks.push(format!("[{tool}] {result}"));
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
                            chunks.push(format!("[{tool}] {result}"));
                        }
                        Ok(_) => {}
                        Err(e) => return Err(AgentError::Llm(e.to_string())),
                    }
                }
            }
        }

        let summary = chunks.join("").trim().to_string();
        if summary.is_empty() {
            return Err(AgentError::Llm(
                "Compaction failed to generate handoff summary".to_string(),
            ));
        }
        Ok(summary)
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

        self.memory.push(Message::system(
            "<system-reminder>TODO.md or PLAN.md still has unchecked items. Continue working through the checklist. If user input is required, call ask_user and then proceed.</system-reminder>",
        ));
        true
    }

    /// Returns the current conversation history.
    #[must_use]
    pub fn history(&self) -> Vec<Message> {
        self.memory.all()
    }

    /// Returns the model profile if available.
    #[must_use]
    pub const fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    /// Ensures the agent is initialized (profiles fetched, etc.).
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

        self.initialized = true;
    }

    /// Builds the message list for an LLM request (structured system context + todo context + memory).
    async fn build_request_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();
        let usage = self.estimate_current_usage();

        if let Some(system_ctx) = self.render_structured_system_context(usage) {
            messages.push(Message::system(system_ctx));
        }

        if let Some(todo_ctx) = self.format_todo_context() {
            messages.push(Message::system(todo_ctx));
        }

        if let Some(sandbox_dir) = self.sandbox_dir.as_deref() {
            let docs = working_docs::read_snapshot(sandbox_dir).await;
            if let Some(plan_md) = docs.plan_md {
                messages.push(Message::system(render_plan_context(&plan_md)));
            }
            if let Some(todo_md) = docs.todo_md {
                messages.push(Message::system(render_working_todo_context(&todo_md)));
            }
        }

        if let Some(handoff_ctx) = self.format_handoff_context(usage) {
            messages.push(Message::system(handoff_ctx));
        }

        messages.extend(self.memory.all());
        messages
    }

    /// Estimates current context usage using the active and fast model windows.
    fn estimate_current_usage(&self) -> f32 {
        let tier_context = self
            .profile
            .as_ref()
            .map(|p| p.context_length as usize)
            .unwrap_or(100_000);

        let fast_context = self
            .fast_profile
            .as_ref()
            .map(|p| p.context_length as usize)
            .unwrap_or(100_000);

        let context_length = tier_context.min(fast_context);
        estimate_context_usage(&self.memory.all(), context_length)
    }

    /// Renders structured XML system context with optional persona and custom blocks.
    fn render_structured_system_context(&self, usage: f32) -> Option<String> {
        if self.config.system_prompt.is_none()
            && self.config.persona_prompt.is_none()
            && self.config.context_blocks.is_empty()
            && self.config.agent_kind != AgentKind::Coding
            && self.config.transcript_path.is_none()
        {
            return None;
        }

        let mut blocks: Vec<ContextBlock> = Vec::new();

        if let Some(system_prompt) = &self.config.system_prompt {
            blocks.push(ContextBlock::new("base_system", system_prompt));
        }

        if let Some(persona_prompt) = &self.config.persona_prompt {
            blocks.push(ContextBlock::new("persona", persona_prompt));
        }

        if self.config.agent_kind == AgentKind::Coding {
            blocks.push(ContextBlock::new(
                "workspace_facts",
                "When discovering workspace guidance, load AGENT.md first. If AGENT.md is missing, load CLAUDE.md. Treat these files as repository policy for coding tasks; this behavior is not required for chatbot-style sessions.",
            ));
        }

        blocks.push(ContextBlock::new(
            "knowledge_and_time",
            concat!(
                "Your knowledge has a training cutoff -- it is frozen and does not update. ",
                "Never answer time-sensitive questions from memory alone. ",
                "Anything that could have changed since your training cutoff (current events, ",
                "recent releases, stock prices, sports results, whether someone is alive) requires verification. ",
                "Use `date` to get the current time, then `websearch` to look it up. ",
                "If the user makes a time-of-day reference, either run `date` to confirm or respond without assuming the time. ",
                "When in doubt about whether something is time-sensitive, err on the side of checking."
            ),
        ));

        blocks.push(ContextBlock::new(
            "permissions",
            concat!(
                "You have read access to the entire filesystem by default. ",
                "Write access is restricted to your sandbox directory and directories the user has explicitly approved. ",
                "When you need to modify files outside your sandbox, check if the directory is already approved. ",
                "If not, use `request_workspace` to ask for permission -- request the project root, not individual subdirectories. ",
                "Explain what changes you plan to make and wait for approval before proceeding."
            ),
        ));

        if let Some(path) = &self.config.transcript_path {
            blocks.push(ContextBlock::new(
                "transcript_memory",
                format!(
                    "Compressed memory may only keep a summary, but the full transcript remains available at {}. If details are missing, recover them by searching/reading transcript content before making irreversible changes.",
                    path
                ),
            ));
        }

        if !self.config.builtin_tool_hints.is_empty() {
            let hints = self
                .config
                .builtin_tool_hints
                .iter()
                .map(|h| format!("{}: {}", h.name, h.hint))
                .collect::<Vec<_>>()
                .join("\n");
            blocks.push(ContextBlock::new("builtin_tool_hints", hints));
        }

        blocks.extend(self.config.context_blocks.clone());
        blocks.sort_by_key(|b| b.priority.rank());

        let mut out = String::from("<context>\n");
        out.push_str(&format!(
            "<context_window usage=\"{usage:.3}\" handoff_threshold=\"{:.3}\" />\n",
            self.config.context_assembler.handoff_threshold
        ));

        for block in blocks {
            out.push_str(&render_xml_block(&block));
        }

        out.push_str("</context>");
        Some(out)
    }

    /// Injects a handoff instruction when context usage approaches the threshold.
    fn format_handoff_context(&self, usage: f32) -> Option<String> {
        if usage < self.config.context_assembler.handoff_threshold {
            return None;
        }

        let mut note = String::from("<system-reminder>\n");
        note.push_str(&self.config.context_assembler.handoff_instruction);
        if let Some(path) = &self.config.transcript_path {
            note.push_str(&format!(" Transcript source: {path}."));
        }
        note.push_str("\n</system-reminder>");
        Some(note)
    }

    /// Continues agent processing after background tasks complete (streaming version).
    async fn continue_after_background_streaming(&mut self) -> Vec<Result<AgentEvent, AgentError>> {
        let mut events = Vec::new();
        let mut iteration = 0;

        loop {
            iteration += 1;
            if iteration > self.config.max_iterations {
                events.push(Err(AgentError::MaxIterations {
                    limit: self.config.max_iterations,
                }));
                return events;
            }

            let messages = self.build_request_messages().await;
            let tool_defs = self.tools.active_definitions();
            let request = LLMRequest::new(messages).with_tool_definitions(tool_defs);

            let mut text_chunks = Vec::new();
            let mut tool_calls = Vec::new();
            let mut error: Option<String> = None;

            // Process stream based on tier
            match self.tier {
                ModelTier::Advanced => {
                    let stream = self.advanced.respond(request);
                    futures_lite::pin!(stream);
                    while let Some(event) = stream.next().await {
                        match event {
                            Ok(Event::Text(text)) => {
                                self.hooks.on_text(&text).await;
                                events.push(Ok(AgentEvent::Text(text.clone())));
                                text_chunks.push(text);
                            }
                            Ok(Event::Reasoning(r)) => events.push(Ok(AgentEvent::Reasoning(r))),
                            Ok(Event::ToolCall(call)) => tool_calls.push(call),
                            Ok(Event::BuiltInToolResult { tool, result }) => {
                                let formatted = format!("[{tool}] {result}");
                                events.push(Ok(AgentEvent::Text(formatted.clone())));
                                text_chunks.push(formatted);
                            }
                            Ok(Event::Usage(u)) => events.push(Ok(AgentEvent::Usage(u))),
                            Err(e) => {
                                error = Some(e.to_string());
                                break;
                            }
                        }
                    }
                }
                ModelTier::Balanced => {
                    let stream = self.balanced.respond(request);
                    futures_lite::pin!(stream);
                    while let Some(event) = stream.next().await {
                        match event {
                            Ok(Event::Text(text)) => {
                                self.hooks.on_text(&text).await;
                                events.push(Ok(AgentEvent::Text(text.clone())));
                                text_chunks.push(text);
                            }
                            Ok(Event::Reasoning(r)) => events.push(Ok(AgentEvent::Reasoning(r))),
                            Ok(Event::ToolCall(call)) => tool_calls.push(call),
                            Ok(Event::BuiltInToolResult { tool, result }) => {
                                let formatted = format!("[{tool}] {result}");
                                events.push(Ok(AgentEvent::Text(formatted.clone())));
                                text_chunks.push(formatted);
                            }
                            Ok(Event::Usage(u)) => events.push(Ok(AgentEvent::Usage(u))),
                            Err(e) => {
                                error = Some(e.to_string());
                                break;
                            }
                        }
                    }
                }
                ModelTier::Fast => {
                    let stream = self.fast.respond(request);
                    futures_lite::pin!(stream);
                    while let Some(event) = stream.next().await {
                        match event {
                            Ok(Event::Text(text)) => {
                                self.hooks.on_text(&text).await;
                                events.push(Ok(AgentEvent::Text(text.clone())));
                                text_chunks.push(text);
                            }
                            Ok(Event::Reasoning(r)) => events.push(Ok(AgentEvent::Reasoning(r))),
                            Ok(Event::ToolCall(call)) => tool_calls.push(call),
                            Ok(Event::BuiltInToolResult { tool, result }) => {
                                let formatted = format!("[{tool}] {result}");
                                events.push(Ok(AgentEvent::Text(formatted.clone())));
                                text_chunks.push(formatted);
                            }
                            Ok(Event::Usage(u)) => events.push(Ok(AgentEvent::Usage(u))),
                            Err(e) => {
                                error = Some(e.to_string());
                                break;
                            }
                        }
                    }
                }
            }

            if let Some(e) = error {
                events.push(Err(AgentError::Llm(e)));
                return events;
            }

            let response_text = text_chunks.join("");

            if tool_calls.is_empty() {
                if !response_text.is_empty() {
                    self.memory.push(Message::assistant(&response_text));
                }
                events.push(Ok(AgentEvent::Complete {
                    final_text: response_text,
                    turns: iteration,
                }));
                return events;
            }

            self.memory.push(Message::assistant_with_tool_calls(
                &response_text,
                tool_calls.clone(),
            ));

            // Execute tool calls
            let tools = &self.tools;
            let tool_futures = tool_calls.iter().map(|call| {
                let args_json = call.arguments.to_string();
                async move {
                    let result = tools.call(&call.name, &args_json).await;
                    let result_str = match result {
                        Ok(output) => output.as_str().unwrap_or("").to_string(),
                        Err(e) => format!("Error: {e}"),
                    };
                    (call.id.clone(), call.name.clone(), result_str)
                }
            });

            let results: Vec<(String, String, String)> =
                futures::future::join_all(tool_futures).await;

            for (call_id, call_name, content) in results {
                events.push(Ok(AgentEvent::ToolCallEnd {
                    id: call_id.clone(),
                    name: call_name,
                    result: Ok(content.clone()),
                }));
                let processed_content = self.process_reload_marker(&content);
                self.memory.push(Message::tool(&call_id, processed_content));
            }

            if let Some(ref receiver) = self.background_receiver {
                let completed_tasks = receiver.take_completed();
                for task in completed_tasks {
                    let result_msg = self.format_background_task_result(&task);
                    self.memory.push(Message::system(&result_msg));
                }
            }
        }
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
    /// When an OutputStore is available, uses lazy URL allocation:
    /// 1. Allocates URLs for large tool outputs before compression
    /// 2. Lets the fast LLM decide which URLs to reference in the summary
    /// 3. Only writes files for URLs actually referenced in the summary
    async fn maybe_compress(&mut self) -> Result<(), AgentError> {
        // Use the minimum of both context windows (most constrained)
        // This ensures:
        // 1. The selected tier doesn't run out of context for reasoning
        // 2. The fast model can see all content during compression
        let tier_context = self
            .profile
            .as_ref()
            .map(|p| p.context_length as usize)
            .unwrap_or(100_000);

        let fast_context = self
            .fast_profile
            .as_ref()
            .map(|p| p.context_length as usize)
            .unwrap_or(100_000);

        let context_length = tier_context.min(fast_context);
        let usage = estimate_context_usage(&self.memory.all(), context_length);

        match &self.config.context {
            ContextStrategy::Unlimited => Ok(()),
            ContextStrategy::Smart(config) => {
                // Use effective_trigger which reserves context for compaction process.
                if usage >= config.effective_trigger() {
                    let preserve_recent = config.preserve_recent;
                    if self.memory.len() > preserve_recent {
                        let _ = self.compact(None).await?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Processes a tool result (currently passthrough).
    ///
    /// Previously handled reload markers, now just returns the content as-is.
    fn process_reload_marker(&self, result: &str) -> String {
        result.to_string()
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
        Some(format!(
            "<system-reminder>\nYour todo list has changed. DO NOT mention this explicitly to the user. Here are the latest contents of your todo list:\n\n{items_json}. Continue on with the tasks at hand if applicable.\n</system-reminder>"
        ))
    }

    /// Formats the current todo list for context injection before each request.
    fn format_todo_context(&self) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();
        if items.is_empty() {
            return None;
        }

        let items_json = format_todo_items_json(&items);
        Some(format!(
            "<system-reminder>\nCurrent todo list (do not mention this explicitly to the user):\n\n{items_json}\n</system-reminder>"
        ))
    }

    /// Formats a completed background task result as a system message.
    fn format_background_task_result(&self, task: &aither_sandbox::CompletedTask) -> String {
        let mut msg = format!(
            "<background-bash-result task_id=\"{}\">\nScript: {}\n",
            task.task_id,
            truncate_script(&task.script, 100)
        );

        match &task.result {
            Ok(result) => {
                msg.push_str(&format!("Exit code: {}\n", result.exit_code));
                // Format stdout
                let stdout_str = result.stdout.to_string();
                if !stdout_str.is_empty() {
                    msg.push_str("Output:\n");
                    msg.push_str(&stdout_str);
                    if !stdout_str.ends_with('\n') {
                        msg.push('\n');
                    }
                }
                // Format stderr if present
                if let Some(ref stderr) = result.stderr {
                    let stderr_str = stderr.to_string();
                    if !stderr_str.is_empty() {
                        msg.push_str("Stderr:\n");
                        msg.push_str(&stderr_str);
                        if !stderr_str.ends_with('\n') {
                            msg.push('\n');
                        }
                    }
                }
            }
            Err(e) => {
                msg.push_str(&format!("Error: {e}\n"));
            }
        }

        msg.push_str("</background-bash-result>");
        msg
    }

    /// Generates a reminder about the next task after a task was completed.
    fn format_next_task_reminder(&self, completed_task: &str) -> Option<String> {
        let list = self.todo_list.as_ref()?;
        let items = list.items();

        // Find the next pending or in_progress task
        let next_task = items
            .iter()
            .find(|item| matches!(item.status, TodoStatus::Pending | TodoStatus::InProgress));

        if let Some(task) = next_task {
            Some(format!(
                "<system-reminder>\nTask \"{}\" completed. Next task: {} ({})\n</system-reminder>",
                completed_task, task.content, task.active_form
            ))
        } else if items.iter().all(|i| i.status == TodoStatus::Completed) {
            Some(format!(
                "<system-reminder>\nTask \"{}\" completed. All tasks in the todo list are now complete!\n</system-reminder>",
                completed_task
            ))
        } else {
            None
        }
    }
}

/// Formats todo items into the JSON-ish list used in system reminders.
fn format_todo_items_json(items: &[TodoItem]) -> String {
    serde_json::to_string(items).unwrap_or_else(|_| "[]".to_string())
}

fn render_plan_context(content: &str) -> String {
    format!("<plan>\n{}</plan>", escape_xml_text(content))
}

fn render_working_todo_context(content: &str) -> String {
    format!("<todo>\n{}</todo>", escape_xml_text(content))
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

fn render_xml_block(block: &ContextBlock) -> String {
    let tag = escape_xml_tag(&block.tag);
    let content = escape_xml_text(&block.content);
    format!("<{tag}>\n{content}\n</{tag}>\n")
}

fn escape_xml_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_xml_tag(tag: &str) -> String {
    tag.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect()
}
