//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages conversation memory, applies context compression, and
//! handles tool execution in an agent-controlled loop.

use std::time::{Duration, Instant};

use async_fs as fs;
use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, Message, model::Profile as ModelProfile},
};
use futures_core::Stream;
use futures_lite::StreamExt;

use crate::{
    compression::{ContentWithUrl, ContextStrategy, estimate_context_usage},
    config::AgentConfig,
    context::ConversationMemory,
    error::AgentError,
    event::AgentEvent,
    hook::{
        Hook, PostToolAction, PreToolAction, StopContext, StopReason, ToolResultContext,
        ToolUseContext,
    },
    todo::{TodoItem, TodoList, TodoStatus},
    tools::AgentTools,
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

        let stream = self.run(prompt);
        futures_lite::pin!(stream);

        let mut final_text = String::new();
        while let Some(event) = stream.next().await {
            match event? {
                AgentEvent::Text(chunk) => final_text.push_str(&chunk),
                AgentEvent::Complete { final_text: text, .. } => {
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
    /// let mut stream = agent.run("Implement the feature");
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
    ) -> impl Stream<Item = Result<AgentEvent, AgentError>> + '_ {
        let prompt = prompt.to_string();
        self.run_streaming(prompt)
    }

    /// Internal streaming implementation.
    fn run_streaming(
        &mut self,
        prompt: String,
    ) -> impl Stream<Item = Result<AgentEvent, AgentError>> + '_ {
        async_stream::try_stream! {
            self.ensure_initialized().await;

            // Apply context compression if needed
            self.maybe_compress().await?;

            // Add user message
            self.memory.push(Message::user(&prompt));

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
                let messages = self.build_request_messages();

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

                // If no tool calls, we're done
                if tool_calls.is_empty() {
                    if !response_text.is_empty() {
                        self.memory.push(Message::assistant(&response_text));
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
                    yield AgentEvent::ToolCallStart {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.to_string(),
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

                    // Yield tool call end event
                    yield AgentEvent::ToolCallEnd {
                        id: call_id.clone(),
                        name: call_name,
                        result: Ok(content.clone()),
                    };

                    if content.contains("not found") || content.contains("Invalid arguments") {
                        has_tool_error = true;
                    }
                    let processed_content = self.process_reload_marker(&content);
                    self.memory.push(Message::tool(&call_id, processed_content));
                }

                // If there was a tool error, inject a reminder
                if has_tool_error {
                    let reminder = concat!(
                        "<system-reminder>\n",
                        "IMPORTANT: You have exactly ONE tool: `bash`. ",
                        "All other capabilities (websearch, webfetch, task, etc.) are bash commands, not separate tools.\n\n",
                        "Correct usage:\n",
                        "- websearch: bash with script `websearch \"query\"`\n",
                        "- webfetch: bash with script `webfetch \"url\"`\n",
                        "- Network commands (websearch, webfetch) work in sandboxed mode - they handle network internally.\n",
                        "- Only use mode=\"network\" when YOUR bash script needs direct network access (curl, wget, etc.).\n",
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

    /// Compacts the conversation by summarizing all messages and starting fresh.
    ///
    /// This generates a summary of the entire conversation, clears history,
    /// and starts a new session with only the summary as context.
    ///
    /// # Errors
    ///
    /// Returns an error if summary generation fails.
    pub async fn compact(&mut self) -> Result<Option<CompactResult>, AgentError> {
        self.ensure_initialized().await;

        let messages = self.memory.all();
        if messages.is_empty() {
            return Ok(None);
        }

        // Get compression config or use defaults
        let config = match &self.config.context {
            ContextStrategy::Smart(config) => config.clone(),
            ContextStrategy::Unlimited => crate::compression::SmartCompressionConfig::default(),
        };

        let preserved = config.extract_preserved(&messages);
        let messages_compacted = messages.len();

        let summary = config
            .generate_summary(&self.fast, &messages, &preserved)
            .await
            .map_err(|e| AgentError::Llm(e.to_string()))?;

        // Clear everything and start fresh with just the summary
        self.memory.clear();
        self.memory.push_summary(Message::system(summary.clone()));

        Ok(Some(CompactResult {
            messages_compacted,
            messages_remaining: 0,
            summary,
        }))
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

    /// Builds the message list for an LLM request (system prompt + todo context + memory).
    fn build_request_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        if let Some(ref system_prompt) = self.config.system_prompt {
            messages.push(Message::system(system_prompt));
        }

        if let Some(todo_ctx) = self.format_todo_context() {
            messages.push(Message::system(todo_ctx));
        }

        messages.extend(self.memory.all());
        messages
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

            let messages = self.build_request_messages();
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
                            Err(e) => { error = Some(e.to_string()); break; }
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
                            Err(e) => { error = Some(e.to_string()); break; }
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
                            Err(e) => { error = Some(e.to_string()); break; }
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

            let results: Vec<(String, String, String)> = futures::future::join_all(tool_futures).await;

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

    /// Minimum content length to consider for URL allocation during compression.
    /// Smaller content stays inline.
    const MIN_CONTENT_FOR_URL: usize = 500;

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
                // Use effective_trigger which reserves context for compaction process
                if usage >= config.effective_trigger() {
                    let preserved = config.extract_preserved(&self.memory.all());
                    let to_compress = self.memory.drain_oldest(config.preserve_recent);

                    if !to_compress.is_empty() {
                        let summary = self
                            .compress_with_urls(config, &to_compress, &preserved)
                            .await?;
                        self.memory.push_summary(Message::system(summary));
                    }
                }
                Ok(())
            }
        }
    }

    /// Compresses messages using URL-aware compression when OutputStore is available.
    ///
    /// This method:
    /// 1. Allocates URLs for large tool outputs
    /// 2. Calls the fast LLM with content + URL mappings
    /// 3. Writes only the files for URLs referenced in the summary
    async fn compress_with_urls(
        &self,
        config: &crate::compression::SmartCompressionConfig,
        to_compress: &[Message],
        preserved: &crate::compression::PreservedContent,
    ) -> Result<String, AgentError> {
        // If no output store, fall back to simple compression
        let Some(output_store) = &self.output_store else {
            return config
                .generate_summary(&self.fast, to_compress, preserved)
                .await
                .map_err(|e| AgentError::Llm(e.to_string()));
        };

        // Allocate URLs for large tool outputs
        let mut pending_urls: Vec<ContentWithUrl> = Vec::new();
        let mut content_to_url: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for msg in to_compress {
            // Only allocate URLs for large tool results
            if msg.role() == aither_core::llm::Role::Tool
                && msg.content().len() >= Self::MIN_CONTENT_FOR_URL
            {
                let content = msg.content().to_string();
                // Avoid duplicate allocations for same content
                if !content_to_url.contains_key(&content) {
                    let url = output_store.allocate_text_url();
                    content_to_url.insert(content.clone(), url.clone());
                    pending_urls.push(ContentWithUrl { content, url });
                }
            }
        }

        // Generate summary with URL-aware compression
        let result = config
            .generate_summary_with_urls(&self.fast, to_compress, preserved, &pending_urls)
            .await
            .map_err(|e| AgentError::Llm(e.to_string()))?;

        // Write only the files for referenced URLs
        // Collect files to write outside the lock, then write them
        let files_to_write: Vec<_> = pending_urls
            .iter()
            .filter(|p| result.referenced_urls.contains(&p.url))
            .map(|p| (p.url.clone(), p.content.clone()))
            .collect();

        if !files_to_write.is_empty() {
            // Get the output directory while holding the lock briefly
            let output_dir = output_store.dir().to_path_buf();

            // Write files without holding the lock
            for (url, content) in files_to_write {
                let filename = url.strip_prefix("outputs/").unwrap_or(&url);
                let filepath = output_dir.join(filename);
                if let Err(e) = fs::write(&filepath, content.as_bytes()).await {
                    tracing::warn!(url = %url, error = %e, "failed to write referenced URL");
                } else {
                    tracing::debug!(url = %url, "wrote referenced URL");
                }
            }
        }

        Ok(result.summary)
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

/// Truncates a script for display in messages.
fn truncate_script(script: &str, max_chars: usize) -> &str {
    let script = script.trim();
    // Find byte index at max_chars boundary
    match script.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &script[..byte_idx],
        None => script, // String is shorter than max_chars
    }
}
