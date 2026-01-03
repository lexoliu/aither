//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages conversation memory, applies context compression, and
//! handles tool execution in an agent-controlled loop.

use std::time::Instant;

use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, Message, model::Profile as ModelProfile},
};
use futures_lite::StreamExt;

use crate::{
    compression::{ContextStrategy, estimate_context_usage},
    config::AgentConfig,
    context::ConversationMemory,
    error::AgentError,
    hook::{Hook, HookAction, StopContext, StopReason, ToolResultContext, ToolUseContext},
    stream::AgentStream,
    tools::AgentTools,
};

/// Maximum number of tool call iterations per query.
const MAX_TOOL_ITERATIONS: usize = 16;

/// An autonomous agent that processes tasks using a language model.
///
/// The agent manages conversation context, handles tool execution in a loop,
/// and applies context compression when needed. Unlike the raw `LanguageModel`
/// trait which only emits events, the agent handles the full tool execution
/// cycle.
///
/// # Type Parameters
///
/// - `LLM`: The language model implementation (must implement `LanguageModel`)
/// - `H`: Composed hooks for customizing behavior (defaults to `()` for no hooks)
///
/// # Example
///
/// ```rust,ignore
/// // Simple usage
/// let agent = Agent::new(claude);
/// let response = agent.query("What is 2+2?").await?;
///
/// // With tools and hooks
/// let agent = Agent::builder(claude)
///     .tool(FileSystemTool::read_only("."))
///     .hook(LoggingHook)
///     .build();
///
/// let response = agent.query("List all Rust files").await?;
/// ```
#[derive(Debug)]
pub struct Agent<LLM, H = ()> {
    /// The language model.
    pub(crate) llm: LLM,

    /// Registered tools (eager and deferred).
    pub(crate) tools: AgentTools,

    /// Composed hooks.
    pub(crate) hooks: H,

    /// Agent configuration.
    pub(crate) config: AgentConfig,

    /// Conversation memory.
    pub(crate) memory: ConversationMemory,

    /// Cached model profile.
    pub(crate) profile: Option<ModelProfile>,

    /// Whether tools have been bootstrapped.
    pub(crate) initialized: bool,
}

impl<LLM: LanguageModel> Agent<LLM, ()> {
    /// Creates a new agent with default configuration.
    #[must_use]
    pub fn new(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::default())
    }

    /// Creates a new agent with the specified configuration.
    #[must_use]
    pub fn with_config(llm: LLM, config: AgentConfig) -> Self {
        Self {
            llm,
            tools: AgentTools::with_config(config.tool_search.clone()),
            hooks: (),
            config,
            memory: ConversationMemory::default(),
            profile: None,
            initialized: false,
        }
    }

    /// Returns a builder for more complex agent construction.
    #[must_use]
    pub fn builder(llm: LLM) -> crate::builder::AgentBuilder<LLM, ()> {
        crate::builder::AgentBuilder::new(llm)
    }
}

impl<LLM, H> Agent<LLM, H>
where
    LLM: LanguageModel,
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
        self.ensure_initialized().await;
        self.run_query(prompt).await
    }

    /// Runs the agent with streaming events.
    ///
    /// Returns a stream of `AgentEvent`s that can be consumed to observe
    /// the agent's progress in real-time.
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
    pub fn run(&mut self, prompt: &str) -> AgentStream<'_, LLM, H> {
        AgentStream::new(self, prompt.to_string())
    }

    /// Registers a tool for the agent to use.
    pub fn register_tool<T: aither_core::llm::Tool + 'static>(&mut self, tool: T) {
        self.tools.register(tool);
    }

    /// Registers a deferred tool (loaded via search).
    pub fn register_deferred_tool<T: aither_core::llm::Tool + 'static>(&mut self, tool: T) {
        self.tools.register_deferred(tool);
    }

    /// Adds a message to the conversation history.
    pub fn push_message(&mut self, message: Message) {
        self.memory.push(message);
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.memory.clear();
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

    /// Ensures the agent is initialized (profile fetched, etc.).
    async fn ensure_initialized(&mut self) {
        if self.initialized {
            return;
        }

        // Fetch model profile
        self.profile = Some(self.llm.profile().await);
        self.initialized = true;
    }

    /// Runs a single query turn with tool loop handling.
    async fn run_query(&mut self, prompt: &str) -> Result<String, AgentError> {
        // Apply context compression if needed
        self.maybe_compress().await?;

        // Add user message
        self.memory.push(Message::user(prompt));

        // Run the tool loop
        let mut iteration = 0;
        let mut final_text = String::new();

        loop {
            iteration += 1;
            if iteration > MAX_TOOL_ITERATIONS {
                return Err(AgentError::MaxIterations {
                    limit: MAX_TOOL_ITERATIONS,
                });
            }

            // Build messages
            let mut messages = Vec::new();

            // System prompt (static for caching)
            if let Some(ref system_prompt) = self.config.system_prompt {
                messages.push(Message::system(system_prompt));
            }

            // Previous conversation
            messages.extend(self.memory.all());

            // Create request with tool definitions
            let tool_defs = self.tools.active_definitions();
            let request = LLMRequest::new(messages).with_tool_definitions(tool_defs);

            // Stream and collect the response
            let stream = self.llm.respond(request);
            futures_lite::pin!(stream);

            let mut text_chunks = Vec::new();
            let mut tool_calls = Vec::new();

            while let Some(event) = stream.next().await {
                match event.map_err(|e| AgentError::Llm(e.to_string()))? {
                    Event::Text(text) => {
                        text_chunks.push(text);
                    }
                    Event::Reasoning(_) => {
                        // Reasoning is for observability, not part of response
                    }
                    Event::ToolCall(call) => {
                        tool_calls.push(call);
                    }
                    Event::BuiltInToolResult { tool, result } => {
                        // Built-in tool results are treated as text
                        text_chunks.push(format!("[{tool}] {result}"));
                    }
                }
            }

            let response_text = text_chunks.join("");

            // If no tool calls, we're done
            if tool_calls.is_empty() {
                final_text = response_text.clone();

                // Store assistant response in memory
                if !response_text.is_empty() {
                    self.memory.push(Message::assistant(&response_text));
                }

                break;
            }

            // Store assistant response with tool calls in memory
            if !response_text.is_empty() {
                self.memory.push(Message::assistant(&response_text));
            }

            // Execute tool calls
            for call in tool_calls {
                let args_json = call.arguments.to_string();
                let message_count = self.memory.len();

                // Pre-tool hook
                let tool_ctx = ToolUseContext {
                    tool_name: &call.name,
                    arguments: &args_json,
                    turn: iteration,
                    message_count,
                };

                match self.hooks.pre_tool_use(&tool_ctx).await {
                    HookAction::Abort(reason) => {
                        return Err(AgentError::HookRejected {
                            hook: "pre_tool_use",
                            reason,
                        });
                    }
                    HookAction::Skip => continue,
                    HookAction::Continue | HookAction::Replace(_) => {}
                }

                // Execute the tool
                let start = Instant::now();
                let result = self.tools.call(&call.name, &args_json).await;
                let duration = start.elapsed();

                // Post-tool hook
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

                match self.hooks.post_tool_use(&result_ctx).await {
                    HookAction::Abort(reason) => {
                        return Err(AgentError::HookRejected {
                            hook: "post_tool_use",
                            reason,
                        });
                    }
                    _ => {}
                }

                // Handle the result
                let output = result.map_err(|e| AgentError::ToolExecution {
                    name: call.name.clone(),
                    error: e.to_string(),
                })?;

                // Add tool result to memory as a formatted message
                // Format: [tool_call_id:tool_name] result
                let tool_content = format!("[{}:{}] {}", call.id, call.name, output);
                self.memory.push(Message::tool(tool_content));
            }

            // Continue the loop to get the next response
        }

        // Notify hooks
        let stop_ctx = StopContext {
            final_text: &final_text,
            turns: iteration,
            reason: StopReason::Complete,
        };

        match self.hooks.on_stop(&stop_ctx).await {
            HookAction::Abort(reason) => {
                return Err(AgentError::HookRejected {
                    hook: "on_stop",
                    reason,
                });
            }
            _ => {}
        }

        Ok(final_text)
    }

    /// Compresses context if needed.
    async fn maybe_compress(&mut self) -> Result<(), AgentError> {
        let context_length = self
            .profile
            .as_ref()
            .map(|p| p.context_length)
            .unwrap_or(100_000);

        let usage = estimate_context_usage(&self.memory.all(), context_length as usize);

        match &self.config.context {
            ContextStrategy::Unlimited => Ok(()),
            ContextStrategy::SlidingWindow { max_messages } => {
                self.memory.drain_oldest(*max_messages);
                Ok(())
            }
            ContextStrategy::Smart(config) => {
                if usage >= config.trigger_threshold {
                    let preserved = config.extract_preserved(&self.memory.all());
                    let to_compress = self.memory.drain_oldest(config.preserve_recent);

                    if !to_compress.is_empty() {
                        let summary = config
                            .generate_summary(&self.llm, &to_compress, &preserved)
                            .await
                            .map_err(|e| AgentError::Llm(e.to_string()))?;

                        self.memory.push_summary(Message::system(summary));
                    }
                }
                Ok(())
            }
        }
    }
}
