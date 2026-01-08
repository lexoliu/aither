//! Builder for constructing agents with custom configuration.
//!
//! The builder pattern allows fluent configuration of agents with
//! tools, hooks, and various settings.

use std::sync::{Arc, RwLock};

use aither_core::{LanguageModel, llm::Tool};
use aither_sandbox::{BackgroundTaskReceiver, OutputStore};

use crate::{
    agent::{Agent, ModelTier},
    compression::ContextStrategy,
    config::{AgentConfig, ToolSearchConfig},
    context::ConversationMemory,
    hook::{HCons, Hook},
    todo::{TodoList, TodoTool},
    tools::AgentTools,
};

#[cfg(feature = "mcp")]
use aither_mcp::McpConnection;

/// Builder for constructing agents with custom configuration.
///
/// Supports tiered LLM configuration:
/// - Advanced: Primary model for main reasoning (most capable)
/// - Balanced: Model for moderate tasks like subagents (defaults to advanced)
/// - Fast: Model for quick tasks like compaction (defaults to balanced)
///
/// # Example
///
/// ```rust,ignore
/// // Simple: all tiers use the same model
/// let agent = Agent::builder(claude).build();
///
/// // Tiered: different models for different tasks
/// let agent = Agent::builder(opus)      // Advanced
///     .balanced_model(sonnet)           // Balanced
///     .fast_model(haiku)                // Fast
///     .system_prompt("You are a helpful assistant.")
///     .tool(FileSystemTool::read_only("."))
///     .build();
/// ```
#[must_use]
pub struct AgentBuilder<Advanced, Balanced = Advanced, Fast = Balanced, H = ()> {
    advanced: Advanced,
    balanced: Balanced,
    fast: Fast,
    tier: ModelTier,
    tools: AgentTools,
    hooks: H,
    config: AgentConfig,
    todo_list: Option<TodoList>,
    output_store: Option<Arc<RwLock<OutputStore>>>,
    background_receiver: Option<BackgroundTaskReceiver>,
}

impl<LLM: LanguageModel + Clone> AgentBuilder<LLM, LLM, LLM, ()> {
    /// Creates a new agent builder with default configuration.
    ///
    /// All model tiers (advanced/balanced/fast) use the same model.
    pub fn new(llm: LLM) -> Self {
        Self {
            advanced: llm.clone(),
            balanced: llm.clone(),
            fast: llm,
            tier: ModelTier::default(),
            tools: AgentTools::new(),
            hooks: (),
            config: AgentConfig::default(),
            todo_list: None,
            output_store: None,
            background_receiver: None,
        }
    }
}

impl<Advanced, Balanced, Fast, H> AgentBuilder<Advanced, Balanced, Fast, H>
where
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    H: Hook,
{
    /// Sets the balanced model for moderate tasks (e.g., subagents).
    ///
    /// This model is used when spawning subagents that don't need
    /// the full capabilities of the advanced model.
    pub fn balanced_model<B2: LanguageModel>(
        self,
        model: B2,
    ) -> AgentBuilder<Advanced, B2, Fast, H> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: model,
            fast: self.fast,
            tier: self.tier,
            tools: self.tools,
            hooks: self.hooks,
            config: self.config,
            todo_list: self.todo_list,
            output_store: self.output_store,
            background_receiver: self.background_receiver,
        }
    }

    /// Sets the fast model for quick tasks (e.g., compaction, ask command).
    ///
    /// This model is used for tasks where speed and cost matter more
    /// than capability, such as context compression.
    pub fn fast_model<F2: LanguageModel>(self, model: F2) -> AgentBuilder<Advanced, Balanced, F2, H> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: model,
            tier: self.tier,
            tools: self.tools,
            hooks: self.hooks,
            config: self.config,
            todo_list: self.todo_list,
            output_store: self.output_store,
            background_receiver: self.background_receiver,
        }
    }

    /// Sets which model tier to use for the agent's main reasoning loop.
    ///
    /// This allows creating subagents that use different capability levels:
    /// - `ModelTier::Advanced`: Use the most capable model (default)
    /// - `ModelTier::Balanced`: Use the balanced model (good for subagents)
    /// - `ModelTier::Fast`: Use the fast model (for quick tasks)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an explore subagent using the balanced model
    /// let explore_agent = Agent::builder(opus)
    ///     .balanced_model(sonnet)
    ///     .fast_model(haiku)
    ///     .tier(ModelTier::Balanced)  // Use sonnet for reasoning
    ///     .build();
    /// ```
    pub fn tier(mut self, tier: ModelTier) -> Self {
        self.tier = tier;
        self
    }

    /// Registers an eager (always-loaded) tool.
    ///
    /// Eager tools are included in every LLM request.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.register(tool);
        self
    }

    /// Registers a dynamic bash tool (type-erased).
    ///
    /// This is used for child bash tools in subagents where the concrete type
    /// is not known at compile time.
    pub fn dyn_bash(mut self, dyn_tool: aither_sandbox::DynBashTool) -> Self {
        self.tools.register_dyn_bash(dyn_tool);
        self
    }

    /// Registers a deferred (searchable) tool.
    ///
    /// Deferred tools are only loaded when the agent searches for them,
    /// reducing context usage when many tools are available.
    pub fn deferred_tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.register_deferred(tool);
        self
    }

    /// Adds a hook to intercept agent operations.
    ///
    /// Hooks are composed using the HCons pattern, allowing multiple
    /// hooks to be chained at compile time.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder(llm)
    ///     .hook(LoggingHook)
    ///     .hook(ConfirmationHook)
    ///     .build();
    /// // Type: Agent<LLM, HCons<ConfirmationHook, HCons<LoggingHook, ()>>>
    /// ```
    pub fn hook<NH: Hook>(
        self,
        hook: NH,
    ) -> AgentBuilder<Advanced, Balanced, Fast, HCons<NH, H>> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: self.fast,
            tier: self.tier,
            tools: self.tools,
            hooks: HCons::new(hook, self.hooks),
            config: self.config,
            todo_list: self.todo_list,
            output_store: self.output_store,
            background_receiver: self.background_receiver,
        }
    }

    /// Sets the system prompt.
    ///
    /// The system prompt is prepended to every conversation and
    /// remains stable for prompt caching.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the maximum number of iterations (turns).
    ///
    /// The agent will stop and return an error if this limit is exceeded.
    pub fn max_iterations(mut self, limit: usize) -> Self {
        self.config.max_iterations = limit;
        self
    }

    /// Sets the context compression strategy.
    pub fn context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.config.context = strategy;
        self
    }

    /// Enables or disables tool search explicitly.
    ///
    /// When enabled, the agent can search for deferred tools dynamically.
    /// By default, tool search is auto-enabled when the tool count
    /// exceeds the threshold.
    pub fn tool_search(mut self, enabled: bool) -> Self {
        self.config.tool_search.enabled = Some(enabled);
        self
    }

    /// Sets the threshold for auto-enabling tool search.
    ///
    /// When the total number of tools exceeds this threshold,
    /// tool search is automatically enabled.
    pub fn tool_search_threshold(mut self, count: usize) -> Self {
        self.config.tool_search.auto_threshold = Some(count);
        self
    }

    /// Sets the tool search configuration.
    pub fn tool_search_config(mut self, config: ToolSearchConfig) -> Self {
        self.config.tool_search = config;
        self
    }

    /// Registers an MCP connection.
    ///
    /// All tools from the MCP server will be available for the agent to use.
    /// You can call this method multiple times to register multiple MCP servers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use aither_mcp::McpConnection;
    ///
    /// // Connect to an MCP server
    /// let conn = McpConnection::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/"]).await?;
    ///
    /// let agent = Agent::builder(llm)
    ///     .mcp(conn)
    ///     .build();
    /// ```
    ///
    /// # Multiple Servers
    ///
    /// ```rust,ignore
    /// let filesystem = McpConnection::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/"]).await?;
    /// let github = McpConnection::spawn("npx", &["-y", "@modelcontextprotocol/server-github"]).await?;
    ///
    /// let agent = Agent::builder(llm)
    ///     .mcp(filesystem)
    ///     .mcp(github)
    ///     .build();
    /// ```
    ///
    /// # Loading from Configuration
    ///
    /// ```rust,ignore
    /// use aither_mcp::{McpConnection, McpServersConfig};
    ///
    /// let config: McpServersConfig = serde_json::from_str(&config_json)?;
    /// let connections = McpConnection::from_configs(&config).await?;
    ///
    /// let mut builder = Agent::builder(llm);
    /// for (_name, conn) in connections {
    ///     builder = builder.mcp(conn);
    /// }
    /// let agent = builder.build();
    /// ```
    #[cfg(feature = "mcp")]
    pub fn mcp(mut self, conn: McpConnection) -> Self {
        self.tools.register_mcp(conn);
        self
    }

    /// Registers a bash tool for script execution in a sandbox.
    ///
    /// The bash tool enables script execution with configurable permission modes
    /// (sandboxed, network, unsafe). It creates its own working directory with
    /// four random words and manages output storage internally.
    ///
    /// This also captures the background task receiver for polling completed
    /// background tasks during the agent loop.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use aither_sandbox::{BashTool, permission::DenyUnsafe};
    ///
    /// // Create bash tool (creates random working dir like amber-forest-thunder-pearl/)
    /// let bash_tool = BashTool::new_in(parent, DenyUnsafe, executor).await?;
    ///
    /// let agent = Agent::builder(llm)
    ///     .bash(bash_tool)
    ///     .build();
    /// ```
    pub fn bash<P, E>(mut self, bash_tool: aither_sandbox::BashTool<P, E>) -> Self
    where
        P: aither_sandbox::PermissionHandler + 'static,
        E: executor_core::Executor + Clone + 'static,
    {
        let output_store = bash_tool.output_store().clone();
        let background_receiver = bash_tool.background_receiver();
        self.tools.register(bash_tool);
        self.output_store = Some(output_store);
        self.background_receiver = Some(background_receiver);
        self
    }

    /// Sets the full agent configuration.
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Enables todo list tracking for managing long tasks.
    ///
    /// When enabled, the agent will:
    /// - Inject the current todo list into the context before each LLM request
    /// - Generate system reminders when tasks are completed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder(llm)
    ///     .todo()
    ///     .build();
    /// ```
    pub fn todo(mut self) -> Self {
        let list = TodoList::new();
        let tool = TodoTool::with_list(list.clone());
        self.tools.register(tool);
        self.todo_list = Some(list);
        self
    }

    /// Enables todo list tracking with a shared list.
    ///
    /// Use this when you want to share a todo list between multiple agents
    /// or access the list externally.
    pub fn todo_with_list(mut self, list: TodoList) -> Self {
        let tool = TodoTool::with_list(list.clone());
        self.tools.register(tool);
        self.todo_list = Some(list);
        self
    }

    /// Builds the agent.
    pub fn build(self) -> Agent<Advanced, Balanced, Fast, H> {
        Agent {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: self.fast,
            tier: self.tier,
            tools: self.tools,
            hooks: self.hooks,
            config: self.config,
            memory: ConversationMemory::default(),
            profile: None,
            fast_profile: None,
            initialized: false,
            todo_list: self.todo_list,
            output_store: self.output_store,
            background_receiver: self.background_receiver,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    use schemars::JsonSchema;
    use serde::Deserialize;

    use futures_core::Stream;

    // Mock error type for testing
    #[derive(Debug)]
    struct MockError;

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock error")
        }
    }

    impl std::error::Error for MockError {}

    // Mock LLM for testing
    #[derive(Clone)]
    struct MockLlm;

    impl LanguageModel for MockLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: aither_core::llm::LLMRequest,
        ) -> impl Stream<Item = Result<aither_core::llm::Event, Self::Error>> + Send {
            futures_lite::stream::empty()
        }

        async fn profile(&self) -> aither_core::llm::model::Profile {
            aither_core::llm::model::Profile::new(
                "mock",
                "test",
                "mock-model",
                "A mock model for testing",
                100_000,
            )
        }
    }

    // Mock tool
    struct MockTool;

    #[derive(Debug, JsonSchema, Deserialize)]
    struct MockArgs;

    impl Tool for MockTool {
        fn name(&self) -> Cow<'static, str> {
            "mock_tool".into()
        }

        fn description(&self) -> Cow<'static, str> {
            "A mock tool".into()
        }

        type Arguments = MockArgs;

        async fn call(&self, _args: Self::Arguments) -> aither_core::Result<aither_core::llm::ToolOutput> {
            Ok(aither_core::llm::ToolOutput::text("ok"))
        }
    }

    // Mock hook
    struct MockHook;

    impl Hook for MockHook {}

    #[test]
    fn test_builder_basic() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert!(agent.tools.eager_definitions().is_empty());
    }

    #[test]
    fn test_builder_with_tool() {
        let agent = AgentBuilder::new(MockLlm).tool(MockTool).build();
        assert_eq!(agent.tools.eager_definitions().len(), 1);
    }

    #[test]
    fn test_builder_with_system_prompt() {
        let agent = AgentBuilder::new(MockLlm)
            .system_prompt("You are helpful.")
            .build();
        assert_eq!(
            agent.config.system_prompt,
            Some("You are helpful.".to_string())
        );
    }

    #[test]
    fn test_builder_with_hook() {
        let _agent = AgentBuilder::new(MockLlm).hook(MockHook).build();
        // Type check: agent has HCons<MockHook, ()> as hook type
    }

    #[test]
    fn test_builder_with_multiple_hooks() {
        let _agent = AgentBuilder::new(MockLlm)
            .hook(MockHook)
            .hook(MockHook)
            .build();
        // Type check: agent has HCons<MockHook, HCons<MockHook, ()>>
    }

    #[test]
    fn test_builder_max_iterations() {
        let agent = AgentBuilder::new(MockLlm).max_iterations(100).build();
        assert_eq!(agent.config.max_iterations, 100);
    }

    #[test]
    fn test_builder_tool_search() {
        let agent = AgentBuilder::new(MockLlm).tool_search(true).build();
        assert_eq!(agent.config.tool_search.enabled, Some(true));
    }
}
