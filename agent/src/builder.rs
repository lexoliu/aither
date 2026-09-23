//! Builder for constructing agents with custom configuration.
//!
//! The builder pattern allows fluent configuration of agents with
//! tools, hooks, and various settings.

#[cfg(feature = "skills")]
use std::sync::Arc;

use aither_core::{LanguageModel, llm::Tool};
use aither_sandbox::{BackgroundTaskReceiver, JobRegistry, PermissionEventReceiver};
#[cfg(feature = "skills")]
use aither_skills::SkillRegistry;

use crate::{
    agent::Agent,
    compression::ContextStrategy,
    config::{AgentConfig, AgentKind},
    context::Context,
    hook::{HCons, Hook},
    model_group::ModelTier,
    todo::{TodoList, TodoTool},
    tools::AgentTools,
    transcript::Transcript,
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
    context: Context,
    todo_list: Option<TodoList>,
    background_receiver: Option<BackgroundTaskReceiver>,
    permission_receiver: Option<PermissionEventReceiver>,
    job_registry: Option<JobRegistry>,
    transcript: Option<Transcript>,
    sandbox_dir: Option<std::path::PathBuf>,
    #[cfg(feature = "skills")]
    skill_registry: Option<Arc<SkillRegistry>>,
}

impl<Advanced, Balanced, Fast, H> std::fmt::Debug for AgentBuilder<Advanced, Balanced, Fast, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("tier", &self.tier)
            .field("config", &self.config)
            .field("todo_enabled", &self.todo_list.is_some())
            .finish_non_exhaustive()
    }
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
            context: Context::default(),
            todo_list: None,
            background_receiver: None,
            permission_receiver: None,
            job_registry: None,
            transcript: None,
            sandbox_dir: None,
            #[cfg(feature = "skills")]
            skill_registry: None,
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
            context: self.context,
            todo_list: self.todo_list,
            background_receiver: self.background_receiver,
            permission_receiver: self.permission_receiver,
            job_registry: self.job_registry,
            transcript: self.transcript,
            sandbox_dir: self.sandbox_dir,
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
        }
    }

    /// Sets the fast model for quick tasks (e.g., compaction, ask command).
    ///
    /// This model is used for tasks where speed and cost matter more
    /// than capability, such as context compression.
    pub fn fast_model<F2: LanguageModel>(
        self,
        model: F2,
    ) -> AgentBuilder<Advanced, Balanced, F2, H> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: model,
            tier: self.tier,
            tools: self.tools,
            hooks: self.hooks,
            config: self.config,
            context: self.context,
            todo_list: self.todo_list,
            background_receiver: self.background_receiver,
            permission_receiver: self.permission_receiver,
            job_registry: self.job_registry,
            transcript: self.transcript,
            sandbox_dir: self.sandbox_dir,
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
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
    pub const fn tier(mut self, tier: ModelTier) -> Self {
        self.tier = tier;
        self
    }

    /// Registers an eager (always-loaded) tool.
    ///
    /// Eager tools are included in every LLM request.
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools
            .register(tool)
            .expect("tool registration must succeed: duplicate name or missing description");
        self
    }

    /// Registers a dynamic terminal tool (type-erased).
    ///
    /// This is used for child terminal capability bundles in subagents where the
    /// concrete terminal type is not known at compile time.
    pub fn dyn_terminal(mut self, dyn_tool: aither_sandbox::DynTerminalTool) -> Self {
        self.background_receiver = Some(dyn_tool.background_receiver());
        self.permission_receiver = Some(dyn_tool.permission_receiver());
        self.job_registry = Some(dyn_tool.job_registry());
        self.sandbox_dir = Some(dyn_tool.working_dir().clone());
        self.tools.register_dyn_terminal(dyn_tool);
        self
    }

    /// Adds a hook to intercept agent operations.
    ///
    /// Hooks are composed using the `HCons` pattern, allowing multiple
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
    pub fn hook<NH: Hook>(self, hook: NH) -> AgentBuilder<Advanced, Balanced, Fast, HCons<NH, H>> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: self.fast,
            tier: self.tier,
            tools: self.tools,
            hooks: HCons::new(hook, self.hooks),
            config: self.config,
            context: self.context,
            todo_list: self.todo_list,
            background_receiver: self.background_receiver,
            permission_receiver: self.permission_receiver,
            job_registry: self.job_registry,
            transcript: self.transcript,
            sandbox_dir: self.sandbox_dir,
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
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
    pub const fn max_iterations(mut self, limit: usize) -> Self {
        self.config.max_iterations = limit;
        self
    }

    /// Sets the context compression strategy.
    pub const fn context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.config.context = strategy;
        self
    }

    /// Sets an optional persona overlay prompt.
    pub fn persona_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.persona_prompt = Some(prompt.into());
        self
    }

    /// Sets agent kind (coding or chatbot).
    pub const fn agent_kind(mut self, kind: AgentKind) -> Self {
        self.config.agent_kind = kind;
        self
    }

    /// Sets transcript path for long-memory recovery guidance.
    pub fn transcript_path(mut self, path: impl Into<String>) -> Self {
        self.config.transcript_path = Some(path.into());
        self
    }

    /// Enables writing a readable transcript to the given path.
    pub fn transcript(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.transcript = Some(Transcript::new(path));
        self
    }

    /// Sets sandbox directory for working document supervision.
    pub fn sandbox_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.sandbox_dir = Some(path.into());
        self
    }

    /// Sets the skill registry used for runtime skill matching and activation.
    #[cfg(feature = "skills")]
    pub fn skill_registry(mut self, registry: Arc<SkillRegistry>) -> Self {
        self.skill_registry = Some(registry);
        self
    }

    /// Inserts or replaces a typed persistent system block.
    pub fn system<T: serde::Serialize>(mut self, value: T) -> Self {
        self.context.insert_system(&value);
        self
    }

    /// Inserts or replaces a persistent system block with an explicit tag.
    pub fn system_named(mut self, tag: impl Into<String>, content: impl Into<String>) -> Self {
        self.context.insert_system_named(tag, content);
        self
    }

    /// Inserts or replaces a persistent system block with raw text.
    ///
    /// Unlike [`system`](Self::system) and [`system_named`](Self::system_named),
    /// the `content` is stored verbatim without any XML wrapping. This is the
    /// preferred entry point for prose system blocks (workspace descriptions,
    /// runtime metadata, environment hints) where XML structure adds tokens
    /// without providing semantic value.
    pub fn system_text(mut self, tag: impl Into<String>, content: impl Into<String>) -> Self {
        self.context.insert_system_text(tag, content);
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

    /// Registers the terminal tool for command execution in a sandbox.
    ///
    /// The terminal tool enables command execution with configurable permission modes
    /// (sandboxed, network, unsafe). It creates its own working directory with
    /// four random words and manages output storage internally.
    ///
    /// This also captures the background task receiver for polling completed
    /// background tasks during the agent loop.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use aither_sandbox::{TerminalTool, ToolRegistryBuilder, permission::DenyUnsafe};
    /// use std::sync::Arc;
    ///
    /// // Create terminal tool (creates random working dir like amber-forest-thunder-pearl/)
    /// let terminal_tool = TerminalTool::new_in(parent, DenyUnsafe, executor).await?;
    /// let registry = Arc::new(ToolRegistryBuilder::new().build(terminal_tool.outputs_dir()));
    /// let terminal_tool = terminal_tool.with_registry(registry);
    ///
    /// let agent = Agent::builder(llm)
    ///     .terminal(terminal_tool)
    ///     .build();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn terminal<P, E, State>(
        mut self,
        terminal_tool: aither_sandbox::TerminalTool<P, E, State>,
    ) -> Self
    where
        P: aither_sandbox::PermissionHandler + 'static,
        E: executor_core::Executor + Clone + 'static,
        State: Clone + 'static,
        aither_sandbox::TerminalTool<P, E, State>: Tool + 'static,
    {
        let background_receiver = terminal_tool.background_receiver();
        let permission_receiver = terminal_tool.permission_receiver();
        let job_registry = terminal_tool.job_registry();
        self.sandbox_dir = Some(terminal_tool.working_dir().clone());
        self.tools
            .register(terminal_tool)
            .expect("tool registration must succeed: duplicate name or missing description");
        self.background_receiver = Some(background_receiver);
        self.permission_receiver = Some(permission_receiver);
        self.job_registry = Some(job_registry);
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
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn todo(mut self) -> Self {
        let list = TodoList::new();
        let tool = TodoTool::with_list(list.clone());
        self.tools
            .register(tool)
            .expect("tool registration must succeed: duplicate name or missing description");
        self.todo_list = Some(list);
        self
    }

    /// Enables todo list tracking with a shared list.
    ///
    /// Use this when you want to share a todo list between multiple agents
    /// or access the list externally.
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn todo_with_list(mut self, list: TodoList) -> Self {
        let tool = TodoTool::with_list(list.clone());
        self.tools
            .register(tool)
            .expect("tool registration must succeed: duplicate name or missing description");
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
            context: self.context,
            profile: None,
            fast_profile: None,
            initialized: false,
            todo_list: self.todo_list,
            background_receiver: self.background_receiver,
            permission_receiver: self.permission_receiver,
            job_registry: self.job_registry,
            transcript: self.transcript,
            sandbox_dir: self.sandbox_dir,
            last_working_docs: None,
            last_request_started_at: None,
            transient_system_messages: Vec::new(),
            cache_stats: crate::CacheStats::new(),
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
            #[cfg(feature = "skills")]
            active_skills: Vec::new(),
            #[cfg(feature = "skills")]
            active_allowed_tools: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[cfg(feature = "skills")]
    use aither_skills::{Skill, SkillRegistry};
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

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = aither_core::llm::model::Profile> + Send {
            std::future::ready(aither_core::llm::model::Profile::new(
                "mock",
                "test",
                "mock-model",
                "A mock model for testing",
                100_000,
            ))
        }
    }

    // Mock tool
    struct MockTool;

    /// Does nothing, for tests that only care that a tool was registered.
    #[derive(Debug, JsonSchema, Deserialize)]
    struct MockArgs;

    impl Tool for MockTool {
        fn name(&self) -> Cow<'static, str> {
            "mock_tool".into()
        }

        type Arguments = MockArgs;
        type Res = aither_core::llm::ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(aither_core::llm::ToolResult::text("ok")))
        }
    }

    // Mock hook
    struct MockHook;

    impl Hook for MockHook {}

    #[test]
    fn test_builder_basic() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert!(agent.tools.definitions().is_empty());
    }

    #[test]
    fn public_model_tier_configures_agent_builder() {
        let agent = AgentBuilder::new(MockLlm)
            .tier(crate::ModelTier::Balanced)
            .build();
        assert_eq!(agent.tier, crate::ModelTier::Balanced);
    }

    #[test]
    fn test_builder_with_tool() {
        let agent = AgentBuilder::new(MockLlm).tool(MockTool).build();
        assert_eq!(agent.tools.definitions().len(), 1);
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
    fn test_builder_default_config() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert_eq!(
            agent.config.max_iterations,
            AgentConfig::default().max_iterations
        );
    }

    #[cfg(feature = "skills")]
    #[test]
    fn test_builder_preserves_skill_registry() {
        let mut registry = SkillRegistry::new();
        registry.register(Skill {
            name: "code-review".to_string(),
            description: "Review code carefully".to_string(),
            instructions: "Use a review checklist.".to_string(),
            allowed_tools: Some(vec!["mock_tool".to_string()]),
            resources: std::collections::HashMap::new(),
        });

        let agent = AgentBuilder::new(MockLlm)
            .skill_registry(Arc::new(registry))
            .build();
        assert!(agent.skill_registry.is_some());
    }

    #[cfg(feature = "skills")]
    #[test]
    fn test_runtime_skill_activation_is_disabled() {
        let mut registry = SkillRegistry::new();
        registry.register(Skill {
            name: "code-review".to_string(),
            description: "Review code carefully".to_string(),
            instructions: "Use a review checklist.".to_string(),
            allowed_tools: Some(vec!["mock_tool".to_string()]),
            resources: std::collections::HashMap::new(),
        });

        let mut agent = AgentBuilder::new(MockLlm)
            .skill_registry(Arc::new(registry))
            .build();
        let events = agent.activate_skills_for_prompt("please review this patch");

        assert!(events.is_empty());
        assert!(agent.active_skills.is_empty());
        assert!(agent.active_allowed_tools.is_none());
    }
}
