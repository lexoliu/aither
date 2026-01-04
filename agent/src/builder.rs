//! Builder for constructing agents with custom configuration.
//!
//! The builder pattern allows fluent configuration of agents with
//! tools, hooks, and various settings.

use aither_core::{LanguageModel, llm::Tool};

use crate::{
    agent::Agent,
    compression::ContextStrategy,
    config::{AgentConfig, ToolSearchConfig},
    context::ConversationMemory,
    hook::{HCons, Hook},
    tools::AgentTools,
};

#[cfg(feature = "mcp")]
use aither_mcp::McpConnection;

/// Builder for constructing agents with custom configuration.
///
/// # Example
///
/// ```rust,ignore
/// let agent = Agent::builder(claude)
///     .system_prompt("You are a helpful assistant.")
///     .tool(FileSystemTool::read_only("."))
///     .tool(CommandTool::new())
///     .hook(LoggingHook)
///     .max_iterations(50)
///     .build();
/// ```
#[must_use]
pub struct AgentBuilder<LLM, H = ()> {
    llm: LLM,
    tools: AgentTools,
    hooks: H,
    config: AgentConfig,
}

impl<LLM: LanguageModel> AgentBuilder<LLM, ()> {
    /// Creates a new agent builder with default configuration.
    pub fn new(llm: LLM) -> Self {
        Self {
            llm,
            tools: AgentTools::new(),
            hooks: (),
            config: AgentConfig::default(),
        }
    }
}

impl<LLM, H> AgentBuilder<LLM, H>
where
    LLM: LanguageModel,
    H: Hook,
{
    /// Registers an eager (always-loaded) tool.
    ///
    /// Eager tools are included in every LLM request.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.register(tool);
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
    pub fn hook<NH: Hook>(self, hook: NH) -> AgentBuilder<LLM, HCons<NH, H>> {
        AgentBuilder {
            llm: self.llm,
            tools: self.tools,
            hooks: HCons::new(hook, self.hooks),
            config: self.config,
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

    /// Sets the full agent configuration.
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the agent.
    pub fn build(self) -> Agent<LLM, H> {
        Agent {
            llm: self.llm,
            tools: self.tools,
            hooks: self.hooks,
            config: self.config,
            memory: ConversationMemory::default(),
            profile: None,
            initialized: false,
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

        async fn call(&mut self, _args: Self::Arguments) -> aither_core::Result {
            Ok("ok".to_string())
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
