//! Agent framework for building autonomous AI agents with planning, execution, and memory control.

use std::path::PathBuf;

use aither_core::{
    LanguageModel,
    llm::{
        LLMRequest, Message, Tool,
        model::{Ability, Parameters, Profile as ModelProfile},
        tool::Tools,
    },
};
use anyhow::anyhow;

#[cfg(feature = "command")]
pub use aither_command as command;
#[cfg(feature = "filesystem")]
pub use aither_fs as filesystem;
#[cfg(feature = "websearch")]
pub use aither_websearch as websearch;

/// Opinionated wrappers around [`Agent`] for coding workflows.
pub mod coder;
/// Execution strategies for running plan steps.
pub mod execute;
/// Memory management and context compression.
pub mod memory;
/// Planning strategies for breaking down goals.
pub mod plan;
/// Deep research agent with streaming findings.
#[cfg(feature = "websearch")]
pub mod research;
/// Sub-agent creation and management.
pub mod sub_agent;
/// Todo list management and tracking.
pub mod todo;

pub use coder::Coder;
#[cfg(feature = "websearch")]
pub use research::DeepResearchAgent;

use crate::{
    execute::Executor,
    memory::{ContextStrategy, ConversationMemory},
    plan::{PlanOutcome, Planner},
    sub_agent::SubAgent,
    todo::TodoList,
};

/// Agent-wide configuration describing context compression, iteration limits, and default tooling.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    context: ContextStrategy,
    max_iterations: usize,
    tooling: ToolingConfig,
}

impl AgentConfig {
    /// Creates a new agent configuration from its fields.
    #[must_use]
    pub const fn new(
        context: ContextStrategy,
        max_iterations: usize,
        tooling: ToolingConfig,
    ) -> Self {
        Self {
            context,
            max_iterations,
            tooling,
        }
    }

    /// Returns the current context strategy.
    #[must_use]
    pub const fn context(&self) -> &ContextStrategy {
        &self.context
    }

    /// Sets the context strategy.
    pub fn set_context(&mut self, context: ContextStrategy) {
        self.context = context;
    }

    /// Returns the iteration cap.
    #[must_use]
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Sets the iteration cap.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Updates the iteration cap in place.
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    /// Returns the tooling configuration.
    #[must_use]
    pub const fn tooling(&self) -> &ToolingConfig {
        &self.tooling
    }

    /// Returns a mutable reference to the tooling configuration.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn tooling_mut(&mut self) -> &mut ToolingConfig {
        &mut self.tooling
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            context: ContextStrategy::SlidingWindow { max_messages: 32 },
            max_iterations: 48,
            tooling: ToolingConfig::none(),
        }
    }
}

/// Configures which tools are mounted automatically when an [`Agent`] starts.
#[derive(Debug, Clone)]
pub struct ToolingConfig {
    filesystem: Option<FileSystemAccess>,
    enable_shell: bool,
}

/// Describes how the agent should mount filesystem access relative to a root directory.
#[derive(Debug, Clone)]
pub struct FileSystemAccess {
    root: Option<PathBuf>,
    allow_writes: bool,
}

impl FileSystemAccess {
    /// Mounts the current working directory with the specified permissions.
    #[must_use]
    pub const fn working_directory(allow_writes: bool) -> Self {
        Self {
            root: None,
            allow_writes,
        }
    }

    /// Mounts a read-only filesystem rooted at the provided path.
    #[must_use]
    pub fn read_only(root: impl Into<PathBuf>) -> Self {
        Self::rooted(root, false)
    }

    /// Mounts a read-write filesystem rooted at the provided path.
    #[must_use]
    pub fn read_write(root: impl Into<PathBuf>) -> Self {
        Self::rooted(root, true)
    }

    fn rooted(root: impl Into<PathBuf>, allow_writes: bool) -> Self {
        Self {
            root: Some(root.into()),
            allow_writes,
        }
    }

    /// Returns the configured root directory.
    #[must_use]
    pub const fn root(&self) -> Option<&PathBuf> {
        self.root.as_ref()
    }

    /// Returns whether write operations are permitted.
    #[must_use]
    pub const fn allow_writes(&self) -> bool {
        self.allow_writes
    }
}

impl ToolingConfig {
    /// Creates a new tooling configuration.
    #[must_use]
    pub const fn new(filesystem: Option<FileSystemAccess>, enable_shell: bool) -> Self {
        Self {
            filesystem,
            enable_shell,
        }
    }

    /// Disables all additional tooling.
    #[must_use]
    pub const fn none() -> Self {
        Self::new(None, false)
    }

    /// Enables tooling suited for coding workflows.
    #[must_use]
    pub const fn coder() -> Self {
        Self::new(Some(FileSystemAccess::working_directory(true)), true)
    }

    /// Returns the optional filesystem access.
    #[must_use]
    pub const fn filesystem(&self) -> Option<&FileSystemAccess> {
        self.filesystem.as_ref()
    }

    /// Sets the filesystem access.
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_filesystem(&mut self, access: Option<FileSystemAccess>) {
        self.filesystem = access;
    }

    /// Enables or disables shell access.
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_enable_shell(&mut self, enable: bool) {
        self.enable_shell = enable;
    }

    /// Returns whether shell tooling is enabled.
    #[must_use]
    pub const fn enable_shell(&self) -> bool {
        self.enable_shell
    }
}

impl Default for ToolingConfig {
    fn default() -> Self {
        Self::none()
    }
}

/// An autonomous agent that plans and executes actions to achieve goals using a language model.
#[derive(Debug)]
pub struct Agent<LLM, PlannerImpl, ExecutorImpl> {
    llm: LLM,
    planner: PlannerImpl,
    executor: ExecutorImpl,
    config: AgentConfig,
    state: AgentState,
    tools_configured: bool,
}

impl<LLM> Agent<LLM, plan::DefaultPlanner, execute::DefaultExecutor>
where
    LLM: LanguageModel,
{
    /// Creates a new agent with default configuration.
    pub fn new(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::default())
    }

    /// Creates a new agent with the specified configuration.
    pub fn with_config(llm: LLM, config: AgentConfig) -> Self {
        Self::custom(llm, plan::DefaultPlanner, execute::DefaultExecutor, config)
    }
}

impl<LLM, PlannerImpl, ExecutorImpl> Agent<LLM, PlannerImpl, ExecutorImpl>
where
    LLM: LanguageModel,
    PlannerImpl: Planner,
    ExecutorImpl: Executor,
{
    /// Creates an agent with custom planner and executor implementations.
    pub fn custom(
        llm: LLM,
        planner: PlannerImpl,
        executor: ExecutorImpl,
        config: AgentConfig,
    ) -> Self {
        Self {
            llm,
            planner,
            executor,
            config,
            state: AgentState::default(),
            tools_configured: false,
        }
    }

    /// Runs the agent to achieve the specified goal.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent exceeds max iterations or planning/execution fails.
    pub async fn run(&mut self, goal: &str) -> aither_core::Result<String> {
        self.ensure_initialized().await;
        if self.state.memory.is_empty() {
            self.state.memory.push(Message::user(goal));
        } else {
            self.state
                .memory
                .push(Message::user(format!("New goal: {goal}")));
        }

        for _ in 0..self.config.max_iterations() {
            self.config
                .context()
                .maintain(&self.llm, &mut self.state.tools, &mut self.state.memory)
                .await?;

            let plan_outcome = self.planner.plan(&self.llm, &mut self.state, goal).await?;

            match plan_outcome {
                PlanOutcome::Completed(result) => {
                    self.state.memory.push(Message::assistant(result.clone()));
                    return Ok(result);
                }
                PlanOutcome::NeedsMoreSteps(todo) => {
                    self.state.attach_todo(todo);
                    self.execute_pending_steps().await?;
                }
            }
        }

        Err(anyhow!(
            "agent hit {} iterations without converging",
            self.config.max_iterations()
        ))
    }

    async fn ensure_initialized(&mut self) {
        if self.state.profile().is_none() {
            let profile = self.llm.profile().await;
            self.state.set_profile(profile);
        }
        if !self.tools_configured {
            self.bootstrap_tools();
            self.tools_configured = true;
        }
    }

    #[allow(clippy::missing_const_for_fn, clippy::needless_pass_by_ref_mut)]
    fn bootstrap_tools(&mut self) {
        #[cfg(feature = "filesystem")]
        if let Some(fs) = self.config.tooling().filesystem() {
            let root = fs
                .root()
                .cloned()
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            self.mount_filesystem(root, fs.allow_writes());
        }
        #[cfg(feature = "command")]
        if self.config.tooling().enable_shell() {
            self.enable_shell(None);
        }
    }

    async fn execute_pending_steps(&mut self) -> aither_core::Result<()> {
        while let Some((index, description)) = self.next_pending_step() {
            self.state
                .memory
                .push(Message::assistant(format!("Executing: {description}")));

            let result = self.executor.execute(&description, &self.state).await?;
            self.state
                .memory
                .push(Message::assistant(format!("Result: {result}")));

            self.request_todo_update(index, &description, &result).await?;

            if self.is_step_pending(&description) {
                break;
            }
        }

        Ok(())
    }

    fn next_pending_step(&self) -> Option<(usize, String)> {
        next_pending(self.state.tools())
    }

    fn is_step_pending(&self, description: &str) -> bool {
        self.state.tools().get::<TodoList>().is_some_and(|todo| {
            todo.pending()
                .any(|item| item.description() == description)
        })
    }

    async fn request_todo_update(
        &mut self,
        index: usize,
        description: &str,
        result: &str,
    ) -> aither_core::Result<()> {
        let mut parameters = Parameters::default();
        parameters.tool_choice = Some(vec![TodoList::TOOL_NAME.to_string()]);

        let mut messages = vec![Message::system(
            "Update the todo list only by calling the todo_list_updater tool. Do not mark tasks complete in plain text.",
        )];
        messages.extend(self.state.memory.all());
        messages.push(Message::assistant(format!(
            "Task {index} ({description}) finished with result: {result}. Call the todo_list_updater tool to record progress and keep the list accurate."
        )));

        let request = LLMRequest::new(messages)
            .with_parameters(parameters)
            .with_tools(self.state.tools_mut());

        let _ = self.llm.respond(request).into_future().await?;
        Ok(())
    }

    /// Returns the model profile if available.
    pub const fn profile(&self) -> Option<&ModelProfile> {
        self.state.profile()
    }

    /// Checks if the model has a native ability.
    pub fn has_native_ability(&self, ability: Ability) -> bool {
        self.state.has_native_ability(ability)
    }

    /// Registers a tool for the agent to use.
    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.state.register_tool(tool);
    }

    /// Mounts a filesystem tool with the specified root directory.
    #[cfg(feature = "filesystem")]
    pub fn mount_filesystem(&mut self, root: impl Into<PathBuf>, allow_writes: bool) {
        let tool = if allow_writes {
            filesystem::FileSystemTool::new(root.into())
        } else {
            filesystem::FileSystemTool::read_only(root.into())
        };
        self.register_tool(tool);
    }

    /// Enables command execution with optional restrictions.
    #[cfg(feature = "command")]
    pub fn enable_shell(&mut self, allowed_commands: impl Into<Option<Vec<String>>>) {
        let mut tool = command::CommandTool::new(
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        );
        if let Some(commands) = allowed_commands.into() {
            tool = tool.restrict_to(commands);
        }
        self.register_tool(tool);
    }

    /// Enables web search with the specified provider.
    #[cfg(feature = "websearch")]
    pub fn enable_websearch<P>(&mut self, provider: P)
    where
        P: websearch::SearchProvider + 'static,
    {
        if self.has_native_ability(Ability::WebSearch) {
            return;
        }
        self.register_tool(websearch::WebSearchTool::new(provider));
    }

    /// Converts this agent into a sub-agent with the specified name and description.
    pub fn into_subagent(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> SubAgent<LLM, PlannerImpl, ExecutorImpl> {
        SubAgent::new(self, name, description)
    }
}

fn next_pending(tools: &Tools) -> Option<(usize, String)> {
    tools.get::<TodoList>().and_then(|todo| {
        todo.items()
            .iter()
            .enumerate()
            .find(|(_, item)| !item.completed())
            .map(|(idx, item)| (idx, item.description().to_string()))
    })
}

/// State container for agent tools, memory, and model profile.
#[derive(Debug, Default)]
pub struct AgentState {
    /// Tools registered for the agent.
    tools: Tools,
    /// Conversation memory.
    memory: ConversationMemory,
    profile: Option<ModelProfile>,
}

impl AgentState {
    /// Attaches a todo list to the agent state and registers it as a tool.
    pub fn attach_todo(&mut self, todo: TodoList) {
        self.register_tool(todo);
    }

    /// Returns all messages in the conversation memory.
    #[must_use]
    pub fn messages(&self) -> Vec<Message> {
        self.memory.all()
    }

    /// Registers a tool for the agent to use.
    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.register(tool);
    }

    /// Returns a reference to the tools.
    #[must_use]
    pub const fn tools(&self) -> &Tools {
        &self.tools
    }

    /// Returns a mutable reference to the tools.
    pub const fn tools_mut(&mut self) -> &mut Tools {
        &mut self.tools
    }

    /// Returns the model profile if available.
    #[must_use]
    pub const fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    /// Sets the model profile.
    pub fn set_profile(&mut self, profile: ModelProfile) {
        self.profile = Some(profile);
    }

    /// Checks if the model has a native ability.
    #[must_use]
    pub fn has_native_ability(&self, ability: Ability) -> bool {
        self.profile
            .as_ref()
            .is_some_and(|profile| profile.abilities.contains(&ability))
    }

    /// Returns a summary of available capabilities including model abilities and tools.
    #[must_use]
    pub fn capability_summary(&self) -> Option<String> {
        let mut sections = Vec::new();
        if let Some(profile) = &self.profile {
            if !profile.abilities.is_empty() {
                let abilities = profile
                    .abilities
                    .iter()
                    .map(|ability| ability_label(*ability))
                    .collect::<Vec<_>>()
                    .join(", ");
                sections.push(format!("Model native abilities: {abilities}."));
            }
        }
        let tool_defs = self.tools.definitions();
        if !tool_defs.is_empty() {
            let tools = tool_defs
                .iter()
                .map(|definition| format!("{} - {}", definition.name(), definition.description()))
                .collect::<Vec<_>>()
                .join("; ");
            sections.push(format!("Tools available: {tools}."));
        }
        if sections.is_empty() {
            None
        } else {
            Some(sections.join(" "))
        }
    }
}

const fn ability_label(ability: Ability) -> &'static str {
    match ability {
        Ability::ToolUse => "tool_use",
        Ability::Vision => "vision",
        Ability::Audio => "audio",
        Ability::WebSearch => "web_search",
        Ability::Pdf => "pdf",
    }
}
