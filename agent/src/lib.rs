//! Agent framework for building autonomous AI agents with planning, execution, and memory control.

use std::path::PathBuf;

use aither_core::{
    LanguageModel,
    llm::{
        Message, Tool,
        model::{Ability, Profile as ModelProfile},
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

/// Execution strategies for running plan steps.
pub mod execute;
/// Memory management and context compression.
pub mod memory;
/// Planning strategies for breaking down goals.
pub mod plan;
/// Sub-agent creation and management.
pub mod sub_agent;
/// Todo list management and tracking.
pub mod todo;

use crate::{
    execute::Executor,
    memory::{ContextStrategy, ConversationMemory},
    plan::{PlanOutcome, Planner},
    sub_agent::SubAgent,
    todo::{SharedTodoList, TodoList},
};

/// Agent-wide configuration describing context compression, iteration limits, and mode presets.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Strategy for managing conversation context and compression.
    pub context: ContextStrategy,
    /// Maximum number of planning iterations before giving up.
    pub max_iterations: usize,
    /// Operating mode that determines default tools and behavior.
    pub mode: AgentMode,
}

impl AgentConfig {
    /// Creates a companion agent configuration with unlimited context.
    #[must_use]
    pub const fn companion() -> Self {
        Self {
            context: ContextStrategy::Unlimited,
            max_iterations: 64,
            mode: AgentMode::Companion,
        }
    }

    /// Creates a coder agent configuration with context summarization.
    #[must_use]
    pub fn coder() -> Self {
        Self {
            context: ContextStrategy::Summarize {
                max_messages: 48,
                retain_recent: 12,
                instructions: "Keep file paths, commands, and compiler diagnostics verbatim."
                    .into(),
            },
            max_iterations: 64,
            mode: AgentMode::Coder(Coder),
        }
    }

    /// Creates a knowledge base agent configuration with filesystem access.
    pub fn knowledge_base(root: impl Into<PathBuf>) -> Self {
        Self {
            context: ContextStrategy::Summarize {
                max_messages: 60,
                retain_recent: 10,
                instructions: "Preserve cited facts, source filenames, and any numerical data."
                    .into(),
            },
            max_iterations: 80,
            mode: AgentMode::KnowledgeBase { root: root.into() },
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            context: ContextStrategy::SlidingWindow { max_messages: 32 },
            max_iterations: 48,
            mode: AgentMode::Generic,
        }
    }
}

/// Describes ready-to-use profiles that tweak defaults and tooling.
#[derive(Debug, Clone)]
pub enum AgentMode {
    /// Generic agent with no special tools or configuration.
    Generic,
    /// Companion agent for conversational interactions.
    Companion,
    /// Coder agent with filesystem and command execution capabilities.
    Coder(Coder),
    /// Knowledge base agent with read-only filesystem access.
    KnowledgeBase {
        /// Root directory for knowledge base access.
        root: PathBuf,
    },
}

/// Dedicated type representing the coder operating mode.
#[derive(Debug, Clone, Copy, Default)]
pub struct Coder;

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

    /// Creates a companion agent for conversational interactions.
    pub fn companion(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::companion())
    }

    /// Creates a coder agent with filesystem and command execution capabilities.
    pub fn coder(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::coder())
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

        for _ in 0..self.config.max_iterations {
            self.config
                .context
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
                    self.execute_pending_steps()?;
                }
            }
        }

        Err(anyhow!(
            "agent hit {} iterations without converging",
            self.config.max_iterations
        ))
    }

    async fn ensure_initialized(&mut self) {
        if self.state.profile().is_none() {
            let profile = self.llm.profile().await;
            self.state.set_profile(profile);
        }
        if !self.tools_configured {
            self.bootstrap_mode_tools();
            self.tools_configured = true;
        }
    }

    #[allow(clippy::missing_const_for_fn, clippy::needless_pass_by_ref_mut)]
    fn bootstrap_mode_tools(&mut self) {
        match &self.config.mode {
            AgentMode::Generic | AgentMode::Companion => {}
            AgentMode::Coder(_) => {
                #[cfg(feature = "filesystem")]
                {
                    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                    self.mount_filesystem(root, true);
                }
                #[cfg(feature = "command")]
                {
                    self.enable_shell(None);
                }
            }
            AgentMode::KnowledgeBase {
                #[cfg_attr(not(feature = "filesystem"), allow(unused_variables))]
                root,
            } => {
                #[cfg(feature = "filesystem")]
                {
                    self.mount_filesystem(root.clone(), false);
                }
            }
        }
    }

    fn execute_pending_steps(&mut self) -> aither_core::Result<()> {
        let Some(todo) = self.state.todo.clone() else {
            return Ok(());
        };

        while let Some((index, description)) = next_pending(&todo) {
            let step_memory = Message::assistant(format!("Executing: {description}"));
            self.state.memory.push(step_memory);

            let result = self.executor.execute(&description, &self.state)?;
            let completion = Message::assistant(format!("Result: {result}"));
            self.state.memory.push(completion);

            todo.with_mut(|list| list.tick(index));
        }

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

fn next_pending(todo: &SharedTodoList) -> Option<(usize, String)> {
    todo.with(|list| {
        list.items()
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
    pub tools: Tools,
    /// Optional todo list for tracking progress.
    pub todo: Option<SharedTodoList>,
    /// Conversation memory.
    pub memory: ConversationMemory,
    profile: Option<ModelProfile>,
}

impl AgentState {
    /// Attaches a todo list to the agent state and registers it as a tool.
    pub fn attach_todo(&mut self, todo: TodoList) {
        let shared = SharedTodoList::new(todo);
        self.register_tool(shared.clone());
        self.todo = Some(shared);
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
