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

pub mod execute;
pub mod memory;
pub mod plan;
pub mod sub_agent;
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
    pub context: ContextStrategy,
    pub max_iterations: usize,
    pub mode: AgentMode,
}

impl AgentConfig {
    pub fn companion() -> Self {
        Self {
            context: ContextStrategy::Unlimited,
            max_iterations: 64,
            mode: AgentMode::Companion,
        }
    }

    pub fn coder() -> Self {
        Self {
            context: ContextStrategy::Summarize {
                max_messages: 48,
                retain_recent: 12,
                instructions: "Keep file paths, commands, and compiler diagnostics verbatim."
                    .into(),
            },
            max_iterations: 64,
            mode: AgentMode::Coder,
        }
    }

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
    Generic,
    Companion,
    Coder,
    KnowledgeBase { root: PathBuf },
}

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
    pub fn new(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::default())
    }

    pub fn with_config(llm: LLM, config: AgentConfig) -> Self {
        Self::custom(llm, plan::DefaultPlanner, execute::DefaultExecutor, config)
    }

    pub fn companion(llm: LLM) -> Self {
        Self::with_config(llm, AgentConfig::companion())
    }

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

    fn bootstrap_mode_tools(&mut self) {
        match &self.config.mode {
            AgentMode::Generic | AgentMode::Companion => {}
            AgentMode::Coder => {
                #[cfg(feature = "filesystem")]
                {
                    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                    self.mount_filesystem(root, true);
                }
                #[cfg(feature = "cli")]
                {
                    self.enable_shell(None);
                }
            }
            AgentMode::KnowledgeBase { root } => {
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

    pub fn profile(&self) -> Option<&ModelProfile> {
        self.state.profile()
    }

    pub fn has_native_ability(&self, ability: Ability) -> bool {
        self.state.has_native_ability(ability)
    }

    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.state.register_tool(tool);
    }

    #[cfg(feature = "filesystem")]
    pub fn mount_filesystem(&mut self, root: impl Into<PathBuf>, allow_writes: bool) {
        let tool = if allow_writes {
            filesystem::FileSystemTool::new(root.into())
        } else {
            filesystem::FileSystemTool::read_only(root.into())
        };
        self.register_tool(tool);
    }

    #[cfg(feature = "cli")]
    pub fn enable_shell(&mut self, allowed_commands: impl Into<Option<Vec<String>>>) {
        let mut tool =
            cli::CommandTool::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        if let Some(commands) = allowed_commands.into() {
            tool = tool.restrict_to(commands);
        }
        self.register_tool(tool);
    }

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

#[derive(Debug, Default)]
pub struct AgentState {
    pub tools: Tools,
    pub todo: Option<SharedTodoList>,
    pub memory: ConversationMemory,
    profile: Option<ModelProfile>,
}

impl AgentState {
    pub fn attach_todo(&mut self, todo: TodoList) {
        let shared = SharedTodoList::new(todo);
        self.register_tool(shared.clone());
        self.todo = Some(shared);
    }

    pub fn messages(&self) -> Vec<Message> {
        self.memory.all()
    }

    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.register(tool);
    }

    pub fn tools(&self) -> &Tools {
        &self.tools
    }

    pub fn tools_mut(&mut self) -> &mut Tools {
        &mut self.tools
    }

    pub fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    pub fn set_profile(&mut self, profile: ModelProfile) {
        self.profile = Some(profile);
    }

    pub fn has_native_ability(&self, ability: Ability) -> bool {
        self.profile
            .as_ref()
            .map_or(false, |profile| profile.abilities.contains(&ability))
    }

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

fn ability_label(ability: Ability) -> &'static str {
    match ability {
        Ability::ToolUse => "tool_use",
        Ability::Vision => "vision",
        Ability::Audio => "audio",
        Ability::WebSearch => "web_search",
        Ability::Pdf => "pdf",
    }
}
