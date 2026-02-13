//! Task tool for spawning specialized subagents.
//!
//! Allows the main agent to delegate complex tasks to specialized subagents
//! that run autonomously and return results.
//!
//! Subagents can be defined in files or registered programmatically.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{Tool, ToolOutput},
};
use aither_sandbox::BashToolFactory;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::AgentBuilder;
use crate::fs_util::path_exists;
use crate::subagent_file::SubagentDefinition;

async fn checked_path_exists(path: &Path) -> anyhow::Result<bool> {
    path_exists(path)
        .await
        .map_err(|error| anyhow::anyhow!("failed to inspect path '{}': {error}", path.display()))
}

/// Builder function type for configuring a subagent.
/// Returns an `AgentBuilder` so we can add hooks before building.
/// The builder returns a single-model agent (all tiers use the same LLM).
pub type SubagentBuilder<LLM> = Arc<dyn Fn(LLM) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync>;

/// Configuration for a subagent type.
pub struct SubagentType<LLM> {
    /// Description shown to the main agent.
    pub description: String,
    /// Builder function that creates the configured agent builder.
    builder: SubagentBuilder<LLM>,
}

impl<LLM> std::fmt::Debug for SubagentType<LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentType")
            .field("description", &self.description)
            .finish()
    }
}

impl<LLM: Clone> SubagentType<LLM> {
    /// Create a new subagent type with a builder function.
    ///
    /// The builder function should return an `AgentBuilder` (not a built `Agent`)
    /// so that hooks can be added before building.
    pub fn new<F>(description: impl Into<String>, builder: F) -> Self
    where
        F: Fn(LLM) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync + 'static,
    {
        Self {
            description: description.into(),
            builder: Arc::new(builder),
        }
    }

    /// Get the agent builder for this subagent type.
    pub fn builder(&self, llm: LLM) -> AgentBuilder<LLM, LLM, LLM, ()> {
        (self.builder)(llm)
    }
}

/// Launch a specialized subagent for complex tasks.
///
/// Subagents are autonomous agents that run in their own context window.
/// Their work does not consume your context, and they return only their
/// final results. This enables:
///
/// - Context isolation for research-heavy tasks
/// - Parallel execution of independent investigations
/// - Specialized capabilities per subagent type
///
/// Each subagent type has focused tools and prompts optimized for its purpose.
/// Choose the subagent type that best matches the task requirements.
///
/// You can also load custom subagents from `.md` files by specifying a path
/// (e.g., `./custom-agent.md` or `.subagents/researcher.md`).
#[derive(Debug, Clone, JsonSchema, Deserialize)]
pub struct TaskArgs {
    /// Subagent type name (e.g., "explore", "plan") or path to a subagent file.
    /// If the value contains '/' or ends with '.md', it's treated as a file path.
    pub subagent: String,
    /// The detailed task prompt for the subagent.
    pub prompt: String,
}

/// A tool that spawns specialized subagents to handle complex tasks.
///
/// Similar to Claude Code's Task tool, this allows the main agent to
/// delegate work to specialized subagents that have different capabilities.
///
/// # Example
///
/// ```rust,ignore
/// use aither_agent::specialized::task::{TaskTool, SubagentType};
///
/// // Create task tool
/// let mut task_tool = TaskTool::new(llm.clone());
///
/// // Register an "explore" subagent type
/// task_tool.register("explore", SubagentType::new(
///     "Codebase explorer for finding files and searching code",
///     |llm| Agent::builder(llm)
///         .system_prompt("You are a codebase explorer...")
///         .tool(FsTool::new())
///         .max_iterations(15)
///         .build()
/// ));
///
/// let agent = Agent::builder(llm)
///     .tool(task_tool)
///     .build();
/// ```
pub struct TaskTool<LLM> {
    llm: LLM,
    types: HashMap<String, SubagentType<LLM>>,
    /// Base directory for resolving relative paths (e.g., sandbox directory).
    base_dir: Option<std::path::PathBuf>,
    /// Factory for creating child bash tools for subagents.
    bash_tool_factory: Option<BashToolFactory>,
}

impl<LLM> std::fmt::Debug for TaskTool<LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_names: Vec<_> = self.types.keys().cloned().collect();
        f.debug_struct("TaskTool")
            .field("type_names", &type_names)
            .field("base_dir", &self.base_dir)
            .finish()
    }
}

impl<LLM: Clone> TaskTool<LLM> {
    /// Creates a new `TaskTool` with the given LLM.
    pub fn new(llm: LLM) -> Self {
        Self {
            llm,
            types: HashMap::new(),
            base_dir: None,
            bash_tool_factory: None,
        }
    }

    /// Sets the base directory for resolving relative paths.
    ///
    /// Paths like `.subagents/...` and `.skills/...` will be resolved
    /// relative to this directory.
    #[must_use]
    pub fn with_base_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    /// Sets the bash tool factory used for subagents.
    #[must_use]
    pub fn with_bash_tool_factory(mut self, factory: BashToolFactory) -> Self {
        self.bash_tool_factory = Some(factory);
        self
    }

    /// Register a subagent type.
    pub fn register(&mut self, name: impl Into<String>, subagent: SubagentType<LLM>) {
        let name = name.into();
        self.types.insert(name, subagent);
    }

    /// Register a subagent type (builder pattern).
    #[must_use]
    pub fn with_type(mut self, name: impl Into<String>, subagent: SubagentType<LLM>) -> Self {
        self.register(name, subagent);
        self
    }

    /// Returns a list of registered subagent types with their descriptions.
    ///
    /// Useful for injecting into system prompts so the main agent knows
    /// what subagents are available.
    pub fn type_descriptions(&self) -> Vec<(&str, &str)> {
        self.types
            .iter()
            .map(|(name, t)| (name.as_str(), t.description.as_str()))
            .collect()
    }
}

impl<LLM: LanguageModel + Clone> TaskTool<LLM> {
    /// Register a subagent from a definition (builder pattern).
    ///
    /// Creates a subagent type from a `SubagentDefinition` loaded from a file.
    /// The subagent will use the definition's system prompt and max iterations.
    #[must_use]
    pub fn with_definition(self, def: SubagentDefinition) -> Self {
        let subagent = SubagentType::new(def.description.clone(), move |llm| {
            crate::AgentBuilder::new(llm)
                .system_prompt(&def.system_prompt)
                .max_iterations(def.max_iterations)
        });
        self.with_type(def.id, subagent)
    }

    /// Register all builtin subagents.
    ///
    /// Loads subagents from embedded definition files (explore, plan, etc.).
    #[must_use]
    pub fn with_builtins(mut self) -> Self {
        for def in crate::builtin_subagents() {
            self = self.with_definition(def);
        }
        self
    }
}

impl<LLM> Tool for TaskTool<LLM>
where
    LLM: LanguageModel + Clone + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("task")
    }

    type Arguments = TaskArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        // Determine if subagent is a file path or a registered type name
        // File paths contain '/' or end with '.md'
        let is_file_path = args.subagent.contains('/') || args.subagent.ends_with(".md");

        let (subagent_id, agent_builder) = if is_file_path {
            // Load subagent from file
            // Resolve paths relative to base_dir if set (e.g., sandbox directory)
            let file_path = &args.subagent;
            let base = self.base_dir.as_deref().unwrap_or(Path::new("."));

            // Try paths in order:
            // 1. As-is relative to base_dir
            // 2. Under .subagents/ relative to base_dir
            // 3. Under .skills/ relative to base_dir
            let path = base.join(file_path);
            let resolved_path = if checked_path_exists(&path).await? {
                path
            } else {
                let subagents_path = base.join(".subagents").join(file_path);
                if checked_path_exists(&subagents_path).await? {
                    subagents_path
                } else {
                    let skills_path = base.join(".skills").join(file_path);
                    if checked_path_exists(&skills_path).await? {
                        skills_path
                    } else {
                        // Return original path for error message
                        path
                    }
                }
            };

            let def = SubagentDefinition::from_file_async(&resolved_path)
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to read subagent file '{}': {e}",
                        resolved_path.display()
                    )
                })?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Invalid subagent definition in '{}'",
                        resolved_path.display()
                    )
                })?;

            let builder = crate::AgentBuilder::new(self.llm.clone())
                .system_prompt(&def.system_prompt)
                .max_iterations(def.max_iterations);

            (def.id, builder)
        } else {
            // Use registered subagent type
            let type_name = &args.subagent;
            let subagent_type = self.types.get(type_name).ok_or_else(|| {
                let available: Vec<&str> =
                    self.types.keys().map(std::string::String::as_str).collect();
                anyhow::anyhow!(
                    "Unknown subagent type '{}'. Available: {}",
                    type_name,
                    available.join(", ")
                )
            })?;
            (type_name.clone(), subagent_type.builder(self.llm.clone()))
        };

        tracing::info!(subagent = %subagent_id, "Starting subagent");

        // Add child bash tool if factory is configured
        let agent_builder = if let Some(factory) = &self.bash_tool_factory {
            let dyn_bash = factory
                .create()
                .await
                .map_err(|e| anyhow::anyhow!("failed to create subagent bash tool: {e}"))?;
            agent_builder.dyn_bash(dyn_bash)
        } else {
            agent_builder
        };

        let mut agent = agent_builder.build();

        // Run the subagent with the prompt
        let result = agent
            .query(&args.prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Subagent '{subagent_id}' error: {e}"))?;

        tracing::info!(subagent = %subagent_id, "Subagent completed");

        Ok(ToolOutput::text(format!(
            "[Subagent '{subagent_id}' completed]\n\n{result}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_args_schema() {
        let schema = schemars::schema_for!(TaskArgs);
        let value = schema.to_value();
        let props = value.get("properties").expect("should have properties");
        assert!(props.get("subagent").is_some());
        assert!(props.get("prompt").is_some());

        // Check required order
        let required = value.get("required").expect("should have required");
        assert!(required.is_array());
    }
}
