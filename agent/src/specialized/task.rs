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

use aither_core::{LanguageModel, llm::{Tool, ToolOutput}};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::hook::{Hook, PostToolAction, PreToolAction, ToolResultContext, ToolUseContext};
use crate::subagent_file::SubagentDefinition;

/// Hook that displays subagent activity to stderr.
pub struct SubagentDisplayHook {
    prefix: String,
}

impl SubagentDisplayHook {
    /// Create a new display hook with a prefix for output.
    pub fn new(subagent_type: &str) -> Self {
        Self {
            prefix: format!("  \x1b[90m[{}]\x1b[0m", subagent_type),
        }
    }
}

impl Hook for SubagentDisplayHook {
    async fn pre_tool_use(&self, ctx: &ToolUseContext<'_>) -> PreToolAction {
        // Show tool being called with arguments preview
        let args_preview: String = ctx.arguments.chars().take(60).collect();
        let args_display = if ctx.arguments.len() > 60 {
            format!("{}...", args_preview)
        } else {
            args_preview
        };
        eprintln!(
            "{} \x1b[36m{}\x1b[0m \x1b[90m{}\x1b[0m",
            self.prefix, ctx.tool_name, args_display
        );
        PreToolAction::Allow
    }

    async fn post_tool_use(&self, ctx: &ToolResultContext<'_>) -> PostToolAction {
        // Show result status with error details if failed
        match ctx.result {
            Ok(_) => {
                eprintln!(
                    "{} \x1b[32m✓\x1b[0m {} \x1b[90m({:?})\x1b[0m",
                    self.prefix, ctx.tool_name, ctx.duration
                );
            }
            Err(err) => {
                // Show error message for failures
                let err_preview: String = err.chars().take(80).collect();
                eprintln!(
                    "{} \x1b[31m✗\x1b[0m {} \x1b[31m{}\x1b[0m",
                    self.prefix, ctx.tool_name, err_preview
                );
            }
        }
        PostToolAction::Keep
    }
}

use crate::AgentBuilder;

/// Builder function type for configuring a subagent.
/// Returns an AgentBuilder so we can add hooks before building.
/// The builder returns a single-model agent (all tiers use the same LLM).
pub type SubagentBuilder<LLM> = Arc<dyn Fn(LLM) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync>;

/// Configuration for a subagent type.
pub struct SubagentType<LLM> {
    /// Description shown to the main agent.
    pub description: String,
    /// Builder function that creates the configured agent builder.
    builder: SubagentBuilder<LLM>,
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

/// Arguments for the Task tool.
#[derive(Debug, Clone, JsonSchema, Deserialize)]
pub struct TaskArgs {
    /// Short description of the task (3-5 words).
    pub description: String,
    /// The detailed task prompt for the subagent.
    pub prompt: String,
    /// The type of specialized subagent to use (e.g., "explore", "plan").
    /// Either `subagent_type` or `subagent_file` must be provided.
    #[serde(default)]
    pub subagent_type: Option<String>,
    /// Path to a subagent definition file (markdown format).
    /// Use this to launch a custom subagent not in the registry.
    #[serde(default)]
    pub subagent_file: Option<String>,
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
    description: String,
}

impl<LLM: Clone> TaskTool<LLM> {
    /// Creates a new TaskTool with the given LLM.
    pub fn new(llm: LLM) -> Self {
        Self {
            llm,
            types: HashMap::new(),
            description: Self::build_description(&[]),
        }
    }

    /// Register a subagent type.
    pub fn register(&mut self, name: impl Into<String>, subagent: SubagentType<LLM>) {
        let name = name.into();
        self.types.insert(name, subagent);
        self.update_description();
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

    fn update_description(&mut self) {
        let type_info: Vec<(&str, &str)> = self
            .types
            .iter()
            .map(|(name, t)| (name.as_str(), t.description.as_str()))
            .collect();
        self.description = Self::build_description(&type_info);
    }

    fn build_description(types: &[(&str, &str)]) -> String {
        const TEMPLATE: &str = include_str!("../prompts/task.md");

        if types.is_empty() {
            return TEMPLATE.replace("{{types}}", "No subagent types registered.");
        }

        let mut types_list = String::new();
        for (name, type_desc) in types {
            types_list.push_str(&format!("- {name}: {type_desc}\n"));
        }

        TEMPLATE.replace("{{types}}", &types_list)
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
        self.with_type(def.id.clone(), subagent)
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

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = TaskArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        // Determine how to create the subagent: from registered type or file
        let (subagent_id, agent_builder) = match (&args.subagent_type, &args.subagent_file) {
            (Some(type_name), None) => {
                // Use registered subagent type
                let subagent_type = self.types.get(type_name).ok_or_else(|| {
                    let available: Vec<&str> = self.types.keys().map(|s| s.as_str()).collect();
                    anyhow::anyhow!(
                        "Unknown subagent type '{}'. Available: {}",
                        type_name,
                        available.join(", ")
                    )
                })?;
                (type_name.clone(), subagent_type.builder(self.llm.clone()))
            }
            (None, Some(file_path)) => {
                // Load subagent from file
                let path = Path::new(file_path);
                let def = SubagentDefinition::from_file(path)
                    .map_err(|e| anyhow::anyhow!("Failed to read subagent file '{}': {e}", file_path))?
                    .ok_or_else(|| anyhow::anyhow!("Invalid subagent definition in '{}'", file_path))?;

                let builder = crate::AgentBuilder::new(self.llm.clone())
                    .system_prompt(&def.system_prompt)
                    .max_iterations(def.max_iterations);

                (def.id, builder)
            }
            (Some(_), Some(_)) => {
                return Err(anyhow::anyhow!(
                    "Cannot specify both 'subagent_type' and 'subagent_file'. Choose one."
                ).into());
            }
            (None, None) => {
                let available: Vec<&str> = self.types.keys().map(|s| s.as_str()).collect();
                return Err(anyhow::anyhow!(
                    "Must specify either 'subagent_type' or 'subagent_file'. Available types: {}",
                    available.join(", ")
                ).into());
            }
        };

        // Log subagent start with user-visible feedback
        tracing::info!(
            subagent = %subagent_id,
            description = %args.description,
            "Starting subagent"
        );
        eprintln!(
            "\x1b[90m[subagent:{}] {}\x1b[0m",
            subagent_id, args.description
        );

        // Build the subagent with display hook for real-time feedback
        let display_hook = SubagentDisplayHook::new(&subagent_id);

        // Add child bash tool if factory is configured
        let agent_builder = if let Some(dyn_bash) = aither_sandbox::create_child_bash_tool() {
            agent_builder.dyn_bash(dyn_bash)
        } else {
            agent_builder
        };

        let mut agent = agent_builder.hook(display_hook).build();

        // Run the subagent with the prompt
        let result = agent
            .query(&args.prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Subagent '{}' error: {e}", subagent_id))?;

        tracing::info!(
            subagent = %subagent_id,
            "Subagent completed"
        );
        eprintln!(
            "\x1b[32m[subagent:{}] done\x1b[0m",
            subagent_id
        );

        Ok(ToolOutput::text(format!(
            "[Subagent '{}' completed: {}]\n\n{}",
            subagent_id, args.description, result
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
        assert!(props.get("description").is_some());
        assert!(props.get("prompt").is_some());
        assert!(props.get("subagent_type").is_some());
    }
}
