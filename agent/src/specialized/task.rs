//! Task tool for spawning specialized subagents.
//!
//! Allows the main agent to delegate complex tasks to specialized subagents
//! that run autonomously and return results.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use aither_core::{LanguageModel, llm::Tool};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::Agent;
use crate::hook::{Hook, PostToolAction, PreToolAction, ToolResultContext, ToolUseContext};

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
pub type SubagentBuilder<LLM> = Arc<dyn Fn(LLM) -> AgentBuilder<LLM, ()> + Send + Sync>;

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
        F: Fn(LLM) -> AgentBuilder<LLM, ()> + Send + Sync + 'static,
    {
        Self {
            description: description.into(),
            builder: Arc::new(builder),
        }
    }

    /// Get the agent builder for this subagent type.
    pub fn builder(&self, llm: LLM) -> AgentBuilder<LLM, ()> {
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
    /// The type of specialized subagent to use.
    /// Available types depend on configuration (e.g., "explore", "plan").
    pub subagent_type: String,
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

    fn update_description(&mut self) {
        let type_info: Vec<(&str, &str)> = self
            .types
            .iter()
            .map(|(name, t)| (name.as_str(), t.description.as_str()))
            .collect();
        self.description = Self::build_description(&type_info);
    }

    fn build_description(types: &[(&str, &str)]) -> String {
        if types.is_empty() {
            return "Launch a specialized subagent to handle complex tasks.\n\n\
                No subagent types registered."
                .to_string();
        }

        let mut desc = String::from(
            "Launch a specialized subagent to handle complex tasks autonomously.\n\n\
            Available subagent types:\n",
        );

        for (name, type_desc) in types {
            desc.push_str(&format!("- {name}: {type_desc}\n"));
        }

        desc.push_str(
            "\nThe subagent will work independently using its specialized tools \
            and return a final result.",
        );
        desc
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

    async fn call(&self, args: Self::Arguments) -> aither_core::Result {
        let subagent_type = self.types.get(&args.subagent_type).ok_or_else(|| {
            let available: Vec<&str> = self.types.keys().map(|s| s.as_str()).collect();
            anyhow::anyhow!(
                "Unknown subagent type '{}'. Available: {}",
                args.subagent_type,
                available.join(", ")
            )
        })?;

        // Log subagent start with user-visible feedback
        tracing::info!(
            subagent = %args.subagent_type,
            description = %args.description,
            "Starting subagent"
        );
        eprintln!(
            "\x1b[90m[subagent:{}] {}\x1b[0m",
            args.subagent_type, args.description
        );

        // Build the subagent with display hook for real-time feedback
        let display_hook = SubagentDisplayHook::new(&args.subagent_type);
        let mut agent = subagent_type
            .builder(self.llm.clone())
            .hook(display_hook)
            .build();

        // Run the subagent with the prompt
        let result = agent
            .query(&args.prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Subagent '{}' error: {e}", args.subagent_type))?;

        tracing::info!(
            subagent = %args.subagent_type,
            "Subagent completed"
        );
        eprintln!(
            "\x1b[32m[subagent:{}] done\x1b[0m",
            args.subagent_type
        );

        Ok(format!(
            "[Subagent '{}' completed: {}]\n\n{}",
            args.subagent_type, args.description, result
        ))
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
