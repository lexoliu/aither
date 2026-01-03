//! Coder - A specialized agent for coding tasks.
//!
//! Pre-configured with filesystem and shell access, optimized for
//! software development workflows.

use std::path::PathBuf;

use aither_core::LanguageModel;

use crate::{
    Agent, AgentBuilder, AgentConfig, ContextStrategy,
    compression::SmartCompressionConfig,
    hook::{HCons, Hook},
};

// System prompt for coding tasks
const CODER_SYSTEM_PROMPT: &str = include_str!("../prompts/coder.txt");

/// A specialized agent for coding tasks.
///
/// Pre-configured with:
/// - Filesystem access (read-write by default)
/// - Shell command execution
/// - Smart context compression
/// - Coding-focused system prompt
///
/// # Example
///
/// ```rust,ignore
/// let coder = Coder::new(claude);
/// coder.query("Add error handling to auth.rs").await?;
/// ```
#[derive(Debug)]
pub struct Coder<LLM, H = ()> {
    agent: Agent<LLM, H>,
}

impl<LLM: LanguageModel> Coder<LLM, ()> {
    /// Creates a new coder agent with default configuration.
    #[must_use]
    pub fn new(llm: LLM) -> Self {
        Self::builder(llm).build()
    }

    /// Returns a builder for customizing the coder.
    #[must_use]
    pub fn builder(llm: LLM) -> CoderBuilder<LLM, ()> {
        CoderBuilder::new(llm)
    }
}

impl<LLM, H> Coder<LLM, H>
where
    LLM: LanguageModel,
    H: Hook,
{
    /// Performs a coding task.
    ///
    /// # Errors
    ///
    /// Returns an error if the task fails.
    pub async fn query(&mut self, task: &str) -> Result<String, crate::AgentError> {
        self.agent.query(task).await
    }

    /// Returns a reference to the underlying agent.
    #[must_use]
    pub const fn agent(&self) -> &Agent<LLM, H> {
        &self.agent
    }

    /// Returns a mutable reference to the underlying agent.
    pub fn agent_mut(&mut self) -> &mut Agent<LLM, H> {
        &mut self.agent
    }
}

/// Builder for constructing a Coder agent.
#[must_use]
pub struct CoderBuilder<LLM, H = ()> {
    builder: AgentBuilder<LLM, H>,
    root: Option<PathBuf>,
    allow_shell: bool,
    read_only: bool,
}

impl<LLM: LanguageModel> CoderBuilder<LLM, ()> {
    /// Creates a new coder builder.
    pub fn new(llm: LLM) -> Self {
        Self {
            builder: Agent::builder(llm)
                .system_prompt(CODER_SYSTEM_PROMPT)
                .context_strategy(ContextStrategy::Smart(SmartCompressionConfig {
                    preserve_recent: 12,
                    ..SmartCompressionConfig::default()
                }))
                .max_iterations(64),
            root: None,
            allow_shell: true,
            read_only: false,
        }
    }
}

impl<LLM, H> CoderBuilder<LLM, H>
where
    LLM: LanguageModel,
    H: Hook,
{
    /// Sets the root directory for filesystem operations.
    ///
    /// Defaults to the current working directory.
    pub fn root(mut self, path: impl Into<PathBuf>) -> Self {
        self.root = Some(path.into());
        self
    }

    /// Disables shell command execution.
    pub fn no_shell(mut self) -> Self {
        self.allow_shell = false;
        self
    }

    /// Sets the filesystem to read-only mode.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Adds a hook to the coder.
    pub fn hook<NH: Hook>(self, hook: NH) -> CoderBuilder<LLM, HCons<NH, H>> {
        CoderBuilder {
            builder: self.builder.hook(hook),
            root: self.root,
            allow_shell: self.allow_shell,
            read_only: self.read_only,
        }
    }

    /// Builds the Coder agent.
    pub fn build(mut self) -> Coder<LLM, H> {
        let root = self
            .root
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        // Register filesystem tool
        #[cfg(feature = "filesystem")]
        {
            let fs_tool = if self.read_only {
                crate::filesystem::FileSystemTool::read_only(root.clone())
            } else {
                crate::filesystem::FileSystemTool::new(root.clone())
            };
            self.builder = self.builder.tool(fs_tool);
        }

        // Register command tool
        #[cfg(feature = "command")]
        if self.allow_shell {
            let cmd_tool = crate::command::CommandTool::new(root);
            self.builder = self.builder.tool(cmd_tool);
        }

        Coder {
            agent: self.builder.build(),
        }
    }
}
