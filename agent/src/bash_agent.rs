//! Bash-centric agent builder where all tools are IPC commands.
//!
//! This module provides a streamlined API for creating agents where:
//! - **LLM perspective**: `bash` is the only tool call entry point
//! - **Developer perspective**: Tools are registered normally but become bash commands
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         Agent                               │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │                    bash tool                         │   │
//! │  │  ┌─────────────────────────────────────────────┐    │   │
//! │  │  │              Sandbox (IPC)                   │    │   │
//! │  │  │  websearch | webfetch | todo | task | ...   │    │   │
//! │  │  └─────────────────────────────────────────────┘    │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use aither_agent::bash_agent::BashAgentBuilder;
//!
//! let agent = BashAgentBuilder::new(llm, bash_tool)
//!     .tool(WebSearchTool::default())  // Becomes `websearch` bash command
//!     .tool(WebFetchTool::new())       // Becomes `webfetch` bash command
//!     .tool(TaskTool::new(llm))        // Becomes `task` bash command
//!     .build();
//! ```

use std::borrow::Cow;

use aither_core::LanguageModel;
use aither_core::llm::Tool;
use aither_core::llm::tool::ToolDefinition;
use async_fs as fs;
use askama::Template;
use executor_core::Executor;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::hook::Hook;
use crate::{Agent, AgentBuilder};
use aither_sandbox::{
    BashTool, BashToolFactory, BashToolFactoryReceiver, PermissionHandler, ToolRegistryBuilder,
    Unconfigured, bash_tool_factory_channel,
};
use aither_sandbox::builtin::{StopTool, TasksTool};

/// System prompt template for bash-centric agents.
#[derive(Template)]
#[template(path = "system.txt", escape = "none")]
struct SystemPrompt {
    os: String,
    os_version: String,
    arch: &'static str,
    user_cwd: String,
    sandbox_dir: String,
    tools: String,
    skills: String,
    has_skills: bool,
    subagents: String,
    has_subagents: bool,
    is_macos: bool,
}

/// Loaded skill metadata for system prompt.
#[derive(Debug, Clone)]
pub struct SkillInfo {
    /// Name of the skill.
    pub name: String,
    /// Short description.
    pub description: String,
}

/// Loaded subagent metadata for system prompt.
#[derive(Debug, Clone)]
pub struct SubagentInfo {
    /// Name/ID of the subagent.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Relative path within .subagents folder.
    pub path: String,
}

/// Builder for creating bash-centric agents.
///
/// All registered tools become IPC commands accessible via bash.
/// The LLM only sees the `bash` tool.
pub struct BashAgentBuilder<LLM, P, E, H = ()>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    inner: AgentBuilder<LLM, LLM, LLM, H>,
    bash_tool: BashTool<P, E, Unconfigured>,
    registry_builder: ToolRegistryBuilder,
    bash_tool_factory: BashToolFactory,
    bash_tool_factory_receiver: Option<BashToolFactoryReceiver>,
    tool_descriptions: Vec<(String, String)>,
    skills: Vec<SkillInfo>,
    subagents: Vec<SubagentInfo>,
}

impl<LLM, P, E, H> std::fmt::Debug for BashAgentBuilder<LLM, P, E, H>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashAgentBuilder")
            .field("tool_count", &self.tool_descriptions.len())
            .field("skill_count", &self.skills.len())
            .field("subagent_count", &self.subagents.len())
            .finish()
    }
}

impl<LLM, P, E> BashAgentBuilder<LLM, P, E, ()>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    /// Creates a new bash-centric agent builder.
    ///
    /// The bash tool is the only direct tool available to the LLM.
    /// Use `.tool()` to register tools as bash commands.
    pub fn new(llm: LLM, bash_tool: BashTool<P, E, Unconfigured>) -> Self {
        let (bash_tool_factory, bash_tool_factory_receiver) = bash_tool_factory_channel();
        let mut registry_builder = ToolRegistryBuilder::new();
        let mut tool_descriptions = Vec::new();

        let job_registry = bash_tool.job_registry();
        let tasks_tool = TasksTool::new(job_registry.clone());
        let stop_tool = StopTool::new(job_registry);

        let tasks_def = ToolDefinition::new(&tasks_tool);
        tool_descriptions.push((
            tasks_def.name().to_string(),
            short_description(tasks_def.description()),
        ));
        registry_builder.configure_tool(tasks_tool);

        let stop_def = ToolDefinition::new(&stop_tool);
        tool_descriptions.push((
            stop_def.name().to_string(),
            short_description(stop_def.description()),
        ));
        registry_builder.configure_tool(stop_tool);

        Self {
            inner: AgentBuilder::new(llm),
            bash_tool,
            registry_builder,
            bash_tool_factory,
            bash_tool_factory_receiver: Some(bash_tool_factory_receiver),
            tool_descriptions,
            skills: Vec::new(),
            subagents: Vec::new(),
        }
    }
}

impl<LLM, P, E, H> BashAgentBuilder<LLM, P, E, H>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
    H: Hook,
{
    /// Registers a tool as an IPC command accessible via bash.
    ///
    /// The tool becomes a bash command with the same name.
    /// For example, `WebSearchTool` becomes the `websearch` command.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.tool(WebSearchTool::default())
    /// // LLM can now call: bash -c 'websearch "rust async"'
    /// ```
    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let name = tool.name().to_string();
        // Extract description from schema (rustdoc on Args struct)
        let def = ToolDefinition::new(&tool);
        let description = def.description();
        self.tool_descriptions
            .push((name, short_description(description)));
        self.registry_builder.configure_tool(tool);
        self
    }

    /// Registers a tool with a custom description.
    pub fn tool_with_desc<T>(mut self, tool: T, description: impl Into<String>) -> Self
    where
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let name = tool.name().to_string();
        self.tool_descriptions.push((name, description.into()));
        self.registry_builder.configure_tool(tool);
        self
    }

    /// Adds a pre-configured tool description (for tools registered elsewhere).
    ///
    /// Use this when the tool was already registered on the registry builder.
    pub fn tool_description(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.tool_descriptions
            .push((name.into(), description.into()));
        self
    }

    /// Sets a custom system prompt (raw string, no template processing).
    pub fn system_prompt_raw(mut self, prompt: impl Into<String>) -> Self {
        self.inner = self.inner.system_prompt(prompt.into());
        self
    }

    /// Generates and sets the default system prompt using the built-in template.
    ///
    /// This should be called after all tools are registered.
    pub fn with_default_prompt(mut self) -> Self {
        // Build tools description
        let tools = self
            .tool_descriptions
            .iter()
            .map(|(name, desc)| format!("- {name}: {desc}"))
            .collect::<Vec<_>>()
            .join("\n");

        // Build skills description
        let has_skills = !self.skills.is_empty();
        let skills = self
            .skills
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");

        // Build subagents description
        let has_subagents = !self.subagents.is_empty();
        let subagents = self
            .subagents
            .iter()
            .map(|s| format!("- {} ({}): {}", s.name, s.path, s.description))
            .collect::<Vec<_>>()
            .join("\n");

        // Get directory paths
        let sandbox_dir = self.bash_tool.working_dir().display().to_string();
        let user_cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());

        // Get system info
        let (os, os_version) = get_os_info();
        let arch = std::env::consts::ARCH;
        let is_macos = cfg!(target_os = "macos");

        let template = SystemPrompt {
            os,
            os_version,
            arch,
            user_cwd,
            sandbox_dir,
            tools,
            skills,
            has_skills,
            subagents,
            has_subagents,
            is_macos,
        };

        let prompt = template
            .render()
            .expect("failed to render system prompt template");
        self.inner = self.inner.system_prompt(prompt);
        self
    }

    /// Adds a hook to intercept agent operations.
    pub fn hook<NH: Hook>(self, hook: NH) -> BashAgentBuilder<LLM, P, E, crate::HCons<NH, H>> {
        BashAgentBuilder {
            inner: self.inner.hook(hook),
            bash_tool: self.bash_tool,
            registry_builder: self.registry_builder,
            bash_tool_factory: self.bash_tool_factory,
            bash_tool_factory_receiver: self.bash_tool_factory_receiver,
            tool_descriptions: self.tool_descriptions,
            skills: self.skills,
            subagents: self.subagents,
        }
    }

    /// Loads skills from a filesystem path and creates a symlink in the sandbox.
    ///
    /// Skills are loaded from SKILL.md files in subdirectories of the provided path.
    /// A symlink `.skills` is created in the sandbox working directory pointing to
    /// the skills directory, allowing the agent to read skill contents.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.with_skills("/path/to/skills")
    /// ```
    #[cfg(feature = "skills")]
    pub fn with_skills(mut self, path: impl AsRef<std::path::Path>) -> Self {
        use aither_skills::SkillLoader;

        let path = path.as_ref();
        if !path.exists() {
            return self;
        }

        // Load all skills from the path
        tracing::debug!(path = %path.display(), "Loading skills from path");
        let loader = SkillLoader::new().add_path(path);
        match loader.load_all() {
            Ok(skills) => {
                tracing::info!(count = skills.len(), path = %path.display(), "Loaded skills");
                for skill in skills {
                    tracing::debug!(name = %skill.name, "Loaded skill");
                    self.skills.push(SkillInfo {
                        name: skill.name,
                        description: skill.description,
                    });
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "Failed to load skills");
            }
        }

        // Create symlink to .skills in sandbox working directory
        // Canonicalize path to ensure symlink works from sandbox directory
        let symlink_path = self.bash_tool.working_dir().join(".skills");
        if let Ok(abs_path) = path.canonicalize() {
            let _ = std::os::unix::fs::symlink(&abs_path, &symlink_path);
        }

        self
    }

    /// Loads skills from a filesystem path asynchronously.
    #[cfg(feature = "skills")]
    pub async fn with_skills_async(mut self, path: impl AsRef<std::path::Path>) -> Self {
        use aither_skills::SkillLoader;

        let path = path.as_ref().to_path_buf();
        if !path_exists(&path).await {
            return self;
        }

        let loader = SkillLoader::new().add_path(&path);
        match loader.load_all_async().await {
            Ok(skills) => {
                tracing::info!(count = skills.len(), path = %path.display(), "Loaded skills");
                for skill in skills {
                    tracing::debug!(name = %skill.name, "Loaded skill");
                    self.skills.push(SkillInfo {
                        name: skill.name,
                        description: skill.description,
                    });
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "Failed to load skills");
            }
        }

        let symlink_path = self.bash_tool.working_dir().join(".skills");
        if let Ok(abs_path) = fs::canonicalize(&path).await {
            #[cfg(unix)]
            {
                let _ = fs::unix::symlink(&abs_path, &symlink_path).await;
            }
            #[cfg(windows)]
            {
                let _ = fs::windows::symlink_dir(&abs_path, &symlink_path).await;
            }
        }

        self
    }

    /// Adds a skill manually without loading from filesystem.
    pub fn skill(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.skills.push(SkillInfo {
            name: name.into(),
            description: description.into(),
        });
        self
    }

    /// Sets up the subagents directory and creates a symlink in the sandbox.
    ///
    /// Subagents are markdown files with YAML frontmatter (name, description).
    /// They can be invoked via: `task --subagent_file <path> --prompt "..."`
    ///
    /// A symlink `.subagents` is created in the sandbox working directory.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.with_subagents("/path/to/subagents")
    /// ```
    pub fn with_subagents(mut self, path: impl AsRef<std::path::Path>) -> Self {
        use crate::subagent_file::SubagentDefinition;

        let path = path.as_ref();
        if !path.exists() {
            return self;
        }

        // Load all subagent definitions from the directory
        if let Ok(defs) = SubagentDefinition::load_from_dir(path) {
            for def in defs {
                // Get relative path for the subagent file
                let filename = format!("{}.md", def.id);
                self.subagents.push(SubagentInfo {
                    name: def.id,
                    description: def.description,
                    path: filename,
                });
            }
        }

        // Create symlink to .subagents in sandbox working directory
        // Canonicalize path to ensure symlink works from sandbox directory
        let symlink_path = self.bash_tool.working_dir().join(".subagents");
        if let Ok(abs_path) = path.canonicalize() {
            let _ = std::os::unix::fs::symlink(&abs_path, &symlink_path);
        }

        self
    }

    /// Sets up the subagents directory and creates a symlink in the sandbox asynchronously.
    pub async fn with_subagents_async(mut self, path: impl AsRef<std::path::Path>) -> Self {
        use crate::subagent_file::SubagentDefinition;

        let path = path.as_ref();
        if !path_exists(path).await {
            return self;
        }

        if let Ok(defs) = SubagentDefinition::load_from_dir_async(path).await {
            for def in defs {
                let filename = format!("{}.md", def.id);
                self.subagents.push(SubagentInfo {
                    name: def.id,
                    description: def.description,
                    path: filename,
                });
            }
        }

        let symlink_path = self.bash_tool.working_dir().join(".subagents");
        if let Ok(abs_path) = fs::canonicalize(path).await {
            #[cfg(unix)]
            {
                let _ = fs::unix::symlink(&abs_path, &symlink_path).await;
            }
            #[cfg(windows)]
            {
                let _ = fs::windows::symlink_dir(&abs_path, &symlink_path).await;
            }
        }

        self
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, limit: usize) -> Self {
        self.inner = self.inner.max_iterations(limit);
        self
    }

    /// Returns the list of registered tool descriptions.
    ///
    /// Useful for dynamically building system prompts.
    pub fn tool_descriptions(&self) -> &[(String, String)] {
        &self.tool_descriptions
    }

    /// Returns a factory for spawning child bash tools (for subagents).
    #[must_use]
    pub fn bash_tool_factory(&self) -> BashToolFactory {
        self.bash_tool_factory.clone()
    }

    /// Returns a mutable reference to the tool registry builder.
    pub fn tool_registry_mut(&mut self) -> &mut ToolRegistryBuilder {
        &mut self.registry_builder
    }

    /// Returns the sandbox working directory path.
    pub fn sandbox_dir(&self) -> Cow<'_, str> {
        self.bash_tool.working_dir().to_string_lossy()
    }

    /// Builds the agent.
    ///
    /// The returned agent only has the `bash` tool.
    /// All registered IPC tools are accessible as bash commands.
    pub fn build(self) -> Agent<LLM, LLM, LLM, H> {
        // Build registry
        let registry = std::sync::Arc::new(self.registry_builder.build(self.bash_tool.outputs_dir()));

        // Configure bash tool
        let bash_tool = self.bash_tool.with_registry(registry);

        // Start factory service for subagents if requested
        if let Some(receiver) = self.bash_tool_factory_receiver {
            bash_tool.start_factory_service(receiver);
        }

        self.inner.bash(bash_tool).build()
    }
}

fn short_description(description: &str) -> String {
    description
        .split('.')
        .next()
        .unwrap_or(description)
        .trim()
        .to_string()
}

async fn path_exists(path: &std::path::Path) -> bool {
    match fs::metadata(path).await {
        Ok(_) => true,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(err) => {
            panic!(
                "failed to access path {}: {}",
                path.display(),
                err
            );
        }
    }
}

/// Returns (os_name, os_version) for the current system.
fn get_os_info() -> (String, String) {
    #[cfg(target_os = "macos")]
    {
        let version = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        ("macOS".to_string(), version)
    }

    #[cfg(target_os = "linux")]
    {
        // Try to get pretty name from os-release
        let version = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("PRETTY_NAME="))
                    .map(|line| {
                        line.trim_start_matches("PRETTY_NAME=")
                            .trim_matches('"')
                            .to_string()
                    })
            })
            .unwrap_or_else(|| {
                // Fallback to uname -r
                std::process::Command::new("uname")
                    .arg("-r")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });
        ("Linux".to_string(), version)
    }

    #[cfg(target_os = "windows")]
    {
        let version = std::process::Command::new("cmd")
            .args(["/C", "ver"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        ("Windows".to_string(), version)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        (std::env::consts::OS.to_string(), "unknown".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that the module compiles and types work
    #[test]
    fn test_types_compile() {
        // This test just ensures the generic constraints are correct
        fn _assert_send_sync<T: Send + Sync>() {}
        // BashAgentBuilder should be constructible with proper types
    }

    #[test]
    fn test_get_os_info() {
        let (os_name, os_version) = get_os_info();
        assert!(!os_name.is_empty());
        assert!(!os_version.is_empty());
        // On macOS, should return "macOS" and a version like "14.0"
        #[cfg(target_os = "macos")]
        assert_eq!(os_name, "macOS");
    }
}
