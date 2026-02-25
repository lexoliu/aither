//! Bash-centric agent builder with stateless bash execution and native terminal controls.
//!
//! This module provides a streamlined API for creating agents where:
//! - **LLM perspective**: core execution happens via `bash`, with native terminal tools
//! - **Developer perspective**: domain tools are registered as bash CLI commands
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
//!     .tool(SubagentTool::new(llm))    // Becomes `subagent` bash command
//!     .build();
//! ```

use std::borrow::Cow;
use std::sync::Arc;

use aither_core::LanguageModel;
use aither_core::llm::Tool;
use aither_core::llm::tool::ToolDefinition;
use askama::Template;
use async_fs as fs;
use executor_core::Executor;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::hook::Hook;
use crate::{
    Agent, AgentBuilder,
    config::{AgentKind, ContextBlock, ContextBlockPriority},
};
use aither_sandbox::builtin::{InputTerminalTool, KillTerminalTool};
use aither_sandbox::{
    BashTool, BashToolFactory, BashToolFactoryReceiver, ContainerExec, ListSshTool, OpenSshTool,
    PermissionHandler, ShellRuntimeAvailability, ShellSessionRegistry, SshServer,
    SshSessionAuthorizer, ToolRegistryBuilder, Unconfigured, bash_tool_factory_channel,
};

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
    host_profile: &'static str,
    host_runtime_context: String,
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
    shell_sessions: ShellSessionRegistry,
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
    /// Creates the agent with native tools (`open_ssh`, `list_ssh`, `kill_terminal`,
    /// `input_terminal`, `bash`) and IPC commands accessible through
    /// bash (registered via `.tool()`).
    pub fn new(llm: LLM, mut bash_tool: BashTool<P, E, Unconfigured>) -> Self {
        let (bash_tool_factory, bash_tool_factory_receiver) = bash_tool_factory_channel();
        let registry_builder = ToolRegistryBuilder::new();
        let mut tool_descriptions = Vec::new();

        let shell_sessions = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: true,
            container: false,
            ssh: false,
        });

        let open_ssh = OpenSshTool::new(shell_sessions.clone());
        let list_ssh = ListSshTool::new(shell_sessions.clone());
        let job_registry = bash_tool.job_registry();
        let kill_terminal_tool = KillTerminalTool::new(job_registry.clone());
        let input_terminal_tool = InputTerminalTool::new(job_registry);
        bash_tool = bash_tool.with_shell_sessions(shell_sessions.clone());

        // SSH management tools are native LLM tool calls (not bash IPC commands).
        let kill_def = ToolDefinition::new(&kill_terminal_tool);
        tool_descriptions.push((
            kill_def.name().to_string(),
            short_description(kill_def.description()),
        ));
        let input_def = ToolDefinition::new(&input_terminal_tool);
        tool_descriptions.push((
            input_def.name().to_string(),
            short_description(input_def.description()),
        ));

        let inner = AgentBuilder::new(llm)
            .tool(open_ssh)
            .tool(list_ssh)
            .tool(kill_terminal_tool)
            .tool(input_terminal_tool);

        Self {
            inner,
            bash_tool,
            registry_builder,
            bash_tool_factory,
            bash_tool_factory_receiver: Some(bash_tool_factory_receiver),
            tool_descriptions,
            skills: Vec::new(),
            subagents: Vec::new(),
            shell_sessions,
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
    async fn resolve_absolute_dir(
        path: &std::path::Path,
        kind: &str,
    ) -> Result<std::path::PathBuf, crate::AgentError> {
        fs::create_dir_all(path).await.map_err(|error| {
            crate::AgentError::Config(format!(
                "failed to create {kind} directory '{}': {error}",
                path.display()
            ))
        })?;

        fs::canonicalize(path).await.map_err(|error| {
            crate::AgentError::Config(format!(
                "failed to canonicalize {kind} directory '{}': {error}",
                path.display()
            ))
        })
    }

    async fn register_readable_dir(
        mut self,
        source_path: &std::path::Path,
        kind: &str,
    ) -> Result<Self, crate::AgentError> {
        let abs_path = Self::resolve_absolute_dir(source_path, kind).await?;
        self.bash_tool = self.bash_tool.with_readable_paths([abs_path]);
        Ok(self)
    }

    async fn attach_readable_symlink(
        mut self,
        source_path: &std::path::Path,
        kind: &str,
        link_name: &str,
    ) -> Result<Self, crate::AgentError> {
        let abs_path = Self::resolve_absolute_dir(source_path, kind).await?;
        let symlink_path = self.bash_tool.working_dir().join(link_name);

        match fs::symlink_metadata(&symlink_path).await {
            Ok(metadata) => {
                if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
                    fs::remove_dir(&symlink_path).await.map_err(|error| {
                        crate::AgentError::Config(format!(
                            "failed to remove existing directory '{}': {error}",
                            symlink_path.display()
                        ))
                    })?;
                } else {
                    fs::remove_file(&symlink_path).await.map_err(|error| {
                        crate::AgentError::Config(format!(
                            "failed to remove existing file '{}': {error}",
                            symlink_path.display()
                        ))
                    })?;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(crate::AgentError::Config(format!(
                    "failed to inspect '{}': {error}",
                    symlink_path.display()
                )));
            }
        }

        #[cfg(unix)]
        {
            fs::unix::symlink(&abs_path, &symlink_path)
                .await
                .map_err(|error| {
                    crate::AgentError::Config(format!(
                        "failed to create symlink '{}' -> '{}': {error}",
                        symlink_path.display(),
                        abs_path.display()
                    ))
                })?;
        }
        #[cfg(windows)]
        {
            fs::windows::symlink_dir(&abs_path, &symlink_path)
                .await
                .map_err(|error| {
                    crate::AgentError::Config(format!(
                        "failed to create symlink '{}' -> '{}': {error}",
                        symlink_path.display(),
                        abs_path.display()
                    ))
                })?;
        }

        self.bash_tool = self.bash_tool.with_readable_paths([abs_path]);
        Ok(self)
    }

    /// Sets runtime shell backend availability for `bash`.
    pub fn shell_runtime_availability(mut self, availability: ShellRuntimeAvailability) -> Self {
        self.bash_tool = self
            .bash_tool
            .with_shell_runtime_availability(availability.clone());
        let _ = self.shell_sessions.set_availability(availability);
        self
    }

    /// Sets preconfigured SSH targets that can be used by `bash` with ssh mode.
    pub fn ssh_servers(self, servers: Vec<SshServer>) -> Self {
        let _ = self.shell_sessions.set_ssh_servers(servers);
        self
    }

    /// Sets the SSH session authorizer for interactive connect/install consent prompts.
    pub fn ssh_authorizer(self, authorizer: Arc<dyn SshSessionAuthorizer>) -> Self {
        let _ = self.shell_sessions.set_ssh_authorizer(authorizer);
        self
    }

    /// Sets the container executor for Docker-based shell backends.
    ///
    /// This is shared across all clones of the internal `ShellSessionRegistry`
    /// via `OnceLock`, so it must only be called once.
    pub fn container_exec<T>(self, exec: Arc<T>) -> Self
    where
        T: ContainerExec + 'static,
    {
        self.shell_sessions.set_container_exec(exec);
        self
    }

    /// Sets the default container ID for Container shell sessions.
    pub fn container_id(self, id: String) -> Self {
        self.shell_sessions.set_container_id(id);
        self
    }

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

    /// Sets an optional persona overlay prompt.
    pub fn persona_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inner = self.inner.persona_prompt(prompt.into());
        self
    }

    /// Sets agent kind (coding/chatbot).
    pub fn agent_kind(mut self, kind: AgentKind) -> Self {
        self.inner = self.inner.agent_kind(kind);
        self
    }

    /// Sets transcript path for long-memory recovery guidance.
    pub fn transcript_path(mut self, path: impl Into<String>) -> Self {
        self.inner = self.inner.transcript_path(path.into());
        self
    }

    /// Enables writing readable transcript entries to the given file path.
    pub fn transcript(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.inner = self.inner.transcript(path);
        self
    }

    /// Adds a structured context block.
    pub fn context_block(mut self, block: ContextBlock) -> Self {
        self.inner = self.inner.context_block(block);
        self
    }

    /// Generates and sets the default system prompt using the built-in template.
    ///
    /// This should be called after all tools are registered.
    pub fn with_default_prompt(mut self) -> Self {
        let availability = self.shell_sessions.availability();
        let host_profile = if availability.container {
            "container"
        } else {
            "leash"
        };
        let runtime_kind = if host_profile == "container" {
            "linux_container"
        } else {
            "user_local_machine"
        };
        let shell_context = format!(
            "<shell_runtime><available_backends local=\"{}\" ssh=\"{}\" /><runtime>{}</runtime></shell_runtime>",
            availability.local || availability.container,
            availability.ssh,
            runtime_kind,
        );
        self.inner = self.inner.context_block(
            ContextBlock::new("shell_runtime", shell_context)
                .with_priority(ContextBlockPriority::High),
        );

        let host_runtime_context = if host_profile == "container" {
            format!(
                "Linux container runtime. Default mode has network enabled. You may install dependencies freely. SSH available: {}.",
                availability.ssh
            )
        } else {
            format!(
                "User machine runtime in sandbox. Default mode has network enabled. Use unsafe for host-level side effects. SSH available: {}.",
                availability.ssh
            )
        };

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
        let user_cwd =
            std::env::current_dir().map_or_else(|_| ".".to_string(), |p| p.display().to_string());

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
            host_profile,
            host_runtime_context,
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
            shell_sessions: self.shell_sessions,
        }
    }

    /// Loads skills from a filesystem path.
    ///
    /// Creates the directory if it does not exist and always creates a
    /// read-only `.skills` symlink inside the sandbox so the agent can
    /// `cat .skills/<name>/SKILL.md` at runtime.  The path is also
    /// registered as a readable (but not writable) sandbox path.
    #[cfg(feature = "skills")]
    async fn load_skills(
        mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        use aither_skills::SkillLoader;

        let path = path.as_ref().to_path_buf();
        let abs_path = Self::resolve_absolute_dir(&path, "skills").await?;
        let loader = SkillLoader::new().add_path(&abs_path);
        let skills = loader.load_all().await.map_err(|error| {
            crate::AgentError::Config(format!(
                "failed to load skills from '{}': {error}",
                abs_path.display()
            ))
        })?;

        tracing::info!(count = skills.len(), path = %abs_path.display(), "Loaded skills");
        for skill in skills {
            tracing::debug!(name = %skill.name, "Loaded skill");
            self.skills.push(SkillInfo {
                name: skill.name,
                description: skill.description,
            });
        }

        Ok(self)
    }

    #[cfg(feature = "skills")]
    pub async fn with_skills(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        let path = path.as_ref().to_path_buf();
        self.load_skills(&path)
            .await?
            .attach_readable_symlink(&path, "skills", ".skills")
            .await
    }

    /// Loads skill metadata and grants read-only access to the source
    /// directory without creating a `.skills` symlink in the sandbox.
    ///
    /// Use this when the runtime already mounts `.skills` inside the sandbox.
    #[cfg(feature = "skills")]
    pub async fn with_skills_readable_only(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        let path = path.as_ref().to_path_buf();
        self.load_skills(&path)
            .await?
            .register_readable_dir(&path, "skills")
            .await
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
    /// builder.with_subagents("/path/to/subagents").await?
    /// ```
    async fn load_subagents(
        mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        use crate::subagent_file::SubagentDefinition;

        let path = path.as_ref().to_path_buf();
        let abs_path = Self::resolve_absolute_dir(&path, "subagents").await?;

        let defs = SubagentDefinition::load_from_dir_async(&abs_path)
            .await
            .map_err(|error| {
                crate::AgentError::Config(format!(
                    "failed to load subagents from '{}': {error}",
                    abs_path.display()
                ))
            })?;

        for def in defs {
            let filename = format!("{}.md", def.id);
            self.subagents.push(SubagentInfo {
                name: def.id,
                description: def.description,
                path: filename,
            });
        }

        Ok(self)
    }

    pub async fn with_subagents(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        let path = path.as_ref().to_path_buf();
        self.load_subagents(&path)
            .await?
            .attach_readable_symlink(&path, "subagents", ".subagents")
            .await
    }

    /// Loads subagent metadata and grants read-only access to the source
    /// directory without creating a `.subagents` symlink in the sandbox.
    ///
    /// Use this when the runtime already mounts `.subagents` inside the sandbox.
    pub async fn with_subagents_readable_only(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        let path = path.as_ref().to_path_buf();
        self.load_subagents(&path)
            .await?
            .register_readable_dir(&path, "subagents")
            .await
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
    pub const fn tool_registry_mut(&mut self) -> &mut ToolRegistryBuilder {
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
        let registry =
            std::sync::Arc::new(self.registry_builder.build(self.bash_tool.outputs_dir()));

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

/// Returns (`os_name`, `os_version`) for the current system.
fn get_os_info() -> (String, String) {
    #[cfg(target_os = "macos")]
    {
        let version = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string());
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

    #[test]
    fn test_system_prompt_container_profile_excludes_leash_modes() {
        let prompt = SystemPrompt {
            os: "macOS".to_string(),
            os_version: "15.0".to_string(),
            arch: "arm64",
            user_cwd: "/tmp/project".to_string(),
            sandbox_dir: "/tmp/sandbox".to_string(),
            tools: "- bash: Execute shell commands".to_string(),
            host_profile: "container",
            host_runtime_context: "runtime=linux_container".to_string(),
            skills: String::new(),
            has_skills: false,
            subagents: String::new(),
            has_subagents: false,
            is_macos: true,
        }
        .render()
        .expect("failed to render container prompt");

        assert!(prompt.contains("<runtime>Linux container runtime</runtime>"));
        assert!(prompt.contains("<shell-runtime>"));
        assert!(!prompt.contains("<shell-modes>"));
        assert!(prompt.contains("install dependencies freely"));
    }

    #[test]
    fn test_system_prompt_leash_profile_includes_shell_modes() {
        let prompt = SystemPrompt {
            os: "macOS".to_string(),
            os_version: "15.0".to_string(),
            arch: "arm64",
            user_cwd: "/tmp/project".to_string(),
            sandbox_dir: "/tmp/sandbox".to_string(),
            tools: "- bash: Execute shell commands".to_string(),
            host_profile: "leash",
            host_runtime_context: "runtime=user_local_machine".to_string(),
            skills: String::new(),
            has_skills: false,
            subagents: String::new(),
            has_subagents: false,
            is_macos: true,
        }
        .render()
        .expect("failed to render leash prompt");

        assert!(prompt.contains("<shell-modes>"));
        assert!(!prompt.contains("<shell-runtime>"));
        assert!(prompt.contains("<unsafe>"));
    }
}
