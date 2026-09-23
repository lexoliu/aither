//! Terminal-first agent builder with stateless terminal execution and native terminal controls.
//!
//! This module provides a streamlined API for creating agents where:
//! - **LLM perspective**: core execution happens via `terminal`, with native terminal tools
//! - **Developer perspective**: domain tools are registered as IPC-backed terminal commands
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         Agent                               │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │                  terminal command tool                  │   │
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
//! use aither_agent::terminal_agent::TerminalAgentBuilder;
//!
//! let agent = TerminalAgentBuilder::new(llm, terminal_tool)
//!     .tool(WebSearchTool::default())  // Becomes `websearch` terminal command
//!     .tool(WebFetchTool::new())       // Becomes `webfetch` terminal command
//!     .tool(SubagentTool::new(llm))    // Becomes `subagent` terminal command
//!     .build();
//! ```

use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use aither_core::LanguageModel;
use aither_core::llm::Tool;
use aither_core::llm::tool::ToolDefinition;
#[cfg(feature = "mcp")]
use aither_mcp::McpConnection;
#[cfg(feature = "skills")]
use aither_skills::SkillRegistry;
use askama::Template;
use async_fs as fs;
use executor_core::Executor;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::hook::Hook;
use crate::{Agent, AgentBuilder, config::AgentKind, context::serialize_xml};
use aither_sandbox::builtin::{InputTerminalTool, KillTerminalTool, ReadTerminalDeltaTool};
use aither_sandbox::{
    CommandPayload, ContainerShellRuntime, PermissionHandler, ShellRuntimeAvailability,
    ShellSessionRegistry, SshServer, SshSessionAuthorizer, TerminalTool, TerminalToolFactory,
    TerminalToolFactoryReceiver, ToolRegistryBuilder, Unconfigured, terminal_tool_factory_channel,
};

/// System prompt template for terminal-first agents.
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

#[derive(serde::Serialize)]
struct ShellRuntimeContextXml {
    available_backends: ShellRuntimeAvailableBackendsXml,
    runtime: &'static str,
}

#[derive(serde::Serialize)]
struct ShellRuntimeAvailableBackendsXml {
    #[serde(rename = "@local")]
    local: bool,
    #[serde(rename = "@ssh")]
    ssh: bool,
}

fn join_text(parts: &[&str]) -> String {
    let capacity = parts.iter().map(|part| part.len()).sum();
    let mut output = String::with_capacity(capacity);
    for part in parts {
        output.push_str(part);
    }
    output
}

fn path_error_text(prefix: &str, path: &std::path::Path, error: &impl std::fmt::Display) -> String {
    let path_text = path.display().to_string();
    let error_text = error.to_string();
    join_text(&[prefix, path_text.as_str(), "': ", error_text.as_str()])
}

fn quoted_path_error_text(
    prefix: &str,
    first_path: &std::path::Path,
    infix: &str,
    second_path: &std::path::Path,
    error: &impl std::fmt::Display,
) -> String {
    let first_path_text = first_path.display().to_string();
    let second_path_text = second_path.display().to_string();
    let error_text = error.to_string();
    join_text(&[
        prefix,
        first_path_text.as_str(),
        infix,
        second_path_text.as_str(),
        "': ",
        error_text.as_str(),
    ])
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
    /// Relative path within the `subagents` folder.
    pub path: String,
}

/// Builder for creating terminal-first agents.
///
/// All registered tools become IPC commands accessible via `terminal`.
/// The LLM only sees the `terminal` tool.
pub struct TerminalAgentBuilder<LLM, P, E, H = ()>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    inner: AgentBuilder<LLM, LLM, LLM, H>,
    terminal_tool: TerminalTool<P, E, Unconfigured>,
    registry_builder: ToolRegistryBuilder,
    terminal_tool_factory: TerminalToolFactory,
    terminal_tool_factory_receiver: Option<TerminalToolFactoryReceiver>,
    tool_descriptions: Vec<(String, String)>,
    skills: Vec<SkillInfo>,
    subagents: Vec<SubagentInfo>,
    shell_sessions: ShellSessionRegistry,
    #[cfg(feature = "skills")]
    skill_registry: Option<SkillRegistry>,
}

impl<LLM, P, E, H> std::fmt::Debug for TerminalAgentBuilder<LLM, P, E, H>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalAgentBuilder")
            .field("tool_count", &self.tool_descriptions.len())
            .field("skill_count", &self.skills.len())
            .field("subagent_count", &self.subagents.len())
            .finish_non_exhaustive()
    }
}

impl<LLM, P, E> TerminalAgentBuilder<LLM, P, E, ()>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    /// Creates a new terminal-first agent builder.
    ///
    /// Creates the agent with native tools (`terminal_kill`, `terminal_input`,
    /// `terminal_read`, `terminal`) and IPC commands accessible through
    /// `terminal` (registered via `.tool()`).
    pub fn new(llm: LLM, mut terminal_tool: TerminalTool<P, E, Unconfigured>) -> Self {
        let (terminal_tool_factory, terminal_tool_factory_receiver) =
            terminal_tool_factory_channel();
        let registry_builder = ToolRegistryBuilder::new();
        let mut tool_descriptions = Vec::new();

        let shell_sessions = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: true,
            container: false,
            ssh: false,
        });

        let job_registry = terminal_tool.job_registry();
        let kill_terminal_tool = KillTerminalTool::new(job_registry.clone());
        let input_terminal_tool = InputTerminalTool::new(job_registry.clone());
        let read_terminal_delta_tool = ReadTerminalDeltaTool::new(job_registry);
        terminal_tool = terminal_tool.with_shell_sessions(shell_sessions.clone());

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
        let read_delta_def = ToolDefinition::new(&read_terminal_delta_tool);
        tool_descriptions.push((
            read_delta_def.name().to_string(),
            short_description(read_delta_def.description()),
        ));

        let inner = AgentBuilder::new(llm)
            .tool(kill_terminal_tool)
            .tool(input_terminal_tool)
            .tool(read_terminal_delta_tool);

        Self {
            inner,
            terminal_tool,
            registry_builder,
            terminal_tool_factory,
            terminal_tool_factory_receiver: Some(terminal_tool_factory_receiver),
            tool_descriptions,
            skills: Vec::new(),
            subagents: Vec::new(),
            shell_sessions,
            #[cfg(feature = "skills")]
            skill_registry: None,
        }
    }
}

impl<LLM, P, E, H> TerminalAgentBuilder<LLM, P, E, H>
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
            let prefix = join_text(&["failed to create ", kind, " directory '"]);
            crate::AgentError::Config(path_error_text(prefix.as_str(), path, &error))
        })?;

        fs::canonicalize(path).await.map_err(|error| {
            let prefix = join_text(&["failed to canonicalize ", kind, " directory '"]);
            crate::AgentError::Config(path_error_text(prefix.as_str(), path, &error))
        })
    }

    async fn register_readable_dir(
        mut self,
        source_path: &std::path::Path,
        kind: &str,
    ) -> Result<Self, crate::AgentError> {
        let abs_path = Self::resolve_absolute_dir(source_path, kind).await?;
        self.terminal_tool = self.terminal_tool.with_readable_paths([abs_path]);
        Ok(self)
    }

    async fn attach_readable_symlink(
        mut self,
        source_path: &std::path::Path,
        kind: &str,
        link_name: &str,
    ) -> Result<Self, crate::AgentError> {
        let abs_path = Self::resolve_absolute_dir(source_path, kind).await?;
        let symlink_path = self.terminal_tool.working_dir().join(link_name);

        match fs::symlink_metadata(&symlink_path).await {
            Ok(metadata) => {
                if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
                    fs::remove_dir(&symlink_path).await.map_err(|error| {
                        crate::AgentError::Config(path_error_text(
                            "failed to remove existing directory '",
                            &symlink_path,
                            &error,
                        ))
                    })?;
                } else {
                    fs::remove_file(&symlink_path).await.map_err(|error| {
                        crate::AgentError::Config(path_error_text(
                            "failed to remove existing file '",
                            &symlink_path,
                            &error,
                        ))
                    })?;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(crate::AgentError::Config(path_error_text(
                    "failed to inspect '",
                    &symlink_path,
                    &error,
                )));
            }
        }

        #[cfg(unix)]
        {
            fs::unix::symlink(&abs_path, &symlink_path)
                .await
                .map_err(|error| {
                    crate::AgentError::Config(quoted_path_error_text(
                        "failed to create symlink '",
                        &symlink_path,
                        "' -> '",
                        &abs_path,
                        &error,
                    ))
                })?;
        }
        #[cfg(windows)]
        {
            fs::windows::symlink_dir(&abs_path, &symlink_path)
                .await
                .map_err(|error| {
                    crate::AgentError::Config(quoted_path_error_text(
                        "failed to create symlink '",
                        &symlink_path,
                        "' -> '",
                        &abs_path,
                        &error,
                    ))
                })?;
        }

        self.terminal_tool = self.terminal_tool.with_readable_paths([abs_path]);
        Ok(self)
    }

    /// Sets runtime shell backend availability for `terminal`.
    #[must_use]
    pub fn shell_runtime_availability(mut self, availability: ShellRuntimeAvailability) -> Self {
        self.shell_sessions = self.shell_sessions.with_availability(availability.clone());
        self.terminal_tool = self
            .terminal_tool
            .with_shell_sessions(self.shell_sessions.clone())
            .with_shell_runtime_availability(availability);
        self
    }

    /// Sets preconfigured SSH targets that can be used by `terminal` with ssh mode.
    ///
    /// # Panics
    ///
    /// Panics when the SSH server configuration is invalid.
    #[must_use]
    pub fn ssh_servers(mut self, servers: Vec<SshServer>) -> Self {
        self.shell_sessions = self
            .shell_sessions
            .with_ssh_servers(servers)
            .expect("invalid ssh server configuration");
        self.terminal_tool = self
            .terminal_tool
            .with_shell_sessions(self.shell_sessions.clone());
        self
    }

    /// Sets the SSH session authorizer for interactive connect/install consent prompts.
    #[must_use]
    pub fn ssh_authorizer(mut self, authorizer: Arc<dyn SshSessionAuthorizer>) -> Self {
        self.shell_sessions = self.shell_sessions.with_ssh_authorizer(authorizer);
        self.terminal_tool = self
            .terminal_tool
            .with_shell_sessions(self.shell_sessions.clone());
        self
    }

    /// Sets the container runtime for container-backed shell sessions.
    #[must_use]
    pub fn container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.shell_sessions = self.shell_sessions.with_container_runtime(runtime.clone());
        self.terminal_tool = self.terminal_tool.with_container_runtime(runtime);
        self
    }

    /// Registers a tool as an IPC command accessible via `terminal`.
    ///
    /// The tool becomes a terminal command with the same name.
    /// For example, `WebSearchTool` becomes the `websearch` command.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.tool(WebSearchTool::default())
    /// // LLM can now call: terminal({ script: "websearch \"rust async\"" })
    /// ```
    #[must_use]
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

    /// Registers a schema-defined command on the terminal IPC bridge.
    ///
    /// This is the dynamic counterpart to [`Self::tool`]. The command wrapper
    /// is still derived entirely from the tool definition schema, so `--help`,
    /// positional arguments, and validation all stay automatic.
    #[must_use]
    pub fn definition_handler<F>(mut self, definition: &ToolDefinition, handler: F) -> Self
    where
        F: Fn(
                serde_json::Value,
            )
                -> Pin<Box<dyn Future<Output = Result<Option<CommandPayload>, String>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        let name = definition.name().to_string();
        let description = definition.description().to_string();
        self.tool_descriptions
            .push((name, short_description(description.as_str())));
        self.registry_builder
            .configure_definition_handler(definition, handler);
        self
    }

    /// Registers all tools from one MCP connection as terminal commands.
    #[cfg(feature = "mcp")]
    #[must_use]
    pub fn mcp_connection(mut self, conn: McpConnection) -> Self {
        let definitions = aither_mcp::register_terminal_commands(&mut self.registry_builder, conn);
        self.tool_descriptions
            .extend(definitions.into_iter().map(|definition| {
                (
                    definition.name().to_string(),
                    short_description(definition.description()),
                )
            }));
        self
    }

    /// Registers a tool with a custom description.
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn system_prompt_raw(mut self, prompt: impl Into<String>) -> Self {
        self.inner = self.inner.system_prompt(prompt.into());
        self
    }

    /// Sets an optional persona overlay prompt.
    #[must_use]
    pub fn persona_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inner = self.inner.persona_prompt(prompt.into());
        self
    }

    /// Sets agent kind (coding/chatbot).
    #[must_use]
    pub fn agent_kind(mut self, kind: AgentKind) -> Self {
        self.inner = self.inner.agent_kind(kind);
        self
    }

    /// Sets transcript path for long-memory recovery guidance.
    #[must_use]
    pub fn transcript_path(mut self, path: impl Into<String>) -> Self {
        self.inner = self.inner.transcript_path(path.into());
        self
    }

    /// Enables writing readable transcript entries to the given file path.
    #[must_use]
    pub fn transcript(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.inner = self.inner.transcript(path);
        self
    }

    /// Inserts or replaces a typed persistent system block.
    #[must_use]
    pub fn system<T: serde::Serialize>(mut self, value: T) -> Self {
        self.inner = self.inner.system(value);
        self
    }

    /// Inserts or replaces a persistent system block with an explicit tag.
    #[must_use]
    pub fn system_named(mut self, tag: impl Into<String>, content: impl Into<String>) -> Self {
        self.inner = self.inner.system_named(tag, content);
        self
    }

    /// Inserts or replaces a persistent system block with raw text.
    ///
    /// Unlike [`system`](Self::system) and [`system_named`](Self::system_named),
    /// the `content` is stored verbatim without any XML wrapping. This is the
    /// preferred entry point for prose system blocks (workspace descriptions,
    /// runtime metadata, environment hints) where XML structure adds tokens
    /// without providing semantic value.
    #[must_use]
    pub fn system_text(mut self, tag: impl Into<String>, content: impl Into<String>) -> Self {
        self.inner = self.inner.system_text(tag, content);
        self
    }

    /// Generates and sets the default system prompt using the built-in template.
    ///
    /// This should be called after all tools are registered.
    ///
    /// # Panics
    ///
    /// Panics when no shell backend is available.
    #[must_use]
    pub fn with_default_prompt(mut self) -> Self {
        let availability = self.shell_sessions.availability();
        let host_profile = if availability.container {
            "container"
        } else if availability.local {
            "heel"
        } else if availability.ssh {
            "remote"
        } else {
            panic!("terminal agent requires at least one shell backend")
        };
        let runtime_kind = match host_profile {
            "container" => "linux_container",
            "remote" => "remote_ssh",
            "heel" => "user_local_machine",
            _ => unreachable!("validated host profile"),
        };
        let shell_context = serialize_xml(
            "shell_runtime",
            &ShellRuntimeContextXml {
                available_backends: ShellRuntimeAvailableBackendsXml {
                    local: availability.local || availability.container,
                    ssh: availability.ssh,
                },
                runtime: runtime_kind,
            },
        );
        self.inner = self.inner.system_named("shell_runtime", shell_context);

        let ssh_context = describe_ssh_servers(&self.shell_sessions.list_ssh_servers());
        let host_runtime_context =
            describe_host_runtime_context(host_profile, &availability, ssh_context.as_str());

        // Build tools description
        let tools = render_tool_descriptions(&self.tool_descriptions);

        // Build skills description
        let has_skills = !self.skills.is_empty();
        let skills = render_skill_descriptions(&self.skills);

        // Build subagents description
        let has_subagents = !self.subagents.is_empty();
        let subagents = render_subagent_descriptions(&self.subagents);

        // Get directory paths
        let sandbox_dir = self.terminal_tool.working_dir().display().to_string();
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
    pub fn hook<NH: Hook>(self, hook: NH) -> TerminalAgentBuilder<LLM, P, E, crate::HCons<NH, H>> {
        TerminalAgentBuilder {
            inner: self.inner.hook(hook),
            terminal_tool: self.terminal_tool,
            registry_builder: self.registry_builder,
            terminal_tool_factory: self.terminal_tool_factory,
            terminal_tool_factory_receiver: self.terminal_tool_factory_receiver,
            tool_descriptions: self.tool_descriptions,
            skills: self.skills,
            subagents: self.subagents,
            shell_sessions: self.shell_sessions,
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
        }
    }

    /// Loads skills from a filesystem path.
    ///
    /// Creates the directory if it does not exist and always creates a
    /// read-only `skills` symlink inside the sandbox so the agent can
    /// `cat skills/<name>/SKILL.md` at runtime. The path is also
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
        let mut registry = self.skill_registry.take().unwrap_or_default();
        registry.load_from(&loader).await.map_err(|error| {
            crate::AgentError::Config(path_error_text(
                "failed to load skills from '",
                &abs_path,
                &error,
            ))
        })?;

        let skills = registry.all().into_iter().cloned().collect::<Vec<_>>();
        self.skills.clear();
        tracing::info!(count = skills.len(), path = %abs_path.display(), "Loaded skills");
        for skill in skills {
            tracing::debug!(name = %skill.name, "Loaded skill");
            self.skills.push(SkillInfo {
                name: skill.name,
                description: skill.description,
            });
        }
        self.skill_registry = Some(registry);

        Ok(self)
    }

    #[cfg(feature = "skills")]
    /// Loads skills and exposes the skill directory inside the sandbox.
    ///
    /// # Errors
    ///
    /// Returns an error when loading skills or creating the sandbox symlink fails.
    pub async fn with_skills(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        let path = path.as_ref().to_path_buf();
        self.load_skills(&path)
            .await?
            .attach_readable_symlink(&path, "skills", "skills")
            .await
    }

    /// Loads skill metadata and grants read-only access to the source
    /// directory without creating a `skills` symlink in the sandbox.
    ///
    /// Use this when the runtime already mounts `skills` inside the sandbox.
    ///
    /// # Errors
    ///
    /// Returns an error when loading skills or registering the readable directory fails.
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
    #[must_use]
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
    /// A symlink `subagents` is created in the sandbox working directory.
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
                crate::AgentError::Config(path_error_text(
                    "failed to load subagents from '",
                    &abs_path,
                    &error,
                ))
            })?;

        for def in defs {
            let mut filename = def.id.clone();
            filename.push_str(".md");
            self.subagents.push(SubagentInfo {
                name: def.id,
                description: def.description,
                path: filename,
            });
        }

        Ok(self)
    }

    /// Loads subagents and exposes the subagent directory inside the sandbox.
    ///
    /// # Errors
    ///
    /// Returns an error when loading subagents or creating the sandbox symlink fails.
    pub async fn with_subagents(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::AgentError> {
        let path = path.as_ref().to_path_buf();
        self.load_subagents(&path)
            .await?
            .attach_readable_symlink(&path, "subagents", "subagents")
            .await
    }

    /// Loads subagent metadata and grants read-only access to the source
    /// directory without creating a `subagents` symlink in the sandbox.
    ///
    /// Use this when the runtime already mounts `subagents` inside the sandbox.
    ///
    /// # Errors
    ///
    /// Returns an error when loading subagents or registering the readable directory fails.
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
    #[must_use]
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

    /// Returns a factory for spawning child terminal tools (for subagents).
    #[must_use]
    pub fn terminal_tool_factory(&self) -> TerminalToolFactory {
        self.terminal_tool_factory.clone()
    }

    /// Returns a mutable reference to the tool registry builder.
    pub const fn tool_registry_mut(&mut self) -> &mut ToolRegistryBuilder {
        &mut self.registry_builder
    }

    /// Returns the sandbox working directory path.
    pub fn sandbox_dir(&self) -> Cow<'_, str> {
        self.terminal_tool.working_dir().to_string_lossy()
    }

    /// Builds the agent.
    ///
    /// The returned agent exposes `terminal` plus native terminal control tools.
    /// All registered IPC tools are accessible as terminal commands.
    pub fn build(self) -> Agent<LLM, LLM, LLM, H> {
        // Build registry
        let registry = std::sync::Arc::new(
            self.registry_builder
                .build(self.terminal_tool.outputs_dir()),
        );

        // Configure terminal tool
        let terminal_tool = self.terminal_tool.with_registry(registry);

        // Start factory service for subagents if requested
        if let Some(receiver) = self.terminal_tool_factory_receiver {
            terminal_tool.start_factory_service(receiver);
        }

        let inner = self.inner.terminal(terminal_tool);
        #[cfg(feature = "skills")]
        let inner = if let Some(registry) = self.skill_registry {
            inner.skill_registry(Arc::new(registry))
        } else {
            inner
        };

        inner.build()
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

fn push_bullet_line(output: &mut String, body: &str) {
    if !output.is_empty() {
        output.push('\n');
    }
    output.push_str("- ");
    output.push_str(body);
}

fn render_tool_descriptions(entries: &[(String, String)]) -> String {
    let mut output = String::new();
    for (name, description) in entries {
        let mut line = String::with_capacity(name.len() + description.len() + 2);
        line.push_str(name.as_str());
        line.push_str(": ");
        line.push_str(description.as_str());
        push_bullet_line(&mut output, line.as_str());
    }
    output
}

fn render_skill_descriptions(entries: &[SkillInfo]) -> String {
    let mut output = String::new();
    for skill in entries {
        let mut line = String::with_capacity(skill.name.len() + skill.description.len() + 2);
        line.push_str(skill.name.as_str());
        line.push_str(": ");
        line.push_str(skill.description.as_str());
        push_bullet_line(&mut output, line.as_str());
    }
    output
}

fn render_subagent_descriptions(entries: &[SubagentInfo]) -> String {
    let mut output = String::new();
    for subagent in entries {
        let mut line = String::with_capacity(
            subagent.name.len() + subagent.path.len() + subagent.description.len() + 5,
        );
        line.push_str(subagent.name.as_str());
        line.push_str(" (");
        line.push_str(subagent.path.as_str());
        line.push_str("): ");
        line.push_str(subagent.description.as_str());
        push_bullet_line(&mut output, line.as_str());
    }
    output
}

fn describe_host_runtime_context(
    host_profile: &str,
    availability: &ShellRuntimeAvailability,
    ssh_context: &str,
) -> String {
    let mut output = String::new();
    match host_profile {
        "container" => {
            output.push_str(
                "Linux container runtime. Default mode has network enabled. You may install dependencies freely. SSH available: ",
            );
            output.push_str(if availability.ssh { "true" } else { "false" });
            output.push_str(". ");
        }
        "remote" => {
            output.push_str(
                "Remote SSH runtime. Default mode runs on the configured SSH target with network enabled. Local CLI commands exposed through terminal are unavailable unless a local backend also exists. ",
            );
        }
        "heel" => {
            output.push_str(
                "User machine runtime in sandbox. Default mode has network enabled. Use unsafe for host-level side effects. SSH available: ",
            );
            output.push_str(if availability.ssh { "true" } else { "false" });
            output.push_str(". ");
        }
        _ => unreachable!("validated host profile"),
    }
    output.push_str(ssh_context);
    output
}

fn describe_ssh_servers(servers: &[SshServer]) -> String {
    if servers.is_empty() {
        "No SSH servers are configured.".to_string()
    } else if servers.len() == 1 {
        let server = &servers[0];
        let mut output = String::with_capacity(server.id().len() + server.target.len() + 89);
        output.push_str("Configured SSH server: ");
        output.push_str(server.id());
        output.push_str(" (");
        output.push_str(server.target.as_str());
        output.push_str("). When using SSH backend, ssh_server_id may be omitted because only one server is configured.");
        output
    } else {
        let mut listings = String::new();
        for server in servers {
            if !listings.is_empty() {
                listings.push_str(", ");
            }
            listings.push_str(server.id());
            listings.push_str(" (");
            listings.push_str(server.target.as_str());
            listings.push(')');
        }
        let mut output = String::with_capacity(listings.len() + 24);
        output.push_str("Configured SSH servers: ");
        output.push_str(listings.as_str());
        output.push('.');
        output
    }
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
    use std::borrow::Cow;

    use aither_core::LanguageModel;
    use aither_core::llm::{Event, LLMRequest, Tool, model::Profile as ModelProfile};
    use aither_sandbox::permission::NoopPermissionHandler;
    use executor_core::tokio::TokioGlobal;
    use futures_core::Stream;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use tempfile::tempdir;

    #[derive(Debug)]
    struct MockError;

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("mock error")
        }
    }

    impl std::error::Error for MockError {}

    #[derive(Clone)]
    struct MockLlm;

    impl LanguageModel for MockLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::stream::empty()
        }

        fn profile(&self) -> impl std::future::Future<Output = ModelProfile> + Send {
            std::future::ready(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                100_000,
            ))
        }
    }

    struct MockTool;

    #[derive(Debug, JsonSchema, Deserialize)]
    struct MockArgs;

    impl Tool for MockTool {
        fn name(&self) -> Cow<'static, str> {
            "mock_tool".into()
        }

        type Arguments = MockArgs;
        type Res = aither_core::llm::ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(aither_core::llm::ToolResult::text("ok")))
        }
    }

    // Test that the module compiles and types work
    #[test]
    fn test_types_compile() {
        // This test just ensures the generic constraints are correct
        fn _assert_send_sync<T: Send + Sync>() {}
        // TerminalAgentBuilder should be constructible with proper types
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
    fn test_system_prompt_container_profile_excludes_heel_modes() {
        let prompt = SystemPrompt {
            os: "macOS".to_string(),
            os_version: "15.0".to_string(),
            arch: "arm64",
            user_cwd: "/tmp/project".to_string(),
            sandbox_dir: "/tmp/sandbox".to_string(),
            tools: "- terminal: Execute terminal commands".to_string(),
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
    fn test_system_prompt_heel_profile_includes_shell_modes() {
        let prompt = SystemPrompt {
            os: "macOS".to_string(),
            os_version: "15.0".to_string(),
            arch: "arm64",
            user_cwd: "/tmp/project".to_string(),
            sandbox_dir: "/tmp/sandbox".to_string(),
            tools: "- terminal: Execute terminal commands".to_string(),
            host_profile: "heel",
            host_runtime_context: "runtime=user_local_machine".to_string(),
            skills: String::new(),
            has_skills: false,
            subagents: String::new(),
            has_subagents: false,
            is_macos: true,
        }
        .render()
        .expect("failed to render heel prompt");

        assert!(prompt.contains("<shell-modes>"));
        assert!(!prompt.contains("<shell-runtime>"));
        assert!(prompt.contains("<unsafe>"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_agent_builder_keeps_ipc_tools_out_of_llm_tool_surface() {
        let dir = tempdir().expect("tempdir should exist");
        let terminal_tool =
            aither_sandbox::TerminalTool::<NoopPermissionHandler, TokioGlobal>::new_exact(
                dir.path(),
                NoopPermissionHandler,
                TokioGlobal,
            )
            .await
            .expect("terminal tool should initialize");

        let agent = TerminalAgentBuilder::new(MockLlm, terminal_tool)
            .tool(MockTool)
            .with_default_prompt()
            .build();

        let mut tool_names = agent
            .tools
            .definitions()
            .into_iter()
            .map(|definition| definition.name().to_string())
            .collect::<Vec<_>>();
        tool_names.sort();

        assert_eq!(
            tool_names,
            vec![
                "terminal".to_string(),
                "terminal_input".to_string(),
                "terminal_kill".to_string(),
                "terminal_read".to_string(),
            ]
        );
    }
}
