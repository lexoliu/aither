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
//! │  │  │  websearch | webfetch | task | ...   │    │   │
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
//!     // Becomes the main agent's `subagent` terminal command.
//!     .subagents(subagents, "Run or continue a subagent")
//!     .build();
//! ```

use std::borrow::Cow;
use std::future::Future;
use std::marker::PhantomData;
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
#[cfg(feature = "skills")]
use async_fs as fs;
use executor_core::Executor;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

#[cfg(feature = "skills")]
use crate::builder::sealed::Sealed as _;
use crate::builder::{AgentSettings, sealed};
use crate::command_catalog::short_description;
use crate::context::InstructionSlot;
use crate::hook::Hook;
use crate::{Agent, AgentBuilder, AgentOptions};
use aither_core::llm::ToolResult;
use aither_sandbox::builtin::{InputTerminalTool, KillTerminalTool, ReadTerminalDeltaTool};
use aither_sandbox::{
    ContainerShellRuntime, PermissionHandler, ShellRuntimeAvailability, ShellSessionRegistry,
    SshServer, SshSessionAuthorizer, TerminalTool, ToolDistribution, ToolRegistryBuilder,
    ToolRuntimeError, Unconfigured,
};

/// System prompt template for terminal-first agents.
#[derive(Template)]
#[template(path = "system.txt", escape = "none")]
#[allow(clippy::struct_excessive_bools)]
struct SystemPrompt {
    os: String,
    os_version: String,
    arch: &'static str,
    user_cwd: String,
    sandbox_dir: String,
    host_profile: &'static str,
    host_runtime_context: String,
    skills: String,
    has_skills: bool,
    is_macos: bool,
    has_websearch_command: bool,
}

#[derive(Template)]
#[template(path = "shell_runtime.txt", escape = "none")]
struct ShellRuntimeContextTemplate {
    local: bool,
    ssh: bool,
    runtime: &'static str,
}

#[cfg(feature = "skills")]
fn join_text(parts: &[&str]) -> String {
    let capacity = parts.iter().map(|part| part.len()).sum();
    let mut output = String::with_capacity(capacity);
    for part in parts {
        output.push_str(part);
    }
    output
}

#[cfg(feature = "skills")]
fn path_error_text(prefix: &str, path: &std::path::Path, error: &impl std::fmt::Display) -> String {
    let path_text = path.display().to_string();
    let error_text = error.to_string();
    join_text(&[prefix, path_text.as_str(), "': ", error_text.as_str()])
}

#[cfg(feature = "skills")]
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

/// [`TerminalAgentBuilder`] state while its shell backends are still being
/// configured: the terminal's tool environments are not prepared yet.
#[derive(Debug, Clone, Copy)]
pub struct Unprepared;

/// [`TerminalAgentBuilder`] state once its terminal holds prepared tool
/// environments, either from [`TerminalAgentBuilder::initialize_tools`] or
/// adopted with [`TerminalAgentBuilder::prepared_shell_sessions`].
///
/// The shell backends are fixed in this state: preparation is bound to them,
/// so the builder no longer offers the methods that change them.
#[derive(Debug, Clone, Copy)]
pub struct Prepared;

/// Builder for creating terminal-first agents.
///
/// All registered tools become IPC commands accessible via `terminal`.
/// The LLM only sees the `terminal` tool.
///
/// `S` tracks whether the terminal's tool environments are prepared
/// ([`Unprepared`] or [`Prepared`]); shell backends are configured only before
/// preparation. `Balanced`, `Fast` and `Compactor` are the models of the
/// agent's balanced and fast tiers and of its compactions, `LLM` until
/// [`balanced_model`](Self::balanced_model), [`fast_model`](Self::fast_model)
/// or [`compaction_model`](Self::compaction_model) sets another. `Schema` is
/// the handoff compactions ask for, [`HandoffDocument`](crate::HandoffDocument)
/// until [`handoff`](Self::handoff) sets another.
///
/// Every agent-level option is a method of [`AgentOptions`], shared with
/// [`AgentBuilder`].
pub struct TerminalAgentBuilder<
    LLM,
    P,
    E,
    H = (),
    S = Unprepared,
    Balanced = LLM,
    Fast = Balanced,
    Compactor = LLM,
    Schema = crate::HandoffDocument,
> where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    inner: AgentBuilder<LLM, Balanced, Fast, H, Compactor, Schema>,
    terminal_tool: TerminalTool<P, E, Unconfigured>,
    registry_builder: ToolRegistryBuilder,
    tool_descriptions: Vec<(String, String)>,
    skills: Vec<SkillInfo>,
    host_load: HostLoadSetup,
    /// The `subagent` command, registered at build with the agent's notices.
    /// Boxed: the tool is large, and the builder moves by value through
    /// every step and across the awaits of its preparation.
    subagents: Option<Box<crate::specialized::SubagentTool<LLM>>>,
    state: PhantomData<S>,
}

/// How the built agent watches the host's memory pressure and load.
struct HostLoadSetup {
    enabled: bool,
    config: crate::HostLoadConfig,
    /// The reader to watch; the host's own when unset.
    reader: Option<crate::AnyHostReader>,
    /// Whether the volumes of the home environment are watched.
    disk: bool,
    thresholds: crate::DiskThresholds,
    /// The reader of the machine's home volume; the one holding the working
    /// directory when unset.
    disk_reader: Option<crate::AnyDiskReader>,
}

impl Default for HostLoadSetup {
    fn default() -> Self {
        Self {
            enabled: true,
            config: crate::HostLoadConfig::default(),
            reader: None,
            disk: true,
            thresholds: crate::DiskThresholds::default(),
            disk_reader: None,
        }
    }
}

impl<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema> std::fmt::Debug
    for TerminalAgentBuilder<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalAgentBuilder")
            .field("tool_count", &self.tool_descriptions.len())
            .field("skill_count", &self.skills.len())
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
    /// Creates the agent with native tools (`terminal`, `monitor`,
    /// `terminal_kill`, `terminal_input`, `terminal_read`, `compact`) and IPC
    /// commands accessible through
    /// `terminal` (registered via `.tool()`). The native control tools address
    /// the same background jobs as `terminal` and are not advertised as commands.
    pub fn new(llm: LLM, mut terminal_tool: TerminalTool<P, E, Unconfigured>) -> Self {
        let registry_builder = ToolRegistryBuilder::new();
        let tool_descriptions = Vec::new();

        let shell_sessions = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: true,
            container: false,
            ssh: false,
        });

        terminal_tool = terminal_tool.with_shell_sessions(shell_sessions);

        // The terminal controls are registered at build, over the notices
        // the options settle on.
        let inner = AgentBuilder::new(llm).compact_tool();

        Self {
            inner,
            terminal_tool,
            registry_builder,
            tool_descriptions,
            skills: Vec::new(),
            host_load: HostLoadSetup::default(),
            subagents: None,
            state: PhantomData,
        }
    }
}

impl<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema> sealed::Sealed
    for TerminalAgentBuilder<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    fn settings(&mut self) -> &mut AgentSettings {
        sealed::Sealed::settings(&mut self.inner)
    }
}

/// The terminal agent's options are its inner [`AgentBuilder`]'s settings.
impl<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema> AgentOptions
    for TerminalAgentBuilder<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
}

impl<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema>
    TerminalAgentBuilder<LLM, P, E, H, S, Balanced, Fast, Compactor, Schema>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    Compactor: LanguageModel,
    H: Hook,
    Schema: crate::HandoffSchema,
{
    /// Moves every part into a builder with another hook chain, models,
    /// handoff schema or state.
    fn rebuild<NH, NS, NB, NF, NC, NSchema>(
        self,
        inner: impl FnOnce(
            AgentBuilder<LLM, Balanced, Fast, H, Compactor, Schema>,
        ) -> AgentBuilder<LLM, NB, NF, NH, NC, NSchema>,
    ) -> TerminalAgentBuilder<LLM, P, E, NH, NS, NB, NF, NC, NSchema> {
        TerminalAgentBuilder {
            inner: inner(self.inner),
            terminal_tool: self.terminal_tool,
            registry_builder: self.registry_builder,
            tool_descriptions: self.tool_descriptions,
            skills: self.skills,
            host_load: self.host_load,
            subagents: self.subagents,
            state: PhantomData,
        }
    }

    #[cfg(feature = "skills")]
    async fn resolve_absolute_dir(
        path: &std::path::Path,
        kind: &str,
    ) -> Result<std::path::PathBuf, crate::AgentError> {
        fs::create_dir_all(path).await.map_err(|error| {
            let prefix = join_text(&["failed to create ", kind, " directory '"]);
            crate::AgentError::Config(path_error_text(prefix.as_str(), path, &error))
        })?;

        fs::canonicalize(path)
            .await
            // Without the Windows verbatim prefix, so these roots compare with
            // the canonical paths subagent grants resolve.
            .map(|canonical| dunce::simplified(&canonical).to_path_buf())
            .map_err(|error| {
                let prefix = join_text(&["failed to canonicalize ", kind, " directory '"]);
                crate::AgentError::Config(path_error_text(prefix.as_str(), path, &error))
            })
    }

    #[cfg(feature = "skills")]
    async fn register_readable_dir(
        mut self,
        source_path: &std::path::Path,
        kind: &str,
    ) -> Result<Self, crate::AgentError> {
        let abs_path = Self::resolve_absolute_dir(source_path, kind).await?;
        self.terminal_tool = self.terminal_tool.with_readable_paths([abs_path]);
        Ok(self)
    }

    #[cfg(feature = "skills")]
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
                Option<aither_core::llm::CallIdentity>,
                serde_json::Value,
            ) -> Pin<Box<dyn Future<Output = Result<ToolResult, String>> + Send>>
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

    /// Registers every command `set` holds as a terminal command, for as long
    /// as it holds it.
    ///
    /// The set is read whenever a command is listed or resolved, so a command
    /// it gains mid-session is runnable from the next terminal execution. The
    /// built agent lists the set's commands in the command table
    /// ([`InstructionSlot::COMMANDS`]) and keeps that table current as live
    /// content (see [`Context::update_live`](crate::Context::update_live)): a
    /// change is announced once as a diff while the prefix is cached, and the
    /// next compaction folds the clean table in.
    #[must_use]
    pub fn command_set(mut self, set: impl aither_sandbox::CommandSet) -> Self {
        self.registry_builder.command_set(set);
        self
    }

    /// Registers the tools of one MCP connection as terminal commands that
    /// follow the server's tool list; see [`Self::command_set`].
    #[cfg(feature = "mcp")]
    #[must_use]
    pub fn mcp_connection(self, conn: McpConnection) -> Self {
        self.command_set(conn)
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

    /// Registers a command built for each terminal that serves it, from that
    /// terminal's [`TerminalBinding`](aither_sandbox::TerminalBinding), with
    /// an explicit description.
    ///
    /// The main agent's terminal and every subagent's terminal get their own
    /// instance, so a command that launches subagents (`SubagentTool`
    /// through [`SubagentTool::for_terminal`](crate::specialized::SubagentTool::for_terminal))
    /// launches them with the authority of the terminal it runs in.
    #[must_use]
    pub fn terminal_bound_tool_with_desc<T, F>(
        mut self,
        name: impl Into<String>,
        bind: F,
        description: impl Into<String>,
    ) -> Self
    where
        F: Fn(&aither_sandbox::TerminalBinding) -> T + Send + Sync + 'static,
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let name = name.into();
        self.tool_descriptions
            .push((name.clone(), description.into()));
        self.registry_builder
            .configure_terminal_bound_tool(name, bind);
        self
    }

    /// Gives the main agent the `subagent` command `subagents` serves, with
    /// `description`, and publishes the agent's requests for its `fork`.
    ///
    /// A subagent's terminal has no `subagent` command, so a subagent, a
    /// fork included, starts none. A bare `subagent` lists the names that
    /// resolve.
    #[must_use]
    pub fn subagents(
        mut self,
        subagents: crate::specialized::SubagentTool<LLM>,
        description: impl Into<String>,
    ) -> Self
    where
        LLM: Sync + 'static,
    {
        self.inner = self.inner.fork_source(subagents.fork_source());
        self.tool_descriptions
            .push(("subagent".to_string(), description.into()));
        self.subagents = Some(Box::new(subagents));
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

    /// Generates and sets the default system prompt using the built-in template.
    ///
    /// This should be called after all tools are registered.
    ///
    /// # Panics
    ///
    /// Panics when no shell backend is available.
    #[must_use]
    pub fn with_default_prompt(mut self) -> Self {
        let availability = self.terminal_tool.shell_sessions().availability();
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
        let shell_context = ShellRuntimeContextTemplate {
            local: availability.local || availability.container,
            ssh: availability.ssh,
            runtime: runtime_kind,
        }
        .render()
        .expect("shell runtime context template must render");
        self.inner = self
            .inner
            .system_text(&InstructionSlot::SHELL_RUNTIME, shell_context);

        let ssh_context =
            describe_ssh_servers(&self.terminal_tool.shell_sessions().list_ssh_servers());
        let host_runtime_context =
            describe_host_runtime_context(host_profile, &availability, ssh_context.as_str());

        // Build skills description
        let has_skills = !self.skills.is_empty();
        let skills = render_skill_descriptions(&self.skills);

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
            host_profile,
            host_runtime_context,
            skills,
            has_skills,
            is_macos,
            has_websearch_command: self.has_command("websearch"),
        };

        let prompt = template
            .render()
            .expect("failed to render system prompt template");
        self.inner = self.inner.system_prompt(prompt);
        self
    }

    /// Adds a hook to intercept agent operations; see [`AgentBuilder::hook`].
    pub fn hook<NH: Hook>(
        self,
        hook: NH,
    ) -> TerminalAgentBuilder<LLM, P, E, crate::HCons<NH, H>, S, Balanced, Fast, Compactor, Schema>
    {
        self.rebuild(|inner| inner.hook(hook))
    }

    /// Sets the handoff schema every compaction asks the model for; see
    /// [`AgentBuilder::handoff`].
    pub fn handoff<NSchema: crate::HandoffSchema>(
        self,
    ) -> TerminalAgentBuilder<LLM, P, E, H, S, Balanced, Fast, Compactor, NSchema> {
        self.rebuild(AgentBuilder::handoff::<NSchema>)
    }

    /// Sets the balanced tier's model and the parameters of the requests
    /// aither sends it; see [`AgentBuilder::balanced_model`].
    pub fn balanced_model<NB: LanguageModel>(
        self,
        model: NB,
        parameters: aither_core::llm::model::Parameters,
    ) -> TerminalAgentBuilder<LLM, P, E, H, S, NB, Fast, Compactor, Schema> {
        self.rebuild(|inner| inner.balanced_model(model, parameters))
    }

    /// Sets the fast tier's model and the parameters of the requests aither
    /// sends it; see [`AgentBuilder::fast_model`].
    pub fn fast_model<NF: LanguageModel>(
        self,
        model: NF,
        parameters: aither_core::llm::model::Parameters,
    ) -> TerminalAgentBuilder<LLM, P, E, H, S, Balanced, NF, Compactor, Schema> {
        self.rebuild(|inner| inner.fast_model(model, parameters))
    }

    /// Sets the model compaction runs on, instead of the session's own, and
    /// the parameters of the compaction requests it serves; see
    /// [`AgentBuilder::compaction_model`].
    pub fn compaction_model<NC: LanguageModel>(
        self,
        model: NC,
        parameters: aither_core::llm::model::Parameters,
    ) -> TerminalAgentBuilder<LLM, P, E, H, S, Balanced, Fast, NC, Schema> {
        self.rebuild(|inner| inner.compaction_model(model, parameters))
    }

    /// Loads skills from a filesystem path into the builder's skill
    /// registry, the one [`AgentOptions::skill_registry`] set or a new one.
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
        let mut registry = match self.settings().skill_registry.take() {
            Some(registry) => Arc::try_unwrap(registry).map_err(|_| {
                crate::AgentError::Config(
                    "skills load into a skill registry nothing else holds".to_string(),
                )
            })?,
            None => SkillRegistry::default(),
        };
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
        self.settings().skill_registry = Some(Arc::new(registry));

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

    /// Sets when machine load counts as high, how long it must hold to
    /// start and to end, how often the host is read,
    /// and the terminal's foreground window while high load lasts.
    ///
    /// The built agent is told once when high load starts and once when it
    /// ends; see [`HostLoadMonitor`](crate::HostLoadMonitor).
    #[must_use]
    pub const fn host_load(mut self, config: crate::HostLoadConfig) -> Self {
        self.host_load.config = config;
        self
    }

    /// Reads the host through `reader` instead of the platform's own.
    #[must_use]
    pub fn host_reader(mut self, reader: impl crate::HostReader) -> Self {
        self.host_load.reader = Some(crate::AnyHostReader::new(reader));
        self
    }

    /// Judges the free space of the home environment's volumes by
    /// `thresholds`, which the caller may change at any time through its own
    /// clone of the handle. Defaults to yellow below 10 GiB, red below 5 and
    /// final below 3.
    #[must_use]
    pub fn disk_thresholds(mut self, thresholds: crate::DiskThresholds) -> Self {
        self.host_load.thresholds = thresholds;
        self
    }

    /// Names the volume of the machine the agent's home environment is on:
    /// the one that holds `path`. Defaults to the volume holding the
    /// terminal's working directory.
    ///
    /// When the agent works in a container, the filesystem of the container
    /// is watched as well, whatever is named here; a remote execution
    /// environment is never watched.
    #[must_use]
    pub fn home_volume(mut self, path: impl AsRef<std::path::Path>) -> Self {
        self.host_load.disk_reader = Some(crate::AnyDiskReader::new(crate::SystemDisk::new(path)));
        self
    }

    /// Reads the free space of the machine's home volume through `reader`
    /// instead of the volume holding the working directory.
    #[must_use]
    pub fn disk_reader(mut self, reader: impl crate::DiskReader) -> Self {
        self.host_load.disk_reader = Some(crate::AnyDiskReader::new(reader));
        self
    }

    /// Does not watch the free space of the home environment's volumes.
    #[must_use]
    pub const fn without_disk_pressure(mut self) -> Self {
        self.host_load.disk = false;
        self
    }

    /// Does not watch the host's memory pressure or machine load.
    #[must_use]
    pub const fn without_host_load(mut self) -> Self {
        self.host_load.enabled = false;
        self
    }

    /// Ends a provider response stream in flight when the machine wakes from
    /// sleep or loses its network, and has the terminal state the sleep and
    /// offline intervals each command ran through.
    ///
    /// See [`AgentBuilder::connectivity`] and
    /// [`TerminalTool::with_connectivity`].
    #[cfg(feature = "connectivity")]
    #[must_use]
    pub fn connectivity(mut self, connectivity: aither_connectivity::Connectivity) -> Self {
        self.terminal_tool = self.terminal_tool.with_connectivity(&connectivity);
        self.inner = self.inner.connectivity(connectivity);
        self
    }

    /// Returns the list of registered tool descriptions.
    ///
    /// Useful for dynamically building system prompts.
    pub fn tool_descriptions(&self) -> &[(String, String)] {
        &self.tool_descriptions
    }

    /// Returns the definitions of the native tools the built agent exposes to
    /// the model.
    ///
    /// The list covers every tool registered on the inner [`AgentBuilder`]
    /// (`terminal_kill`, `terminal_input`, `terminal_read`) plus `terminal`,
    /// which `build()` registers last, in the order the built agent's tool
    /// registry holds them.
    ///
    /// `terminal` only implements [`Tool`] once it holds a command registry,
    /// but its definition — the name, the `TerminalArgs` schema and the
    /// description the schema carries — is independent of the registry's
    /// contents, so an empty registry yields the same definition `build()`
    /// registers.
    #[must_use]
    pub fn native_tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = self.inner.tool_definitions();
        let jobs = self.terminal_tool.job_registry();
        definitions.push(ToolDefinition::new(&KillTerminalTool::new(jobs.clone())));
        definitions.push(ToolDefinition::new(&InputTerminalTool::new(jobs.clone())));
        definitions.push(ToolDefinition::new(&ReadTerminalDeltaTool::new(
            jobs,
            aither_sandbox::AnySandboxNotices::default(),
        )));
        let terminal = self.terminal_tool.clone().with_registry(Arc::new(
            ToolRegistryBuilder::new().build(self.terminal_tool.output_store().clone()),
        ));
        definitions.push(ToolDefinition::new(&terminal));
        definitions.push(ToolDefinition::new(&aither_sandbox::MonitorTool::new(
            terminal,
        )));
        // The registry keys tools by name, so the definition list the built
        // agent sends is name-ordered.
        definitions.sort_by(|a, b| a.name().cmp(b.name()));
        definitions
    }

    /// Whether a CLI command with `name` is registered in the terminal registry.
    fn has_command(&self, name: &str) -> bool {
        self.tool_descriptions.iter().any(|(n, _)| n == name)
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
    ///
    /// # Panics
    ///
    /// Panics if the `knowledge_and_time` prompt template fails to render.
    pub fn build(self) -> Agent<LLM, Balanced, Fast, H, Compactor, Schema>
    where
        LLM: 'static,
    {
        // Render knowledge/time guidance against the actual command registry so
        // it never references a command that was not registered.
        let websearch_command = self.has_command("websearch");
        let notices = self.inner.chosen_notices().clone();

        // Build registry
        let mut registry_builder = self.registry_builder;
        registry_builder.notices(notices.sandbox());
        if let Some(subagents) = self.subagents {
            let subagents = (*subagents)
                .with_handoff::<Schema>()
                .with_notices(notices.clone())
                .with_working_document(
                    self.inner.chosen_document().cloned(),
                    self.inner.chosen_stop_hook().cloned(),
                );
            registry_builder.main_exclusive(|main| {
                main.configure_terminal_bound_usage_tool("subagent", move |binding| {
                    subagents.for_terminal(binding)
                });
            });
        }
        let registry =
            std::sync::Arc::new(registry_builder.build(self.terminal_tool.output_store().clone()));

        let commands = crate::command_catalog::CommandCatalog::new(
            self.tool_descriptions,
            registry.command_sets().to_vec(),
        );
        // Configure terminal tool
        let terminal_tool = self.terminal_tool.with_registry(registry);

        let job_registry = terminal_tool.job_registry();
        let inner = self
            .inner
            .tool(KillTerminalTool::new(job_registry.clone()))
            .tool(InputTerminalTool::new(job_registry.clone()))
            .tool(ReadTerminalDeltaTool::new(job_registry, notices.sandbox()))
            .command_catalog(commands);
        let inner = if inner.has_instruction(&InstructionSlot::KNOWLEDGE_AND_TIME) {
            inner
        } else {
            inner.system_named(
                &InstructionSlot::KNOWLEDGE_AND_TIME,
                &crate::agent::KnowledgeAndTimeTemplate { websearch_command }
                    .render()
                    .expect("failed to render knowledge_and_time template"),
            )
        };
        let host_load = self.host_load;
        let reader = host_load
            .enabled
            .then_some(host_load.reader)
            .flatten()
            .or_else(|| {
                host_load.enabled.then(|| {
                    // A host without a source is still watched where the
                    // watch does not need it: a container's memory limit.
                    match crate::SystemReader::new() {
                        Ok(reader) => crate::AnyHostReader::new(reader),
                        Err(error) => {
                            tracing::warn!(%error, "host pressure and load are not watched");
                            crate::AnyHostReader::new(crate::UnwatchedHost)
                        }
                    }
                })
            });
        let inner = match reader {
            Some(reader) => {
                let monitor = crate::HostLoadMonitor::new(
                    reader,
                    host_load.config,
                    terminal_tool.foreground_wait(),
                );
                let monitor = if host_load.disk {
                    let local = host_load.disk_reader.unwrap_or_else(|| {
                        crate::AnyDiskReader::new(crate::SystemDisk::new(
                            terminal_tool.working_dir(),
                        ))
                    });
                    monitor.with_disk(
                        host_load.thresholds,
                        crate::disk::home_volumes(local, terminal_tool.container_probe()),
                    )
                } else {
                    monitor
                };
                inner.host_load(
                    terminal_tool.executor(),
                    monitor,
                    Some(terminal_tool.job_memory()),
                )
            }
            None => inner,
        };
        inner
            .terminal(terminal_tool.clone())
            .tool(aither_sandbox::MonitorTool::new(terminal_tool))
            .build()
    }
}

impl<LLM, P, E, H, Balanced, Fast, Compactor, Schema>
    TerminalAgentBuilder<LLM, P, E, H, Unprepared, Balanced, Fast, Compactor, Schema>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    Compactor: LanguageModel,
    H: Hook,
    Schema: crate::HandoffSchema,
{
    /// Sets runtime shell backend availability for `terminal`.
    #[must_use]
    pub fn shell_runtime_availability(mut self, availability: ShellRuntimeAvailability) -> Self {
        self.terminal_tool = self
            .terminal_tool
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
        let sessions = self
            .terminal_tool
            .shell_sessions()
            .clone()
            .with_ssh_servers(servers)
            .expect("invalid ssh server configuration");
        self.terminal_tool = self.terminal_tool.with_shell_sessions(sessions);
        self
    }

    /// Sets the SSH session authorizer for interactive connect/install consent prompts.
    #[must_use]
    pub fn ssh_authorizer<T: SshSessionAuthorizer + 'static>(mut self, authorizer: Arc<T>) -> Self {
        let sessions = self
            .terminal_tool
            .shell_sessions()
            .clone()
            .with_ssh_authorizer(authorizer);
        self.terminal_tool = self.terminal_tool.with_shell_sessions(sessions);
        self
    }

    /// Sets the container runtime for container-backed shell sessions.
    #[must_use]
    pub fn container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.terminal_tool = self.terminal_tool.with_container_runtime(runtime);
        self
    }

    /// Prepares baseline tools for the configured backends before building the agent.
    ///
    /// The prepared registry is available afterwards from
    /// [`TerminalAgentBuilder::shell_sessions`], so an application that
    /// rebuilds its agent can adopt it with [`Self::prepared_shell_sessions`]
    /// instead of preparing again.
    ///
    /// # Errors
    /// Fails if provisioning, integrity validation or sandbox protection fails.
    pub async fn initialize_tools(
        mut self,
        distribution: &ToolDistribution,
        cache_root: &std::path::Path,
    ) -> Result<
        TerminalAgentBuilder<LLM, P, E, H, Prepared, Balanced, Fast, Compactor, Schema>,
        ToolRuntimeError,
    > {
        self.terminal_tool = self
            .terminal_tool
            .initialize_tools(distribution, cache_root)
            .await?;
        Ok(self.rebuild(std::convert::identity))
    }

    /// Adopts a registry another builder already prepared with
    /// [`Self::initialize_tools`], taking its shell backends with it.
    ///
    /// Only this terminal's own checks run: its working directory and writable
    /// paths must not expose the managed tools. Downloads, integrity hashing,
    /// version probes and SSH bootstrap — including its connection consent
    /// prompt — are not repeated.
    ///
    /// # Errors
    /// Rejects a registry that is not prepared, one whose home backend,
    /// container or SSH server set changed after preparation, and writable
    /// grants that expose the managed tools.
    pub async fn prepared_shell_sessions(
        mut self,
        sessions: ShellSessionRegistry,
    ) -> Result<
        TerminalAgentBuilder<LLM, P, E, H, Prepared, Balanced, Fast, Compactor, Schema>,
        ToolRuntimeError,
    > {
        self.terminal_tool = self.terminal_tool.adopt_prepared_tools(sessions).await?;
        Ok(self.rebuild(std::convert::identity))
    }
}

impl<LLM, P, E, H, Balanced, Fast, Compactor, Schema>
    TerminalAgentBuilder<LLM, P, E, H, Prepared, Balanced, Fast, Compactor, Schema>
where
    LLM: LanguageModel + Clone,
    P: PermissionHandler + 'static,
    E: Executor + Clone + 'static,
{
    /// The prepared registry, for a later builder to adopt with
    /// [`TerminalAgentBuilder::prepared_shell_sessions`].
    #[must_use]
    pub const fn shell_sessions(&self) -> &ShellSessionRegistry {
        self.terminal_tool.shell_sessions()
    }
}

fn push_bullet_line(output: &mut String, body: &str) {
    if !output.is_empty() {
        output.push('\n');
    }
    output.push_str("- ");
    output.push_str(body);
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
                    .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
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
            .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string());
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

    #[test]
    fn shell_runtime_context_is_prose() {
        let rendered = ShellRuntimeContextTemplate {
            local: true,
            ssh: false,
            runtime: "user_local_machine",
        }
        .render()
        .unwrap();
        assert_eq!(
            rendered,
            "Shell runtime: user_local_machine. Local backend: available. SSH backend: unavailable."
        );
    }
    #[test]
    fn the_prompt_carries_no_working_document_protocol() {
        let prompt = SystemPrompt {
            os: String::new(),
            os_version: String::new(),
            arch: "",
            user_cwd: String::new(),
            sandbox_dir: String::new(),
            host_profile: "heel",
            host_runtime_context: String::new(),
            skills: String::new(),
            has_skills: false,
            is_macos: false,
            has_websearch_command: false,
        }
        .render()
        .unwrap();
        // The embedder states the protocol of its own working document.
        for needle in ["tasks.md", "BLOCKING", "## Doing"] {
            assert!(!prompt.contains(needle), "{needle:?} in {prompt}");
        }
    }

    use std::borrow::Cow;

    use crate::test_support::{compile_baseline_fixture, fixture_distribution};
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

    impl aither_core::llm::Classify for MockError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    #[derive(Clone)]
    struct MockLlm {
        requests: async_channel::Sender<LLMRequest>,
    }

    impl LanguageModel for MockLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests
                .try_send(request)
                .expect("record model request");
            futures_lite::stream::empty()
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                100_000,
            )))
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
            _cx: aither_core::llm::ToolContext,
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
        assert_ne!(os_name, "");
        assert_ne!(os_version, "");
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
            host_profile: "container",
            host_runtime_context: "runtime=linux_container".to_string(),
            skills: String::new(),
            has_skills: false,
            is_macos: true,
            has_websearch_command: true,
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
            host_profile: "heel",
            host_runtime_context: "runtime=user_local_machine".to_string(),
            skills: String::new(),
            has_skills: false,
            is_macos: true,
            has_websearch_command: true,
        }
        .render()
        .expect("failed to render heel prompt");

        assert!(prompt.contains("<shell-modes>"));
        assert!(!prompt.contains("<shell-runtime>"));
        assert!(prompt.contains("<unsafe>"));
    }

    async fn noop_terminal(
        dir: &std::path::Path,
    ) -> aither_sandbox::TerminalTool<NoopPermissionHandler, TokioGlobal> {
        aither_sandbox::TerminalTool::new_exact(dir, NoopPermissionHandler, TokioGlobal)
            .await
            .expect("terminal tool should initialize")
    }

    /// Asks for one terminal call running every baseline command, then stops.
    #[derive(Clone)]
    struct BaselineCommandLlm {
        requests: async_channel::Sender<LLMRequest>,
        turns: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LanguageModel for BaselineCommandLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests
                .try_send(request)
                .expect("record model request");
            let turn = self.turns.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let event = if turn == 0 {
                Event::ToolCall(aither_core::llm::ToolCall::new(
                    "baseline",
                    "terminal",
                    serde_json::json!({
                        "description": "Run every baseline command",
                        "script": "apply_patch --version && rg --version && ast-grep --version && jq --version",
                        "mode": "sandboxed",
                        "raw": true,
                        "timeout": 30
                    }),
                ))
            } else {
                Event::Text("done".into())
            };
            futures_lite::stream::once(Ok(event))
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                100_000,
            )))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rebuilt_agent_adopts_prepared_tools_and_runs_baseline_commands() {
        let source = tempdir().unwrap();
        let binary = compile_baseline_fixture(source.path()).await;
        let distribution = fixture_distribution(&binary, "1.0.0").await;
        let cache = crate::test_support::TempDir::new();
        let (requests, _captured) = async_channel::bounded(4);
        let first_dir = tempdir().unwrap();
        let first =
            TerminalAgentBuilder::new(MockLlm { requests }, noop_terminal(first_dir.path()).await)
                .initialize_tools(&distribution, cache.path())
                .await
                .expect("first agent prepares its tools");
        let prepared = first.shell_sessions().clone();
        let _first_agent = first.build();

        let (requests, captured) = async_channel::bounded(4);
        let second_dir = tempdir().unwrap();
        let mut second = TerminalAgentBuilder::new(
            BaselineCommandLlm {
                requests,
                turns: Arc::default(),
            },
            noop_terminal(second_dir.path()).await,
        )
        .max_iterations(2)
        .with_default_prompt()
        .prepared_shell_sessions(prepared)
        .await
        .expect("second agent adopts the prepared registry")
        .build();
        let _ =
            crate::test_support::finish_run(second.run("Check the baseline commands", [])).await;

        let first_request = captured
            .try_recv()
            .expect("first model request was recorded");
        assert!(
            first_request
                .messages()
                .iter()
                .any(|message| message.content().contains("<prepared_tools>")),
            "the adopted catalog reaches the model"
        );
        let second_request = captured
            .try_recv()
            .expect("the terminal result was sent back to the model");
        let transcript = second_request
            .messages()
            .iter()
            .map(aither_core::llm::Message::content)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            transcript.contains("baseline-tool 1.0.0"),
            "baseline commands ran from the adopted tools: {transcript}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn adoption_skips_the_version_probe_that_preparation_runs() {
        let source = tempdir().unwrap();
        let binary = compile_baseline_fixture(source.path()).await;
        // The fixture reports 1.0.0, so the probe rejects a 2.0.0 pin while
        // downloads and integrity checks accept it.
        let distribution = fixture_distribution(&binary, "2.0.0").await;
        let cache = crate::test_support::TempDir::new();
        let (requests, _captured) = async_channel::bounded(4);

        let probed_dir = tempdir().unwrap();
        let probed = TerminalAgentBuilder::new(
            MockLlm {
                requests: requests.clone(),
            },
            noop_terminal(probed_dir.path()).await,
        )
        .initialize_tools(&distribution, cache.path())
        .await;
        assert!(matches!(
            probed,
            Err(aither_sandbox::ToolRuntimeError::VersionMismatch(_))
        ));

        let prepared = ShellSessionRegistry::new(ShellRuntimeAvailability::default())
            .initialize_tools(&distribution, cache.path())
            .await
            .expect("registry preparation runs no terminal probe");
        let adopted_dir = tempdir().unwrap();
        TerminalAgentBuilder::new(
            MockLlm { requests },
            noop_terminal(adopted_dir.path()).await,
        )
        .prepared_shell_sessions(prepared)
        .await
        .expect("adoption does not probe versions again")
        .build();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_agent_builder_exposes_native_terminal_controls_and_keeps_ipc_tools_out() {
        let dir = tempdir().expect("tempdir should exist");
        let terminal_tool = noop_terminal(dir.path()).await;
        let source = tempdir().unwrap();
        let cache = crate::test_support::TempDir::new();
        let binary = compile_baseline_fixture(source.path()).await;
        let distribution = fixture_distribution(&binary, "1.0.0").await;
        let (requests, captured) = async_channel::bounded(4);
        let builder = TerminalAgentBuilder::new(MockLlm { requests }, terminal_tool)
            .max_iterations(1)
            .tool(MockTool)
            .with_default_prompt()
            .initialize_tools(&distribution, cache.path())
            .await
            .unwrap();
        let mut catalog_names = builder
            .tool_descriptions()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        catalog_names.sort_unstable();
        assert_eq!(catalog_names, ["mock_tool"]);
        let mut agent = builder.build();

        let mut tool_names = agent
            .tools
            .definitions()
            .into_iter()
            .map(|definition| definition.name().to_string())
            .collect::<Vec<_>>();
        tool_names.sort();

        assert_eq!(
            tool_names,
            [
                "compact",
                "monitor",
                "terminal",
                "terminal_input",
                "terminal_kill",
                "terminal_read"
            ]
        );
        let _ = crate::test_support::finish_run(agent.run("Inspect the repository", [])).await;
        let request = captured
            .try_recv()
            .expect("first model request was recorded");
        assert_eq!(
            request
                .tool_definitions()
                .iter()
                .map(ToolDefinition::name)
                .collect::<Vec<_>>(),
            vec![
                "compact",
                "monitor",
                "terminal",
                "terminal_input",
                "terminal_kill",
                "terminal_read"
            ]
        );
        let catalog = request
            .messages()
            .iter()
            .find(|message| message.content().contains("<prepared_tools>"))
            .expect("first request contains prepared catalog")
            .content();
        for command in ["apply_patch", "rg", "ast-grep", "jq"] {
            assert!(catalog.contains(command));
        }
    }

    /// The parameters a host gives `TerminalAgentBuilder` reach the provider
    /// on the agent's turns, and a compaction model's own on its compaction,
    /// each beside the cache controls and structured output aither sets.
    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_agent_requests_carry_the_hosts_parameters_under_aithers_fields() {
        use crate::request_parameters::test_support::{
            assert_cache_controls, host_parameters, host_part,
        };
        use aither_core::llm::model::ReasoningEffort;

        let dir = tempdir().expect("tempdir should exist");
        let source = tempdir().unwrap();
        let cache = crate::test_support::TempDir::new();
        let binary = compile_baseline_fixture(source.path()).await;
        let distribution = fixture_distribution(&binary, "1.0.0").await;
        let session = host_parameters(ReasoningEffort::High, 0.3);
        let compacting = host_parameters(ReasoningEffort::Low, 0.9);
        let (requests, turns) = async_channel::unbounded();
        let (compactor_requests, compactions) = async_channel::unbounded();
        let mut agent =
            TerminalAgentBuilder::new(MockLlm { requests }, noop_terminal(dir.path()).await)
                .request_parameters(session.clone())
                .compaction_model(
                    MockLlm {
                        requests: compactor_requests,
                    },
                    compacting.clone(),
                )
                .max_iterations(1)
                .with_default_prompt()
                .initialize_tools(&distribution, cache.path())
                .await
                .expect("tools prepare")
                .build();
        let _ = crate::test_support::finish_run(agent.run("Inspect the repository", [])).await;
        // The scripted compactor answers nothing, so the compaction fails
        // once its request is out; only that request matters here.
        let _ = agent.compact(None).await;

        let turn = turns.try_recv().expect("the turn reached the model");
        assert_eq!(host_part(&turn), session, "the host's parameters arrive");
        assert_cache_controls(&turn);
        assert!(!turn.parameters().structured_outputs);

        let compaction = compactions
            .try_recv()
            .expect("the compaction reached the compaction model");
        assert_eq!(
            host_part(&compaction),
            compacting,
            "the compaction model's own parameters arrive"
        );
        assert_cache_controls(&compaction);
        assert!(compaction.parameters().structured_outputs);
    }

    /// The agent options reach a terminal agent through the same trait as
    /// `AgentBuilder`, and the steps that change the builder's type keep
    /// them.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_terminal_agent_runs_on_the_options_the_host_sets() {
        struct NoopHook;
        impl Hook for NoopHook {}

        let source = tempdir().unwrap();
        let binary = compile_baseline_fixture(source.path()).await;
        let distribution = fixture_distribution(&binary, "1.0.0").await;
        let cache = crate::test_support::TempDir::new();
        let dir = tempdir().unwrap();
        let (requests, _captured) = async_channel::unbounded();
        let policy = crate::ContextPolicy {
            compact_at: 0.5,
            compact_ceiling: 300_000,
            start_distance: 20_000,
            safe_buffer: 5_000,
            ..crate::ContextPolicy::default()
        };
        let agent = TerminalAgentBuilder::new(
            MockLlm {
                requests: requests.clone(),
            },
            noop_terminal(dir.path()).await,
        )
        .context_policy(policy)
        .tier(crate::ModelTier::Balanced)
        .persona_prompt("persona")
        .hook(NoopHook)
        .balanced_model(
            MockLlm {
                requests: requests.clone(),
            },
            aither_core::llm::model::Parameters::default(),
        )
        .compaction_model(
            MockLlm { requests },
            aither_core::llm::model::Parameters::default(),
        )
        .initialize_tools(&distribution, cache.path())
        .await
        .unwrap()
        .build();

        assert_eq!(agent.config.context_policy, policy);
        assert_eq!(agent.tier, crate::ModelTier::Balanced);
        assert_eq!(agent.config.persona_prompt.as_deref(), Some("persona"));
        assert!(agent.compaction_model.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn native_tool_definitions_match_the_built_agents_tool_definitions() {
        let source = tempdir().unwrap();
        let binary = compile_baseline_fixture(source.path()).await;
        let distribution = fixture_distribution(&binary, "1.0.0").await;
        let cache = crate::test_support::TempDir::new();
        let dir = tempdir().unwrap();
        let (requests, _captured) = async_channel::bounded(4);
        let builder =
            TerminalAgentBuilder::new(MockLlm { requests }, noop_terminal(dir.path()).await)
                .initialize_tools(&distribution, cache.path())
                .await
                .unwrap();
        let expected = builder.native_tool_definitions();
        let agent = builder.build();
        assert_eq!(expected, agent.tools.active_definitions());
    }

    /// Asks for one native `terminal_kill` call, then stops.
    #[derive(Clone)]
    struct KillTaskLlm {
        requests: async_channel::Sender<LLMRequest>,
        turns: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LanguageModel for KillTaskLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests
                .try_send(request)
                .expect("record model request");
            let turn = self.turns.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let event = if turn == 0 {
                Event::ToolCall(aither_core::llm::ToolCall::new(
                    "stop",
                    "terminal_kill",
                    serde_json::json!({ "task_id": "background" }),
                ))
            } else {
                Event::Text("done".into())
            };
            futures_lite::stream::once(Ok(event))
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                100_000,
            )))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn native_terminal_kill_stops_a_job_of_the_agents_terminal() {
        let source = tempdir().unwrap();
        let binary = compile_baseline_fixture(source.path()).await;
        let distribution = fixture_distribution(&binary, "1.0.0").await;
        let cache = crate::test_support::TempDir::new();
        let dir = tempdir().unwrap();
        let terminal_tool = noop_terminal(dir.path()).await;
        let jobs = terminal_tool.job_registry();
        jobs.register(
            424_242,
            &aither_sandbox::test_ids::lease("background"),
            "execution",
            "sleep 60",
            aither_sandbox::TerminalMode::Sandboxed,
            None,
        )
        .await;
        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        jobs.attach_kill_switch(424_242, kill_tx).await;
        let (requests, captured) = async_channel::bounded(8);
        let mut agent = TerminalAgentBuilder::new(
            KillTaskLlm {
                requests,
                turns: Arc::default(),
            },
            terminal_tool,
        )
        .max_iterations(3)
        .initialize_tools(&distribution, cache.path())
        .await
        .unwrap()
        .build();

        let _ = crate::test_support::finish_run(agent.run("Stop the background task", [])).await;

        kill_rx
            .try_recv()
            .expect("the native terminal_kill call reached the terminal's job");
        assert!(!jobs.has_running().await);
        let _ = captured.try_recv().expect("first request");
        let reply = captured
            .try_recv()
            .expect("the kill result reached the model");
        let transcript = reply
            .messages()
            .iter()
            .map(aither_core::llm::Message::content)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            transcript.contains("' terminated."),
            "native terminal_kill reported the kill: {transcript}"
        );
    }

    /// An embedder's policy refusing a foreground `sleep` in its own words.
    struct NoSleep;

    impl aither_sandbox::CommandPolicy for NoSleep {
        type State = ();

        fn check(
            &self,
            script: &aither_sandbox::ScriptCheck<'_>,
            (): &(),
        ) -> impl std::future::Future<Output = aither_sandbox::PolicyDecision> + Send {
            let sleeps = matches!(
                script.shell,
                Some(Ok(shell)) if shell
                    .commands()
                    .iter()
                    .any(|command| !command.is_background() && command.program() == "sleep")
            );
            std::future::ready(if sleeps {
                aither_sandbox::PolicyDecision::Refuse(aither_sandbox::PolicyRefusal::new(
                    "Not run: wait for the job's completion instead of sleeping.",
                ))
            } else {
                aither_sandbox::PolicyDecision::Allow
            })
        }
    }

    /// Asks for one foreground `sleep`, then stops.
    #[derive(Clone)]
    struct SleepLlm {
        requests: async_channel::Sender<LLMRequest>,
        turns: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LanguageModel for SleepLlm {
        type Error = MockError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests
                .try_send(request)
                .expect("record model request");
            let turn = self.turns.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let event = if turn == 0 {
                Event::ToolCall(aither_core::llm::ToolCall::new(
                    "wait",
                    "terminal",
                    serde_json::json!({
                        "description": "Wait for the build",
                        "script": "sleep 5",
                        "mode": "sandboxed",
                        "timeout": 30
                    }),
                ))
            } else {
                Event::Text("done".into())
            };
            futures_lite::stream::once(Ok(event))
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "mock-model",
                "mock model",
                100_000,
            )))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_command_policy_refusal_reaches_the_conversation_in_the_embedders_words() {
        let source = tempdir().unwrap();
        let binary = compile_baseline_fixture(source.path()).await;
        let distribution = fixture_distribution(&binary, "1.0.0").await;
        let cache = crate::test_support::TempDir::new();
        let dir = tempdir().unwrap();
        let (requests, captured) = async_channel::bounded(8);
        let mut builder = TerminalAgentBuilder::new(
            SleepLlm {
                requests,
                turns: Arc::default(),
            },
            noop_terminal(dir.path()).await,
        )
        .max_iterations(3);
        builder.tool_registry_mut().command_policy(NoSleep);
        let mut agent = builder
            .initialize_tools(&distribution, cache.path())
            .await
            .unwrap()
            .build();

        let _ = crate::test_support::finish_run(agent.run("Wait for the build", [])).await;

        let _ = captured.try_recv().expect("first request");
        let reply = captured.try_recv().expect("the refusal reached the model");
        let transcript = reply
            .messages()
            .iter()
            .map(aither_core::llm::Message::content)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            transcript.contains("Not run: wait for the job's completion instead of sleeping."),
            "the embedder's refusal reached the model: {transcript}"
        );
    }
}
