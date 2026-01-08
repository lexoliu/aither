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
//!     .ipc_tool(WebSearchTool::default())  // Becomes `websearch` bash command
//!     .ipc_tool(WebFetchTool::new())       // Becomes `webfetch` bash command
//!     .ipc_tool(TaskTool::new(llm))        // Becomes `task` bash command
//!     .build();
//! ```

use std::borrow::Cow;

use aither_core::llm::Tool;
use aither_core::LanguageModel;
use askama::Template;
use executor_core::Executor;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::hook::Hook;
use crate::{Agent, AgentBuilder};
use aither_sandbox::{configure_tool, BashTool, PermissionHandler};

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
    is_macos: bool,
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
    bash_tool: BashTool<P, E>,
    tool_descriptions: Vec<(String, String)>,
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
    /// Use `.ipc_tool()` to register tools as bash commands.
    pub fn new(llm: LLM, bash_tool: BashTool<P, E>) -> Self {
        // Set up bash tool factory for subagents
        {
            let bash_for_factory = bash_tool.clone();
            aither_sandbox::set_bash_tool_factory(move || {
                bash_for_factory.child().to_dyn()
            });
        }

        Self {
            inner: AgentBuilder::new(llm),
            bash_tool,
            tool_descriptions: Vec::new(),
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
    /// builder.ipc_tool(WebSearchTool::default())
    /// // LLM can now call: bash -c 'websearch "rust async"'
    /// ```
    pub fn ipc_tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let name = tool.name().to_string();
        let description = tool.description();
        // Truncate description to first sentence
        let short_desc = description
            .split('.')
            .next()
            .unwrap_or(&description)
            .trim()
            .to_string();

        self.tool_descriptions.push((name, short_desc));
        configure_tool(tool);
        self
    }

    /// Registers a tool with a custom description.
    pub fn ipc_tool_with_desc<T>(mut self, tool: T, description: impl Into<String>) -> Self
    where
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let name = tool.name().to_string();
        self.tool_descriptions.push((name, description.into()));
        configure_tool(tool);
        self
    }

    /// Adds a pre-configured tool description (for tools registered elsewhere).
    ///
    /// Use this when the tool was already registered via `configure_tool()`
    /// or `configure_raw_handler()`.
    pub fn tool_description(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.tool_descriptions.push((name.into(), description.into()));
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
        let tools = self.tool_descriptions
            .iter()
            .map(|(name, desc)| format!("- {name}: {desc}"))
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
            is_macos,
        };

        let prompt = template.render().expect("failed to render system prompt template");
        self.inner = self.inner.system_prompt(prompt);
        self
    }

    /// Adds a hook to intercept agent operations.
    pub fn hook<NH: Hook>(self, hook: NH) -> BashAgentBuilder<LLM, P, E, crate::HCons<NH, H>> {
        BashAgentBuilder {
            inner: self.inner.hook(hook),
            bash_tool: self.bash_tool,
            tool_descriptions: self.tool_descriptions,
        }
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

    /// Returns the sandbox working directory path.
    pub fn sandbox_dir(&self) -> Cow<'_, str> {
        self.bash_tool.working_dir().to_string_lossy()
    }

    /// Builds the agent.
    ///
    /// The returned agent only has the `bash` tool.
    /// All registered IPC tools are accessible as bash commands.
    pub fn build(self) -> Agent<LLM, LLM, LLM, H> {
        self.inner.bash(self.bash_tool).build()
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
