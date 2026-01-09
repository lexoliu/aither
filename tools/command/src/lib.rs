use std::{borrow::Cow, path::PathBuf};

use aither_core::llm::{Tool, ToolOutput, tool::json};
use anyhow::{Context, Result, bail};
use async_process::Command;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Execute a shell command with arguments.
///
/// Runs a program with the specified arguments and returns stdout, stderr,
/// and exit code. Use this for executing system commands when you need
/// fine-grained control over arguments.
///
/// For complex scripts with pipelines, use `bash` instead.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommandArgs {
    /// Program name to execute (e.g., "ls", "grep", "git", "cargo").
    pub program: String,
    /// Command-line arguments as separate strings (e.g., ["-la"] for ls, ["-r", "TODO", "src/"] for grep).
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory for command execution. Omit to use default.
    pub cwd: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommandOutput {
    pub program: String,
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone)]
pub struct CommandTool {
    allowed: Option<Vec<String>>,
    default_cwd: PathBuf,
    max_output: usize,
    name: String,
}

impl CommandTool {
    pub fn new(default_cwd: impl Into<PathBuf>) -> Self {
        let default_cwd = default_cwd.into();
        Self {
            allowed: None,
            default_cwd,
            max_output: 16 * 1024,
            name: "command".into(),
        }
    }

    pub fn restrict_to(mut self, commands: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowed = Some(commands.into_iter().map(Into::into).collect());
        self
    }

    pub fn max_output(mut self, bytes: usize) -> Self {
        self.max_output = bytes;
        self
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    fn ensure_allowed(&self, program: &str) -> Result<()> {
        if let Some(allowed) = &self.allowed
            && !allowed.iter().any(|entry| entry == program)
        {
            bail!(
                "Program '{program}' is not allowed. Allowed commands: {}",
                allowed.join(", ")
            );
        }
        Ok(())
    }

    fn truncate(&self, text: String) -> String {
        if text.len() <= self.max_output {
            return text;
        }

        let mut truncated = text;
        truncated.truncate(self.max_output);
        truncated.push_str("\n...output truncated...");
        truncated
    }
}

impl Tool for CommandTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    type Arguments = CommandArgs;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        self.ensure_allowed(&arguments.program)?;

        let working_dir = arguments.cwd.unwrap_or_else(|| self.default_cwd.clone());
        let output = Command::new(&arguments.program)
            .args(&arguments.args)
            .current_dir(&working_dir)
            .output()
            .await
            .with_context(|| {
                format!(
                    "failed to execute '{}' in {}",
                    arguments.program,
                    working_dir.display()
                )
            })?;

        let response = CommandOutput {
            program: arguments.program,
            status: output.status.code().unwrap_or_default(),
            stdout: self.truncate(String::from_utf8_lossy(&output.stdout).to_string()),
            stderr: self.truncate(String::from_utf8_lossy(&output.stderr).to_string()),
        };

        Ok(ToolOutput::text(json(&response)))
    }
}
