use std::{borrow::Cow, path::PathBuf};

use aither_core::llm::{Tool, tool::json};
use anyhow::{Context, Result, bail};
use async_process::Command;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommandArgs {
    /// Executable name only (e.g., "ls", "cat"). No arguments here.
    pub program: String,
    /// Arguments array (e.g., ["-l", "-a"] for "ls -la").
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory (optional).
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
    description: String,
}

impl CommandTool {
    pub fn new(default_cwd: impl Into<PathBuf>) -> Self {
        let default_cwd = default_cwd.into();
        Self {
            allowed: None,
            default_cwd,
            max_output: 16 * 1024,
            name: "command_runner".into(),
            description: "Executes shell commands inside a sandboxed environment.".into(),
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

    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
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

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = CommandArgs;

    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
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

        Ok(json(&response))
    }
}
