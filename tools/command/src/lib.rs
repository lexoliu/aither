use std::{borrow::Cow, fmt, path::PathBuf, process::ExitStatus};

use aither_core::llm::ToolContext;
use aither_core::llm::{Tool, ToolResult};
use anyhow::{Context, Result, bail};
use askama::Template;
use async_process::Command;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Execute a shell command with arguments.
///
/// Runs a program with the specified arguments and returns stdout, stderr,
/// and exit code. Use this for executing system commands when you need
/// fine-grained control over arguments.
///
/// For complex scripts with pipelines, use `terminal` instead.
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

/// What a finished command wrote, rendered for the model by `templates/output.txt`.
#[derive(Debug, Clone, Template)]
#[template(path = "output.txt", escape = "none")]
pub struct CommandOutput {
    pub program: String,
    pub termination: Termination,
    pub stdout: String,
    pub stderr: String,
}

/// How a command ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Termination {
    /// The process exited with this status code.
    Exited(i32),
    /// The process was killed by this signal number.
    Signaled(i32),
}

impl Termination {
    /// Classify a finished process.
    ///
    /// # Errors
    /// Fails when the process neither exited with a code nor was killed by a
    /// signal, which the platform does not report.
    pub fn of(status: ExitStatus) -> Result<Self> {
        if let Some(code) = status.code() {
            return Ok(Self::Exited(code));
        }
        #[cfg(unix)]
        if let Some(signal) = std::os::unix::process::ExitStatusExt::signal(&status) {
            return Ok(Self::Signaled(signal));
        }
        bail!("process ended with neither an exit code nor a signal: {status}")
    }
}

impl fmt::Display for Termination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exited(code) => write!(f, "exited with status {code}"),
            Self::Signaled(number) => {
                #[cfg(unix)]
                if let Ok(signal) = nix::sys::signal::Signal::try_from(*number) {
                    return write!(f, "was killed by signal {signal} ({number})");
                }
                write!(f, "was killed by signal {number}")
            }
        }
    }
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

        // Cut at the last character boundary within the limit.
        let mut end = self.max_output;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let mut truncated = text;
        truncated.truncate(end);
        truncated.push_str("\n...output truncated...");
        truncated
    }
}

impl Tool for CommandTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    type Arguments = CommandArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
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
            termination: Termination::of(output.status)?,
            stdout: self.truncate(String::from_utf8_lossy(&output.stdout).to_string()),
            stderr: self.truncate(String::from_utf8_lossy(&output.stderr).to_string()),
        };

        Ok(ToolResult::text(response.render()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run(tool: &CommandTool, program: &str, args: &[&str]) -> String {
        let result = tool
            .call(
                CommandArgs {
                    program: program.into(),
                    args: args.iter().map(ToString::to_string).collect(),
                    cwd: None,
                },
                ToolContext::new(),
            )
            .await
            .unwrap();
        match result {
            ToolResult::Text { text } => text,
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn truncation_stays_on_character_boundaries() {
        // 'é' is two bytes: every odd limit lands inside a character.
        let text = "é".repeat(8);
        for limit in 0..=text.len() {
            let tool = CommandTool::new(".").max_output(limit);
            let cut = tool.truncate(text.clone());
            if limit >= text.len() {
                assert_eq!(cut, text);
            } else {
                let kept = cut.strip_suffix("\n...output truncated...").unwrap();
                assert_eq!(kept.len(), limit - limit % 2);
            }
        }
        let tool = CommandTool::new(".").max_output(1);
        assert_eq!(tool.truncate("日本語".into()), "\n...output truncated...");
    }

    #[tokio::test]
    async fn long_non_ascii_output_is_truncated_without_panic() {
        let tool = CommandTool::new(".").max_output(5);
        let text = run(&tool, "printf", &["日本語日本語"]).await;
        assert!(
            text.contains("stdout:\n日\n...output truncated..."),
            "{text}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn signal_termination_is_reported_with_the_signal() {
        let tool = CommandTool::new(".");
        let text = run(&tool, "sh", &["-c", "kill -KILL $$"]).await;
        assert!(
            text.starts_with("sh was killed by signal SIGKILL (9)."),
            "{text}"
        );
        assert!(!text.contains("status 0"), "{text}");
    }

    #[tokio::test]
    async fn exit_status_is_still_reported() {
        let tool = CommandTool::new(".");
        let text = run(&tool, "sh", &["-c", "exit 3"]).await;
        assert!(text.starts_with("sh exited with status 3."), "{text}");
    }

    #[test]
    fn output_is_prose_with_labelled_streams() {
        let output = CommandOutput {
            program: "ls".into(),
            termination: Termination::Exited(1),
            stdout: "a\nb".into(),
            stderr: "denied".into(),
        };
        assert_eq!(
            output.render().unwrap(),
            "ls exited with status 1.\nstdout:\na\nb\nstderr:\ndenied\n"
        );
    }

    #[test]
    fn silent_command_says_so() {
        let output = CommandOutput {
            program: "true".into(),
            termination: Termination::Exited(0),
            stdout: String::new(),
            stderr: String::new(),
        };
        assert_eq!(
            output.render().unwrap(),
            "true exited with status 0.\n(no output)\n"
        );
    }
}
