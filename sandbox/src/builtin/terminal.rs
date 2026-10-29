//! Native terminal control tools for background terminal tasks.

use std::borrow::Cow;

use aither_core::llm::ToolContext;
use aither_core::llm::{CommandStream, Tool, ToolResult};
use askama::Template;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::job_registry::{JobRegistry, JobStatus};

/// Tool for terminating a background terminal task.
#[derive(Debug, Clone)]
pub struct KillTerminalTool {
    registry: JobRegistry,
}

impl KillTerminalTool {
    /// Creates a terminal kill tool.
    #[must_use]
    pub const fn new(registry: JobRegistry) -> Self {
        Self { registry }
    }
}

/// Terminate a background terminal task by its task id.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KillTerminalArgs {
    /// Task identifier returned by terminal when the task is backgrounded.
    pub task_id: String,
}

impl Tool for KillTerminalTool {
    fn name(&self) -> Cow<'static, str> {
        "terminal_kill".into()
    }

    type Arguments = KillTerminalArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = args.task_id.trim();
        if task_id.is_empty() {
            return Err(anyhow::anyhow!("task_id must not be empty"));
        }

        let killed = self.registry.kill_by_task_id(task_id).await;
        let text = TerminalKillTemplate { task_id, killed }
            .render()
            .expect("terminal kill template must render");
        Ok(ToolResult::text(text))
    }
}

/// Tool for sending stdin to a background terminal task.
#[derive(Debug, Clone)]
pub struct InputTerminalTool {
    registry: JobRegistry,
}

impl InputTerminalTool {
    /// Creates a terminal input tool.
    #[must_use]
    pub const fn new(registry: JobRegistry) -> Self {
        Self { registry }
    }
}

/// Send input to a background terminal task that is waiting on stdin.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InputTerminalArgs {
    /// Task identifier returned by terminal when the task is backgrounded.
    pub task_id: String,
    /// Raw bytes encoded as UTF-8 text written to terminal stdin.
    pub input: String,
    /// Append a trailing newline before writing (default true).
    #[serde(default = "default_append_newline")]
    pub append_newline: bool,
}

const fn default_append_newline() -> bool {
    true
}

impl Tool for InputTerminalTool {
    fn name(&self) -> Cow<'static, str> {
        "terminal_input".into()
    }

    type Arguments = InputTerminalArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = args.task_id.trim();
        if task_id.is_empty() {
            return Err(anyhow::anyhow!("task_id must not be empty"));
        }

        let mut bytes = args.input.into_bytes();
        if args.append_newline {
            bytes.push(b'\n');
        }

        self.registry
            .terminal_input(task_id, bytes)
            .await
            .map_err(anyhow::Error::msg)?;

        let text = TerminalInputTemplate { task_id }
            .render()
            .expect("terminal input template must render");
        Ok(ToolResult::text(text))
    }
}

/// Tool for reading incremental output from a background terminal task.
#[derive(Debug, Clone)]
pub struct ReadTerminalDeltaTool {
    registry: JobRegistry,
}

impl ReadTerminalDeltaTool {
    /// Creates a terminal delta reader tool.
    #[must_use]
    pub const fn new(registry: JobRegistry) -> Self {
        Self { registry }
    }
}

/// Arguments for [`ReadTerminalDeltaTool`].
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReadTerminalDeltaArgs {
    /// Task identifier returned by terminal when the task is backgrounded.
    pub task_id: String,
    /// Read only bytes after this cursor offset.
    #[serde(default)]
    pub cursor: usize,
    /// Maximum bytes to return in this call.
    #[serde(default = "default_max_bytes")]
    pub max_bytes: usize,
}

const fn default_max_bytes() -> usize {
    32 * 1024
}

fn job_status_text(status: &JobStatus) -> String {
    match status {
        JobStatus::Running => "is running".to_string(),
        JobStatus::Completed { exit_code } => format!("completed with exit code {exit_code}"),
        JobStatus::Failed { error } => format!("failed: {error}"),
        JobStatus::Killed => "was killed".to_string(),
    }
}

#[derive(Template)]
#[template(path = "terminal_kill.txt", escape = "none")]
struct TerminalKillTemplate<'a> {
    task_id: &'a str,
    killed: bool,
}

#[derive(Template)]
#[template(path = "terminal_input.txt", escape = "none")]
struct TerminalInputTemplate<'a> {
    task_id: &'a str,
}

#[derive(Template)]
#[template(path = "terminal_read.txt", escape = "none")]
struct TerminalReadTemplate<'a> {
    delta: &'a str,
    has_more: bool,
    cursor: usize,
    task_id: &'a str,
    status_text: &'a str,
    bytes_read: usize,
    total_bytes: usize,
}

fn render_terminal_delta(delta: &crate::job_registry::TerminalDelta) -> String {
    let delta_text = CommandStream::Inline(String::from_utf8_lossy(&delta.bytes).into_owned())
        .render_for_model();
    let status_text = job_status_text(&delta.status);
    TerminalReadTemplate {
        delta: &delta_text,
        has_more: delta.cursor < delta.total_bytes,
        cursor: delta.cursor,
        task_id: &delta.task_id,
        status_text: &status_text,
        bytes_read: delta.bytes.len(),
        total_bytes: delta.total_bytes,
    }
    .render()
    .expect("terminal read template must render")
}

impl Tool for ReadTerminalDeltaTool {
    fn name(&self) -> Cow<'static, str> {
        "terminal_read".into()
    }

    type Arguments = ReadTerminalDeltaArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = args.task_id.trim();
        if task_id.is_empty() {
            return Err(anyhow::anyhow!("task_id must not be empty"));
        }
        if args.max_bytes == 0 {
            return Err(anyhow::anyhow!("max_bytes must be greater than 0"));
        }

        let delta = self
            .registry
            .terminal_read(task_id, args.cursor, args.max_bytes)
            .await
            .map_err(anyhow::Error::msg)?;

        let text = render_terminal_delta(&delta);
        Ok(ToolResult::text(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_registry::TerminalDelta;

    fn delta(bytes: &str, cursor: usize, total_bytes: usize, status: JobStatus) -> TerminalDelta {
        TerminalDelta {
            task_id: "amber-forest-thunder-pearl".to_string(),
            bytes: bytes.as_bytes().to_vec(),
            cursor,
            total_bytes,
            status,
        }
    }

    #[test]
    fn terminal_read_without_more_buffered_states_the_next_cursor() {
        assert_eq!(
            render_terminal_delta(&delta("hello", 1005, 1005, JobStatus::Running)),
            "hello\nTask 'amber-forest-thunder-pearl' is running; read 5 new bytes (1005 buffered in total); next cursor is 1005.\n"
        );
    }

    #[test]
    fn terminal_read_with_more_buffered_says_how_to_continue() {
        assert_eq!(
            render_terminal_delta(&delta(
                "hello",
                5,
                1005,
                JobStatus::Completed { exit_code: 0 }
            )),
            "hello\nTask 'amber-forest-thunder-pearl' completed with exit code 0; read 5 new bytes (1005 buffered in total); next cursor is 5.\nMore output is buffered: call terminal_read with cursor=5.\n"
        );
    }
}
