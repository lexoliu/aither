//! Native terminal control tools for background terminal tasks.

use std::borrow::Cow;

use aither_core::llm::ToolContext;
use aither_core::llm::{Tool, ToolResult};
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
        ToolResult::json(&serde_json::json!({
            "ok": killed,
            "task_id": task_id,
            "killed": killed,
            "message": if killed {
                "Background task terminated"
            } else {
                "Background task not found or already stopped"
            }
        }))
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

        ToolResult::json(&serde_json::json!({
            "ok": true,
            "task_id": task_id,
        }))
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

fn job_status_payload(status: &JobStatus) -> serde_json::Value {
    match status {
        JobStatus::Running => serde_json::json!({ "kind": "running" }),
        JobStatus::Completed { exit_code } => {
            serde_json::json!({ "kind": "completed", "exit_code": exit_code })
        }
        JobStatus::Failed { error } => {
            serde_json::json!({ "kind": "failed", "error": error })
        }
        JobStatus::Killed => serde_json::json!({ "kind": "killed" }),
    }
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

        ToolResult::json(&serde_json::json!({
            "ok": true,
            "task_id": delta.task_id,
            "cursor": delta.cursor,
            "total_bytes": delta.total_bytes,
            "bytes_read": delta.bytes.len(),
            "has_more": delta.cursor < delta.total_bytes,
            "delta": String::from_utf8_lossy(&delta.bytes),
            "status": job_status_payload(&delta.status),
        }))
    }
}
