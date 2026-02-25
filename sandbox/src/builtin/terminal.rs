//! Native terminal control tools for background bash tasks.

use std::borrow::Cow;

use aither_core::llm::{Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::job_registry::JobRegistry;

#[derive(Debug, Clone)]
pub struct KillTerminalTool {
    registry: JobRegistry,
}

impl KillTerminalTool {
    #[must_use]
    pub const fn new(registry: JobRegistry) -> Self {
        Self { registry }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KillTerminalArgs {
    /// Task identifier returned by bash when the task is backgrounded.
    pub task_id: String,
}

impl Tool for KillTerminalTool {
    fn name(&self) -> Cow<'static, str> {
        "kill_terminal".into()
    }

    type Arguments = KillTerminalArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let task_id = args.task_id.trim();
        if task_id.is_empty() {
            return Err(anyhow::anyhow!("task_id must not be empty"));
        }

        let killed = self.registry.kill_by_task_id(task_id).await;
        ToolOutput::json(&serde_json::json!({
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

#[derive(Debug, Clone)]
pub struct InputTerminalTool {
    registry: JobRegistry,
}

impl InputTerminalTool {
    #[must_use]
    pub const fn new(registry: JobRegistry) -> Self {
        Self { registry }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InputTerminalArgs {
    /// Task identifier returned by bash when the task is backgrounded.
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
        "input_terminal".into()
    }

    type Arguments = InputTerminalArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let task_id = args.task_id.trim();
        if task_id.is_empty() {
            return Err(anyhow::anyhow!("task_id must not be empty"));
        }

        let mut bytes = args.input.into_bytes();
        if args.append_newline {
            bytes.push(b'\n');
        }

        self.registry
            .input_terminal(task_id, bytes)
            .await
            .map_err(anyhow::Error::msg)?;

        ToolOutput::json(&serde_json::json!({
            "ok": true,
            "task_id": task_id,
        }))
    }
}
