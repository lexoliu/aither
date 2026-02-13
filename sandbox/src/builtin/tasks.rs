//! Background tasks tool - list background tasks.
//!
//! Implements `Tool` trait and can be registered via `ToolRegistryBuilder::configure_tool`.
//!
//! Exposed as `jobs` to align with standard shell intuition.
//!
//! # Usage
//!
//! ```bash
//! tasks             # List all background tasks
//! ```

use std::borrow::Cow;

use aither_core::llm::{Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::job_registry::{JobRegistry, JobStatus};

/// List background tasks.
///
/// Shows all tasks including running, completed, failed, and killed.
#[derive(Debug, Clone)]
pub struct TasksTool {
    registry: JobRegistry,
}

impl TasksTool {
    /// Creates a new tasks tool with the given registry.
    #[must_use]
    pub const fn new(registry: JobRegistry) -> Self {
        Self { registry }
    }
}

/// Arguments for the tasks command.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TasksArgs {
    /// Optional filter (reserved for future use)
    #[serde(default)]
    pub filter: Option<String>,
}

impl Tool for TasksTool {
    fn name(&self) -> Cow<'static, str> {
        "jobs".into()
    }

    type Arguments = TasksArgs;

    async fn call(&self, _args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let jobs = self.registry.list().await;

        if jobs.is_empty() {
            return Ok(ToolOutput::text("No background tasks."));
        }

        let mut output = String::new();
        for job in &jobs {
            let status_str = match &job.status {
                JobStatus::Running => "running".to_string(),
                JobStatus::Completed { exit_code } => format!("exit {exit_code}"),
                JobStatus::Failed { error } => format!("failed: {error}"),
                JobStatus::Killed => "killed".to_string(),
            };

            let script_preview = if job.script.len() > 60 {
                format!("{}...", &job.script[..60])
            } else {
                job.script.clone()
            };

            let output_str = job
                .output_path
                .as_ref()
                .map_or_else(|| "(no output)".to_string(), |p| p.display().to_string());

            output.push_str(&format!(
                "PID {} [{}]\n  shell_id: {}\n  script: {}\n  output: {}\n\n",
                job.pid, status_str, job.shell_id, script_preview, output_str
            ));
        }

        Ok(ToolOutput::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_registry::job_registry_channel;
    use crate::permission::BashMode;
    use aither_core::llm::Tool;
    use executor_core::Executor;
    use executor_core::Task;
    use executor_core::tokio::TokioGlobal;
    use std::path::PathBuf;

    #[test]
    fn test_tasks_args_schema() {
        let schema = schemars::schema_for!(TasksArgs);
        let json = serde_json::to_string_pretty(&schema).unwrap();
        // Should have filter as optional
        assert!(json.contains("filter"));
    }

    #[tokio::test]
    async fn test_tasks_tool_lists_jobs() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();

        // Register a running job
        registry
            .register(
                12345,
                "shell-test",
                "sleep 100",
                BashMode::Sandboxed,
                Some(PathBuf::from("/tmp/12345.txt")),
            )
            .await;

        // Create tool with registry
        let tool = TasksTool::new(registry.clone());

        // Call tool
        let output = tool.call(TasksArgs::default()).await.unwrap();
        let text = output.as_str().unwrap();

        // Verify output contains job info
        assert!(text.contains("PID 12345"));
        assert!(text.contains("running"));
        assert!(text.contains("sleep 100"));
    }

    #[tokio::test]
    async fn test_tasks_tool_empty_registry() {
        let (registry, service) = job_registry_channel();
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();
        let tool = TasksTool::new(registry);

        let output = tool.call(TasksArgs::default()).await.unwrap();
        let text = output.as_str().unwrap();

        assert!(text.contains("No background tasks"));
    }
}
