//! Stop tool - terminate a background task by PID.
//!
//! Implements `Tool` trait and can be registered via `ToolRegistryBuilder::configure_tool`.
//! Allows terminating tasks across sandbox boundaries.
//!
//! Named `stop` to avoid collision with bash builtin `kill`.
//!
//! # Usage
//!
//! ```bash
//! stop 12345        # Stop process with PID 12345
//! ```

use std::borrow::Cow;

use aither_core::llm::{Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::job_registry::JobRegistry;

/// Stop a background task by PID.
///
/// This tool allows sandboxed processes to terminate background tasks.
/// The kill signal is sent from the parent process (outside the sandbox),
/// bypassing sandbox restrictions.
#[derive(Debug, Clone)]
pub struct StopTool {
    registry: JobRegistry,
}

impl StopTool {
    /// Creates a new stop tool with the given registry.
    pub fn new(registry: JobRegistry) -> Self {
        Self { registry }
    }
}

/// Arguments for the stop command.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StopArgs {
    /// Process ID to stop.
    pub pid: u32,
}

impl Tool for StopTool {
    fn name(&self) -> Cow<'static, str> {
        "stop".into()
    }

    type Arguments = StopArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        if self.registry.kill(args.pid).await {
            tracing::info!(pid = %args.pid, "stopped background task");
            Ok(ToolOutput::text(format!("Stopped process {}", args.pid)))
        } else {
            tracing::warn!(pid = %args.pid, "failed to stop process - not found or already dead");
            Ok(ToolOutput::text(format!(
                "Failed to stop process {}: not found or already dead",
                args.pid
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_args_schema() {
        let schema = schemars::schema_for!(StopArgs);
        let json = serde_json::to_string_pretty(&schema).unwrap();
        // Should have pid as required
        assert!(json.contains("pid"));
        assert!(json.contains("required"));
    }
}
