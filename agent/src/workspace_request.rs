//! Workspace request tool.

use std::borrow::Cow;

use aither_core::llm::tool::{Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tool_request::{
    ToolRequest, ToolRequestBroker, ToolRequestQueue, channel as tool_request_channel,
};

const DEFAULT_REASON: &str = include_str!("texts/request_workspace_default_reason.txt");

/// Request access to a directory outside the sandbox.
///
/// This command asks for permission to read/write files in the specified
/// directory. The user will be prompted to approve or deny the request.
/// Use this when you need to modify files in the user's project directory
/// or other locations outside your sandbox.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RequestWorkspaceArgs {
    /// The directory path to request access to (absolute or relative to cwd).
    pub path: String,

    /// Why access is needed (shown to user during approval prompt).
    #[serde(default = "default_reason")]
    pub reason: String,
}

fn default_reason() -> String {
    DEFAULT_REASON.trim().to_string()
}

/// Result of a workspace access request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorkspaceAccess {
    /// The requested directory path.
    pub path: String,
    /// Reason for the request.
    pub reason: String,
    /// Whether access was approved.
    pub approved: bool,
}

/// A workspace access request sent from the tool to the UI.
pub type WorkspaceRequest = ToolRequest<RequestWorkspaceArgs, bool>;

/// Broker for workspace requests (held by the tool).
pub type WorkspaceRequestBroker = ToolRequestBroker<RequestWorkspaceArgs, bool>;

/// Queue for workspace requests (held by the UI).
pub type WorkspaceRequestQueue = ToolRequestQueue<RequestWorkspaceArgs, bool>;

/// Create a new workspace request channel pair.
#[must_use]
pub fn channel() -> (WorkspaceRequestBroker, WorkspaceRequestQueue) {
    tool_request_channel()
}

/// Tool for requesting workspace access.
#[derive(Debug, Clone)]
pub struct RequestWorkspaceTool {
    broker: WorkspaceRequestBroker,
}

impl RequestWorkspaceTool {
    /// Create a new workspace request tool.
    pub fn new(broker: WorkspaceRequestBroker) -> Self {
        Self { broker }
    }
}

impl Tool for RequestWorkspaceTool {
    fn name(&self) -> Cow<'static, str> {
        "request_workspace".into()
    }

    type Arguments = RequestWorkspaceArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let approved = self.broker.request(args.clone()).await?;
        let access = WorkspaceAccess {
            path: args.path,
            reason: args.reason,
            approved,
        };
        Ok(ToolOutput::json(&access)?)
    }
}
