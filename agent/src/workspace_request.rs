//! Workspace request tool.

use std::borrow::Cow;

use aither_core::llm::ToolContext;
use aither_core::llm::tool::{Tool, ToolResult};
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

/// Internal request payload sent from a session-bound tool to the UI/runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceRequestPayload {
    /// Session issuing the request.
    pub session_id: String,
    /// User-visible request body.
    pub request: RequestWorkspaceArgs,
}

/// A workspace access request sent from the tool to the UI.
pub type WorkspaceRequest = ToolRequest<WorkspaceRequestPayload, bool>;

/// Broker for workspace requests (held by the tool).
pub type WorkspaceRequestBroker = ToolRequestBroker<WorkspaceRequestPayload, bool>;

/// Queue for workspace requests (held by the UI).
pub type WorkspaceRequestQueue = ToolRequestQueue<WorkspaceRequestPayload, bool>;

/// Create a new workspace request channel pair.
#[must_use]
pub fn channel() -> (WorkspaceRequestBroker, WorkspaceRequestQueue) {
    tool_request_channel()
}

/// Tool for requesting workspace access.
#[derive(Debug, Clone)]
pub struct RequestWorkspaceTool {
    broker: WorkspaceRequestBroker,
    session_id: String,
}

impl RequestWorkspaceTool {
    /// Create a new workspace request tool.
    #[must_use]
    pub fn new(session_id: impl Into<String>, broker: WorkspaceRequestBroker) -> Self {
        Self {
            broker,
            session_id: session_id.into(),
        }
    }
}

impl Tool for RequestWorkspaceTool {
    fn name(&self) -> Cow<'static, str> {
        "request_workspace".into()
    }

    type Arguments = RequestWorkspaceArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let approved = self
            .broker
            .request(WorkspaceRequestPayload {
                session_id: self.session_id.clone(),
                request: args.clone(),
            })
            .await?;
        let access = WorkspaceAccess {
            path: args.path,
            reason: args.reason,
            approved,
        };
        ToolResult::json(&access)
    }
}
