//! Managed browser tool protocol and tool implementation.

use std::borrow::Cow;

use aither_agent::tool_request::{
    ToolRequestBroker, ToolRequestQueue, channel as tool_request_channel,
};
use aither_core::llm::ToolContext;
use aither_core::llm::{IntoToolResult, Tool, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Runtime status for the managed browser service.
pub struct BrowserServiceStatus {
    /// Whether the service is currently running.
    pub running: bool,
    /// CDP endpoint when available.
    pub cdp_endpoint: Option<String>,
}

/// Amount of page detail returned by a browser read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum BrowserReadDetail {
    /// Short summary suitable for quick inspection.
    Brief,
    /// Balanced summary with common page metadata.
    #[default]
    Standard,
    /// Full structured capture.
    Full,
}

/// Screenshot mode for a browser read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum BrowserScreenshotMode {
    /// Do not capture a screenshot.
    Off,
    /// Capture the current viewport.
    #[default]
    Viewport,
    /// Capture the full page.
    FullPage,
}

/// Output format for browser reads.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserReadFormat {
    /// JSON-structured output.
    #[default]
    Json,
}

/// Options controlling browser read capture depth.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BrowserReadOptions {
    /// Detail level for textual and accessibility output.
    #[serde(default)]
    pub detail: BrowserReadDetail,
    /// Screenshot capture mode.
    #[serde(default)]
    pub screenshot: BrowserScreenshotMode,
}

/// Interactive element discovered on a page.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserInteractiveElement {
    /// Accessibility role.
    pub role: String,
    /// DOM tag name.
    pub tag_name: String,
    /// Visible text content.
    pub text: Option<String>,
    /// Accessible name.
    pub accessible_name: Option<String>,
    /// Link target for anchor-like elements.
    pub href: Option<String>,
    /// Input type for form controls.
    pub input_type: Option<String>,
    /// Whether the element is disabled.
    pub disabled: bool,
}

/// Accessibility property on a browser accessibility node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAccessibilityProperty {
    /// Property name.
    pub name: String,
    /// Property value.
    pub value: String,
}

/// Accessibility node captured from the page tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAccessibilityNode {
    /// Accessibility role.
    pub role: String,
    /// Accessible name.
    pub name: Option<String>,
    /// Accessible description.
    pub description: Option<String>,
    /// Node value.
    pub value: Option<String>,
    /// Tree depth.
    pub depth: usize,
    /// Number of children.
    pub child_count: usize,
    /// Whether the browser marked the node ignored.
    pub ignored: bool,
    /// Additional accessibility properties.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<BrowserAccessibilityProperty>,
}

/// Summary statistics and nodes from an accessibility snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAccessibilitySnapshot {
    /// Total node count.
    pub total_nodes: usize,
    /// Count of interesting nodes.
    pub interesting_nodes: usize,
    /// Count of interactive nodes.
    pub interactive_nodes: usize,
    /// Count of heading nodes.
    pub heading_nodes: usize,
    /// Count of landmark nodes.
    pub landmark_nodes: usize,
    /// Captured nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<BrowserAccessibilityNode>,
}

/// Inline screenshot captured from the managed browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserCapturedScreenshot {
    /// MIME type.
    pub mime_type: String,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Base64-encoded image bytes.
    pub data_base64: String,
}

/// Screenshot stored as an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserArtifactScreenshot {
    /// MIME type.
    pub mime_type: String,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Filesystem artifact path.
    pub artifact_path: String,
    /// API path for retrieving the artifact.
    pub api_path: String,
}

/// Structured summary of the current browser page.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPageSummary {
    /// Current page URL.
    pub url: String,
    /// Page title.
    pub title: String,
    /// Main heading, when detected.
    pub main_heading: Option<String>,
    /// Heading texts.
    pub headings: Vec<String>,
    /// Text excerpt.
    pub text_excerpt: String,
    /// Interactive elements.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interactive_elements: Vec<BrowserInteractiveElement>,
    /// Accessibility summary.
    pub accessibility: BrowserAccessibilitySnapshot,
}

/// Page capture returned by the browser service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPageCapture {
    /// Page summary.
    pub page: BrowserPageSummary,
    /// Optional inline screenshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot: Option<BrowserCapturedScreenshot>,
}

/// Page snapshot returned to the tool caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPageSnapshot {
    /// Flattened page summary.
    #[serde(flatten)]
    pub page: BrowserPageSummary,
    /// Optional screenshot artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot: Option<BrowserArtifactScreenshot>,
}

/// Request sent to the managed browser service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum BrowserServiceRequest {
    /// Query service status.
    Status,
    /// Ensure the service is running.
    EnsureStarted,
    /// Stop the service.
    Stop,
    /// Read and capture a page URL.
    Read {
        /// URL to open.
        url: String,
        /// Read options.
        #[serde(default)]
        options: BrowserReadOptions,
    },
}

/// Response returned by the managed browser service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BrowserServiceResponse {
    /// Command succeeded.
    Ok {
        /// Status payload.
        payload: BrowserServiceStatus,
    },
    /// Page read succeeded.
    ReadOk {
        /// Page capture payload.
        payload: Box<BrowserPageCapture>,
    },
    /// Service stopped.
    Stopped,
    /// Command failed.
    Error {
        /// Error message.
        message: String,
    },
}

/// Arguments accepted by the browser tool.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum BrowserArgs {
    /// Show managed browser runtime status.
    Status,
    /// Start the managed browser runtime.
    #[serde(rename = "open")]
    Start,
    /// Stop the managed browser runtime.
    #[serde(rename = "close")]
    Stop,
    /// Return the managed browser CDP endpoint. Requires `open` first.
    CdpEndpoint,
    /// Open a URL in the managed browser and inspect it with structured text,
    /// accessibility output, and an optional screenshot artifact.
    Read {
        /// URL to open and inspect.
        url: String,
        /// Read detail level.
        #[serde(default)]
        detail: BrowserReadDetail,
        /// Screenshot mode.
        #[serde(default)]
        screenshot: BrowserScreenshotMode,
        /// Output format.
        #[serde(default)]
        format: BrowserReadFormat,
    },
}

impl BrowserArgs {
    /// Builds read options from tool arguments.
    #[must_use]
    pub const fn read_options(
        detail: BrowserReadDetail,
        screenshot: BrowserScreenshotMode,
    ) -> BrowserReadOptions {
        BrowserReadOptions { detail, screenshot }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
/// Request sent from the tool to the browser host.
pub struct BrowserRequest {
    /// Logical session ID.
    pub session_id: String,
    /// Browser command arguments.
    pub args: BrowserArgs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Result returned by the browser tool.
pub struct BrowserResult {
    /// Whether the browser service is running.
    pub running: bool,
    /// CDP endpoint when available.
    pub cdp_endpoint: Option<String>,
    /// Optional page snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<BrowserPageSnapshot>,
}

/// Error returned by the browser tool.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum BrowserToolError {
    /// Browser commands are not available in this runtime.
    #[error("browser command is unavailable in this runtime")]
    UnavailableInRuntime,
    /// Browser request failed.
    #[error("{message}")]
    RequestFailed {
        /// Failure message.
        message: String,
    },
}

impl BrowserToolError {
    /// Creates a request-failed error.
    #[must_use]
    pub fn request_failed(message: impl Into<String>) -> Self {
        Self::RequestFailed {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
/// Tool output wrapper that converts browser errors into tool results.
pub struct BrowserToolOutput(Result<BrowserResult, BrowserToolError>);

impl From<Result<BrowserResult, BrowserToolError>> for BrowserToolOutput {
    fn from(value: Result<BrowserResult, BrowserToolError>) -> Self {
        Self(value)
    }
}

impl IntoToolResult for BrowserToolOutput {
    fn into_tool_result(self) -> aither_core::Result<ToolResult> {
        match self.0 {
            Ok(result) => ToolResult::json(&result),
            Err(error) => Ok(ToolResult::error(error.to_string())),
        }
    }
}

/// Broker used by the browser tool to request host-side browser work.
pub type BrowserBroker = ToolRequestBroker<BrowserRequest, Result<BrowserResult, BrowserToolError>>;
/// Queue consumed by the host-side browser service.
pub type BrowserQueue = ToolRequestQueue<BrowserRequest, Result<BrowserResult, BrowserToolError>>;

/// Creates a browser request channel pair.
#[must_use]
pub fn channel() -> (BrowserBroker, BrowserQueue) {
    tool_request_channel()
}

#[derive(Debug, Clone)]
/// Tool implementation for managed browser operations.
pub struct BrowserTool {
    session_id: String,
    broker: Option<BrowserBroker>,
}

impl BrowserTool {
    /// Creates a browser tool bound to a session and optional broker.
    #[must_use]
    pub const fn new(session_id: String, broker: Option<BrowserBroker>) -> Self {
        Self { session_id, broker }
    }
}

impl Tool for BrowserTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("browser")
    }

    type Arguments = BrowserArgs;
    type Res = BrowserToolOutput;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let Some(broker) = self.broker.as_ref() else {
            return Ok(BrowserToolOutput::from(Err(
                BrowserToolError::UnavailableInRuntime,
            )));
        };
        let response = broker
            .request(BrowserRequest {
                session_id: self.session_id.clone(),
                args,
            })
            .await?;
        Ok(BrowserToolOutput::from(response))
    }
}
