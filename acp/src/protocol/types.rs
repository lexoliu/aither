//! ACP protocol types.
//!
//! Defines all message types for the Agent Client Protocol.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// ACP protocol version.
pub const PROTOCOL_VERSION: &str = "1.0";

// =============================================================================
// Initialization
// =============================================================================

/// Initialize request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol version the client supports.
    pub protocol_version: String,
    /// Client capabilities.
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
    /// Client information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_info: Option<Implementation>,
}

/// Initialize response result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Protocol version the agent supports.
    pub protocol_version: String,
    /// Agent capabilities.
    pub agent_capabilities: AgentCapabilities,
    /// Agent information.
    pub agent_info: Implementation,
    /// Supported authentication methods.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auth_methods: Vec<AuthMethod>,
}

/// Implementation info (client or agent).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Implementation {
    /// Name.
    pub name: String,
    /// Display title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Version.
    pub version: String,
}

/// Client capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// File system capabilities.
    #[serde(default)]
    pub fs: FileSystemCapability,
    /// Whether terminal is supported.
    #[serde(default)]
    pub terminal: bool,
}

/// File system capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileSystemCapability {
    /// Can read text files.
    #[serde(default)]
    pub read_text_file: bool,
    /// Can write text files.
    #[serde(default)]
    pub write_text_file: bool,
}

/// Agent capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    /// Whether session loading is supported.
    #[serde(default)]
    pub load_session: bool,
    /// Prompt capabilities.
    #[serde(default)]
    pub prompt_capabilities: PromptCapabilities,
    /// MCP capabilities.
    #[serde(default)]
    pub mcp_capabilities: McpCapabilities,
    /// Session capabilities.
    #[serde(default)]
    pub session_capabilities: SessionCapabilities,
}

/// Prompt capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCapabilities {
    /// Image support.
    #[serde(default)]
    pub image: bool,
    /// Audio support.
    #[serde(default)]
    pub audio: bool,
    /// Embedded context support.
    #[serde(default)]
    pub embedded_context: bool,
}

/// MCP capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpCapabilities {
    /// HTTP MCP support.
    #[serde(default)]
    pub http: bool,
    /// SSE MCP support.
    #[serde(default)]
    pub sse: bool,
}

/// Session capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionCapabilities {}

/// Authentication method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthMethod {
    /// Unique ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

// =============================================================================
// Session Management
// =============================================================================

/// MCP server specification for session setup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerSpec {
    /// Server name.
    pub name: String,
    /// Command to run.
    pub command: String,
    /// Command arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables.
    #[serde(default)]
    pub env: Vec<EnvVar>,
}

/// Environment variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    /// Variable name.
    pub name: String,
    /// Variable value.
    pub value: String,
}

/// Create new session request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewParams {
    /// Working directory.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,
}

/// Create new session response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewResult {
    /// Session ID.
    pub session_id: String,
}

// =============================================================================
// Prompt
// =============================================================================

/// Prompt request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    /// Session ID.
    pub session_id: String,
    /// Prompt content.
    pub prompt: Vec<ContentBlock>,
}

/// Prompt response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResult {
    /// Reason for stopping.
    pub stop_reason: StopReason,
}

/// Stop reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Normal end of turn.
    EndTurn,
    /// Cancelled by user.
    Cancelled,
    /// Error occurred.
    Error,
    /// Max tokens reached.
    MaxTokens,
}

/// Stop session request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStopParams {
    /// Session ID.
    pub session_id: String,
}

// =============================================================================
// Content Types
// =============================================================================

/// Content block (text, image, resource).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentBlock {
    /// Text content.
    Text(TextContent),
    /// Image content.
    Image(ImageContent),
    /// Resource content.
    Resource(ResourceContent),
}

/// Text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    /// Text value.
    pub text: String,
    /// Annotations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
}

/// Image content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded data.
    pub data: String,
    /// MIME type.
    pub mime_type: String,
}

/// Resource content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    /// Resource data.
    pub resource: EmbeddedResource,
}

/// Embedded resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedResource {
    /// Resource URI.
    pub uri: String,
    /// MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Content chunk for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentChunk {
    /// Content.
    pub content: ContentBlock,
}

// =============================================================================
// Session Updates (Notifications)
// =============================================================================

/// Session update notification parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNotification {
    /// Session ID.
    pub session_id: String,
    /// Update data.
    pub update: SessionUpdate,
}

/// Session update types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum SessionUpdate {
    /// Agent thinking chunk.
    AgentThoughtChunk(ContentChunk),
    /// Agent message chunk.
    AgentMessageChunk(ContentChunk),
    /// User message chunk (for session replay).
    UserMessageChunk(ContentChunk),
    /// Plan update.
    Plan(Plan),
    /// Tool call initiated.
    ToolCall(ToolCall),
    /// Tool call progress/completion.
    ToolCallUpdate(ToolCallUpdate),
}

// =============================================================================
// Plan
// =============================================================================

/// Execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Plan entries.
    pub entries: Vec<PlanEntry>,
}

/// Plan entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanEntry {
    /// Entry content.
    pub content: String,
    /// Entry status.
    pub status: PlanEntryStatus,
    /// Priority.
    #[serde(default)]
    pub priority: PlanEntryPriority,
}

/// Plan entry status.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryStatus {
    /// Not started.
    #[default]
    Pending,
    /// In progress.
    InProgress,
    /// Completed.
    Completed,
}

/// Plan entry priority.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanEntryPriority {
    /// Low priority.
    Low,
    /// Medium priority.
    #[default]
    Medium,
    /// High priority.
    High,
}

// =============================================================================
// Tool Calls
// =============================================================================

/// Tool call initiated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Unique tool call ID.
    pub tool_call_id: String,
    /// Human-readable title.
    pub title: String,
    /// Tool kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    /// Current status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
    /// Content produced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ToolCallContent>,
    /// Affected locations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<ToolCallLocation>,
    /// Raw input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
    /// Raw output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Value>,
}

/// Tool call update.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallUpdate {
    /// Tool call ID.
    pub tool_call_id: String,
    /// Updated status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
    /// New content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ToolCallContent>>,
    /// Updated title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Updated kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    /// Updated locations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<ToolCallLocation>>,
    /// Raw input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
    /// Raw output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Value>,
}

/// Tool kind.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolKind {
    /// Read operation.
    Read,
    /// Write/edit operation.
    Edit,
    /// Search operation.
    Search,
    /// Execute/run operation.
    Execute,
    /// Other operation.
    #[default]
    Other,
}

/// Tool call status.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    /// Pending execution.
    #[default]
    Pending,
    /// In progress.
    InProgress,
    /// Completed successfully.
    Completed,
    /// Failed with error.
    Error,
}

/// Tool call content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolCallContent {
    /// Regular content.
    Content {
        /// The content block.
        content: ContentBlock,
    },
    /// File diff.
    Diff(Diff),
    /// Terminal reference.
    Terminal {
        /// Terminal ID from terminal/create.
        terminal_id: String,
    },
}

/// File diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diff {
    /// File path.
    pub path: PathBuf,
    /// Original text (None for new files).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_text: Option<String>,
    /// New text.
    pub new_text: String,
}

/// Tool call location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallLocation {
    /// File path.
    pub path: PathBuf,
    /// Line number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}
