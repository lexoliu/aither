//! ACP protocol types.
//!
//! Defines all message types for the Agent Client Protocol.

use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// ACP protocol version implemented by this crate.
pub const PROTOCOL_VERSION: u16 = 1;

// =============================================================================
// Initialization
// =============================================================================

/// Initialize request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol version the client supports.
    pub protocol_version: u16,
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
    pub protocol_version: u16,
    /// Agent capabilities.
    #[serde(default)]
    pub agent_capabilities: AgentCapabilities,
    /// Agent information.
    #[serde(default)]
    pub agent_info: Option<Implementation>,
    /// Supported authentication methods.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auth_methods: Vec<AuthMethod>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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

/// Session capabilities supported by the agent.
///
/// Each flag is an optional empty object: omitted or `null` means the method
/// family is not supported, `{}` means it is.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCapabilities {
    /// Whether the agent supports `session/list`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list: Option<SessionListCapabilities>,
    /// Whether the agent supports `session/delete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delete: Option<SessionDeleteCapabilities>,
    /// Whether the agent supports `additionalDirectories` on session
    /// lifecycle requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<SessionAdditionalDirectoriesCapabilities>,
    /// Whether the agent supports `session/resume`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<SessionResumeCapabilities>,
    /// Whether the agent supports `session/close`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub close: Option<SessionCloseCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/list` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionListCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/delete` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionDeleteCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for `additionalDirectories` support on session lifecycle
/// requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionAdditionalDirectoriesCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/resume` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionResumeCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/close` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionCloseCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

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

/// MCP server specification for session setup (stdio transport).
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// Working directory. Must be absolute.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,
    /// Additional workspace roots. Each path must be absolute.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Create new session response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewResult {
    /// Session ID.
    pub session_id: String,
    /// Initial mode state, if the agent supports session modes.
    #[serde(default)]
    pub modes: Option<SessionModeState>,
    /// Initial configuration options, if the agent supports them.
    #[serde(default)]
    pub config_options: Option<Vec<ConfigOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Load session request parameters (`session/load`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionLoadParams {
    /// Session ID to load.
    pub session_id: String,
    /// Working directory. Must match the session's `cwd`.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,
    /// Additional workspace roots. Each path must be absolute.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Load session response (`session/load`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionLoadResult {
    /// Initial mode state, if the agent supports session modes.
    #[serde(default)]
    pub modes: Option<SessionModeState>,
    /// Configuration options, if the agent supports them.
    #[serde(default)]
    pub config_options: Option<Vec<ConfigOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Set session mode request parameters (`session/set_mode`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSetModeParams {
    /// Session ID.
    pub session_id: String,
    /// Mode to activate.
    pub mode_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Set session mode response (`session/set_mode`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionSetModeResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Set config option request parameters (`session/set_config_option`).
///
/// The `value` field flattens into the params: a [`SessionConfigValue::Select`]
/// produces `{"value": "id"}` while [`SessionConfigValue::Boolean`] produces
/// `{"type": "boolean", "value": true}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSetConfigOptionParams {
    /// Session ID.
    pub session_id: String,
    /// Config option to set.
    pub config_id: String,
    /// New value for the option.
    #[serde(flatten)]
    pub value: SessionConfigValue,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Set config option response (`session/set_config_option`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSetConfigOptionResult {
    /// The full set of configuration options and their current values.
    #[serde(default)]
    pub config_options: Vec<ConfigOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// New value for a session config option in `session/set_config_option`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SessionConfigValue {
    /// `select` option: the chosen value ID.
    Select {
        /// Selected value ID.
        value: String,
    },
    /// `boolean` option: the new flag state.
    Boolean {
        /// Wire tag; always `"boolean"`.
        #[serde(rename = "type")]
        kind: BooleanConfigTag,
        /// Flag state.
        value: bool,
    },
}

/// Wire tag marking a `boolean` config option write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BooleanConfigTag {
    /// The literal `"boolean"`.
    #[serde(rename = "boolean")]
    Boolean,
}

impl From<String> for SessionConfigValue {
    fn from(value: String) -> Self {
        Self::Select { value }
    }
}

impl From<&str> for SessionConfigValue {
    fn from(value: &str) -> Self {
        Self::Select {
            value: value.to_string(),
        }
    }
}

impl From<bool> for SessionConfigValue {
    fn from(value: bool) -> Self {
        Self::Boolean {
            kind: BooleanConfigTag::Boolean,
            value,
        }
    }
}

/// The set of modes a session can be in and the one currently active.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModeState {
    /// The currently active mode.
    pub current_mode_id: String,
    /// Modes the agent can operate in.
    #[serde(default)]
    pub available_modes: Vec<SessionMode>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A mode the agent can operate in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMode {
    /// Stable identifier used to select this mode.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional human-readable details.
    #[serde(default)]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A session configuration option and its current state.
///
/// `select` options carry `current_value` (a value ID) and `options`;
/// `boolean` options carry a boolean `current_value` and no `options`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigOption {
    /// Unique identifier for the configuration option.
    pub id: String,
    /// Human-readable label for the option.
    pub name: String,
    /// Optional description for the client to display to the user.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional semantic category (e.g. `"mode"`, `"model"`,
    /// `"thought_level"`). Unknown categories are kept verbatim.
    #[serde(default)]
    pub category: Option<String>,
    /// Option type: `"select"`, `"boolean"`, or an agent-defined value.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    /// Currently selected value.
    #[serde(default)]
    pub current_value: Option<ConfigOptionValue>,
    /// Selectable values for `select` options.
    #[serde(default)]
    pub options: Option<ConfigSelectOptions>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Current value of a [`ConfigOption`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigOptionValue {
    /// Selected value ID for `select` options.
    Selected(String),
    /// Flag state for `boolean` options.
    Toggle(bool),
}

/// Possible values for a `select` config option: a flat list or a list of
/// groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigSelectOptions {
    /// Flat list of options with no grouping.
    Flat(Vec<ConfigSelectOption>),
    /// Options grouped under headers.
    Grouped(Vec<ConfigSelectGroup>),
}

/// A possible value for a `select` config option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSelectOption {
    /// Unique identifier for this option value.
    pub value: String,
    /// Human-readable label for this option value.
    pub name: String,
    /// Optional description for this option value.
    #[serde(default)]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A group of `select` config options under a header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSelectGroup {
    /// Group identifier.
    pub group: String,
    /// Human-readable group label.
    pub name: String,
    /// Options in this group.
    pub options: Vec<ConfigSelectOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Prompt response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResult {
    /// Reason for stopping.
    pub stop_reason: StopReason,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Stop reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Max turn requests reached.
    MaxTurnRequests,
    /// The agent refused to respond.
    Refusal,
}

/// Stop session request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStopParams {
    /// Session ID.
    pub session_id: String,
}

/// Cancel notification parameters (`session/cancel`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCancelParams {
    /// Session ID.
    pub session_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
#[serde(rename_all = "camelCase")]
pub struct ContentChunk {
    /// Content.
    pub content: ContentBlock,
    /// Optional identifier correlating chunks that belong to one message.
    #[serde(default)]
    pub message_id: Option<String>,
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A session update streamed from the agent in a `session/update`
/// notification.
///
/// Updates whose `sessionUpdate` tag or payload this crate does not model
/// deserialize into [`SessionUpdate::Other`], keeping the raw object, so an
/// unknown update can never fail parsing.
#[derive(Debug, Clone)]
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
    /// Available commands are ready or have changed.
    AvailableCommandsUpdate(AvailableCommandsUpdate),
    /// The session's current mode changed.
    CurrentModeUpdate(CurrentModeUpdate),
    /// Session configuration options were updated.
    ConfigOptionUpdate(ConfigOptionUpdate),
    /// Session metadata (title, timestamps) changed.
    SessionInfoUpdate(SessionInfoUpdate),
    /// Context window or cost usage changed.
    UsageUpdate(UsageUpdate),
    /// An update this crate does not model; the raw object is preserved.
    Other(Value),
}

/// Serialization view of [`SessionUpdate`] for known variants.
#[derive(Serialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
enum SessionUpdateRepr<'a> {
    AgentThoughtChunk(&'a ContentChunk),
    AgentMessageChunk(&'a ContentChunk),
    UserMessageChunk(&'a ContentChunk),
    Plan(&'a Plan),
    ToolCall(&'a ToolCall),
    ToolCallUpdate(&'a ToolCallUpdate),
    AvailableCommandsUpdate(&'a AvailableCommandsUpdate),
    CurrentModeUpdate(&'a CurrentModeUpdate),
    ConfigOptionUpdate(&'a ConfigOptionUpdate),
    SessionInfoUpdate(&'a SessionInfoUpdate),
    UsageUpdate(&'a UsageUpdate),
}

/// Deserialization view of [`SessionUpdate`] for known variants.
#[derive(Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
enum SessionUpdateOwned {
    AgentThoughtChunk(ContentChunk),
    AgentMessageChunk(ContentChunk),
    UserMessageChunk(ContentChunk),
    Plan(Plan),
    ToolCall(ToolCall),
    ToolCallUpdate(ToolCallUpdate),
    AvailableCommandsUpdate(AvailableCommandsUpdate),
    CurrentModeUpdate(CurrentModeUpdate),
    ConfigOptionUpdate(ConfigOptionUpdate),
    SessionInfoUpdate(SessionInfoUpdate),
    UsageUpdate(UsageUpdate),
}

impl Serialize for SessionUpdate {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let repr = match self {
            Self::AgentThoughtChunk(chunk) => SessionUpdateRepr::AgentThoughtChunk(chunk),
            Self::AgentMessageChunk(chunk) => SessionUpdateRepr::AgentMessageChunk(chunk),
            Self::UserMessageChunk(chunk) => SessionUpdateRepr::UserMessageChunk(chunk),
            Self::Plan(plan) => SessionUpdateRepr::Plan(plan),
            Self::ToolCall(call) => SessionUpdateRepr::ToolCall(call),
            Self::ToolCallUpdate(update) => SessionUpdateRepr::ToolCallUpdate(update),
            Self::AvailableCommandsUpdate(update) => {
                SessionUpdateRepr::AvailableCommandsUpdate(update)
            }
            Self::CurrentModeUpdate(update) => SessionUpdateRepr::CurrentModeUpdate(update),
            Self::ConfigOptionUpdate(update) => SessionUpdateRepr::ConfigOptionUpdate(update),
            Self::SessionInfoUpdate(update) => SessionUpdateRepr::SessionInfoUpdate(update),
            Self::UsageUpdate(update) => SessionUpdateRepr::UsageUpdate(update),
            Self::Other(raw) => return raw.serialize(serializer),
        };
        repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SessionUpdate {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Ok(SessionUpdateOwned::deserialize(&value).map_or_else(
            // Unknown `sessionUpdate` tag or a payload this crate does not
            // model: keep the raw object instead of failing.
            |_| Self::Other(value),
            Self::from,
        ))
    }
}

impl From<SessionUpdateOwned> for SessionUpdate {
    fn from(owned: SessionUpdateOwned) -> Self {
        match owned {
            SessionUpdateOwned::AgentThoughtChunk(chunk) => Self::AgentThoughtChunk(chunk),
            SessionUpdateOwned::AgentMessageChunk(chunk) => Self::AgentMessageChunk(chunk),
            SessionUpdateOwned::UserMessageChunk(chunk) => Self::UserMessageChunk(chunk),
            SessionUpdateOwned::Plan(plan) => Self::Plan(plan),
            SessionUpdateOwned::ToolCall(call) => Self::ToolCall(call),
            SessionUpdateOwned::ToolCallUpdate(update) => Self::ToolCallUpdate(update),
            SessionUpdateOwned::AvailableCommandsUpdate(update) => {
                Self::AvailableCommandsUpdate(update)
            }
            SessionUpdateOwned::CurrentModeUpdate(update) => Self::CurrentModeUpdate(update),
            SessionUpdateOwned::ConfigOptionUpdate(update) => Self::ConfigOptionUpdate(update),
            SessionUpdateOwned::SessionInfoUpdate(update) => Self::SessionInfoUpdate(update),
            SessionUpdateOwned::UsageUpdate(update) => Self::UsageUpdate(update),
        }
    }
}

/// `available_commands_update` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvailableCommandsUpdate {
    /// Commands the agent can execute.
    #[serde(default)]
    pub available_commands: Vec<AvailableCommand>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A command the agent can execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableCommand {
    /// Command name.
    pub name: String,
    /// Human-readable description of what the command does.
    pub description: String,
    /// Input specification, if the command takes input.
    #[serde(default)]
    pub input: Option<AvailableCommandInput>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Input specification for an [`AvailableCommand`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableCommandInput {
    /// Hint shown when no input has been provided.
    pub hint: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `current_mode_update` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentModeUpdate {
    /// The ID of the now-current mode.
    pub current_mode_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `config_option_update` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigOptionUpdate {
    /// The full set of configuration options and their current values.
    #[serde(default)]
    pub config_options: Vec<ConfigOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `session_info_update` payload. All fields are optional to support partial
/// updates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoUpdate {
    /// Human-readable title for the session. `null` clears it.
    #[serde(default)]
    pub title: Option<String>,
    /// ISO 8601 timestamp of last activity. `null` clears it.
    #[serde(default)]
    pub updated_at: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `usage_update` payload: context window and cost usage for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageUpdate {
    /// Tokens currently in context.
    pub used: u64,
    /// Total context window size in tokens.
    pub size: u64,
    /// Cumulative session cost.
    #[serde(default)]
    pub cost: Option<Cost>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Cost information for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cost {
    /// Total cumulative cost for the session.
    pub amount: f64,
    /// ISO 4217 currency code (e.g. `"USD"`).
    pub currency: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

// =============================================================================
// Plan
// =============================================================================

/// Execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Plan entries.
    pub entries: Vec<PlanEntry>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Plan entry status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Tool kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Read operation.
    Read,
    /// Write/edit operation.
    Edit,
    /// Delete operation.
    Delete,
    /// Move/rename operation.
    Move,
    /// Search operation.
    Search,
    /// Execute/run operation.
    Execute,
    /// Internal reasoning or planning.
    Think,
    /// Retrieving external data.
    Fetch,
    /// Switching the session mode.
    SwitchMode,
    /// Other operation.
    #[default]
    Other,
}

/// Tool call status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    Failed,
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
        #[serde(rename = "terminalId")]
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

// =============================================================================
// Permissions (agent → client requests)
// =============================================================================

/// `session/request_permission` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionParams {
    /// Session ID.
    pub session_id: String,
    /// The tool call that needs permission.
    pub tool_call: ToolCall,
    /// Options the user can choose from.
    #[serde(default)]
    pub options: Vec<PermissionOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A selectable option in a permission request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionOption {
    /// Unique option ID reported back in the outcome.
    pub option_id: String,
    /// Human-readable label.
    pub name: String,
    /// What the option does.
    pub kind: PermissionOptionKind,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// What a [`PermissionOption`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionOptionKind {
    /// Allow this call only.
    AllowOnce,
    /// Allow this call and future ones like it.
    AllowAlways,
    /// Reject this call only.
    RejectOnce,
    /// Reject this call and future ones like it.
    RejectAlways,
}

/// `session/request_permission` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestPermissionResult {
    /// The outcome of the permission request.
    pub outcome: RequestPermissionOutcome,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Outcome of a permission request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RequestPermissionOutcome {
    /// The prompt turn was cancelled before the user responded.
    Cancelled,
    /// The user selected one of the provided options.
    Selected {
        /// ID of the selected option.
        #[serde(rename = "optionId")]
        option_id: String,
    },
}

// =============================================================================
// File system (agent → client requests)
// =============================================================================

/// `fs/read_text_file` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadTextFileParams {
    /// Session ID.
    pub session_id: String,
    /// File path to read.
    pub path: PathBuf,
    /// First line to read (1-based); absent reads from the start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Maximum number of lines to read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `fs/read_text_file` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadTextFileResult {
    /// File content.
    pub content: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `fs/write_text_file` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteTextFileParams {
    /// Session ID.
    pub session_id: String,
    /// File path to write.
    pub path: PathBuf,
    /// Full content to write.
    pub content: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `fs/write_text_file` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WriteTextFileResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

// =============================================================================
// Terminals (agent → client requests)
// =============================================================================

/// `terminal/create` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalCreateParams {
    /// Session ID.
    pub session_id: String,
    /// Command to execute.
    pub command: String,
    /// Command arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables.
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// Working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Truncate retained output to this many bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_byte_limit: Option<u64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/create` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalCreateResult {
    /// Terminal ID used by the other `terminal/*` methods.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/output` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalOutputParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/output` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalOutputResult {
    /// Retained output.
    pub output: String,
    /// Whether the output was truncated at the byte limit.
    pub truncated: bool,
    /// Exit status once the process has exited.
    #[serde(default)]
    pub exit_status: Option<TerminalExitStatus>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/wait_for_exit` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalWaitForExitParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Exit status of a terminal process; also the `terminal/wait_for_exit`
/// response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalExitStatus {
    /// Exit code, if the process exited normally.
    #[serde(default)]
    pub exit_code: Option<i64>,
    /// Signal that terminated the process, if any.
    #[serde(default)]
    pub signal: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/kill` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalKillParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/kill` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TerminalKillResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/release` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalReleaseParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/release` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TerminalReleaseResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}
