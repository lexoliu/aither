//! ACP protocol types.
//!
//! Defines all message types for the Agent Client Protocol.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::{AgentAuthCapabilities, AuthMethod, ClientAuthCapabilities, ElicitationCapabilities};

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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// What the client offers for `authenticate`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<ClientAuthCapabilities>,
    /// What the client offers for `elicitation/create`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elicitation: Option<ElicitationCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Capability keys this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
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
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// What the agent offers for `authenticate`/`logout`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AgentAuthCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Capability keys this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
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
    /// Session capability keys this crate does not model (e.g. `fork`,
    /// `subagents`), preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
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

// =============================================================================
// Session Management
// =============================================================================

/// An MCP server offered to the agent in `session/new`, `session/load`, or
/// `session/resume`.
///
/// Local stdio servers carry no `type` field; remote servers are tagged
/// `"type": "http"` or `"type": "sse"`. Whether an agent accepts remote
/// servers is negotiated through [`McpCapabilities`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpServer {
    /// A remote server reached over streamable HTTP.
    #[serde(rename = "http")]
    Http(McpServerHttp),
    /// A remote server reached over SSE.
    #[serde(rename = "sse")]
    Sse(McpServerSse),
    /// A local process the agent spawns; sent with no `type` field.
    #[serde(untagged)]
    Stdio(McpServerStdio),
}

impl McpServer {
    /// The server name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Http(server) => &server.name,
            Self::Sse(server) => &server.name,
            Self::Stdio(server) => &server.name,
        }
    }
}

impl From<McpServerStdio> for McpServer {
    fn from(server: McpServerStdio) -> Self {
        Self::Stdio(server)
    }
}

impl From<McpServerHttp> for McpServer {
    fn from(server: McpServerHttp) -> Self {
        Self::Http(server)
    }
}

impl From<McpServerSse> for McpServer {
    fn from(server: McpServerSse) -> Self {
        Self::Sse(server)
    }
}

/// A local MCP server the agent spawns as a child process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStdio {
    /// Server name.
    pub name: String,
    /// Command to run.
    pub command: String,
    /// Command arguments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Environment variables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<EnvVar>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl McpServerStdio {
    /// Build a stdio server spec.
    #[must_use]
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
            meta: None,
        }
    }

    /// Set command arguments.
    #[must_use]
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set environment variables.
    #[must_use]
    pub fn env(mut self, env: Vec<EnvVar>) -> Self {
        self.env = env;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// A remote MCP server reached over streamable HTTP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerHttp {
    /// Server name.
    pub name: String,
    /// Endpoint URL.
    pub url: String,
    /// HTTP headers to send with requests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<HttpHeader>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl McpServerHttp {
    /// Build an HTTP server spec.
    #[must_use]
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            headers: Vec::new(),
            meta: None,
        }
    }

    /// Set request headers.
    #[must_use]
    pub fn headers(mut self, headers: Vec<HttpHeader>) -> Self {
        self.headers = headers;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// A remote MCP server reached over SSE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerSse {
    /// Server name.
    pub name: String,
    /// Endpoint URL.
    pub url: String,
    /// HTTP headers to send with requests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<HttpHeader>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl McpServerSse {
    /// Build an SSE server spec.
    #[must_use]
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            headers: Vec::new(),
            meta: None,
        }
    }

    /// Set request headers.
    #[must_use]
    pub fn headers(mut self, headers: Vec<HttpHeader>) -> Self {
        self.headers = headers;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Environment variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    /// Variable name.
    pub name: String,
    /// Variable value.
    pub value: String,
}

/// HTTP header on a remote MCP server request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpHeader {
    /// Header name.
    pub name: String,
    /// Header value.
    pub value: String,
}

impl HttpHeader {
    /// Build a header.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// Create new session request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewParams {
    /// Working directory. Must be absolute.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    /// Additional workspace roots. Each path must be absolute and is only
    /// honoured when the agent advertises
    /// [`SessionCapabilities::additional_directories`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionNewParams {
    /// Build params for a session rooted at `cwd`.
    #[must_use]
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            mcp_servers: Vec::new(),
            additional_directories: Vec::new(),
            meta: None,
        }
    }

    /// Set the MCP servers to connect.
    #[must_use]
    pub fn mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Set additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, additional_directories: Vec<PathBuf>) -> Self {
        self.additional_directories = additional_directories;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
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
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
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
    pub mcp_servers: Vec<McpServer>,
    /// Additional workspace roots. Each path must be absolute and must be
    /// re-sent here — the agent does not restore them implicitly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionLoadParams {
    /// Build params loading `session_id` rooted at `cwd`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            session_id: session_id.into(),
            cwd: cwd.into(),
            mcp_servers: Vec::new(),
            additional_directories: Vec::new(),
            meta: None,
        }
    }

    /// Set the MCP servers to connect.
    #[must_use]
    pub fn mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Set additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, additional_directories: Vec<PathBuf>) -> Self {
        self.additional_directories = additional_directories;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
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
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Resume session request parameters (`session/resume`).
///
/// Unlike `session/load`, resuming restores the session's context inside the
/// agent without replaying its history back as `session/update`
/// notifications — for clients that do not need the transcript re-sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResumeParams {
    /// Session ID to resume.
    pub session_id: String,
    /// Working directory. Must match the session's `cwd`.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    /// Additional workspace roots. Each path must be absolute and must be
    /// re-sent here — the agent does not restore them implicitly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionResumeParams {
    /// Build params resuming `session_id` rooted at `cwd`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            session_id: session_id.into(),
            cwd: cwd.into(),
            mcp_servers: Vec::new(),
            additional_directories: Vec::new(),
            meta: None,
        }
    }

    /// Set the MCP servers to connect.
    #[must_use]
    pub fn mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Set additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, additional_directories: Vec<PathBuf>) -> Self {
        self.additional_directories = additional_directories;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Resume session response (`session/resume`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResumeResult {
    /// Initial mode state, if the agent supports session modes.
    #[serde(default)]
    pub modes: Option<SessionModeState>,
    /// Configuration options, if the agent supports them.
    #[serde(default)]
    pub config_options: Option<Vec<ConfigOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
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

impl SessionSetModeParams {
    /// Build params activating `mode_id` on `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, mode_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            mode_id: mode_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
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

impl SessionSetConfigOptionParams {
    /// Build params setting `config_id` on `session_id`. `value` accepts
    /// `&str`/`String` for `select` options and `bool` for `boolean` options.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        config_id: impl Into<String>,
        value: impl Into<SessionConfigValue>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            config_id: config_id.into(),
            value: value.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
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

impl PromptParams {
    /// Build params prompting `session_id` with `prompt`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, prompt: Vec<ContentBlock>) -> Self {
        Self {
            session_id: session_id.into(),
            prompt,
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
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

impl SessionCancelParams {
    /// Build params cancelling `session_id`'s current turn.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

impl From<&str> for SessionCancelParams {
    fn from(session_id: &str) -> Self {
        Self::new(session_id)
    }
}

impl From<String> for SessionCancelParams {
    fn from(session_id: String) -> Self {
        Self::new(session_id)
    }
}

// =============================================================================
// Content Types
// =============================================================================

/// Content block (text, image, audio, resource, resource link).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentBlock {
    /// Text content.
    Text(TextContent),
    /// Image content.
    Image(ImageContent),
    /// Audio content.
    Audio(AudioContent),
    /// Resource content.
    Resource(ResourceContent),
    /// A link to a resource without embedding its contents.
    #[serde(rename = "resource_link")]
    ResourceLink(ResourceLink),
}

/// Text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    /// Text value.
    pub text: String,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Image content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded data. Absent when `uri` carries the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// URI carrying the image instead of inline `data`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// MIME type.
    pub mime_type: String,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Audio content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioContent {
    /// Base64-encoded data.
    pub data: String,
    /// MIME type.
    pub mime_type: String,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Resource content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    /// Resource data.
    pub resource: EmbeddedResource,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Embedded resource: carries either `text` or base64 `blob` contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedResource {
    /// Resource URI.
    pub uri: String,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Text contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded binary contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A link to a resource without embedding its contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceLink {
    /// Resource URI.
    pub uri: String,
    /// Resource name.
    pub name: String,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ResourceLink {
    /// Build a resource link.
    #[must_use]
    pub fn new(uri: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            name: name.into(),
            title: None,
            description: None,
            mime_type: None,
            size: None,
            annotations: None,
            meta: None,
        }
    }
}

/// Content chunk for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentChunk {
    /// Content.
    pub content: ContentBlock,
    /// Optional identifier correlating chunks that belong to one message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
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
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
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
    /// Human-readable title; agents in the wild omit it, so tolerate absence.
    #[serde(default)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `audio` content blocks use the ACP `audio` tag and `mimeType` key,
    /// and round-trip back into the same variant.
    #[test]
    fn audio_content_block_serializes_with_audio_tag() {
        let block = ContentBlock::Audio(AudioContent {
            data: "AAAA".to_string(),
            mime_type: "audio/ogg".to_string(),
            annotations: None,
            meta: None,
        });
        let json = serde_json::to_value(&block).expect("serializes");
        assert_eq!(json["type"], "audio");
        assert_eq!(json["mimeType"], "audio/ogg");
        let parsed: ContentBlock = serde_json::from_value(json).expect("deserializes");
        assert!(matches!(parsed, ContentBlock::Audio(_)));
    }

    /// `session/resume` params use camelCase keys and skip empty
    /// `additionalDirectories`; the advertised capability deserializes from
    /// an empty object.
    #[test]
    fn session_resume_wire_format() {
        let params = SessionResumeParams {
            session_id: "s1".to_string(),
            cwd: PathBuf::from("/tmp/chat"),
            mcp_servers: vec![],
            additional_directories: vec![],
            meta: None,
        };
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["cwd"], "/tmp/chat");
        assert!(json.get("additionalDirectories").is_none());

        let caps: SessionCapabilities =
            serde_json::from_str(r#"{"resume":{}}"#).expect("deserializes");
        assert!(caps.resume.is_some());
        assert!(caps.close.is_none());
    }
}
