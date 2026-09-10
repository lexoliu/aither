//! MCP connections for agent integration.
//!
//! Provides a unified interface for connecting to MCP servers
//! via different transports (child process, HTTP, stdio).

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use aither_core::llm::tool::ToolDefinition;
use aither_sandbox::{CommandPayload, ToolRegistryBuilder};
use async_lock::Mutex;
use serde::Deserialize;

use crate::protocol::{CallToolResult, McpError, McpToolDefinition};
#[cfg(feature = "http")]
use crate::transport::HttpTransport;
use crate::transport::{ChildProcessTransport, StdioTransport};

use super::McpClient;

/// Configuration for a single MCP server.
///
/// This matches the format used by Claude Desktop and other MCP clients.
///
/// # Example JSON
///
/// ```json
/// {
///   "command": "npx",
///   "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfig {
    /// The command to run (for process-based servers).
    pub command: Option<String>,

    /// Arguments to pass to the command.
    #[serde(default)]
    pub args: Vec<String>,

    /// URL for HTTP-based servers (requires the `http` feature).
    pub url: Option<String>,

    /// Optional environment variables for the process.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Configuration for multiple MCP servers.
///
/// # Example JSON
///
/// ```json
/// {
///   "filesystem": {
///     "command": "npx",
///     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
///   },
///   "github": {
///     "command": "npx",
///     "args": ["-y", "@modelcontextprotocol/server-github"]
///   }
/// }
/// ```
pub type McpServersConfig = HashMap<String, McpServerConfig>;

/// A single MCP connection with its cached tools.
///
/// This enum handles all transport types internally, hiding the
/// transport abstraction from users. Use the constructor methods
/// ([`spawn`](Self::spawn), [`stdio`](Self::stdio), and `http` with
/// the `http` feature) to create connections.
#[non_exhaustive]
#[allow(missing_docs)]
pub enum McpConnection {
    /// Connection via spawned child process.
    Process {
        client: McpClient<ChildProcessTransport>,
        tools: Vec<McpToolDefinition>,
        server_name: Option<String>,
    },
    /// Connection via HTTP (requires the `http` feature).
    #[cfg(feature = "http")]
    Http {
        client: McpClient<HttpTransport>,
        tools: Vec<McpToolDefinition>,
        server_name: Option<String>,
    },
    /// Connection via stdio (for use as a subprocess).
    Stdio {
        client: McpClient<StdioTransport>,
        tools: Vec<McpToolDefinition>,
        server_name: Option<String>,
    },
}

/// Service wrapper that serializes MCP tool calls through a command channel.
#[derive(Clone, Debug)]
pub struct McpToolService {
    conn: Arc<Mutex<McpConnection>>,
    tools: Vec<McpToolDefinition>,
}

impl std::fmt::Debug for McpConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process {
                server_name, tools, ..
            } => f
                .debug_struct("McpConnection::Process")
                .field("server_name", server_name)
                .field("tool_count", &tools.len())
                .finish(),
            #[cfg(feature = "http")]
            Self::Http {
                server_name, tools, ..
            } => f
                .debug_struct("McpConnection::Http")
                .field("server_name", server_name)
                .field("tool_count", &tools.len())
                .finish(),
            Self::Stdio {
                server_name, tools, ..
            } => f
                .debug_struct("McpConnection::Stdio")
                .field("server_name", server_name)
                .field("tool_count", &tools.len())
                .finish(),
        }
    }
}

impl McpConnection {
    /// Connect to an MCP server using configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = McpServerConfig {
    ///     command: Some("npx".to_string()),
    ///     args: vec!["-y".to_string(), "@modelcontextprotocol/server-filesystem".to_string()],
    ///     url: None,
    ///     env: HashMap::new(),
    /// };
    /// let conn = McpConnection::from_config(&config).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or connection fails.
    pub async fn from_config(config: &McpServerConfig) -> Result<Self, McpError> {
        if let Some(ref url) = config.url {
            // HTTP-based server
            #[cfg(feature = "http")]
            return Self::http(url).await;
            #[cfg(not(feature = "http"))]
            return Err(McpError::InvalidConfig(format!(
                "HTTP MCP server '{url}' requires the `http` feature"
            )));
        }
        if let Some(ref command) = config.command {
            // Process-based server
            let args: Vec<&str> = config
                .args
                .iter()
                .map(std::string::String::as_str)
                .collect();
            Self::spawn(command, &args).await
        } else {
            Err(McpError::InvalidConfig(
                "Config must have either 'command' or 'url'".to_string(),
            ))
        }
    }

    /// Connect to multiple MCP servers from a configuration map.
    ///
    /// Returns a vector of (name, connection) pairs for all connections.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config: McpServersConfig = serde_json::from_str(r#"{
    ///     "filesystem": {
    ///         "command": "npx",
    ///         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    ///     }
    /// }"#)?;
    ///
    /// let connections = McpConnection::from_configs(&config).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any connection fails.
    pub async fn from_configs(configs: &McpServersConfig) -> Result<Vec<(String, Self)>, McpError> {
        let mut connections = Vec::new();

        for (name, config) in configs {
            let conn = Self::from_config(config).await.map_err(|e| {
                McpError::InvalidConfig(format!("Failed to connect to MCP server '{name}': {e}"))
            })?;
            tracing::info!("Connected to MCP server: {name}");
            connections.push((name.clone(), conn));
        }

        Ok(connections)
    }

    /// Connect to an MCP server via a spawned child process.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let conn = McpConnection::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem"]).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be spawned or connection fails.
    pub async fn spawn(program: &str, args: &[&str]) -> Result<Self, McpError> {
        let transport = ChildProcessTransport::spawn(program, args)?;
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Process {
            client,
            tools,
            server_name,
        })
    }

    /// Connect to an MCP server via HTTP.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let conn = McpConnection::http("http://localhost:3000/mcp").await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP connection fails.
    #[cfg(feature = "http")]
    pub async fn http(url: &str) -> Result<Self, McpError> {
        let transport = HttpTransport::new(url);
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Http {
            client,
            tools,
            server_name,
        })
    }

    /// Connect to an MCP server via HTTP with authentication.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP connection fails.
    #[cfg(feature = "http")]
    pub async fn http_with_auth(url: &str, auth: &str) -> Result<Self, McpError> {
        let transport = HttpTransport::new(url).with_auth(auth);
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Http {
            client,
            tools,
            server_name,
        })
    }

    /// Connect via stdio (when running as a subprocess).
    ///
    /// # Errors
    ///
    /// Returns an error if stdio cannot be initialized.
    pub async fn stdio() -> Result<Self, McpError> {
        let transport = StdioTransport::new();
        let mut client = McpClient::connect(transport).await?;
        let tools = client.list_tools().await?;
        let server_name = client.server_info().map(|i| i.name.clone());

        Ok(Self::Stdio {
            client,
            tools,
            server_name,
        })
    }

    /// Returns the server name if available.
    #[must_use]
    pub fn server_name(&self) -> Option<&str> {
        match self {
            Self::Process { server_name, .. } | Self::Stdio { server_name, .. } => {
                server_name.as_deref()
            }
            #[cfg(feature = "http")]
            Self::Http { server_name, .. } => server_name.as_deref(),
        }
    }

    /// Returns the MCP tool definitions.
    #[must_use]
    pub fn mcp_definitions(&self) -> &[McpToolDefinition] {
        match self {
            Self::Process { tools, .. } | Self::Stdio { tools, .. } => tools,
            #[cfg(feature = "http")]
            Self::Http { tools, .. } => tools,
        }
    }

    /// Returns aither-compatible tool definitions.
    ///
    /// A tool whose schema the server sent in a form JSON Schema does not allow
    /// is dropped with a warning: it is unusable, and one bad entry should not
    /// cost the caller every other tool on the server.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        convert_definitions(self.mcp_definitions())
    }

    /// Check if this connection has a tool with the given name.
    #[must_use]
    pub fn has_tool(&self, name: &str) -> bool {
        self.mcp_definitions().iter().any(|d| d.name == name)
    }

    /// Call a tool on this MCP server.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool call fails.
    pub async fn call(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        match self {
            Self::Process { client, .. } => client.call_tool(name, arguments).await,
            #[cfg(feature = "http")]
            Self::Http { client, .. } => client.call_tool(name, arguments).await,
            Self::Stdio { client, .. } => client.call_tool(name, arguments).await,
        }
    }

    /// Close the connection.
    ///
    /// # Errors
    ///
    /// Returns an error if closing fails.
    pub async fn close(&mut self) -> Result<(), McpError> {
        match self {
            Self::Process { client, .. } => client.close().await,
            #[cfg(feature = "http")]
            Self::Http { client, .. } => client.close().await,
            Self::Stdio { client, .. } => client.close().await,
        }
    }
}

fn call_result_to_terminal_payload(
    result: CallToolResult,
) -> Result<Option<CommandPayload>, String> {
    let CallToolResult { content, is_error } = result;
    let all_text = content
        .iter()
        .all(|item| matches!(item, crate::Content::Text(_)));

    if all_text {
        let text = content
            .into_iter()
            .filter_map(|item| match item {
                crate::Content::Text(text_content) => Some(text_content.text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if is_error {
            Err(text)
        } else {
            Ok(Some(CommandPayload::Text { content: text }))
        }
    } else {
        let value = serde_json::to_value(CallToolResult { content, is_error })
            .map_err(|error| format!("failed to encode MCP tool result: {error}"))?;
        if is_error {
            Err(value.to_string())
        } else {
            Ok(Some(CommandPayload::Json { value }))
        }
    }
}

/// Registers all tools from an MCP connection as schema-driven terminal commands.
///
/// Each command derives its help text, positional arguments, and validation
/// behavior from the MCP tool's JSON schema via
/// [`aither_sandbox::ToolRegistryBuilder::configure_definition_handler`].
pub fn register_terminal_commands(
    registry: &mut ToolRegistryBuilder,
    conn: McpConnection,
) -> Vec<ToolDefinition> {
    let definitions = conn.definitions();
    let conn = std::sync::Arc::new(async_lock::Mutex::new(conn));

    for definition in &definitions {
        let tool_name = definition.name().to_string();
        let conn = conn.clone();
        registry.configure_definition_handler(definition, move |arguments| {
            let conn = conn.clone();
            let tool_name = tool_name.clone();
            Box::pin(async move {
                let result = {
                    let mut conn = conn.lock().await;
                    conn.call(tool_name.as_str(), arguments)
                        .await
                        .map_err(|error| error.to_string())?
                };
                call_result_to_terminal_payload(result)
            })
        });
    }

    definitions
}

impl McpToolService {
    /// Creates a new MCP tool service.
    #[must_use]
    pub fn new(conn: McpConnection) -> Self {
        let tools = conn.mcp_definitions().to_vec();
        Self {
            conn: Arc::new(Mutex::new(conn)),
            tools,
        }
    }

    /// Returns the MCP tool definitions.
    #[must_use]
    pub fn mcp_definitions(&self) -> &[McpToolDefinition] {
        &self.tools
    }

    /// Returns aither-compatible tool definitions.
    ///
    /// See [`McpConnection::definitions`] for how an unusable schema is handled.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        convert_definitions(&self.tools)
    }

    /// Check if this service has a tool with the given name.
    #[must_use]
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|d| d.name == name)
    }

    /// Call a tool on this MCP service.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool call fails.
    pub async fn call(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        let mut conn = self.conn.lock().await;
        conn.call(name, arguments).await
    }
}

/// Converts the server's tool list into aither definitions.
///
/// The schemas come off the wire from a third-party server, so one that is not
/// a JSON schema at all is a rejection to log, not a reason to fail: that tool
/// is skipped and the rest are returned.
fn convert_definitions(defs: &[McpToolDefinition]) -> Vec<ToolDefinition> {
    defs.iter()
        .filter_map(|def| {
            let name: Cow<'static, str> = Cow::Owned(def.name.clone());
            let description: Cow<'static, str> =
                Cow::Owned(def.description.clone().unwrap_or_default());
            match ToolDefinition::from_parts(name, description, def.input_schema.clone()) {
                Ok(definition) => Some(definition),
                Err(err) => {
                    tracing::warn!(error = %err, "skipping MCP tool with an unusable schema");
                    None
                }
            }
        })
        .collect()
}
