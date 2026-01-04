//! MCP server that exposes aither tools.

use aither_core::llm::tool::Tools;
use tracing::debug;

use crate::protocol::{
    CallToolParams, CallToolResult, InitializeParams, InitializeResult, JsonRpcError,
    JsonRpcMessage, JsonRpcRequest, JsonRpcResponse, ListToolsResult, McpError, McpToolDefinition,
    ServerCapabilities, ServerInfo, TextContent, ToolsCapability, PROTOCOL_VERSION,
};
use crate::transport::{BidirectionalTransport, StdioTransport};

/// MCP server that exposes aither tools to external clients.
///
/// # Example
///
/// ```ignore
/// use aither_mcp::McpServer;
/// use aither_core::llm::tool::Tools;
///
/// let mut tools = Tools::new();
/// tools.register(my_tool);
///
/// let mut server = McpServer::stdio(tools, "my-server", "1.0.0")?;
/// server.run().await?;
/// ```
pub struct McpServer<T: BidirectionalTransport> {
    transport: T,
    tools: Tools,
    info: ServerInfo,
    initialized: bool,
}

impl<T: BidirectionalTransport> std::fmt::Debug for McpServer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServer")
            .field("info", &self.info)
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl McpServer<StdioTransport> {
    /// Create an MCP server using stdio transport.
    ///
    /// This is the standard way to create an MCP server that communicates
    /// via stdin/stdout (e.g., when run by Claude Desktop).
    ///
    /// # Arguments
    ///
    /// * `tools` - The aither tools to expose.
    /// * `name` - The server name.
    /// * `version` - The server version.
    ///
    /// # Errors
    ///
    /// Returns an error if stdio cannot be initialized.
    pub fn stdio(
        tools: Tools,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self, McpError> {
        let transport = StdioTransport::new().map_err(|e| McpError::Transport(e.to_string()))?;
        Ok(Self {
            transport,
            tools,
            info: ServerInfo {
                name: name.into(),
                version: Some(version.into()),
            },
            initialized: false,
        })
    }
}

impl<T: BidirectionalTransport> McpServer<T> {
    /// Create a new MCP server with a custom transport.
    ///
    /// For most use cases, prefer `McpServer::stdio()` instead.
    ///
    /// # Arguments
    ///
    /// * `transport` - The transport to use for communication.
    /// * `tools` - The aither tools to expose.
    /// * `name` - The server name.
    /// * `version` - The server version.
    #[must_use]
    pub(crate) fn new(
        transport: T,
        tools: Tools,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            tools,
            info: ServerInfo {
                name: name.into(),
                version: Some(version.into()),
            },
            initialized: false,
        }
    }

    /// Run the server main loop.
    ///
    /// This processes incoming requests until the connection is closed.
    ///
    /// # Errors
    ///
    /// Returns an error if a fatal transport error occurs.
    pub async fn run(&mut self) -> Result<(), McpError> {
        debug!("MCP server starting: {}", self.info.name);

        loop {
            match self.transport.recv().await? {
                Some(msg) => {
                    if let Err(e) = self.handle_message(msg).await {
                        debug!("Error handling message: {e}");
                    }
                }
                None => {
                    debug!("Connection closed");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle an incoming JSON-RPC message.
    async fn handle_message(&mut self, msg: JsonRpcMessage) -> Result<(), McpError> {
        match msg {
            JsonRpcMessage::Request(req) => {
                let response = self.handle_request(req).await;
                self.transport.respond(response).await?;
            }
            JsonRpcMessage::Notification(notif) => {
                debug!("Received notification: {}", notif.method);
                // Handle notifications (e.g., initialized)
                if notif.method == "notifications/initialized" {
                    debug!("Client initialized");
                }
            }
            JsonRpcMessage::Response(_) => {
                // We don't expect responses as a server
                debug!("Unexpected response message");
            }
        }
        Ok(())
    }

    /// Handle an incoming request.
    async fn handle_request(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: {}", req.method);

        match req.method.as_str() {
            "initialize" => self.handle_initialize(req),
            "tools/list" => self.handle_list_tools(req),
            "tools/call" => self.handle_call_tool(req).await,
            method => JsonRpcResponse::error(req.id, JsonRpcError::method_not_found(method)),
        }
    }

    /// Handle initialize request.
    fn handle_initialize(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let _params: InitializeParams = match req.params.map(serde_json::from_value).transpose() {
            Ok(p) => p.unwrap_or_default(),
            Err(e) => {
                return JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::invalid_params(e.to_string()),
                );
            }
        };

        self.initialized = true;

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability { list_changed: None }),
                ..Default::default()
            },
            server_info: self.info.clone(),
            instructions: None,
        };

        JsonRpcResponse::success(req.id, result)
    }

    /// Handle tools/list request.
    fn handle_list_tools(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let definitions = self.tools.definitions();

        let mcp_tools: Vec<McpToolDefinition> = definitions
            .iter()
            .map(|def| McpToolDefinition {
                name: def.name().to_string(),
                description: Some(def.description().to_string()),
                input_schema: def.arguments_openai_schema(),
            })
            .collect();

        let result = ListToolsResult {
            tools: mcp_tools,
            next_cursor: None,
        };

        JsonRpcResponse::success(req.id, result)
    }

    /// Handle tools/call request.
    async fn handle_call_tool(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let params: CallToolParams = match req.params.map(serde_json::from_value).transpose() {
            Ok(Some(p)) => p,
            Ok(None) => {
                return JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::invalid_params("Missing params"),
                );
            }
            Err(e) => {
                return JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::invalid_params(e.to_string()),
                );
            }
        };

        let args_str = serde_json::to_string(&params.arguments).unwrap_or_default();

        match self.tools.call(&params.name, &args_str).await {
            Ok(output) => {
                let result = CallToolResult {
                    content: vec![crate::protocol::Content::Text(TextContent {
                        text: output,
                        annotations: None,
                    })],
                    is_error: false,
                };
                JsonRpcResponse::success(req.id, result)
            }
            Err(e) => {
                let result = CallToolResult {
                    content: vec![crate::protocol::Content::Text(TextContent {
                        text: e.to_string(),
                        annotations: None,
                    })],
                    is_error: true,
                };
                JsonRpcResponse::success(req.id, result)
            }
        }
    }
}
