//! MCP client for connecting to MCP servers.

use tracing::debug;

use crate::protocol::{
    CallToolParams, CallToolResult, InitializeParams, InitializeResult, JsonRpcRequest,
    ListToolsResult, McpError, McpToolDefinition, Resource, ResourceContents, ServerCapabilities,
    ServerInfo,
};
use crate::transport::Transport;

/// MCP client for connecting to and interacting with MCP servers.
///
/// # Example
///
/// ```ignore
/// use aither_mcp::{McpClient, ChildProcessTransport};
///
/// let transport = ChildProcessTransport::spawn("npx", &["-y", "@mcp/server"])?;
/// let mut client = McpClient::connect(transport).await?;
///
/// // List available tools
/// let tools = client.list_tools().await?;
///
/// // Call a tool
/// let result = client.call_tool("my_tool", serde_json::json!({"arg": "value"})).await?;
/// ```
#[derive(Debug)]
pub struct McpClient<T: Transport> {
    transport: T,
    server_info: Option<ServerInfo>,
    capabilities: ServerCapabilities,
}

impl<T: Transport> McpClient<T> {
    /// Connect to an MCP server and perform initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection or initialization fails.
    pub async fn connect(transport: T) -> Result<Self, McpError> {
        let mut client = Self {
            transport,
            server_info: None,
            capabilities: ServerCapabilities::default(),
        };

        client.initialize().await?;

        Ok(client)
    }

    /// Perform MCP initialization handshake.
    async fn initialize(&mut self) -> Result<(), McpError> {
        let params = InitializeParams::default();
        let request = JsonRpcRequest::with_params(0i64, "initialize", params);

        let response = self.transport.request(request).await?;
        let result: InitializeResult =
            serde_json::from_value(response.into_result().map_err(McpError::JsonRpc)?)?;

        debug!(
            "Connected to MCP server: {} v{}",
            result.server_info.name,
            result.server_info.version.as_deref().unwrap_or("unknown")
        );

        self.server_info = Some(result.server_info);
        self.capabilities = result.capabilities;

        // Send initialized notification
        let notif = crate::protocol::JsonRpcNotification::new("notifications/initialized");
        self.transport.notify(notif).await?;

        Ok(())
    }

    /// Get the server information.
    #[must_use]
    pub fn server_info(&self) -> Option<&ServerInfo> {
        self.server_info.as_ref()
    }

    /// Get the server capabilities.
    #[must_use]
    pub fn capabilities(&self) -> &ServerCapabilities {
        &self.capabilities
    }

    /// List available tools from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn list_tools(&mut self) -> Result<Vec<McpToolDefinition>, McpError> {
        let request = JsonRpcRequest::new(0i64, "tools/list");
        let response = self.transport.request(request).await?;
        let result: ListToolsResult =
            serde_json::from_value(response.into_result().map_err(McpError::JsonRpc)?)?;

        debug!("Listed {} tools", result.tools.len());
        Ok(result.tools)
    }

    /// Call a tool on the server.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name.
    /// * `arguments` - The tool arguments as a JSON value.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        let params = CallToolParams {
            name: name.to_string(),
            arguments,
        };
        let request = JsonRpcRequest::with_params(0i64, "tools/call", params);

        let response = self.transport.request(request).await?;
        let result: CallToolResult =
            serde_json::from_value(response.into_result().map_err(McpError::JsonRpc)?)?;

        debug!("Tool {} returned {} content items", name, result.content.len());
        Ok(result)
    }

    /// List available resources from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or resources are not supported.
    pub async fn list_resources(&mut self) -> Result<Vec<Resource>, McpError> {
        #[derive(serde::Deserialize)]
        struct ListResourcesResult {
            resources: Vec<Resource>,
        }

        let request = JsonRpcRequest::new(0i64, "resources/list");
        let response = self.transport.request(request).await?;
        let result: ListResourcesResult =
            serde_json::from_value(response.into_result().map_err(McpError::JsonRpc)?)?;

        debug!("Listed {} resources", result.resources.len());
        Ok(result.resources)
    }

    /// Read a resource from the server.
    ///
    /// # Arguments
    ///
    /// * `uri` - The resource URI.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn read_resource(&mut self, uri: &str) -> Result<ResourceContents, McpError> {
        #[derive(serde::Serialize)]
        struct ReadResourceParams {
            uri: String,
        }

        #[derive(serde::Deserialize)]
        struct ReadResourceResult {
            contents: Vec<ResourceContents>,
        }

        let params = ReadResourceParams {
            uri: uri.to_string(),
        };
        let request = JsonRpcRequest::with_params(0i64, "resources/read", params);

        let response = self.transport.request(request).await?;
        let result: ReadResourceResult =
            serde_json::from_value(response.into_result().map_err(McpError::JsonRpc)?)?;

        result
            .contents
            .into_iter()
            .next()
            .ok_or_else(|| McpError::Transport("No content returned".to_string()))
    }

    /// Close the client connection.
    ///
    /// # Errors
    ///
    /// Returns an error if closing fails.
    pub async fn close(&mut self) -> Result<(), McpError> {
        self.transport.close().await
    }
}
