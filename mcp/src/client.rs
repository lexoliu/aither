//! MCP client implementation.
//!
//! The client connects to MCP servers and provides methods to list and call
//! tools, read resources, and so on. [`McpConnection`] wraps a client together
//! with the tools it discovered, ready to hand to an agent.
//!
//! Each client owns one receive loop (see [`McpClient::connect`]) that is the
//! only reader and writer of its transport. It routes responses to the
//! requests awaiting them, answers the server's `ping`, rejects server
//! requests for capabilities the client does not advertise, and hands
//! notifications to the [`McpHandler`] given at connect time.

mod connection;
mod handler;
#[cfg(all(test, feature = "http"))]
mod tests;
mod toolset;

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use arc_swap::ArcSwapOption;
use async_channel::{Receiver, Sender};
use executor_core::{Executor, Task};
use serde::de::DeserializeOwned;
use tracing::debug;

pub use handler::{McpHandler, ServerNotification};
pub use toolset::{
    McpConnection, McpServerConfig, McpServersConfig, McpToolService, call_result_to_tool_result,
    register_terminal_commands,
};

use connection::Command;

use crate::protocol::{
    CallToolParams, CallToolResult, InitializeParams, InitializeResult, JsonRpcNotification,
    JsonRpcRequest, ListToolsResult, McpError, McpToolDefinition, Resource, ResourceContents,
    ServerCapabilities, ServerInfo,
};
use crate::transport::BidirectionalTransport;

/// MCP client for connecting to and interacting with MCP servers.
///
/// # Example
///
/// ```ignore
/// use aither_mcp::{McpClient, ChildProcessTransport};
/// use executor_core::tokio::TokioGlobal;
///
/// let transport = ChildProcessTransport::spawn("npx", &["-y", "@mcp/server"])?;
/// let client = McpClient::connect(&TokioGlobal, transport, ()).await?;
///
/// // List available tools
/// let tools = client.list_tools().await?;
///
/// // Call a tool
/// let result = client.call_tool("my_tool", serde_json::json!({"arg": "value"})).await?;
/// ```
#[derive(Debug)]
pub struct McpClient {
    /// Requests and notifications for the receive loop to write.
    commands: Sender<Command>,
    /// Closes when the receive loop has finished.
    finished: Receiver<()>,
    /// Next request ID; the receive loop draws from it for its own requests.
    next_id: Arc<AtomicI64>,
    /// The server's tools as last listed. Only the receive loop replaces it
    /// once the connection is up, always with a complete list, so a reader
    /// sees the list before or after a refresh and never a mixture.
    tools: Arc<ArcSwapOption<Vec<McpToolDefinition>>>,
    server_info: ServerInfo,
    capabilities: ServerCapabilities,
}

impl McpClient {
    /// Connect to an MCP server and perform initialization.
    ///
    /// `executor` runs the connection's receive loop, which takes ownership of
    /// `transport`. The loop ends with the connection: when the transport
    /// reports EOF or fails, or when [`close`](Self::close) is called or the
    /// client is dropped. Requests pending at that point fail with an error
    /// that says why.
    ///
    /// `handler` receives the server's notifications.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection or initialization fails.
    pub async fn connect<T, H, E>(executor: &E, transport: T, handler: H) -> Result<Self, McpError>
    where
        T: BidirectionalTransport + 'static,
        H: McpHandler,
        E: Executor,
    {
        let (commands, command_rx) = async_channel::unbounded();
        let (alive, finished) = async_channel::bounded(1);
        let next_id = Arc::new(AtomicI64::new(1));
        let tools = Arc::new(ArcSwapOption::empty());
        executor
            .spawn(connection::run(
                transport,
                command_rx,
                // Weak, so the loop's own handle does not keep the
                // connection open after the client is dropped.
                commands.downgrade(),
                Arc::new(handler),
                connection::ToolCache {
                    tools: tools.clone(),
                    next_id: next_id.clone(),
                },
                alive,
            ))
            .detach();

        let mut client = Self {
            commands,
            finished,
            next_id,
            tools,
            server_info: ServerInfo {
                name: String::new(),
                version: None,
            },
            capabilities: ServerCapabilities::default(),
        };
        client.initialize().await?;
        Ok(client)
    }

    /// Perform MCP initialization handshake.
    async fn initialize(&mut self) -> Result<(), McpError> {
        let result: InitializeResult = self
            .request("initialize", Some(InitializeParams::default()))
            .await?;

        debug!(
            "Connected to MCP server: {} v{}",
            result.server_info.name,
            result.server_info.version.as_deref().unwrap_or("unknown")
        );

        self.server_info = result.server_info;
        self.capabilities = result.capabilities;

        // Send initialized notification
        let (reply, sent) = async_channel::bounded(1);
        self.commands
            .send(Command::Notification {
                notification: JsonRpcNotification::new("notifications/initialized"),
                reply,
            })
            .await
            .map_err(|_| McpError::ConnectionClosed)?;
        sent.recv().await.map_err(|_| McpError::ConnectionClosed)?
    }

    /// Send a request through the receive loop and decode its result.
    async fn request<P, R>(&self, method: &str, params: Option<P>) -> Result<R, McpError>
    where
        P: serde::Serialize,
        R: DeserializeOwned,
    {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let mut request = JsonRpcRequest::new(id, method);
        request.params = params.map(serde_json::to_value).transpose()?;

        let (reply, response) = async_channel::bounded(1);
        self.commands
            .send(Command::Request { request, reply })
            .await
            .map_err(|_| McpError::ConnectionClosed)?;
        let response = response
            .recv()
            .await
            .map_err(|_| McpError::ConnectionClosed)??;

        Ok(serde_json::from_value(
            response.into_result().map_err(McpError::JsonRpc)?,
        )?)
    }

    /// Get the server information.
    #[must_use]
    pub const fn server_info(&self) -> &ServerInfo {
        &self.server_info
    }

    /// Get the server capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> &ServerCapabilities {
        &self.capabilities
    }

    /// List available tools from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>, McpError> {
        let result: ListToolsResult = self.request("tools/list", None::<()>).await?;

        debug!("Listed {} tools", result.tools.len());
        Ok(result.tools)
    }

    /// The server's tools as of the latest listing the client holds.
    ///
    /// Empty until a listing is stored. [`McpConnection`] stores the first
    /// one when it connects; from then on the receive loop re-lists the tools
    /// whenever the server sends `notifications/tools/list_changed` and
    /// replaces the whole list at once.
    #[must_use]
    pub fn tools(&self) -> Arc<Vec<McpToolDefinition>> {
        self.tools.load_full().unwrap_or_default()
    }

    /// Store `tools` as the cached list unless a refresh already stored a
    /// newer one.
    fn seed_tools(&self, tools: Vec<McpToolDefinition>) {
        let tools = Arc::new(tools);
        self.tools
            .rcu(|current| current.clone().or_else(|| Some(tools.clone())));
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
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        let params = CallToolParams {
            name: name.to_string(),
            arguments,
            meta: None,
        };
        let result: CallToolResult = self.request("tools/call", Some(params)).await?;

        debug!(
            "Tool {} returned {} content items",
            name,
            result.content.len()
        );
        Ok(result)
    }

    /// List available resources from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or resources are not supported.
    pub async fn list_resources(&self) -> Result<Vec<Resource>, McpError> {
        #[derive(serde::Deserialize)]
        struct ListResourcesResult {
            resources: Vec<Resource>,
        }

        let result: ListResourcesResult = self.request("resources/list", None::<()>).await?;

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
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceContents, McpError> {
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
        let result: ReadResourceResult = self.request("resources/read", Some(params)).await?;

        result
            .contents
            .into_iter()
            .next()
            .ok_or_else(|| McpError::Transport("No content returned".to_string()))
    }

    /// Close the client connection.
    ///
    /// Stops the receive loop, which closes the transport (terminating a
    /// child process) and fails every request still pending, and waits for it
    /// to finish. Closing an already closed connection does nothing.
    pub async fn close(&self) {
        self.commands.close();
        // The loop drops its end when it is done; nothing is ever sent.
        let _ = self.finished.recv().await;
    }
}
