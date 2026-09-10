//! MCP server that exposes aither tools.

use std::collections::HashMap;
use std::sync::Arc;

use aither_core::llm::tool::Tools;
use futures_lite::future;
use futures_util::future::{AbortHandle, Abortable, Aborted, BoxFuture};
use futures_util::stream::{FuturesUnordered, StreamExt};
use tracing::debug;

use crate::protocol::{
    CallToolParams, CallToolResult, CancelledParams, InitializeParams, InitializeResult,
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    ListToolsResult, McpError, McpToolDefinition, PROTOCOL_VERSION, RequestId, ServerCapabilities,
    ServerInfo, TextContent, ToolsCapability,
};
use crate::transport::{BidirectionalTransport, StdioTransport};

/// A `tools/call` execution that resolves to its request ID and response, or
/// to [`Aborted`] once a `notifications/cancelled` notification names its ID.
type InFlightCall = Abortable<BoxFuture<'static, (RequestId, JsonRpcResponse)>>;

/// The `tools/call` executions currently running, resolved in any order.
type InFlightCalls = FuturesUnordered<InFlightCall>;

/// One event driving an iteration of the server loop.
enum ServerEvent {
    /// The client sent a message, closed the connection, or the transport failed.
    Message(Result<Option<JsonRpcMessage>, McpError>),
    /// An in-flight `tools/call` produced its response or was aborted.
    Call(Result<(RequestId, JsonRpcResponse), Aborted>),
}

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
/// let mut server = McpServer::stdio(tools, "my-server", "1.0.0");
/// server.run().await?;
/// ```
pub struct McpServer<T: BidirectionalTransport> {
    transport: T,
    tools: Arc<Tools>,
    info: ServerInfo,
    instructions: Option<String>,
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
    #[must_use]
    pub fn stdio(tools: Tools, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self::new(StdioTransport::new(), tools, name, version)
    }
}

impl<T: BidirectionalTransport + Sync> McpServer<T> {
    /// Create a new MCP server with a custom transport.
    ///
    /// For most use cases, prefer [`McpServer::stdio`] instead.
    ///
    /// # Arguments
    ///
    /// * `transport` - The transport to use for communication.
    /// * `tools` - The aither tools to expose.
    /// * `name` - The server name.
    /// * `version` - The server version.
    #[must_use]
    pub fn new(
        transport: T,
        tools: Tools,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            tools: Arc::new(tools),
            info: ServerInfo {
                name: name.into(),
                version: Some(version.into()),
            },
            instructions: None,
            initialized: false,
        }
    }

    /// Adds server-wide instructions returned during MCP initialization.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Run the server main loop.
    ///
    /// Processes incoming messages until the connection is closed.
    /// `tools/call` requests execute concurrently: each becomes an in-flight
    /// future raced against the next incoming message, and its response is
    /// written as soon as the call completes, in any order. A
    /// `notifications/cancelled` notification drops the in-flight call whose
    /// request ID it carries; per the MCP specification no response is then
    /// sent for it.
    ///
    /// # Errors
    ///
    /// Returns an error if a fatal transport error occurs.
    ///
    /// # Panics
    ///
    /// Panics only on an internal bug: the in-flight call set is polled only
    /// while it is non-empty, in which case it always yields a completed call.
    pub async fn run(&mut self) -> Result<(), McpError> {
        debug!("MCP server starting: {}", self.info.name);

        let mut calls = InFlightCalls::new();
        let mut cancellations: HashMap<RequestId, AbortHandle> = HashMap::new();

        loop {
            let event = if calls.is_empty() {
                ServerEvent::Message(self.transport.recv().await)
            } else {
                future::race(
                    async { ServerEvent::Message(self.transport.recv().await) },
                    async { ServerEvent::Call(calls.next().await.expect("call set is not empty")) },
                )
                .await
            };

            match event {
                ServerEvent::Message(Ok(Some(JsonRpcMessage::Request(req))))
                    if req.method == "tools/call" =>
                {
                    self.enqueue_call(req, &calls, &mut cancellations);
                }
                ServerEvent::Message(Ok(Some(msg))) => {
                    if let Err(e) = self.handle_message(msg, &mut cancellations).await {
                        debug!("Error handling message: {e}");
                    }
                }
                ServerEvent::Message(Ok(None)) => {
                    debug!("Connection closed");
                    break;
                }
                ServerEvent::Message(Err(e)) => return Err(e),
                ServerEvent::Call(Ok((id, response))) => {
                    cancellations.remove(&id);
                    self.transport.respond(response).await?;
                }
                ServerEvent::Call(Err(Aborted)) => {
                    debug!("In-flight tool call was cancelled");
                }
            }
        }

        Ok(())
    }

    /// Push a `tools/call` request onto the in-flight set and keep its abort
    /// handle in `cancellations` for `notifications/cancelled`.
    ///
    /// Synchronous so that `&calls` — which is not [`Sync`] — never crosses an
    /// await point and the server loop's future stays `Send`.
    fn enqueue_call(
        &self,
        req: JsonRpcRequest,
        calls: &InFlightCalls,
        cancellations: &mut HashMap<RequestId, AbortHandle>,
    ) {
        let (handle, registration) = AbortHandle::new_pair();
        let tools = Arc::clone(&self.tools);
        let call_id = req.id.clone();
        let map_id = req.id.clone();
        let call: BoxFuture<'static, (RequestId, JsonRpcResponse)> =
            Box::pin(async move { (call_id, Self::handle_call_tool(&tools, req).await) });
        calls.push(Abortable::new(call, registration));
        cancellations.insert(map_id, handle);
    }

    /// Handle an incoming JSON-RPC message other than a `tools/call` request,
    /// which [`run`](Self::run) enqueues before this is reached. Every other
    /// request is answered inline through the transport before this returns.
    async fn handle_message(
        &mut self,
        msg: JsonRpcMessage,
        cancellations: &mut HashMap<RequestId, AbortHandle>,
    ) -> Result<(), McpError> {
        match msg {
            JsonRpcMessage::Request(req) => {
                let response = self.handle_request(req);
                self.transport.respond(response).await?;
            }
            JsonRpcMessage::Notification(notif) => {
                Self::handle_notification(notif, cancellations);
            }
            JsonRpcMessage::Response(_) => {
                // We don't expect responses as a server
                debug!("Unexpected response message");
            }
        }
        Ok(())
    }

    /// Handle an incoming request other than `tools/call`, which
    /// [`run`](Self::run) enqueues before this is reached.
    fn handle_request(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: {}", req.method);

        match req.method.as_str() {
            "initialize" => self.handle_initialize(req),
            "tools/list" => self.handle_list_tools(req),
            method => JsonRpcResponse::error(req.id, JsonRpcError::method_not_found(method)),
        }
    }

    /// Handle an incoming JSON-RPC notification.
    ///
    /// `notifications/cancelled` aborts the in-flight `tools/call` whose
    /// request ID it names. Cancelling an unknown or already-finished request
    /// is ignored, as the specification allows.
    fn handle_notification(
        notif: JsonRpcNotification,
        cancellations: &mut HashMap<RequestId, AbortHandle>,
    ) {
        debug!("Received notification: {}", notif.method);

        match notif.method.as_str() {
            "notifications/initialized" => {
                debug!("Client initialized");
            }
            "notifications/cancelled" => {
                match notif
                    .params
                    .map(serde_json::from_value::<CancelledParams>)
                    .transpose()
                {
                    Ok(Some(params)) => {
                        if let Some(handle) = cancellations.remove(&params.request_id) {
                            debug!("Cancelling tool call {:?}", params.request_id);
                            handle.abort();
                        } else {
                            debug!("No in-flight call {:?}", params.request_id);
                        }
                    }
                    Ok(None) => debug!("notifications/cancelled without params"),
                    Err(e) => debug!("Invalid notifications/cancelled params: {e}"),
                }
            }
            _ => {}
        }
    }

    /// Handle initialize request.
    fn handle_initialize(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let _params: InitializeParams = match req.params.map(serde_json::from_value).transpose() {
            Ok(p) => p.unwrap_or_default(),
            Err(e) => {
                return JsonRpcResponse::error(req.id, JsonRpcError::invalid_params(e.to_string()));
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
            instructions: self.instructions.clone(),
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
    ///
    /// Takes the shared tool table rather than `&self` so the returned future
    /// owns no borrow of the server and can run inside the in-flight set.
    async fn handle_call_tool(tools: &Tools, req: JsonRpcRequest) -> JsonRpcResponse {
        let params: CallToolParams = match req.params.map(serde_json::from_value).transpose() {
            Ok(Some(p)) => p,
            Ok(None) => {
                return JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::invalid_params("Missing params"),
                );
            }
            Err(e) => {
                return JsonRpcResponse::error(req.id, JsonRpcError::invalid_params(e.to_string()));
            }
        };

        let args_str = serde_json::to_string(&params.arguments).unwrap_or_default();

        match tools.call(&params.name, &args_str).await {
            Ok(output) => {
                let text = match output.render_for_model() {
                    Ok(text) => text,
                    Err(error) => {
                        return JsonRpcResponse::error(
                            req.id,
                            JsonRpcError::internal_error(error.to_string()),
                        );
                    }
                };
                let result = CallToolResult {
                    content: vec![crate::protocol::Content::Text(TextContent {
                        text,
                        annotations: None,
                    })],
                    is_error: output.is_error(),
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

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    use aither_core::llm::tool::{ToolDefinition, ToolResult};
    use serde_json::json;

    use super::*;
    use crate::transport::{DuplexTransport, Transport};

    /// The handler signature accepted by [`Tools::register_dyn`].
    type ToolHandler = dyn Fn(&str) -> Pin<Box<dyn Future<Output = aither_core::Result<ToolResult>> + Send>>
        + Send
        + Sync;

    /// Sends once dropped, marking that the future holding it was dropped.
    struct DropMarker(async_channel::Sender<()>);

    impl Drop for DropMarker {
        fn drop(&mut self) {
            let _ = self.0.try_send(());
        }
    }

    /// Register `name` running `handler` on the JSON arguments string.
    fn register(tools: &mut Tools, name: &'static str, handler: Box<ToolHandler>) {
        let definition = ToolDefinition::from_parts(
            name.into(),
            "test tool".into(),
            json!({"type": "object", "properties": {}}),
        )
        .expect("valid schema");
        tools.register_dyn(definition, handler).expect("registers");
    }

    /// A `tools/call` request for `name` with empty arguments.
    fn call(id: i64, name: &str) -> JsonRpcRequest {
        JsonRpcRequest::with_params(
            id,
            "tools/call",
            CallToolParams {
                name: name.to_string(),
                arguments: json!({}),
            },
        )
    }

    /// Receive the next response arriving at `client`.
    async fn next_response(client: &mut DuplexTransport) -> JsonRpcResponse {
        match client.recv().await.expect("recv") {
            Some(JsonRpcMessage::Response(response)) => response,
            other => panic!("expected a response, got {other:?}"),
        }
    }

    /// Two `tools/call` requests execute concurrently: the call blocked on the
    /// gate answers only after the second call opens it, so the second
    /// request's response arrives first.
    #[tokio::test]
    async fn tool_calls_run_concurrently() {
        let (gate_tx, gate_rx) = async_channel::unbounded::<()>();

        let mut tools = Tools::new();
        register(
            &mut tools,
            "wait",
            Box::new(move |_args| {
                let gate_rx = gate_rx.clone();
                Box::pin(async move {
                    gate_rx.recv().await.expect("gate open");
                    Ok(ToolResult::text("waited"))
                })
            }),
        );
        register(
            &mut tools,
            "release",
            Box::new(move |_args| {
                let gate_tx = gate_tx.clone();
                Box::pin(async move {
                    gate_tx.send(()).await.expect("gate send");
                    Ok(ToolResult::text("released"))
                })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });

        client.send_request(call(1, "wait")).await.expect("send");
        client.send_request(call(2, "release")).await.expect("send");

        assert_eq!(next_response(&mut client).await.id, RequestId::Number(2));
        assert_eq!(next_response(&mut client).await.id, RequestId::Number(1));

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    /// `notifications/cancelled` drops the in-flight call: the wait tool's
    /// future is dropped while it blocks on the gate, and no response is
    /// written for it even though a later call opens the gate.
    #[tokio::test]
    async fn cancelled_call_is_dropped() {
        let (gate_tx, gate_rx) = async_channel::unbounded::<()>();
        let (started_tx, started_rx) = async_channel::unbounded::<()>();
        let (dropped_tx, dropped_rx) = async_channel::unbounded::<()>();

        let mut tools = Tools::new();
        register(
            &mut tools,
            "wait",
            Box::new(move |_args| {
                let gate_rx = gate_rx.clone();
                let started_tx = started_tx.clone();
                let dropped_tx = dropped_tx.clone();
                Box::pin(async move {
                    let _marker = DropMarker(dropped_tx);
                    started_tx.send(()).await.expect("started send");
                    gate_rx.recv().await.expect("gate open");
                    Ok(ToolResult::text("waited"))
                })
            }),
        );
        register(
            &mut tools,
            "release",
            Box::new(move |_args| {
                let gate_tx = gate_tx.clone();
                Box::pin(async move {
                    gate_tx.send(()).await.expect("gate send");
                    Ok(ToolResult::text("released"))
                })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });

        client.send_request(call(1, "wait")).await.expect("send");

        // Wait until the call is actually in flight and blocked on the gate,
        // so the cancel drops a running future rather than a pending one.
        started_rx.recv().await.expect("started");

        client
            .notify(JsonRpcNotification::with_params(
                "notifications/cancelled",
                CancelledParams {
                    request_id: RequestId::Number(1),
                    reason: None,
                },
            ))
            .await
            .expect("notify");
        client.send_request(call(2, "release")).await.expect("send");

        assert_eq!(next_response(&mut client).await.id, RequestId::Number(2));

        // The call was dropped rather than left running: its marker fired.
        tokio::time::timeout(Duration::from_secs(5), dropped_rx.recv())
            .await
            .expect("call dropped")
            .expect("marker channel");

        // And no response for it was ever written, though the gate was opened.
        assert!(future::poll_once(client.recv()).await.is_none());

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }
}
