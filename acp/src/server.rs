//! ACP server that exposes aither agents to code editors.

use std::collections::HashMap;

use aither_mcp::transport::{BidirectionalTransport, StdioTransport};
use tracing::debug;

use crate::protocol::{
    AcpError, AgentCapabilities, Implementation, InitializeParams, InitializeResult, JsonRpcError,
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpCapabilities,
    PROTOCOL_VERSION, PromptCapabilities, PromptParams, PromptResult, Result, SessionCapabilities,
    SessionNewParams, SessionNewResult, SessionNotification, SessionStopParams, SessionUpdate,
    StopReason,
};
use crate::session::AcpSession;

/// ACP server that exposes aither agents to code editors.
///
/// # Example
///
/// ```ignore
/// use aither_acp::AcpServer;
///
/// let mut server = AcpServer::stdio("my-agent", "1.0.0")?;
/// server.run(|config| async {
///     // Create agent for this session
///     let agent = Agent::builder(llm)
///         .system_prompt("You are helpful")
///         .build();
///     Ok(agent)
/// }).await?;
/// ```
pub struct AcpServer<T: BidirectionalTransport> {
    transport: T,
    info: Implementation,
    sessions: HashMap<String, AcpSession>,
    initialized: bool,
}

impl<T: BidirectionalTransport> std::fmt::Debug for AcpServer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcpServer")
            .field("info", &self.info)
            .field("sessions", &self.sessions.len())
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl AcpServer<StdioTransport> {
    /// Create an ACP server using stdio transport.
    ///
    /// This is the standard way to create an ACP server that communicates
    /// via stdin/stdout (e.g., when run by Zed or other editors).
    ///
    /// # Arguments
    ///
    /// * `name` - The agent name.
    /// * `version` - The agent version.
    ///
    /// # Errors
    ///
    /// Returns an error if stdio cannot be initialized.
    pub fn stdio(name: impl Into<String>, version: impl Into<String>) -> Result<Self> {
        let transport = StdioTransport::new().map_err(|e| AcpError::Transport(e.to_string()))?;
        Ok(Self {
            transport,
            info: Implementation {
                name: name.into(),
                title: None,
                version: version.into(),
            },
            sessions: HashMap::new(),
            initialized: false,
        })
    }
}

impl<T: BidirectionalTransport> AcpServer<T> {
    /// Run the server main loop.
    ///
    /// This processes incoming requests until the connection is closed.
    ///
    /// # Errors
    ///
    /// Returns an error if a fatal transport error occurs.
    pub async fn run(&mut self) -> Result<()> {
        debug!("ACP server starting: {}", self.info.name);

        loop {
            if let Some(msg) = self.recv().await? {
                if let Err(e) = self.handle_message(msg).await {
                    debug!("Error handling message: {e}");
                }
            } else {
                debug!("Connection closed");
                break;
            }
        }

        Ok(())
    }

    /// Receive the next message from the transport.
    async fn recv(&mut self) -> Result<Option<JsonRpcMessage>> {
        self.transport
            .recv()
            .await
            .map_err(|e| AcpError::Transport(e.to_string()))
    }

    /// Send a notification to the client.
    async fn notify(&mut self, notif: JsonRpcNotification) -> Result<()> {
        self.transport
            .notify(notif)
            .await
            .map_err(|e| AcpError::Transport(e.to_string()))
    }

    /// Send a response to the client.
    async fn respond(&mut self, response: JsonRpcResponse) -> Result<()> {
        self.transport
            .respond(response)
            .await
            .map_err(|e| AcpError::Transport(e.to_string()))
    }

    /// Handle an incoming JSON-RPC message.
    async fn handle_message(&mut self, msg: JsonRpcMessage) -> Result<()> {
        match msg {
            JsonRpcMessage::Request(req) => {
                let response = self.handle_request(req).await;
                self.respond(response).await?;
            }
            JsonRpcMessage::Notification(notif) => {
                debug!("Received notification: {}", notif.method);
                // Handle notifications
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
            "session/new" => self.handle_session_new(req).await,
            "session/prompt" => self.handle_session_prompt(req).await,
            "session/stop" => self.handle_session_stop(req).await,
            method => JsonRpcResponse::error(req.id, JsonRpcError::method_not_found(method)),
        }
    }

    /// Handle initialize request.
    fn handle_initialize(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let _params: Option<InitializeParams> = match req
            .params
            .map(serde_json::from_value)
            .transpose()
        {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(req.id, JsonRpcError::invalid_params(e.to_string()));
            }
        };

        self.initialized = true;

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            agent_capabilities: AgentCapabilities {
                load_session: false, // No session persistence
                prompt_capabilities: PromptCapabilities {
                    image: false,
                    audio: false,
                    embedded_context: true,
                },
                mcp_capabilities: McpCapabilities {
                    http: true,
                    sse: false,
                },
                session_capabilities: SessionCapabilities {},
            },
            agent_info: self.info.clone(),
            auth_methods: vec![],
        };

        JsonRpcResponse::success(req.id, result)
    }

    /// Handle session/new request.
    async fn handle_session_new(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let params: SessionNewParams = match req.params.map(serde_json::from_value).transpose() {
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

        // Create new session
        let session = AcpSession::new(params.cwd, params.mcp_servers);
        let session_id = session.id().to_string();

        self.sessions.insert(session_id.clone(), session);

        JsonRpcResponse::success(req.id, SessionNewResult { session_id })
    }

    /// Handle session/prompt request.
    async fn handle_session_prompt(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let params: PromptParams = match req.params.map(serde_json::from_value).transpose() {
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

        let session = match self.sessions.get_mut(&params.session_id) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::invalid_params(format!(
                        "Session not found: {}",
                        params.session_id
                    )),
                );
            }
        };

        // Process prompt and stream updates
        // For now, we'll implement a simple placeholder
        // The actual implementation will integrate with the agent

        // Extract text from prompt
        let prompt_text = params
            .prompt
            .iter()
            .filter_map(|block| {
                if let crate::protocol::ContentBlock::Text(text) = block {
                    Some(text.text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Run the agent and stream updates
        let stop_reason = match session.prompt(&prompt_text, |update| {
            // Send session update notification
            let session_id = params.session_id.clone();
            let notif = SessionNotification { session_id, update };
            // Note: In a real implementation, we'd need async notification sending
            // For now, we'll collect updates and send them after
            debug!("Session update: {:?}", notif);
        }) {
            Ok(()) => StopReason::EndTurn,
            Err(e) => {
                debug!("Agent error: {e}");
                StopReason::Error
            }
        };

        JsonRpcResponse::success(req.id, PromptResult { stop_reason })
    }

    /// Handle session/stop request.
    async fn handle_session_stop(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let params: SessionStopParams = match req.params.map(serde_json::from_value).transpose() {
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

        match self.sessions.get_mut(&params.session_id) {
            Some(session) => {
                session.stop();
                JsonRpcResponse::success(req.id, serde_json::Value::Null)
            }
            None => JsonRpcResponse::error(
                req.id,
                JsonRpcError::invalid_params(format!("Session not found: {}", params.session_id)),
            ),
        }
    }

    /// Send a session update notification.
    pub async fn send_update(&mut self, session_id: &str, update: SessionUpdate) -> Result<()> {
        let notif = JsonRpcNotification::with_params(
            "session/update",
            SessionNotification {
                session_id: session_id.to_string(),
                update,
            },
        );
        self.notify(notif).await
    }
}
