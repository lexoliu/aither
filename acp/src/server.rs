//! ACP server that exposes aither agents to code editors.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aither_agent::Agent;
use aither_core::LanguageModel;
use aither_mcp::transport::{BidirectionalTransport, StdioTransport};
use futures_lite::future::or;
use serde::de::DeserializeOwned;
use tracing::debug;

use crate::protocol::{
    AcpError, AgentCapabilities, ContentBlock, Implementation, InitializeParams, InitializeResult,
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    McpCapabilities, PROTOCOL_VERSION, PromptCapabilities, PromptParams, PromptResult, RequestId,
    Result, SessionCancelParams, SessionCapabilities, SessionCloseCapabilities, SessionCloseParams,
    SessionCloseResult, SessionDeleteCapabilities, SessionDeleteParams, SessionDeleteResult,
    SessionInfo, SessionListCapabilities, SessionListResult, SessionNewParams, SessionNewResult,
    SessionNotification, SessionUpdate, StopReason,
};
use crate::session::AcpSession;

/// ACP server that exposes aither agents to code editors.
///
/// # Example
///
/// ```ignore
/// use aither_acp::AcpServer;
///
/// let mut server = AcpServer::stdio("my-agent", "1.0.0");
/// server.run(|config| async {
///     // Create agent for this session
///     let agent = Agent::builder(llm)
///         .system_prompt("You are helpful")
///         .build();
///     Ok(agent)
/// }).await?;
/// ```
pub struct AcpServer<T: BidirectionalTransport, LLM: LanguageModel> {
    transport: T,
    info: Implementation,
    sessions: HashMap<String, AcpSession<LLM>>,
    /// Cancellation flags indexed by session ID.
    ///
    /// A prompt turn borrows its session out of `sessions`, so a
    /// `session/cancel` notification cannot reach it through the map; the
    /// flag is shared state that always can.
    cancellations: HashMap<String, Arc<AtomicBool>>,
    initialized: bool,
    /// Builds a fresh agent for each new session.
    ///
    /// Sessions are independent conversations, so each gets its own agent
    /// rather than sharing one and interleaving their contexts. Building one
    /// is async because setting up a sandbox touches the filesystem.
    make_agent: AgentFactory<LLM>,
}

/// Builds the agent backing a new session, given that session's working directory.
pub type AgentFactory<LLM> = Box<
    dyn Fn(PathBuf) -> Pin<Box<dyn Future<Output = Result<Agent<LLM, LLM, LLM>>> + Send>> + Send,
>;

impl<T: BidirectionalTransport, LLM: LanguageModel> std::fmt::Debug for AcpServer<T, LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcpServer")
            .field("info", &self.info)
            .field("sessions", &self.sessions.len())
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl<LLM: LanguageModel> AcpServer<StdioTransport, LLM> {
    /// Create an ACP server using stdio transport.
    ///
    /// This is the standard way to create an ACP server that communicates
    /// via stdin/stdout (e.g., when run by Zed or other editors).
    ///
    /// # Arguments
    ///
    /// * `name` - The agent name.
    /// * `version` - The agent version.
    /// * `make_agent` - Builds the agent backing each new session, given the
    ///   session's working directory.
    pub fn stdio<F, Fut>(name: impl Into<String>, version: impl Into<String>, make_agent: F) -> Self
    where
        F: Fn(PathBuf) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Agent<LLM, LLM, LLM>>> + Send + 'static,
    {
        Self {
            transport: StdioTransport::new(),
            info: Implementation {
                name: name.into(),
                title: None,
                version: version.into(),
            },
            sessions: HashMap::new(),
            cancellations: HashMap::new(),
            initialized: false,
            make_agent: Box::new(move |cwd| Box::pin(make_agent(cwd))),
        }
    }
}

/// Which side of a prompt-turn `select` produced a value.
enum Progress<E> {
    /// The agent finished the turn.
    Turn(std::result::Result<(), String>),
    /// A client message arrived mid-turn.
    Message(std::result::Result<Option<JsonRpcMessage>, E>),
    /// The agent produced a session update.
    Update(SessionUpdate),
}

impl<T: BidirectionalTransport, LLM: LanguageModel> AcpServer<T, LLM> {
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
                self.handle_notification(&notif);
            }
            JsonRpcMessage::Response(_) => {
                // We don't expect responses as a server
                debug!("Unexpected response message");
            }
        }
        Ok(())
    }

    /// Handle an incoming notification. `session/cancel` sets the session's
    /// shared cancellation flag; everything else is logged and ignored, as
    /// the protocol requires for unrecognized notifications.
    fn handle_notification(&self, notif: &JsonRpcNotification) {
        match notif.method.as_str() {
            "session/cancel" => match notif.params.clone().map(serde_json::from_value) {
                Some(Ok(SessionCancelParams { session_id, .. })) => {
                    if let Some(flag) = self.cancellations.get(&session_id) {
                        flag.store(true, Ordering::SeqCst);
                    } else {
                        debug!("session/cancel for unknown session: {session_id}");
                    }
                }
                Some(Err(error)) => {
                    debug!("malformed session/cancel notification: {error}");
                }
                None => debug!("session/cancel notification without params"),
            },
            "notifications/initialized" => debug!("Client initialized"),
            method => debug!("ignoring client notification: {method}"),
        }
    }

    /// Handle an incoming request.
    async fn handle_request(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: {}", req.method);

        match req.method.as_str() {
            "initialize" => self.handle_initialize(req),
            "session/new" => self.handle_session_new(req).await,
            "session/prompt" => self.handle_session_prompt(req).await,
            "session/list" => self.handle_session_list(req),
            "session/close" => self.handle_session_close(req),
            "session/delete" => self.handle_session_delete(req),
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
            protocol_version: PROTOCOL_VERSION,
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
                session_capabilities: SessionCapabilities {
                    list: Some(SessionListCapabilities::default()),
                    delete: Some(SessionDeleteCapabilities::default()),
                    close: Some(SessionCloseCapabilities::default()),
                    ..SessionCapabilities::default()
                },
                ..AgentCapabilities::default()
            },
            agent_info: Some(self.info.clone()),
            auth_methods: vec![],
            meta: None,
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

        let agent = match (self.make_agent)(params.cwd.clone()).await {
            Ok(agent) => agent,
            Err(err) => {
                return JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::internal_error(format!("could not start a session: {err}")),
                );
            }
        };

        let session = AcpSession::new(params.cwd, params.mcp_servers, agent);
        let session_id = session.id().to_string();
        self.cancellations
            .insert(session_id.clone(), session.cancellation());

        self.sessions.insert(session_id.clone(), session);

        JsonRpcResponse::success(
            req.id,
            SessionNewResult {
                session_id,
                ..SessionNewResult::default()
            },
        )
    }

    /// Handle session/prompt request.
    ///
    /// While the turn runs, updates stream to the client as they are
    /// produced and inbound messages are still read, so `session/cancel`
    /// takes effect mid-turn. Other requests received mid-turn are deferred
    /// and answered after the turn, preserving message order.
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

        // The session leaves the map for the turn's duration so `self`
        // stays free for transport I/O and notification handling; it goes
        // back once the turn resolves.
        let Some(mut session) = self.sessions.remove(&params.session_id) else {
            return JsonRpcResponse::error(
                req.id,
                JsonRpcError::invalid_params(format!("Session not found: {}", params.session_id)),
            );
        };

        let prompt_text = prompt_text(&params.prompt);

        let (update_tx, update_rx) = async_channel::unbounded::<SessionUpdate>();
        let mut deferred = Vec::new();

        let outcome = {
            let mut turn = std::pin::pin!(session.prompt(&prompt_text, move |update| {
                // The receiver lives across the loop; the only failure is an
                // unreachable disconnect.
                drop(update_tx.try_send(update));
            }));
            loop {
                let progress = or(
                    or(async { Progress::Turn(turn.as_mut().await) }, async {
                        Progress::Message(self.transport.recv().await)
                    }),
                    async {
                        // The sender lives inside `turn`, which has already
                        // resolved by the time `recv` fails.
                        update_rx
                            .recv()
                            .await
                            .map_or(Progress::Turn(Ok(())), Progress::Update)
                    },
                )
                .await;
                match progress {
                    Progress::Turn(outcome) => break outcome,
                    Progress::Update(update) => {
                        if let Err(error) = self.send_update(&params.session_id, update).await {
                            debug!("failed to send session update: {error}");
                        }
                    }
                    Progress::Message(Ok(Some(JsonRpcMessage::Notification(notif)))) => {
                        self.handle_notification(&notif);
                    }
                    Progress::Message(Ok(Some(msg))) => deferred.push(msg),
                    Progress::Message(Ok(None)) => break Err("connection closed".to_string()),
                    Progress::Message(Err(error)) => {
                        break Err(format!("transport error: {error}"));
                    }
                }
            }
        };

        self.sessions.insert(params.session_id.clone(), session);

        // Flush updates produced before the final select iteration.
        while let Ok(update) = update_rx.try_recv() {
            if let Err(error) = self.send_update(&params.session_id, update).await {
                debug!("failed to send session update: {error}");
            }
        }

        // Answer requests and notifications that arrived mid-turn, in order.
        // `handle_message` recurses back through `session/prompt`, so the
        // call is boxed to keep the future's size finite.
        for msg in deferred {
            if let Err(error) = Box::pin(self.handle_message(msg)).await {
                debug!("error handling deferred message: {error}");
            }
        }

        let stop_reason = match outcome {
            Ok(())
                if self
                    .sessions
                    .get(&params.session_id)
                    .is_some_and(AcpSession::is_cancelled) =>
            {
                StopReason::Cancelled
            }
            Ok(()) => StopReason::EndTurn,
            Err(e) => {
                debug!("Agent error: {e}");
                StopReason::Error
            }
        };

        JsonRpcResponse::success(
            req.id,
            PromptResult {
                stop_reason,
                meta: None,
            },
        )
    }

    /// Handle session/list request.
    fn handle_session_list(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let sessions = self
            .sessions
            .values()
            .map(|session| SessionInfo {
                session_id: session.id().to_string(),
                cwd: session.cwd().clone(),
                additional_directories: Vec::new(),
                title: None,
                updated_at: None,
                meta: None,
                extra: BTreeMap::new(),
            })
            .collect();
        JsonRpcResponse::success(
            req.id,
            SessionListResult {
                sessions,
                ..SessionListResult::default()
            },
        )
    }

    /// Handle session/close request: drops the session without deleting any
    /// history (this server keeps none).
    fn handle_session_close(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        match parse_session_params::<SessionCloseParams>(&req) {
            Ok(session_id) if self.end_session(&session_id) => {
                JsonRpcResponse::success(req.id, SessionCloseResult::default())
            }
            Ok(session_id) => session_not_found(req.id, &session_id),
            Err(response) => *response,
        }
    }

    /// Handle session/delete request.
    fn handle_session_delete(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        match parse_session_params::<SessionDeleteParams>(&req) {
            Ok(session_id) if self.end_session(&session_id) => {
                JsonRpcResponse::success(req.id, SessionDeleteResult::default())
            }
            Ok(session_id) => session_not_found(req.id, &session_id),
            Err(response) => *response,
        }
    }

    /// Remove a session and its cancellation flag. Returns `false` when the
    /// session does not exist.
    fn end_session(&mut self, session_id: &str) -> bool {
        self.cancellations.remove(session_id);
        self.sessions.remove(session_id).is_some()
    }

    /// Send a session update notification.
    ///
    /// # Errors
    ///
    /// Returns an error if the notification cannot be written to the client.
    pub async fn send_update(&mut self, session_id: &str, update: SessionUpdate) -> Result<()> {
        let notif = JsonRpcNotification::with_params(
            "session/update",
            SessionNotification {
                session_id: session_id.to_string(),
                update,
                meta: None,
                extra: BTreeMap::new(),
            },
        );
        self.notify(notif).await
    }
}

/// A params type that identifies a session.
trait SessionScoped: DeserializeOwned {
    /// The `sessionId` field.
    fn session_id(&self) -> &str;
}

impl SessionScoped for SessionCloseParams {
    fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl SessionScoped for SessionDeleteParams {
    fn session_id(&self) -> &str {
        &self.session_id
    }
}

/// The concatenated text of a prompt's text blocks.
fn prompt_text(prompt: &[ContentBlock]) -> String {
    prompt
        .iter()
        .filter_map(|block| {
            if let ContentBlock::Text(text) = block {
                Some(text.text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse a session-scoped request's params into its `session_id`,
/// producing the error response on failure.
///
/// The error variant is boxed: a `JsonRpcResponse` is far larger than the
/// `Ok` payload it replaces.
fn parse_session_params<P: SessionScoped>(
    req: &JsonRpcRequest,
) -> std::result::Result<String, Box<JsonRpcResponse>> {
    match req
        .params
        .clone()
        .map(serde_json::from_value::<P>)
        .transpose()
    {
        Ok(Some(params)) => Ok(params.session_id().to_string()),
        Ok(None) => Err(Box::new(JsonRpcResponse::error(
            req.id.clone(),
            JsonRpcError::invalid_params("Missing params"),
        ))),
        Err(e) => Err(Box::new(JsonRpcResponse::error(
            req.id.clone(),
            JsonRpcError::invalid_params(e.to_string()),
        ))),
    }
}

/// The error response for a request naming a session that does not exist.
fn session_not_found(id: RequestId, session_id: &str) -> JsonRpcResponse {
    JsonRpcResponse::error(
        id,
        JsonRpcError::invalid_params(format!("Session not found: {session_id}")),
    )
}
