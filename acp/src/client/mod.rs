//! ACP client.
//!
//! [`AcpClient`] talks to an ACP agent over any
//! [`BidirectionalTransport`], or over stdio to a spawned child process.
//! The transport is owned by a connection task returned alongside the client;
//! it is the only reader and the only writer, so a `session/prompt` request
//! can stay outstanding while `session/update` notifications and
//! agent-to-client requests are routed through the same connection.

mod error;
mod handler;

pub use error::ClientError;
pub use handler::ClientHandler;

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use aither_mcp::protocol::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError,
    RequestId,
};
use aither_mcp::transport::{BidirectionalTransport, ChildProcessTransport};
use async_channel::{Receiver, Sender};
use futures_lite::future::or;
use futures_util::stream::{FuturesUnordered, StreamExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::{debug, warn};

use crate::protocol::{
    ConfigOption, ContentBlock, Implementation, InitializeParams, InitializeResult, McpServerSpec,
    PROTOCOL_VERSION, PromptParams, PromptResult, SessionCancelParams, SessionConfigValue,
    SessionLoadParams, SessionLoadResult, SessionNewParams, SessionNewResult, SessionNotification,
    SessionSetConfigOptionParams, SessionSetConfigOptionResult, SessionSetModeParams,
    SessionSetModeResult,
};

/// An outbound message plus, for requests, the channel that receives the
/// matching response.
enum Outbound {
    /// A client-to-agent request; `reply` receives its response.
    Request {
        /// The request to write to the transport.
        request: JsonRpcRequest,
        /// Channel that receives the matching response, or the terminal error.
        reply: Sender<Result<JsonRpcResponse, ClientError>>,
    },
    /// A client-to-agent notification.
    Notification(JsonRpcNotification),
}

/// ACP client.
///
/// Construct it with [`connect`](Self::connect) over an existing
/// [`BidirectionalTransport`] or with [`spawn`](Self::spawn) to run an agent
/// as a child process over stdio. Both return the client plus a connection
/// future that the caller must drive — for example `tokio::spawn(driver)` —
/// because the transport is polled only while that future is being polled.
///
/// Methods are safe to call concurrently and from any clone of the client;
/// request IDs and response routing are managed by the connection task.
pub struct AcpClient<H: ClientHandler> {
    /// Requests and notifications drained by the connection task.
    outbound: Sender<Outbound>,
    /// Request ID counter shared by all clones of this client.
    next_id: Arc<AtomicI64>,
    /// Handler for agent-to-client traffic; also consulted for the client
    /// capabilities sent during `initialize`.
    handler: Arc<H>,
    /// Identity reported to the agent in `initialize`.
    info: Implementation,
}

impl<H: ClientHandler> std::fmt::Debug for AcpClient<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcpClient").finish_non_exhaustive()
    }
}

impl<H: ClientHandler> Clone for AcpClient<H> {
    fn clone(&self) -> Self {
        Self {
            outbound: self.outbound.clone(),
            next_id: self.next_id.clone(),
            handler: self.handler.clone(),
            info: self.info.clone(),
        }
    }
}

impl<H: ClientHandler> AcpClient<H> {
    /// Connect to an agent over an existing bidirectional transport.
    ///
    /// Returns the client and the connection future that owns the transport.
    /// The future routes responses to pending requests, dispatches
    /// agent-to-client requests to `handler`, and forwards notifications to
    /// [`ClientHandler::session_update`]. It resolves when the connection
    /// closes; drive it on the caller's executor, e.g. `tokio::spawn`.
    ///
    /// This is the only workable shape here because `aither-mcp` transports
    /// are `&mut`-owned with no internal task: nothing can read or write the
    /// transport while no one polls it, and no runtime is assumed.
    pub fn connect<T>(transport: T, handler: H) -> (Self, impl Future<Output = ()> + Send)
    where
        T: BidirectionalTransport,
    {
        let handler = Arc::new(handler);
        let (outbound_tx, outbound_rx) = async_channel::unbounded();
        let client = Self {
            outbound: outbound_tx,
            next_id: Arc::new(AtomicI64::new(1)),
            handler: handler.clone(),
            info: Implementation {
                name: "aither-acp".to_string(),
                title: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };
        let connection = drive(transport, outbound_rx, handler);
        (client, connection)
    }

    /// Spawn an agent child process and connect to it over stdio.
    ///
    /// The child's stdin/stdout carry the JSON-RPC stream and stderr is
    /// inherited. The returned future owns the process and the transport;
    /// when the child exits, every pending request fails with
    /// [`ClientError::Closed`] carrying the exit status.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] if the process cannot be spawned or
    /// its pipes cannot be captured.
    pub fn spawn<E, C>(
        program: &str,
        args: &[&str],
        env: E,
        cwd: C,
        handler: H,
    ) -> Result<(Self, impl Future<Output = ()> + Send + use<H, E, C>), ClientError>
    where
        E: IntoIterator<Item = (String, String)>,
        C: AsRef<Path>,
    {
        let mut command = async_process::Command::new(program);
        command.args(args).envs(env).current_dir(cwd);
        let transport = ChildProcessTransport::from_command(&mut command)
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        Ok(Self::connect(transport, handler))
    }

    /// Set the identity reported to the agent during [`initialize`](Self::initialize).
    ///
    /// The default is `aither-acp` with this crate's version.
    #[must_use]
    pub fn with_client_info(mut self, info: Implementation) -> Self {
        self.info = info;
        self
    }

    /// The handler this client dispatches agent requests to.
    #[must_use]
    pub fn handler(&self) -> &H {
        &self.handler
    }

    /// `initialize` request.
    ///
    /// Advertises `protocol_version` [`PROTOCOL_VERSION`] and the
    /// capabilities returned by [`ClientHandler::capabilities`].
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn initialize(&self) -> Result<InitializeResult, ClientError> {
        self.call(
            "initialize",
            &InitializeParams {
                protocol_version: PROTOCOL_VERSION,
                client_capabilities: self.handler.capabilities(),
                client_info: Some(self.info.clone()),
            },
        )
        .await
    }

    /// `session/new` request.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn new_session(
        &self,
        cwd: impl Into<PathBuf>,
        mcp_servers: Vec<McpServerSpec>,
    ) -> Result<SessionNewResult, ClientError> {
        self.call(
            "session/new",
            &SessionNewParams {
                cwd: cwd.into(),
                mcp_servers,
                additional_directories: Vec::new(),
                meta: None,
            },
        )
        .await
    }

    /// `session/load` request.
    ///
    /// Only valid when the agent advertised
    /// [`AgentCapabilities::load_session`](crate::protocol::AgentCapabilities::load_session).
    /// While it runs the agent replays the session's history as
    /// `session/update` notifications.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn load_session(
        &self,
        session_id: &str,
        cwd: impl Into<PathBuf>,
        mcp_servers: Vec<McpServerSpec>,
    ) -> Result<SessionLoadResult, ClientError> {
        self.call(
            "session/load",
            &SessionLoadParams {
                session_id: session_id.to_string(),
                cwd: cwd.into(),
                mcp_servers,
                additional_directories: Vec::new(),
                meta: None,
            },
        )
        .await
    }

    /// `session/set_mode` request.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn set_mode(&self, session_id: &str, mode_id: &str) -> Result<(), ClientError> {
        self.call::<_, SessionSetModeResult>(
            "session/set_mode",
            &SessionSetModeParams {
                session_id: session_id.to_string(),
                mode_id: mode_id.to_string(),
                meta: None,
            },
        )
        .await?;
        Ok(())
    }

    /// `session/set_config_option` request.
    ///
    /// `value` accepts `&str`/`String` for `select` options and `bool` for
    /// `boolean` options. Returns the full updated option list.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn set_config_option(
        &self,
        session_id: &str,
        config_id: &str,
        value: impl Into<SessionConfigValue>,
    ) -> Result<Vec<ConfigOption>, ClientError> {
        let result: SessionSetConfigOptionResult = self
            .call(
                "session/set_config_option",
                &SessionSetConfigOptionParams {
                    session_id: session_id.to_string(),
                    config_id: config_id.to_string(),
                    value: value.into(),
                    meta: None,
                },
            )
            .await?;
        Ok(result.config_options)
    }

    /// `session/prompt` request.
    ///
    /// The request stays outstanding for the whole turn while
    /// `session/update` notifications keep flowing to the handler. Resolves
    /// with the turn's [`PromptResult`] when the agent responds.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn prompt(
        &self,
        session_id: &str,
        prompt: Vec<ContentBlock>,
    ) -> Result<PromptResult, ClientError> {
        self.call(
            "session/prompt",
            &PromptParams {
                session_id: session_id.to_string(),
                prompt,
                meta: None,
            },
        )
        .await
    }

    /// `session/cancel` notification.
    ///
    /// Asks the agent to cancel the current prompt turn on `session_id`.
    /// Per the protocol the agent must respond `cancelled` to the pending
    /// `session/prompt` and to any pending permission requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed.
    pub async fn cancel(&self, session_id: &str) -> Result<(), ClientError> {
        let notification = JsonRpcNotification::with_params(
            "session/cancel",
            SessionCancelParams {
                session_id: session_id.to_string(),
                meta: None,
            },
        );
        self.outbound
            .send(Outbound::Notification(notification))
            .await
            .map_err(|_| ClientError::Closed { status: None })
    }

    /// Close the connection.
    ///
    /// The channel to the connection task is closed; the task then collects
    /// the peer's exit status (killing a child that is still running), fails
    /// every pending request, and returns.
    /// Await the connection future returned by [`connect`](Self::connect) or
    /// [`spawn`](Self::spawn) to observe shutdown.
    pub fn close(self) {
        self.outbound.close();
    }

    /// Send a request through the connection task and await the response.
    async fn call<P, R>(&self, method: &str, params: &P) -> Result<R, ClientError>
    where
        P: Serialize + Sync,
        R: DeserializeOwned,
    {
        let id = RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst));
        let request = JsonRpcRequest::with_params(id, method, params);
        let (reply_tx, reply_rx) = async_channel::bounded(1);

        self.outbound
            .send(Outbound::Request {
                request,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ClientError::Closed { status: None })?;

        let response = reply_rx
            .recv()
            .await
            .map_err(|_| ClientError::Closed { status: None })??;

        let value = response.into_result()?;
        serde_json::from_value(value)
            .map_err(|error| ClientError::Protocol(format!("malformed {method} result: {error}")))
    }
}

/// The connection task: sole owner of the transport and router of every
/// message in both directions.
///
/// Runs until the transport reports EOF or an error, or until every
/// [`AcpClient`] handle is gone or calls `close`. On termination it collects
/// the peer's exit status and fails every pending request.
///
/// Agent-to-client requests (`session/request_permission`, `fs/*`,
/// `terminal/*`) are not answered inline: they are long-lived — a permission
/// prompt waits on a human, `terminal/wait_for_exit` waits on a process — so
/// each is pushed into `in_flight` and polled alongside the other sources,
/// and its response is written when the handler future completes.
async fn drive<T, H>(mut transport: T, outbound: Receiver<Outbound>, handler: Arc<H>)
where
    T: BidirectionalTransport,
    H: ClientHandler,
{
    /// One step of the connection loop.
    enum Event {
        /// Something the agent sent, or the end of the transport.
        Incoming(Result<Option<JsonRpcMessage>, McpError>),
        /// Something the client sent, or all client handles closing.
        Outbound(Result<Outbound, async_channel::RecvError>),
        /// A handler future finished and produced the response to write back.
        Handled(JsonRpcResponse),
    }

    let mut pending: HashMap<RequestId, Sender<Result<JsonRpcResponse, ClientError>>> =
        HashMap::new();
    // Agent requests currently being answered by the handler.
    let mut in_flight: FuturesUnordered<Pin<Box<dyn Future<Output = JsonRpcResponse> + Send>>> =
        FuturesUnordered::new();

    let terminal = loop {
        let event = or(
            or(async { Event::Outbound(outbound.recv().await) }, async {
                Event::Incoming(transport.recv().await)
            }),
            async {
                match in_flight.next().await {
                    Some(response) => Event::Handled(response),
                    // An empty `FuturesUnordered` resolves to `None`
                    // immediately; park instead of spinning the loop.
                    None => std::future::pending::<Event>().await,
                }
            },
        )
        .await;

        match event {
            Event::Outbound(Ok(Outbound::Request { request, reply })) => {
                pending.insert(request.id.clone(), reply);
                if let Err(error) = transport.send_request(request).await {
                    break ClientError::Transport(error.to_string());
                }
            }
            Event::Outbound(Ok(Outbound::Notification(notification))) => {
                if let Err(error) = transport.notify(notification).await {
                    break ClientError::Transport(error.to_string());
                }
            }
            Event::Outbound(Err(_)) | Event::Incoming(Ok(None)) => {
                break ClientError::Closed { status: None };
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Response(response)))) => {
                if let Some(reply) = pending.remove(&response.id) {
                    let _ = reply.try_send(Ok(response));
                } else {
                    debug!(id = ?response.id, "ignoring response with no pending request");
                }
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Request(request)))) => {
                in_flight.push(Box::pin(handle_agent_request(handler.clone(), request)));
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Notification(notification)))) => {
                handle_agent_notification(handler.as_ref(), notification).await;
            }
            Event::Handled(response) => {
                if let Err(error) = transport.respond(response).await {
                    break ClientError::Transport(error.to_string());
                }
            }
            Event::Incoming(Err(error)) => break ClientError::Transport(error.to_string()),
        }
    };

    // Shutdown: drop the in-flight handler futures, collect the peer's exit
    // status (reaping a child that is still running), close the transport,
    // and fail everything still in flight so no caller hangs.
    drop(in_flight);
    let status = transport.exit_status().await;
    let _ = transport.close().await;

    let terminal = match terminal {
        ClientError::Closed { .. } => ClientError::Closed { status },
        other => other,
    };
    debug!(error = %terminal, "ACP connection closed");

    while let Ok(outbound) = outbound.try_recv() {
        if let Outbound::Request { reply, .. } = outbound {
            let _ = reply.try_send(Err(terminal.clone()));
        }
    }
    for (_, reply) in pending {
        let _ = reply.try_send(Err(terminal.clone()));
    }
}

/// Dispatch an agent-to-client request to the handler and produce the
/// JSON-RPC response to send back.
///
/// Takes the handler by `Arc` so the returned future is `'static` and can be
/// polled inside the connection task's in-flight set while the loop keeps
/// reading and writing other messages.
async fn handle_agent_request<H: ClientHandler>(
    handler: Arc<H>,
    request: JsonRpcRequest,
) -> JsonRpcResponse {
    match request.method.as_str() {
        "session/request_permission" => {
            dispatch(&request, |params| handler.request_permission(params)).await
        }
        "fs/read_text_file" => dispatch(&request, |params| handler.read_text_file(params)).await,
        "fs/write_text_file" => dispatch(&request, |params| handler.write_text_file(params)).await,
        "terminal/create" => dispatch(&request, |params| handler.terminal_create(params)).await,
        "terminal/output" => dispatch(&request, |params| handler.terminal_output(params)).await,
        "terminal/wait_for_exit" => {
            dispatch(&request, |params| handler.terminal_wait_for_exit(params)).await
        }
        "terminal/kill" => dispatch(&request, |params| handler.terminal_kill(params)).await,
        "terminal/release" => dispatch(&request, |params| handler.terminal_release(params)).await,
        method => JsonRpcResponse::error(request.id, JsonRpcError::method_not_found(method)),
    }
}

/// Parse a request's params, call `f`, and wrap the result as a JSON-RPC
/// response carrying `request.id`.
async fn dispatch<P, R, F, Fut>(request: &JsonRpcRequest, f: F) -> JsonRpcResponse
where
    P: DeserializeOwned,
    R: Serialize,
    F: FnOnce(P) -> Fut,
    Fut: Future<Output = Result<R, JsonRpcError>>,
{
    let Some(params) = request.params.clone() else {
        return JsonRpcResponse::error(
            request.id.clone(),
            JsonRpcError::invalid_params("missing params"),
        );
    };
    let params: P = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError::invalid_params(error.to_string()),
            );
        }
    };
    match f(params).await {
        Ok(result) => JsonRpcResponse::success(request.id.clone(), result),
        Err(error) => JsonRpcResponse::error(request.id.clone(), error),
    }
}

/// Dispatch an agent-to-client notification to the handler.
async fn handle_agent_notification<H: ClientHandler>(
    handler: &H,
    notification: JsonRpcNotification,
) {
    if notification.method == "session/update" {
        match notification.params.map(serde_json::from_value) {
            Some(Ok(update)) => {
                let update: SessionNotification = update;
                handler.session_update(update).await;
            }
            Some(Err(error)) => warn!(%error, "malformed session/update notification"),
            None => warn!("session/update notification without params"),
        }
    } else {
        debug!(method = %notification.method, "ignoring unknown agent notification");
    }
}
