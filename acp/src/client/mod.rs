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
use std::marker::PhantomData;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::task::{Context, Poll, ready};

use aither_mcp::protocol::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError,
    RequestId,
};
use aither_mcp::transport::{BidirectionalTransport, ChildProcessTransport};
use async_channel::{Receiver, Sender};
use futures_lite::future::or;
use futures_lite::stream::Stream;
use futures_util::stream::{FuturesUnordered, StreamExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::{debug, warn};

use crate::protocol::{
    AuthenticateParams, AuthenticateResult, ConfigOption, ElicitationCompleteParams, ExtMethod,
    Implementation, InitializeParams, InitializeResult, LogoutParams, LogoutResult,
    PROTOCOL_VERSION, PromptParams, PromptResult, SessionCancelParams, SessionCloseParams,
    SessionCloseResult, SessionDeleteParams, SessionDeleteResult, SessionListParams,
    SessionListResult, SessionLoadParams, SessionLoadResult, SessionNewParams, SessionNewResult,
    SessionNotification, SessionResumeParams, SessionResumeResult, SessionSetConfigOptionParams,
    SessionSetConfigOptionResult, SessionSetModeParams, SessionSetModeResult,
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
                meta: None,
            },
        )
        .await
    }

    /// `authenticate` request: runs one of the [`AuthMethod`]s the agent
    /// advertised in [`InitializeResult::auth_methods`].
    ///
    /// [`AuthMethod`]: crate::protocol::AuthMethod
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn authenticate(
        &self,
        params: impl Into<AuthenticateParams>,
    ) -> Result<AuthenticateResult, ClientError> {
        self.call("authenticate", &params.into()).await
    }

    /// `logout` request.
    ///
    /// Only valid when the agent advertised
    /// [`AgentCapabilities::auth.logout`](crate::protocol::AgentAuthCapabilities::logout).
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn logout(&self, params: LogoutParams) -> Result<LogoutResult, ClientError> {
        self.call("logout", &params).await
    }

    /// `session/new` request.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn new_session(
        &self,
        params: SessionNewParams,
    ) -> Result<SessionNewResult, ClientError> {
        self.call("session/new", &params).await
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
        params: SessionLoadParams,
    ) -> Result<SessionLoadResult, ClientError> {
        self.call("session/load", &params).await
    }

    /// `session/resume` request.
    ///
    /// Only valid when the agent advertised
    /// [`SessionCapabilities::resume`](crate::protocol::SessionCapabilities::resume).
    /// Restores the session's context inside the agent without replaying its
    /// history as `session/update` notifications.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn resume_session(
        &self,
        params: SessionResumeParams,
    ) -> Result<SessionResumeResult, ClientError> {
        self.call("session/resume", &params).await
    }

    /// `session/list` request.
    ///
    /// Only valid when the agent advertised
    /// [`SessionCapabilities::list`](crate::protocol::SessionCapabilities::list).
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> Result<SessionListResult, ClientError> {
        self.call("session/list", &params).await
    }

    /// `session/delete` request: removes the session from the agent's
    /// history.
    ///
    /// Only valid when the agent advertised
    /// [`SessionCapabilities::delete`](crate::protocol::SessionCapabilities::delete).
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn delete_session(
        &self,
        params: impl Into<SessionDeleteParams>,
    ) -> Result<SessionDeleteResult, ClientError> {
        self.call("session/delete", &params.into()).await
    }

    /// `session/close` request: ends a live session without deleting its
    /// history.
    ///
    /// Only valid when the agent advertised
    /// [`SessionCapabilities::close`](crate::protocol::SessionCapabilities::close).
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn close_session(
        &self,
        params: impl Into<SessionCloseParams>,
    ) -> Result<SessionCloseResult, ClientError> {
        self.call("session/close", &params.into()).await
    }

    /// `session/set_mode` request.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn set_mode(&self, params: SessionSetModeParams) -> Result<(), ClientError> {
        self.call::<_, SessionSetModeResult>("session/set_mode", &params)
            .await?;
        Ok(())
    }

    /// `session/set_config_option` request. Returns the full updated option
    /// list.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn set_config_option(
        &self,
        params: SessionSetConfigOptionParams,
    ) -> Result<Vec<ConfigOption>, ClientError> {
        let result: SessionSetConfigOptionResult =
            self.call("session/set_config_option", &params).await?;
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
    pub async fn prompt(&self, params: PromptParams) -> Result<PromptResult, ClientError> {
        self.start_prompt(&params).await?.await
    }

    /// `session/prompt` request whose response half is returned separately.
    ///
    /// Unlike [`prompt`](Self::prompt), this resolves as soon as the request
    /// is on the connection, so a later write (such as `session/cancel`)
    /// cannot overtake it on the wire. Awaiting the [`ResponseFuture`]
    /// yields the turn's [`PromptResult`].
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed before the request is
    /// sent.
    pub async fn start_prompt(
        &self,
        params: &PromptParams,
    ) -> Result<ResponseFuture<PromptResult>, ClientError> {
        self.start_request("session/prompt", params).await
    }

    /// `session/cancel` notification.
    ///
    /// Asks the agent to cancel the current prompt turn on
    /// `params.session_id`. Per the protocol the agent must respond
    /// `cancelled` to the pending `session/prompt` and to any pending
    /// permission requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed.
    pub async fn cancel(&self, params: impl Into<SessionCancelParams>) -> Result<(), ClientError> {
        let notification = JsonRpcNotification::with_params("session/cancel", params.into());
        self.outbound
            .send(Outbound::Notification(notification))
            .await
            .map_err(|_| ClientError::Closed { status: None })
    }

    /// Send a `_`-prefixed extension request and deserialize its result.
    ///
    /// This is the channel for provider-neutral extensions modelled under
    /// [`crate::ext`] and vendor-specific methods under [`crate::vendor`].
    /// The [`ExtMethod`] type enforces the protocol rule that custom method
    /// names begin with `_`; for legacy vendor methods that predate that
    /// rule, use [`request`](Self::request).
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn ext_request<P, R>(&self, method: &ExtMethod, params: &P) -> Result<R, ClientError>
    where
        P: Serialize + Sync,
        R: DeserializeOwned,
    {
        self.call(method.as_str(), params).await
    }

    /// Send a `_`-prefixed extension notification.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed.
    pub async fn ext_notify<P>(&self, method: &ExtMethod, params: &P) -> Result<(), ClientError>
    where
        P: Serialize + Sync,
    {
        let notification = JsonRpcNotification::with_params(method.as_str(), params);
        self.outbound
            .send(Outbound::Notification(notification))
            .await
            .map_err(|_| ClientError::Closed { status: None })
    }

    /// Send a JSON-RPC request for an arbitrary method and deserialize its
    /// result.
    ///
    /// Prefer the typed methods and [`ext_request`](Self::ext_request); this
    /// exists for vendor methods that predate the `_` naming rule (see
    /// [`crate::vendor::codex`]) and for methods this crate does not model.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn request<P, R>(&self, method: &str, params: &P) -> Result<R, ClientError>
    where
        P: Serialize + Sync,
        R: DeserializeOwned,
    {
        self.call(method, params).await
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

    /// Send a JSON-RPC request for an arbitrary method and return a future
    /// resolving with its deserialized result.
    ///
    /// The request is on the connection when this returns, so messages sent
    /// afterwards cannot overtake it on the wire. Awaiting the
    /// [`ResponseFuture`] yields the same result [`request`](Self::request)
    /// would.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed before the request is
    /// sent.
    pub async fn start_request<P, R>(
        &self,
        method: &str,
        params: &P,
    ) -> Result<ResponseFuture<R>, ClientError>
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

        Ok(ResponseFuture {
            method: method.to_string(),
            reply: Box::pin(reply_rx),
            marker: PhantomData,
        })
    }

    /// Send a request through the connection task and await the response.
    async fn call<P, R>(&self, method: &str, params: &P) -> Result<R, ClientError>
    where
        P: Serialize + Sync,
        R: DeserializeOwned,
    {
        self.start_request(method, params).await?.await
    }
}

/// The response half of a request sent with [`AcpClient::start_request`].
///
/// Awaiting it yields the deserialized result, an agent-reported error, or
/// [`ClientError::Closed`] when the connection drops first. Dropping it
/// leaves the request outstanding — the agent still sees and answers it.
#[derive(Debug)]
pub struct ResponseFuture<R> {
    /// Method name, for error context.
    method: String,
    /// Receives the matching response from the connection task.
    reply: Pin<Box<Receiver<Result<JsonRpcResponse, ClientError>>>>,
    /// Result type without owning a value.
    marker: PhantomData<fn() -> R>,
}

impl<R: DeserializeOwned> Future for ResponseFuture<R> {
    type Output = Result<R, ClientError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let response = match ready!(this.reply.as_mut().poll_next(cx)) {
            Some(response) => response?,
            None => return Poll::Ready(Err(ClientError::Closed { status: None })),
        };
        Poll::Ready(
            response
                .into_result()
                .map_err(ClientError::from)
                .and_then(|value| {
                    serde_json::from_value(value).map_err(|error| {
                        ClientError::Protocol(format!("malformed {} result: {error}", this.method))
                    })
                }),
        )
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
        "elicitation/create" => {
            dispatch(&request, |params| handler.elicitation_create(params)).await
        }
        method if method.starts_with('_') => {
            let method = ExtMethod::try_new(request.method.clone())
                .expect("extension method name starts with `_`");
            match handler.ext_request(method, request.params.clone()).await {
                Ok(result) => JsonRpcResponse::success(request.id, result),
                Err(error) => JsonRpcResponse::error(request.id, error),
            }
        }
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
    match notification.method.as_str() {
        "session/update" => match notification.params.clone().map(serde_json::from_value) {
            Some(Ok(update)) => {
                let update: SessionNotification = update;
                handler.session_update(update).await;
            }
            Some(Err(error)) => warn!(%error, "malformed session/update notification"),
            None => warn!("session/update notification without params"),
        },
        "elicitation/complete" => match notification.params.clone().map(serde_json::from_value) {
            Some(Ok(params)) => {
                let params: ElicitationCompleteParams = params;
                handler.elicitation_complete(params).await;
            }
            Some(Err(error)) => warn!(%error, "malformed elicitation/complete notification"),
            None => warn!("elicitation/complete notification without params"),
        },
        _ => handler.notification(notification).await,
    }
}
