//! HTTP transport for MCP (Streamable HTTP).
//!
//! This transport uses HTTP POST for sending requests to the server.
//! A response is read by its `Content-Type`: `application/json` is a single
//! JSON-RPC response, `text/event-stream` is an SSE stream read until the
//! response to the request. The transport follows MCP session management via
//! the `Mcp-Session-Id` header and sends the negotiated `MCP-Protocol-Version`
//! on every request after `initialize`.

use std::collections::VecDeque;
use std::fmt::Display;
use std::future::{self, Future};
use std::pin::Pin;
use std::sync::atomic::{AtomicI64, Ordering};
use std::task::Poll;

use futures_lite::{Stream, StreamExt};
use futures_util::stream::FuturesUnordered;
use serde_json::Value;
use tracing::{debug, warn};
use zenwave::sse::SseStream;
use zenwave::{Client, ResponseExt, client, header};

use super::traits::{BidirectionalTransport, Result, Transport};
use crate::protocol::{
    InitializeResult, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    McpError, RequestId,
};

/// MCP session ID header name.
const MCP_SESSION_ID_HEADER: &str = "Mcp-Session-Id";

/// MCP protocol version header name.
const MCP_PROTOCOL_VERSION_HEADER: &str = "MCP-Protocol-Version";

/// HTTP transport for MCP communication (Streamable HTTP).
///
/// This transport sends requests via HTTP POST and handles session management
/// via the `Mcp-Session-Id` header as per MCP specification 2025-06-18.
///
/// It is also a [`BidirectionalTransport`]: the SSE stream that answers a POST
/// can carry server requests and notifications besides the response.
/// [`send_request`] leaves the whole stream to [`recv`], which hands every
/// message out, and [`respond`] POSTs the reply to a server request back, as
/// the specification requires. [`Transport::request`] reads its stream to the
/// matching response and discards everything else, as its contract says.
///
/// [`recv`]: BidirectionalTransport::recv
/// [`respond`]: BidirectionalTransport::respond
/// [`send_request`]: BidirectionalTransport::send_request
pub struct HttpTransport {
    /// Base URL of the MCP server.
    base_url: String,
    /// Optional authorization header value.
    auth: Option<String>,
    /// Session ID returned by server during initialization.
    session_id: Option<String>,
    /// Protocol version negotiated by `initialize`, sent on every later request.
    protocol_version: Option<String>,
    /// ID of an `initialize` sent by `send_request` whose response is pending.
    initialize_id: Option<RequestId>,
    /// Messages read off a response but not yet handed to `recv`.
    inbound: VecDeque<JsonRpcMessage>,
    /// SSE responses still open, polled by `recv`.
    streams: Vec<Pin<Box<SseStream>>>,
    /// Requests handed to `send_request` whose reply has not arrived yet,
    /// driven by `recv` so that waiting on one never blocks the others or the
    /// server's own messages.
    sending: FuturesUnordered<Pin<Box<dyn Future<Output = Result<Sent>> + Send>>>,
    /// Next request ID.
    next_id: AtomicI64,
    /// Whether the transport is closed.
    closed: bool,
}

impl std::fmt::Debug for HttpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpTransport")
            .field("base_url", &self.base_url)
            .field("session_id", &self.session_id)
            .field("protocol_version", &self.protocol_version)
            .field("open_streams", &self.streams.len())
            .field("requests_in_flight", &self.sending.len())
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

/// What a POST was answered with.
enum Reply {
    /// One JSON-RPC response as `application/json`.
    Json(JsonRpcResponse),
    /// An SSE stream of JSON-RPC messages.
    Events(Pin<Box<SseStream>>),
    /// `202 Accepted` with no body.
    Accepted,
}

/// What a POST needs of the transport's state, copied out so the POST can be
/// in flight while the transport itself is used.
struct Endpoint {
    url: String,
    auth: Option<String>,
    session_id: Option<String>,
    protocol_version: Option<String>,
}

/// What `recv` was woken by.
enum Arrived<E> {
    /// A queued request's reply.
    Sent(Result<Sent>),
    /// An event on an open SSE stream.
    Event(E),
    /// Nothing is in flight any more.
    Idle,
}

/// A POST's reply and the session ID it handed out.
struct Sent {
    session_id: Option<String>,
    reply: Reply,
}

/// Read the next JSON-RPC message off an SSE stream; `None` once it ends.
async fn next_message(events: &mut Pin<Box<SseStream>>) -> Result<Option<JsonRpcMessage>> {
    loop {
        let Some(event) = events.next().await else {
            return Ok(None);
        };
        let event = event.map_err(|e| McpError::Transport(format!("SSE parse failed: {e}")))?;
        if let Some(message) = parse_event(&event)? {
            return Ok(Some(message));
        }
    }
}

/// Decode an SSE event; a priming event carries an ID and no data.
fn parse_event(event: &zenwave::sse::Event) -> Result<Option<JsonRpcMessage>> {
    if event.text_data().is_empty() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(event.text_data())?))
}

impl HttpTransport {
    /// Create a new HTTP transport.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the MCP server.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            auth: None,
            session_id: None,
            protocol_version: None,
            initialize_id: None,
            inbound: VecDeque::new(),
            streams: Vec::new(),
            sending: FuturesUnordered::new(),
            next_id: AtomicI64::new(1),
            closed: false,
        }
    }

    /// Set the authorization header.
    #[must_use]
    pub fn with_auth(mut self, auth: impl Into<String>) -> Self {
        self.auth = Some(auth.into());
        self
    }

    /// Generate the next request ID.
    fn next_request_id(&self) -> RequestId {
        RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// What the next POST needs of the transport's state.
    fn endpoint(&self) -> Endpoint {
        Endpoint {
            url: self.base_url.clone(),
            auth: self.auth.clone(),
            session_id: self.session_id.clone(),
            protocol_version: self.protocol_version.clone(),
        }
    }

    /// Start `req` as a POST that `recv` drives to its reply.
    fn queue_request(&mut self, req: &JsonRpcRequest) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }
        debug!("MCP HTTP TX: {:?}", req);
        if req.method == "initialize" {
            self.initialize_id = Some(req.id.clone());
        }
        self.sending.push(Box::pin(Self::send(
            self.endpoint(),
            serde_json::to_value(req)?,
        )));
        Ok(())
    }

    /// Remember the session ID a reply handed out.
    fn adopt_session(&mut self, session_id: Option<String>) {
        if session_id.is_some() {
            self.session_id = session_id;
        }
    }

    /// POST a message and classify the reply by its status and `Content-Type`.
    ///
    /// Owns everything it needs, so it can wait on the server while the
    /// transport does other work.
    async fn send(endpoint: Endpoint, body: Value) -> Result<Sent> {
        let (response, session_id) = Self::post(&endpoint, &body).await?;
        let response = response
            .error_for_status()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP request failed: {e}")))?;

        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<mime::Mime>().ok());

        let reply = match content_type.as_ref().map(mime::Mime::essence_str) {
            Some("application/json") => Reply::Json(
                response
                    .into_json::<JsonRpcResponse>()
                    .await
                    .map_err(|e| McpError::Transport(format!("Response parse failed: {e}")))?,
            ),
            Some("text/event-stream") => Reply::Events(Box::pin(response.into_sse())),
            None if response.status() == zenwave::StatusCode::ACCEPTED => Reply::Accepted,
            _ => {
                return Err(McpError::Transport(format!(
                    "unsupported response Content-Type: {content_type:?}"
                )));
            }
        };
        Ok(Sent { session_id, reply })
    }

    /// Remember the protocol version an `initialize` response negotiated.
    fn record_protocol_version(&mut self, response: &JsonRpcResponse) -> Result<()> {
        if let Some(value) = response.result.clone() {
            let init: InitializeResult = serde_json::from_value(value)?;
            self.protocol_version = Some(init.protocol_version);
        }
        Ok(())
    }

    /// POST `body` with the endpoint's headers; also returns the session ID
    /// the server handed out, if any.
    async fn post(
        endpoint: &Endpoint,
        body: &Value,
    ) -> Result<(zenwave::Response, Option<String>)> {
        fn failed(e: &impl Display) -> McpError {
            McpError::Transport(format!("HTTP request failed: {e}"))
        }

        let mut backend = client();
        let mut builder = backend
            .post(&endpoint.url)
            .map_err(|e| failed(&e))?
            .header(header::CONTENT_TYPE.as_str(), "application/json")
            .map_err(|e| failed(&e))?
            .header(
                header::ACCEPT.as_str(),
                "application/json, text/event-stream",
            )
            .map_err(|e| failed(&e))?
            .header(header::USER_AGENT.as_str(), "aither-mcp/0.1")
            .map_err(|e| failed(&e))?;

        if let Some(auth) = &endpoint.auth {
            builder = builder
                .header(header::AUTHORIZATION.as_str(), auth.clone())
                .map_err(|e| failed(&e))?;
        }

        if let Some(session_id) = &endpoint.session_id {
            builder = builder
                .header(MCP_SESSION_ID_HEADER, session_id.clone())
                .map_err(|e| failed(&e))?;
        }

        if let Some(version) = &endpoint.protocol_version {
            builder = builder
                .header(MCP_PROTOCOL_VERSION_HEADER, version.clone())
                .map_err(|e| failed(&e))?;
        }

        let response = builder
            .json_body(body)
            .map_err(|e| failed(&e))?
            .await
            .map_err(|e| failed(&e))?;

        let session_id = response
            .headers()
            .get(MCP_SESSION_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        if let Some(session_id) = &session_id {
            debug!("MCP HTTP session ID: {}", session_id);
        }

        Ok((response, session_id))
    }
}

impl Transport for HttpTransport {
    async fn request(&mut self, mut req: JsonRpcRequest) -> Result<JsonRpcResponse> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }

        let id = self.next_request_id();
        req.id = id.clone();

        debug!("MCP HTTP TX: {:?}", req);

        let sent = Self::send(self.endpoint(), serde_json::to_value(&req)?).await?;
        self.adopt_session(sent.session_id);
        let result = match sent.reply {
            Reply::Json(response) => response,
            Reply::Events(mut events) => loop {
                match next_message(&mut events).await? {
                    Some(JsonRpcMessage::Response(response)) if response.id == id => {
                        break response;
                    }
                    // `Transport::request` discards everything else; callers
                    // that want server messages use `send_request` + `recv`.
                    Some(other) => debug!("MCP HTTP RX (not the response): {:?}", other),
                    None => return Err(McpError::ConnectionClosed),
                }
            },
            Reply::Accepted => {
                return Err(McpError::Transport(
                    "request answered 202 Accepted without a response".to_owned(),
                ));
            }
        };

        debug!("MCP HTTP RX: {:?}", result);

        if req.method == "initialize" {
            self.record_protocol_version(&result)?;
        }

        Ok(result)
    }

    async fn notify(&mut self, notif: JsonRpcNotification) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }

        debug!("MCP HTTP notify: {:?}", notif);

        // A notification expects no response body (the server answers 202);
        // the request still has to be driven, and the body is discarded.
        let (_, session_id) = Self::post(&self.endpoint(), &serde_json::to_value(&notif)?).await?;
        self.adopt_session(session_id);

        Ok(())
    }

    fn close(&mut self) -> impl Future<Output = Result<()>> + Send {
        self.closed = true;
        self.streams.clear();
        self.sending.clear();
        self.inbound.clear();
        std::future::ready(Ok(()))
    }
}

impl BidirectionalTransport for HttpTransport {
    /// Next server message read off a response.
    ///
    /// A message can only arrive on the response to a POST, so with no queued
    /// message and no open stream this waits until the caller drops it, like a
    /// quiet pipe would. Streams are polled in place, so a dropped future
    /// loses nothing.
    async fn recv(&mut self) -> Result<Option<JsonRpcMessage>> {
        loop {
            if self.closed {
                return Ok(None);
            }
            if let Some(message) = self.inbound.pop_front() {
                if let JsonRpcMessage::Response(response) = &message
                    && self.initialize_id.as_ref() == Some(&response.id)
                {
                    self.initialize_id = None;
                    self.record_protocol_version(response)?;
                }
                return Ok(Some(message));
            }
            if self.streams.is_empty() && self.sending.is_empty() {
                future::pending::<()>().await;
            }

            let streams = &mut self.streams;
            let sending = &mut self.sending;
            let arrived = future::poll_fn(|cx| {
                if let Poll::Ready(Some(sent)) = Pin::new(&mut *sending).poll_next(cx) {
                    return Poll::Ready(Arrived::Sent(sent));
                }
                let mut index = 0;
                while index < streams.len() {
                    match streams[index].as_mut().poll_next(cx) {
                        Poll::Ready(Some(event)) => return Poll::Ready(Arrived::Event(event)),
                        Poll::Ready(None) => {
                            streams.swap_remove(index);
                        }
                        Poll::Pending => index += 1,
                    }
                }
                if streams.is_empty() && sending.is_empty() {
                    Poll::Ready(Arrived::Idle)
                } else {
                    Poll::Pending
                }
            })
            .await;

            let event = match arrived {
                Arrived::Idle => continue,
                Arrived::Sent(sent) => {
                    let sent = sent?;
                    self.adopt_session(sent.session_id);
                    match sent.reply {
                        Reply::Json(response) => {
                            self.inbound.push_back(JsonRpcMessage::Response(response));
                        }
                        Reply::Events(events) => self.streams.push(events),
                        Reply::Accepted => {}
                    }
                    continue;
                }
                Arrived::Event(event) => event,
            };
            let event = event.map_err(|e| McpError::Transport(format!("SSE parse failed: {e}")))?;
            match parse_event(&event) {
                Ok(Some(message)) => self.inbound.push_back(message),
                Ok(None) => {}
                Err(McpError::Serialization(error)) => {
                    warn!(%error, "ignoring invalid JSON-RPC input");
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn respond(&mut self, response: JsonRpcResponse) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }
        debug!("MCP HTTP respond: {:?}", response);
        // The server acknowledges a response with 202 and no body.
        let (reply, session_id) =
            Self::post(&self.endpoint(), &serde_json::to_value(&response)?).await?;
        self.adopt_session(session_id);
        reply
            .error_for_status()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP request failed: {e}")))?;
        Ok(())
    }

    /// Queue the request; `recv` performs the POST and reads its reply, so
    /// the request is only on the wire while `recv` is being polled. A failed
    /// POST ends the connection through `recv`'s error.
    fn send_request(&mut self, req: JsonRpcRequest) -> impl Future<Output = Result<()>> + Send {
        std::future::ready(self.queue_request(&req))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::test_server::{serve, sse};

    #[tokio::test]
    async fn json_response() {
        let body = json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}).to_string();
        let (url, _) = serve(vec![("Content-Type: application/json\r\n", body)]).await;
        let mut transport = HttpTransport::new(url);

        let response = transport
            .request(JsonRpcRequest::new(0i64, "tools/list"))
            .await
            .unwrap();

        assert_eq!(response.id, RequestId::Number(1));
        assert_eq!(response.result, Some(json!({"ok": true})));
    }

    #[tokio::test]
    async fn sse_response_skips_interleaved_messages() {
        let body = sse(&[
            json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}),
            json!({"jsonrpc": "2.0", "id": 99, "method": "ping"}),
            json!({"jsonrpc": "2.0", "id": 7, "result": {"other": true}}),
            json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}),
        ]);
        let (url, _) = serve(vec![("Content-Type: text/event-stream\r\n", body)]).await;
        let mut transport = HttpTransport::new(url);

        let response = transport
            .request(JsonRpcRequest::new(0i64, "tools/call"))
            .await
            .unwrap();

        assert_eq!(response.id, RequestId::Number(1));
        assert_eq!(response.result, Some(json!({"ok": true})));
        // `request` discards what is not its response.
        assert!(transport.inbound.is_empty());
        assert!(transport.streams.is_empty());
    }

    #[tokio::test]
    async fn send_request_streams_server_messages_to_recv() {
        let body = sse(&[
            json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}),
            json!({"jsonrpc": "2.0", "id": 99, "method": "ping"}),
            json!({"jsonrpc": "2.0", "id": 5, "result": {"ok": true}}),
        ]);
        let (url, seen) = serve(vec![
            ("Content-Type: text/event-stream\r\n", body),
            ("", String::new()),
        ])
        .await;
        let mut transport = HttpTransport::new(url);

        transport
            .send_request(JsonRpcRequest::new(5i64, "tools/call"))
            .await
            .unwrap();

        // Every message on the stream is delivered, in order.
        let Some(JsonRpcMessage::Notification(notification)) = transport.recv().await.unwrap()
        else {
            panic!("expected the notification first");
        };
        assert_eq!(notification.method, "notifications/progress");
        let Some(JsonRpcMessage::Request(ping)) = transport.recv().await.unwrap() else {
            panic!("expected the ping request");
        };
        assert_eq!((ping.method.as_str(), &ping.id), ("ping", &99.into()));
        let Some(JsonRpcMessage::Response(response)) = transport.recv().await.unwrap() else {
            panic!("expected the response");
        };
        assert_eq!(response.id, RequestId::Number(5));

        // The ping is answered with a POST of the response.
        transport
            .respond(JsonRpcResponse::success(ping.id, json!({})))
            .await
            .unwrap();
        let seen = seen.lock().unwrap();
        assert!(seen[1].contains(r#""id":99"#), "{}", seen[1]);
        assert!(seen[1].contains(r#""result":{}"#), "{}", seen[1]);
    }

    #[tokio::test]
    async fn sse_response_without_the_result_is_an_error() {
        let body = sse(&[json!({"jsonrpc": "2.0", "method": "notifications/progress"})]);
        let (url, _) = serve(vec![("Content-Type: text/event-stream\r\n", body)]).await;
        let mut transport = HttpTransport::new(url);

        let err = transport
            .request(JsonRpcRequest::new(0i64, "tools/call"))
            .await
            .unwrap_err();

        assert!(matches!(err, McpError::ConnectionClosed), "{err:?}");
    }

    #[tokio::test]
    async fn unknown_content_type_is_an_error() {
        let (url, _) = serve(vec![("Content-Type: text/plain\r\n", "nope".to_string())]).await;
        let mut transport = HttpTransport::new(url);

        let err = transport
            .request(JsonRpcRequest::new(0i64, "tools/list"))
            .await
            .unwrap_err();

        assert!(matches!(err, McpError::Transport(_)), "{err:?}");
    }

    #[tokio::test]
    async fn protocol_version_follows_initialize() {
        let init = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "test"}
            }
        })
        .to_string();
        let list = json!({"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}).to_string();
        let (url, seen) = serve(vec![
            (
                "Content-Type: application/json\r\nMcp-Session-Id: abc\r\n",
                init,
            ),
            ("Content-Type: application/json\r\n", String::new()),
            ("Content-Type: application/json\r\n", list),
        ])
        .await;
        let mut transport = HttpTransport::new(url);

        transport
            .request(JsonRpcRequest::new(0i64, "initialize"))
            .await
            .unwrap();
        // The 202-style notification reply has no body to parse.
        transport
            .notify(JsonRpcNotification::new("notifications/initialized"))
            .await
            .unwrap();
        transport
            .request(JsonRpcRequest::new(0i64, "tools/list"))
            .await
            .unwrap();

        let seen = seen.lock().unwrap();
        assert!(!seen[0].contains("mcp-protocol-version"), "{}", seen[0]);
        for head in &seen[1..] {
            assert!(head.contains("mcp-protocol-version: 2025-06-18"), "{head}");
            assert!(head.contains("mcp-session-id: abc"), "{head}");
        }
    }
}
