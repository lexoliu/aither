//! HTTP transport for MCP (Streamable HTTP).
//!
//! This transport uses HTTP POST for sending requests to the server,
//! with support for MCP session management via `Mcp-Session-Id` header.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::RwLock;

use tracing::debug;
use zenwave::{Client, ResponseExt, client, header};

use super::traits::{Result, Transport};
use crate::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, RequestId};

/// MCP session ID header name.
const MCP_SESSION_ID_HEADER: &str = "Mcp-Session-Id";

/// HTTP transport for MCP communication (Streamable HTTP).
///
/// This transport sends requests via HTTP POST and handles session management
/// via the `Mcp-Session-Id` header as per MCP specification 2025-03-26.
#[derive(Debug)]
pub struct HttpTransport {
    /// Base URL of the MCP server.
    base_url: String,
    /// Optional authorization header value.
    auth: Option<String>,
    /// Session ID returned by server during initialization.
    session_id: RwLock<Option<String>>,
    /// Next request ID.
    next_id: AtomicI64,
    /// Whether the transport is closed.
    closed: bool,
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
            session_id: RwLock::new(None),
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
}

impl Transport for HttpTransport {
    async fn request(&mut self, mut req: JsonRpcRequest) -> Result<JsonRpcResponse> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }

        // Assign request ID
        let id = self.next_request_id();
        req.id = id;

        debug!("MCP HTTP TX: {:?}", req);

        // Build and send HTTP request
        let mut backend = client();
        let mut builder = backend.post(&self.base_url);

        builder = builder.header(header::CONTENT_TYPE.as_str(), "application/json");
        builder = builder.header(
            header::ACCEPT.as_str(),
            "application/json, text/event-stream",
        );
        builder = builder.header(header::USER_AGENT.as_str(), "aither-mcp/0.1");

        if let Some(auth) = &self.auth {
            builder = builder.header(header::AUTHORIZATION.as_str(), auth.clone());
        }

        // Include session ID if we have one
        if let Some(session_id) = self.session_id.read().unwrap().as_ref() {
            builder = builder.header(MCP_SESSION_ID_HEADER, session_id.clone());
        }

        // Send request and get response with headers
        let response = builder
            .json_body(&req)
            .await
            .map_err(|e| McpError::Transport(format!("HTTP request failed: {e}")))?;

        // Extract session ID from response headers
        if let Some(session_id) = response.headers().get(MCP_SESSION_ID_HEADER) {
            if let Ok(session_str) = session_id.to_str() {
                debug!("MCP HTTP session ID: {}", session_str);
                *self.session_id.write().unwrap() = Some(session_str.to_string());
            }
        }

        // Parse response body as JSON
        let result: JsonRpcResponse = response
            .into_json()
            .await
            .map_err(|e| McpError::Transport(format!("Response parse failed: {e}")))?;

        debug!("MCP HTTP RX: {:?}", result);

        Ok(result)
    }

    async fn notify(&mut self, notif: JsonRpcNotification) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }

        debug!("MCP HTTP notify: {:?}", notif);

        // Build and send HTTP request (fire and forget)
        let mut backend = client();
        let mut builder = backend.post(&self.base_url);

        builder = builder.header(header::CONTENT_TYPE.as_str(), "application/json");
        builder = builder.header(
            header::ACCEPT.as_str(),
            "application/json, text/event-stream",
        );
        builder = builder.header(header::USER_AGENT.as_str(), "aither-mcp/0.1");

        if let Some(auth) = &self.auth {
            builder = builder.header(header::AUTHORIZATION.as_str(), auth.clone());
        }

        // Include session ID if we have one
        if let Some(session_id) = self.session_id.read().unwrap().as_ref() {
            builder = builder.header(MCP_SESSION_ID_HEADER, session_id.clone());
        }

        // Notifications may return empty body, so we ignore parse errors
        let _ = builder.json_body(&notif).await;

        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
}
