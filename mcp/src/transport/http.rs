//! HTTP transport for MCP.
//!
//! This transport uses HTTP POST for sending requests to the server.

use std::sync::atomic::{AtomicI64, Ordering};

use tracing::debug;
use zenwave::{Client, client, header};

use super::traits::{Result, Transport};
use crate::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, RequestId};

/// HTTP transport for MCP communication.
///
/// This transport sends requests via HTTP POST.
#[derive(Debug)]
pub struct HttpTransport {
    /// Base URL of the MCP server.
    base_url: String,
    /// Optional authorization header value.
    auth: Option<String>,
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
        builder = builder.header(header::ACCEPT.as_str(), "application/json");
        builder = builder.header(header::USER_AGENT.as_str(), "aither-mcp/0.1");

        if let Some(auth) = &self.auth {
            builder = builder.header(header::AUTHORIZATION.as_str(), auth.clone());
        }

        let response: JsonRpcResponse = builder
            .json_body(&req)
            .json()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP request failed: {e}")))?;

        debug!("MCP HTTP RX: {:?}", response);

        Ok(response)
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
        builder = builder.header(header::ACCEPT.as_str(), "application/json");
        builder = builder.header(header::USER_AGENT.as_str(), "aither-mcp/0.1");

        if let Some(auth) = &self.auth {
            builder = builder.header(header::AUTHORIZATION.as_str(), auth.clone());
        }

        let _: serde_json::Value = builder
            .json_body(&notif)
            .json()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP request failed: {e}")))?;

        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
}
