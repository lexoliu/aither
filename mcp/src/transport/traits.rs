//! Transport trait definitions.

use std::future::Future;

use crate::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError,
};

/// Result type for transport operations.
pub type Result<T> = std::result::Result<T, McpError>;

/// Transport trait for sending JSON-RPC messages.
///
/// This trait provides the core functionality for MCP communication,
/// allowing requests to be sent and responses to be received.
///
/// Note: Uses `&mut self` to avoid locks - transports should be owned
/// by a single task/context.
pub trait Transport: Send {
    /// Send a request and wait for the response.
    fn request(
        &mut self,
        req: JsonRpcRequest,
    ) -> impl Future<Output = Result<JsonRpcResponse>> + Send;

    /// Send a notification (no response expected).
    fn notify(&mut self, notif: JsonRpcNotification) -> impl Future<Output = Result<()>> + Send;

    /// Close the transport connection.
    fn close(&mut self) -> impl Future<Output = Result<()>> + Send;
}

/// Bidirectional transport that can also receive incoming messages.
///
/// This is used for server-side transports where we need to listen
/// for incoming requests from clients.
pub trait BidirectionalTransport: Transport {
    /// Receive the next incoming message.
    ///
    /// Returns `None` if the connection is closed.
    fn recv(&mut self) -> impl Future<Output = Result<Option<JsonRpcMessage>>> + Send;

    /// Send a response to a request.
    fn respond(&mut self, response: JsonRpcResponse) -> impl Future<Output = Result<()>> + Send;
}
