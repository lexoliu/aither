//! Transport trait definitions.

use std::future::Future;
use std::process::ExitStatus;

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
    ///
    /// While a request is outstanding the transport reads inbound messages
    /// itself and discards anything that is not the matching response. Callers
    /// that need to observe notifications or peer-initiated requests must not
    /// use this method; drive the transport with
    /// [`BidirectionalTransport::send_request`] and
    /// [`BidirectionalTransport::recv`] instead.
    fn request(
        &mut self,
        req: JsonRpcRequest,
    ) -> impl Future<Output = Result<JsonRpcResponse>> + Send;

    /// Send a notification (no response expected).
    fn notify(&mut self, notif: JsonRpcNotification) -> impl Future<Output = Result<()>> + Send;

    /// Close the transport connection.
    fn close(&mut self) -> impl Future<Output = Result<()>> + Send;

    /// Report the exit status of the peer once the connection has closed.
    ///
    /// Waits for the peer to terminate when it is a child process, killing it
    /// if it is still running after the connection broke. Transports that are
    /// not backed by a process report `None`.
    fn exit_status(&mut self) -> impl Future<Output = Option<ExitStatus>> + Send {
        std::future::ready(None)
    }
}

/// Bidirectional transport that can also receive incoming messages.
///
/// This is used for server-side transports where we need to listen
/// for incoming requests from clients.
pub trait BidirectionalTransport: Transport {
    /// Receive the next incoming message.
    ///
    /// Returns `None` if the connection is closed. Dropping the returned
    /// future before it completes does not lose bytes: a partially read
    /// message is resumed by the next call, so `recv` may safely race other
    /// futures in a `select`-style loop.
    fn recv(&mut self) -> impl Future<Output = Result<Option<JsonRpcMessage>>> + Send;

    /// Send a response to a request.
    fn respond(&mut self, response: JsonRpcResponse) -> impl Future<Output = Result<()>> + Send;

    /// Send a request without waiting for its response.
    ///
    /// The request is written to the peer as-is; the caller owns the request
    /// ID and matches the response itself when it arrives through
    /// [`recv`](Self::recv). This is the counterpart to [`Transport::request`]
    /// for full-duplex connections where a single reader routes every
    /// inbound message.
    fn send_request(&mut self, req: JsonRpcRequest) -> impl Future<Output = Result<()>> + Send;
}
