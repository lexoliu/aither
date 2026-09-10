//! Errors reported by the ACP client.

use std::process::ExitStatus;

use aither_mcp::protocol::JsonRpcError;
use thiserror::Error;

/// Error returned by [`AcpClient`](super::AcpClient) methods.
///
/// Distinguishes a broken transport (including the agent process exiting)
/// from JSON-RPC errors returned by the agent and from agent messages that
/// violate the protocol.
#[derive(Debug, Clone, Error)]
pub enum ClientError {
    /// A transport-level failure: spawn, I/O, or an internal channel error.
    ///
    /// The underlying error is stringified so the error can be cloned and
    /// delivered to every pending request at once.
    #[error("transport error: {0}")]
    Transport(String),

    /// The connection closed or the agent process exited.
    ///
    /// For child-process connections `status` carries the exit status when it
    /// could be collected; in-memory and remote transports report `None`.
    #[error(
        "agent connection closed{}",
        .status.map(|status| format!(" (exit status: {status})")).unwrap_or_default()
    )]
    Closed {
        /// Exit status of the agent process, when the transport can report one.
        status: Option<ExitStatus>,
    },

    /// The agent returned a JSON-RPC error response.
    #[error("JSON-RPC error: {0}")]
    JsonRpc(#[from] JsonRpcError),

    /// The agent sent a message that violates the protocol: a response that
    /// does not match its request's shape, or a malformed payload.
    #[error("protocol violation: {0}")]
    Protocol(String),
}
