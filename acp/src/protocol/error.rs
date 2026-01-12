//! ACP error types.

use thiserror::Error;

// Re-export JSON-RPC error from MCP to ensure type compatibility
pub use aither_mcp::protocol::JsonRpcError;

/// ACP error type.
#[derive(Debug, Error)]
pub enum AcpError {
    /// JSON-RPC error.
    #[error("JSON-RPC error: {0}")]
    JsonRpc(#[from] JsonRpcError),

    /// Transport error.
    #[error("Transport error: {0}")]
    Transport(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Session not found.
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Invalid session state.
    #[error("Invalid session state: {0}")]
    InvalidState(String),

    /// Agent error.
    #[error("Agent error: {0}")]
    Agent(String),

    /// Connection closed.
    #[error("Connection closed")]
    ConnectionClosed,
}
