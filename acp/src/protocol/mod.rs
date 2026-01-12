//! ACP protocol definitions.
//!
//! Defines message types, error codes, and protocol constants
//! for the Agent Client Protocol.

mod error;
mod types;

pub use error::{AcpError, JsonRpcError};
pub use types::*;

// Re-export JSON-RPC message types from MCP (shared protocol layer)
pub use aither_mcp::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId,
};

/// Result type for ACP operations.
pub type Result<T> = std::result::Result<T, AcpError>;
