//! ACP protocol definitions.
//!
//! Defines message types, error codes, and protocol constants
//! for the Agent Client Protocol.

mod auth;
mod elicitation;
mod error;
mod ext;
mod session;
mod types;

pub use auth::*;
pub use elicitation::*;
pub use error::{AcpError, JsonRpcError};
pub use ext::*;
pub use session::*;
pub use types::*;

// Re-export JSON-RPC message types from MCP (shared protocol layer)
pub use aither_mcp::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId,
};

/// Result type for ACP operations.
pub type Result<T> = std::result::Result<T, AcpError>;
