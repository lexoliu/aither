//! MCP error types and JSON-RPC error codes.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// JSON-RPC error codes as defined by the specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ErrorCode(pub i32);

impl ErrorCode {
    /// Parse error - Invalid JSON was received.
    pub const PARSE_ERROR: Self = Self(-32700);
    /// Invalid Request - The JSON sent is not a valid Request object.
    pub const INVALID_REQUEST: Self = Self(-32600);
    /// Method not found - The method does not exist / is not available.
    pub const METHOD_NOT_FOUND: Self = Self(-32601);
    /// Invalid params - Invalid method parameter(s).
    pub const INVALID_PARAMS: Self = Self(-32602);
    /// Internal error - Internal JSON-RPC error.
    pub const INTERNAL_ERROR: Self = Self(-32603);
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::PARSE_ERROR => write!(f, "Parse error"),
            Self::INVALID_REQUEST => write!(f, "Invalid request"),
            Self::METHOD_NOT_FOUND => write!(f, "Method not found"),
            Self::INVALID_PARAMS => write!(f, "Invalid params"),
            Self::INTERNAL_ERROR => write!(f, "Internal error"),
            Self(code) => write!(f, "Error {code}"),
        }
    }
}

/// JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: ErrorCode,
    /// Human-readable error message.
    pub message: String,
    /// Additional error data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcError {
    /// Create a new JSON-RPC error.
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Create a new JSON-RPC error with additional data.
    #[must_use]
    pub fn with_data(code: ErrorCode, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    /// Create a parse error.
    #[must_use]
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::PARSE_ERROR, message)
    }

    /// Create an invalid request error.
    #[must_use]
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::INVALID_REQUEST, message)
    }

    /// Create a method not found error.
    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self::new(
            ErrorCode::METHOD_NOT_FOUND,
            format!("Method not found: {method}"),
        )
    }

    /// Create an invalid params error.
    #[must_use]
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::INVALID_PARAMS, message)
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::INTERNAL_ERROR, message)
    }
}

impl fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for JsonRpcError {}

/// MCP-specific error type.
#[derive(Debug, Error)]
pub enum McpError {
    /// JSON-RPC protocol error.
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

    /// Connection closed.
    #[error("Connection closed")]
    ConnectionClosed,

    /// Timeout.
    #[error("Request timeout")]
    Timeout,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
