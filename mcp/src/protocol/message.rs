//! JSON-RPC 2.0 message types for MCP.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::JsonRpcError;

/// JSON-RPC request ID. Can be a string or number.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// String ID.
    String(String),
    /// Numeric ID.
    Number(i64),
}

impl From<i64> for RequestId {
    fn from(id: i64) -> Self {
        Self::Number(id)
    }
}

impl From<String> for RequestId {
    fn from(id: String) -> Self {
        Self::String(id)
    }
}

impl From<&str> for RequestId {
    fn from(id: &str) -> Self {
        Self::String(id.to_string())
    }
}

/// JSON-RPC 2.0 request message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version, always "2.0".
    pub jsonrpc: String,
    /// Request ID for correlating responses.
    pub id: RequestId,
    /// Method name.
    pub method: String,
    /// Method parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    #[must_use]
    pub fn new(id: impl Into<RequestId>, method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params: None,
        }
    }

    /// Create a new JSON-RPC request with parameters.
    #[must_use]
    pub fn with_params(
        id: impl Into<RequestId>,
        method: impl Into<String>,
        params: impl Serialize,
    ) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params: Some(serde_json::to_value(params).expect("Failed to serialize params")),
        }
    }
}

/// JSON-RPC 2.0 response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// Protocol version, always "2.0".
    pub jsonrpc: String,
    /// Request ID this response corresponds to.
    pub id: RequestId,
    /// Result on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Create a successful response.
    #[must_use]
    pub fn success(id: RequestId, result: impl Serialize) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::to_value(result).expect("Failed to serialize result")),
            error: None,
        }
    }

    /// Create an error response.
    #[must_use]
    pub fn error(id: RequestId, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Check if this response is an error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Get the result, returning an error if this is an error response.
    pub fn into_result(self) -> Result<Value, JsonRpcError> {
        if let Some(error) = self.error {
            Err(error)
        } else {
            Ok(self.result.unwrap_or(Value::Null))
        }
    }
}

/// JSON-RPC 2.0 notification message (no response expected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    /// Protocol version, always "2.0".
    pub jsonrpc: String,
    /// Method name.
    pub method: String,
    /// Method parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    /// Create a new notification.
    #[must_use]
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params: None,
        }
    }

    /// Create a new notification with parameters.
    #[must_use]
    pub fn with_params(method: impl Into<String>, params: impl Serialize) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params: Some(serde_json::to_value(params).expect("Failed to serialize params")),
        }
    }
}

/// Any JSON-RPC message (request, response, or notification).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcMessage {
    /// A request message.
    Request(JsonRpcRequest),
    /// A response message.
    Response(JsonRpcResponse),
    /// A notification message.
    Notification(JsonRpcNotification),
}

impl JsonRpcMessage {
    /// Check if this is a request.
    #[must_use]
    pub fn is_request(&self) -> bool {
        matches!(self, Self::Request(_))
    }

    /// Check if this is a response.
    #[must_use]
    pub fn is_response(&self) -> bool {
        matches!(self, Self::Response(_))
    }

    /// Check if this is a notification.
    #[must_use]
    pub fn is_notification(&self) -> bool {
        matches!(self, Self::Notification(_))
    }

    /// Try to get as a request.
    #[must_use]
    pub fn as_request(&self) -> Option<&JsonRpcRequest> {
        match self {
            Self::Request(req) => Some(req),
            _ => None,
        }
    }

    /// Try to get as a response.
    #[must_use]
    pub fn as_response(&self) -> Option<&JsonRpcResponse> {
        match self {
            Self::Response(res) => Some(res),
            _ => None,
        }
    }

    /// Try to get as a notification.
    #[must_use]
    pub fn as_notification(&self) -> Option<&JsonRpcNotification> {
        match self {
            Self::Notification(notif) => Some(notif),
            _ => None,
        }
    }
}
