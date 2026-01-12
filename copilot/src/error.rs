//! Error types for GitHub Copilot API integration.

use std::time::Duration;

/// Errors that can arise when calling the GitHub Copilot API.
#[derive(Debug, thiserror::Error)]
pub enum CopilotError {
    /// HTTP layer errors from zenwave.
    #[error("HTTP error: {0}")]
    Http(#[from] zenwave::Error),

    /// Response body parsing failures.
    #[error("Body error: {0}")]
    Body(#[from] zenwave::BodyError),

    /// SSE parsing failures.
    #[error("SSE error: {0}")]
    Stream(#[from] zenwave::sse::ParseError),

    /// JSON serialization/deserialization errors.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// API contract violations or unsupported operations.
    #[error("{0}")]
    Api(String),

    /// OAuth authorization pending (user hasn't completed flow yet).
    #[error("Authorization pending")]
    AuthorizationPending,

    /// OAuth device code expired.
    #[error("Device code expired")]
    DeviceCodeExpired,

    /// OAuth access denied by user.
    #[error("Access denied")]
    AccessDenied,

    /// Rate limit exceeded (HTTP 429).
    #[error("Rate limit exceeded: {message}{}", retry_after.map(|d| format!(" (retry after {}s)", d.as_secs())).unwrap_or_default())]
    RateLimit {
        /// Message from the API.
        message: String,
        /// Suggested retry delay from Retry-After header.
        retry_after: Option<Duration>,
    },

    /// Server error (HTTP 5xx).
    #[error("Server error {status}: {message}")]
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Error message.
        message: String,
    },

    /// Request timed out.
    #[error("Request timed out")]
    Timeout,
}
