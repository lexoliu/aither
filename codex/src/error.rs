//! Error type for the Codex provider.

/// Errors from Codex OAuth flows and API calls.
#[derive(Debug, thiserror::Error)]
pub enum CodexError {
    /// HTTP transport or status error.
    #[error("HTTP request failed: {0}")]
    Http(#[from] zenwave::Error),
    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Error body returned by an endpoint (`detail` field).
    #[error("Codex API error: {0}")]
    Api(String),
    /// The user has not completed device authorization yet.
    #[error("device authorization is still pending")]
    AuthorizationPending,
    /// The OAuth loopback listener failed or the callback did not arrive.
    #[error("OAuth callback error: {0}")]
    Callback(String),
    /// The callback `state` did not match the issued one.
    #[error("OAuth state mismatch")]
    StateMismatch,
    /// I/O error (e.g. binding the loopback listener).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Error from the underlying `OpenAI` client.
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] aither_openai::OpenAIError),
}
