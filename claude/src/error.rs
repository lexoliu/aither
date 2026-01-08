//! Error types for the Claude API client.

use core::fmt;
use zenwave::{BodyError, Error as ZenwaveError, sse::ParseError as SseParseError};

/// Errors that can arise when calling the Claude API.
#[derive(Debug)]
pub enum ClaudeError {
    /// HTTP layer errors.
    Http(ZenwaveError),
    /// Response body parsing failures.
    Body(BodyError),
    /// SSE parsing failures.
    Stream(SseParseError),
    /// JSON serialization/deserialization errors.
    Json(serde_json::Error),
    /// API contract violations or unsupported operations.
    Api(String),
}

impl fmt::Display for ClaudeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "HTTP error: {err}"),
            Self::Body(err) => write!(f, "Body error: {err}"),
            Self::Stream(err) => write!(f, "SSE error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::Api(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ClaudeError {}

impl From<ZenwaveError> for ClaudeError {
    fn from(value: ZenwaveError) -> Self {
        Self::Http(value)
    }
}

impl From<BodyError> for ClaudeError {
    fn from(value: BodyError) -> Self {
        Self::Body(value)
    }
}

impl From<SseParseError> for ClaudeError {
    fn from(value: SseParseError) -> Self {
        Self::Stream(value)
    }
}

impl From<serde_json::Error> for ClaudeError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
