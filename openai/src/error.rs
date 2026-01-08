use std::fmt;
use std::time::Duration;
use zenwave::{BodyError, Error as ZenwaveError, sse::ParseError as SseParseError};

/// Errors that can arise when calling the `OpenAI` API.
#[derive(Debug)]
pub enum OpenAIError {
    /// HTTP layer errors.
    Http(ZenwaveError),
    /// Response body parsing failures.
    Body(BodyError),
    /// SSE parsing failures.
    Stream(SseParseError),
    /// JSON serialization/deserialization errors.
    Json(serde_json::Error),
    /// Base64 decoding failures (e.g., image payloads).
    Decode(base64::DecodeError),
    /// API contract violations or unsupported operations.
    Api(String),
    /// Rate limit exceeded (HTTP 429).
    RateLimit {
        /// Message from the API.
        message: String,
        /// Suggested retry delay from Retry-After header.
        retry_after: Option<Duration>,
    },
    /// Server error (HTTP 5xx).
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Error message.
        message: String,
    },
    /// Request timed out.
    Timeout,
}

impl fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "HTTP error: {err}"),
            Self::Body(err) => write!(f, "Body error: {err}"),
            Self::Stream(err) => write!(f, "SSE error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::Decode(err) => write!(f, "Base64 decode error: {err}"),
            Self::Api(message) => write!(f, "{message}"),
            Self::RateLimit { message, retry_after } => {
                write!(f, "Rate limit exceeded: {message}")?;
                if let Some(delay) = retry_after {
                    write!(f, " (retry after {}s)", delay.as_secs())?;
                }
                Ok(())
            }
            Self::ServerError { status, message } => {
                write!(f, "Server error {status}: {message}")
            }
            Self::Timeout => write!(f, "Request timed out"),
        }
    }
}

impl std::error::Error for OpenAIError {}

impl From<BodyError> for OpenAIError {
    fn from(value: BodyError) -> Self {
        Self::Body(value)
    }
}

impl From<SseParseError> for OpenAIError {
    fn from(value: SseParseError) -> Self {
        Self::Stream(value)
    }
}

impl From<serde_json::Error> for OpenAIError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<base64::DecodeError> for OpenAIError {
    fn from(value: base64::DecodeError) -> Self {
        Self::Decode(value)
    }
}
