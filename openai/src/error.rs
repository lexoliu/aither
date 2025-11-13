use std::fmt;
use zenwave::{BodyError, Error as HttpError, sse::ParseError as SseParseError};

/// Errors that can arise when calling the `OpenAI` API.
#[derive(Debug)]
pub enum OpenAIError {
    /// HTTP layer errors.
    Http(HttpError),
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
        }
    }
}

impl std::error::Error for OpenAIError {}

impl From<HttpError> for OpenAIError {
    fn from(value: HttpError) -> Self {
        Self::Http(value)
    }
}

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
