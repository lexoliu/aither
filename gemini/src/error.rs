use std::fmt;

use base64::DecodeError;
use zenwave::{BodyError, error::BoxHttpError};

/// Errors raised by the Gemini backend.
#[derive(Debug)]
pub enum GeminiError {
    /// HTTP transport errors.
    Http(BoxHttpError),
    /// Request/response body encoding failures.
    Body(BodyError),
    /// JSON serialization/deserialization problems.
    Json(serde_json::Error),
    /// Base64 decode failure (inline images/audio).
    Decode(DecodeError),
    /// API level errors or unsupported operations.
    Api(String),
}

impl fmt::Display for GeminiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "HTTP error: {err}"),
            Self::Body(err) => write!(f, "Body error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::Decode(err) => write!(f, "Base64 decode error: {err}"),
            Self::Api(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for GeminiError {}

impl From<BodyError> for GeminiError {
    fn from(value: BodyError) -> Self {
        Self::Body(value)
    }
}

impl From<serde_json::Error> for GeminiError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<DecodeError> for GeminiError {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
    }
}
