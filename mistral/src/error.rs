use std::fmt;

/// Errors from `aither-mistral`.
#[derive(Debug)]
pub enum MistralError {
    /// Required model configuration is missing.
    MissingModel(&'static str),
    /// An API call returned an invalid payload.
    Api(String),
    /// Generic runtime error from mistral.rs.
    Runtime(anyhow::Error),
    /// JSON parsing failure.
    Json(serde_json::Error),
    /// Base64 decoding failure.
    Decode(base64::DecodeError),
}

impl fmt::Display for MistralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingModel(name) => write!(f, "missing model configuration: {name}"),
            Self::Api(message) => write!(f, "{message}"),
            Self::Runtime(err) => write!(f, "runtime error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::Decode(err) => write!(f, "decode error: {err}"),
        }
    }
}

impl std::error::Error for MistralError {}

impl From<anyhow::Error> for MistralError {
    fn from(value: anyhow::Error) -> Self {
        Self::Runtime(value)
    }
}

impl From<serde_json::Error> for MistralError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<base64::DecodeError> for MistralError {
    fn from(value: base64::DecodeError) -> Self {
        Self::Decode(value)
    }
}
