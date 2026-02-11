use thiserror::Error;

/// Errors raised by the local llama.cpp backend.
#[derive(Debug, Error)]
pub enum LlamaError {
    /// Model loading failed or model path is invalid.
    #[error("model error: {0}")]
    Model(String),
    /// Context initialization failed.
    #[error("context error: {0}")]
    Context(String),
    /// Tokenization or detokenization failed.
    #[error("token error: {0}")]
    Token(String),
    /// Decoding/sampling failure from llama.cpp.
    #[error("decode error: {0}")]
    Decode(String),
    /// The request cannot be represented by this backend.
    #[error("unsupported request: {0}")]
    Unsupported(String),
    /// JSON serialization/deserialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
