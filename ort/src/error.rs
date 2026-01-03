//! Error types for the ONNX Runtime embedding crate.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur when loading or running ONNX embedding models.
#[derive(Debug, Error)]
pub enum OrtError {
    /// Failed to load or run the ONNX model.
    #[error("onnx runtime error: {0}")]
    Ort(ort::Error),

    /// Failed to load the tokenizer.
    #[error("failed to load tokenizer from {path}: {message}")]
    Tokenizer {
        /// Path to the tokenizer file.
        path: PathBuf,
        /// Error message from tokenizers crate.
        message: String,
    },

    /// Model path was not specified in the builder.
    #[error("model path not specified")]
    MissingModelPath,

    /// Tokenizer file not found in the model directory.
    #[error("tokenizer.json not found in {0}")]
    TokenizerNotFound(PathBuf),

    /// Model file not found in the specified path.
    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),

    /// Output tensor shape mismatch.
    #[error("unexpected output shape: expected 3 dimensions, got {0}")]
    InvalidOutputShape(usize),

    /// Tokenization failed.
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// Ndarray shape error.
    #[error("shape error: {0}")]
    Shape(String),
}

impl From<ort::Error> for OrtError {
    fn from(e: ort::Error) -> Self {
        Self::Ort(e)
    }
}

impl OrtError {
    /// Creates a tokenizer error from a path and error message.
    pub fn tokenizer(path: impl Into<PathBuf>, message: impl ToString) -> Self {
        Self::Tokenizer {
            path: path.into(),
            message: message.to_string(),
        }
    }
}
