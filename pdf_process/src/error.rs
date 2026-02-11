use thiserror::Error;

/// Errors emitted by PDF processing pipeline.
#[derive(Debug, Error)]
pub enum PdfProcessError {
    /// The input bytes do not decode as a valid PDF structure.
    #[error("failed to parse PDF: {0}")]
    Parse(String),
    /// The source PDF could not be read from the filesystem.
    #[error("failed to read PDF: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, PdfProcessError>;
