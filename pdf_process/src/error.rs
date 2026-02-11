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
    /// Pdfium dynamic library path must be configured to render page images.
    #[error("pdfium library path is required; set PdfProcessOptions.pdfium_library_path")]
    MissingPdfiumLibraryPath,
    /// Pdfium dynamic library could not be loaded.
    #[error("failed to load pdfium library: {0}")]
    PdfiumLoad(String),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, PdfProcessError>;
