//! PDF to XML conversion optimized for LLM context windows.
//!
//! This crate converts PDF input into compact, deterministic XML with
//! chunk metadata suitable for prompt assembly and retrieval workflows.

mod chunking;
mod error;
mod model;
mod ocr;
mod parser;
mod xml;

pub use error::{PdfProcessError, Result};
pub use model::{
    ChunkStrategy, MetadataVerbosity, OcrBackend, OcrMode, PdfProcessOptions, ProcessedDocument,
};

use std::path::{Path, PathBuf};

/// PDF processor entrypoint.
#[derive(Debug, Clone)]
pub struct PdfProcessor {
    source: PdfSource,
}

#[derive(Debug, Clone)]
enum PdfSource {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

impl PdfProcessor {
    /// Build a processor from a PDF file path.
    #[must_use]
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            source: PdfSource::Path(path.into()),
        }
    }

    /// Build a processor from PDF bytes.
    #[must_use]
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            source: PdfSource::Bytes(bytes.into()),
        }
    }

    /// Convert PDF input into token-efficient XML.
    pub fn to_xml(&self, options: PdfProcessOptions) -> Result<String> {
        let processed = self.to_model(options)?;
        Ok(xml::to_xml(&processed))
    }

    /// Convert PDF input into the structured intermediate model.
    pub fn to_model(&self, options: PdfProcessOptions) -> Result<ProcessedDocument> {
        match &self.source {
            PdfSource::Path(path) => parser::parse_from_path(path, &options),
            PdfSource::Bytes(bytes) => parser::parse_from_bytes(bytes, "memory", &options),
        }
    }

    /// Returns source path if available.
    #[must_use]
    pub fn source_path(&self) -> Option<&Path> {
        match &self.source {
            PdfSource::Path(path) => Some(path.as_path()),
            PdfSource::Bytes(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_for_invalid_pdf_errors() {
        let processor = PdfProcessor::from_bytes(b"not-a-pdf".to_vec());
        let result = processor.to_xml(PdfProcessOptions::default());
        assert!(result.is_err());
    }
}
