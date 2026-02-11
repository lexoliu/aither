//! PDF to XML conversion optimized for LLM context windows.
//!
//! This crate converts PDF input into compact, deterministic XML with
//! chunk metadata suitable for prompt assembly and retrieval workflows.

mod chunking;
mod error;
mod model;
mod ocr;
mod parser;
mod pdfium;
mod render;
mod xml;

pub use error::{PdfProcessError, Result};
pub use model::{
    ChunkStrategy, MetadataVerbosity, OcrBackend, OcrMode, PaddleOcrConfig, PdfProcessOptions,
    ProcessedDocument,
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

    /// Export document.xml and rendered page images into a folder bundle.
    pub async fn export_bundle(
        &self,
        output_dir: impl AsRef<Path>,
        options: PdfProcessOptions,
    ) -> Result<()> {
        let output_dir = output_dir.as_ref().to_path_buf();
        async_fs::create_dir_all(&output_dir).await?;

        let source = self.source.clone();
        let options_for_model = options.clone();
        let mut processed = blocking::unblock(move || match source {
            PdfSource::Path(path) => parser::parse_from_path(&path, &options_for_model),
            PdfSource::Bytes(bytes) => {
                parser::parse_from_bytes(&bytes, "memory", &options_for_model)
            }
        })
        .await?;
        let pages_dir = output_dir.join("pages");

        match &self.source {
            PdfSource::Path(path) => {
                let path = path.clone();
                let mut pages = std::mem::take(&mut processed.pages);
                let options = options.clone();
                let pages_dir_block = pages_dir.clone();
                pages = blocking::unblock(move || {
                    render::render_pages_from_path(&path, &mut pages, &pages_dir_block, &options)?;
                    Ok::<_, crate::error::PdfProcessError>(pages)
                })
                .await?;
                processed.pages = pages;
            }
            PdfSource::Bytes(bytes) => {
                let bytes = bytes.clone();
                let mut pages = std::mem::take(&mut processed.pages);
                let options = options.clone();
                let pages_dir_block = pages_dir.clone();
                pages = blocking::unblock(move || {
                    render::render_pages_from_bytes(
                        &bytes,
                        &mut pages,
                        &pages_dir_block,
                        &options,
                    )?;
                    Ok::<_, crate::error::PdfProcessError>(pages)
                })
                .await?;
                processed.pages = pages;
            }
        }

        let xml = xml::to_xml(&processed);
        async_fs::write(output_dir.join("document.xml"), xml).await?;
        Ok(())
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
