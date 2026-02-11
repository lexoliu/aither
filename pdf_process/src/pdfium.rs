use std::path::PathBuf;

use pdfium_render::prelude::{Pdfium, PdfiumError};

use crate::error::{PdfProcessError, Result};
use crate::model::PdfProcessOptions;

pub(crate) fn bind_pdfium(options: &PdfProcessOptions) -> Result<Pdfium> {
    let path = options
        .pdfium_library_path
        .as_ref()
        .ok_or(PdfProcessError::MissingPdfiumLibraryPath)
        .map(PathBuf::from)?;

    let bindings = Pdfium::bind_to_library(path).map_err(map_pdfium_load_error)?;
    Ok(Pdfium::new(bindings))
}

fn map_pdfium_load_error(error: PdfiumError) -> PdfProcessError {
    PdfProcessError::PdfiumLoad(error.to_string())
}
