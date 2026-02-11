use std::path::Path;

use crate::error::{PdfProcessError, Result};
use crate::model::{Page, PdfProcessOptions};
use crate::pdfium::bind_pdfium;
use pdfium_render::prelude::*;

pub(crate) fn render_pages_from_path(
    source_pdf: &Path,
    pages: &mut [Page],
    output_pages_dir: &Path,
    options: &PdfProcessOptions,
) -> Result<()> {
    let pdfium = bind_pdfium(options)?;
    let doc = pdfium
        .load_pdf_from_file(source_pdf, None)
        .map_err(|e| PdfProcessError::Parse(e.to_string()))?;
    render_pages_impl(doc, pages, output_pages_dir, options)
}

pub(crate) fn render_pages_from_bytes(
    source_pdf: &[u8],
    pages: &mut [Page],
    output_pages_dir: &Path,
    options: &PdfProcessOptions,
) -> Result<()> {
    let pdfium = bind_pdfium(options)?;
    let doc = pdfium
        .load_pdf_from_byte_vec(source_pdf.to_vec(), None)
        .map_err(|e| PdfProcessError::Parse(e.to_string()))?;
    render_pages_impl(doc, pages, output_pages_dir, options)
}

fn render_pages_impl(
    doc: PdfDocument<'_>,
    pages: &mut [Page],
    output_pages_dir: &Path,
    options: &PdfProcessOptions,
) -> Result<()> {
    std::fs::create_dir_all(output_pages_dir)?;

    for page in pages {
        let src_index = page.source_page.saturating_sub(1) as u16;
        let pdf_page = doc
            .pages()
            .get(src_index)
            .map_err(|e| PdfProcessError::Parse(e.to_string()))?;

        let width_pt = pdf_page.width().value.max(1.0);
        let dpi = options.page_image_dpi.max(72);
        let target_width = ((width_pt / 72.0) * f32::from(dpi)).round().max(256.0) as i32;

        let render_config = PdfRenderConfig::new()
            .set_target_width(target_width)
            .rotate_if_landscape(PdfPageRenderRotation::None, true);

        let image = pdf_page
            .render_with_config(&render_config)
            .map_err(|e| PdfProcessError::Parse(e.to_string()))?
            .as_image()
            .into_rgb8();

        let filename = format!("page_{:04}.png", page.index);
        let abs_path = output_pages_dir.join(&filename);
        image
            .save(&abs_path)
            .map_err(|e| PdfProcessError::Parse(e.to_string()))?;

        page.image_ref = Some(format!("pages/{filename}"));
        if page.vision_ref.is_none() && matches!(page.mode.as_str(), "vision_only" | "hybrid") {
            page.vision_ref = page.image_ref.clone();
        }
    }

    Ok(())
}
