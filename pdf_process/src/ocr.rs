use std::path::Path;

use crate::model::PdfProcessOptions;

#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "ocr"), allow(dead_code))]
pub(crate) enum OcrSource<'a> {
    Path(&'a Path),
    Bytes(&'a [u8]),
}

#[cfg(feature = "ocr")]
pub(crate) fn ocr_page(
    source: OcrSource<'_>,
    page_number: u32,
    options: &PdfProcessOptions,
) -> Option<String> {
    use crate::model::OcrBackend;

    if !matches!(options.ocr_backend, OcrBackend::Paddle) {
        return None;
    }

    let det = options.ocr_det_model_path.as_deref()?;
    let rec = options.ocr_rec_model_path.as_deref()?;
    let dict = options.ocr_char_dict_path.as_deref()?;

    let image = render_page_to_rgb(source, page_number, options.ocr_dpi)?;

    let mut builder = oar_ocr::oarocr::OAROCRBuilder::new(det, rec, dict)
        .image_batch_size(1)
        .region_batch_size(16);

    if let Some(cls) = options.ocr_cls_model_path.as_deref() {
        builder = builder.with_text_line_orientation_classification(cls);
    }

    let ocr = builder.build().ok()?;
    let mut results = ocr.predict(vec![image]).ok()?;
    let result = results.pop()?;

    let mut lines = Vec::new();
    for region in result.text_regions {
        if let Some(text) = region.text {
            let text = text.trim();
            if !text.is_empty() {
                lines.push(text.to_string());
            }
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

#[cfg(feature = "ocr")]
fn render_page_to_rgb(
    source: OcrSource<'_>,
    page_number: u32,
    dpi: u16,
) -> Option<image::RgbImage> {
    use pdfium_render::prelude::*;

    let pdfium = Pdfium::default();
    let doc = match source {
        OcrSource::Path(path) => pdfium.load_pdf_from_file(path, None).ok()?,
        OcrSource::Bytes(bytes) => pdfium.load_pdf_from_byte_slice(bytes, None).ok()?,
    };

    let page_index = page_number.checked_sub(1)? as u16;
    let page = doc.pages().get(page_index).ok()?;

    let width_pt = page.width().value.max(1.0);
    let target_width = ((width_pt / 72.0) * f32::from(dpi)).round().max(256.0) as i32;

    let render_config = PdfRenderConfig::new()
        .set_target_width(target_width)
        .rotate_if_landscape(PdfPageRenderRotation::None, true);

    let image = page
        .render_with_config(&render_config)
        .ok()?
        .as_image()
        .into_rgb8();
    Some(image)
}

#[cfg(not(feature = "ocr"))]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn ocr_page(
    _source: OcrSource<'_>,
    _page_number: u32,
    _options: &PdfProcessOptions,
) -> Option<String> {
    None
}
