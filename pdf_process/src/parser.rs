use std::path::Path;

use lopdf::Document;

use crate::chunking::build_chunks;
use crate::error::{PdfProcessError, Result};
use crate::model::{
    DocumentMeta, MetadataVerbosity, OcrMode, Page, PageMode, PdfProcessOptions, ProcessedDocument,
};
use crate::ocr::{OcrSource, ocr_page};

pub(crate) fn parse_from_path(
    path: &Path,
    options: &PdfProcessOptions,
) -> Result<ProcessedDocument> {
    let doc = Document::load(path).map_err(|e| PdfProcessError::Parse(e.to_string()))?;
    parse_document(
        doc,
        path.display().to_string(),
        OcrSource::Path(path),
        options,
    )
}

pub(crate) fn parse_from_bytes(
    bytes: &[u8],
    source_name: &str,
    options: &PdfProcessOptions,
) -> Result<ProcessedDocument> {
    let doc = Document::load_mem(bytes).map_err(|e| PdfProcessError::Parse(e.to_string()))?;
    parse_document(
        doc,
        source_name.to_string(),
        OcrSource::Bytes(bytes),
        options,
    )
}

fn parse_document(
    doc: Document,
    source: String,
    ocr_source: OcrSource<'_>,
    options: &PdfProcessOptions,
) -> Result<ProcessedDocument> {
    let page_map = doc.get_pages();
    let mut page_numbers: Vec<u32> = page_map.keys().copied().collect();
    page_numbers.sort_unstable();

    let selected = select_pages(&page_numbers, options.page_range.clone());
    let mut pages = Vec::with_capacity(selected.len());

    for (idx, page_number) in selected.iter().enumerate() {
        let text_raw = doc
            .extract_text(&[*page_number])
            .unwrap_or_else(|_| String::new());
        let normalized = normalize_text(&text_raw);
        let quality = text_quality_score(&normalized);
        let initial_mode = classify_mode(quality, options.ocr_mode);

        let ocr_text = if matches!(
            initial_mode,
            PageMode::Ocr | PageMode::Hybrid | PageMode::VisionOnly
        ) {
            ocr_page(ocr_source, *page_number, options).map(|t| normalize_text(&t))
        } else {
            None
        };

        let (mode, text) = match initial_mode {
            PageMode::Ocr => {
                if let Some(ocr) = ocr_text.clone() {
                    (PageMode::Ocr, ocr)
                } else {
                    (PageMode::VisionOnly, String::new())
                }
            }
            PageMode::Hybrid => {
                if let Some(ocr) = ocr_text.clone() {
                    if ocr.chars().count() > normalized.chars().count() {
                        (PageMode::Hybrid, ocr)
                    } else {
                        (PageMode::Hybrid, normalized)
                    }
                } else {
                    (PageMode::Hybrid, normalized)
                }
            }
            PageMode::VisionOnly => {
                if let Some(ocr) = ocr_text {
                    (PageMode::Ocr, ocr)
                } else {
                    (PageMode::VisionOnly, String::new())
                }
            }
            PageMode::Native => (PageMode::Native, normalized),
        };

        let text_chars = text.chars().count();
        let token_estimate = estimate_tokens(&text);
        let vision_ref = if options.include_vision_refs
            && matches!(mode, PageMode::VisionOnly | PageMode::Hybrid)
        {
            Some(format!("page:{}", idx + 1))
        } else {
            None
        };

        pages.push(Page {
            index: idx + 1,
            mode,
            text,
            text_chars,
            token_estimate,
            vision_ref,
        });
    }

    let metadata = extract_metadata(&doc, options.metadata_verbosity);
    let chunks = build_chunks(
        &pages,
        options.chunk_strategy,
        options.target_chunk_chars,
        options.target_chunk_tokens,
    );

    Ok(ProcessedDocument {
        source,
        page_count: page_numbers.len(),
        metadata,
        pages,
        chunks,
    })
}

fn select_pages(pages: &[u32], range: Option<std::ops::RangeInclusive<usize>>) -> Vec<u32> {
    match range {
        None => pages.to_vec(),
        Some(range) => pages
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                let page_index = idx + 1;
                if range.contains(&page_index) {
                    Some(*p)
                } else {
                    None
                }
            })
            .collect(),
    }
}

fn normalize_text(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn text_quality_score(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let chars = text.chars().count() as f32;
    let printable = text
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .count() as f32;
    let words = text.split_whitespace().count() as f32;
    let printable_ratio = printable / chars;
    let density = (words / (chars / 5.0).max(1.0)).min(1.0);
    (0.7 * printable_ratio) + (0.3 * density)
}

fn classify_mode(score: f32, ocr_mode: OcrMode) -> PageMode {
    match ocr_mode {
        OcrMode::Force => PageMode::Ocr,
        OcrMode::Off => {
            if score < 0.2 {
                PageMode::VisionOnly
            } else {
                PageMode::Native
            }
        }
        OcrMode::Auto => {
            if score < 0.1 {
                PageMode::VisionOnly
            } else if score < 0.4 {
                PageMode::Hybrid
            } else {
                PageMode::Native
            }
        }
    }
}

fn extract_metadata(doc: &Document, verbosity: MetadataVerbosity) -> DocumentMeta {
    let mut meta = DocumentMeta::default();
    if let Ok(info_ref) = doc.trailer.get(b"Info")
        && let Ok(info_ref) = info_ref.as_reference()
        && let Ok(dict) = doc.get_dictionary(info_ref)
    {
        meta.title = dict
            .get(b"Title")
            .ok()
            .and_then(|v| v.as_str().ok())
            .map(to_clean_string);
        meta.author = dict
            .get(b"Author")
            .ok()
            .and_then(|v| v.as_str().ok())
            .map(to_clean_string);
        if matches!(verbosity, MetadataVerbosity::Standard) {
            meta.creation_date = dict
                .get(b"CreationDate")
                .ok()
                .and_then(|v| v.as_str().ok())
                .map(to_clean_string);
        }
    }
    meta
}

fn to_clean_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim().to_string()
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}
