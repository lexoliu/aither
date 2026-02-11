use std::ops::RangeInclusive;

/// OCR strategy for scanned pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrMode {
    /// Never perform OCR fallback.
    Off,
    /// Apply OCR only when native extraction quality is too low.
    Auto,
    /// Force OCR mode for all pages.
    Force,
}

/// OCR backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrBackend {
    /// Run OCR in-process with PaddleOCR ONNX models.
    Paddle,
}

/// Chunking strategy for XML content units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// Split by fixed character windows.
    Fixed,
    /// Split near sentence boundaries first.
    Sentence,
    /// Prefer sentence splits, fallback to fixed windows.
    Hybrid,
}

/// Metadata detail level in generated XML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataVerbosity {
    /// Include only minimal retrieval-critical metadata.
    Minimal,
    /// Include additional document metadata when available.
    Standard,
}

/// Paddle OCR model configuration.
#[derive(Debug, Clone)]
pub struct PaddleOcrConfig {
    /// Path to Paddle/OAR text detection model (.onnx).
    pub det_model_path: String,
    /// Optional path to text line orientation model (.onnx).
    pub cls_model_path: Option<String>,
    /// Path to Paddle/OAR text recognition model (.onnx).
    pub rec_model_path: String,
    /// Path to OCR character dictionary file.
    pub char_dict_path: String,
}

/// Runtime options for PDF processing.
#[derive(Debug, Clone)]
pub struct PdfProcessOptions {
    /// Optional inclusive 1-based page range.
    pub page_range: Option<RangeInclusive<usize>>,
    /// Strategy used when splitting canonical text into chunk units.
    pub chunk_strategy: ChunkStrategy,
    /// Soft character budget for each chunk.
    pub target_chunk_chars: usize,
    /// Soft token estimate budget for each chunk.
    pub target_chunk_tokens: usize,
    /// OCR fallback behavior for low-quality pages.
    pub ocr_mode: OcrMode,
    /// OCR backend implementation used when OCR is enabled.
    pub ocr_backend: OcrBackend,
    /// OCR language hint for downstream model selection.
    pub ocr_language: String,
    /// OCR render DPI for rasterizing PDF pages before recognition.
    pub ocr_dpi: u16,
    /// Render DPI for page image assets in folder export mode.
    pub page_image_dpi: u16,
    /// Absolute path to Pdfium dynamic library file (for example `/opt/lib/libpdfium.dylib`).
    pub pdfium_library_path: Option<String>,
    /// Paddle OCR model bundle used when OCR is enabled.
    pub paddle_ocr: Option<PaddleOcrConfig>,
    /// Emit vision references for non-text-heavy pages.
    pub include_vision_refs: bool,
    /// Metadata detail level in XML output.
    pub metadata_verbosity: MetadataVerbosity,
}

impl Default for PdfProcessOptions {
    fn default() -> Self {
        Self {
            page_range: None,
            chunk_strategy: ChunkStrategy::Hybrid,
            target_chunk_chars: 1800,
            target_chunk_tokens: 450,
            ocr_mode: OcrMode::Off,
            ocr_backend: OcrBackend::Paddle,
            ocr_language: "en".to_string(),
            ocr_dpi: 300,
            page_image_dpi: 144,
            pdfium_library_path: None,
            paddle_ocr: None,
            include_vision_refs: true,
            metadata_verbosity: MetadataVerbosity::Minimal,
        }
    }
}

/// Processed document ready to serialize to XML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessedDocument {
    /// Source identifier (path or virtual label).
    pub source: String,
    /// Total pages in the original PDF.
    pub page_count: usize,
    /// Extracted document metadata.
    pub metadata: DocumentMeta,
    /// Canonical per-page outputs.
    pub pages: Vec<Page>,
    /// Budget-aware chunk units derived from canonical content.
    pub chunks: Vec<Chunk>,
}

/// Minimal PDF metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentMeta {
    /// Optional title from PDF info dictionary.
    pub title: Option<String>,
    /// Optional author from PDF info dictionary.
    pub author: Option<String>,
    /// Optional creation date when standard verbosity is enabled.
    pub creation_date: Option<String>,
}

/// Page modality derived from extraction quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageMode {
    /// Native text extraction produced quality text.
    Native,
    /// OCR text was used as canonical text.
    Ocr,
    /// No usable text; page represented by vision reference only.
    VisionOnly,
    /// Text exists but extraction quality is mixed.
    Hybrid,
}

impl PageMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Ocr => "ocr",
            Self::VisionOnly => "vision_only",
            Self::Hybrid => "hybrid",
        }
    }
}

/// Parsed page in canonical text form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// 1-based page index in selected output set.
    pub index: usize,
    /// Original 1-based page index in source PDF.
    pub source_page: usize,
    /// Modality classification for this page.
    pub mode: PageMode,
    /// Canonical text for this page (native or OCR).
    pub text: String,
    /// Character count of canonical text.
    pub text_chars: usize,
    /// Approximate token count for canonical text.
    pub token_estimate: usize,
    /// Optional vision reference identifier.
    pub vision_ref: Option<String>,
    /// Optional rendered page image path (relative in bundle export mode).
    pub image_ref: Option<String>,
}

/// Chunk unit for prompt budget-aware packing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Stable chunk identifier.
    pub id: String,
    /// First page index contributing to this chunk.
    pub page_start: usize,
    /// Last page index contributing to this chunk.
    pub page_end: usize,
    /// Approximate token count for chunk content.
    pub token_estimate: usize,
    /// Chunk modality label.
    pub modality: ChunkModality,
    /// Canonical chunk text.
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkModality {
    /// Chunk contains text-derived content only.
    Text,
    /// Chunk describes vision-only content.
    Vision,
    /// Chunk has mixed text and vision provenance.
    Hybrid,
}

impl ChunkModality {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Hybrid => "hybrid",
        }
    }
}
