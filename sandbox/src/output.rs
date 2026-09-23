//! Output management for terminal tool execution.
//!
//! Outputs are classified by size and type:
//! - **Inline**: Super tiny text (< 5 lines) - always in context, never gets URL
//! - **Loaded**: Small text/images - in context, URL generated only on offload
//! - **Stored**: Large text/binary/video - file created immediately

use std::{
    collections::HashMap,
    io::Cursor,
    path::{Path, PathBuf},
};

use askama::Template;
use async_fs as fs;
use image::{GenericImageView, ImageFormat, ImageReader, imageops::FilterType};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};
use tracing::debug;

/// Expected output format for terminal execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// Plain text output (default)
    #[default]
    Text,
    /// Image data (png, jpg, etc.)
    Image,
    /// Video data
    Video,
    /// Binary data
    Binary,
    /// Auto-detect from content
    Auto,
}

/// Requested delivery resolution for media returned through terminal output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MediaResolution {
    /// Use the source media bytes as-is.
    #[default]
    Auto,
    /// Downscale to a 512px bounding box before loading into context.
    Low,
    /// Downscale to a 1024px bounding box before loading into context.
    Medium,
    /// Downscale to a 2048px bounding box before loading into context.
    High,
    /// Preserve the original media resolution without resizing.
    Native,
}

impl MediaResolution {
    const fn max_dimension(self) -> Option<u32> {
        match self {
            Self::Auto | Self::Native => None,
            Self::Low => Some(512),
            Self::Medium => Some(1024),
            Self::High => Some(2048),
        }
    }
}

/// Content that can be loaded into agent context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Text content, potentially truncated
    Text {
        /// The text content
        text: String,
        /// Whether the content was truncated
        truncated: bool,
    },
    /// Image content as base64
    Image {
        /// Base64-encoded image data
        data: String,
        /// MIME type (e.g., "image/png")
        media_type: String,
    },
}

/// Output entry with lazy file creation.
///
/// Serializes to a flat JSON object with optional fields.
#[derive(Debug, Clone)]
pub enum OutputEntry {
    /// No output (`ToolResult::Done`) - nothing to store
    Empty,

    /// Super tiny content - always in context, NEVER gets URL.
    /// Examples: "done", "3 files deleted", short status messages.
    Inline {
        /// The content to display inline.
        content: Content,
    },

    /// Content is loaded in context, no file yet.
    /// URL will be generated when offloaded.
    Loaded {
        /// The content to display.
        content: Content,
        /// Raw bytes for potential later file creation
        raw: Vec<u8>,
        /// Format for extension when creating file
        format: OutputFormat,
    },

    /// Content is stored in file, URL available.
    Stored {
        /// Relative path: "outputs/purple-ocean-swift-meadow.txt"
        url: String,
        /// Absolute filesystem path for the stored payload.
        path: String,
        /// Preview content (if large text)
        content: Option<Content>,
    },
}

impl Serialize for OutputEntry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Empty => {
                let map = serializer.serialize_map(Some(0))?;
                map.end()
            }
            // Loaded output carries provenance the caller does not need on the
            // wire, so it serializes exactly like inline output.
            Self::Inline { content } | Self::Loaded { content, .. } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            Self::Stored { url, path, content } => {
                let count = if content.is_some() { 3 } else { 2 };
                let mut map = serializer.serialize_map(Some(count))?;
                map.serialize_entry("url", url)?;
                map.serialize_entry("path", path)?;
                if let Some(c) = content {
                    map.serialize_entry("content", c)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for OutputEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            url: Option<String>,
            path: Option<String>,
            content: Option<Content>,
        }

        let h = Helper::deserialize(deserializer)?;

        Ok(match (h.url, h.path, h.content) {
            (Some(url), path, content) => Self::Stored {
                url,
                path: path.unwrap_or_default(),
                content,
            },
            (None, _, Some(content)) => Self::Inline { content },
            (None, _, None) => Self::Empty,
        })
    }
}

impl std::fmt::Display for OutputEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => Ok(()),
            Self::Inline { content } | Self::Loaded { content, .. } => match content {
                Content::Text { text, truncated } => {
                    write!(f, "{text}")?;
                    if *truncated {
                        write!(
                            f,
                            "\n[truncated: full output is available in the stored file]"
                        )?;
                    }
                    Ok(())
                }
                Content::Image { media_type, .. } => {
                    write!(f, "[Image: {media_type}]")
                }
            },
            Self::Stored { url, path, content } => {
                if let Some(content) = content {
                    match content {
                        Content::Text { text, truncated } => {
                            write!(f, "{text}")?;
                            if *truncated {
                                write!(
                                    f,
                                    "\n[full content at {url}; absolute path: {path}; output was truncated in-chat]"
                                )?;
                            }
                        }
                        Content::Image { media_type, .. } => {
                            write!(f, "[Image: {media_type}] at {url} ({path})")?;
                        }
                    }
                } else {
                    write!(f, "[content at {url} ({path})]")?;
                }
                Ok(())
            }
        }
    }
}

impl OutputEntry {
    /// Returns the stored file path for entries written to disk.
    #[must_use]
    pub fn stored_path(&self, base_dir: &Path) -> Option<PathBuf> {
        match self {
            Self::Stored { url, path, .. } => {
                if !path.is_empty() {
                    return Some(PathBuf::from(path));
                }
                let filename = url.strip_prefix("outputs/").unwrap_or(url);
                Some(base_dir.join(filename))
            }
            _ => None,
        }
    }
}

/// Internal reference to a stored output.
#[derive(Debug, Clone)]
pub struct OutputRef {
    /// The output entry state
    pub entry: OutputEntry,
    /// Detected or specified format
    pub format: OutputFormat,
    /// Size in bytes
    pub size: usize,
}

/// Maximum output size (in bytes) delivered inline to the model.
/// Outputs exceeding this are saved to file; the model still receives a head
/// preview capped at this budget together with the file reference, so it can
/// always make progress without an extra read round-trip.
/// This is the single source of truth for output size limits.
pub const INLINE_OUTPUT_LIMIT: usize = 16_000;

#[derive(Template)]
#[template(path = "stored_output_preview.txt", escape = "none")]
struct StoredOutputPreviewTemplate<'a> {
    preview: &'a str,
    reason: &'a str,
    line_count: usize,
    byte_count: usize,
    url: &'a str,
}

/// Renders the model-facing content for a stored text output: a head preview
/// within the caller's budget plus the file reference and read guidance.
fn stored_text_content(
    text: &str,
    reason: &str,
    line_count: usize,
    byte_count: usize,
    url: &str,
    max_lines: usize,
    byte_budget: usize,
) -> Content {
    let preview = head_preview(text, max_lines, byte_budget);
    let rendered = StoredOutputPreviewTemplate {
        preview: preview.trim_end_matches('\n'),
        reason,
        line_count,
        byte_count,
        url,
    }
    .render()
    .expect("stored output preview template must render");
    Content::Text {
        text: rendered,
        truncated: true,
    }
}

/// Returns the longest prefix of `text` spanning at most `max_lines` lines and
/// `byte_budget` bytes, cutting at a line boundary when possible and never
/// splitting a UTF-8 character.
pub fn head_preview(text: &str, max_lines: usize, byte_budget: usize) -> &str {
    let mut end = 0;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if index == max_lines || end + line.len() > byte_budget {
            break;
        }
        end += line.len();
    }
    if end == 0 {
        // The first line alone exceeds the byte budget: cut inside it at a
        // character boundary.
        let mut cut = byte_budget.min(text.len());
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        end = cut;
    }
    text.get(..end)
        .expect("prefix end is aligned to a line or char boundary")
}

/// A pending URL allocation that hasn't been written to disk yet.
///
/// Used during context compaction to allocate URLs for content before
/// deciding which ones are actually referenced in the summary.
#[derive(Debug, Clone)]
pub struct PendingUrl {
    /// The allocated URL (e.g., "outputs/amber-oak-swift-river.txt")
    pub url: String,
    /// Raw bytes to write if this URL is referenced
    pub raw: Vec<u8>,
    /// Format for file extension
    pub format: OutputFormat,
}

/// Manages output storage for terminal executions.
#[derive(Debug)]
pub struct OutputStore {
    /// Base directory for outputs (e.g., $WORKDIR/outputs)
    dir: PathBuf,
    /// Tracked output entries by ID
    entries: HashMap<String, OutputRef>,
    /// Counter for unique IDs
    next_id: u64,
}

impl OutputStore {
    /// Creates a new output store in the given directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub async fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir).await?;
        Ok(Self {
            dir,
            entries: HashMap::new(),
            next_id: 0,
        })
    }

    /// Returns the base directory path.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Saves output data and returns an `OutputEntry` with appropriate state.
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails (for large outputs).
    pub async fn save(
        &mut self,
        data: &[u8],
        format: OutputFormat,
        resolution: MediaResolution,
    ) -> std::io::Result<OutputEntry> {
        let entry = Self::save_to_dir(&self.dir, data, format, resolution).await?;

        // Track the reference
        let id = format!("output_{}", self.next_id);
        self.next_id += 1;

        let format = if format == OutputFormat::Auto {
            detect_format(data)
        } else {
            format
        };
        let output_ref = OutputRef {
            entry: entry.clone(),
            format,
            size: data.len(),
        };
        self.entries.insert(id, output_ref);

        Ok(entry)
    }

    /// Saves output data to a directory.
    ///
    /// Uses the system-wide `INLINE_OUTPUT_LIMIT` constant:
    /// - Empty data → `Empty`
    /// - Below limit → `Inline` (shown directly)
    /// - Above limit → `Stored` (file created, path returned)
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails.
    pub async fn save_to_dir(
        dir: &Path,
        data: &[u8],
        format: OutputFormat,
        resolution: MediaResolution,
    ) -> std::io::Result<OutputEntry> {
        Self::save_to_dir_with_limit(dir, data, format, resolution, Some(INLINE_OUTPUT_LIMIT)).await
    }

    /// Saves output data with an explicit size limit.
    ///
    /// When output exceeds the limit, it's saved to file and only a reference is returned.
    /// The LLM must use head/tail/grep to read the file content like a human.
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails.
    pub async fn save_to_dir_with_limit(
        dir: &Path,
        data: &[u8],
        format: OutputFormat,
        resolution: MediaResolution,
        limit: Option<usize>,
    ) -> std::io::Result<OutputEntry> {
        if data.is_empty() {
            return Ok(OutputEntry::Empty);
        }

        let format = if format == OutputFormat::Auto {
            detect_format(data)
        } else {
            format
        };

        let exceeds_limit = limit.is_some_and(|max| data.len() > max);
        if exceeds_limit && !matches!(format, OutputFormat::Image) {
            return save_large_output(dir, data, format).await;
        }

        match format {
            OutputFormat::Text | OutputFormat::Auto => save_text_output(dir, data, format).await,
            OutputFormat::Image => {
                let prepared = prepare_image_for_context(data, resolution)?;
                let preview_content = Content::Image {
                    data: base64_encode(&prepared.bytes),
                    media_type: prepared.media_type,
                };
                let preview_fits_inline = limit.is_none_or(|max| prepared.bytes.len() <= max);
                let source_fits_inline = limit.is_none_or(|max| data.len() <= max);

                if source_fits_inline && preview_fits_inline {
                    Ok(OutputEntry::Inline {
                        content: preview_content,
                    })
                } else {
                    let (url, path) = create_file(dir, data, format).await?;
                    Ok(OutputEntry::Stored {
                        path: path.display().to_string(),
                        url,
                        content: preview_fits_inline.then_some(preview_content),
                    })
                }
            }
            OutputFormat::Video | OutputFormat::Binary => {
                // Binary/video are always Stored
                save_large_output(dir, data, format).await
            }
        }
    }

    /// Offloads a Loaded entry to disk, generating a URL.
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails.
    pub async fn offload(&mut self, entry: &OutputEntry) -> std::io::Result<OutputEntry> {
        match entry {
            OutputEntry::Loaded {
                content,
                raw,
                format,
            } => {
                let (url, path) = create_file(&self.dir, raw, *format).await?;
                Ok(OutputEntry::Stored {
                    path: path.display().to_string(),
                    url,
                    content: Some(content.clone()),
                })
            }
            // Already stored or inline/empty - return as-is
            other => Ok(other.clone()),
        }
    }

    /// Retrieves an output reference by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&OutputRef> {
        self.entries.get(id)
    }

    /// Allocates a URL for a Loaded entry without writing to disk.
    ///
    /// This is used during context compaction to allocate URLs before
    /// calling the fast LLM. The file is only written if the URL is
    /// actually referenced in the compacted summary.
    ///
    /// Returns `None` for:
    /// - `Empty`: No content to store
    /// - `Inline`: Super tiny content, never gets a URL
    /// - `Stored`: Already has a URL (no action needed)
    ///
    /// Returns `Some(PendingUrl)` for `Loaded` entries.
    #[must_use]
    pub fn allocate_url(&self, entry: &OutputEntry) -> Option<PendingUrl> {
        match entry {
            OutputEntry::Loaded { raw, format, .. } => {
                let ext = extension_for_data(raw, *format);
                let name = generate_word_filename();
                let url = format!("outputs/{name}.{ext}");
                Some(PendingUrl {
                    url,
                    raw: raw.clone(),
                    format: *format,
                })
            }
            // Empty, Inline, or Stored - no URL allocation needed
            _ => None,
        }
    }

    /// Allocates a URL for text content without writing to disk.
    ///
    /// This is used during context compaction to allocate URLs for
    /// tool output content before calling the fast LLM.
    ///
    /// Returns the allocated URL (e.g., "outputs/amber-oak-swift-river.txt").
    #[must_use]
    pub fn allocate_text_url(&self) -> String {
        let name = generate_word_filename();
        format!("outputs/{name}.txt")
    }

    /// Writes text content to a URL path.
    ///
    /// Call this only for URLs that were actually referenced in the
    /// compacted summary.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub async fn write_text(&self, url: &str, content: &str) -> std::io::Result<PathBuf> {
        let filename = url.strip_prefix("outputs/").unwrap_or(url);
        let filepath = self.dir.join(filename);
        fs::write(&filepath, content.as_bytes()).await?;
        debug!(url = %url, size = content.len(), "wrote text output");
        Ok(filepath)
    }

    /// Writes a pending URL's data to disk.
    ///
    /// Call this only for URLs that were actually referenced in the
    /// compacted summary.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub async fn write_pending(&self, pending: &PendingUrl) -> std::io::Result<PathBuf> {
        let filename = pending.url.strip_prefix("outputs/").unwrap_or(&pending.url);
        let filepath = self.dir.join(filename);
        fs::write(&filepath, &pending.raw).await?;
        debug!(url = %pending.url, size = pending.raw.len(), "wrote pending output");
        Ok(filepath)
    }

    /// Reads the raw data for a stored output URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub async fn read(&self, url: &str) -> std::io::Result<Vec<u8>> {
        let filename = url.strip_prefix("outputs/").unwrap_or(url);
        let filepath = self.dir.join(filename);
        fs::read(&filepath).await
    }

    /// Cleans up all stored outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if files cannot be deleted.
    pub async fn cleanup(&mut self) -> std::io::Result<()> {
        for (_, output_ref) in self.entries.drain() {
            if let OutputEntry::Stored { url, path, .. } = output_ref.entry {
                let filename = url.strip_prefix("outputs/").unwrap_or(&url);
                let filepath = if path.is_empty() {
                    self.dir.join(filename)
                } else {
                    PathBuf::from(path)
                };
                if let Err(e) = fs::remove_file(&filepath).await {
                    tracing::warn!(path = %filepath.display(), error = %e, "failed to remove output file");
                }
            }
        }
        Ok(())
    }
}

/// Saves raw data to a file without any processing, returning the URL.
///
/// Used to preserve the original uncompressed output alongside a compressed version.
///
/// # Errors
///
/// Returns an error if file creation fails.
pub async fn save_raw_to_file(dir: &Path, data: &[u8]) -> std::io::Result<String> {
    let (url, _) = create_file(dir, data, OutputFormat::Text).await?;
    debug!(url = %url, size = data.len(), "saved raw output");
    Ok(url)
}

/// Saves text output with simple size-based classification.
///
/// - Below `INLINE_OUTPUT_LIMIT` → show inline
/// - Above `INLINE_OUTPUT_LIMIT` → save to file, return head preview plus file reference
async fn save_text_output(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<OutputEntry> {
    if data.len() <= INLINE_OUTPUT_LIMIT {
        let content = Content::Text {
            text: String::from_utf8_lossy(data).into_owned(),
            truncated: false,
        };
        Ok(OutputEntry::Inline { content })
    } else {
        save_large_output(dir, data, format).await
    }
}

/// Saves text output with a line-count limit.
///
/// When the output exceeds `max_lines` (or the byte limit), the full content
/// is saved to a file and the returned content carries a head preview within
/// the caller's budget plus the file reference and read guidance.
pub async fn save_text_with_line_limit(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    byte_limit: Option<usize>,
) -> std::io::Result<OutputEntry> {
    if data.is_empty() {
        return Ok(OutputEntry::Empty);
    }

    let format = if format == OutputFormat::Auto {
        detect_format(data)
    } else {
        format
    };

    // For non-text formats, fall back to the standard byte-based logic
    if !matches!(format, OutputFormat::Text | OutputFormat::Auto) {
        return OutputStore::save_to_dir_with_limit(dir, data, format, resolution, byte_limit)
            .await;
    }

    let text = String::from_utf8_lossy(data);
    let line_count = text.lines().count();

    // Check byte limit first
    let exceeds_bytes = byte_limit.is_some_and(|max| data.len() > max);
    let exceeds_lines = line_count > max_lines;

    if !exceeds_bytes && !exceeds_lines {
        let content = Content::Text {
            text: text.into_owned(),
            truncated: false,
        };
        return Ok(OutputEntry::Inline { content });
    }

    // Save full output to file, but still hand the model a head preview
    // within its requested budget so it never has to re-read blind.
    let (url, path) = create_file(dir, data, format).await?;

    let byte_budget = byte_limit.unwrap_or(INLINE_OUTPUT_LIMIT);
    let reason = if exceeds_lines {
        format!("output exceeded max_lines limit ({line_count} lines > {max_lines} max)")
    } else {
        format!("output exceeded the {byte_budget}-byte inline limit")
    };
    let content = stored_text_content(
        &text,
        &reason,
        line_count,
        data.len(),
        &url,
        max_lines,
        byte_budget,
    );

    Ok(OutputEntry::Stored {
        path: path.display().to_string(),
        url,
        content: Some(content),
    })
}

/// Saves output that exceeds limit - stores full content, returning a head
/// preview plus the file reference for text formats.
async fn save_large_output(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<OutputEntry> {
    // Store full content to file
    let (url, path) = create_file(dir, data, format).await?;

    let content = match format {
        OutputFormat::Text | OutputFormat::Auto => {
            let text = String::from_utf8_lossy(data);
            let line_count = text.lines().count();
            let reason = format!("output exceeded the {INLINE_OUTPUT_LIMIT}-byte inline limit");
            Some(stored_text_content(
                &text,
                &reason,
                line_count,
                data.len(),
                &url,
                usize::MAX,
                INLINE_OUTPUT_LIMIT,
            ))
        }
        _ => None, // Binary/video/image just get file path
    };

    Ok(OutputEntry::Stored {
        path: path.display().to_string(),
        url,
        content,
    })
}

#[derive(Debug)]
struct PreparedImage {
    bytes: Vec<u8>,
    media_type: String,
}

fn prepare_image_for_context(
    data: &[u8],
    resolution: MediaResolution,
) -> std::io::Result<PreparedImage> {
    let media_type = detect_image_media_type(data);
    let Some(max_dimension) = resolution.max_dimension() else {
        return Ok(PreparedImage {
            bytes: data.to_vec(),
            media_type,
        });
    };

    let image = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(io_error)?
        .decode()
        .map_err(io_error)?;

    let (width, height) = image.dimensions();
    if width <= max_dimension && height <= max_dimension {
        return Ok(PreparedImage {
            bytes: data.to_vec(),
            media_type,
        });
    }

    let format = image_format_from_media_type(media_type.as_str()).ok_or_else(|| {
        std::io::Error::other(format!(
            "unsupported image media type for terminal resizing: {media_type}"
        ))
    })?;
    let resized = image.resize(max_dimension, max_dimension, FilterType::Lanczos3);
    let mut cursor = Cursor::new(Vec::new());
    resized.write_to(&mut cursor, format).map_err(io_error)?;

    Ok(PreparedImage {
        bytes: cursor.into_inner(),
        media_type,
    })
}

/// Creates a file with a four-random-words name.
async fn create_file(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<(String, PathBuf)> {
    let ext = extension_for_data(data, format);
    let name = generate_word_filename();
    let filename = format!("{name}.{ext}");
    let url = format!("outputs/{filename}");
    let filepath = dir.join(&filename);

    fs::write(&filepath, data).await?;
    debug!(url = %url, size = data.len(), format = ?format, "saved output");

    Ok((url, filepath))
}

fn io_error(error: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

/// Generates a filename using four random words.
pub fn generate_word_filename() -> String {
    crate::naming::random_word_slug(4)
}

/// Detects the output format from content.
fn detect_format(data: &[u8]) -> OutputFormat {
    // Check for common image magic bytes
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        // PNG
        return OutputFormat::Image;
    }
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        // JPEG
        return OutputFormat::Image;
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return OutputFormat::Image;
    }
    if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        return OutputFormat::Image;
    }

    // Check for video formats
    if data.len() > 12 && &data[4..8] == b"ftyp" {
        // MP4/MOV
        return OutputFormat::Video;
    }

    // Check if it's printable UTF-8 text (no null bytes or control chars except newline/tab)
    if let Ok(text) = std::str::from_utf8(data) {
        let is_text = text
            .chars()
            .all(|c| c == '\n' || c == '\r' || c == '\t' || (c >= ' ' && c != '\x7f'));
        if is_text {
            return OutputFormat::Text;
        }
    }

    OutputFormat::Binary
}

/// Returns file extension for format.
const fn format_extension(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Text | OutputFormat::Auto => "txt",
        OutputFormat::Image | OutputFormat::Binary => "bin",
        OutputFormat::Video => "mp4",
    }
}

fn image_extension(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        Some("png")
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpg")
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("gif")
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        Some("webp")
    } else {
        None
    }
}

fn extension_for_data(data: &[u8], format: OutputFormat) -> &'static str {
    if matches!(format, OutputFormat::Image) {
        return image_extension(data).unwrap_or_else(|| format_extension(format));
    }
    format_extension(format)
}

/// Detects image MIME type from data.
fn detect_image_media_type(data: &[u8]) -> String {
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        "image/png".to_string()
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg".to_string()
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        "image/gif".to_string()
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        "image/webp".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}

fn image_format_from_media_type(media_type: &str) -> Option<ImageFormat> {
    match media_type {
        "image/png" => Some(ImageFormat::Png),
        "image/jpeg" => Some(ImageFormat::Jpeg),
        "image/gif" => Some(ImageFormat::Gif),
        "image/webp" => Some(ImageFormat::WebP),
        _ => None,
    }
}

/// Base64 encodes data using standard alphabet with padding.
fn base64_encode(data: &[u8]) -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use base64::Engine;
    use image::{DynamicImage, ImageBuffer, Rgba};

    #[test]
    fn test_detect_format() {
        // PNG magic bytes
        assert_eq!(
            detect_format(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            OutputFormat::Image
        );

        // Plain text
        assert_eq!(detect_format(b"Hello, world!"), OutputFormat::Text);

        // Binary with null bytes
        assert_eq!(
            detect_format(&[0x00, 0x01, 0x02, 0x03]),
            OutputFormat::Binary
        );
    }

    #[test]
    fn test_output_entry_inline_serialize() {
        let entry = OutputEntry::Inline {
            content: Content::Text {
                text: "done".to_string(),
                truncated: false,
            },
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"content\""));
        assert!(!json.contains("\"url\""));
    }

    #[test]
    fn test_output_entry_stored_serialize() {
        let entry = OutputEntry::Stored {
            url: "outputs/test.txt".to_string(),
            path: "/tmp/test.txt".to_string(),
            content: Some(Content::Text {
                text: "preview".to_string(),
                truncated: true,
            }),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"url\""));
        assert!(json.contains("\"content\""));
    }

    #[test]
    fn test_output_entry_empty_serialize() {
        let entry = OutputEntry::Empty;
        let json = serde_json::to_string(&entry).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_generate_word_filename() {
        let name = generate_word_filename();
        assert_eq!(name.split('-').count(), 4, "Should have 4 words: {name}");
    }

    #[tokio::test]
    async fn save_image_low_resolution_downscales_inline_content() {
        let dir = tempfile::tempdir().unwrap();
        let jpeg = encode_test_image(4000, 3000, ImageFormat::Jpeg);

        let entry = OutputStore::save_to_dir_with_limit(
            dir.path(),
            &jpeg,
            OutputFormat::Image,
            MediaResolution::Low,
            None,
        )
        .await
        .unwrap();

        let OutputEntry::Inline {
            content: Content::Image { data, media_type },
        } = entry
        else {
            panic!("expected inline image preview");
        };

        assert_eq!(media_type, "image/jpeg");

        let preview_bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .unwrap();
        let preview = ImageReader::new(Cursor::new(preview_bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        assert!(preview.width() <= 512);
        assert!(preview.height() <= 512);
    }

    #[tokio::test]
    async fn save_large_image_keeps_stored_file_and_preview_when_resolution_fits_limit() {
        const LIMIT: usize = 100_000;

        let dir = tempfile::tempdir().unwrap();
        let jpeg = encode_test_image(4000, 3000, ImageFormat::Jpeg);
        assert!(jpeg.len() > LIMIT);

        let entry = OutputStore::save_to_dir_with_limit(
            dir.path(),
            &jpeg,
            OutputFormat::Image,
            MediaResolution::Low,
            Some(LIMIT),
        )
        .await
        .unwrap();

        let OutputEntry::Stored {
            url,
            path,
            content: Some(Content::Image { data, media_type }),
        } = entry
        else {
            panic!("expected stored image preview");
        };

        assert_eq!(media_type, "image/jpeg");
        assert!(
            std::path::Path::new(&path)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("jpg")
                    || extension.eq_ignore_ascii_case("jpeg"))
        );
        assert!(dir.path().join(url.trim_start_matches("outputs/")).exists());

        let preview_bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .unwrap();
        let preview = ImageReader::new(Cursor::new(preview_bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        assert!(preview.width() <= 512);
        assert!(preview.height() <= 512);
    }

    fn encode_test_image(width: u32, height: u32, format: ImageFormat) -> Vec<u8> {
        let image = ImageBuffer::from_fn(width, height, |x, y| {
            let r = u8::try_from((x * 255) / width.max(1)).expect("red channel fits in u8");
            let g = u8::try_from((y * 255) / height.max(1)).expect("green channel fits in u8");
            let b = u8::try_from(((x + y) * 255) / (width + height).max(1))
                .expect("blue channel fits in u8");
            Rgba([r, g, b, 255])
        });
        let dynamic = DynamicImage::ImageRgba8(image);
        let mut cursor = Cursor::new(Vec::new());
        dynamic.write_to(&mut cursor, format).unwrap();
        cursor.into_inner()
    }

    #[test]
    fn head_preview_respects_line_budget() {
        let text = "one\ntwo\nthree\n";
        assert_eq!(head_preview(text, 2, usize::MAX), "one\ntwo\n");
        assert_eq!(head_preview(text, 10, usize::MAX), text);
    }

    #[test]
    fn head_preview_cuts_at_line_boundary_within_byte_budget() {
        let text = "aaaa\nbbbb\ncccc\n";
        // Budget of 12 fits "aaaa\n" + "bbbb\n" (10 bytes) but not the next line.
        assert_eq!(head_preview(text, usize::MAX, 12), "aaaa\nbbbb\n");
    }

    #[test]
    fn head_preview_never_splits_multibyte_chars() {
        // Each '界' is 3 bytes; a 500-byte budget must cut on a char boundary.
        let text = "界".repeat(400);
        let preview = head_preview(&text, usize::MAX, 500);
        assert!(!preview.is_empty());
        assert!(preview.len() <= 500);
        assert!(preview.chars().all(|c| c == '界'));
    }

    #[tokio::test]
    async fn stored_text_output_includes_head_preview_and_guidance() {
        let dir = tempfile::tempdir().unwrap();
        let mut data = String::new();
        for i in 0..2000 {
            data.extend([format!("line number {i} with some padding text\n")]);
        }
        assert!(data.len() > INLINE_OUTPUT_LIMIT);

        let entry = save_text_with_line_limit(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
            10_000,
            Some(INLINE_OUTPUT_LIMIT),
        )
        .await
        .unwrap();

        let OutputEntry::Stored {
            content: Some(Content::Text { text, truncated }),
            url,
            ..
        } = entry
        else {
            panic!("expected stored entry with text content");
        };
        assert!(truncated);
        assert!(
            text.starts_with("line number 0 "),
            "preview must lead with real content: {text}"
        );
        assert!(
            text.contains(&url),
            "guidance must reference the saved file"
        );
        assert!(text.contains("Use head/tail/grep/sed"));
    }

    #[tokio::test]
    async fn stored_text_output_preview_honors_max_lines() {
        let dir = tempfile::tempdir().unwrap();
        let data = "alpha\nbeta\ngamma\ndelta\n".repeat(2);

        let entry = save_text_with_line_limit(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
            3,
            Some(INLINE_OUTPUT_LIMIT),
        )
        .await
        .unwrap();

        let OutputEntry::Stored {
            content: Some(Content::Text { text, .. }),
            ..
        } = entry
        else {
            panic!("expected stored entry with text content");
        };
        assert!(
            text.starts_with("alpha\nbeta\ngamma\n[Preview only:"),
            "preview must stop at max_lines: {text}"
        );
    }
}
