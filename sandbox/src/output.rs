//! Output management for terminal tool execution.
//!
//! Outputs are classified by size and type:
//! - **Inline**: text within the limits - in the result, never gets a file
//! - **Stored**: large text, binary, video and every image - file created
//!   immediately; text keeps a preview in the result

use std::{
    collections::HashMap,
    io::Cursor,
    path::{Path, PathBuf},
};

use aither_core::llm::{CommandStream, StoredFile};
use async_fs as fs;
use image::{GenericImageView, ImageFormat, ImageReader, imageops::FilterType};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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

/// An output stream holding no output.
#[must_use]
pub const fn empty_stream() -> CommandStream {
    CommandStream::Inline {
        text: String::new(),
    }
}

/// Returns the file a stream was stored in, for streams written to disk.
#[must_use]
pub fn stored_path(stream: &CommandStream) -> Option<PathBuf> {
    match stream {
        CommandStream::Stored { file, .. } | CommandStream::Image { file, .. } => {
            Some(PathBuf::from(&file.path))
        }
        CommandStream::Inline { .. } => None,
    }
}

fn stored_file(url: String, path: &Path) -> StoredFile {
    StoredFile {
        url,
        path: path.display().to_string(),
    }
}

/// Internal reference to a stored output.
#[derive(Debug, Clone)]
pub struct OutputRef {
    /// How the output was delivered
    pub stream: CommandStream,
    /// Detected or specified format
    pub format: OutputFormat,
    /// Size in bytes
    pub size: usize,
}

/// Maximum output size (in bytes) delivered inline to the model.
/// Outputs exceeding this are saved to file; the model still receives a
/// head+tail preview capped at this budget together with the file reference,
/// so it can always make progress without an extra read round-trip.
/// This is the single source of truth for output size limits.
pub const INLINE_OUTPUT_LIMIT: usize = 16_000;

/// A text output that exceeded a limit, with the limits that cut it.
struct CutText<'a> {
    /// The original bytes, which are what gets stored.
    data: &'a [u8],
    /// `data` as text.
    text: &'a str,
    max_lines: usize,
    byte_budget: usize,
}

/// Delivers a text output that exceeded a limit.
///
/// The model-facing form is the folded view when the text is source code and
/// the folded view fits `byte_budget`, otherwise the head+tail preview. That
/// form, with its stored-output notice, is compared in bytes with the
/// original: when it is not smaller the original is returned inline and
/// uncut, and nothing is written to disk.
async fn cut_text_output(
    dir: &Path,
    format: OutputFormat,
    cut: CutText<'_>,
) -> std::io::Result<CommandStream> {
    let CutText {
        data,
        text,
        max_lines,
        byte_budget,
    } = cut;
    let (url, path) = allocate_file(dir, data, format);

    let folded = crate::output_compress::folded_code_content(text)
        .filter(|folded| folded.len() <= byte_budget);
    let preview = folded.unwrap_or_else(|| {
        crate::output_compress::stored_preview(text, max_lines, byte_budget)
            .trim_end_matches('\n')
            .to_owned()
    });
    let stream = CommandStream::Stored {
        file: stored_file(url, &path),
        preview,
        truncated: true,
    };

    let rendered_len = stream.render_for_model().len();
    if rendered_len >= data.len() {
        debug!(
            original = data.len(),
            cut = rendered_len,
            "cut form is not smaller than the output, returning it inline"
        );
        return Ok(CommandStream::Inline {
            text: text.to_owned(),
        });
    }

    fs::write(&path, data).await?;
    debug!(size = data.len(), format = ?format, "saved output");
    Ok(stream)
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

    /// Saves output data and returns the stream the result carries.
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails (for large outputs).
    pub async fn save(
        &mut self,
        data: &[u8],
        format: OutputFormat,
        resolution: MediaResolution,
    ) -> std::io::Result<CommandStream> {
        let stream = Self::save_to_dir(&self.dir, data, format, resolution).await?;

        // Track the reference
        let id = format!("output_{}", self.next_id);
        self.next_id += 1;

        let format = if format == OutputFormat::Auto {
            detect_format(data)
        } else {
            format
        };
        let output_ref = OutputRef {
            stream: stream.clone(),
            format,
            size: data.len(),
        };
        self.entries.insert(id, output_ref);

        Ok(stream)
    }

    /// Saves output data to a directory.
    ///
    /// Uses the system-wide `INLINE_OUTPUT_LIMIT` constant:
    /// - Empty data → an empty `Inline` stream
    /// - Below limit → `Inline` (shown directly)
    /// - Above limit → `Stored` (file created, preview kept for text)
    /// - Images → `Image` (always a file)
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails.
    pub async fn save_to_dir(
        dir: &Path,
        data: &[u8],
        format: OutputFormat,
        resolution: MediaResolution,
    ) -> std::io::Result<CommandStream> {
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
    ) -> std::io::Result<CommandStream> {
        if data.is_empty() {
            return Ok(empty_stream());
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
                let (url, path) = create_file(dir, &prepared.bytes, format).await?;
                Ok(CommandStream::Image {
                    media_type: prepared.media_type,
                    file: stored_file(url, &path),
                })
            }
            OutputFormat::Video | OutputFormat::Binary => {
                // Binary/video are always Stored
                save_large_output(dir, data, format).await
            }
        }
    }

    /// Retrieves an output reference by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&OutputRef> {
        self.entries.get(id)
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
            if let Some(filepath) = stored_path(&output_ref.stream)
                && let Err(e) = fs::remove_file(&filepath).await
            {
                tracing::warn!(path = %filepath.display(), error = %e, "failed to remove output file");
            }
        }
        Ok(())
    }
}

/// Saves raw data to a file without any processing, returning its URL and
/// absolute path.
///
/// Used to preserve the original of a lossy compression stage alongside the
/// reduced version the model reads.
///
/// # Errors
///
/// Returns an error if file creation fails.
pub async fn save_raw_to_file(dir: &Path, data: &[u8]) -> std::io::Result<(String, PathBuf)> {
    let (url, path) = create_file(dir, data, OutputFormat::Text).await?;
    debug!(url = %url, size = data.len(), "saved raw output");
    Ok((url, path))
}

/// Saves a binary payload into `dir` under a four-random-words name, with an
/// image extension when `mime` is an image the bytes confirm.
///
/// Returns the relative URL and the absolute path of the file.
///
/// # Errors
///
/// Returns an error if file creation fails.
pub async fn save_binary_to_file(
    dir: &Path,
    data: &[u8],
    mime: &str,
) -> std::io::Result<(String, PathBuf)> {
    let format = if mime.starts_with("image/") {
        OutputFormat::Image
    } else {
        OutputFormat::Binary
    };
    create_file(dir, data, format).await
}

/// Saves text output with simple size-based classification.
///
/// - Below `INLINE_OUTPUT_LIMIT` → show inline
/// - Above `INLINE_OUTPUT_LIMIT` → save to file, return a head+tail preview plus file reference
async fn save_text_output(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<CommandStream> {
    if data.len() <= INLINE_OUTPUT_LIMIT {
        Ok(CommandStream::Inline {
            text: String::from_utf8_lossy(data).into_owned(),
        })
    } else {
        save_large_output(dir, data, format).await
    }
}

/// Saves text output with a line-count limit.
///
/// When the output exceeds `max_lines` (or the byte limit), the full content
/// is saved to a file and the returned content carries a head+tail preview
/// within the caller's budget plus the file reference and read guidance.
pub async fn save_text_with_line_limit(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    byte_limit: Option<usize>,
) -> std::io::Result<CommandStream> {
    if data.is_empty() {
        return Ok(empty_stream());
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
        return Ok(CommandStream::Inline {
            text: text.into_owned(),
        });
    }

    // Cut: store the full output and hand the model a preview within its
    // requested budget so it never has to re-read blind, unless the preview
    // would not be smaller than the output.
    let byte_budget = byte_limit.unwrap_or(INLINE_OUTPUT_LIMIT);
    cut_text_output(
        dir,
        format,
        CutText {
            data,
            text: &text,
            max_lines,
            byte_budget,
        },
    )
    .await
}

/// Saves output that exceeds limit - stores full content, returning a
/// preview plus the file reference for text formats.
async fn save_large_output(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<CommandStream> {
    if matches!(format, OutputFormat::Text | OutputFormat::Auto) {
        let text = String::from_utf8_lossy(data);
        return cut_text_output(
            dir,
            format,
            CutText {
                data,
                text: &text,
                max_lines: usize::MAX,
                byte_budget: INLINE_OUTPUT_LIMIT,
            },
        )
        .await;
    }

    // Binary/video/image just get the file path.
    let (url, path) = create_file(dir, data, format).await?;
    Ok(CommandStream::Stored {
        file: stored_file(url, &path),
        preview: String::new(),
        truncated: false,
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

/// Picks the four-random-words name a file for `data` will have, without
/// writing it.
fn allocate_file(dir: &Path, data: &[u8], format: OutputFormat) -> (String, PathBuf) {
    let ext = extension_for_data(data, format);
    let name = generate_word_filename();
    let filename = format!("{name}.{ext}");
    (format!("outputs/{filename}"), dir.join(&filename))
}

/// Creates a file with a four-random-words name.
async fn create_file(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<(String, PathBuf)> {
    let (url, filepath) = allocate_file(dir, data, format);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fmt::Write, io::Cursor};

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
    fn test_generate_word_filename() {
        let name = generate_word_filename();
        assert_eq!(name.split('-').count(), 4, "Should have 4 words: {name}");
    }

    #[tokio::test]
    async fn images_are_always_stored_downscaled_to_the_requested_resolution() {
        let dir = tempfile::tempdir().unwrap();
        let jpeg = encode_test_image(1200, 900, ImageFormat::Jpeg);

        let stream = OutputStore::save_to_dir_with_limit(
            dir.path(),
            &jpeg,
            OutputFormat::Image,
            MediaResolution::Low,
            None,
        )
        .await
        .unwrap();

        let CommandStream::Image { media_type, file } = stream else {
            panic!("expected a stored image");
        };
        assert_eq!(media_type, "image/jpeg");
        assert!(
            Path::new(&file.path)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("jpg")
                    || extension.eq_ignore_ascii_case("jpeg"))
        );
        assert!(
            dir.path()
                .join(file.url.trim_start_matches("outputs/"))
                .exists()
        );

        let stored = ImageReader::open(&file.path)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        assert!(stored.width() <= 512);
        assert!(stored.height() <= 512);
    }

    #[tokio::test]
    async fn small_images_are_stored_too() {
        let dir = tempfile::tempdir().unwrap();
        let png = encode_test_image(8, 8, ImageFormat::Png);

        let stream =
            OutputStore::save_to_dir(dir.path(), &png, OutputFormat::Auto, MediaResolution::Auto)
                .await
                .unwrap();

        let CommandStream::Image { media_type, file } = stream else {
            panic!("expected a stored image");
        };
        assert_eq!(media_type, "image/png");
        assert_eq!(std::fs::read(&file.path).unwrap(), png);
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

        let rendered = entry.render_for_model();
        let CommandStream::Stored {
            preview: text,
            truncated,
            file,
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
            rendered.contains(&file.url) && rendered.contains(&file.path),
            "guidance must reference the saved file: {rendered}"
        );
        assert!(rendered.contains("Use head/tail/grep/sed"));
    }

    #[tokio::test]
    async fn stored_text_output_preview_honors_max_lines() {
        let dir = tempfile::tempdir().unwrap();
        let pad = "p".repeat(400);
        let data = format!("alpha{pad}\nbeta{pad}\ngamma{pad}\ndelta{pad}\n").repeat(2);

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

        let CommandStream::Stored { preview: text, .. } = entry else {
            panic!("expected stored entry with text content");
        };
        // Eight lines, a budget of three: one for the head, two for the tail.
        assert!(
            text.starts_with(&format!("alpha{pad}\n[... lines 2-6 of 8 omitted")),
            "preview must stop at max_lines: {text}"
        );
        assert!(
            text.ends_with(&format!("gamma{pad}\ndelta{pad}")),
            "tail must close the preview: {text}"
        );
    }

    #[tokio::test]
    async fn stored_preview_shows_the_failure_at_the_tail_with_matching_line_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let mut data = String::new();
        for index in 0..4000 {
            writeln!(data, "test suite::case_{index} ... ok").expect("write to String");
        }
        data.push_str("test suite::case_4000 ... FAILED\n");
        data.push_str("thread 'suite::case_4000' panicked at sandbox/src/lib.rs:42:5:\n");
        data.push_str("assertion `left == right` failed\n");
        data.push_str("error: test failed\n");
        let total_lines = data.lines().count();

        let entry = save_text_with_line_limit(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
            usize::MAX,
            Some(INLINE_OUTPUT_LIMIT),
        )
        .await
        .unwrap();

        let CommandStream::Stored {
            preview: text,
            file,
            ..
        } = entry
        else {
            panic!("expected stored entry with text content");
        };
        let path = file.path;

        // The failure is in the preview, at its end.
        assert!(text.contains("error: test failed"), "{text}");
        assert!(text.contains("FAILED"), "{text}");

        // The omitted-range marker's line numbers address the stored file.
        let marker = text
            .lines()
            .find(|line| line.starts_with("[... lines"))
            .expect("omitted range marker");
        let range = marker.split_whitespace().nth(2).expect("range");
        let (first, last) = range.split_once('-').expect("range bounds");
        let first: usize = first.parse().expect("first line number");
        let last: usize = last.parse().expect("last line number");
        assert!(
            marker.contains(&format!("of {total_lines} omitted")),
            "{marker}"
        );

        let stored = std::fs::read_to_string(&path).expect("stored file is readable");
        let stored_lines: Vec<&str> = stored.lines().collect();
        assert_eq!(stored_lines.len(), total_lines);

        let preview_lines: Vec<&str> = text.lines().collect();
        let marker_index = preview_lines
            .iter()
            .position(|line| line.starts_with("[... lines"))
            .expect("marker index");
        // Line `first - 1` of the stored file is the last head line, and line
        // `last + 1` is the first tail line.
        assert_eq!(stored_lines[first - 2], preview_lines[marker_index - 1]);
        assert_eq!(stored_lines[last], preview_lines[marker_index + 1]);
        assert_eq!(stored_lines[total_lines - 1], "error: test failed");
    }

    fn stored_file_count(dir: &Path) -> usize {
        std::fs::read_dir(dir).expect("read dir").count()
    }

    #[tokio::test]
    async fn output_just_over_max_lines_is_returned_inline_when_the_cut_is_larger() {
        let dir = tempfile::tempdir().unwrap();
        let data = (0..212).fold(String::new(), |mut data, i| {
            use std::fmt::Write as _;
            writeln!(data, "ok {i}").expect("writing to a String cannot fail");
            data
        });

        let entry = save_text_with_line_limit(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
            200,
            Some(INLINE_OUTPUT_LIMIT),
        )
        .await
        .unwrap();

        let CommandStream::Inline { text } = entry else {
            panic!("expected the original inline");
        };
        assert_eq!(text, data);
        assert_eq!(stored_file_count(dir.path()), 0, "nothing may be stored");
    }

    #[tokio::test]
    async fn large_output_that_shrinks_is_stored_with_its_preview() {
        let dir = tempfile::tempdir().unwrap();
        let data = "a reasonably long line of command output text\n".repeat(1000);

        let entry = save_text_with_line_limit(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
            200,
            Some(INLINE_OUTPUT_LIMIT),
        )
        .await
        .unwrap();

        let CommandStream::Stored {
            preview: text,
            truncated,
            file,
        } = entry
        else {
            panic!("expected stored entry with text content");
        };
        assert!(truncated);
        assert!(text.len() < data.len());
        assert_eq!(std::fs::read_to_string(file.path).unwrap(), data);
        assert_eq!(stored_file_count(dir.path()), 1);
    }

    #[tokio::test]
    async fn oversized_single_line_is_stored_only_when_the_cut_is_smaller() {
        let dir = tempfile::tempdir().unwrap();
        let data = "x".repeat(INLINE_OUTPUT_LIMIT * 2);

        let entry = OutputStore::save_to_dir(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
        )
        .await
        .unwrap();

        let CommandStream::Stored { preview: text, .. } = entry else {
            panic!("expected stored entry with text content");
        };
        assert!(text.len() < data.len());
    }

    #[cfg(feature = "code-folding")]
    fn rust_source(functions: usize) -> String {
        let mut source = String::new();
        for index in 0..functions {
            writeln!(source, "fn compute_{index}(input: u64) -> u64 {{").unwrap();
            for step in 0..8 {
                writeln!(
                    source,
                    "    let step_{step} = input.wrapping_mul({step} + 3);"
                )
                .unwrap();
            }
            writeln!(source, "    input + step_7\n}}\n").unwrap();
        }
        source
    }

    #[cfg(feature = "code-folding")]
    #[tokio::test]
    async fn source_code_over_the_limit_is_stored_with_a_folded_preview() {
        let dir = tempfile::tempdir().unwrap();
        let data = rust_source(40);
        assert!(data.lines().count() > 200);

        let entry = save_text_with_line_limit(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
            200,
            Some(INLINE_OUTPUT_LIMIT),
        )
        .await
        .unwrap();

        let CommandStream::Stored {
            preview: text,
            file,
            ..
        } = entry
        else {
            panic!("expected stored entry with a folded preview");
        };
        let path = file.path;
        assert!(text.contains("fn compute_0"), "{text}");
        assert!(text.contains("Code folded"), "{text}");
        assert!(!text.contains("step_3 ="), "bodies must be folded: {text}");
        assert!(text.len() < data.len());
        assert_eq!(std::fs::read_to_string(path).unwrap(), data);
    }

    #[cfg(feature = "code-folding")]
    #[tokio::test]
    async fn small_source_code_is_left_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let data = rust_source(3);

        let entry = save_text_with_line_limit(
            dir.path(),
            data.as_bytes(),
            OutputFormat::Text,
            MediaResolution::Auto,
            200,
            Some(INLINE_OUTPUT_LIMIT),
        )
        .await
        .unwrap();

        let CommandStream::Inline { text } = entry else {
            panic!("expected the source inline");
        };
        assert_eq!(text, data);
        assert_eq!(stored_file_count(dir.path()), 0);
    }
}
