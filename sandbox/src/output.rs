//! Output management for bash tool execution.
//!
//! Outputs are classified by size and type:
//! - **Inline**: Super tiny text (< 5 lines) - always in context, never gets URL
//! - **Loaded**: Small text/images - in context, URL generated only on offload
//! - **Stored**: Large text/binary/video - file created immediately

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use async_fs as fs;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};
use tracing::debug;

/// Expected output format for bash execution.
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
    /// No output (`ToolOutput::Done`) - nothing to store
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
            Self::Inline { content } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            Self::Loaded { content, .. } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            Self::Stored { url, content } => {
                let count = if content.is_some() { 2 } else { 1 };
                let mut map = serializer.serialize_map(Some(count))?;
                map.serialize_entry("url", url)?;
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
            content: Option<Content>,
        }

        let h = Helper::deserialize(deserializer)?;

        Ok(match (h.url, h.content) {
            (Some(url), content) => Self::Stored { url, content },
            (None, Some(content)) => Self::Inline { content },
            (None, None) => Self::Empty,
        })
    }
}

impl std::fmt::Display for OutputEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => Ok(()),
            Self::Inline { content } | Self::Loaded { content, .. } => {
                match content {
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
                }
            }
            Self::Stored { url, content } => {
                if let Some(content) = content {
                    match content {
                        Content::Text { text, truncated } => {
                            write!(f, "{text}")?;
                            if *truncated {
                                write!(
                                    f,
                                    "\n[full content at {url}; output was truncated in-chat]"
                                )?;
                            }
                        }
                        Content::Image { media_type, .. } => {
                            write!(f, "[Image: {media_type}] at {url}")?;
                        }
                    }
                } else {
                    write!(f, "[content at {url}]")?;
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
            Self::Stored { url, .. } => {
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

/// Maximum output size to show inline (roughly what fits in a terminal without scrolling).
/// Outputs exceeding this are saved to file, and only the file path is returned.
/// This is the single source of truth for output size limits.
pub const INLINE_OUTPUT_LIMIT: usize = 4000;

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

/// Manages output storage for bash executions.
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
    ) -> std::io::Result<OutputEntry> {
        let entry = Self::save_to_dir(&self.dir, data, format).await?;

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
    ) -> std::io::Result<OutputEntry> {
        Self::save_to_dir_with_limit(dir, data, format, Some(INLINE_OUTPUT_LIMIT)).await
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

        // Check if we exceed limit - if so, always store as file
        if let Some(max) = limit {
            if data.len() > max {
                return save_large_output(dir, data, format).await;
            }
        }

        match format {
            OutputFormat::Text | OutputFormat::Auto => save_text_output(dir, data, format).await,
            OutputFormat::Image => {
                // Small images inline as base64
                if data.len() <= limit.unwrap_or(INLINE_OUTPUT_LIMIT) {
                    let media_type = detect_image_media_type(data);
                    let content = Content::Image {
                        data: base64_encode(data),
                        media_type,
                    };
                    Ok(OutputEntry::Inline { content })
                } else {
                    save_large_output(dir, data, format).await
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
                let (url, _) = create_file(&self.dir, raw, *format).await?;
                Ok(OutputEntry::Stored {
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
                let ext = format_extension(*format);
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
            if let OutputEntry::Stored { url, .. } = output_ref.entry {
                let filename = url.strip_prefix("outputs/").unwrap_or(&url);
                let filepath = self.dir.join(filename);
                if let Err(e) = fs::remove_file(&filepath).await {
                    tracing::warn!(path = %filepath.display(), error = %e, "failed to remove output file");
                }
            }
        }
        Ok(())
    }
}

/// Saves text output with simple size-based classification.
///
/// - Below `INLINE_OUTPUT_LIMIT` → show inline
/// - Above `INLINE_OUTPUT_LIMIT` → save to file, return only file reference
async fn save_text_output(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<OutputEntry> {
    let text = String::from_utf8_lossy(data);

    if data.len() <= INLINE_OUTPUT_LIMIT {
        // Small enough to show inline
        let content = Content::Text {
            text: text.into_owned(),
            truncated: false,
        };
        Ok(OutputEntry::Inline { content })
    } else {
        // Large - save to file, return only reference
        let (url, _) = create_file(dir, data, format).await?;
        let line_count = text.lines().count();
        let content = Content::Text {
            text: format!(
                "Output saved to {} ({} lines, {} bytes)",
                url,
                line_count,
                data.len()
            ),
            truncated: true,
        };
        Ok(OutputEntry::Stored {
            url,
            content: Some(content),
        })
    }
}

/// Saves output that exceeds limit - stores full content, returns only file reference.
async fn save_large_output(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<OutputEntry> {
    // Store full content to file
    let (url, _) = create_file(dir, data, format).await?;

    // Return just the file reference (no preview - LLM must read file like a human)
    let content = match format {
        OutputFormat::Text | OutputFormat::Auto => {
            let text = String::from_utf8_lossy(data);
            let line_count = text.lines().count();
            Some(Content::Text {
                text: format!(
                    "Output saved to {} ({} lines, {} bytes)",
                    url,
                    line_count,
                    data.len()
                ),
                truncated: true,
            })
        }
        _ => None, // Binary/video/image just get file path
    };

    Ok(OutputEntry::Stored { url, content })
}

/// Creates a file with a four-random-words name.
async fn create_file(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<(String, PathBuf)> {
    let ext = format_extension(format);
    let name = generate_word_filename();
    let filename = format!("{name}.{ext}");
    let url = format!("outputs/{filename}");
    let filepath = dir.join(&filename);

    fs::write(&filepath, data).await?;
    debug!(url = %url, size = data.len(), format = ?format, "saved output");

    Ok((url, filepath))
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
        OutputFormat::Image => "bin", // Generic, actual type detected from content
        OutputFormat::Video => "mp4",
        OutputFormat::Binary => "bin",
    }
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

/// Base64 encodes data using standard alphabet with padding.
fn base64_encode(data: &[u8]) -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let parts: Vec<&str> = name.split('-').collect();
        assert_eq!(parts.len(), 4, "Should have 4 words: {}", name);
    }
}
