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

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};
use async_fs as fs;
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
    /// No output (ToolOutput::Done) - nothing to store
    Empty,

    /// Super tiny content - always in context, NEVER gets URL.
    /// Examples: "done", "3 files deleted", short status messages.
    Inline { content: Content },

    /// Content is loaded in context, no file yet.
    /// URL will be generated when offloaded.
    Loaded {
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
        /// Summary for large outputs
        summary: Option<String>,
    },
}

impl Serialize for OutputEntry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            OutputEntry::Empty => {
                let map = serializer.serialize_map(Some(0))?;
                map.end()
            }
            OutputEntry::Inline { content } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            OutputEntry::Loaded { content, .. } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            OutputEntry::Stored { url, content, summary } => {
                let mut count = 1; // url is always present
                if content.is_some() {
                    count += 1;
                }
                if summary.is_some() {
                    count += 1;
                }
                let mut map = serializer.serialize_map(Some(count))?;
                map.serialize_entry("url", url)?;
                if let Some(c) = content {
                    map.serialize_entry("content", c)?;
                }
                if let Some(s) = summary {
                    map.serialize_entry("summary", s)?;
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
            summary: Option<String>,
        }

        let h = Helper::deserialize(deserializer)?;

        Ok(match (h.url, h.content) {
            (Some(url), content) => OutputEntry::Stored { url, content, summary: h.summary },
            (None, Some(content)) => OutputEntry::Inline { content },
            (None, None) => OutputEntry::Empty,
        })
    }
}

impl std::fmt::Display for OutputEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputEntry::Empty => Ok(()),
            OutputEntry::Inline { content } | OutputEntry::Loaded { content, .. } => {
                match content {
                    Content::Text { text, truncated } => {
                        write!(f, "{text}")?;
                        if *truncated {
                            write!(f, "\n[truncated]")?;
                        }
                        Ok(())
                    }
                    Content::Image { media_type, .. } => {
                        write!(f, "[Image: {media_type}]")
                    }
                }
            }
            OutputEntry::Stored { url, content, summary } => {
                if let Some(content) = content {
                    match content {
                        Content::Text { text, truncated } => {
                            write!(f, "{text}")?;
                            if *truncated {
                                write!(f, "\n[truncated, full content at {url}]")?;
                            }
                        }
                        Content::Image { media_type, .. } => {
                            write!(f, "[Image: {media_type}] at {url}")?;
                        }
                    }
                } else if let Some(summary) = summary {
                    write!(f, "{summary}\n[full content at {url}]")?;
                } else {
                    write!(f, "[content at {url}]")?;
                }
                Ok(())
            }
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

/// Maximum lines for tiny text (never gets URL).
const MAX_INLINE_LINES: usize = 5;

/// Maximum lines for small text (lazy URL on offload).
const MAX_LOADED_LINES: usize = 500;

/// Preview lines for truncated large text.
const PREVIEW_LINES: usize = 50;

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

        let format = if format == OutputFormat::Auto { detect_format(data) } else { format };
        let output_ref = OutputRef {
            entry: entry.clone(),
            format,
            size: data.len(),
        };
        self.entries.insert(id, output_ref);

        Ok(entry)
    }

    /// Saves output data to a directory without tracking.
    ///
    /// Determines the appropriate `OutputEntry` variant based on content:
    /// - Empty data → `Empty`
    /// - Super tiny text (< 5 lines) → `Inline`
    /// - Small text (< 500 lines) or images → `Loaded`
    /// - Large text, binary, video → `Stored` (file created immediately)
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails.
    pub async fn save_to_dir(
        dir: &Path,
        data: &[u8],
        format: OutputFormat,
    ) -> std::io::Result<OutputEntry> {
        if data.is_empty() {
            return Ok(OutputEntry::Empty);
        }

        let format = if format == OutputFormat::Auto {
            detect_format(data)
        } else {
            format
        };

        match format {
            OutputFormat::Text | OutputFormat::Auto => {
                save_text_output(dir, data, format).await
            }
            OutputFormat::Image => {
                // Images are always Loaded (lazy URL)
                let media_type = detect_image_media_type(data);
                let content = Content::Image {
                    data: base64_encode(data),
                    media_type,
                };
                Ok(OutputEntry::Loaded {
                    content,
                    raw: data.to_vec(),
                    format,
                })
            }
            OutputFormat::Video | OutputFormat::Binary => {
                // Binary/video are always Stored immediately
                let (url, _) = create_file(dir, data, format).await?;
                let summary = format!(
                    "{} data ({} bytes)",
                    if format == OutputFormat::Video { "Video" } else { "Binary" },
                    data.len()
                );
                Ok(OutputEntry::Stored {
                    url,
                    content: None,
                    summary: Some(summary),
                })
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
            OutputEntry::Loaded { content, raw, format } => {
                let (url, _) = create_file(&self.dir, raw, *format).await?;
                Ok(OutputEntry::Stored {
                    url,
                    content: Some(content.clone()),
                    summary: None,
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
        let filename = pending
            .url
            .strip_prefix("outputs/")
            .unwrap_or(&pending.url);
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

/// Saves text output with appropriate classification.
async fn save_text_output(
    dir: &Path,
    data: &[u8],
    format: OutputFormat,
) -> std::io::Result<OutputEntry> {
    let text = String::from_utf8_lossy(data);
    let lines: Vec<&str> = text.lines().collect();
    let line_count = lines.len();

    if line_count <= MAX_INLINE_LINES {
        // Super tiny - Inline, never gets URL
        let content = Content::Text {
            text: text.into_owned(),
            truncated: false,
        };
        Ok(OutputEntry::Inline { content })
    } else if line_count <= MAX_LOADED_LINES {
        // Small - Loaded, lazy URL on offload
        let content = Content::Text {
            text: text.into_owned(),
            truncated: false,
        };
        Ok(OutputEntry::Loaded {
            content,
            raw: data.to_vec(),
            format,
        })
    } else {
        // Large - Stored immediately with preview
        let (url, _) = create_file(dir, data, format).await?;
        let preview: String = lines[..PREVIEW_LINES].join("\n");
        let content = Content::Text {
            text: format!(
                "{preview}\n\n... truncated ({} more lines). Use `ask` or read the file to see more.",
                line_count - PREVIEW_LINES
            ),
            truncated: true,
        };
        let summary = format!("Text output with {line_count} lines ({} bytes)", data.len());
        Ok(OutputEntry::Stored {
            url,
            content: Some(content),
            summary: Some(summary),
        })
    }
}

/// Creates a file with a four-random-words name.
async fn create_file(dir: &Path, data: &[u8], format: OutputFormat) -> std::io::Result<(String, PathBuf)> {
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
fn generate_word_filename() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Simple word list - common adjectives and nouns
    const ADJECTIVES: &[&str] = &[
        "amber", "azure", "bold", "brave", "bright", "calm", "clear", "cool",
        "coral", "crisp", "dark", "deep", "dry", "dusk", "fair", "fast",
        "fine", "firm", "free", "fresh", "frost", "glad", "gold", "good",
        "gray", "green", "happy", "hard", "high", "jade", "keen", "kind",
        "late", "lean", "light", "live", "long", "loud", "mild", "neat",
        "new", "next", "nice", "old", "open", "pale", "pink", "plain",
        "proud", "pure", "quick", "quiet", "rare", "red", "rich", "ripe",
        "rough", "ruby", "safe", "sage", "sharp", "short", "shy", "silk",
        "slim", "slow", "small", "smart", "smooth", "snow", "soft", "solid",
        "stark", "still", "stone", "strong", "sweet", "swift", "tall", "teal",
        "thick", "thin", "tight", "warm", "west", "white", "wide", "wild",
        "wise", "young", "zero",
    ];

    const NOUNS: &[&str] = &[
        "acorn", "arch", "aria", "ash", "aurora", "bay", "beam", "bear",
        "bell", "bird", "blade", "bloom", "bolt", "book", "bowl", "breeze",
        "brook", "bud", "cape", "cave", "chill", "cliff", "cloud", "coast",
        "comet", "coral", "cove", "crane", "creek", "crest", "crow", "dawn",
        "deer", "delta", "dew", "dove", "dream", "drift", "dune", "dust",
        "eagle", "earth", "echo", "edge", "elm", "ember", "fawn", "fern",
        "field", "finch", "fire", "flame", "flash", "flight", "flint", "flood",
        "flow", "foam", "fog", "forge", "frost", "gate", "gaze", "glade",
        "gleam", "glen", "glow", "gorge", "grain", "grass", "grove", "gust",
        "hail", "harbor", "hawk", "haze", "heath", "helm", "hill", "hollow",
        "hope", "horn", "hush", "ice", "inlet", "isle", "ivy", "jade",
        "jay", "jet", "lake", "lark", "leaf", "light", "lily", "lion",
        "marsh", "mead", "mist", "moon", "moss", "moth", "mount", "muse",
        "nest", "night", "north", "oak", "oasis", "ocean", "orchid", "owl",
        "palm", "pass", "path", "peak", "pearl", "petal", "pine", "plain",
        "plum", "pond", "pool", "rain", "rapids", "raven", "ray", "reef",
        "ridge", "ring", "rise", "river", "road", "rock", "root", "rose",
        "sage", "sand", "sea", "seed", "shade", "shadow", "shell", "shore",
        "silk", "sky", "slate", "slope", "snow", "song", "south", "spark",
        "spire", "spray", "spring", "spruce", "star", "steam", "steel", "stem",
        "stone", "storm", "strait", "stream", "sun", "surf", "swan", "swift",
        "thorn", "thyme", "tide", "trail", "tree", "vale", "valley", "vapor",
        "vine", "vista", "void", "wave", "well", "wheat", "willow", "wind",
        "wing", "winter", "wolf", "wood", "wren", "yarn", "yew", "zen",
    ];

    // Use time-based seed + counter for randomness without external deps
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0) as u64;

    // Simple xorshift for pseudo-random selection
    let mut rng = seed;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let w1 = ADJECTIVES[(next() as usize) % ADJECTIVES.len()];
    let w2 = NOUNS[(next() as usize) % NOUNS.len()];
    let w3 = ADJECTIVES[(next() as usize) % ADJECTIVES.len()];
    let w4 = NOUNS[(next() as usize) % NOUNS.len()];

    format!("{w1}-{w2}-{w3}-{w4}")
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
        let is_text = text.chars().all(|c| {
            c == '\n' || c == '\r' || c == '\t' || (c >= ' ' && c != '\x7f')
        });
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
        assert_eq!(detect_format(&[0x00, 0x01, 0x02, 0x03]), OutputFormat::Binary);
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
            summary: Some("100 lines".to_string()),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"url\""));
        assert!(json.contains("\"content\""));
        assert!(json.contains("\"summary\""));
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
