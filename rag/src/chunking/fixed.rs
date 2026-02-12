//! Fixed-size text chunking.

use crate::dedup::content_hash;
use crate::error::Result;
use crate::types::{Chunk, Document};

use super::Chunker;

/// Chunks text into fixed-size pieces with configurable overlap.
///
/// This chunker splits text by character count, respecting word boundaries
/// where possible. Overlap ensures context is preserved across chunk boundaries.
///
/// # Example
///
/// ```rust
/// use aither_rag::chunking::{Chunker, FixedSizeChunker};
/// use aither_rag::Document;
///
/// let chunker = FixedSizeChunker::new(100, 20);
/// let doc = Document::new("doc1", "Long text content...");
/// let chunks = chunker.chunk(&doc).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FixedSizeChunker {
    /// Maximum size of each chunk in characters.
    chunk_size: usize,
    /// Number of overlapping characters between consecutive chunks.
    overlap: usize,
}

impl FixedSizeChunker {
    /// Creates a new fixed-size chunker.
    ///
    /// # Arguments
    /// * `chunk_size` - Maximum characters per chunk
    /// * `overlap` - Characters to overlap between consecutive chunks
    ///
    /// # Panics
    /// Panics if `overlap >= chunk_size`.
    #[must_use]
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        assert!(
            overlap < chunk_size,
            "overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        );
        Self {
            chunk_size,
            overlap,
        }
    }

    /// Creates a chunker with default settings (512 chars, 64 overlap).
    #[must_use]
    pub fn default_settings() -> Self {
        Self::new(512, 64)
    }
}

impl Default for FixedSizeChunker {
    fn default() -> Self {
        Self::default_settings()
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        let text = &doc.text;

        // If text fits in one chunk, return it as-is
        if text.len() <= self.chunk_size {
            let hash = content_hash(text);
            return Ok(vec![Chunk::with_metadata(
                format!("{}#chunk_0", doc.id),
                text.clone(),
                &doc.id,
                0,
                hash,
                doc.metadata.clone(),
            )]);
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_idx = 0;
        let step = self.chunk_size - self.overlap;

        while start < text.len() {
            let end = (start + self.chunk_size).min(text.len());

            // Find word boundary for cleaner cuts (look for whitespace before end)
            let actual_end = if end < text.len() {
                text[start..end]
                    .rfind(char::is_whitespace)
                    .map_or(end, |pos| start + pos)
            } else {
                end
            };

            // Don't create tiny trailing chunks
            if actual_end <= start {
                break;
            }

            let chunk_text = text[start..actual_end].trim();
            if !chunk_text.is_empty() {
                let hash = content_hash(chunk_text);
                let mut metadata = doc.metadata.clone();
                metadata.insert("chunk_start".into(), start.to_string());
                metadata.insert("chunk_end".into(), actual_end.to_string());

                chunks.push(Chunk::with_metadata(
                    format!("{}#chunk_{}", doc.id, chunk_idx),
                    chunk_text.to_string(),
                    &doc.id,
                    chunk_idx,
                    hash,
                    metadata,
                ));
                chunk_idx += 1;
            }

            // Move forward, accounting for overlap
            start = if actual_end > self.overlap {
                actual_end.saturating_sub(self.overlap)
            } else {
                actual_end
            };

            // Prevent infinite loop if we can't make progress
            if start >= text.len() || (actual_end == text.len() && start + step >= text.len()) {
                break;
            }
        }

        Ok(chunks)
    }

    fn name(&self) -> &'static str {
        "fixed_size"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_text_single_chunk() {
        let chunker = FixedSizeChunker::new(100, 20);
        let doc = Document::new("doc1", "Short text");
        let chunks = chunker.chunk(&doc).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "doc1#chunk_0");
        assert_eq!(chunks[0].text, "Short text");
        assert_eq!(chunks[0].source_id, "doc1");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn text_splits_into_multiple_chunks() {
        let chunker = FixedSizeChunker::new(50, 10);
        let doc = Document::new(
            "doc1",
            "This is a longer text that should be split into multiple chunks for testing purposes.",
        );
        let chunks = chunker.chunk(&doc).unwrap();

        assert!(chunks.len() > 1);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.id, format!("doc1#chunk_{i}"));
            assert_eq!(chunk.source_id, "doc1");
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn chunks_have_content_hash() {
        let chunker = FixedSizeChunker::new(100, 20);
        let doc = Document::new("doc1", "Some text content");
        let chunks = chunker.chunk(&doc).unwrap();

        assert_ne!(chunks[0].content_hash, 0);
    }

    #[test]
    fn metadata_preserved() {
        let chunker = FixedSizeChunker::new(100, 20);
        let mut metadata = crate::types::Metadata::new();
        metadata.insert("source".into(), "test".into());
        let doc = Document::with_metadata("doc1", "Some text", metadata);
        let chunks = chunker.chunk(&doc).unwrap();

        assert_eq!(chunks[0].metadata.get("source"), Some(&"test".to_string()));
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn overlap_must_be_less_than_chunk_size() {
        let _ = FixedSizeChunker::new(50, 50);
    }

    #[test]
    fn default_settings() {
        let chunker = FixedSizeChunker::default();
        assert_eq!(chunker.chunk_size, 512);
        assert_eq!(chunker.overlap, 64);
    }
}
