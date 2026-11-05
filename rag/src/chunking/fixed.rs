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

        // Every position below counts characters, so every cut and every
        // overlap start is a character boundary by construction.
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let total = chars.len();
        let byte_at = |position: usize| chars.get(position).map_or(text.len(), |&(byte, _)| byte);

        // If text fits in one chunk, return it as-is
        if total <= self.chunk_size {
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

        loop {
            let end = (start + self.chunk_size).min(total);

            // Prefer cutting at whitespace, but only where the next chunk
            // (which starts `overlap` characters earlier) still begins after
            // this one; otherwise cut hard at `end`, which always does since
            // `overlap < chunk_size`.
            let actual_end = if end < total {
                chars[start..end]
                    .iter()
                    .rposition(|&(_, c)| c.is_whitespace())
                    .map(|pos| start + pos)
                    .filter(|&cut| cut - start > self.overlap)
                    .unwrap_or(end)
            } else {
                end
            };

            let chunk_text = text[byte_at(start)..byte_at(actual_end)].trim();
            if !chunk_text.is_empty() {
                let hash = content_hash(chunk_text);
                let mut metadata = doc.metadata.clone();
                metadata.insert("chunk_start".into(), byte_at(start).to_string());
                metadata.insert("chunk_end".into(), byte_at(actual_end).to_string());

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

            if actual_end == total {
                break;
            }
            // Strictly greater than the previous start: actual_end - start > overlap.
            start = actual_end - self.overlap;
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
