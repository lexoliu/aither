//! Sentence-based text chunking.

use unicode_segmentation::UnicodeSegmentation;

use crate::dedup::content_hash;
use crate::error::Result;
use crate::types::{Chunk, Document};

use super::Chunker;

/// Chunks text by sentence boundaries.
///
/// This chunker groups sentences together until the maximum chunk size is
/// reached, ensuring that chunks don't break in the middle of sentences.
///
/// # Example
///
/// ```rust
/// use aither_rag::chunking::{Chunker, SentenceChunker};
/// use aither_rag::Document;
///
/// let chunker = SentenceChunker::new(500);
/// let doc = Document::new("doc1", "First sentence. Second sentence. Third sentence.");
/// let chunks = chunker.chunk(&doc).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    /// Maximum size of each chunk in characters.
    max_chunk_size: usize,
}

impl SentenceChunker {
    /// Creates a new sentence chunker.
    ///
    /// # Arguments
    /// * `max_chunk_size` - Maximum characters per chunk
    #[must_use]
    pub const fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Creates a chunker with default settings (512 chars).
    #[must_use]
    pub const fn default_settings() -> Self {
        Self::new(512)
    }
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::default_settings()
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        let text = &doc.text;

        // If text fits in one chunk, return it as-is
        if text.len() <= self.max_chunk_size {
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

        let sentences: Vec<&str> = text.unicode_sentences().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_idx = 0;
        let mut chunk_start = 0;

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            // If adding this sentence would exceed max size
            if !current_chunk.is_empty()
                && current_chunk.len() + sentence.len() + 1 > self.max_chunk_size
            {
                // Save current chunk
                let hash = content_hash(&current_chunk);
                let mut metadata = doc.metadata.clone();
                metadata.insert("chunk_start".into(), chunk_start.to_string());
                metadata.insert(
                    "chunk_end".into(),
                    (chunk_start + current_chunk.len()).to_string(),
                );

                chunks.push(Chunk::with_metadata(
                    format!("{}#chunk_{}", doc.id, chunk_idx),
                    current_chunk.clone(),
                    &doc.id,
                    chunk_idx,
                    hash,
                    metadata,
                ));

                chunk_start += current_chunk.len() + 1;
                chunk_idx += 1;
                current_chunk.clear();
            }

            // Add sentence to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            let hash = content_hash(&current_chunk);
            let mut metadata = doc.metadata.clone();
            metadata.insert("chunk_start".into(), chunk_start.to_string());
            metadata.insert(
                "chunk_end".into(),
                (chunk_start + current_chunk.len()).to_string(),
            );

            chunks.push(Chunk::with_metadata(
                format!("{}#chunk_{}", doc.id, chunk_idx),
                current_chunk,
                &doc.id,
                chunk_idx,
                hash,
                metadata,
            ));
        }

        Ok(chunks)
    }

    fn name(&self) -> &'static str {
        "sentence"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_text_single_chunk() {
        let chunker = SentenceChunker::new(500);
        let doc = Document::new("doc1", "Short sentence.");
        let chunks = chunker.chunk(&doc).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "doc1#chunk_0");
        assert_eq!(chunks[0].text, "Short sentence.");
    }

    #[test]
    fn multiple_sentences_split() {
        let chunker = SentenceChunker::new(50);
        let doc = Document::new(
            "doc1",
            "First sentence here. Second sentence here. Third sentence here.",
        );
        let chunks = chunker.chunk(&doc).unwrap();

        assert!(chunks.len() > 1);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.source_id, "doc1");
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn sentences_grouped_within_limit() {
        let chunker = SentenceChunker::new(100);
        let doc = Document::new("doc1", "Short. Also short. Still short.");
        let chunks = chunker.chunk(&doc).unwrap();

        // All sentences should fit in one chunk
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn default_settings() {
        let chunker = SentenceChunker::default();
        assert_eq!(chunker.max_chunk_size, 512);
    }
}
