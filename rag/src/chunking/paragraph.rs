//! Paragraph-based text chunking.

use crate::dedup::content_hash;
use crate::error::Result;
use crate::types::{Chunk, Document};

use super::Chunker;

/// Chunks text by paragraph boundaries (blank lines).
#[derive(Debug, Clone)]
pub struct ParagraphChunker {
    max_chunk_size: usize,
}

impl ParagraphChunker {
    /// Creates a paragraph chunker with a maximum chunk size in bytes.
    #[must_use]
    pub const fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Creates a chunker with default settings (1024 bytes).
    #[must_use]
    pub const fn default_settings() -> Self {
        Self::new(1024)
    }

    fn split_large_paragraph<'a>(&self, paragraph: &'a str) -> Vec<&'a str> {
        if paragraph.len() <= self.max_chunk_size {
            return vec![paragraph];
        }

        let mut pieces = Vec::new();
        let mut start = 0usize;

        while start < paragraph.len() {
            let mut end = (start + self.max_chunk_size).min(paragraph.len());

            if end < paragraph.len() {
                // Prefer a whitespace split inside the window for cleaner boundaries.
                if let Some(rel) = paragraph[start..end].rfind(char::is_whitespace) {
                    let candidate = start + rel;
                    if candidate > start {
                        end = candidate;
                    }
                }
            }

            if end <= start {
                break;
            }

            let piece = paragraph[start..end].trim();
            if !piece.is_empty() {
                pieces.push(piece);
            }
            start = end;
        }

        pieces
    }

    fn paragraphs<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        for block in text.split("\n\n") {
            let paragraph = block.trim();
            if paragraph.is_empty() {
                continue;
            }
            let split = self.split_large_paragraph(paragraph);
            out.extend(split);
        }
        out
    }
}

impl Default for ParagraphChunker {
    fn default() -> Self {
        Self::default_settings()
    }
}

impl Chunker for ParagraphChunker {
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        let text = doc.text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        if text.len() <= self.max_chunk_size {
            let hash = content_hash(text);
            return Ok(vec![Chunk::with_metadata(
                format!("{}#chunk_0", doc.id),
                text.to_string(),
                &doc.id,
                0,
                hash,
                doc.metadata.clone(),
            )]);
        }

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut idx = 0usize;

        for paragraph in self.paragraphs(text) {
            if current.is_empty() {
                current.push_str(paragraph);
                continue;
            }

            if current.len() + 2 + paragraph.len() > self.max_chunk_size {
                let hash = content_hash(&current);
                chunks.push(Chunk::with_metadata(
                    format!("{}#chunk_{}", doc.id, idx),
                    current.clone(),
                    &doc.id,
                    idx,
                    hash,
                    doc.metadata.clone(),
                ));
                idx += 1;
                current.clear();
                current.push_str(paragraph);
            } else {
                current.push_str("\n\n");
                current.push_str(paragraph);
            }
        }

        if !current.is_empty() {
            let hash = content_hash(&current);
            chunks.push(Chunk::with_metadata(
                format!("{}#chunk_{}", doc.id, idx),
                current,
                &doc.id,
                idx,
                hash,
                doc.metadata.clone(),
            ));
        }

        Ok(chunks)
    }

    fn name(&self) -> &'static str {
        "paragraph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_on_paragraph_boundaries() {
        let chunker = ParagraphChunker::new(80);
        let doc = Document::new("doc", "a\n\n b\n\n c");
        let chunks = chunker.chunk(&doc).unwrap();

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("a"));
        assert!(chunks[0].text.contains("b"));
        assert!(chunks[0].text.contains("c"));
    }

    #[test]
    fn large_paragraph_gets_split() {
        let chunker = ParagraphChunker::new(16);
        let doc = Document::new("doc", "abcdefghijklmnopqrstuvwxyz");
        let chunks = chunker.chunk(&doc).unwrap();

        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| !c.text.is_empty()));
    }
}
