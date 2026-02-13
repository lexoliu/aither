//! Tree-sitter based semantic code chunking.

use crate::dedup::content_hash;
use crate::error::{RagError, Result};
use crate::types::{Chunk, Document};

use super::{Chunker, ParagraphChunker};

/// Semantic code chunker using tree-sitter for Rust code.
///
/// For non-Rust files, this chunker falls back to paragraph chunking.
#[derive(Debug, Clone)]
pub struct CodeChunker {
    max_chunk_size: usize,
    fallback: ParagraphChunker,
}

impl CodeChunker {
    /// Creates a new code chunker.
    #[must_use]
    pub fn new(max_chunk_size: usize) -> Self {
        Self {
            max_chunk_size,
            fallback: ParagraphChunker::new(max_chunk_size),
        }
    }

    /// Creates a chunker with default settings (2048 bytes).
    #[must_use]
    pub fn default_settings() -> Self {
        Self::new(2048)
    }

    fn detect_rust(doc: &Document) -> bool {
        doc.metadata
            .get("path")
            .is_some_and(|path| path.ends_with(".rs"))
    }

    fn chunk_rust(&self, doc: &Document) -> Result<Vec<Chunk>> {
        use tree_sitter::Parser;

        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::language())
            .map_err(|e| RagError::Chunking(format!("tree-sitter set language failed: {e}")))?;

        let tree = parser
            .parse(&doc.text, None)
            .ok_or_else(|| RagError::Chunking("tree-sitter parse failed".to_string()))?;

        let root = tree.root_node();
        let bytes = doc.text.as_bytes();
        let mut chunks = Vec::new();
        let mut idx = 0usize;

        for node in root.named_children(&mut root.walk()) {
            if !Self::is_semantic_node(node.kind()) {
                continue;
            }
            let Ok(text) = node.utf8_text(bytes) else {
                continue;
            };
            let text = text.trim();
            if text.is_empty() {
                continue;
            }

            if text.len() <= self.max_chunk_size {
                chunks.push(Self::mk_chunk(doc, idx, text));
                idx += 1;
            } else {
                // Very large item: split with paragraph fallback while preserving semantic grouping.
                let sub_doc = Document::with_metadata(
                    format!("{}#semantic_{}", doc.id, idx),
                    text.to_string(),
                    doc.metadata.clone(),
                );
                let sub_chunks = self.fallback.chunk(&sub_doc)?;
                for sub in sub_chunks {
                    chunks.push(Self::mk_chunk(doc, idx, &sub.text));
                    idx += 1;
                }
            }
        }

        if chunks.is_empty() {
            return self.fallback.chunk(doc);
        }

        Ok(chunks)
    }

    fn mk_chunk(doc: &Document, index: usize, text: &str) -> Chunk {
        Chunk::with_metadata(
            format!("{}#chunk_{}", doc.id, index),
            text.to_string(),
            &doc.id,
            index,
            content_hash(text),
            doc.metadata.clone(),
        )
    }

    fn is_semantic_node(kind: &str) -> bool {
        matches!(
            kind,
            "function_item"
                | "struct_item"
                | "enum_item"
                | "trait_item"
                | "impl_item"
                | "mod_item"
                | "const_item"
                | "static_item"
                | "type_item"
                | "use_declaration"
                | "macro_definition"
        )
    }
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self::default_settings()
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        if Self::detect_rust(doc) {
            self.chunk_rust(doc)
        } else {
            self.fallback.chunk(doc)
        }
    }

    fn name(&self) -> &'static str {
        "code_semantic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_code_semantic_chunks() {
        let chunker = CodeChunker::new(4096);
        let mut md = crate::types::Metadata::new();
        md.insert("path".into(), "src/lib.rs".into());
        let doc = Document::with_metadata(
            "doc1",
            "use std::fmt;\n\nstruct A { x: i32 }\n\nfn hello() -> i32 { 42 }",
            md,
        );

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.text.contains("fn hello")));
    }
}
