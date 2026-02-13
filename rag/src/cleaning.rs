//! Text cleaning pipeline executed before chunking.

use crate::types::Document;

/// Trait for document cleaning strategies.
pub trait Cleaner: Send + Sync {
    /// Cleans the input document and returns a normalized version.
    fn clean(&self, doc: &Document) -> Document;

    /// Returns the cleaner name.
    fn name(&self) -> &'static str;
}

/// Default cleaner used by RAG before chunking.
///
/// It performs lightweight normalization:
/// - normalize line endings (`\r\n`, `\r` -> `\n`)
/// - trim trailing whitespace on each line
/// - collapse excessive blank lines (max 2)
/// - trim outer whitespace
#[derive(Debug, Clone, Default)]
pub struct BasicCleaner;

impl BasicCleaner {
    fn normalize_line_endings(text: &str) -> String {
        text.replace("\r\n", "\n").replace('\r', "\n")
    }

    fn trim_trailing_whitespace_per_line(text: &str) -> String {
        text.lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn collapse_blank_lines(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut blank_run = 0usize;
        let mut prev_was_newline = false;

        for line in text.lines() {
            let is_blank = line.trim().is_empty();
            if is_blank {
                blank_run += 1;
                if blank_run <= 2 && !out.is_empty() && !prev_was_newline {
                    out.push('\n');
                    prev_was_newline = true;
                }
            } else {
                blank_run = 0;
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(line);
                prev_was_newline = false;
            }
        }

        out
    }
}

impl Cleaner for BasicCleaner {
    fn clean(&self, doc: &Document) -> Document {
        let normalized = Self::normalize_line_endings(&doc.text);
        let trimmed = Self::trim_trailing_whitespace_per_line(&normalized);
        let collapsed = Self::collapse_blank_lines(&trimmed);

        Document::with_metadata(
            doc.id.clone(),
            collapsed.trim().to_string(),
            doc.metadata.clone(),
        )
    }

    fn name(&self) -> &'static str {
        "basic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_text() {
        let doc = Document::new("d1", "a\r\n\r\n\r\n b  \n\n\n\nc");
        let cleaned = BasicCleaner.clean(&doc);
        assert_eq!(cleaned.text, "a\n\n b\n\nc");
    }
}
