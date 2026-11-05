//! Progress and character-boundary guarantees of the fixed-size chunker.

use super::Chunker;
use super::fixed::FixedSizeChunker;
use crate::types::Document;

/// Chunks `text` and checks what must hold for any input: every chunk is a
/// slice of the text, fits the size limit, starts strictly after the previous
/// one, and together the chunks cover the whole text.
fn assert_well_formed(chunker: &FixedSizeChunker, text: &str, size: usize) {
    let doc = Document::new("doc", text);
    let chunks = chunker.chunk(&doc).unwrap();
    assert!(!chunks.is_empty());

    let mut previous_start = None;
    let mut covered_to = 0;
    for chunk in &chunks {
        assert!(
            text.contains(&chunk.text),
            "chunk is not a slice of the text"
        );
        assert!(chunk.text.chars().count() <= size, "chunk exceeds the size");

        let start: usize = chunk.metadata["chunk_start"].parse().unwrap_or(0);
        let end: usize = chunk.metadata["chunk_end"].parse().unwrap_or(text.len());
        assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
        if let Some(previous) = previous_start {
            assert!(start > previous, "chunk {} did not advance", chunk.index);
        }
        previous_start = Some(start);
        assert!(start <= covered_to, "gap before chunk {}", chunk.index);
        covered_to = covered_to.max(end);
    }
    assert_eq!(covered_to, text.len(), "tail of the text is not covered");
}

#[test]
fn early_space_then_long_unbroken_run_terminates() {
    // One space well past the overlap, then a run with no whitespace: the
    // word-boundary cut used to land on the same position forever.
    let text = format!("{} {}", "a".repeat(100), "b".repeat(2000));
    let chunker = FixedSizeChunker::default();
    assert_well_formed(&chunker, &text, 512);

    // 2101 characters: a short first chunk (cut at the space), then steps of
    // 512 - 64. A bounded count, not an endless stream of revisits.
    let chunks = chunker.chunk(&Document::new("doc", &text)).unwrap();
    assert_eq!(chunks.len(), 6);
}

#[test]
fn space_within_overlap_of_chunk_start_terminates() {
    let text = format!("{} {}", "a".repeat(10), "b".repeat(2000));
    assert_well_formed(&FixedSizeChunker::default(), &text, 512);
}

#[test]
fn cjk_text_longer_than_a_chunk_does_not_panic() {
    let text = "日本語のテキスト".repeat(300);
    assert_well_formed(&FixedSizeChunker::default(), &text, 512);
    assert_well_formed(&FixedSizeChunker::new(50, 10), &text, 50);
}

#[test]
fn mixed_width_text_with_spaces_is_cut_on_boundaries() {
    let text = "naïve café 日本 😀 emoji ".repeat(120);
    for (size, overlap) in [(7, 0), (7, 3), (50, 10), (51, 50), (512, 64)] {
        assert_well_formed(&FixedSizeChunker::new(size, overlap), &text, size);
    }
}

#[test]
fn size_counts_characters_not_bytes() {
    let text = "日".repeat(100);
    let chunks = FixedSizeChunker::new(100, 10)
        .chunk(&Document::new("doc", &text))
        .unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
}
