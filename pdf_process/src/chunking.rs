use crate::model::{Chunk, ChunkModality, ChunkStrategy, Page};

pub(crate) fn build_chunks(
    pages: &[Page],
    strategy: ChunkStrategy,
    target_chunk_chars: usize,
    target_chunk_tokens: usize,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let max_chars = target_chunk_chars.max(256);
    let max_tokens = target_chunk_tokens.max(64);

    for page in pages {
        if page.text.trim().is_empty() {
            continue;
        }

        let segments = split_text(&page.text, strategy, max_chars);
        for (idx, segment) in segments.into_iter().enumerate() {
            if segment.is_empty() {
                continue;
            }
            let modality = match (page.mode.as_str(), page.vision_ref.is_some()) {
                ("vision_only", _) => ChunkModality::Vision,
                ("hybrid", true) => ChunkModality::Hybrid,
                _ => ChunkModality::Text,
            };
            let mut token_estimate = estimate_tokens(&segment);
            if token_estimate > max_tokens {
                token_estimate = max_tokens;
            }
            chunks.push(Chunk {
                id: format!("p{}c{}", page.index, idx),
                page_start: page.index,
                page_end: page.index,
                token_estimate,
                modality,
                content: segment,
            });
        }
    }

    chunks
}

fn split_text(text: &str, strategy: ChunkStrategy, max_chars: usize) -> Vec<String> {
    match strategy {
        ChunkStrategy::Fixed => split_fixed(text, max_chars),
        ChunkStrategy::Sentence => split_sentence(text, max_chars),
        ChunkStrategy::Hybrid => {
            let sentence = split_sentence(text, max_chars);
            if sentence.len() <= 1 {
                split_fixed(text, max_chars)
            } else {
                sentence
            }
        }
    }
}

fn split_fixed(text: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let end = (cursor + max_chars).min(text.len());
        let slice = &text[cursor..end];
        out.push(slice.trim().to_string());
        cursor = end;
    }
    out
}

fn split_sentence(text: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for part in sentence_like_segments(text) {
        if current.is_empty() {
            current.push_str(part);
            continue;
        }
        if current.len() + part.len() + 1 > max_chars {
            out.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(part);
    }

    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }

    if out.is_empty() {
        split_fixed(text, max_chars)
    } else {
        out
    }
}

fn sentence_like_segments(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if matches!(c, '.' | '!' | '?' | '\n') {
            let end = i + c.len_utf8();
            let segment = text[start..end].trim();
            if !segment.is_empty() {
                out.push(segment);
            }
            start = end;
        }
    }
    if start < text.len() {
        let segment = text[start..].trim();
        if !segment.is_empty() {
            out.push(segment);
        }
    }
    out
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Page, PageMode};

    #[test]
    fn chunks_respect_non_empty_content() {
        let pages = vec![Page {
            index: 1,
            source_page: 1,
            mode: PageMode::Native,
            text: "A. B. C.".to_string(),
            text_chars: 8,
            token_estimate: 2,
            vision_ref: None,
            image_ref: None,
        }];
        let chunks = build_chunks(&pages, ChunkStrategy::Sentence, 8, 20);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.content.is_empty()));
    }
}
