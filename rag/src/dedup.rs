//! Content deduplication using xxhash.

use xxhash_rust::xxh3::xxh3_64;

/// Computes a content hash for deduplication.
#[must_use]
pub fn content_hash(text: &str) -> u64 {
    xxh3_64(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_content_same_hash() {
        let text = "Hello, world!";
        assert_eq!(content_hash(text), content_hash(text));
    }

    #[test]
    fn different_content_different_hash() {
        let text1 = "Hello, world!";
        let text2 = "Goodbye, world!";
        assert_ne!(content_hash(text1), content_hash(text2));
    }

    #[test]
    fn empty_string_hash() {
        let hash = content_hash("");
        assert_ne!(hash, 0); // xxh3 produces non-zero for empty input
    }
}
