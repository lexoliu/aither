//! Shared name generation utilities.

use human_hash::humanize;
use uuid::Uuid;

/// Generates a hyphenated slug using the default human-hash wordlist.
///
/// # Panics
///
/// Panics if `words` is zero.
pub fn random_word_slug(words: usize) -> String {
    assert!(words > 0, "word count must be positive");
    let id = Uuid::new_v4();
    humanize(&id, words)
}
