//! Recognizing the output URLs the sandbox hands back.
//!
//! When a terminal command produces more output than belongs in the window,
//! the sandbox writes it to `outputs/` under a four-word name and reports the
//! path instead of the content. Those paths are the only way the model can
//! get back to the full output, so compaction has to find them in a
//! conversation that is about to be thrown away and carry them into the
//! handoff's file index.
//!
//! Recognition is by shape rather than by bookkeeping: four lowercase
//! hyphenated words and an alphanumeric extension. Anything else that happens
//! to start with `outputs/` — a directory the model mentioned, a path it
//! invented — is not a real artifact and is not worth a line in the handoff.

use std::collections::HashSet;

/// Extracts the sandbox output URLs referenced anywhere in `text`.
///
/// # Example
///
/// ```
/// use aither_agent::extract_referenced_urls;
///
/// let text = "Output saved to outputs/amber-oak-swift-river.txt (5000 bytes)";
/// let urls = extract_referenced_urls(text);
/// assert!(urls.contains("outputs/amber-oak-swift-river.txt"));
/// ```
#[must_use]
pub fn extract_referenced_urls(text: &str) -> HashSet<String> {
    let mut urls = HashSet::new();
    for word in text.split_whitespace() {
        let word = word.trim_matches(|c: char| {
            c == '"' || c == '\'' || c == '`' || c == ',' || c == '.' || c == ')' || c == ']'
        });
        if let Some(filename) = word.strip_prefix("outputs/")
            && is_valid_output_filename(filename)
        {
            urls.insert(word.to_string());
        }
    }
    urls
}

/// Whether `filename` matches the sandbox's `word-word-word-word.ext` shape.
#[must_use]
pub fn is_valid_output_filename(filename: &str) -> bool {
    let Some((name, extension)) = filename.rsplit_once('.') else {
        return false;
    };
    if extension.is_empty() || !extension.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    let parts: Vec<&str> = name.split('-').collect();
    parts.len() == 4
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_every_referenced_output() {
        let text = "The agent read config from outputs/amber-oak-swift-river.txt and saved data to outputs/bold-creek-calm-dawn.json.";
        let urls = extract_referenced_urls(text);
        assert_eq!(urls.len(), 2);
        assert!(urls.contains("outputs/amber-oak-swift-river.txt"));
        assert!(urls.contains("outputs/bold-creek-calm-dawn.json"));
    }

    #[test]
    fn surrounding_punctuation_is_stripped() {
        let text = r#"See "outputs/jade-mist-clear-pool.txt" for details."#;
        assert!(extract_referenced_urls(text).contains("outputs/jade-mist-clear-pool.txt"));
    }

    #[test]
    fn paths_that_are_not_artifacts_are_ignored() {
        let text = "Not valid: outputs/single.txt outputs/two-words.txt outputs/ outputs/one-two-three.txt";
        assert!(extract_referenced_urls(text).is_empty());
    }

    #[test]
    fn the_filename_shape_is_enforced() {
        assert!(is_valid_output_filename("amber-oak-swift-river.txt"));
        assert!(is_valid_output_filename("bold-creek-calm-dawn.json"));
        assert!(!is_valid_output_filename("single.txt"));
        assert!(!is_valid_output_filename("two-words.txt"));
        assert!(!is_valid_output_filename("no-extension"));
        assert!(!is_valid_output_filename("Upper-Case-Words.txt"));
    }
}
