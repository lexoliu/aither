//! Recognizing the output URLs the sandbox hands back.
//!
//! When a terminal command produces more output than belongs in the window,
//! the sandbox writes it to `outputs/` under a name leased from its id
//! allocator and reports the path instead of the content, relative
//! (`outputs/<name>`) in some notices and absolute inside the session's
//! outputs directory in others. Those paths are the only way the model can get
//! back to the full output, so compaction has to find them in a conversation
//! that is about to be thrown away and carry them into the handoff's file
//! index.
//!
//! Recognition is by shape rather than by bookkeeping: one lowercase word, the
//! shape every leased id has, and an alphanumeric extension. Anything else
//! that happens to start with `outputs/` — a directory the model mentioned, a
//! path it invented — is not a real artifact and is not worth a line in the
//! handoff.

use std::collections::HashSet;
use std::path::Path;

/// Characters that can wrap a path in prose without being part of it.
const fn is_path_punctuation(c: char) -> bool {
    matches!(c, '"' | '\'' | '`' | ',' | '.' | ')' | ']')
}

/// Extracts the sandbox output paths referenced anywhere in `text`: relative
/// `outputs/<name>` words, and, when `outputs_dir` is the session's outputs
/// directory, `<outputs_dir>/<name>` as the notices print it.
///
/// # Example
///
/// ```
/// use std::path::Path;
/// use aither_agent::extract_referenced_urls;
///
/// let text = "Output saved to outputs/river.txt (5000 bytes)";
/// let urls = extract_referenced_urls(text, None);
/// assert!(urls.contains("outputs/river.txt"));
///
/// let dir = Path::new("/work/outputs");
/// let saved = dir.join("creek.txt").display().to_string();
/// let text = format!("saved to {saved}.");
/// let urls = extract_referenced_urls(&text, Some(dir));
/// assert!(urls.contains(&saved));
/// ```
#[must_use]
pub fn extract_referenced_urls(text: &str, outputs_dir: Option<&Path>) -> HashSet<String> {
    let mut urls = HashSet::new();
    for word in text.split_whitespace() {
        let word = word.trim_matches(is_path_punctuation);
        if let Some(filename) = word.strip_prefix("outputs/")
            && is_valid_output_filename(filename)
        {
            urls.insert(word.to_string());
        }
    }
    if let Some(outputs_dir) = outputs_dir {
        // The directory may hold spaces, so the absolute form is found by its
        // prefix rather than by splitting words. Joining an empty component
        // ends the directory with the platform's separator, as the notice's
        // own `join` does.
        let prefix = outputs_dir.join("").display().to_string();
        for (start, _) in text.match_indices(prefix.as_str()) {
            let rest = &text[start + prefix.len()..];
            let name = rest.split_whitespace().next().unwrap_or_default();
            let name = name.trim_end_matches(is_path_punctuation);
            if is_valid_output_filename(name) {
                urls.insert(format!("{prefix}{name}"));
            }
        }
    }
    urls
}

/// Whether `filename` matches the sandbox's `word.ext` shape: one lowercase
/// word, as every leased id is, and an alphanumeric extension.
#[must_use]
pub fn is_valid_output_filename(filename: &str) -> bool {
    let Some((name, extension)) = filename.rsplit_once('.') else {
        return false;
    };
    !extension.is_empty()
        && extension.bytes().all(|byte| byte.is_ascii_alphanumeric())
        && !name.is_empty()
        && name.bytes().all(|byte| byte.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_every_referenced_output() {
        let text =
            "The agent read config from outputs/river.txt and saved data to outputs/creek.json.";
        let urls = extract_referenced_urls(text, None);
        assert_eq!(urls.len(), 2);
        assert!(urls.contains("outputs/river.txt"));
        assert!(urls.contains("outputs/creek.json"));
    }

    #[test]
    fn surrounding_punctuation_is_stripped() {
        let text = r#"See "outputs/pool.txt" for details."#;
        assert!(extract_referenced_urls(text, None).contains("outputs/pool.txt"));
    }

    #[test]
    fn the_saved_output_notice_the_sandbox_renders_is_found() {
        let dir = Path::new("/home/me/My Sessions/s1/outputs");
        let path = dir.join("river.txt");
        let notice =
            aither_sandbox::output_saved_text("preview line", &path.to_string_lossy(), 900, 48_000);
        let urls = extract_referenced_urls(&notice, Some(dir));
        assert_eq!(
            urls,
            HashSet::from([path.to_string_lossy().into_owned()]),
            "notice: {notice}"
        );
    }

    #[test]
    fn absolute_paths_outside_the_outputs_directory_are_ignored() {
        let text = "saved to /elsewhere/outputs/river.txt and /work/outputs/not-valid.txt";
        assert!(extract_referenced_urls(text, Some(Path::new("/work/outputs"))).is_empty());
    }

    #[test]
    fn paths_that_are_not_artifacts_are_ignored() {
        let text = "Not valid: outputs/two-words.txt outputs/ outputs/Notes.txt outputs/draft2.txt outputs/river";
        assert!(extract_referenced_urls(text, None).is_empty());
    }

    #[test]
    fn the_filename_shape_is_enforced() {
        assert!(is_valid_output_filename("river.txt"));
        assert!(is_valid_output_filename("creek.json"));
        assert!(!is_valid_output_filename("two-words.txt"));
        assert!(!is_valid_output_filename("no_extension"));
        assert!(!is_valid_output_filename("Upper.txt"));
        assert!(!is_valid_output_filename(".txt"));
    }
}
