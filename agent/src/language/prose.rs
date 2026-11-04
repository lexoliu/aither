//! Judging the language of a text's prose.
//!
//! Code blocks, code spans, URLs, paths and identifiers are removed first.
//! What is left is split into runs of one script family; a short run is a
//! technical term of another language and does not change the verdict, as
//! long as a longer run exists to judge. Each judged run is identified by
//! `whatlang`, restricted to the languages a session can use so that short
//! runs are not confused with unrelated ones.

use std::sync::Arc;

use whatlang::{Detector, Lang};

use super::Language;

/// A Latin-script run of at most this many words is a term.
const LATIN_TERM_WORDS: usize = 3;
/// A run of any other script of at most this many letters is a term.
const OTHER_TERM_LETTERS: usize = 4;

/// Collapses every run of whitespace to one space and trims the ends.
pub(super) fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether the prose of `text` is in `target`; `others` are the other
/// languages the session uses, which the detector may also answer. A text
/// with no prose is in every language.
pub(super) fn is_in(text: &str, target: Language, others: &[Language]) -> bool {
    let prose = prose(text);
    let segments = segments(&prose);
    if segments.is_empty() {
        return true;
    }
    let mut allowed = vec![target.lang(), Lang::Eng];
    allowed.extend(others.iter().map(|language| language.lang()));
    allowed.dedup();
    let detector = Detector::with_allowlist(allowed);
    let judged: Vec<&Segment> = if segments.iter().all(Segment::is_term) {
        // Nothing but terms: the text itself is the verdict.
        return detector.detect_lang(&prose) == Some(target.lang());
    } else {
        segments
            .iter()
            .filter(|segment| !segment.is_term())
            .collect()
    };
    judged
        .iter()
        .all(|segment| detector.detect_lang(&segment.text) == Some(target.lang()))
}

/// Removes the quotes of `text` that match, after whitespace normalization,
/// something in `corpus`. A quote is a span in double quotes (straight,
/// curly, corner or guillemets) or a markdown blockquote line.
pub(super) fn without_quotes(text: &str, corpus: &[Arc<str>]) -> String {
    let matches = |quoted: &str| {
        let quoted = normalize(quoted);
        !quoted.is_empty() && corpus.iter().any(|message| message.contains(&quoted))
    };
    let mut unquoted = String::with_capacity(text.len());
    for line in text.lines() {
        if let Some(quoted) = line.trim_start().strip_prefix('>')
            && matches(quoted)
        {
            continue;
        }
        unquoted.push_str(line);
        unquoted.push('\n');
    }
    let mut kept = String::with_capacity(unquoted.len());
    let mut chars = unquoted.chars();
    while let Some(c) = chars.next() {
        let Some(close) = closing_quote(c) else {
            kept.push(c);
            continue;
        };
        let mut quoted = String::new();
        let mut closed = false;
        for inner in chars.by_ref() {
            if inner == close {
                closed = true;
                break;
            }
            quoted.push(inner);
        }
        if closed && matches(&quoted) {
            kept.push(' ');
        } else {
            kept.push(c);
            kept.push_str(&quoted);
            if closed {
                kept.push(close);
            }
        }
    }
    kept
}

const fn closing_quote(open: char) -> Option<char> {
    match open {
        '"' => Some('"'),
        '\u{201c}' => Some('\u{201d}'),
        '\u{300c}' => Some('\u{300d}'),
        '\u{300e}' => Some('\u{300f}'),
        '\u{ab}' => Some('\u{bb}'),
        _ => None,
    }
}

/// The text with code, URLs, paths and identifiers removed.
fn prose(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut fenced = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        for (index, part) in line.split('`').enumerate() {
            if index % 2 == 0 {
                push_words(part, &mut out);
            }
        }
        out.push('\n');
    }
    out
}

fn push_words(text: &str, out: &mut String) {
    for token in text.split_whitespace() {
        if token.contains("://") || token.starts_with("www.") {
            out.push(' ');
            continue;
        }
        let mut chunk = String::new();
        let mut chunk_is_latin = true;
        for c in token.chars() {
            let latin = c.is_ascii() || (c as u32) < 0x250;
            if latin != chunk_is_latin {
                flush(&chunk, chunk_is_latin, out);
                chunk.clear();
                chunk_is_latin = latin;
            }
            chunk.push(c);
        }
        flush(&chunk, chunk_is_latin, out);
        out.push(' ');
    }
}

fn flush(chunk: &str, latin: bool, out: &mut String) {
    if chunk.is_empty() {
        return;
    }
    if latin && is_identifier(chunk) {
        out.push(' ');
    } else {
        out.push_str(chunk);
    }
}

/// Whether a run of Latin-range characters is a path, an identifier, a flag
/// or an acronym rather than a word of prose.
fn is_identifier(chunk: &str) -> bool {
    let core = chunk.trim_matches(|c: char| {
        matches!(
            c,
            '(' | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
                | ','
                | ';'
                | '!'
                | '?'
                | '"'
                | '\''
                | '*'
                | ':'
                | '.'
        )
    });
    if core.is_empty() {
        return false;
    }
    if core.starts_with('-') && core.len() > 1 {
        return true;
    }
    if core.contains(['_', '/', '\\', '~', '@', '#', '=', '.', '$'])
        || core.contains("::")
        || core.contains(|c: char| c.is_ascii_digit())
    {
        return true;
    }
    let mut previous: Option<char> = None;
    for c in core.chars() {
        if previous.is_some_and(char::is_lowercase) && c.is_uppercase() {
            return true;
        }
        previous = Some(c);
    }
    core.chars().count() >= 2 && core.chars().all(char::is_uppercase)
}

/// A run of letters of one script family and the text between them.
struct Segment {
    text: String,
    latin: bool,
    letters: usize,
}

impl Segment {
    fn is_term(&self) -> bool {
        if self.latin {
            self.text.split_whitespace().count() <= LATIN_TERM_WORDS
        } else {
            self.letters <= OTHER_TERM_LETTERS
        }
    }
}

fn segments(prose: &str) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    for c in prose.chars() {
        if c.is_alphabetic() {
            let latin = (c as u32) < 0x250;
            match segments.last_mut() {
                Some(segment) if segment.latin == latin => {
                    segment.letters += 1;
                    segment.text.push(c);
                }
                _ => segments.push(Segment {
                    text: c.to_string(),
                    latin,
                    letters: 1,
                }),
            }
        } else if let Some(segment) = segments.last_mut() {
            segment.text.push(c);
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chinese() -> Language {
        "Chinese".parse().unwrap()
    }

    #[test]
    fn code_identifiers_and_paths_are_not_prose() {
        let text = "请检查 `tokio::spawn` 和 spawn_blocking，见 src/agent/mod.rs 与 https://example.com/a。\n```rust\nfn main() {}\n```";
        assert!(is_in(text, chinese(), &[]));
    }

    #[test]
    fn a_short_foreign_term_does_not_change_the_verdict() {
        assert!(is_in(
            "好的，我来用 Rust 的 trait object 实现这个功能。",
            chinese(),
            &[]
        ));
        assert!(is_in(
            "Use the 数据库 here to store the results.",
            Language::ENGLISH,
            &[chinese()]
        ));
    }

    #[test]
    fn a_foreign_sentence_is_prose() {
        assert!(!is_in("Which database should we use?", chinese(), &[]));
        assert!(!is_in(
            "好的。 Please update the tasks file now.",
            chinese(),
            &[]
        ));
        assert!(!is_in(
            "我们需要更新这个文件",
            Language::ENGLISH,
            &[chinese()]
        ));
    }

    #[test]
    fn a_text_without_prose_is_in_every_language() {
        assert!(is_in("`cargo test` https://example.com 42", chinese(), &[]));
    }

    #[test]
    fn matching_quotes_are_removed_and_others_kept() {
        let corpus = [Arc::<str>::from("请把数据库换成 PostgreSQL，不要用 MySQL")];
        let text = "User said \u{201c}请把数据库换成   PostgreSQL，不要用\nMySQL\u{201d}.";
        assert!(!without_quotes(text, &corpus).contains('库'));
        let other = "User said \u{201c}他们想要完全不同的方案\u{201d}.";
        assert!(without_quotes(other, &corpus).contains("方案"));
    }
}
