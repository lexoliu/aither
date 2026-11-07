//! References to tracked work, and the words of a row's text.

use std::fmt;
use std::hash::{Hash, Hasher};

/// What a [`WorkRef`] names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum WorkKind {
    /// An issue.
    Issue,
    /// A pull request.
    PullRequest,
}

/// A reference to an issue or a pull request.
///
/// [`parse_all`](Self::parse_all) reads `#12` or `owner/repo#12` as an
/// issue, and `PR #12` or `PR owner/repo#12` as a pull request. An unqualified
/// reference means the session's own repository; what that is, and where
/// the item lives, is the [`WorkTracker`](super::WorkTracker)'s business.
///
/// Repository names compare without regard to ASCII case, as forges treat
/// them; the written spelling is kept for display.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkRef {
    kind: WorkKind,
    repository: Option<String>,
    number: u64,
}

impl PartialEq for WorkRef {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.number == other.number
            && match (&self.repository, &other.repository) {
                (Some(left), Some(right)) => left.eq_ignore_ascii_case(right),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for WorkRef {}

impl Hash for WorkRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.number.hash(state);
        match &self.repository {
            Some(repository) => {
                state.write_u8(1);
                for byte in repository.bytes() {
                    state.write_u8(byte.to_ascii_lowercase());
                }
            }
            None => state.write_u8(0),
        }
    }
}

impl WorkRef {
    /// Issue `number` of the session's repository.
    #[must_use]
    pub const fn issue(number: u64) -> Self {
        Self {
            kind: WorkKind::Issue,
            repository: None,
            number,
        }
    }

    /// Pull request `number` of the session's repository.
    #[must_use]
    pub const fn pull_request(number: u64) -> Self {
        Self {
            kind: WorkKind::PullRequest,
            repository: None,
            number,
        }
    }

    /// The same reference, qualified by `repository` (`owner/repo`).
    #[must_use]
    pub fn in_repository(mut self, repository: impl Into<String>) -> Self {
        self.repository = Some(repository.into());
        self
    }

    /// Whether this names an issue or a pull request.
    #[must_use]
    pub const fn kind(&self) -> WorkKind {
        self.kind
    }

    /// The repository it is qualified by; `None` for the session's own.
    #[must_use]
    pub fn repository(&self) -> Option<&str> {
        self.repository.as_deref()
    }

    /// The item's number.
    #[must_use]
    pub const fn number(&self) -> u64 {
        self.number
    }

    /// Whether `self` and `other` can name the same item: the same kind and
    /// number, in the same repository or with either one unqualified.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.number == other.number
            && match (&self.repository, &other.repository) {
                (Some(left), Some(right)) => left.eq_ignore_ascii_case(right),
                _ => true,
            }
    }

    /// Every reference written in `text`, in order.
    #[must_use]
    pub fn parse_all(text: &str) -> Vec<Self> {
        let mut references = Vec::new();
        let mut previous: Option<&str> = None;
        for token in words(text) {
            let pull_request = previous.is_some_and(|word| word.eq_ignore_ascii_case("PR"));
            if let Some(reference) = parse_token(token, pull_request) {
                references.push(reference);
            }
            previous = Some(token);
        }
        references
    }
}

impl fmt::Display for WorkRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.kind == WorkKind::PullRequest {
            f.write_str("PR ")?;
        }
        if let Some(repository) = &self.repository {
            f.write_str(repository)?;
        }
        write!(f, "#{}", self.number)
    }
}

/// The words of `text`, split at whitespace and at the punctuation that
/// surrounds a reference in prose or markdown.
///
/// [`DocumentRow::names`](super::DocumentRow::names) matches ids against
/// them; a schema reads the payload of its own marks with them, so markdown
/// and punctuation around a word do not hide it.
pub fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '`' | '"' | '\'' | '*'
            )
    })
    .map(|token| token.trim_end_matches(['.', '!', '?']))
    .filter(|token| !token.is_empty())
}

/// Parses one token; `pull_request` when the word before it was `PR`.
fn parse_token(token: &str, pull_request: bool) -> Option<WorkRef> {
    let (scope, rest) = token.split_once('#')?;
    // The number ends at the first non-digit, which must not continue a
    // word: `#8。` is issue 8, `#8th` is no reference.
    let (digits, tail) = rest
        .find(|c: char| !c.is_ascii_digit())
        .map_or((rest, ""), |end| rest.split_at(end));
    if digits.is_empty() || tail.chars().next().is_some_and(char::is_alphanumeric) {
        return None;
    }
    let number = digits.parse().ok()?;
    let (kind, repository) = if scope.is_empty() {
        (None, None)
    } else if scope.eq_ignore_ascii_case("PR") {
        (Some(WorkKind::PullRequest), None)
    } else if is_repository(scope) {
        (None, Some(scope.to_string()))
    } else {
        return None;
    };
    let kind = kind.unwrap_or(if pull_request {
        WorkKind::PullRequest
    } else {
        WorkKind::Issue
    });
    Some(WorkRef {
        kind,
        repository,
        number,
    })
}

/// Whether `scope` is an `owner/repo` name.
fn is_repository(scope: &str) -> bool {
    let valid = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    };
    scope
        .split_once('/')
        .is_some_and(|(owner, repo)| valid(owner) && valid(repo))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_written_form_parses() {
        let refs = WorkRef::parse_all(
            "- #12 ship it (PR #3, owner/repo#4, PR other/repo#5); see `#6`. PR#7",
        );
        assert_eq!(
            refs,
            vec![
                WorkRef::issue(12),
                WorkRef::pull_request(3),
                WorkRef::issue(4).in_repository("owner/repo"),
                WorkRef::pull_request(5).in_repository("other/repo"),
                WorkRef::issue(6),
                WorkRef::pull_request(7),
            ]
        );
    }

    #[test]
    fn anchors_headings_and_words_are_not_references() {
        assert_eq!(
            WorkRef::parse_all("## Roots page.html#3 #abc # 12 C#"),
            Vec::new()
        );
    }

    #[test]
    fn non_ascii_text_around_a_reference_is_kept_apart() {
        assert_eq!(
            WorkRef::parse_all("修复 #8。和 PR #9 #10th"),
            vec![WorkRef::issue(8), WorkRef::pull_request(9)]
        );
    }

    #[test]
    fn display_writes_the_form_it_parses_from() {
        for text in ["#12", "owner/repo#4", "PR #3", "PR owner/repo#5"] {
            let parsed = WorkRef::parse_all(text);
            assert_eq!(parsed.len(), 1, "{text}");
            assert_eq!(parsed[0].to_string(), text);
        }
    }

    #[test]
    fn an_unqualified_reference_matches_any_repository() {
        let qualified = WorkRef::issue(4).in_repository("owner/repo");
        assert!(WorkRef::issue(4).matches(&qualified));
        assert!(!WorkRef::issue(4).in_repository("x/y").matches(&qualified));
        assert!(!WorkRef::pull_request(4).matches(&qualified));
    }

    #[test]
    fn repository_names_compare_without_case() {
        let written = WorkRef::issue(4).in_repository("May-Agent/may");
        let canonical = WorkRef::issue(4).in_repository("may-agent/MAY");
        assert!(written.matches(&canonical));
        assert_eq!(written, canonical);
        let set: std::collections::HashSet<_> = [written.clone(), canonical].into();
        assert_eq!(set.len(), 1, "equal references hash alike");
        assert_ne!(written, WorkRef::issue(4), "qualified is not unqualified");
        assert_eq!(written.to_string(), "May-Agent/may#4");
    }
}
