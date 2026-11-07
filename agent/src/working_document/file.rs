//! Reading the working document from disk and diffing its versions.

use std::path::{Path, PathBuf};

use similar::{ChangeTag, TextDiff};

/// How many changed lines a diff shows before it is cut.
const DIFF_LINES: usize = 48;

/// Where a working document lives, and the name the model knows it by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentFile {
    path: PathBuf,
    name: String,
}

/// The document as one read found it: its content, `None` when absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Snapshot(pub Option<String>);

impl Snapshot {
    /// Its content, `None` when the document was absent.
    pub fn content(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl DocumentFile {
    /// The document `name`, a path relative to `sandbox_dir`.
    pub fn new(sandbox_dir: &Path, name: &str) -> Self {
        Self {
            path: sandbox_dir.join(name),
            name: name.to_string(),
        }
    }

    /// The name the model knows it by, as the embedder wrote it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What it holds now; absent when it cannot be read, which is logged.
    pub async fn read(&self) -> Snapshot {
        match async_fs::read_to_string(&self.path).await {
            Ok(content) => Snapshot(Some(content)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Snapshot(None),
            Err(error) => {
                tracing::warn!(
                    path = %self.path.display(),
                    %error,
                    "the working document cannot be read"
                );
                Snapshot(None)
            }
        }
    }
}

/// The changed lines from `previous` to `current`, each prefixed `- ` or
/// `+ `, cut after [`DIFF_LINES`] with a `...` line; `None` when nothing
/// changed, and empty when only blank lines did.
pub fn diff(previous: &Snapshot, current: &Snapshot) -> Option<String> {
    if previous == current {
        return None;
    }
    let diff = TextDiff::from_lines(
        previous.content().unwrap_or_default(),
        current.content().unwrap_or_default(),
    );
    let mut lines = Vec::new();
    for change in diff.iter_all_changes() {
        let prefix = match change.tag() {
            ChangeTag::Delete => "- ",
            ChangeTag::Insert => "+ ",
            ChangeTag::Equal => continue,
        };
        let value = change.value().trim_end_matches('\n');
        if value.is_empty() {
            continue;
        }
        lines.push([prefix, value].concat());
        if lines.len() == DIFF_LINES {
            lines.push("...".to_string());
            break;
        }
    }
    Some(lines.join("\n"))
}

/// The lines a change from `previous` to `current` added, or `None` when it
/// added none.
#[cfg(feature = "languages")]
pub fn added(previous: &Snapshot, current: &Snapshot) -> Option<String> {
    if previous == current {
        return None;
    }
    let diff = TextDiff::from_lines(
        previous.content().unwrap_or_default(),
        current.content().unwrap_or_default(),
    );
    let added: String = diff
        .iter_all_changes()
        .filter(|change| change.tag() == ChangeTag::Insert)
        .map(|change| change.value())
        .collect();
    (!added.trim().is_empty()).then_some(added)
}
