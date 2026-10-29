//! Parser for the compact contextual patch format.
//!
//! Turns patch text into typed [`Action`]s. Leniencies kept on purpose: a
//! surrounding markdown fence is stripped, a patch that starts with a file
//! directive and carries neither envelope marker is wrapped implicitly, empty
//! lines inside a hunk are empty context lines, and empty context lines at a
//! hunk's edges are cosmetic and dropped.

use crate::error::PatchError;

const BEGIN: &str = "*** Begin Patch";
const END: &str = "*** End Patch";
const ADD: &str = "*** Add File: ";
const UPDATE: &str = "*** Update File: ";
const DELETE: &str = "*** Delete File: ";
const MOVE: &str = "*** Move to: ";
const END_OF_FILE: &str = "*** End of File";

/// One file-level operation of a patch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Create `path` holding `lines`.
    Add { path: String, lines: Vec<String> },
    /// Remove `path`; `expected`, when non-empty, must equal the file's lines.
    Delete { path: String, expected: Vec<String> },
    /// Apply `hunks` to `path`, optionally moving the result to `new_path`.
    Update {
        path: String,
        new_path: Option<String>,
        hunks: Vec<Hunk>,
    },
}

/// A run of context, deleted and inserted lines between `@@` markers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hunk {
    /// Text of the `@@ anchor` line: the hunk must match after the first line
    /// equal to it.
    pub anchor: Option<String>,
    pub lines: Vec<HunkLine>,
    /// The hunk's old side must end at the last line of the file.
    pub at_end_of_file: bool,
}

/// One body line of a hunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HunkLine {
    Context(String),
    Delete(String),
    Insert(String),
}

impl HunkLine {
    /// The line as it must appear in the file, or `None` for an insertion.
    pub fn old_side(&self) -> Option<&str> {
        match self {
            Self::Context(text) | Self::Delete(text) => Some(text),
            Self::Insert(_) => None,
        }
    }
}

impl Hunk {
    /// Drops empty context lines from both edges and returns the hunk unless
    /// nothing is left.
    fn finish(mut self) -> Option<Self> {
        let is_blank = |line: &HunkLine| matches!(line, HunkLine::Context(text) if text.is_empty());
        let start = self.lines.iter().position(|l| !is_blank(l))?;
        let end = self.lines.iter().rposition(|l| !is_blank(l))?;
        self.lines.truncate(end + 1);
        self.lines.drain(..start);
        Some(self)
    }
}

/// Parses `text` into the file operations it describes.
pub fn parse(text: &str) -> Result<Vec<Action>, PatchError> {
    let text = unwrap_fence(text.trim());
    let wrapped;
    let text = if starts_with_directive(&text) && !text.contains(BEGIN) && !text.contains(END) {
        wrapped = format!("{BEGIN}\n{text}\n{END}");
        wrapped.as_str()
    } else {
        text.as_str()
    };

    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return Err(format_error(
            "Patch text is too short (must include start and end markers).",
        ));
    }
    if lines[0] != BEGIN {
        return Err(format_error("Patch must start with '*** Begin Patch'"));
    }
    if lines[lines.len() - 1] != END {
        return Err(format_error("Patch must end with '*** End Patch'"));
    }

    let mut actions = Vec::new();
    let mut index = 1;
    while index < lines.len() - 1 {
        let line = lines[index].trim();
        index += 1;
        if let Some(path) = line.strip_prefix(ADD) {
            actions.push(parse_add(path.trim(), &lines, &mut index));
        } else if let Some(path) = line.strip_prefix(UPDATE) {
            actions.push(parse_update(path.trim(), &lines, &mut index));
        } else if let Some(path) = line.strip_prefix(DELETE) {
            actions.push(parse_delete(path.trim(), &lines, &mut index));
        }
    }
    if actions.is_empty() {
        return Err(format_error("No file directive found in patch."));
    }
    Ok(actions)
}

fn format_error(message: &str) -> PatchError {
    PatchError::InvalidFormat(message.to_owned())
}

fn starts_with_directive(text: &str) -> bool {
    [UPDATE, ADD, DELETE]
        .iter()
        .any(|directive| text.starts_with(directive.trim_end()))
}

/// Strips a leading markdown fence line and its matching trailing fence line.
fn unwrap_fence(text: &str) -> String {
    if !text.starts_with("```") {
        return text.to_owned();
    }
    let mut lines: Vec<&str> = text.lines().skip(1).collect();
    if lines.last().is_some_and(|line| line.trim() == "```") {
        lines.pop();
    }
    lines.join("\n").trim().to_owned()
}

/// Parses the body of an Add directive; `index` ends on the next `*** ` line.
fn parse_add(path: &str, lines: &[&str], index: &mut usize) -> Action {
    let mut content = Vec::new();
    // A bare empty line is a blank line of the new file whose `+` was
    // omitted; only a trailing run of them (a separator) is not content.
    let mut trailing_blank = 0;
    while *index < lines.len() && !lines[*index].starts_with("*** ") {
        let line = lines[*index];
        if let Some(text) = line.strip_prefix('+') {
            content.push(text.to_owned());
            trailing_blank = 0;
        } else if line.is_empty() {
            content.push(String::new());
            trailing_blank += 1;
        }
        *index += 1;
    }
    content.truncate(content.len() - trailing_blank);
    Action::Add {
        path: path.to_owned(),
        lines: content,
    }
}

fn parse_delete(path: &str, lines: &[&str], index: &mut usize) -> Action {
    let mut expected = Vec::new();
    while *index < lines.len() && !lines[*index].starts_with("*** ") {
        if let Some(text) = lines[*index].strip_prefix('-') {
            expected.push(text.to_owned());
        }
        *index += 1;
    }
    Action::Delete {
        path: path.to_owned(),
        expected,
    }
}

fn parse_update(path: &str, lines: &[&str], index: &mut usize) -> Action {
    let mut hunks = Vec::new();
    let mut new_path = None;
    let mut current = Hunk::default();
    while *index < lines.len() && !lines[*index].starts_with(END) {
        let line = lines[*index];
        if [ADD, UPDATE, DELETE]
            .iter()
            .any(|directive| line.starts_with(directive.trim_end()))
        {
            break;
        }
        *index += 1;
        if let Some(target) = line.strip_prefix(MOVE) {
            new_path = Some(target.trim().to_owned());
        } else if line == END_OF_FILE {
            current.at_end_of_file = true;
            hunks.extend(std::mem::take(&mut current).finish());
        } else if let Some(anchor) = line.strip_prefix("@@") {
            hunks.extend(std::mem::take(&mut current).finish());
            let anchor = anchor.trim_start();
            current.anchor = (!anchor.is_empty()).then(|| anchor.to_owned());
        } else if line.is_empty() {
            current.lines.push(HunkLine::Context(String::new()));
        } else if let Some(text) = line.strip_prefix(' ') {
            current.lines.push(HunkLine::Context(text.to_owned()));
        } else if let Some(text) = line.strip_prefix('+') {
            current.lines.push(HunkLine::Insert(text.to_owned()));
        } else if let Some(text) = line.strip_prefix('-') {
            current.lines.push(HunkLine::Delete(text.to_owned()));
        }
    }
    hunks.extend(current.finish());
    Action::Update {
        path: path.to_owned(),
        new_path,
        hunks,
    }
}

#[cfg(test)]
mod tests {
    use super::{Action, HunkLine, parse};

    #[test]
    fn splits_hunks_at_markers_and_keeps_anchors() {
        let actions = parse(
            "*** Update File: a\n@@ fn x\n c\n-d\n+e\n@@\n f\n*** End of File\n*** End Patch",
        )
        .unwrap_err();
        // Missing Begin marker with a trailing End marker stays an error.
        assert!(actions.to_string().contains("Invalid patch format"));

        let actions =
            parse("*** Begin Patch\n*** Update File: a\n@@ fn x\n c\n-d\n+e\n@@\n f\n*** End of File\n*** End Patch")
                .expect("parses");
        let Action::Update { hunks, .. } = &actions[0] else {
            panic!("expected update");
        };
        assert_eq!(hunks.len(), 2);
        assert_eq!(hunks[0].anchor.as_deref(), Some("fn x"));
        assert_eq!(hunks[0].lines[1], HunkLine::Delete("d".to_owned()));
        assert!(hunks[1].at_end_of_file && hunks[1].anchor.is_none());
    }

    #[test]
    fn trims_blank_context_at_hunk_edges_only() {
        let actions = parse("*** Begin Patch\n*** Update File: a\n@@\n\n x\n\n y\n\n*** End Patch")
            .expect("parses");
        let Action::Update { hunks, .. } = &actions[0] else {
            panic!("expected update");
        };
        assert_eq!(
            hunks[0].lines,
            [
                HunkLine::Context("x".to_owned()),
                HunkLine::Context(String::new()),
                HunkLine::Context("y".to_owned()),
            ]
        );
    }

    #[test]
    fn wraps_a_bare_directive_and_unwraps_a_fence() {
        assert!(parse("*** Add File: a.txt\n+hi").is_ok());
        assert!(parse("```\n*** Update File: a.txt\n@@\n-a\n+b\n```").is_ok());
        assert!(parse("*** Begin Patch\n*** Update File: a.txt\n@@\n ctx\n+new").is_err());
        assert!(parse("*** Update File: a.txt\n@@\n ctx\n+new\n*** End Patch").is_err());
    }
}
