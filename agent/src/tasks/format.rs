//! The `tasks.md` format.
//!
//! The document holds current state only, in up to five `##` sections:
//! `Roots` (the root issues the session serves), `Doing` (work in progress),
//! `Queued` (work waiting to start), `Notes` and `Decisions`. The first three
//! hold `- ` rows, one per item, with indented lines continuing the row
//! above; the last two are free text, which ends only at a `## ` heading
//! naming a section outside a fenced code block. A `# ` title may precede
//! them.

use super::reference::{WorkRef, tokens};

/// A section of `tasks.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    /// The root issues the session serves.
    Roots,
    /// Work in progress.
    Doing,
    /// Work waiting to start.
    Queued,
    /// Free-text notes.
    Notes,
    /// The user's decisions, quoted verbatim.
    Decisions,
}

impl Section {
    const ALL: [Self; 5] = [
        Self::Roots,
        Self::Doing,
        Self::Queued,
        Self::Notes,
        Self::Decisions,
    ];

    /// The section's heading text, after `## `.
    #[must_use]
    pub const fn heading(self) -> &'static str {
        match self {
            Self::Roots => "Roots",
            Self::Doing => "Doing",
            Self::Queued => "Queued",
            Self::Notes => "Notes",
            Self::Decisions => "Decisions",
        }
    }

    /// Whether the section holds rows rather than free text.
    const fn holds_rows(self) -> bool {
        matches!(self, Self::Roots | Self::Doing | Self::Queued)
    }

    /// Whether a row of this section is open work, which can hold a turn.
    #[must_use]
    pub const fn is_open_work(self) -> bool {
        matches!(self, Self::Doing | Self::Queued)
    }
}

/// One row of a row section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    section: Section,
    line: usize,
    text: String,
}

impl Row {
    /// The section it is in.
    #[must_use]
    pub const fn section(&self) -> Section {
        self.section
    }

    /// Its first line's number, from 1.
    #[must_use]
    pub const fn line(&self) -> usize {
        self.line
    }

    /// Its text without the `- ` marker, continuation lines joined by
    /// newlines.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The first line of its text, which names it in an alert.
    #[must_use]
    pub fn summary(&self) -> &str {
        self.text.lines().next().unwrap_or_default()
    }

    /// The issues and pull requests it names, in order.
    #[must_use]
    pub fn references(&self) -> Vec<WorkRef> {
        WorkRef::parse_all(&self.text)
    }

    /// Whether it names `id` — a question, job or subagent id — as a word.
    #[must_use]
    pub fn names(&self, id: &str) -> bool {
        tokens(&self.text).any(|token| token == id)
    }

    /// The payload of its `BLOCKING <what>` mark: the token right after
    /// `BLOCKING`, a request id or the first word of a free-text reason,
    /// split by the same rules as [`names`](Self::names), so markdown and
    /// punctuation around either word do not hide it. A trailing `BLOCKING`
    /// alone is no mark.
    #[must_use]
    pub fn blocking_mark(&self) -> Option<&str> {
        let mut words = tokens(&self.text);
        words.find(|word| *word == "BLOCKING")?;
        words.next()
    }
}

/// A way a line breaks the format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// Text before the first section.
    OutsideSection,
    /// A `##` heading that names no section.
    UnknownSection(String),
    /// A section heading that already occurred.
    RepeatedSection(Section),
    /// A line in a row section that is not a row.
    NotARow,
    /// A checkbox row: rows are not ticked, a finished one is deleted.
    Checkbox,
}

/// A violation and the line it is on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineViolation {
    /// The line number, from 1.
    pub line: usize,
    /// What is wrong with it.
    pub violation: Violation,
}

/// A parsed `tasks.md`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TasksFile {
    rows: Vec<Row>,
    violations: Vec<LineViolation>,
    lines: usize,
}

impl TasksFile {
    /// Parses `content`, collecting every violation rather than stopping at
    /// the first.
    #[must_use]
    pub fn parse(content: &str) -> Self {
        let mut file = Self::default();
        let mut section: Option<Section> = None;
        let mut seen: Vec<Section> = Vec::new();
        // Lines after an unknown heading belong to it and are not reported
        // again one by one.
        let mut in_unknown = false;
        let mut titled = false;
        // The fence a free-text section is inside: its character and length.
        let mut fence: Option<(char, usize)> = None;
        for (index, line) in content.lines().enumerate() {
            let number = index + 1;
            file.lines = number;
            let free_text = section.is_some_and(|current| !current.holds_rows());
            if free_text {
                match (fence, Fence::of(line)) {
                    (None, Some(opening)) => {
                        fence = Some((opening.marker, opening.length));
                        continue;
                    }
                    (Some((marker, length)), Some(closing)) if closing.closes(marker, length) => {
                        fence = None;
                        continue;
                    }
                    (Some(_), _) => continue,
                    (None, None) => {}
                }
            }
            if line.trim().is_empty() {
                continue;
            }
            if let Some(heading) = line.strip_prefix("## ") {
                let heading = heading.trim();
                let named = Section::ALL
                    .into_iter()
                    .find(|candidate| candidate.heading() == heading);
                // Free text holds any other `## ` line as its own text.
                if free_text && named.is_none() {
                    continue;
                }
                section = named;
                in_unknown = section.is_none();
                if let Some(found) = section {
                    if seen.contains(&found) {
                        file.violate(number, Violation::RepeatedSection(found));
                    }
                    seen.push(found);
                } else {
                    file.violate(number, Violation::UnknownSection(heading.to_string()));
                }
                continue;
            }
            if in_unknown {
                continue;
            }
            match section {
                None => {
                    if !titled && seen.is_empty() && line.starts_with("# ") {
                        titled = true;
                    } else {
                        file.violate(number, Violation::OutsideSection);
                    }
                }
                Some(current) if !current.holds_rows() => {}
                Some(current) => file.row_line(current, number, line),
            }
        }
        file
    }

    fn violate(&mut self, line: usize, violation: Violation) {
        self.violations.push(LineViolation { line, violation });
    }

    /// Takes one line of a row section.
    fn row_line(&mut self, section: Section, number: usize, line: &str) {
        if let Some(text) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            let text = text.trim();
            if ["[ ]", "[x]", "[X]"]
                .iter()
                .any(|checkbox| text.starts_with(checkbox))
            {
                self.violate(number, Violation::Checkbox);
                return;
            }
            self.rows.push(Row {
                section,
                line: number,
                text: text.to_string(),
            });
            return;
        }
        let continues = line.starts_with([' ', '\t'])
            && self.rows.last().is_some_and(|row| row.section == section);
        match self.rows.last_mut() {
            Some(row) if continues => {
                row.text.push('\n');
                row.text.push_str(line.trim());
            }
            _ => self.violate(number, Violation::NotARow),
        }
    }

    /// Every row, in document order.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Every violation, in document order.
    #[must_use]
    pub fn violations(&self) -> &[LineViolation] {
        &self.violations
    }

    /// The number of lines the document has.
    #[must_use]
    pub const fn lines(&self) -> usize {
        self.lines
    }

    /// The references the `Roots` rows name, each once, in order.
    #[must_use]
    pub fn roots(&self) -> Vec<WorkRef> {
        let mut roots: Vec<WorkRef> = Vec::new();
        for reference in self
            .rows
            .iter()
            .filter(|row| row.section == Section::Roots)
            .flat_map(Row::references)
        {
            if !roots.contains(&reference) {
                roots.push(reference);
            }
        }
        roots
    }

    /// The rows that name `reference`.
    pub fn rows_naming<'a>(&'a self, reference: &'a WorkRef) -> impl Iterator<Item = &'a Row> {
        self.rows.iter().filter(move |row| {
            row.references()
                .iter()
                .any(|named| named.matches(reference))
        })
    }

    /// The rows that name `id` as a word.
    pub fn rows_naming_id<'a>(&'a self, id: &'a str) -> impl Iterator<Item = &'a Row> {
        self.rows.iter().filter(move |row| row.names(id))
    }
}

/// A code fence line: three or more backticks or tildes, indented by at
/// most three spaces.
struct Fence<'a> {
    marker: char,
    length: usize,
    /// What follows the marker: an info string on an opening fence; a
    /// closing fence has none.
    rest: &'a str,
}

impl<'a> Fence<'a> {
    fn of(line: &'a str) -> Option<Self> {
        let unindented = line.trim_start_matches(' ');
        if line.len() - unindented.len() > 3 {
            return None;
        }
        let marker = unindented
            .chars()
            .next()
            .filter(|c| matches!(c, '`' | '~'))?;
        let rest = unindented.trim_start_matches(marker);
        let length = unindented.len() - rest.len();
        (length >= 3).then_some(Self {
            marker,
            length,
            rest,
        })
    }

    /// Whether this line closes a fence opened with `length` of `marker`.
    fn closes(&self, marker: char, length: usize) -> bool {
        self.marker == marker && self.length >= length && self.rest.trim().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WELL_FORMED: &str = "# Ship search\n\n## Roots\n- #129 event-driven tasks.md\n\n## Doing\n- #130 wire the backend, PR #12 open\n  next: answer review\n- #131 pick a db BLOCKING req-7\n\n## Queued\n- #132 docs\n\n## Notes\nfree text, - [ ] even this\n### a heading\n\n## Decisions\n> \"use sqlite\"\n";

    #[test]
    fn a_well_formed_document_has_rows_and_no_violations() {
        let file = TasksFile::parse(WELL_FORMED);
        assert_eq!(file.violations(), &[]);
        let rows: Vec<_> = file
            .rows()
            .iter()
            .map(|row| (row.section(), row.line(), row.summary()))
            .collect();
        assert_eq!(
            rows,
            vec![
                (Section::Roots, 4, "#129 event-driven tasks.md"),
                (Section::Doing, 7, "#130 wire the backend, PR #12 open"),
                (Section::Doing, 9, "#131 pick a db BLOCKING req-7"),
                (Section::Queued, 12, "#132 docs"),
            ]
        );
        assert!(file.rows()[1].text().ends_with("next: answer review"));
        assert_eq!(file.rows()[2].blocking_mark(), Some("req-7"));
        assert_eq!(file.roots(), vec![WorkRef::issue(129)]);
        assert_eq!(file.lines(), 19);
    }

    #[test]
    fn each_kind_of_violation_is_reported_with_its_line() {
        let file = TasksFile::parse(
            "stray\n## Doing\n- [ ] old style\nplain\n## Plans\n- hidden\n## Doing\n- ok\n",
        );
        assert_eq!(
            file.violations(),
            &[
                LineViolation {
                    line: 1,
                    violation: Violation::OutsideSection,
                },
                LineViolation {
                    line: 3,
                    violation: Violation::Checkbox,
                },
                LineViolation {
                    line: 4,
                    violation: Violation::NotARow,
                },
                LineViolation {
                    line: 5,
                    violation: Violation::UnknownSection("Plans".to_string()),
                },
                LineViolation {
                    line: 7,
                    violation: Violation::RepeatedSection(Section::Doing),
                },
            ]
        );
    }

    #[test]
    fn rows_are_found_by_reference_and_by_id() {
        let file = TasksFile::parse(WELL_FORMED);
        let merged = WorkRef::pull_request(12);
        let naming: Vec<_> = file.rows_naming(&merged).map(Row::line).collect();
        assert_eq!(naming, vec![7]);
        let asking: Vec<_> = file.rows_naming_id("req-7").map(Row::line).collect();
        assert_eq!(asking, vec![9]);
        assert_eq!(file.rows_naming_id("req").count(), 0);
    }

    #[test]
    fn a_mark_reads_through_markdown_and_punctuation_like_names() {
        for text in [
            "- pick a db `BLOCKING` `req-7`",
            "- pick a db (BLOCKING: req-7)",
            "- pick a db **BLOCKING** req-7.",
        ] {
            let file = TasksFile::parse(&format!("## Doing\n{text}\n"));
            let row = &file.rows()[0];
            assert_eq!(row.blocking_mark(), Some("req-7"), "{text}");
            assert!(row.names("req-7"), "{text}");
        }
        let trailing = TasksFile::parse("## Doing\n- pick a db `BLOCKING`\n");
        assert_eq!(trailing.rows()[0].blocking_mark(), None);
    }

    #[test]
    fn free_text_holds_headings_and_fences_of_its_own() {
        let file = TasksFile::parse(
            "## Notes\n## Findings\n```md\n## Doing\n- not a row\n```\n~~~~\n## Roots\n~~~\n~~~~\n## Decisions\n> \"## Queued is fine\"\n## Doing\n- #4 real row\n",
        );
        assert_eq!(file.violations(), &[]);
        let rows: Vec<_> = file
            .rows()
            .iter()
            .map(|row| (row.section(), row.summary()))
            .collect();
        assert_eq!(rows, vec![(Section::Doing, "#4 real row")]);
    }

    #[test]
    fn a_known_heading_outside_a_fence_still_ends_free_text() {
        let file = TasksFile::parse("## Notes\ntext\n## Notes\n");
        assert_eq!(
            file.violations(),
            &[LineViolation {
                line: 3,
                violation: Violation::RepeatedSection(Section::Notes),
            }]
        );
    }
}
