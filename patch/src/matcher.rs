//! Hunk matching: places each hunk's old side in the file and builds the
//! patched lines.
//!
//! A hunk's old side (context and deleted lines, in order) must match the file
//! contiguously, at or after the previous hunk's end and after the `@@` anchor
//! line when one is given. Zero matches is a conflict and more than one is
//! ambiguous; the matcher never picks an occurrence on its own.

use std::{borrow::Cow, collections::HashSet, ops::Range};

use crate::{
    error::PatchError,
    parse::{Hunk, HunkLine},
};

/// How lines are compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Whitespace {
    /// Byte-for-byte equality.
    Strict,
    /// Leading and trailing whitespace is ignored and inner runs collapse.
    Lenient,
}

impl Whitespace {
    fn key(self, line: &str) -> Cow<'_, str> {
        match self {
            Self::Strict => Cow::Borrowed(line),
            Self::Lenient => {
                let mut key = String::with_capacity(line.len());
                for word in line.split_whitespace() {
                    if !key.is_empty() {
                        key.push(' ');
                    }
                    key.push_str(word);
                }
                Cow::Owned(key)
            }
        }
    }
}

/// Matches `hunks` strictly first and leniently when the strict match is a
/// conflict or ambiguous, returning the start index of each hunk and the
/// mode that placed them.
pub fn place_tolerant(
    path: &str,
    file: &[String],
    hunks: &[Hunk],
) -> Result<(Vec<usize>, Whitespace), PatchError> {
    match place_hunks(path, file, hunks, Whitespace::Strict) {
        Err(PatchError::Conflict { .. } | PatchError::Ambiguous { .. }) => {
            place_hunks(path, file, hunks, Whitespace::Lenient)
                .map(|placement| (placement, Whitespace::Lenient))
        }
        other => other.map(|placement| (placement, Whitespace::Strict)),
    }
}

/// Applies `hunks` with [`place_tolerant`] and returns the patched lines.
pub fn apply_tolerant(
    path: &str,
    file: &[String],
    hunks: &[Hunk],
) -> Result<Vec<String>, PatchError> {
    place_tolerant(path, file, hunks).map(|(placement, _)| build(file, hunks, &placement))
}

/// Where each hunk of an update landed in a file, found without changing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Located {
    /// The comparison that placed the hunks: [`Whitespace::Strict`] when an
    /// exact match placed them, [`Whitespace::Lenient`] when only the
    /// whitespace-tolerant fallback did or when no placement was found.
    pub whitespace: Whitespace,
    /// One entry per hunk, in order: the lines of the current file its old
    /// side occupies (zero-based, end exclusive; empty for a pure insertion),
    /// or `None` when the hunk is unlocated.
    pub ranges: Vec<Option<Range<usize>>>,
}

/// Locates every hunk of an update in `file` without applying anything.
///
/// This is the matching [`apply_tolerant`] performs, strict first and then
/// whitespace-tolerant: when the hunks place together, every range is the one
/// the patch would change. When they do not place together, each hunk is
/// located on its own and is `None` unless it matches exactly one place.
#[must_use]
pub fn locate_hunks(file: &[String], hunks: &[Hunk]) -> Located {
    if let Ok((placement, whitespace)) = place_tolerant("", file, hunks) {
        return Located {
            whitespace,
            ranges: hunks
                .iter()
                .zip(placement)
                .map(|(hunk, start)| Some(start..start + old_len(hunk)))
                .collect(),
        };
    }
    let ranges = hunks
        .iter()
        .map(|hunk| {
            place_tolerant("", file, std::slice::from_ref(hunk))
                .ok()
                .map(|(placement, _)| placement[0]..placement[0] + old_len(hunk))
        })
        .collect();
    Located {
        whitespace: Whitespace::Lenient,
        ranges,
    }
}

/// The number of file lines a hunk's old side spans.
fn old_len(hunk: &Hunk) -> usize {
    hunk.lines
        .iter()
        .filter(|line| line.old_side().is_some())
        .count()
}

/// Finds the one consistent placement of `hunks`: the start index of each.
///
/// The hunks are placed in file order, each at or after the previous hunk's
/// end and after its `@@` anchor line. A placement exists only when exactly
/// one placement of all hunks is consistent, so a hunk that is ambiguous alone
/// is resolved by the hunks after it.
///
/// `path` only labels errors.
fn place_hunks(
    path: &str,
    file: &[String],
    hunks: &[Hunk],
    mode: Whitespace,
) -> Result<Vec<usize>, PatchError> {
    let keys: Vec<Cow<'_, str>> = file.iter().map(|line| mode.key(line)).collect();
    let olds: Vec<Vec<Cow<'_, str>>> = hunks
        .iter()
        .map(|hunk| {
            hunk.lines
                .iter()
                .filter_map(HunkLine::old_side)
                .map(|line| mode.key(line))
                .collect()
        })
        .collect();
    let mut search = Search {
        file,
        keys: &keys,
        olds: &olds,
        hunks,
        mode,
        solutions: Vec::new(),
        dead: HashSet::new(),
        failure: None,
    };
    search.place(0, 0, &mut Vec::new());

    match search.solutions.as_slice() {
        [] => {
            let (hunk, reason) = search.failure.expect("no solution implies a failed hunk");
            Err(PatchError::Conflict {
                path: path.to_owned(),
                hunk: hunk + 1,
                reason,
            })
        }
        [only] => Ok(only.clone()),
        [first, second, ..] => {
            let index = first
                .iter()
                .zip(second)
                .position(|(a, b)| a != b)
                .expect("distinct solutions differ");
            let cursor = if index == 0 {
                0
            } else {
                first[index - 1] + olds[index - 1].len()
            };
            let starts = search
                .candidates(index, cursor)
                .expect("the hunk was placed in both solutions");
            Err(PatchError::Ambiguous {
                path: path.to_owned(),
                hunk: index + 1,
                starts: starts.iter().map(|start| start + 1).collect(),
            })
        }
    }
}

/// Backtracking placement of hunks, stopping at the second solution.
struct Search<'a> {
    file: &'a [String],
    keys: &'a [Cow<'a, str>],
    olds: &'a [Vec<Cow<'a, str>>],
    hunks: &'a [Hunk],
    mode: Whitespace,
    /// Complete placements found so far (start index per hunk), at most two.
    solutions: Vec<Vec<usize>>,
    /// `(hunk, cursor)` states known to have no solution below them.
    dead: HashSet<(usize, usize)>,
    /// The furthest hunk that could not be placed, with the reason.
    failure: Option<(usize, String)>,
}

impl Search<'_> {
    /// Where hunk `index` may start when the previous hunk ended at `cursor`;
    /// `Err` carries the reason when its anchor is missing.
    fn candidates(&self, index: usize, cursor: usize) -> Result<Vec<usize>, String> {
        let hunk = &self.hunks[index];
        let lower = if let Some(anchor) = &hunk.anchor {
            let anchor_key = self.mode.key(anchor);
            let found = self.keys[cursor..]
                .iter()
                .position(|key| *key == anchor_key);
            let Some(found) = found else {
                return Err(format!(
                    "the anchor line {anchor:?} does not exist at or after file line {}",
                    cursor + 1
                ));
            };
            cursor + found + 1
        } else {
            cursor
        };
        let starts = candidates(self.keys, &self.olds[index], lower, hunk.at_end_of_file);
        if starts.is_empty() {
            return Err(diagnose(
                self.file,
                self.keys,
                &self.olds[index],
                lower,
                hunk,
            ));
        }
        Ok(starts)
    }

    /// Places hunks `index..`; returns whether any solution lies below.
    fn place(&mut self, index: usize, cursor: usize, placed: &mut Vec<usize>) -> bool {
        if index == self.hunks.len() {
            self.solutions.push(placed.clone());
            return true;
        }
        if self.dead.contains(&(index, cursor)) {
            return false;
        }
        let starts = match self.candidates(index, cursor) {
            Ok(starts) => starts,
            Err(reason) => {
                if self
                    .failure
                    .as_ref()
                    .is_none_or(|(deepest, _)| index > *deepest)
                {
                    self.failure = Some((index, reason));
                }
                self.dead.insert((index, cursor));
                return false;
            }
        };
        let mut found = false;
        for start in starts {
            placed.push(start);
            found |= self.place(index + 1, start + self.olds[index].len(), placed);
            placed.pop();
            if self.solutions.len() >= 2 {
                return true;
            }
        }
        if !found {
            self.dead.insert((index, cursor));
        }
        found
    }
}

/// Builds the patched lines from a consistent placement.
fn build(file: &[String], hunks: &[Hunk], placement: &[usize]) -> Vec<String> {
    let mut out = Vec::with_capacity(file.len());
    let mut cursor = 0;
    for (hunk, &start) in hunks.iter().zip(placement) {
        out.extend_from_slice(&file[cursor..start]);
        let mut at = start;
        for line in &hunk.lines {
            match line {
                HunkLine::Context(_) => {
                    out.push(file[at].clone());
                    at += 1;
                }
                HunkLine::Delete(_) => at += 1,
                HunkLine::Insert(text) => out.push(text.clone()),
            }
        }
        cursor = at;
    }
    out.extend_from_slice(&file[cursor..]);
    out
}

/// Every index at or after `lower` where `old` matches `keys` contiguously.
///
/// An empty old side (pure insertion) has exactly one place: the end of the
/// file for an end-of-file hunk, otherwise `lower`.
fn candidates(
    keys: &[Cow<'_, str>],
    old: &[Cow<'_, str>],
    lower: usize,
    at_end: bool,
) -> Vec<usize> {
    if old.is_empty() {
        return vec![if at_end { keys.len() } else { lower }];
    }
    let Some(last) = keys.len().checked_sub(old.len()) else {
        return Vec::new();
    };
    (lower..=last)
        .filter(|&start| keys[start..start + old.len()] == *old)
        .filter(|&start| !at_end || start + old.len() == keys.len())
        .collect()
}

/// Explains why `old` fits nowhere: names the first line of the best partial
/// match that did not match.
fn diagnose(
    file: &[String],
    keys: &[Cow<'_, str>],
    old: &[Cow<'_, str>],
    lower: usize,
    hunk: &Hunk,
) -> String {
    let old_lines: Vec<&str> = hunk.lines.iter().filter_map(HunkLine::old_side).collect();
    let mut best = (0, lower);
    for start in lower..keys.len() {
        let matched = keys[start..]
            .iter()
            .zip(old)
            .take_while(|(key, wanted)| key == wanted)
            .count();
        if matched > best.0 {
            best = (matched, start);
        }
    }
    let (matched, start) = best;
    if matched == old.len() {
        return format!(
            "the context/deleted lines match at file line {} but `*** End of File` requires them to end the file",
            start + 1
        );
    }
    let wanted = old_lines[matched];
    match file.get(start + matched) {
        Some(found) if matched > 0 => format!(
            "context/deleted line {} of the hunk, {wanted:?}, does not match; the closest candidate starts at file line {} and has {found:?} there",
            matched + 1,
            start + 1,
        ),
        _ => format!(
            "the first context/deleted line, {wanted:?}, does not exist at or after file line {}",
            lower + 1
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{Whitespace, build, place_hunks};
    use crate::parse::{Action, parse};

    fn run(file: &str, hunks: &str, mode: Whitespace) -> Result<String, String> {
        let patch = format!("*** Begin Patch\n*** Update File: f\n{hunks}\n*** End Patch");
        let Action::Update { hunks, .. } = parse(&patch).expect("parses").remove(0) else {
            panic!("expected update");
        };
        let lines: Vec<String> = file.lines().map(String::from).collect();
        place_hunks("f", &lines, &hunks, mode)
            .map(|placement| build(&lines, &hunks, &placement).join("\n"))
            .map_err(|error| error.to_string())
    }

    #[test]
    fn applies_several_changes_separated_by_context() {
        let out = run(
            "a\nb\nc\nd\ne",
            "@@\n a\n-b\n+B\n c\n-d\n+D\n e",
            Whitespace::Strict,
        );
        assert_eq!(out.as_deref(), Ok("a\nB\nc\nD\ne"));
    }

    #[test]
    fn applies_hunks_in_file_order() {
        let out = run(
            "1\n2\n3\n4\n5",
            "@@\n-2\n+two\n@@\n-4\n+four",
            Whitespace::Strict,
        );
        assert_eq!(out.as_deref(), Ok("1\ntwo\n3\nfour\n5"));
    }

    #[test]
    fn anchor_selects_a_later_occurrence() {
        let file = "fn a\nx\nfn b\nx";
        let out = run(file, "@@ fn b\n-x\n+y", Whitespace::Strict);
        assert_eq!(out.as_deref(), Ok("fn a\nx\nfn b\ny"));
        let unanchored = run(file, "@@\n-x\n+y", Whitespace::Strict).unwrap_err();
        assert!(unanchored.contains("Ambiguous patch"), "{unanchored}");
    }

    #[test]
    fn later_hunk_resolves_an_ambiguous_earlier_one() {
        // `-x` alone matches lines 1 and 3, but `-a` can only follow the first.
        let out = run("x\na\nx\nb", "@@\n-x\n+X\n@@\n-a\n+A", Whitespace::Strict);
        assert_eq!(out.as_deref(), Ok("X\nA\nx\nb"));
    }

    #[test]
    fn two_consistent_placements_are_ambiguous() {
        let error = run(
            "x
x
x",
            "@@
-x
+X
@@
-x
+Y",
            Whitespace::Strict,
        )
        .unwrap_err();
        assert!(error.contains("Ambiguous patch"), "{error}");
        assert!(error.contains("hunk 2"), "{error}");
    }

    #[test]
    fn ambiguity_lists_every_place() {
        let error = run("s\ns\ns", "@@\n-s\n+u", Whitespace::Strict).unwrap_err();
        assert!(error.contains("lines 1, 2, 3"), "{error}");
    }

    #[test]
    fn conflict_names_the_hunk_and_the_first_unmatched_line() {
        let error = run(
            "a\nb\nc\nd",
            "@@\n-a\n+A\n@@\n c\n-nope\n+x",
            Whitespace::Strict,
        )
        .unwrap_err();
        assert!(error.contains("hunk 2"), "{error}");
        assert!(error.contains("\"nope\""), "{error}");
    }

    #[test]
    fn lenient_mode_ignores_whitespace_differences() {
        assert!(run("  a  \nb", "@@\n a\n-b\n+B", Whitespace::Strict).is_err());
        let out = run("  a  \nb", "@@\n a\n-b\n+B", Whitespace::Lenient);
        assert_eq!(out.as_deref(), Ok("  a  \nB"));
    }

    #[test]
    fn end_of_file_hunk_must_reach_the_end() {
        let out = run("x\ny\nx", "@@\n-x\n+z\n*** End of File", Whitespace::Strict);
        assert_eq!(out.as_deref(), Ok("x\ny\nz"));
        let error = run("x\ny", "@@\n-x\n+z\n*** End of File", Whitespace::Strict).unwrap_err();
        assert!(error.contains("End of File"), "{error}");
    }
}
