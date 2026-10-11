//! Preview stage for output whose full text is stored on disk.
//!
//! The model sees the head and the tail of the stored text with an explicit
//! marker for the omitted middle. Every line number a marker mentions is a
//! line number of the stored file, so the model can jump straight to the
//! omitted range with `sed`/`head`/`tail` instead of reading the whole file.
//!
//! Because the full text is kept, this stage may also drop content: runs of
//! identical consecutive lines are replaced by one copy plus a marker naming
//! the range they covered.

use askama::Template;

/// The head of a preview gets this fraction of the line and byte budget --
/// one third -- and the tail gets the rest.
///
/// Compiler errors, panics, test failures and stack traces sit at the end of
/// a command's output, while the head is usually a banner, so the tail is
/// worth twice the head.
const HEAD_BUDGET_DIVISOR: usize = 3;

#[derive(Template)]
#[template(path = "omitted_lines_marker.txt", escape = "none")]
struct OmittedLinesMarker {
    first: usize,
    last: usize,
    total: usize,
}

#[derive(Template)]
#[template(path = "repeated_lines_marker.txt", escape = "none")]
struct RepeatedLinesMarker {
    first: usize,
    last: usize,
    count: usize,
}

/// One preview item: a line of the stored text, plus a marker when it stands
/// for a run of identical lines.
struct Segment<'a> {
    /// 1-based line number of this item in the stored text.
    first: usize,
    /// 1-based line number of the last stored line this item stands for.
    last: usize,
    /// The line itself, including its trailing newline when it has one.
    line: &'a str,
    /// Repeat marker, present only when `last > first`.
    marker: Option<String>,
}

impl Segment<'_> {
    /// Bytes this item contributes to the preview.
    fn bytes(&self) -> usize {
        self.line.len() + self.marker.as_ref().map_or(0, |m| m.len() + 1)
    }

    /// Lines this item contributes to the preview.
    fn lines(&self) -> usize {
        1 + usize::from(self.marker.is_some())
    }

    fn render_into(&self, out: &mut String) {
        out.push_str(self.line);
        if !self.line.ends_with('\n') {
            out.push('\n');
        }
        if let Some(marker) = &self.marker {
            out.push_str(marker);
            out.push('\n');
        }
    }
}

/// Renders the repeat marker for a run of `count` identical lines spanning
/// stored lines `first..=last`, or `None` when the run is better shown as is.
///
/// Describing a run only helps when the description is smaller than the
/// repeats it stands in for, so a pair of short lines stays verbatim.
fn collapsed_run_marker(first: usize, last: usize, count: usize, line: &str) -> Option<String> {
    if count < 2 {
        return None;
    }
    let marker = RepeatedLinesMarker { first, last, count }
        .render()
        .expect("repeated lines marker template must render");
    (marker.len() + 1 < line.len() * (count - 1)).then_some(marker)
}

/// Splits `text` into preview items, collapsing runs of identical consecutive
/// lines whenever the marker is shorter than the lines it replaces.
fn segments(text: &str) -> Vec<Segment<'_>> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let mut segments = Vec::with_capacity(lines.len());
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let mut end = index + 1;
        while end < lines.len() && lines[end] == line {
            end += 1;
        }
        let count = end - index;
        match collapsed_run_marker(index + 1, end, count, line) {
            Some(marker) => segments.push(Segment {
                first: index + 1,
                last: end,
                line,
                marker: Some(marker),
            }),
            None => segments.extend((index..end).map(|number| Segment {
                first: number + 1,
                last: number + 1,
                line,
                marker: None,
            })),
        }
        index = end;
    }
    segments
}

/// Renders a head+tail preview of `text`, which the caller has stored on disk
/// verbatim.
///
/// The preview holds at most `max_lines` lines and `byte_budget` bytes, split
/// one third to the head and two thirds to the tail. When nothing is left
/// out, `text` is returned unchanged. Runs of identical lines are collapsed
/// and the omitted middle is announced by a marker, both carrying line
/// numbers of `text` itself.
#[must_use]
pub fn stored_preview(text: &str, max_lines: usize, byte_budget: usize) -> String {
    let total_lines = text.split_inclusive('\n').count();
    if total_lines <= max_lines && text.len() <= byte_budget {
        return text.to_owned();
    }

    let segments = segments(text);
    let head_line_cap = max_lines / HEAD_BUDGET_DIVISOR;
    let head_byte_cap = byte_budget / HEAD_BUDGET_DIVISOR;
    let tail_line_cap = max_lines - head_line_cap;
    let tail_byte_cap = byte_budget - head_byte_cap;

    let head_end = take_head(&segments, head_line_cap, head_byte_cap);
    let tail_start = take_tail(&segments[head_end..], tail_line_cap, tail_byte_cap) + head_end;

    if head_end == 0 && tail_start == segments.len() {
        return cut_single_line(text, total_lines, byte_budget);
    }
    if head_end == tail_start {
        // Every segment fits: the only reduction is the collapsed runs.
        let mut out = String::new();
        render_range(&segments, &mut out);
        return out;
    }

    let omitted = OmittedLinesMarker {
        first: head_end
            .checked_sub(1)
            .map_or(1, |last_head| segments[last_head].last + 1),
        last: segments[tail_start].first - 1,
        total: total_lines,
    };
    let mut out = String::new();
    render_range(&segments[..head_end], &mut out);
    out.push_str(
        &omitted
            .render()
            .expect("omitted lines marker template must render"),
    );
    out.push('\n');
    render_range(&segments[tail_start..], &mut out);
    out
}

/// Number of leading segments that fit in the head budget.
fn take_head(segments: &[Segment<'_>], line_cap: usize, byte_cap: usize) -> usize {
    let mut lines = 0;
    let mut bytes = 0;
    let mut count = 0;
    for segment in segments {
        lines += segment.lines();
        bytes += segment.bytes();
        if lines > line_cap || bytes > byte_cap {
            break;
        }
        count += 1;
    }
    count
}

/// Index of the first trailing segment that fits in the tail budget.
fn take_tail(segments: &[Segment<'_>], line_cap: usize, byte_cap: usize) -> usize {
    let mut lines = 0;
    let mut bytes = 0;
    let mut start = segments.len();
    for (index, segment) in segments.iter().enumerate().rev() {
        lines += segment.lines();
        bytes += segment.bytes();
        if lines > line_cap || bytes > byte_cap {
            break;
        }
        start = index;
    }
    start
}

fn render_range(segments: &[Segment<'_>], out: &mut String) {
    for segment in segments {
        segment.render_into(out);
    }
}

/// Handles the degenerate case where not even one line fits the budget: cut
/// inside the first line at a character boundary and announce the rest.
fn cut_single_line(text: &str, total_lines: usize, byte_budget: usize) -> String {
    let mut cut = byte_budget.min(text.len());
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let head = text
        .get(..cut)
        .expect("cut is aligned to a character boundary");
    let mut out = String::with_capacity(cut + 128);
    out.push_str(head);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    let omitted = OmittedLinesMarker {
        first: head.split_inclusive('\n').count().max(1),
        last: total_lines,
        total: total_lines,
    };
    out.push_str(
        &omitted
            .render()
            .expect("omitted lines marker template must render"),
    );
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(count: usize) -> String {
        use std::fmt::Write;
        (1..=count).fold(String::new(), |mut acc, index| {
            writeln!(acc, "line {index}").expect("write to String");
            acc
        })
    }

    #[test]
    fn returns_text_unchanged_when_it_fits() {
        let text = "one\ntwo\nthree\n";
        assert_eq!(stored_preview(text, 10, usize::MAX), text);
    }

    #[test]
    fn keeps_head_and_tail_with_an_omitted_marker() {
        let text = numbered(90);
        let preview = stored_preview(&text, 9, usize::MAX);
        assert!(preview.starts_with("line 1\n"), "{preview}");
        assert!(preview.ends_with("line 90\n"), "{preview}");
        assert!(
            preview.contains("[... lines 4-84 of 90 omitted"),
            "{preview}"
        );
    }

    #[test]
    fn omitted_range_joins_head_and_tail_exactly() {
        let text = numbered(100);
        let preview = stored_preview(&text, 12, usize::MAX);
        let lines: Vec<&str> = preview.lines().collect();
        let marker_index = lines
            .iter()
            .position(|line| line.starts_with("[... lines"))
            .expect("marker present");
        let last_head = lines[marker_index - 1];
        let first_tail = lines[marker_index + 1];
        let stored: Vec<&str> = text.lines().collect();
        // The marker's range starts right after the last head line and ends
        // right before the first tail line, in stored-file numbering.
        let first_omitted: usize = lines[marker_index]
            .split_whitespace()
            .nth(2)
            .and_then(|range| range.split('-').next())
            .and_then(|n| n.parse().ok())
            .expect("first omitted line number");
        let last_omitted: usize = lines[marker_index]
            .split_whitespace()
            .nth(2)
            .and_then(|range| range.split('-').nth(1))
            .and_then(|n| n.parse().ok())
            .expect("last omitted line number");
        assert_eq!(stored[first_omitted - 2], last_head);
        assert_eq!(stored[last_omitted], first_tail);
    }

    #[test]
    fn tail_gets_the_larger_share_of_the_budget() {
        let text = numbered(100);
        let preview = stored_preview(&text, 12, usize::MAX);
        let head = preview
            .lines()
            .take_while(|line| !line.starts_with("[... lines"))
            .count();
        let tail = preview
            .lines()
            .skip_while(|line| !line.starts_with("[... lines"))
            .count()
            - 1;
        assert!(tail > head, "tail {tail} should exceed head {head}");
    }

    #[test]
    fn collapses_identical_runs_with_a_repeat_marker() {
        let mut text = String::from("start\n");
        for _ in 0..200 {
            text.push_str("retrying connection to registry, attempt failed\n");
        }
        text.push_str("fatal: giving up\n");
        let preview = stored_preview(&text, 30, usize::MAX);
        assert!(preview.contains("repeated 200 times"), "{preview}");
        assert!(preview.ends_with("fatal: giving up\n"), "{preview}");
        assert_eq!(
            preview
                .matches("retrying connection to registry, attempt failed")
                .count(),
            1,
            "{preview}"
        );
    }

    #[test]
    fn does_not_collapse_a_short_run_that_the_marker_would_not_shrink() {
        let text = "a\na\nb\n";
        let preview = stored_preview(text, usize::MAX, 4);
        assert!(!preview.contains("repeated"), "{preview}");
    }

    #[test]
    fn respects_the_byte_budget() {
        let text = numbered(500);
        let preview = stored_preview(&text, usize::MAX, 400);
        let body: usize = preview
            .lines()
            .filter(|line| !line.starts_with("[... lines"))
            .map(|line| line.len() + 1)
            .sum();
        assert!(body <= 400, "body {body} over budget: {preview}");
    }

    #[test]
    fn cuts_inside_a_single_oversized_line_at_a_char_boundary() {
        let text = format!("{}\n", "héllo".repeat(100));
        let preview = stored_preview(&text, usize::MAX, 51);
        assert!(preview.is_char_boundary(preview.len()));
        assert!(preview.contains("omitted"), "{preview}");
    }

    #[test]
    fn text_without_a_trailing_newline_still_previews() {
        let text = "one\ntwo\nthree";
        let preview = stored_preview(text, 2, usize::MAX);
        assert!(preview.ends_with("three\n"), "{preview}");
    }

    #[test]
    fn empty_text_previews_empty() {
        assert_eq!(stored_preview("", 10, 10), "");
    }
}
