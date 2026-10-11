//! Blank-run collapsing stage.

/// A line counts as blank only when it has no characters at all.
///
/// Lines made of spaces are left alone on purpose: in a unified diff every
/// context line carries a leading space, so a blank line of the patched file
/// reaches us as `" "`. Treating that as blank would drop context lines and
/// the patch would stop applying.
const fn is_blank(line: &str) -> bool {
    line.is_empty()
}

/// Collapses every run of two or more blank lines into a single blank line.
///
/// A lone blank line is always kept, so paragraphs, hunk separators and the
/// blank context lines of a patch survive. The result ends with a newline
/// when it is non-empty.
#[must_use]
pub fn collapse_blank_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_blank_run = false;
    for line in text.lines() {
        if is_blank(line) {
            if in_blank_run {
                continue;
            }
            in_blank_run = true;
        } else {
            in_blank_run = false;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_runs_to_one_blank_line() {
        assert_eq!(
            collapse_blank_runs("alpha\n\n\n\nbeta\n"),
            "alpha\n\nbeta\n"
        );
    }

    #[test]
    fn keeps_a_single_blank_line() {
        assert_eq!(collapse_blank_runs("alpha\n\nbeta\n"), "alpha\n\nbeta\n");
    }

    #[test]
    fn keeps_space_only_lines() {
        // A diff context line for a blank source line is a single space.
        assert_eq!(collapse_blank_runs(" \n \nx\n"), " \n \nx\n");
    }

    #[test]
    fn collapses_a_trailing_run() {
        assert_eq!(collapse_blank_runs("alpha\n\n\n"), "alpha\n\n");
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(collapse_blank_runs(""), "");
    }
}
