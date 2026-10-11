//! Output cleaning pipeline for terminal tool results.
//!
//! The pipeline is a sequence of stages, each in its own module:
//!
//! 1. [`render`] renders the byte stream the way a terminal would display it,
//!    consuming escape sequences and resolving `\r` to the last frame of each
//!    line.
//! 2. [`collapse`] collapses runs of blank lines to a single blank line.
//! 3. [`json_tsv`] turns a JSON array of records into TSV.
//! 4. [`fold`] folds the bodies of blocks in output that is source code.
//! 5. [`preview`] reduces stored text to a head+tail preview for the model.
//!
//! Stages 1 and 2 are the only ones applied unconditionally: together they
//! are what a terminal would have shown, so output delivered inline -- where
//! no copy is kept -- keeps every character the command actually produced,
//! down to whether it ended with a line terminator.
//!
//! Stages 3 and 4 are lossy. They run only when the text they replace can be
//! written to disk, and the text the model reads names that file, so nothing
//! is lost. Stage 5 belongs to the stored path in [`crate::output`], which
//! owns the inline budget.

mod collapse;
mod fold;
mod json_tsv;
mod preview;
mod render;

use std::path::Path;

use askama::Template;

pub use preview::stored_preview;

#[derive(Template)]
#[template(path = "json_tsv_output.txt", escape = "none")]
struct JsonTsvOutput<'a> {
    tsv: &'a str,
    url: &'a str,
    path: &'a str,
}

#[derive(Template)]
#[template(path = "folded_code_output.txt", escape = "none")]
struct FoldedCodeOutput<'a> {
    folded: &'a str,
    url: &'a str,
    path: &'a str,
}

/// Cleans foreground terminal text for the model.
///
/// Returns `None` when the text is already what a terminal would show and no
/// stage applied, so the caller can keep the original bytes.
///
/// `store_dir` receives the original of any lossy stage; the returned text
/// names that file. The returned text is **not** yet split between inline and
/// stored delivery -- that is [`crate::output`]'s job, and the text handed to
/// it is what gets stored, so the line numbers in a preview marker refer to
/// it exactly.
///
/// # Errors
///
/// Returns an error when a lossy stage applied but its original could not be
/// written to `store_dir`.
pub async fn compress_text(store_dir: &Path, text: &str) -> std::io::Result<Option<String>> {
    if text.is_empty() {
        return Ok(None);
    }

    let cleaned = collapse::collapse_blank_runs(&render::render_terminal(text));

    if let Some(tsv) = json_tsv::to_tsv(&cleaned) {
        let (url, path) = crate::output::save_raw_to_file(store_dir, cleaned.as_bytes()).await?;
        let path = path.display().to_string();
        let rendered = JsonTsvOutput {
            tsv: tsv.trim_end_matches('\n'),
            url: &url,
            path: &path,
        }
        .render()
        .expect("json tsv output template must render");
        return Ok(Some(rendered));
    }

    if let Some(folded) = fold::fold_source_code(&cleaned) {
        let (url, path) = crate::output::save_raw_to_file(store_dir, cleaned.as_bytes()).await?;
        let path = path.display().to_string();
        let rendered = FoldedCodeOutput {
            folded: folded.trim_end_matches('\n'),
            url: &url,
            path: &path,
        }
        .render()
        .expect("folded code output template must render");
        return Ok(Some(rendered));
    }

    Ok((cleaned != text).then_some(cleaned))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn compress(text: &str) -> Option<String> {
        let dir = tempfile::tempdir().expect("tempdir");
        compress_text(dir.path(), text)
            .await
            .expect("compression succeeds")
    }

    #[tokio::test]
    async fn empty_output_is_left_alone() {
        assert!(compress("").await.is_none());
    }

    #[tokio::test]
    async fn coloured_cargo_output_has_no_escape_residue() {
        let coloured = "\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m\u{1b}[1m aither-sandbox\u{1b}[0m v0.1.0 (/src)\n\u{1b}[1m\u{1b}[31merror\u{1b}[0m\u{1b}[1m: expected `;`\u{1b}[0m\n";
        let cleaned = compress(coloured).await.expect("escapes are stripped");
        assert_eq!(
            cleaned,
            "   Compiling aither-sandbox v0.1.0 (/src)\nerror: expected `;`\n"
        );
        for residue in ['\u{1b}', '\u{7}'] {
            assert!(!cleaned.contains(residue), "{cleaned:?}");
        }
        assert!(!cleaned.contains("[0m"), "{cleaned:?}");
        assert!(!cleaned.contains("[1m"), "{cleaned:?}");
    }

    #[tokio::test]
    async fn coloured_git_status_has_no_escape_residue() {
        let coloured = "On branch \u{1b}[1;36moutput-cleaning\u{1b}[m\nChanges not staged for commit:\n\t\u{1b}[31mmodified:   sandbox/src/output.rs\u{1b}[m\n";
        let cleaned = compress(coloured).await.expect("escapes are stripped");
        assert_eq!(
            cleaned,
            "On branch output-cleaning\nChanges not staged for commit:\n\tmodified:   sandbox/src/output.rs\n"
        );
    }

    #[tokio::test]
    async fn progress_line_keeps_only_its_last_frame() {
        let progress =
            "\u{1b}[2K\rFetching 10/90\u{1b}[2K\rFetching 50/90\u{1b}[2K\rFetching 90/90\ndone\n";
        let cleaned = compress(progress).await.expect("frames are resolved");
        assert_eq!(cleaned, "Fetching 90/90\ndone\n");
    }

    #[tokio::test]
    async fn blank_runs_collapse_but_single_blank_lines_survive() {
        let cleaned = compress("alpha\n\n\n\nbeta\n\ngamma\n")
            .await
            .expect("blank runs collapse");
        assert_eq!(cleaned, "alpha\n\nbeta\n\ngamma\n");
    }

    #[tokio::test]
    async fn json_table_becomes_tsv_naming_a_readable_original() {
        let json = r#"[
            {"id": 1, "name": "Alice", "email": "alice@aither.test", "role": "admin"},
            {"id": 2, "name": "Bob", "email": "bob@aither.test", "role": "user"},
            {"id": 3, "name": "Charlie", "email": "charlie@aither.test", "role": "user"},
            {"id": 4, "name": "Diana", "email": "diana@aither.test", "role": "admin"}
        ]"#;
        let dir = tempfile::tempdir().expect("tempdir");
        let cleaned = compress_text(dir.path(), json)
            .await
            .expect("compression succeeds")
            .expect("json becomes tsv");

        assert!(cleaned.contains('\t'), "{cleaned}");
        assert!(cleaned.contains("Original JSON saved to"), "{cleaned}");

        // The text names a file that holds JSON parsing back to the input.
        let named = cleaned
            .rsplit_once(" (")
            .and_then(|(_, tail)| tail.strip_suffix(").]"))
            .expect("absolute path in the reference");
        let stored = std::fs::read_to_string(named).expect("named original is readable");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&stored).expect("stored original is JSON"),
            serde_json::from_str::<serde_json::Value>(json).expect("input is JSON"),
            "stored original must round-trip"
        );
    }

    #[tokio::test]
    async fn plain_text_that_needs_no_cleaning_is_left_alone() {
        assert!(compress("all good\n").await.is_none());
    }

    #[tokio::test]
    async fn output_cut_off_mid_escape_leaks_no_brackets() {
        // A killed process can stop in the middle of a sequence; the last
        // frame still has to be readable and free of bracket noise, and the
        // newline the process never wrote is not invented.
        let cleaned = compress("partial output\u{1b}[32")
            .await
            .expect("truncated sequence is consumed");
        assert_eq!(cleaned, "partial output");
    }

    #[tokio::test]
    async fn output_without_a_final_newline_is_left_alone() {
        // `printf abc` is already what a terminal shows, so the pipeline has
        // nothing to change and must not append a terminator.
        assert!(compress("abc").await.is_none());
    }

    #[tokio::test]
    async fn unterminated_progress_line_renders_as_its_last_frame() {
        let cleaned = compress("abc\r\u{1b}[2Kxyz")
            .await
            .expect("the frame is resolved");
        assert_eq!(cleaned, "xyz");
    }

    #[tokio::test]
    async fn unified_diff_still_applies_after_cleaning() {
        // A blank line inside the hunk reaches the diff as a single space, so
        // the patch exercises blank-run collapsing as well as colour.
        let original =
            "fn main() {\n    let a = 1;\n\n\n    let b = 2;\n    println!(\"{a}{b}\");\n}\n";
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path();
        run_git(repo, &["init", "--quiet"]);
        std::fs::write(repo.join("main.rs"), original).expect("write original");
        run_git(repo, &["add", "main.rs"]);
        run_git(repo, &["commit", "--quiet", "-m", "seed"]);

        std::fs::write(
            repo.join("main.rs"),
            original.replace("let b = 2;", "let b = 20;"),
        )
        .expect("write patched");
        let diff = git_output(repo, &["diff", "--color=always", "--", "main.rs"]);
        assert!(diff.contains('\u{1b}'), "git diff should be coloured");

        let cleaned = compress_text(repo, &diff)
            .await
            .expect("compression succeeds")
            .expect("coloured diff is cleaned");
        assert!(!cleaned.contains('\u{1b}'), "{cleaned}");

        // Restore the original, then check the cleaned patch still applies.
        run_git(repo, &["checkout", "--", "main.rs"]);
        std::fs::write(repo.join("cleaned.patch"), &cleaned).expect("write patch");
        let check = std::process::Command::new("git")
            .current_dir(repo)
            .args(["apply", "--check", "cleaned.patch"])
            .output()
            .expect("run git apply");
        assert!(
            check.status.success(),
            "git apply --check failed: {}\npatch:\n{cleaned}",
            String::from_utf8_lossy(&check.stderr)
        );
    }

    /// Runs `git` in `repo` with identity and signing pinned, so the test does
    /// not depend on the developer's global configuration.
    fn git(repo: &Path, args: &[&str]) -> std::process::Output {
        std::process::Command::new("git")
            .current_dir(repo)
            .args([
                "-c",
                "user.name=aither test",
                "-c",
                "user.email=test@aither.invalid",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .expect("run git")
    }

    fn run_git(repo: &Path, args: &[&str]) {
        let output = git(repo, args);
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn git_output(repo: &Path, args: &[&str]) -> String {
        let output = git(repo, args);
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("git output is utf-8")
    }
}
