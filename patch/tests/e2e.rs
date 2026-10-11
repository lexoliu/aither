//! End-to-end tests for the `apply_patch` executable: real process invocations
//! with the patch on stdin, exactly as a shell heredoc delivers it.

use std::{
    fs,
    io::Write as _,
    path::Path,
    process::{Command, Output, Stdio},
};

use tempfile::tempdir;

/// Runs `apply_patch` inside `cwd`, feeding `stdin` on its standard input.
fn run_apply_patch(cwd: &Path, stdin: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_apply_patch"))
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn apply_patch");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(stdin.as_bytes())
        .expect("write patch");
    child.wait_with_output().expect("wait for apply_patch")
}

/// Runs `apply_patch` with arguments and no stdin content.
fn run_with_args(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_apply_patch"))
        .current_dir(cwd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("run apply_patch")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn version_is_deterministic_and_identifies_the_command() {
    let dir = tempdir().expect("tempdir");
    let output = run_with_args(dir.path(), &["--version"]);
    assert!(output.status.success());
    assert_eq!(stdout(&output), "apply_patch 0.6.0\n");
    let short = run_with_args(dir.path(), &["-V"]);
    assert_eq!(stdout(&short), stdout(&output));
}

#[test]
fn help_describes_the_patch_format() {
    let dir = tempdir().expect("tempdir");
    let output = run_with_args(dir.path(), &["--help"]);
    assert!(output.status.success());
    let help = stdout(&output);
    assert!(help.contains("*** Begin Patch"));
    assert!(help.contains("<<'PATCH'"));
    assert!(help.contains("Move to:"));
}

#[test]
fn unexpected_arguments_exit_2_with_usage() {
    let dir = tempdir().expect("tempdir");
    let output = run_with_args(dir.path(), &["file.txt"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("usage"));
}

#[test]
fn full_heredoc_roundtrip_add_update_delete_move() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("keep.rs"), "fn old() {}\nfn rest() {}\n").expect("write");
    fs::write(dir.path().join("trash.txt"), "junk\n").expect("write");

    let patch = "*** Begin Patch\n\
        *** Update File: keep.rs\n@@\n-fn old() {}\n+fn new() {}\n\
        *** Update File: keep.rs\n*** Move to: renamed.rs\n\
        *** Add File: fresh/created.txt\n+made\n\
        *** Delete File: trash.txt\n\
        *** End Patch";
    let output = run_apply_patch(dir.path(), patch);
    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert_eq!(
        stdout(&output),
        "M keep.rs\nR keep.rs -> renamed.rs\nA fresh/created.txt\nD trash.txt\n"
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("renamed.rs")).expect("read"),
        "fn new() {}\nfn rest() {}\n"
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("fresh/created.txt")).expect("read"),
        "made\n"
    );
    assert!(!dir.path().join("trash.txt").exists());
}

#[test]
fn malformed_patch_exits_1_with_stderr_and_touches_nothing() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("a.txt"), "untouched\n").expect("write");
    let output = run_apply_patch(dir.path(), "*** Begin Patch\n*** Update File:");
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("apply_patch:"));
    assert_eq!(
        fs::read_to_string(dir.path().join("a.txt")).expect("read"),
        "untouched\n"
    );
}

#[test]
fn mismatched_context_exits_1_and_reports_the_line() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("a.txt"), "real\n").expect("write");
    let patch = "*** Begin Patch\n*** Update File: a.txt\n@@\n phantom\n+new\n*** End Patch";
    let output = run_apply_patch(dir.path(), patch);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("phantom"),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn ambiguous_patch_is_diagnosed_not_applied() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("d.txt"), "same\nsame\nsame\n").expect("write");
    let patch = "*** Begin Patch\n*** Update File: d.txt\n@@\n-same\n+unique\n*** End Patch";
    let output = run_apply_patch(dir.path(), patch);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).to_lowercase().contains("ambiguous"),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn multi_file_failure_leaves_all_files_untouched() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("good.txt"), "before\n").expect("write");
    fs::write(dir.path().join("other.txt"), "stays\n").expect("write");
    let patch = "*** Begin Patch\n\
        *** Update File: good.txt\n@@\n-before\n+after\n\
        *** Update File: other.txt\n@@\n-ghost\n+never\n\
        *** End Patch";
    let output = run_apply_patch(dir.path(), patch);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        fs::read_to_string(dir.path().join("good.txt")).expect("read"),
        "before\n"
    );
}

#[test]
fn empty_stdin_exits_1() {
    let dir = tempdir().expect("tempdir");
    let output = run_apply_patch(dir.path(), "");
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("empty"));
}

#[test]
fn missing_file_error_names_the_path() {
    let dir = tempdir().expect("tempdir");
    let patch = "*** Begin Patch\n*** Delete File: nowhere.txt\n*** End Patch";
    let output = run_apply_patch(dir.path(), patch);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("nowhere.txt"));
}

#[test]
fn crlf_and_final_newline_survive_a_patch() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("w.txt"), "a\r\nb").expect("write");
    let patch = "*** Begin Patch\n*** Update File: w.txt\n@@\n-a\n+A\n*** End Patch";
    let output = run_apply_patch(dir.path(), patch);
    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert_eq!(
        fs::read_to_string(dir.path().join("w.txt")).expect("read"),
        "A\r\nb"
    );
}
