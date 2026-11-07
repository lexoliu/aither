use std::ffi::OsString;
use std::path::Path;

use super::{EXIT_OK, EXIT_UNREADABLE_INPUT, EXIT_UNSUPPORTED, VERSION, run, supports};

/// Files `a` (five lines), `b` (three), `nonl` (two, the last without a
/// newline) and `empty`, in a directory of their own.
fn fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("tempdir");
    for (name, content) in [
        ("a", "a1\na2\na3\na4\na5\n"),
        ("b", "b1\nb2\nb3\n"),
        ("nonl", "n1\nn2"),
        ("empty", ""),
    ] {
        std::fs::write(directory.path().join(name), content).expect("write fixture");
    }
    directory
}

/// Runs `aither-sed` with `args`, where an argument naming a fixture file
/// stands for its path, and returns the exit status, stdout and stderr.
fn sed(directory: &Path, args: &[&str], stdin: &str) -> (u8, Vec<u8>, String) {
    let args = args
        .iter()
        .map(|argument| {
            if ["a", "b", "nonl", "empty", "missing"].contains(argument) {
                directory.join(argument).into_os_string()
            } else {
                OsString::from(argument)
            }
        })
        .collect();
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let status = run(args, stdin.as_bytes(), &mut stdout, &mut stderr);
    (status, stdout, String::from_utf8(stderr).expect("stderr"))
}

fn printed(args: &[&str], stdin: &str) -> String {
    let directory = fixture();
    let (status, stdout, stderr) = sed(directory.path(), args, stdin);
    assert_eq!(status, EXIT_OK, "{args:?}: {stderr}");
    assert_eq!(stderr, "", "{args:?}");
    String::from_utf8(stdout).expect("stdout")
}

#[test]
fn prints_addressed_lines() {
    for (args, expected) in [
        (&["-n", "2p", "a"][..], "a2\n"),
        (&["-n", "2,4p", "a"], "a2\na3\na4\n"),
        (&["-n", "4,$p", "a"], "a4\na5\n"),
        (&["-n", "$p", "a"], "a5\n"),
        (&["-n", "1,$p", "b"], "b1\nb2\nb3\n"),
        (&["-n", "9p", "a"], ""),
        (&["-n", "4,9p", "a"], "a4\na5\n"),
        (&["-n", "08p", "a"], ""),
        (&["-n", "02p", "a"], "a2\n"),
    ] {
        assert_eq!(printed(args, ""), expected, "{args:?}");
    }
}

#[test]
fn a_range_ending_before_its_start_prints_only_the_start() {
    assert_eq!(printed(&["-n", "3,2p", "a"], ""), "a3\n");
    assert_eq!(printed(&["-n", "3,3p", "a"], ""), "a3\n");
    assert_eq!(printed(&["-n", "3,0p", "a"], ""), "a3\n");
}

#[test]
fn commands_join_with_semicolons_and_newlines() {
    for (args, expected) in [
        (&["-n", "1p;3,4p;$p", "a"][..], "a1\na3\na4\na5\n"),
        (&["-n", "1p\n3p", "a"], "a1\na3\n"),
        (&["-n", " 1 p ; 3 , 4p ", "a"], "a1\na3\na4\n"),
        (&["-n", "3,\t$ p", "a"], "a3\na4\na5\n"),
        (&["-n", ";1p;;2p;", "a"], "a1\na2\n"),
        (&["-n", "1p;1p;", "a"], "a1\na1\n"),
    ] {
        assert_eq!(printed(args, ""), expected, "{args:?}");
    }
}

#[test]
fn overlapping_commands_print_a_line_once_each_in_input_order() {
    assert_eq!(
        printed(&["-n", "2,4p;3,5p", "a"], ""),
        "a2\na3\na3\na4\na4\na5\n"
    );
    assert_eq!(printed(&["-n", "4p;2p", "a"], ""), "a2\na4\n");
    assert_eq!(printed(&["-n", "$p;5p;4,$p", "a"], ""), "a4\na5\na5\na5\n");
}

#[test]
fn script_comes_from_the_first_operand_or_one_dash_e() {
    for args in [
        &["-n", "-e", "2p", "a"][..],
        &["-n", "-e2p", "a"],
        &["-ne", "2p", "a"],
        &["-ne2p", "a"],
        &["--quiet", "2p", "a"],
        &["--silent", "2p", "a"],
        &["-n", "--", "2p", "a"],
        &["2p", "a", "-n"],
        &["-n", "-n", "2p", "a"],
        &["-n", "-e", "2p", "--", "a"],
    ] {
        assert_eq!(printed(args, ""), "a2\n", "{args:?}");
    }
    // With `-e`, the first operand is a file, not the script.
    assert_eq!(printed(&["-n", "-e", "1p", "a", "b"], ""), "a1\n");
}

#[test]
fn line_numbers_continue_across_files() {
    for (args, expected) in [
        (&["-n", "3p", "a", "b"][..], "a3\n"),
        (&["-n", "4,7p", "a", "b"], "a4\na5\nb1\nb2\n"),
        (&["-n", "5,$p", "a", "b"], "a5\nb1\nb2\nb3\n"),
        (&["-n", "6p;8p", "a", "b"], "b1\nb3\n"),
        (&["-n", "$p", "a", "b"], "b3\n"),
        (&["-n", "$p", "a", "empty"], "a5\n"),
        (&["-n", "$p", "empty", "a"], "a5\n"),
        (&["-n", "$p", "empty"], ""),
        (&["-n", "1p", "empty", "b"], "b1\n"),
    ] {
        assert_eq!(printed(args, ""), expected, "{args:?}");
    }
}

#[test]
fn standard_input_is_the_default_and_dash() {
    let stdin = "x\ny\nz\n";
    assert_eq!(printed(&["-n", "2,3p"], stdin), "y\nz\n");
    assert_eq!(printed(&["-n", "$p", "-"], stdin), "z\n");
    assert_eq!(printed(&["-n", "2,3p", "a", "-", "b"], stdin), "a2\na3\n");
    assert_eq!(printed(&["-n", "5,7p", "a", "-", "b"], stdin), "a5\nx\ny\n");
    assert_eq!(printed(&["-n", "1p", "--"], stdin), "x\n");
    // Standard input is read once; a second `-` is at its end.
    assert_eq!(printed(&["-n", "1,$p", "-", "-"], stdin), "x\ny\nz\n");
}

#[test]
fn a_line_without_its_newline_gets_one_only_when_output_follows() {
    for (args, expected) in [
        (&["-n", "1,2p", "nonl"][..], "n1\nn2"),
        (&["-n", "2p", "nonl"], "n2"),
        (&["-n", "1p", "nonl"], "n1\n"),
        (&["-n", "$p", "nonl"], "n2"),
        (&["-n", "2p;2p", "nonl"], "n2\nn2"),
        (&["-n", "2p;1p", "nonl"], "n1\nn2"),
        (&["-n", "1,2p", "nonl", "a"], "n1\nn2"),
        (&["-n", "2p", "nonl", "b"], "n2"),
        (&["-n", "$p", "nonl", "a"], "a5\n"),
        (&["-n", "2,3p", "nonl", "a"], "n2\na1\n"),
        (&["-n", "2,3p", "a", "nonl"], "a2\na3\n"),
        (&["-n", "4,$p", "a", "nonl"], "a4\na5\nn1\nn2"),
        (&["-n", "$p", "nonl", "empty"], "n2"),
        (&["-n", "$p", "empty", "nonl"], "n2"),
        (&["-n", "1,$p", "nonl", "nonl"], "n1\nn2\nn1\nn2"),
    ] {
        assert_eq!(printed(args, ""), expected, "{args:?}");
    }
    assert_eq!(printed(&["-n", "1,$p"], "x\ny"), "x\ny");
    assert_eq!(printed(&["-n", "$p"], "x\ny"), "y");
}

#[test]
fn bytes_that_are_not_text_pass_through() {
    let directory = tempfile::tempdir().expect("tempdir");
    let file = directory.path().join("binary");
    std::fs::write(&file, b"\xff\xfe\nok\n\x00\n").expect("write");
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let args = vec![
        OsString::from("-n"),
        OsString::from("1p;3p"),
        file.into_os_string(),
    ];
    assert_eq!(run(args, &b""[..], &mut stdout, &mut stderr), EXIT_OK);
    assert_eq!(stdout, b"\xff\xfe\n\x00\n");
}

#[test]
fn an_unreadable_file_is_reported_and_the_rest_still_print() {
    let directory = fixture();
    let (status, stdout, stderr) = sed(directory.path(), &["-n", "1,4p", "a", "missing", "b"], "");
    assert_eq!(status, EXIT_UNREADABLE_INPUT);
    assert_eq!(String::from_utf8_lossy(&stdout), "a1\na2\na3\na4\n");
    assert!(
        stderr.starts_with("sed: can't read ")
            && stderr.ends_with("missing: No such file or directory\n"),
        "{stderr}"
    );

    let (status, stdout, stderr) = sed(directory.path(), &["-n", "$p", "a", "missing"], "");
    assert_eq!(status, EXIT_UNREADABLE_INPUT);
    assert_eq!(String::from_utf8_lossy(&stdout), "a5\n");
    assert!(
        stderr.contains("missing: No such file or directory"),
        "{stderr}"
    );

    // Past the last needed line, the inputs after it are still checked.
    let (status, stdout, stderr) = sed(directory.path(), &["-n", "1p", "a", "missing"], "");
    assert_eq!(status, EXIT_UNREADABLE_INPUT);
    assert_eq!(String::from_utf8_lossy(&stdout), "a1\n");
    assert!(
        stderr.contains("missing: No such file or directory"),
        "{stderr}"
    );
}

#[test]
fn a_directory_is_a_read_error() {
    let directory = fixture();
    let dir = directory.path().to_string_lossy().into_owned();
    let (status, stdout, stderr) = sed(directory.path(), &["-n", "1p", &dir, "a"], "");
    assert_eq!(status, EXIT_UNREADABLE_INPUT);
    assert_eq!(String::from_utf8_lossy(&stdout), "a1\n");
    assert!(stderr.contains("read error on"), "{stderr}");
    assert!(stderr.contains("Is a directory"), "{stderr}");
}

#[test]
fn reading_stops_past_the_largest_needed_line() {
    /// Reads one line per call and fails when asked for more than `lines`.
    struct Endless {
        lines: usize,
        sent: usize,
    }
    impl std::io::Read for Endless {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            assert!(self.sent < self.lines, "read past line {}", self.lines);
            self.sent += 1;
            buffer[..2].copy_from_slice(b"x\n");
            Ok(2)
        }
    }
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let args = vec![OsString::from("-n"), OsString::from("2,3p")];
    // One read returns one line here, so a read past line 3 would panic.
    let status = run(
        args,
        Endless { lines: 3, sent: 0 },
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, EXIT_OK);
    assert_eq!(stdout, b"x\nx\n");
}

#[test]
fn everything_else_fails_with_the_message() {
    let directory = fixture();
    for args in [
        &["-i", "2p", "a"][..],
        &["-n", "-i", "2p", "a"],
        &["-n", "-s", "2p", "a"],
        &["-n", "-E", "2p", "a"],
        &["-n", "--expression=2p", "a"],
        &["-n", "--in-place", "2p", "a"],
        &["-n", "--help"],
        &["s/a/b/", "a"],
        &["-n", "s/a/b/p", "a"],
        &["-n", "/a2/p", "a"],
        &["-n", "/a2/,/a4/p", "a"],
        &["-n", "1d", "a"],
        &["-n", "2,3d", "a"],
        &["-n", "1!p", "a"],
        &["-n", "2~2p", "a"],
        &["-n", "0p", "a"],
        &["-n", "0,3p", "a"],
        &["-n", "1p;s/a/b/", "a"],
        &["-n", "1,2,3p", "a"],
        &["-n", "$,3p", "a"],
        &["-n", "$,$p", "a"],
        &["-n", "1pp", "a"],
        &["-n", "1p p", "a"],
        &["-n", "1 2p", "a"],
        &["-n", "1p # note", "a"],
        &["-n", "{1p}", "a"],
        &["-n", "p", "a"],
        &["-n", "1", "a"],
        &["-n", "", "a"],
        &["-n", ";", "a"],
        &["-n", "a"],
        &["-n"],
        &[],
        &["-n", "-e", "1p", "-e", "2p", "a"],
        &["-n", "-e"],
        &["2p", "a"],
        &["--version", "a"],
    ] {
        let (status, stdout, stderr) = sed(directory.path(), args, "");
        assert_eq!(status, EXIT_UNSUPPORTED, "{args:?}: {stderr}");
        assert!(stdout.is_empty(), "{args:?}");
        assert!(
            supports(args.iter().map(OsString::from).collect()).is_err(),
            "{args:?}"
        );
        assert_eq!(
            stderr,
            "`sed` in scripts only prints line ranges (`sed -n 'N,Mp' file`). \
             Use `apply_patch` to edit files or `python3` to transform text.\n",
            "{args:?}"
        );
    }
    // Nothing was edited.
    let (_, kept, _) = sed(directory.path(), &["-n", "1,$p", "a"], "");
    assert_eq!(String::from_utf8_lossy(&kept), "a1\na2\na3\na4\na5\n");
}

#[test]
fn supports_accepts_the_line_range_forms() {
    for args in [
        &["-n", "2,3p", "a"][..],
        &["-n", "$p"],
        &["--quiet", "-e", "1p;4,$p", "a", "b"],
        &["--version"],
    ] {
        assert!(
            supports(args.iter().map(OsString::from).collect()).is_ok(),
            "{args:?}"
        );
    }
}

#[test]
fn version_names_the_tool_and_its_version() {
    let (status, stdout, _) = sed(fixture().path(), &["--version"], "");
    assert_eq!(status, EXIT_OK);
    assert_eq!(String::from_utf8_lossy(&stdout), format!("{VERSION}\n"));
    assert_eq!(VERSION, format!("aither-sed {}", env!("CARGO_PKG_VERSION")));
}
