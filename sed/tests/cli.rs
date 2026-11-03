//! The executable as the sandbox runs it.

use std::io::Write;
use std::process::{Command, Stdio};

fn aither_sed(args: &[&str], stdin: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_aither-sed"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn aither-sed");
    // A refused script exits without reading its input, which then meets a
    // closed pipe.
    match child
        .stdin
        .take()
        .expect("stdin")
        .write_all(stdin.as_bytes())
    {
        Err(error) if error.kind() != std::io::ErrorKind::BrokenPipe => {
            panic!("write stdin: {error}")
        }
        _ => {}
    }
    child.wait_with_output().expect("wait")
}

#[test]
fn version_is_the_name_and_the_package_version() {
    let output = aither_sed(&["--version"], "");
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        concat!("aither-sed ", env!("CARGO_PKG_VERSION"), "\n")
    );
}

#[test]
fn prints_a_range_from_standard_input() {
    let output = aither_sed(&["-n", "2,3p"], "x\ny\nz\nw\n");
    assert!(output.status.success());
    assert_eq!(output.stdout, b"y\nz\n");
}

/// Every supported form prints what the host's GNU `sed` prints, byte for
/// byte, with the same exit status.
#[cfg(target_os = "linux")]
#[test]
fn every_supported_form_matches_gnu_sed() {
    let directory = tempfile::tempdir().expect("tempdir");
    for (name, content) in [
        ("a", "a1\na2\na3\na4\na5\n"),
        ("b", "b1\nb2\nb3\n"),
        ("nonl", "n1\nn2"),
        ("empty", ""),
    ] {
        std::fs::write(directory.path().join(name), content).expect("write fixture");
    }
    let cases: &[&[&str]] = &[
        &["-n", "2p", "a"],
        &["-n", "2,4p", "a"],
        &["-n", "4,$p", "a"],
        &["-n", "$p", "a"],
        &["-n", "4,2p", "a"],
        &["-n", "9p", "a"],
        &["-n", "2,3p;1p", "a"],
        &["-n", "1,4p;3,5p", "a"],
        &["-n", " 2 , 3 p ; 5p ", "a"],
        &["-n", "2p\n4p", "a"],
        &["-ne", "3p", "a"],
        &["-n", "-e", "3p", "a"],
        &["--quiet", "3p", "a"],
        &["-n", "4,6p", "a", "b"],
        &["-n", "$p", "a", "b"],
        &["-n", "2p", "nonl"],
        &["-n", "$p", "nonl"],
        &["-n", "1,$p", "nonl", "b"],
        &["-n", "2p", "nonl", "b"],
        &["-n", "2,3p", "nonl", "b"],
        &["-n", "1,$p", "b", "nonl"],
        &["-n", "$p", "nonl", "empty"],
        &["-n", "1,$p", "empty", "a"],
        &["-n", "2p", "a", "missing", "b"],
        &["-n", "7p", "a", "missing", "b"],
        &["-n", "2,3p", "-"],
        &["-n", "2,3p"],
    ];
    let mut mismatches = Vec::new();
    for args in cases {
        let run = |program: &str| {
            let mut child = Command::new(program)
                .args(*args)
                .current_dir(directory.path())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn");
            let mut stdin = child.stdin.take().expect("stdin");
            // Only the forms that read standard input get it; the others may
            // exit before it is written.
            if args.len() == 2 || args.last() == Some(&"-") {
                stdin.write_all(b"s1\ns2\ns3\ns4").expect("write stdin");
            }
            drop(stdin);
            child.wait_with_output().expect("wait")
        };
        let ours = run(env!("CARGO_BIN_EXE_aither-sed"));
        let gnu = run("sed");
        if (ours.status.code(), &ours.stdout) != (gnu.status.code(), &gnu.stdout) {
            mismatches.push(format!(
                "{args:?}: ours {:?} {:?}, GNU {:?} {:?}",
                ours.status.code(),
                String::from_utf8_lossy(&ours.stdout),
                gnu.status.code(),
                String::from_utf8_lossy(&gnu.stdout),
            ));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
fn an_unsupported_script_exits_non_zero_with_the_message() {
    let output = aither_sed(&["s/a/b/"], "a\n");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stdout, b"");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("`apply_patch`") && stderr.contains("`python3`"),
        "{stderr}"
    );
}
