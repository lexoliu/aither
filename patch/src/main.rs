//! The `apply_patch` executable: reads a compact contextual patch on standard
//! input and applies it to the filesystem relative to the current directory.
//!
//! It is an ordinary sandbox-executed binary — all parsing, matching, and
//! writes happen in this process under the shell's sandbox policy.

use std::io::{self, Read};
use std::process::ExitCode;

/// Printed by `--version`; consumed by the tool-runtime initializer to verify
/// artifact identity, so it stays a bare `apply_patch <semver>` line.
const VERSION: &str = concat!("apply_patch ", env!("CARGO_PKG_VERSION"));

const HELP: &str = "\
apply_patch — apply a compact contextual patch to files

Usage: apply_patch < PATCH

Reads the patch from standard input and applies it relative to the current
directory. Typical use is a quoted heredoc:

  apply_patch <<'PATCH'
  *** Begin Patch
  *** Update File: src/main.rs
  @@ fn main
   let args = parse();
  -run(args);
  +run(&args);
  *** Add File: notes/todo.txt
  +first task
  *** Update File: old/name.rs
  *** Move to: new/name.rs
  *** Delete File: stale.txt
  *** End Patch
  PATCH

Format: one or more `Add File`, `Update File`, or `Delete File` sections
between `*** Begin Patch` and `*** End Patch`. Inside an Update section,
lines starting with a space are context, `-` removes, `+` inserts, `@@`
anchors a hunk, `*** Move to:` renames, and `*** End of File` anchors a
hunk at the tail of the file.

Every hunk is validated before any file is written; a patch that fails to
parse or match changes nothing. Matched context must be unambiguous.

Options:
  -h, --help     show this help and exit
  -V, --version  print the version and exit

Output: one summary line per file (`A` added, `M` modified, `D` deleted,
`R` renamed). Exit status: 0 success, 1 patch or filesystem failure,
2 bad arguments.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => run(),
        [flag] if flag == "-h" || flag == "--help" => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        [flag] if flag == "-V" || flag == "--version" => {
            println!("{VERSION}");
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "apply_patch: unexpected arguments (takes no arguments; patch on stdin)\nusage: apply_patch < PATCH  (see --help)"
            );
            ExitCode::from(2)
        }
    }
}

/// Reads the patch from stdin and applies it under the current directory.
fn run() -> ExitCode {
    let mut input = Vec::new();
    let read_result = io::stdin().lock().read_to_end(&mut input);
    if let Err(error) = read_result {
        eprintln!("apply_patch: failed to read standard input: {error}");
        return ExitCode::FAILURE;
    }
    let Ok(patch) = String::from_utf8(input) else {
        eprintln!("apply_patch: patch input is not valid UTF-8");
        return ExitCode::FAILURE;
    };
    let base = match std::env::current_dir() {
        Ok(base) => base,
        Err(error) => {
            eprintln!("apply_patch: failed to resolve current directory: {error}");
            return ExitCode::FAILURE;
        }
    };
    match aither_patch::apply_patch_text(&patch, &base) {
        Ok(outcomes) => {
            for outcome in &outcomes {
                println!("{}", outcome.summary());
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("apply_patch: {error}");
            ExitCode::FAILURE
        }
    }
}
