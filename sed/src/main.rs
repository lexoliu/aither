//! The `aither-sed` executable.

use std::io::{self, BufWriter};
use std::process::ExitCode;

fn main() -> ExitCode {
    let status = aither_sed::run(
        std::env::args_os().skip(1).collect(),
        io::stdin().lock(),
        BufWriter::new(io::stdout().lock()),
        io::stderr().lock(),
    );
    ExitCode::from(status)
}
