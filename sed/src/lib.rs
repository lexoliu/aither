//! `aither-sed`: the `sed` of sandboxed agent scripts, which prints line
//! ranges and nothing else.
//!
//! It implements exactly the GNU `sed -n` forms whose script is made of `p`
//! commands addressed by `N`, `N,M`, `N,$` or `$`, joined by `;`, over files,
//! `-` or standard input. Any other option or script is refused with a message
//! naming the tools to use instead.

mod input;
mod script;

use std::ffi::OsString;
use std::io::{self, Read, Write};

use input::{Input, Line};
use script::Script;

/// Printed by `--version`; the tool-runtime initializer reads the version
/// from its second word.
pub const VERSION: &str = concat!("aither-sed ", env!("CARGO_PKG_VERSION"));

const UNSUPPORTED: &str = include_str!("../templates/unsupported.txt");

/// Exit status when every input was read.
pub const EXIT_OK: u8 = 0;
/// Exit status of an option or script this `sed` does not implement.
pub const EXIT_UNSUPPORTED: u8 = 1;
/// Exit status when an input could not be read; the others are still printed.
pub const EXIT_UNREADABLE_INPUT: u8 = 2;
/// Exit status when the output cannot be written.
pub const EXIT_WRITE: u8 = 4;

/// An option or script outside the implemented `sed -n 'N,Mp'` forms.
#[derive(Debug, thiserror::Error)]
#[error("{}", UNSUPPORTED)]
pub struct Unsupported;

/// What the command line asks for.
#[derive(Debug)]
enum Request {
    Version,
    Print {
        script: Script,
        files: Vec<OsString>,
    },
}

/// Whether `aither-sed` implements `args` (the program name excluded), the
/// same check [`run`] makes before it reads anything: a host's `sed` given
/// arguments this accepts prints the same lines.
///
/// # Errors
/// Returns [`Unsupported`] for an option or script outside the implemented
/// forms.
pub fn supports(args: Vec<OsString>) -> Result<(), Unsupported> {
    parse_arguments(args).map(drop)
}

/// Runs `aither-sed` with `args` (the program name excluded) and returns its
/// exit status. Lines go to `stdout`, diagnostics to `stderr`.
pub fn run(
    args: Vec<OsString>,
    stdin: impl Read,
    mut stdout: impl Write,
    mut stderr: impl Write,
) -> u8 {
    let (script, files) = match parse_arguments(args) {
        Ok(Request::Version) => return write_line(&mut stdout, VERSION),
        Ok(Request::Print { script, files }) => (script, files),
        Err(error) => {
            let _ = writeln!(stderr, "{error}");
            return EXIT_UNSUPPORTED;
        }
    };
    match print_ranges(&script, files, stdin, &mut stdout, &mut stderr) {
        Ok(status) => status,
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => EXIT_WRITE,
        Err(error) => {
            let _ = writeln!(
                stderr,
                "sed: couldn't write to stdout: {}",
                input::describe(&error)
            );
            EXIT_WRITE
        }
    }
}

fn write_line(out: &mut impl Write, text: &str) -> u8 {
    match writeln!(out, "{text}") {
        Ok(()) => EXIT_OK,
        Err(_) => EXIT_WRITE,
    }
}

/// Reads GNU `sed`'s command line, accepting only `-n`/`--quiet`/`--silent`,
/// one script (first operand or one `-e`) and the files after it.
fn parse_arguments(args: Vec<OsString>) -> Result<Request, Unsupported> {
    if let [only] = args.as_slice()
        && only == "--version"
    {
        return Ok(Request::Version);
    }
    let mut quiet = false;
    let mut script: Option<String> = None;
    let mut operands = Vec::new();
    let mut arguments = args.into_iter();
    while let Some(argument) = arguments.next() {
        let Some(text) = argument.to_str().filter(|text| text.starts_with('-')) else {
            operands.push(argument);
            continue;
        };
        match text {
            "-" => operands.push(argument),
            "--" => {
                operands.extend(arguments);
                break;
            }
            "--quiet" | "--silent" => quiet = true,
            _ if text.starts_with("--") => return Err(Unsupported),
            _ => {
                let mut flags = text[1..].chars();
                while let Some(flag) = flags.next() {
                    match flag {
                        'n' => quiet = true,
                        'e' => {
                            let attached = flags.as_str();
                            let value = if attached.is_empty() {
                                arguments
                                    .next()
                                    .and_then(|value| value.into_string().ok())
                                    .ok_or(Unsupported)?
                            } else {
                                attached.to_owned()
                            };
                            if script.replace(value).is_some() {
                                return Err(Unsupported);
                            }
                            break;
                        }
                        _ => return Err(Unsupported),
                    }
                }
            }
        }
    }
    let mut operands = operands.into_iter();
    let script = match script {
        Some(script) => script,
        None => operands
            .next()
            .and_then(|operand| operand.into_string().ok())
            .ok_or(Unsupported)?,
    };
    if !quiet {
        return Err(Unsupported);
    }
    Ok(Request::Print {
        script: Script::parse(&script)?,
        files: operands.collect(),
    })
}

/// Prints, for each input line in order, once per command that addresses it.
fn print_ranges(
    script: &Script,
    files: Vec<OsString>,
    stdin: impl Read,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> io::Result<u8> {
    let files = if files.is_empty() {
        vec![OsString::from("-")]
    } else {
        files
    };
    let mut input = Input::new(files, stdin, stderr);
    let mut printer = Printer {
        out: stdout,
        owes_newline: false,
    };
    let needs_last = script.needs_last_line();
    let limit = script.last_needed_line();
    let mut number = 0_u64;
    let mut current = input.next_line()?;
    while let Some(line) = current {
        number += 1;
        // `$` is the last line of the last input, so it needs one line of
        // lookahead; other scripts read no further than they print.
        let following = if needs_last { input.next_line()? } else { None };
        let copies = script.copies(number, needs_last && following.is_none());
        for _ in 0..copies {
            printer.print(&line)?;
        }
        if needs_last {
            current = following;
        } else if limit.is_some_and(|limit| number >= limit) {
            input.check_remaining();
            break;
        } else {
            current = input.next_line()?;
        }
    }
    printer.out.flush()?;
    Ok(input.status())
}

/// Writes lines, restoring the newline a line read without one lacks when
/// more output follows it.
struct Printer<'a, W: Write> {
    out: &'a mut W,
    owes_newline: bool,
}

impl<W: Write> Printer<'_, W> {
    fn print(&mut self, line: &Line) -> io::Result<()> {
        if self.owes_newline {
            self.out.write_all(b"\n")?;
        }
        self.out.write_all(&line.text)?;
        self.owes_newline = line.unterminated;
        if !line.unterminated {
            self.out.write_all(b"\n")?;
        }
        Ok(())
    }
}

// aither-sed ships only on Unix, where its diagnostics carry the system's
// wording GNU sed prints.
#[cfg(all(test, unix))]
mod tests;
