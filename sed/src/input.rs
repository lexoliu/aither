//! The inputs in order, read as lines through buffered readers.

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;

use crate::{EXIT_OK, EXIT_UNREADABLE_INPUT};

/// One input line without its newline.
#[derive(Debug)]
pub struct Line {
    pub text: Vec<u8>,
    /// Whether the line ended its input without a newline. GNU `sed` prints
    /// it without one too, and writes the newline only when more output
    /// follows.
    pub unterminated: bool,
}

enum Source<R> {
    Stdin(BufReader<R>),
    File(BufReader<File>),
}

impl<R: Read> Source<R> {
    fn read_until_newline(&mut self, buffer: &mut Vec<u8>) -> io::Result<usize> {
        match self {
            Self::Stdin(reader) => reader.read_until(b'\n', buffer),
            Self::File(reader) => reader.read_until(b'\n', buffer),
        }
    }
}

/// The operands read one after another; an unreadable one is reported and
/// skipped.
pub struct Input<'a, R, E> {
    names: std::vec::IntoIter<OsString>,
    stdin: Option<R>,
    open: Option<(OsString, Source<R>)>,
    stderr: &'a mut E,
    unreadable: bool,
}

impl<'a, R: Read, E: Write> Input<'a, R, E> {
    pub fn new(names: Vec<OsString>, stdin: R, stderr: &'a mut E) -> Self {
        Self {
            names: names.into_iter(),
            stdin: Some(stdin),
            open: None,
            stderr,
            unreadable: false,
        }
    }

    /// The exit status once reading stopped.
    pub const fn status(&self) -> u8 {
        if self.unreadable {
            EXIT_UNREADABLE_INPUT
        } else {
            EXIT_OK
        }
    }

    /// The next line of the inputs, or `None` after the last.
    pub fn next_line(&mut self) -> io::Result<Option<Line>> {
        loop {
            if self.open.is_none() && !self.open_next() {
                return Ok(None);
            }
            let (name, source) = self.open.as_mut().expect("an input is open");
            let mut text = Vec::new();
            match source.read_until_newline(&mut text) {
                Ok(0) => self.open = None,
                Ok(_) => {
                    let terminated = text.last() == Some(&b'\n');
                    if terminated {
                        text.pop();
                    }
                    return Ok(Some(Line {
                        text,
                        unterminated: !terminated,
                    }));
                }
                Err(error) => {
                    let name = display(name);
                    self.report(&format!("read error on {name}: {}", describe(&error)))?;
                    self.open = None;
                }
            }
        }
    }

    /// Reports the operands after the last line read that cannot be opened,
    /// as GNU `sed` does when it reads them.
    pub fn check_remaining(&mut self) {
        while let Some(name) = self.names.next() {
            if name != "-"
                && let Err(error) = File::open(Path::new(&name))
            {
                let _ = self.report(&format!(
                    "can't read {}: {}",
                    display(&name),
                    describe(&error)
                ));
            }
        }
    }

    /// Opens the next operand that can be read; `false` when none is left.
    fn open_next(&mut self) -> bool {
        while let Some(name) = self.names.next() {
            if name == "-" {
                // A second `-` finds standard input already at its end.
                if let Some(stdin) = self.stdin.take() {
                    self.open = Some((name, Source::Stdin(BufReader::new(stdin))));
                    return true;
                }
                continue;
            }
            match File::open(Path::new(&name)) {
                Ok(file) => {
                    self.open = Some((name, Source::File(BufReader::new(file))));
                    return true;
                }
                Err(error) => {
                    let _ = self.report(&format!(
                        "can't read {}: {}",
                        display(&name),
                        describe(&error)
                    ));
                }
            }
        }
        false
    }

    fn report(&mut self, message: &str) -> io::Result<()> {
        self.unreadable = true;
        writeln!(self.stderr, "sed: {message}")
    }
}

fn display(name: &OsString) -> String {
    if name == "-" {
        "standard input".to_owned()
    } else {
        name.to_string_lossy().into_owned()
    }
}

/// An I/O error as `strerror` words it, without Rust's `(os error N)` suffix.
pub fn describe(error: &io::Error) -> String {
    let text = error.to_string();
    match text.rfind(" (os error ") {
        Some(end) => text[..end].to_owned(),
        None => text,
    }
}
