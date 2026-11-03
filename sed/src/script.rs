//! The script language: `p` commands addressed by `N`, `N,M`, `N,$` or `$`.

use crate::Unsupported;

/// Which lines a `p` command prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Address {
    /// `N`, or `N,M` with `M <= N`, which GNU `sed` treats as `N` alone.
    Line(u64),
    /// `N,M` with `M > N`.
    Range(u64, u64),
    /// `N,$`.
    From(u64),
    /// `$`.
    Last,
}

impl Address {
    const fn matches(self, number: u64, last: bool) -> bool {
        match self {
            Self::Line(line) => number == line,
            Self::Range(from, to) => from <= number && number <= to,
            Self::From(from) => from <= number,
            Self::Last => last,
        }
    }
}

/// A parsed script: its `p` commands in order.
#[derive(Debug)]
pub struct Script {
    commands: Vec<Address>,
}

impl Script {
    /// Parses `text`: commands joined by `;` or newlines, each an address, `p`
    /// and GNU `sed`'s blanks between them.
    pub fn parse(text: &str) -> Result<Self, Unsupported> {
        let mut cursor = Cursor(text.as_bytes());
        let mut commands = Vec::new();
        loop {
            while cursor.skip(|byte| byte == b';' || byte == b'\n' || is_blank(byte)) {}
            if cursor.0.is_empty() {
                break;
            }
            commands.push(cursor.command()?);
        }
        if commands.is_empty() {
            return Err(Unsupported);
        }
        Ok(Self { commands })
    }

    /// Whether some command addresses `$`.
    pub fn needs_last_line(&self) -> bool {
        self.commands
            .iter()
            .any(|command| matches!(command, Address::Last))
    }

    /// The last line any command prints, or `None` when one runs to the end
    /// of the input.
    pub fn last_needed_line(&self) -> Option<u64> {
        self.commands
            .iter()
            .map(|command| match *command {
                Address::Line(line) | Address::Range(_, line) => Some(line),
                Address::From(_) | Address::Last => None,
            })
            .try_fold(0, |largest, line| line.map(|line| largest.max(line)))
    }

    /// How many commands print line `number`.
    pub fn copies(&self, number: u64, last: bool) -> usize {
        self.commands
            .iter()
            .filter(|command| command.matches(number, last))
            .count()
    }
}

/// The blanks GNU `sed` skips inside a command: everything `isspace` accepts
/// but the newline.
const fn is_blank(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | 0x0b | 0x0c | b'\r')
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    /// Consumes one byte satisfying `accept`.
    fn skip(&mut self, accept: impl Fn(u8) -> bool) -> bool {
        match self.0.split_first() {
            Some((&byte, rest)) if accept(byte) => {
                self.0 = rest;
                true
            }
            _ => false,
        }
    }

    fn blanks(&mut self) {
        while self.skip(is_blank) {}
    }

    /// A line number; one larger than `u64` saturates.
    fn number(&mut self) -> Result<u64, Unsupported> {
        let mut value = 0_u64;
        let mut digits = 0;
        while let Some((&byte, rest)) = self.0.split_first() {
            if !byte.is_ascii_digit() {
                break;
            }
            value = value
                .saturating_mul(10)
                .saturating_add(u64::from(byte - b'0'));
            digits += 1;
            self.0 = rest;
        }
        if digits == 0 {
            return Err(Unsupported);
        }
        Ok(value)
    }

    /// One command: an address, then `p`, then the end of the command.
    fn command(&mut self) -> Result<Address, Unsupported> {
        let address = if self.skip(|byte| byte == b'$') {
            Address::Last
        } else {
            // GNU `sed` has no line 0 to start at; an end of 0 is merely
            // before the start.
            let from = self.number().and_then(|from| match from {
                0 => Err(Unsupported),
                from => Ok(from),
            })?;
            self.blanks();
            if self.skip(|byte| byte == b',') {
                self.blanks();
                if self.skip(|byte| byte == b'$') {
                    Address::From(from)
                } else {
                    let to = self.number()?;
                    if to > from {
                        Address::Range(from, to)
                    } else {
                        Address::Line(from)
                    }
                }
            } else {
                Address::Line(from)
            }
        };
        self.blanks();
        if !self.skip(|byte| byte == b'p') {
            return Err(Unsupported);
        }
        self.blanks();
        match self.0.first() {
            None | Some(b';' | b'\n') => Ok(address),
            Some(_) => Err(Unsupported),
        }
    }
}
