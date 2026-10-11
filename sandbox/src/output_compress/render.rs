//! Terminal rendering stage: turns a raw byte stream from a PTY-less child
//! into the text a terminal would have displayed.
//!
//! This is a line-oriented renderer, not a screen emulator: there is no known
//! terminal width, so lines are never wrapped or truncated and vertical cursor
//! motion carries no meaning. What it does model is everything that changes
//! the content of the *current* line -- `\r`, backspace, horizontal cursor
//! motion and erase-in-line -- so a progress bar that rewrites one line many
//! times collapses to the frame that was on screen when the line ended.
//!
//! Escape sequences are recognised by [`vte`], Alacritty's implementation of
//! the DEC ANSI parser state machine, so a CSI, OSC or DCS sequence is
//! consumed as a unit instead of leaving its tail (`[32m`, `[2K`) in the text.

use vte::{Params, Parser, Perform};

/// Renders `text` as a terminal would display it.
///
/// Escape sequences are consumed, `\r` rewinds to the start of the current
/// line so later output overwrites earlier output, and `\r\n` ends a line
/// exactly like a bare `\n`. Printable content, including tabs and trailing
/// whitespace, is preserved byte for byte, so the result is lossless for
/// output that contains no escapes.
///
/// A line terminator is never invented: output that ends without one --
/// `printf abc`, `base64 -w0`, a file with no final newline, a stream cut off
/// mid-escape -- renders without one too.
#[must_use]
pub fn render_terminal(text: &str) -> String {
    let mut screen = LineRenderer::with_capacity(text.len());
    let mut parser = Parser::new();
    parser.advance(&mut screen, text.as_bytes());
    screen.finish()
}

/// Accumulates rendered lines, tracking a cursor column inside the line that
/// is currently being built.
struct LineRenderer {
    rendered: String,
    line: Vec<char>,
    cursor: usize,
}

impl LineRenderer {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            rendered: String::with_capacity(capacity),
            line: Vec::new(),
            cursor: 0,
        }
    }

    /// Flushes the unterminated last line, if any, and returns the rendered
    /// text.
    ///
    /// A stream that ends without a newline -- a lone `\r` at the end of a
    /// progress line, or a shell prompt -- still has a last frame to show,
    /// but it is emitted without a terminator the stream never carried.
    fn finish(mut self) -> String {
        self.rendered.extend(self.line.drain(..));
        self.rendered
    }

    fn end_line(&mut self) {
        self.rendered.extend(self.line.drain(..));
        self.rendered.push('\n');
        self.cursor = 0;
    }

    /// Writes `c` at the cursor, overwriting what is there and extending the
    /// line with spaces when the cursor sits past its end.
    fn put(&mut self, c: char) {
        while self.line.len() < self.cursor {
            self.line.push(' ');
        }
        if self.cursor == self.line.len() {
            self.line.push(c);
        } else {
            self.line[self.cursor] = c;
        }
        self.cursor += 1;
    }

    /// Erase-in-line (CSI K): 0 erases to the end of the line, 1 to its
    /// start, 2 the whole line.
    fn erase_in_line(&mut self, mode: u16) {
        match mode {
            0 => self.line.truncate(self.cursor.min(self.line.len())),
            1 => {
                let upto = self.cursor.min(self.line.len());
                for cell in &mut self.line[..upto] {
                    *cell = ' ';
                }
            }
            2 => self.line.clear(),
            _ => {}
        }
    }
}

/// First parameter of a CSI sequence, defaulting to `default` when it is
/// absent or zero, as the VT specification requires.
fn first_param(params: &Params, default: u16) -> u16 {
    match params
        .iter()
        .next()
        .and_then(|group| group.first().copied())
    {
        Some(0) | None => default,
        Some(value) => value,
    }
}

impl Perform for LineRenderer {
    fn print(&mut self, c: char) {
        self.put(c);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            b'\n' | 0x0b | 0x0c => self.end_line(),
            b'\r' => self.cursor = 0,
            // Kept literally rather than expanded to a tab stop: output that
            // is read back as TSV or as a patch depends on the tab itself.
            b'\t' => self.put('\t'),
            0x08 => self.cursor = self.cursor.saturating_sub(1),
            // Every other C0/C1 control -- bell, shift in/out, nulls from a
            // truncated write -- has no textual content.
            _ => {}
        }
    }

    fn csi_dispatch(&mut self, params: &Params, _intermediates: &[u8], ignore: bool, action: char) {
        if ignore {
            return;
        }
        match action {
            // CHA / HPA: absolute column, 1-based.
            'G' | '`' => self.cursor = (first_param(params, 1) as usize).saturating_sub(1),
            // CUF: cursor forward.
            'C' => self.cursor = self.cursor.saturating_add(first_param(params, 1) as usize),
            // CUB: cursor back.
            'D' => self.cursor = self.cursor.saturating_sub(first_param(params, 1) as usize),
            'K' => self.erase_in_line(first_param(params, 0)),
            // Colours (SGR), vertical motion, scrolling, mode changes and
            // queries change no content on this line.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_sgr_colour_sequences_without_residue() {
        let coloured = "\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m aither-sandbox v0.1.0\n";
        assert_eq!(
            render_terminal(coloured),
            "   Compiling aither-sandbox v0.1.0\n"
        );
    }

    #[test]
    fn renders_carriage_return_progress_as_last_frame() {
        let progress = "Downloading  10%\rDownloading  55%\rDownloading 100%\n";
        assert_eq!(render_terminal(progress), "Downloading 100%\n");
    }

    #[test]
    fn erase_in_line_clears_longer_previous_frame() {
        let spinner = "working on a long step\r\u{1b}[2Kdone\n";
        assert_eq!(render_terminal(spinner), "done\n");
    }

    #[test]
    fn carriage_return_newline_ends_the_line() {
        assert_eq!(render_terminal("alpha\r\nbeta\r\n"), "alpha\nbeta\n");
    }

    #[test]
    fn lone_carriage_return_at_end_keeps_the_frame() {
        assert_eq!(render_terminal("progress 99%\r"), "progress 99%");
    }

    #[test]
    fn carriage_return_at_end_still_rewinds_for_what_follows() {
        assert_eq!(render_terminal("abc\r\u{1b}[2Kxyz"), "xyz");
    }

    #[test]
    fn output_without_a_final_newline_gains_none() {
        // `printf abc`, `base64 -w0` and files with no final newline must
        // come through byte for byte.
        assert_eq!(render_terminal("abc"), "abc");
        assert_eq!(render_terminal("first\nsecond"), "first\nsecond");
    }

    #[test]
    fn column_absolute_rewrites_from_the_requested_column() {
        // `[1G` is what a spinner emits instead of `\r`.
        assert_eq!(render_terminal("abcdef\u{1b}[3GXY\n"), "abXYef\n");
    }

    #[test]
    fn backspace_rewinds_one_column() {
        assert_eq!(render_terminal("abc\u{8}d\n"), "abd\n");
    }

    #[test]
    fn tabs_and_trailing_spaces_survive() {
        assert_eq!(render_terminal("a\tb  \n"), "a\tb  \n");
    }

    #[test]
    fn osc_window_title_leaves_no_text() {
        assert_eq!(
            render_terminal("\u{1b}]0;cargo test\u{7}running 1 test\n"),
            "running 1 test\n"
        );
    }

    #[test]
    fn csi_with_too_many_parameters_leaves_no_residue() {
        // More parameters than the state machine tracks: the sequence is
        // flagged as ignored, so it must move no cursor and print no digits.
        let params = (0..40)
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(";");
        assert_eq!(render_terminal(&format!("x\u{1b}[{params}Gy\n")), "xy\n");
    }

    #[test]
    fn two_character_escape_consumes_its_final_byte() {
        // `ESC c` is a reset, not the letter `c`.
        assert_eq!(render_terminal("a\u{1b}cb\n"), "ab\n");
    }

    #[test]
    fn trailing_escape_at_end_of_stream_is_dropped() {
        assert_eq!(render_terminal("tail\u{1b}["), "tail");
    }

    #[test]
    fn null_and_bell_bytes_are_dropped() {
        assert_eq!(render_terminal("hello\0world\u{7}!\n"), "helloworld!\n");
    }

    #[test]
    fn empty_input_renders_empty() {
        assert_eq!(render_terminal(""), "");
    }

    #[test]
    fn multibyte_text_is_preserved() {
        assert_eq!(render_terminal("héllo → 世界\n"), "héllo → 世界\n");
    }
}
