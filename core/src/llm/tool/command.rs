//! Typed record of a command run, carried by `ToolResult::Command`.
//!
//! The model reads the prose [`CommandResult::render_for_model`] renders from
//! these fields; hosts read the fields and never parse that prose.

use alloc::string::String;

use askama::Template;

/// A file a command's output was written to.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StoredFile {
    /// Location relative to the session sandbox (for example `outputs/x.txt`).
    pub url: String,
    /// Absolute filesystem path of the same file.
    pub path: String,
}

/// One output stream of a command.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum CommandStream {
    /// Output small enough to keep in the result.
    Inline(String),
    /// Output written to a file.
    Stored {
        /// Where the full output is stored.
        file: StoredFile,
        /// What the model is shown (head/tail or folded view); empty when the
        /// output has no text view.
        preview: String,
        /// Whether `preview` is partial.
        truncated: bool,
    },
    /// Image output, always written to a file.
    Image {
        /// MIME type of the image.
        media_type: String,
        /// Where the image is stored.
        file: StoredFile,
    },
}

/// How a command run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "status", rename_all = "snake_case"))]
pub enum CommandOutcome {
    /// The command ran to completion.
    Exited {
        /// Process exit code.
        code: i32,
    },
    /// A background task; its output is being written to `output`.
    Running {
        /// Identifier of the background task.
        task_id: String,
        /// File the background output is written to.
        output: StoredFile,
    },
    /// The foreground run was killed at its time bound.
    Killed {
        /// Identifier of the killed run.
        task_id: String,
        /// Model-facing reason the run was killed.
        notice: String,
    },
}

/// Complete typed record of one command run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CommandResult {
    /// Standard output.
    pub stdout: CommandStream,
    /// Standard error, when the command wrote any.
    pub stderr: Option<CommandStream>,
    /// How the run ended.
    pub outcome: CommandOutcome,
}

#[derive(Template)]
#[template(path = "command_stream.txt", escape = "none")]
struct CommandStreamTemplate<'a> {
    stream: &'a CommandStream,
}

#[derive(Template)]
#[template(path = "command_result.txt", escape = "none")]
struct CommandResultTemplate<'a> {
    stdout: &'a str,
    stderr: Option<&'a str>,
    outcome: &'a CommandOutcome,
}

impl CommandStream {
    /// Returns `true` when the stream holds no output.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Inline(text) if text.is_empty())
    }

    /// Renders the stream as the newline-terminated block the model reads;
    /// an empty stream renders as the empty string.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render, which formatting in-memory
    /// strings cannot do.
    #[must_use]
    pub fn render_for_model(&self) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut block = CommandStreamTemplate { stream: self }
            .render()
            .expect("command stream template must render");
        if !block.ends_with('\n') {
            block.push('\n');
        }
        block
    }
}

impl CommandResult {
    /// Renders the prose the model reads.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render, which formatting in-memory
    /// strings cannot do.
    #[must_use]
    pub fn render_for_model(&self) -> String {
        let stdout = self.stdout.render_for_model();
        let stderr = self
            .stderr
            .as_ref()
            .map(CommandStream::render_for_model)
            .filter(|block| !block.is_empty());
        CommandResultTemplate {
            stdout: &stdout,
            stderr: stderr.as_deref(),
            outcome: &self.outcome,
        }
        .render()
        .expect("command result template must render")
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    fn file() -> StoredFile {
        StoredFile {
            url: "outputs/x.txt".to_string(),
            path: "/sandbox/outputs/x.txt".to_string(),
        }
    }

    fn result(
        stdout: CommandStream,
        stderr: Option<CommandStream>,
        outcome: CommandOutcome,
    ) -> String {
        CommandResult {
            stdout,
            stderr,
            outcome,
        }
        .render_for_model()
    }

    #[test]
    fn placeholder_only_when_both_streams_are_empty() {
        assert_eq!(
            result(
                CommandStream::Inline(String::new()),
                None,
                CommandOutcome::Exited { code: 0 }
            ),
            "(no output)\n"
        );
        assert_eq!(
            result(
                CommandStream::Inline(String::new()),
                Some(CommandStream::Inline("oops".to_string())),
                CommandOutcome::Exited { code: 2 },
            ),
            "stderr:\noops\nExit code: 2.\n"
        );
    }

    #[test]
    fn stored_streams_state_url_and_absolute_path() {
        let stored = CommandStream::Stored {
            file: file(),
            preview: "head".to_string(),
            truncated: true,
        };
        assert_eq!(
            result(stored, None, CommandOutcome::Exited { code: 0 }),
            "head\n[Partial view: the full output is saved to outputs/x.txt (absolute path: /sandbox/outputs/x.txt). Use head/tail/grep/sed on that file to read the rest.]\n"
        );
        let image = CommandStream::Image {
            media_type: "image/png".to_string(),
            file: file(),
        };
        assert_eq!(
            result(image, None, CommandOutcome::Exited { code: 0 }),
            "[Image: image/png, saved to outputs/x.txt (absolute path: /sandbox/outputs/x.txt).]\n"
        );
    }

    #[test]
    fn running_and_killed_outcomes_render_their_facts() {
        assert_eq!(
            result(
                CommandStream::Inline(String::new()),
                None,
                CommandOutcome::Running {
                    task_id: "t1".to_string(),
                    output: file(),
                },
            ),
            "Started in background as task 't1'; output is being written to outputs/x.txt (absolute path: /sandbox/outputs/x.txt).\n"
        );
        assert_eq!(
            result(
                CommandStream::Inline("partial\n".to_string()),
                None,
                CommandOutcome::Killed {
                    task_id: "t2".to_string(),
                    notice: "Killed at the time bound.".to_string(),
                },
            ),
            "partial\nKilled at the time bound.\nTask ID: 't2'.\n"
        );
    }
}
