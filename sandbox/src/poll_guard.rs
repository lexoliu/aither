//! Refuses waiting by polling instead of by mechanism.
//!
//! Two shapes of the same mistake: a foreground `terminal` script that runs
//! `sleep`, and a command that reads a file or a background task's output
//! again when nothing changed since the last read. A finished background
//! task already reports its completion to the agent, and a background script
//! that exits when its condition holds (`until [ -f done ]; do sleep 5;
//! done`) is the way to wait on anything else — so both are refused ahead of
//! running, with the refusal naming the right pattern.
//!
//! Detection walks the parsed script through [`crate::kill_guard`]'s
//! traversal, so `sleep` in a string or a file name does not trigger it.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;

use crate::kill_guard::{file_name, walk_invocations};

/// A call refused because it waits or re-reads instead of using the
/// mechanism that reports the change.
///
/// Tools return it as their error inside a [`Refused`](crate::Refused), so
/// an agent loop can tell a refused poll from any other failure by
/// downcasting; [`SandboxNotices::poll_refusal`](crate::SandboxNotices::poll_refusal)
/// renders its text.
#[derive(Debug)]
pub enum PollRefusal {
    /// The command invokes `sleep` in a foreground call.
    Sleep,
    /// Every file the command reads is unchanged since its last read.
    UnchangedFileRead {
        /// The file paths the command re-reads, as written in the command.
        paths: Vec<PathBuf>,
    },
    /// A `terminal_read` would return what the task's last read returned.
    UnchangedTaskRead {
        /// The background task whose output did not change.
        task_id: String,
    },
}

impl PollRefusal {
    /// Whether the refused call re-read output that has not changed: it
    /// learned nothing, so only a background event can give the model
    /// something new. A refused `sleep` is not one — the model has not yet
    /// looked at anything.
    #[must_use]
    pub const fn is_unchanged_read(&self) -> bool {
        matches!(
            self,
            Self::UnchangedFileRead { .. } | Self::UnchangedTaskRead { .. }
        )
    }
}

/// Whether `script` invokes `sleep` anywhere the script itself would wait on
/// it — pipelines, compound commands, substitutions, wrappers and
/// `sh -c`/`eval` strings included. A `sleep` in an `&`-asynchronous
/// position does not count: that is spawned work, not the call waiting.
///
/// An unparsable script reports `false`: the kill guard already refuses
/// those before this check runs.
#[must_use]
pub fn contains_sleep(script: &str) -> bool {
    let found = Cell::new(false);
    let _ = walk_invocations(script, &|argv, backgrounded| {
        if !backgrounded
            && argv
                .first()
                .is_some_and(|program| file_name(program) == "sleep")
        {
            found.set(true);
        }
    });
    found.get()
}

/// Programs that print a file's content. `tail -f` follows instead of
/// re-reading and is not collected.
const READERS: &[&str] = &["cat", "head", "tail", "less", "more", "bat", "view"];

/// The arguments `script` reads files from: positional words of `cat`,
/// `head`, `tail`, `less`, `more`, `bat` and `view` invocations, excluding
/// `tail`'s follow flags. Words computed at run time are not seen; option
/// values that are not files are filtered by the caller's `stat`.
#[must_use]
pub fn file_reads(script: &str) -> Vec<String> {
    let targets = RefCell::new(Vec::new());
    let _ = walk_invocations(script, &|argv, _backgrounded| {
        let Some((program, arguments)) = argv.split_first() else {
            return;
        };
        let name = file_name(program);
        if !READERS.contains(&name.as_str()) {
            return;
        }
        if name == "tail"
            && arguments.iter().any(|argument| {
                argument == "-f"
                    || argument == "-F"
                    || argument.starts_with("--follow")
                    || argument
                        .strip_prefix('-')
                        .is_some_and(|flags| !flags.starts_with('-') && flags.contains('f'))
            })
        {
            return;
        }
        for argument in arguments {
            if !argument.starts_with('-') {
                targets.borrow_mut().push(argument.clone());
            }
        }
    });
    targets.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleep_is_found_in_command_position() {
        assert!(contains_sleep("sleep 5"));
        assert!(contains_sleep("sleep 60; tail -50 out.txt"));
        assert!(contains_sleep("timeout 5 sleep 3"));
        assert!(contains_sleep("env sleep 1"));
        assert!(contains_sleep("until [ -f done ]; do sleep 5; done"));
        assert!(contains_sleep("sh -c 'sleep 2'"));
        assert!(contains_sleep("echo hi && sleep 1"));
    }

    #[test]
    fn sleep_in_an_async_position_is_not_the_call_waiting() {
        assert!(!contains_sleep("sleep 30 & echo $! > child.pid; wait"));
        assert!(!contains_sleep("(sleep 1; echo one > one.txt) & wait"));
        assert!(!contains_sleep("sleep 300 & pid=$!; kill $pid"));
        // A second command after `&` still blocks the call.
        assert!(contains_sleep("sleep 1 & sleep 2"));
        assert!(contains_sleep("sleep 1 & sleep"));
    }

    #[test]
    fn sleep_in_strings_and_names_is_not_found() {
        assert!(!contains_sleep("echo 'sleep'"));
        assert!(!contains_sleep("echo sleep"));
        assert!(!contains_sleep("cat sleep.txt"));
        assert!(!contains_sleep("grep sleep log.txt"));
        assert!(!contains_sleep("sleepy 5"));
    }

    #[test]
    fn file_reads_collects_reader_targets() {
        assert_eq!(file_reads("cat a.txt"), ["a.txt"]);
        assert_eq!(file_reads("tail -50 out.txt"), ["out.txt"]);
        assert_eq!(file_reads("sleep 60; tail -50 out.txt"), ["out.txt"]);
        assert_eq!(file_reads("head -n 5 a.txt b.txt"), ["5", "a.txt", "b.txt"]);
        assert_eq!(file_reads("tail -f log.txt"), [] as [String; 0]);
        assert_eq!(file_reads("tail -fn5 log.txt"), [] as [String; 0]);
        assert_eq!(file_reads("tail --follow=name log.txt"), [] as [String; 0]);
        assert_eq!(file_reads("echo tail out.txt"), [] as [String; 0]);
        // A quoted string is one file name, not a parsed invocation.
        assert_eq!(file_reads("cat 'tail -f x'"), ["tail -f x"]);
        assert_eq!(file_reads("grep 'tail -f' x"), [] as [String; 0]);
    }
}
