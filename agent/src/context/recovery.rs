//! Cleaning a context a compaction failed on.
//!
//! A compaction replays the whole window, so what makes it fail is usually
//! what makes the window big or noisy: system alerts and reminders that
//! accumulated, and tool outputs of tens of thousands of tokens. Recovery
//! removes the first and moves the second out of the window — to files the
//! agent can still read, leaving the path and a short head in place — so the
//! synchronous rerun summarises a context the summariser can cope with.

use std::path::Path;

use askama::Template;

use super::reassembly::truncate_on_char_boundary;
use super::window::Context;

/// What recovery took out of the window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// System alerts and reminders removed.
    pub alerts_removed: usize,
    /// Tool outputs moved to files.
    pub outputs_spilled: usize,
}

/// Where spilled outputs go, relative to the working directory.
const SPILL_DIR: &str = "outputs";

#[derive(Template)]
#[template(path = "spilled_tool_output.txt", escape = "none")]
struct SpilledToolOutput<'a> {
    head: &'a str,
    bytes: usize,
    path: &'a str,
    absolute_path: &'a str,
}

/// How much of a context recovery may move out of the window.
#[derive(Debug, Clone, Copy)]
pub struct SpillLimits {
    /// Bytes a tool output may hold before it is moved.
    pub threshold: usize,
    /// Bytes of a moved output that stay in place.
    pub head: usize,
}

/// Removes the alerts and reminders numbered below `fork_point` (the fork
/// point of the compaction about to run) from `context`, and moves the tool
/// outputs over `limits.threshold` into `working_dir`.
///
/// With no working directory there is nowhere the agent could read a file
/// from, so outputs stay where they are and only the alerts go.
///
/// # Errors
///
/// Returns the I/O error of the first file that could not be written; the
/// output it held is left in the window.
pub async fn clean_for_recovery(
    context: &mut Context,
    fork_point: u64,
    working_dir: Option<&Path>,
    limits: SpillLimits,
) -> std::io::Result<RecoveryReport> {
    let alerts_removed = context.prune_system_messages_below(fork_point);
    let mut outputs_spilled = 0;
    if let Some(working_dir) = working_dir {
        let dir = working_dir.join(SPILL_DIR);
        for output in context.long_tool_outputs(limits.threshold) {
            async_fs::create_dir_all(&dir).await?;
            let name = format!("compaction-{}.txt", file_stem(&output.tool_call_id));
            let absolute = dir.join(&name);
            async_fs::write(&absolute, output.content.as_bytes()).await?;
            let relative = format!("{SPILL_DIR}/{name}");
            let replacement = SpilledToolOutput {
                head: &truncate_on_char_boundary(&output.content, limits.head),
                bytes: output.content.len(),
                path: &relative,
                absolute_path: &absolute.to_string_lossy(),
            }
            .render()
            .expect("spilled tool output template must render");
            context.replace_tool_output(output.seq, replacement);
            outputs_spilled += 1;
        }
    }
    Ok(RecoveryReport {
        alerts_removed,
        outputs_spilled,
    })
}

/// A tool call id as a file name stem: ids come from providers and may hold
/// characters a path cannot.
fn file_stem(id: &str) -> String {
    id.bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
                char::from(byte)
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use aither_core::llm::Message;

    use super::*;

    fn limits() -> SpillLimits {
        SpillLimits {
            threshold: 100,
            head: 10,
        }
    }

    #[tokio::test]
    async fn long_outputs_move_to_files_and_alerts_go() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut context = Context::new();
        context.push(Message::user("run it"));
        context.push(Message::system("<system-reminder>stale</system-reminder>"));
        context.push(Message::tool("call/1", "é".repeat(200)));
        context.push(Message::tool("call-2", "short"));

        let report = clean_for_recovery(&mut context, u64::MAX, Some(dir.path()), limits())
            .await
            .expect("recovery");

        assert_eq!(
            report,
            RecoveryReport {
                alerts_removed: 1,
                outputs_spilled: 1
            }
        );
        let spilled = dir.path().join("outputs").join("compaction-call_1.txt");
        assert_eq!(
            std::fs::read_to_string(&spilled).expect("spill file"),
            "é".repeat(200)
        );
        let messages = context.conversation_messages();
        assert_eq!(messages.len(), 3);
        let stub = messages[1].content();
        assert!(stub.starts_with("ééééé\n"), "short head kept: {stub}");
        assert!(stub.contains("outputs/compaction-call_1.txt"));
        assert!(stub.contains(&*spilled.to_string_lossy()));
        assert_eq!(messages[2].content(), "short");
    }

    #[tokio::test]
    async fn only_alerts_below_the_fork_point_go() {
        let mut context = Context::new();
        context.push(Message::system("old alert"));
        let fork_point = context.fork_point();
        context.push(Message::system("new alert"));

        let report = clean_for_recovery(&mut context, fork_point, None, limits())
            .await
            .expect("recovery");

        assert_eq!(report.alerts_removed, 1);
        let kept: Vec<_> = context.recent().iter().map(|m| m.content()).collect();
        assert_eq!(kept, ["new alert"]);
    }

    #[tokio::test]
    async fn without_a_working_directory_only_the_alerts_go() {
        let mut context = Context::new();
        context.push(Message::system("alert"));
        context.push(Message::tool("call-1", "x".repeat(500)));

        let report = clean_for_recovery(&mut context, u64::MAX, None, limits())
            .await
            .expect("recovery");

        assert_eq!(report.alerts_removed, 1);
        assert_eq!(report.outputs_spilled, 0);
        assert_eq!(context.conversation_messages()[0].content().len(), 500);
    }
}
