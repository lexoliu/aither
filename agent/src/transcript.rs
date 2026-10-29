//! Readable transcript for LLM context recovery.
//!
//! A clean, append-only markdown file placed in the agent's sandbox so it can
//! read its own history after compaction. Contains the conversational flow
//! (user messages, assistant text, tool invocations with brief results) but
//! omits internal details (reasoning tokens, signatures, model metadata).
//!
//! When compaction occurs a short marker is written -- no summary is included
//! so the model knows information was lost and will actively recover from
//! files or by re-reading the transcript.

use std::fmt::Write;
use std::path::{Path, PathBuf};

use aither_core::llm::ToolResult;
use async_fs::OpenOptions;
use futures_lite::AsyncWriteExt;

/// Append-only writer for the readable transcript.
#[derive(Debug, Clone)]
pub struct Transcript {
    path: PathBuf,
}

impl Transcript {
    /// Creates a transcript that appends to `path`.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The file this transcript is written to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends a user turn.
    pub async fn write_user_message(&self, content: &str) {
        let mut block = String::new();
        let _ = writeln!(block, "\n## User\n");
        let _ = writeln!(block, "{content}\n");
        self.append(&block).await;
    }

    /// Appends an assistant turn. Empty content is skipped.
    pub async fn write_assistant_text(&self, content: &str) {
        if content.is_empty() {
            return;
        }
        let mut block = String::new();
        let _ = writeln!(block, "\n## Assistant\n");
        let _ = writeln!(block, "{content}\n");
        self.append(&block).await;
    }

    /// Appends the command a tool was invoked with.
    pub async fn write_tool_call(&self, name: &str, command: &str) {
        let mut block = String::new();
        let _ = writeln!(block, "\n### Tool: {name}\n");
        let _ = writeln!(block, "```");
        let _ = writeln!(block, "{command}");
        let _ = writeln!(block, "```\n");
        self.append(&block).await;
    }

    /// Append a tool result block.
    pub async fn write_tool_result(&self, name: &str, result: &ToolResult) {
        let mut block = String::new();
        let rendered = result.render_for_model();
        let truncated = truncate_lines(&rendered, 200);
        if result.is_error() {
            let _ = writeln!(block, "-> {name} (error): {truncated}\n");
        } else {
            let _ = writeln!(block, "-> {name}: {truncated}\n");
        }
        self.append(&block).await;
    }

    /// Marker written on compaction. Deliberately excludes the summary so the
    /// model knows context was lost and should recover from files.
    pub async fn write_compact_marker(&self) {
        let marker = "\n---\n\n*[Context was compacted here. Earlier messages were summarized and removed. Details may be missing -- recover from files or re-read this transcript if needed.]*\n\n---\n\n";
        self.append(marker).await;
    }

    async fn append(&self, content: &str) {
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await;

        match result {
            Ok(mut file) => {
                if let Err(e) = file.write_all(content.as_bytes()).await {
                    tracing::warn!("Failed to append to transcript: {}", e);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to open transcript for append: {}", e);
            }
        }
    }
}

fn truncate_lines(s: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= max_lines {
        return s.to_string();
    }
    let kept: Vec<&str> = lines[..max_lines].to_vec();
    let omitted = lines.len() - max_lines;
    format!("{}\n... ({omitted} lines omitted)", kept.join("\n"))
}
