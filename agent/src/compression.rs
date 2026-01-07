//! Smart context compression for managing conversation history.

use std::collections::{HashMap, HashSet};

use aither_core::{LanguageModel, llm::Message};

use crate::context::ConversationMemory;

/// Strategy for managing conversation context.
#[derive(Debug, Clone)]
pub enum ContextStrategy {
    /// No compression - keep all messages until context is full, then stop.
    Unlimited,

    /// Smart compression with selective preservation (default).
    Smart(SmartCompressionConfig),
}

impl Default for ContextStrategy {
    fn default() -> Self {
        Self::Smart(SmartCompressionConfig::default())
    }
}

/// Configuration for smart context compression.
#[derive(Debug, Clone)]
pub struct SmartCompressionConfig {
    /// Trigger compression at this fraction of context window (default: 0.7).
    pub trigger_threshold: f32,

    /// Emergency compaction threshold (default: 0.9).
    pub emergency_threshold: f32,

    /// Number of recent messages to always keep verbatim.
    pub preserve_recent: usize,

    /// Types of content to preserve during compression.
    pub preserve: PreserveConfig,

    /// Compression level (trade-off between quality and size).
    pub level: CompressionLevel,
}

impl Default for SmartCompressionConfig {
    fn default() -> Self {
        Self {
            trigger_threshold: 0.7,
            emergency_threshold: 0.9,
            preserve_recent: 8,
            preserve: PreserveConfig::default(),
            level: CompressionLevel::Standard,
        }
    }
}

/// Configuration for what content to preserve during compression.
#[derive(Debug, Clone)]
pub struct PreserveConfig {
    /// Keep file paths verbatim.
    pub file_paths: bool,
    /// Keep error messages verbatim.
    pub errors: bool,
    /// Keep shell commands verbatim.
    pub commands: bool,
    /// Keep code snippets verbatim (if false, summarize instead).
    pub code_snippets: bool,
    /// Keep tool results verbatim (if false, compress).
    pub tool_results: bool,
}

impl Default for PreserveConfig {
    fn default() -> Self {
        Self {
            file_paths: true,
            errors: true,
            commands: true,
            code_snippets: false,
            tool_results: false,
        }
    }
}

/// Compression aggressiveness level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CompressionLevel {
    /// Keep more detail, less compression.
    Light,
    /// Balanced compression.
    #[default]
    Standard,
    /// Maximum compression.
    Aggressive,
}

/// Content extracted and preserved during compression.
#[derive(Debug, Default)]
pub struct PreservedContent {
    /// File paths found in messages.
    pub file_paths: Vec<String>,
    /// Error messages found.
    pub errors: Vec<String>,
    /// Shell commands found.
    pub commands: Vec<String>,
}

/// Estimate tokens in a string (rough approximation: ~4 chars per token).
#[must_use]
pub fn estimate_tokens(content: &str) -> usize {
    content.len() / 4
}

/// Estimate context usage as a fraction of the window.
#[must_use]
pub fn estimate_context_usage(messages: &[Message], context_window: usize) -> f32 {
    let message_tokens: usize = messages.iter().map(|m| estimate_tokens(m.content())).sum();
    message_tokens as f32 / context_window as f32
}

// Prompt templates loaded from files
const COMPRESSION_SYSTEM_PROMPT: &str = include_str!("prompts/compression_system.txt");
const COMPRESSION_USER_TEMPLATE: &str = include_str!("prompts/compression_user.txt");

impl SmartCompressionConfig {
    /// Extract content that should be preserved from messages.
    #[must_use]
    pub fn extract_preserved(&self, messages: &[Message]) -> PreservedContent {
        let mut preserved = PreservedContent::default();

        for msg in messages {
            let content = msg.content();

            if self.preserve.file_paths {
                preserved.file_paths.extend(extract_file_paths(content));
            }

            if self.preserve.errors {
                preserved.errors.extend(extract_errors(content));
            }

            if self.preserve.commands {
                preserved.commands.extend(extract_commands(content));
            }
        }

        preserved
    }

    /// Identify indices of stale tool calls that can be safely removed.
    ///
    /// Stale tool calls include:
    /// - Tool results that were just acknowledgments
    /// - Read operations for files that were later modified
    #[must_use]
    pub fn find_stale_tool_calls(&self, messages: &[Message]) -> HashSet<usize> {
        let mut stale = HashSet::new();
        let mut file_versions: HashMap<String, usize> = HashMap::new();

        for (idx, msg) in messages.iter().enumerate() {
            let content = msg.content();

            // Check for trivial results
            if is_trivial_result(content) {
                stale.insert(idx);
            }

            // Track file reads that may be superseded
            if content.contains("read_file") || content.contains("Read") {
                if let Some(path) = extract_single_path(content) {
                    if let Some(&later_idx) = file_versions.get(&path) {
                        if later_idx > idx {
                            stale.insert(idx);
                        }
                    }
                }
            }

            // Track file modifications
            if content.contains("write_file")
                || content.contains("edit_file")
                || content.contains("Write")
                || content.contains("Edit")
            {
                if let Some(path) = extract_single_path(content) {
                    file_versions.insert(path, idx);
                }
            }
        }

        stale
    }

    /// Generate a compressed summary of messages.
    ///
    /// # Errors
    ///
    /// Returns an error if the LLM fails to generate a summary.
    pub async fn generate_summary<LLM: LanguageModel>(
        &self,
        llm: &LLM,
        messages: &[Message],
        preserved: &PreservedContent,
    ) -> aither_core::Result<String> {
        let prompt = COMPRESSION_USER_TEMPLATE
            .replace("{file_paths}", &preserved.file_paths.join(", "))
            .replace("{errors}", &preserved.errors.join("\n"))
            .replace("{commands}", &preserved.commands.join("\n"))
            .replace("{dialogue}", &format_messages(messages));

        llm.generate(aither_core::llm::oneshot(COMPRESSION_SYSTEM_PROMPT, prompt))
            .await
    }
}

/// Extract file paths from content.
fn extract_file_paths(content: &str) -> Vec<String> {
    let mut paths = Vec::new();

    // Match common path patterns
    for word in content.split_whitespace() {
        let word = word.trim_matches(|c: char| c == '"' || c == '\'' || c == '`' || c == ',');

        // Unix-style absolute paths
        if word.starts_with('/') && word.len() > 1 && !word.contains(' ') {
            paths.push(word.to_string());
        }

        // Relative paths with extensions
        if (word.contains('/') || word.contains('.'))
            && !word.starts_with("http")
            && !word.contains(' ')
            && word.len() > 2
        {
            // Check for common file extensions
            let extensions = [
                ".rs", ".py", ".js", ".ts", ".go", ".java", ".c", ".cpp", ".h", ".md", ".txt",
                ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css",
            ];
            if extensions.iter().any(|ext| word.ends_with(ext)) {
                paths.push(word.to_string());
            }
        }
    }

    paths.sort();
    paths.dedup();
    paths
}

/// Extract a single path from content (for file operation tracking).
fn extract_single_path(content: &str) -> Option<String> {
    extract_file_paths(content).into_iter().next()
}

/// Extract error messages from content.
fn extract_errors(content: &str) -> Vec<String> {
    let mut errors = Vec::new();

    // Look for common error patterns
    for line in content.lines() {
        let lower = line.to_lowercase();
        if lower.contains("error")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("exception")
        {
            errors.push(line.trim().to_string());
        }
    }

    errors
}

/// Extract shell commands from content.
fn extract_commands(content: &str) -> Vec<String> {
    let mut commands = Vec::new();

    // Look for command patterns (lines starting with $ or common commands)
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('$') || trimmed.starts_with('>') {
            commands.push(trimmed.to_string());
        }

        // Common command prefixes
        let cmd_prefixes = [
            "cargo ", "npm ", "git ", "docker ", "make ", "rustc ", "python ", "node ", "go ",
            "cd ", "ls ", "cat ", "mkdir ", "rm ", "mv ", "cp ",
        ];
        if cmd_prefixes.iter().any(|p| trimmed.starts_with(p)) {
            commands.push(trimmed.to_string());
        }
    }

    commands
}

/// Check if a tool result is trivial (just an acknowledgment).
fn is_trivial_result(result: &str) -> bool {
    let trivial_patterns = [
        "ok",
        "success",
        "done",
        "file written",
        "file saved",
        "completed",
    ];
    let lower = result.to_lowercase();
    trivial_patterns.iter().any(|t| lower.contains(t)) && result.len() < 50
}

/// Format messages for compression prompt.
fn format_messages(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|msg| format!("{:?}: {}", msg.role(), msg.content()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_file_paths() {
        let content = "I modified /Users/user/project/src/main.rs and also checked lib.rs";
        let paths = extract_file_paths(content);
        assert!(paths.contains(&"/Users/user/project/src/main.rs".to_string()));
        assert!(paths.contains(&"lib.rs".to_string()));
    }

    #[test]
    fn test_trivial_result() {
        assert!(is_trivial_result("OK"));
        assert!(is_trivial_result("File written successfully"));
        assert!(!is_trivial_result(
            "This is a long result with actual content that should be preserved"
        ));
    }

    #[test]
    fn test_estimate_tokens() {
        let content = "This is a test string with some content";
        let tokens = estimate_tokens(content);
        assert!(tokens > 0);
        assert!(tokens < content.len());
    }
}
