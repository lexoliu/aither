//! Adapter for converting aither agent events to ACP session updates.

use aither_agent::AgentEvent;
use aither_core::llm::ToolResult;

use crate::protocol::{
    ContentBlock, ContentChunk, SessionUpdate, TextContent, ToolCall, ToolCallStatus,
    ToolCallUpdate, ToolKind,
};

/// Converts a stream of [`AgentEvent`]s into ACP [`SessionUpdate`]s.
///
/// `AgentEvent::ToolCallDelta` carries incremental argument fragments; the
/// mapper accumulates them per call so each emitted `ToolCallUpdate` reports
/// the arguments received so far.
#[derive(Debug, Default)]
pub struct SessionUpdateMapper {
    /// Arguments accumulated per tool call, keyed by call id.
    tool_arguments: std::collections::HashMap<String, String>,
}

impl SessionUpdateMapper {
    /// Convert an `AgentEvent` to an ACP `SessionUpdate`.
    ///
    /// Returns `None` for events that don't have a direct ACP mapping
    /// (like `TurnComplete` and `Complete`, which are handled separately).
    #[must_use]
    pub fn map(&mut self, event: &AgentEvent) -> Option<SessionUpdate> {
        match event {
            AgentEvent::Text(text) => Some(SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent {
                    text: text.clone(),
                    annotations: None,
                    meta: None,
                }),
                message_id: None,
                meta: None,
            })),

            AgentEvent::Reasoning(text) => Some(SessionUpdate::AgentThoughtChunk(ContentChunk {
                content: ContentBlock::Text(TextContent {
                    text: text.clone(),
                    annotations: None,
                    meta: None,
                }),
                message_id: None,
                meta: None,
            })),

            AgentEvent::ToolCallStart {
                id,
                name,
                arguments,
            } => {
                self.tool_arguments.insert(id.clone(), arguments.clone());
                Some(SessionUpdate::ToolCall(ToolCall {
                    tool_call_id: id.clone(),
                    title: format_tool_title(name, arguments),
                    kind: Some(infer_tool_kind(name)),
                    status: Some(ToolCallStatus::Pending),
                    content: vec![],
                    locations: vec![],
                    raw_input: serde_json::from_str(arguments).ok(),
                    raw_output: None,
                    meta: None,
                }))
            }

            AgentEvent::ToolCallDelta {
                id,
                name,
                arguments_fragment,
            } => {
                let arguments = self.tool_arguments.entry(id.clone()).or_default();
                arguments.push_str(arguments_fragment);
                let arguments = arguments.as_str();
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    tool_call_id: id.clone(),
                    status: Some(ToolCallStatus::Pending),
                    content: None,
                    title: Some(format_tool_title(name, arguments)),
                    kind: Some(infer_tool_kind(name)),
                    locations: None,
                    raw_input: Some(serde_json::Value::String(arguments.to_string())),
                    raw_output: None,
                    meta: None,
                }))
            }

            AgentEvent::ToolCallEnd {
                id,
                name: _,
                result,
            } => {
                self.tool_arguments.remove(id);
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    tool_call_id: id.clone(),
                    status: Some(tool_result_status(result)),
                    content: None,
                    title: None,
                    kind: None,
                    locations: None,
                    raw_input: None,
                    raw_output: Some(tool_result_raw_output(result)),
                    meta: None,
                }))
            }

            AgentEvent::RunStart { .. }
            | AgentEvent::Checkpoint { .. }
            | AgentEvent::BackgroundTaskStarted { .. }
            | AgentEvent::BackgroundTaskCompleted { .. }
            | AgentEvent::TerminalInputNeeded { .. }
            | AgentEvent::RunPaused { .. }
            | AgentEvent::RunResumed { .. }
            | AgentEvent::SkillActivated { .. }
            | AgentEvent::TurnComplete { .. }
            | AgentEvent::Complete { .. }
            | AgentEvent::Error(_)
            | AgentEvent::Usage(_) => None,
        }
    }
}

const fn tool_result_status(result: &ToolResult) -> ToolCallStatus {
    if result.is_error() {
        ToolCallStatus::Failed
    } else {
        ToolCallStatus::Completed
    }
}

fn tool_result_raw_output(result: &ToolResult) -> serde_json::Value {
    match serde_json::to_value(result) {
        Ok(value) => value,
        Err(error) => {
            panic!("failed to serialize ToolResult for ACP raw_output: {error}");
        }
    }
}

/// Format a human-readable title for a tool call.
fn format_tool_title(name: &str, arguments: &str) -> String {
    // Try to extract relevant info from arguments for better titles
    match name {
        "terminal" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments)
                && let Some(cmd) = args.get("command").and_then(|v| v.as_str())
            {
                // Truncate long commands
                let truncated = if cmd.len() > 50 {
                    format!("{}...", &cmd[..47])
                } else {
                    cmd.to_string()
                };
                return format!("Running: {truncated}");
            }
            "Running command".to_string()
        }
        "read" | "Read" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments)
                && let Some(path) = args.get("file_path").and_then(|v| v.as_str())
            {
                return format!("Reading {path}");
            }
            "Reading file".to_string()
        }
        "write" | "Write" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments)
                && let Some(path) = args.get("file_path").and_then(|v| v.as_str())
            {
                return format!("Writing {path}");
            }
            "Writing file".to_string()
        }
        "edit" | "Edit" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments)
                && let Some(path) = args.get("file_path").and_then(|v| v.as_str())
            {
                return format!("Editing {path}");
            }
            "Editing file".to_string()
        }
        "glob" | "Glob" => "Searching files".to_string(),
        "grep" | "Grep" => "Searching content".to_string(),
        "websearch" | "WebSearch" => "Searching web".to_string(),
        "webfetch" | "WebFetch" => "Fetching URL".to_string(),
        "todo" | "TodoWrite" => "Updating tasks".to_string(),
        _ => format!("Running {name}"),
    }
}

/// Infer the tool kind from the tool name.
fn infer_tool_kind(name: &str) -> ToolKind {
    match name.to_lowercase().as_str() {
        "read" | "glob" | "grep" | "webfetch" => ToolKind::Read,
        "write" | "edit" => ToolKind::Edit,
        "websearch" => ToolKind::Search,
        "terminal" | "command" => ToolKind::Execute,
        _ => ToolKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_event_conversion() {
        let event = AgentEvent::Text("Hello".to_string());
        let update = SessionUpdateMapper::default().map(&event).unwrap();

        if let SessionUpdate::AgentMessageChunk(chunk) = update {
            if let ContentBlock::Text(text) = chunk.content {
                assert_eq!(text.text, "Hello");
            } else {
                panic!("Expected text content");
            }
        } else {
            panic!("Expected AgentMessageChunk");
        }
    }

    #[test]
    fn test_tool_call_conversion() {
        let event = AgentEvent::ToolCallStart {
            id: "123".to_string(),
            name: "terminal".to_string(),
            arguments: r#"{"command": "ls -la"}"#.to_string(),
        };
        let update = SessionUpdateMapper::default().map(&event).unwrap();

        if let SessionUpdate::ToolCall(call) = update {
            assert_eq!(call.tool_call_id, "123");
            assert_eq!(call.title, "Running: ls -la");
            assert_eq!(call.kind, Some(ToolKind::Execute));
            assert_eq!(call.status, Some(ToolCallStatus::Pending));
        } else {
            panic!("Expected ToolCall");
        }
    }

    /// `ToolCallDelta` fragments are incremental; the mapper must accumulate
    /// them so every `ToolCallUpdate` still carries the arguments so far.
    #[test]
    fn test_tool_call_deltas_accumulate() {
        let mut mapper = SessionUpdateMapper::default();
        let first = AgentEvent::ToolCallDelta {
            id: "call-1".to_string(),
            name: "terminal".to_string(),
            arguments_fragment: r#"{"command""#.to_string(),
        };
        let second = AgentEvent::ToolCallDelta {
            id: "call-1".to_string(),
            name: "terminal".to_string(),
            arguments_fragment: r#":"ls -la"}"#.to_string(),
        };

        let _ = mapper.map(&first);
        let update = mapper.map(&second).unwrap();

        if let SessionUpdate::ToolCallUpdate(update) = update {
            assert_eq!(
                update.raw_input,
                Some(serde_json::Value::String(
                    r#"{"command":"ls -la"}"#.to_string()
                ))
            );
            assert_eq!(update.title.as_deref(), Some("Running: ls -la"));
        } else {
            panic!("Expected ToolCallUpdate");
        }
    }
}
