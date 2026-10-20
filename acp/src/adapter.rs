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

/// Maximum characters shown in a tool-call title before it is truncated.
const TOOL_TITLE_MAX_CHARS: usize = 50;

/// Truncate `text` to [`TOOL_TITLE_MAX_CHARS`] characters on a char boundary,
/// appending an ellipsis when anything was cut.
fn truncate_title(text: &str) -> String {
    match text.char_indices().nth(TOOL_TITLE_MAX_CHARS) {
        Some((byte_index, _)) => format!("{}...", &text[..byte_index]),
        None => text.to_string(),
    }
}

/// Format a human-readable title for a tool call.
///
/// Covers `terminal` and its control tools (`terminal_kill`, `terminal_input`,
/// `terminal_read`), the host-mediated `ask_user` and `request_workspace`
/// tools, and `subagent`. Every other capability is a CLI command executed
/// through `terminal` and gets the generic title.
fn format_tool_title(name: &str, arguments: &str) -> String {
    match name {
        "terminal" => {
            if let Ok(args) = serde_json::from_str::<aither_agent::sandbox::TerminalArgs>(arguments)
                && !args.description.is_empty()
            {
                // `description` is the UI-facing explanation the model writes
                // for the command; `script` is the shell payload.
                return truncate_title(&args.description);
            }
            "Running command".to_string()
        }
        "terminal_kill" => {
            if let Ok(args) =
                serde_json::from_str::<aither_agent::sandbox::builtin::KillTerminalArgs>(arguments)
            {
                return format!("Stopping task {}", args.task_id);
            }
            "Stopping task".to_string()
        }
        "terminal_input" => {
            if let Ok(args) =
                serde_json::from_str::<aither_agent::sandbox::builtin::InputTerminalArgs>(arguments)
            {
                return format!("Sending input to task {}", args.task_id);
            }
            "Sending input to task".to_string()
        }
        "terminal_read" => {
            if let Ok(args) = serde_json::from_str::<
                aither_agent::sandbox::builtin::ReadTerminalDeltaArgs,
            >(arguments)
            {
                return format!("Reading output of task {}", args.task_id);
            }
            "Reading task output".to_string()
        }
        "ask_user" => {
            if let Ok(args) = serde_json::from_str::<aither_agent::ask_user::AskUserArgs>(arguments)
                && let Some(question) = args
                    .into_questions()
                    .into_iter()
                    .map(|question| question.question)
                    .find(|question| !question.is_empty())
            {
                return format!("Asking: {}", truncate_title(&question));
            }
            "Asking user".to_string()
        }
        "request_workspace" => {
            if let Ok(args) = serde_json::from_str::<
                aither_agent::workspace_request::RequestWorkspaceArgs,
            >(arguments)
            {
                return format!("Requesting access to {}", truncate_title(&args.path));
            }
            "Requesting workspace access".to_string()
        }
        "subagent" => {
            if let Ok(args) =
                serde_json::from_str::<aither_agent::specialized::SubagentArgs>(arguments)
            {
                return format!("Running {} subagent", truncate_title(&args.subagent));
            }
            "Running subagent".to_string()
        }
        _ => format!("Running {name}"),
    }
}

/// Infer the tool kind from the tool name.
fn infer_tool_kind(name: &str) -> ToolKind {
    match name {
        "terminal" | "terminal_kill" | "terminal_input" => ToolKind::Execute,
        "terminal_read" => ToolKind::Read,
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
            arguments: r#"{"description":"List files","script":"ls -la","timeout":0}"#.to_string(),
        };
        let update = SessionUpdateMapper::default().map(&event).unwrap();

        if let SessionUpdate::ToolCall(call) = update {
            assert_eq!(call.tool_call_id, "123");
            assert_eq!(call.title, "List files");
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
            arguments_fragment: r#"{"description""#.to_string(),
        };
        let second = AgentEvent::ToolCallDelta {
            id: "call-1".to_string(),
            name: "terminal".to_string(),
            arguments_fragment: r#":"List files","script":"ls -la","timeout":0}"#.to_string(),
        };

        let _ = mapper.map(&first);
        let update = mapper.map(&second).unwrap();

        if let SessionUpdate::ToolCallUpdate(update) = update {
            assert_eq!(
                update.raw_input,
                Some(serde_json::Value::String(
                    r#"{"description":"List files","script":"ls -la","timeout":0}"#.to_string()
                ))
            );
            assert_eq!(update.title.as_deref(), Some("List files"));
        } else {
            panic!("Expected ToolCallUpdate");
        }
    }

    #[test]
    fn test_terminal_control_tool_titles() {
        let cases = [
            (
                "terminal_kill",
                r#"{"task_id":"task-7"}"#,
                "Stopping task task-7",
                ToolKind::Execute,
            ),
            (
                "terminal_input",
                r#"{"task_id":"task-7","input":"yes"}"#,
                "Sending input to task task-7",
                ToolKind::Execute,
            ),
            (
                "terminal_read",
                r#"{"task_id":"task-7"}"#,
                "Reading output of task task-7",
                ToolKind::Read,
            ),
        ];
        for (name, arguments, title, kind) in cases {
            assert_eq!(format_tool_title(name, arguments), title, "{name}");
            assert_eq!(infer_tool_kind(name), kind, "{name}");
        }
    }

    #[test]
    fn test_host_mediated_tool_titles() {
        let cases = [
            (
                "ask_user",
                r#"{"question":"Which database?","option":["PostgreSQL","SQLite"]}"#,
                "Asking: Which database?",
                ToolKind::Other,
            ),
            (
                "ask_user",
                r#"{"questions":[{"section":"db","question":"Which database?","option":[]}]}"#,
                "Asking: Which database?",
                ToolKind::Other,
            ),
            (
                "request_workspace",
                r#"{"path":"/home/user/project"}"#,
                "Requesting access to /home/user/project",
                ToolKind::Other,
            ),
            (
                "subagent",
                r#"{"subagent":"explore","prompt":"find the adapter code"}"#,
                "Running explore subagent",
                ToolKind::Other,
            ),
        ];
        for (name, arguments, title, kind) in cases {
            assert_eq!(format_tool_title(name, arguments), title, "{name}");
            assert_eq!(infer_tool_kind(name), kind, "{name}");
        }
    }

    /// Arguments that cannot be deserialized fall back to a generic title.
    #[test]
    fn test_tool_title_fallbacks() {
        assert_eq!(format_tool_title("terminal", "{}"), "Running command");
        assert_eq!(
            format_tool_title(
                "terminal",
                r#"{"description":"","script":"true","timeout":0}"#
            ),
            "Running command"
        );
        assert_eq!(format_tool_title("terminal", "not json"), "Running command");
        assert_eq!(format_tool_title("terminal_kill", "{}"), "Stopping task");
        assert_eq!(
            format_tool_title("terminal_input", "{}"),
            "Sending input to task"
        );
        assert_eq!(
            format_tool_title("terminal_read", "{}"),
            "Reading task output"
        );
        assert_eq!(format_tool_title("ask_user", "{}"), "Asking user");
        assert_eq!(
            format_tool_title("request_workspace", "{}"),
            "Requesting workspace access"
        );
        assert_eq!(format_tool_title("subagent", "{}"), "Running subagent");
        assert_eq!(
            format_tool_title("unknown_tool", "{}"),
            "Running unknown_tool"
        );
        assert_eq!(infer_tool_kind("unknown_tool"), ToolKind::Other);
    }

    /// Titles are cut on a char boundary: a multi-byte description longer than
    /// the limit must not panic on byte slicing.
    #[test]
    fn test_tool_title_truncates_on_char_boundary() {
        let description = "あ".repeat(TOOL_TITLE_MAX_CHARS + 10);
        let arguments = serde_json::json!({
            "description": description,
            "script": "true",
            "timeout": 0,
        })
        .to_string();
        let title = format_tool_title("terminal", &arguments);
        assert_eq!(title, format!("{}...", "あ".repeat(TOOL_TITLE_MAX_CHARS)));

        // A description that lands mid-character at the old byte cut (47)
        // must not panic.
        let boundary_case = serde_json::json!({
            "description": format!("{:46}é{}", "x", "y".repeat(20)),
            "script": "true",
            "timeout": 0,
        })
        .to_string();
        let title = format_tool_title("terminal", &boundary_case);
        assert!(title.contains('é'));
        assert_eq!(title.chars().count(), TOOL_TITLE_MAX_CHARS + 3);
        assert!(title.ends_with("..."));
    }
}
