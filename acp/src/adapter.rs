//! Adapter for converting aither agent events to ACP session updates.

use aither_agent::{AgentEvent, TodoItem, TodoStatus};

use crate::protocol::{
    ContentBlock, ContentChunk, Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus, SessionUpdate,
    TextContent, ToolCall, ToolCallStatus, ToolCallUpdate, ToolKind,
};

/// Convert an `AgentEvent` to an ACP `SessionUpdate`.
///
/// Returns `None` for events that don't have a direct ACP mapping
/// (like `TurnComplete` and `Complete`, which are handled separately).
#[must_use]
pub fn agent_event_to_session_update(event: &AgentEvent) -> Option<SessionUpdate> {
    match event {
        AgentEvent::Text(text) => Some(SessionUpdate::AgentMessageChunk(ContentChunk {
            content: ContentBlock::Text(TextContent {
                text: text.clone(),
                annotations: None,
            }),
        })),

        AgentEvent::Reasoning(text) => Some(SessionUpdate::AgentThoughtChunk(ContentChunk {
            content: ContentBlock::Text(TextContent {
                text: text.clone(),
                annotations: None,
            }),
        })),

        AgentEvent::ToolCallStart {
            id,
            name,
            arguments,
        } => Some(SessionUpdate::ToolCall(ToolCall {
            tool_call_id: id.clone(),
            title: format_tool_title(name, arguments),
            kind: Some(infer_tool_kind(name)),
            status: Some(ToolCallStatus::Pending),
            content: vec![],
            locations: vec![],
            raw_input: serde_json::from_str(arguments).ok(),
            raw_output: None,
        })),

        AgentEvent::ToolCallEnd {
            id,
            name: _,
            result,
        } => {
            let (status, output) = match result {
                Ok(output) => (ToolCallStatus::Completed, Some(output.clone())),
                Err(error) => (ToolCallStatus::Error, Some(error.clone())),
            };

            Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                tool_call_id: id.clone(),
                status: Some(status),
                content: None,
                title: None,
                kind: None,
                locations: None,
                raw_input: None,
                raw_output: output.map(serde_json::Value::String),
            }))
        }

        // These events are handled at a higher level
        AgentEvent::TurnComplete { .. } => None,
        AgentEvent::Complete { .. } => None,
        AgentEvent::Error(_) => None,
        AgentEvent::Usage(_) => None,
    }
}

/// Convert a todo list to an ACP Plan.
#[must_use]
pub fn todos_to_plan(todos: &[TodoItem]) -> Plan {
    Plan {
        entries: todos
            .iter()
            .map(|item| PlanEntry {
                content: item.content.clone(),
                status: todo_status_to_plan_status(item.status),
                priority: PlanEntryPriority::Medium,
            })
            .collect(),
    }
}

/// Convert `TodoStatus` to `PlanEntryStatus`.
#[must_use]
pub const fn todo_status_to_plan_status(status: TodoStatus) -> PlanEntryStatus {
    match status {
        TodoStatus::Pending => PlanEntryStatus::Pending,
        TodoStatus::InProgress => PlanEntryStatus::InProgress,
        TodoStatus::Completed => PlanEntryStatus::Completed,
    }
}

/// Format a human-readable title for a tool call.
fn format_tool_title(name: &str, arguments: &str) -> String {
    // Try to extract relevant info from arguments for better titles
    match name {
        "bash" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments) {
                if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                    // Truncate long commands
                    let truncated = if cmd.len() > 50 {
                        format!("{}...", &cmd[..47])
                    } else {
                        cmd.to_string()
                    };
                    return format!("Running: {truncated}");
                }
            }
            "Running command".to_string()
        }
        "read" | "Read" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments) {
                if let Some(path) = args.get("file_path").and_then(|v| v.as_str()) {
                    return format!("Reading {path}");
                }
            }
            "Reading file".to_string()
        }
        "write" | "Write" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments) {
                if let Some(path) = args.get("file_path").and_then(|v| v.as_str()) {
                    return format!("Writing {path}");
                }
            }
            "Writing file".to_string()
        }
        "edit" | "Edit" => {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(arguments) {
                if let Some(path) = args.get("file_path").and_then(|v| v.as_str()) {
                    return format!("Editing {path}");
                }
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
        "bash" | "command" => ToolKind::Execute,
        _ => ToolKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_event_conversion() {
        let event = AgentEvent::Text("Hello".to_string());
        let update = agent_event_to_session_update(&event).unwrap();

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
            name: "bash".to_string(),
            arguments: r#"{"command": "ls -la"}"#.to_string(),
        };
        let update = agent_event_to_session_update(&event).unwrap();

        if let SessionUpdate::ToolCall(call) = update {
            assert_eq!(call.tool_call_id, "123");
            assert_eq!(call.title, "Running: ls -la");
            assert_eq!(call.kind, Some(ToolKind::Execute));
            assert_eq!(call.status, Some(ToolCallStatus::Pending));
        } else {
            panic!("Expected ToolCall");
        }
    }

    #[test]
    fn test_todos_to_plan() {
        let todos = vec![
            TodoItem {
                content: "Task 1".to_string(),
                status: TodoStatus::Completed,
                active_form: "Completing task 1".to_string(),
            },
            TodoItem {
                content: "Task 2".to_string(),
                status: TodoStatus::InProgress,
                active_form: "Working on task 2".to_string(),
            },
        ];

        let plan = todos_to_plan(&todos);
        assert_eq!(plan.entries.len(), 2);
        assert_eq!(plan.entries[0].status, PlanEntryStatus::Completed);
        assert_eq!(plan.entries[1].status, PlanEntryStatus::InProgress);
    }
}
