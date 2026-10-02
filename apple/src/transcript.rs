//! `Message` conversation → native transcript preparation.
//!
//! The conversion is deliberately structural: roles map to transcript entry
//! kinds, and content is never relabeled or concatenated into a pseudo
//! conversation. Shapes the framework cannot express fail here, before any
//! native generation starts.

use std::collections::{HashMap, HashSet};

use aither_core::llm::{Attachment, Message};

use crate::error::AppleError;

/// A transcript entry whose attachments are not yet resolved.
#[derive(Debug)]
pub enum PreparedEntry {
    Prompt {
        text: String,
        attachments: Vec<Attachment>,
    },
    Response {
        text: String,
    },
    ToolCalls(Vec<PreparedToolCall>),
    ToolOutput {
        id: String,
        name: String,
        text: String,
    },
}

#[derive(Debug)]
pub struct PreparedToolCall {
    pub id: String,
    pub name: String,
    /// Arguments serialized as a JSON document.
    pub arguments: String,
}

/// The result of [`prepare`]: instructions, history and the final prompt,
/// with attachments still carrying their source `Attachment`.
#[derive(Debug)]
pub struct Prepared {
    /// One string per leading system message.
    pub instructions: Vec<String>,
    pub history: Vec<PreparedEntry>,
    /// The final prompt's text (empty when the turn continues from tool
    /// outputs).
    pub prompt_text: String,
    pub prompt_attachments: Vec<Attachment>,
}

fn unsupported(reason: impl Into<String>) -> AppleError {
    AppleError::UnsupportedRequest(reason.into())
}

/// Converts `messages` into a native transcript plan.
///
/// # Validation
///
/// * At least one message is required.
/// * System messages are collected into one instructions entry and must lead
///   the conversation; a system message anywhere else is rejected.
/// * Assistant turns become a `response` entry plus a `toolCalls` entry.
/// * Tool outputs must reference a call id issued by an earlier assistant
///   turn, and are rejected otherwise — replaying a dangling output would
///   corrupt the transcript.
/// * Reasoning state from other providers cannot be replayed natively and is
///   rejected rather than dropped.
/// * The last message must be `User` (becomes the generated-from prompt) or
///   `Tool` (the session resumes after tool outputs with an empty prompt).
pub fn prepare(messages: &[Message]) -> Result<Prepared, AppleError> {
    if messages.is_empty() {
        return Err(unsupported("request contains no messages"));
    }

    let mut instructions = Vec::new();
    let mut entries: Vec<PreparedEntry> = Vec::new();
    // Tool call id -> tool name, for reconstructing `ToolOutput` names.
    let mut call_names: HashMap<String, String> = HashMap::new();
    let mut seen_ids = HashSet::new();
    let mut seen_content = false;

    for message in messages {
        if !matches!(message, Message::Tool { .. }) && !call_names.is_empty() {
            return Err(unsupported(
                "tool batch is missing outputs before the next message",
            ));
        }
        match message {
            Message::System { content } => {
                if seen_content {
                    return Err(unsupported(
                        "system message after the start of the conversation; the native \
                         transcript admits instructions only as the first entry",
                    ));
                }
                instructions.push(content.clone());
            }
            Message::User {
                content,
                attachments,
            } => {
                seen_content = true;
                entries.push(PreparedEntry::Prompt {
                    text: content.clone(),
                    attachments: attachments.clone(),
                });
            }
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
            } => {
                seen_content = true;
                if !reasoning.is_empty() {
                    return Err(unsupported(
                        "assistant reasoning state cannot be replayed to the on-device \
                         model; strip reasoning state when switching providers",
                    ));
                }
                if !content.is_empty() || tool_calls.is_empty() {
                    entries.push(PreparedEntry::Response {
                        text: content.clone(),
                    });
                }
                if !tool_calls.is_empty() {
                    let calls = prepare_calls(tool_calls, &mut call_names, &mut seen_ids)?;
                    entries.push(PreparedEntry::ToolCalls(calls));
                }
            }
            Message::Tool {
                content,
                tool_call_id,
            } => {
                seen_content = true;
                let Some(name) = call_names.remove(tool_call_id) else {
                    return Err(unsupported(format!(
                        "tool output '{tool_call_id}' has no matching tool call in the \
                         conversation"
                    )));
                };
                entries.push(PreparedEntry::ToolOutput {
                    id: tool_call_id.clone(),
                    name: name.clone(),
                    text: content.clone(),
                });
            }
        }
    }

    if !call_names.is_empty() {
        return Err(unsupported("tool batch is missing outputs"));
    }

    let (prompt_text, prompt_attachments) = take_prompt(messages, &mut entries)?;

    Ok(Prepared {
        instructions,
        history: entries,
        prompt_text,
        prompt_attachments,
    })
}

fn prepare_calls(
    tool_calls: &[aither_core::llm::ToolCall],
    call_names: &mut HashMap<String, String>,
    seen_ids: &mut HashSet<String>,
) -> Result<Vec<PreparedToolCall>, AppleError> {
    tool_calls
        .iter()
        .map(|call| {
            if call.reasoning_state.is_some() {
                return Err(unsupported("tool call reasoning state cannot be replayed"));
            }
            if !seen_ids.insert(call.id.clone()) {
                return Err(unsupported(format!("duplicate tool call id {}", call.id)));
            }
            call_names.insert(call.id.clone(), call.name.clone());
            Ok(PreparedToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            })
        })
        .collect::<Result<Vec<_>, AppleError>>()
}

fn take_prompt(
    messages: &[Message],
    entries: &mut Vec<PreparedEntry>,
) -> Result<(String, Vec<Attachment>), AppleError> {
    // The final message determines what we generate from.
    let prompt = match messages.last() {
        Some(Message::User { .. }) => {
            let Some(PreparedEntry::Prompt { text, attachments }) = entries.pop() else {
                unreachable!("last message is a user prompt")
            };
            (text, attachments)
        }
        Some(Message::Tool { .. }) => {
            // Resuming after tool outputs: the native session continues from
            // the transcript with an empty prompt.
            (String::new(), Vec::new())
        }
        Some(other) => {
            return Err(unsupported(format!(
                "last message has role {:?}; the model can only generate from a \
                 user prompt or tool outputs",
                other.role()
            )));
        }
        None => unreachable!("non-empty messages"),
    };

    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::{Message, ToolCall};

    fn prepare_err(messages: &[Message]) -> String {
        match prepare(messages) {
            Err(AppleError::UnsupportedRequest(reason)) => reason,
            other => panic!("expected UnsupportedRequest, got {other:?}"),
        }
    }

    fn call_message(ids: &[&str]) -> Message {
        Message::Assistant {
            content: String::new(),
            reasoning: Vec::new(),
            tool_calls: ids
                .iter()
                .map(|id| ToolCall {
                    id: (*id).into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({}),
                    reasoning_state: None,
                })
                .collect(),
        }
    }

    #[test]
    fn rejects_duplicate_and_incomplete_batches() {
        for messages in [
            vec![call_message(&["a", "a"]), Message::tool("a", "ok")],
            vec![call_message(&["a", "b"]), Message::tool("a", "ok")],
            vec![
                call_message(&["a"]),
                Message::tool("a", "ok"),
                Message::tool("a", "twice"),
            ],
            vec![call_message(&["a"]), Message::user("interrupt")],
        ] {
            assert!(prepare(&messages).is_err());
        }
        let prepared = prepare(&[Message::assistant(""), Message::user("continue")]).unwrap();
        assert!(
            matches!(&prepared.history[0], PreparedEntry::Response { text } if text.is_empty())
        );
    }

    #[test]
    fn empty_conversation_rejected() {
        assert_eq!(prepare_err(&[]), "request contains no messages");
    }

    #[test]
    fn system_user_splits_instructions_and_prompt() {
        let prepared = prepare(&[Message::system("be terse"), Message::user("hello")])
            .expect("valid conversation");
        assert_eq!(prepared.instructions, ["be terse"]);
        assert_eq!(prepared.prompt_text, "hello");
        assert!(prepared.history.is_empty());
    }

    #[test]
    fn mid_conversation_system_rejected() {
        let err = prepare_err(&[Message::user("a"), Message::system("late")]);
        assert!(err.contains("system message"), "{err}");
    }

    #[test]
    fn assistant_then_tool_output_preserves_ids() {
        let prepared = prepare(&[
            Message::user("weather?"),
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "weather".into(),
                    arguments: serde_json::json!({"city": "sf"}),
                    reasoning_state: None,
                }],
                reasoning: Vec::new(),
            },
            Message::tool("call_1", "sunny"),
            Message::user("thanks"),
        ])
        .expect("valid conversation");
        assert_eq!(prepared.history.len(), 3);
        let last = &prepared.history[2];
        match last {
            PreparedEntry::ToolOutput { id, name, text } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "weather");
                assert_eq!(text, "sunny");
            }
            _ => panic!("expected tool output"),
        }
        assert_eq!(prepared.prompt_text, "thanks");
    }

    #[test]
    fn dangling_tool_output_rejected() {
        let err = prepare_err(&[Message::user("hi"), Message::tool("missing_call", "data")]);
        assert!(err.contains("no matching tool call"), "{err}");
    }

    #[test]
    fn assistant_last_message_rejected() {
        let err = prepare_err(&[Message::user("hi"), Message::assistant("draft")]);
        assert!(err.contains("last message"), "{err}");
    }

    #[test]
    fn tool_outputs_as_last_continue_with_empty_prompt() {
        let prepared = prepare(&[
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_9".into(),
                    name: "ping".into(),
                    arguments: serde_json::json!({}),
                    reasoning_state: None,
                }],
                reasoning: Vec::new(),
            },
            Message::tool("call_9", "pong"),
        ])
        .expect("tool continuation");
        assert_eq!(prepared.prompt_text, "");
        assert!(matches!(
            prepared.history.last(),
            Some(PreparedEntry::ToolOutput { id, name, .. }) if id == "call_9" && name == "ping"
        ));
    }
}
