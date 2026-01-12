//! Message types for AI language model conversations.
//!
//! This module provides types for representing messages in conversations with AI language models.
//! Messages are represented as an enum with variants for different roles (User, Assistant, System, Tool).

use core::fmt::Debug;

use alloc::{string::String, vec::Vec};
use url::Url;

use super::event::ToolCall;

/// Conversation participant role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Role {
    /// User message - input from human user.
    User,
    /// AI assistant message - responses from the AI.
    Assistant,
    /// System message - context/instructions for the AI.
    System,
    /// Tool message - output from tool/function calls.
    Tool,
}

/// A message in a conversation.
///
/// Different message types have different fields:
/// - User/System: content with optional attachments
/// - Assistant: content with optional tool calls
/// - Tool: content with required tool_call_id
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "role", rename_all = "snake_case"))]
pub enum Message {
    /// User message with content and optional attachments.
    User {
        /// Text content of the message.
        content: String,
        /// Attachment URLs (images, documents, etc.)
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Vec::is_empty")
        )]
        attachments: Vec<Url>,
    },
    /// Assistant message with content and optional tool calls.
    Assistant {
        /// Text content of the message.
        content: String,
        /// Tool calls made by the assistant.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Vec::is_empty")
        )]
        tool_calls: Vec<ToolCall>,
    },
    /// System message with instructions/context.
    System {
        /// Text content of the message.
        content: String,
    },
    /// Tool result message.
    Tool {
        /// Result content from the tool.
        content: String,
        /// ID of the tool call this is responding to.
        tool_call_id: String,
    },
}

impl Message {
    /// Returns the message sender role.
    #[must_use]
    pub const fn role(&self) -> Role {
        match self {
            Self::User { .. } => Role::User,
            Self::Assistant { .. } => Role::Assistant,
            Self::System { .. } => Role::System,
            Self::Tool { .. } => Role::Tool,
        }
    }

    /// Returns the text content of the message.
    #[must_use]
    pub fn content(&self) -> &str {
        match self {
            Self::User { content, .. }
            | Self::Assistant { content, .. }
            | Self::System { content }
            | Self::Tool { content, .. } => content,
        }
    }

    /// Returns the attachment URLs (only for User messages).
    #[must_use]
    pub fn attachments(&self) -> &[Url] {
        match self {
            Self::User { attachments, .. } => attachments,
            _ => &[],
        }
    }

    /// Returns tool calls made by the assistant (only for Assistant messages).
    #[must_use]
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Self::Assistant { tool_calls, .. } => tool_calls,
            _ => &[],
        }
    }

    /// Returns the tool call ID (only for Tool messages).
    #[must_use]
    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Self::Tool { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        }
    }

    /// Creates a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            attachments: Vec::new(),
        }
    }

    /// Creates a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Creates an assistant message with tool calls.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls,
        }
    }

    /// Creates a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    /// Creates a new tool result message.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Adds an attachment URL to the message (only works for User messages).
    #[must_use]
    pub fn with_attachment<U: TryInto<Url, Error: Debug>>(mut self, url: U) -> Self {
        if let Self::User { attachments, .. } = &mut self {
            attachments.push(url.try_into().unwrap());
        }
        self
    }

    /// Adds multiple attachment URLs to the message.
    #[must_use]
    pub fn with_attachments<U: TryInto<Url, Error: Debug>>(
        mut self,
        urls: impl IntoIterator<Item = U>,
    ) -> Self {
        if let Self::User { attachments, .. } = &mut self {
            attachments.extend(urls.into_iter().map(|url| url.try_into().unwrap()));
        }
        self
    }

    /// Adds tool calls to the message (only works for Assistant messages).
    #[must_use]
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        if let Self::Assistant { tool_calls, .. } = &mut self {
            *tool_calls = calls;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_eq!(Role::Assistant, Role::Assistant);
        assert_eq!(Role::System, Role::System);
        assert_eq!(Role::Tool, Role::Tool);
        assert_ne!(Role::User, Role::Assistant);
    }

    #[test]
    fn message_creation() {
        let user = Message::user("Hello");
        assert_eq!(user.role(), Role::User);
        assert_eq!(user.content(), "Hello");

        let assistant = Message::assistant("Hi there!");
        assert_eq!(assistant.role(), Role::Assistant);
        assert_eq!(assistant.content(), "Hi there!");

        let system = Message::system("Be helpful");
        assert_eq!(system.role(), Role::System);
        assert_eq!(system.content(), "Be helpful");

        let tool = Message::tool("call_123", "Success");
        assert_eq!(tool.role(), Role::Tool);
        assert_eq!(tool.content(), "Success");
        assert_eq!(tool.tool_call_id(), Some("call_123"));
    }

    #[test]
    fn assistant_with_tool_calls() {
        let tool_calls = vec![ToolCall::new(
            "call_1",
            "get_weather",
            serde_json::json!({"city": "NYC"}),
        )];

        let msg = Message::assistant_with_tool_calls("", tool_calls.clone());
        assert_eq!(msg.tool_calls().len(), 1);
        assert_eq!(msg.tool_calls()[0].name, "get_weather");
    }

    #[test]
    fn message_with_attachment() {
        let url = "https://example.com".parse::<Url>().unwrap();
        let message = Message::user("Hello").with_attachment(url.clone());
        assert_eq!(message.attachments().len(), 1);
        assert_eq!(message.attachments()[0], url);
    }

    #[test]
    fn message_clone() {
        let original = Message::user("Original");
        let cloned = original.clone();
        assert_eq!(original.content(), cloned.content());
    }
}
