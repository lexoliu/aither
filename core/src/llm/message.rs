//! Message types for AI language model conversations.
//!
//! This module provides types for representing messages in conversations with AI language models.
//! Messages are represented as an enum with variants for different roles (User, Assistant, System, Tool).

use alloc::{string::String, vec::Vec};
use mime::Mime;
use url::Url;

use super::event::ToolCall;
use super::reasoning::ReasoningState;

/// A typed media attachment supplied with a user message.
///
/// Keeping the MIME type beside the URL lets each provider choose the correct
/// protocol content block without guessing from a provider-generated URL.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Attachment {
    url: Url,
    #[cfg_attr(feature = "serde", serde(with = "mime_serde"))]
    media_type: Mime,
}

impl Attachment {
    /// Creates an attachment with an explicit MIME type.
    #[must_use]
    pub const fn new(url: Url, media_type: Mime) -> Self {
        Self { url, media_type }
    }

    /// Returns the attachment URL.
    #[must_use]
    pub const fn url(&self) -> &Url {
        &self.url
    }

    /// Returns the declared MIME type.
    #[must_use]
    pub const fn media_type(&self) -> &Mime {
        &self.media_type
    }

    /// Replaces the URL while preserving the declared MIME type.
    #[must_use]
    pub fn with_url(self, url: Url) -> Self {
        Self {
            url,
            media_type: self.media_type,
        }
    }

    /// Splits the attachment into its URL and MIME type.
    #[must_use]
    pub fn into_parts(self) -> (Url, Mime) {
        (self.url, self.media_type)
    }
}

#[cfg(feature = "serde")]
mod mime_serde {
    use alloc::string::String;
    use core::str::FromStr;
    use mime::Mime;
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub fn serialize<S>(media_type: &Mime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(media_type.as_ref())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Mime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Mime::from_str(&raw).map_err(D::Error::custom)
    }
}

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
/// - Tool: content with required `tool_call_id`
#[derive(Debug, Clone, PartialEq, Eq)]
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
        attachments: Vec<Attachment>,
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
        /// Opaque reasoning state produced with this turn.
        ///
        /// Replayed to the provider on the next request. Order is significant —
        /// Anthropic rejects a turn whose thinking blocks were reordered or
        /// partially dropped — so append rather than rebuild.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Vec::is_empty")
        )]
        reasoning: Vec<ReasoningState>,
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

    /// Returns the typed attachments (only for User messages).
    #[must_use]
    pub fn attachments(&self) -> &[Attachment] {
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
            reasoning: Vec::new(),
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
            reasoning: Vec::new(),
        }
    }

    /// Creates an assistant message carrying the reasoning state of its turn.
    ///
    /// Use this when rebuilding a conversation for another request: without the
    /// reasoning, providers that verify their own thinking see a turn that has
    /// lost the reasoning behind its tool calls.
    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
        reasoning: Vec<ReasoningState>,
    ) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls,
            reasoning,
        }
    }

    /// Reasoning state recorded on this message, if any.
    ///
    /// Only assistant turns carry reasoning; every other role returns empty.
    #[must_use]
    pub fn reasoning(&self) -> &[ReasoningState] {
        match self {
            Self::Assistant { reasoning, .. } => reasoning,
            _ => &[],
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

    /// Adds a typed attachment to the message (only works for User messages).
    #[must_use]
    pub fn with_attachment(mut self, attachment: Attachment) -> Self {
        if let Self::User { attachments, .. } = &mut self {
            attachments.push(attachment);
        }
        self
    }

    /// Adds multiple typed attachments to the message.
    #[must_use]
    pub fn with_attachments(mut self, values: impl IntoIterator<Item = Attachment>) -> Self {
        if let Self::User { attachments, .. } = &mut self {
            attachments.extend(values);
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

        let msg = Message::assistant_with_tool_calls("", tool_calls);
        assert_eq!(msg.tool_calls().len(), 1);
        assert_eq!(msg.tool_calls()[0].name, "get_weather");
    }

    #[test]
    fn message_with_attachment() {
        let attachment = Attachment::new(
            "https://example.com/image.png".parse::<Url>().unwrap(),
            mime::IMAGE_PNG,
        );
        let message = Message::user("Hello").with_attachment(attachment.clone());
        assert_eq!(message.attachments(), &[attachment]);
    }

    #[test]
    fn message_with_attachments() {
        let attachments = vec![
            Attachment::new(
                "https://example.com/a.png".parse::<Url>().unwrap(),
                mime::IMAGE_PNG,
            ),
            Attachment::new(
                "https://example.com/b.pdf".parse::<Url>().unwrap(),
                mime::APPLICATION_PDF,
            ),
        ];
        let message = Message::user("Hello").with_attachments(attachments.clone());
        assert_eq!(message.attachments(), attachments.as_slice());
    }

    #[test]
    fn attachments_are_ignored_for_non_user_messages() {
        let attachment = Attachment::new(
            "https://example.com/a.png".parse::<Url>().unwrap(),
            mime::IMAGE_PNG,
        );
        let message = Message::assistant("Hello").with_attachment(attachment);
        assert!(message.attachments().is_empty());
    }

    #[test]
    fn message_clone() {
        let original = Message::user("Original");
        let cloned = original.clone();
        assert_eq!(original.content(), cloned.content());
    }
}
