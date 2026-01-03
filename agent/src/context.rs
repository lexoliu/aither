//! Conversation context management.
//!
//! This module provides the `ConversationMemory` type for managing
//! conversation history, including both compressed summaries and
//! recent messages.

use aither_core::llm::Message;

/// Conversation memory that keeps both long-term summaries and recent messages.
///
/// The memory is structured in two parts:
/// - **Summaries**: Compressed context from earlier in the conversation
/// - **Recent**: Verbatim recent messages
///
/// This structure allows for efficient context management while preserving
/// important historical information.
#[derive(Debug, Clone, Default)]
pub struct ConversationMemory {
    /// Compressed summaries of earlier conversation.
    summaries: Vec<Message>,
    /// Recent messages kept verbatim.
    recent: Vec<Message>,
}

impl ConversationMemory {
    /// Creates a new empty conversation memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new message to the recent conversation history.
    pub fn push(&mut self, message: Message) {
        self.recent.push(message);
    }

    /// Extends the recent conversation history with multiple messages.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        for message in messages {
            self.push(message);
        }
    }

    /// Adds a summary message to the long-term summaries.
    pub fn push_summary(&mut self, summary: Message) {
        self.summaries.push(summary);
    }

    /// Returns the number of recent messages stored.
    #[must_use]
    pub const fn len_recent(&self) -> usize {
        self.recent.len()
    }

    /// Returns the number of summary messages stored.
    #[must_use]
    pub const fn len_summaries(&self) -> usize {
        self.summaries.len()
    }

    /// Returns the total number of messages (summaries + recent).
    #[must_use]
    pub fn len(&self) -> usize {
        self.summaries.len() + self.recent.len()
    }

    /// Returns all messages, combining summaries and recent messages.
    ///
    /// Summaries come first, followed by recent messages.
    #[must_use]
    pub fn all(&self) -> Vec<Message> {
        self.summaries
            .iter()
            .cloned()
            .chain(self.recent.iter().cloned())
            .collect()
    }

    /// Returns only the recent messages.
    #[must_use]
    pub fn recent(&self) -> &[Message] {
        &self.recent
    }

    /// Returns only the summary messages.
    #[must_use]
    pub fn summaries(&self) -> &[Message] {
        &self.summaries
    }

    /// Returns the last message, prioritizing recent over summaries.
    #[must_use]
    pub fn last(&self) -> Option<&Message> {
        self.recent.last().or_else(|| self.summaries.last())
    }

    /// Checks if the memory is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.summaries.is_empty() && self.recent.is_empty()
    }

    /// Drains the oldest messages from recent history.
    ///
    /// Keeps only the specified number of most recent messages.
    /// Returns the drained (older) messages.
    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        if keep >= self.recent.len() {
            return Vec::new();
        }
        self.recent.drain(..self.recent.len() - keep).collect()
    }

    /// Clears all messages from memory.
    pub fn clear(&mut self) {
        self.summaries.clear();
        self.recent.clear();
    }

    /// Creates a fork (clone) of this memory.
    ///
    /// Useful for creating branches in conversation history.
    #[must_use]
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Creates a checkpoint that can be restored later.
    #[must_use]
    pub fn checkpoint(&self) -> MemoryCheckpoint {
        MemoryCheckpoint {
            summaries: self.summaries.clone(),
            recent: self.recent.clone(),
        }
    }

    /// Restores from a checkpoint.
    pub fn restore(&mut self, checkpoint: MemoryCheckpoint) {
        self.summaries = checkpoint.summaries;
        self.recent = checkpoint.recent;
    }
}

/// A snapshot of conversation memory that can be restored.
#[derive(Debug, Clone)]
pub struct MemoryCheckpoint {
    summaries: Vec<Message>,
    recent: Vec<Message>,
}

impl MemoryCheckpoint {
    /// Returns the total number of messages in this checkpoint.
    #[must_use]
    pub fn len(&self) -> usize {
        self.summaries.len() + self.recent.len()
    }

    /// Returns `true` if this checkpoint is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.summaries.is_empty() && self.recent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_all() {
        let mut memory = ConversationMemory::new();
        memory.push(Message::user("Hello"));
        memory.push(Message::assistant("Hi there!"));

        let messages = memory.all();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_push_summary() {
        let mut memory = ConversationMemory::new();
        memory.push_summary(Message::system("Summary of earlier conversation."));
        memory.push(Message::user("New message"));

        let messages = memory.all();
        assert_eq!(messages.len(), 2);
        // Summary should come first
        assert!(messages[0].content().contains("Summary"));
    }

    #[test]
    fn test_drain_oldest() {
        let mut memory = ConversationMemory::new();
        for i in 0..10 {
            memory.push(Message::user(format!("Message {i}")));
        }

        let drained = memory.drain_oldest(3);
        assert_eq!(drained.len(), 7);
        assert_eq!(memory.len_recent(), 3);
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut memory = ConversationMemory::new();
        memory.push(Message::user("Hello"));
        memory.push(Message::assistant("Hi!"));

        let checkpoint = memory.checkpoint();

        memory.push(Message::user("More messages"));
        memory.push(Message::user("Even more"));

        assert_eq!(memory.len(), 4);

        memory.restore(checkpoint);
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn test_fork() {
        let mut memory = ConversationMemory::new();
        memory.push(Message::user("Hello"));

        let fork = memory.fork();
        memory.push(Message::assistant("Hi!"));

        assert_eq!(memory.len(), 2);
        assert_eq!(fork.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut memory = ConversationMemory::new();
        memory.push_summary(Message::system("Summary"));
        memory.push(Message::user("Hello"));

        memory.clear();
        assert!(memory.is_empty());
    }
}
