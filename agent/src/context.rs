//! Unified context manager for agent conversation state.
//!
//! The [`Context`] struct owns the entire context window state: system blocks
//! (stable cacheable prefix) and recent conversation messages (including
//! reminders and handoff summaries).
//!
//! # Design principles
//!
//! - **No custom trait**: only `T: Serialize` is required for `insert_system`.
//! - **Immediate serialization**: `insert_system` serializes `T` to XML and stores the string.
//!   No trait objects, no `Arc`, no `dyn`.
//! - **Identity = snake_case struct name**: derived from `std::any::type_name::<T>()`.
//! - **Explicit cache lifecycle**: system blocks form a stable cacheable prefix;
//!   everything else (conversation, reminders, handoff) lives in `recent`.
//!
//! # Message layout
//!
//! ```text
//! [System: <base_system>…<persona>…<memory>…]  ← system_blocks (CACHEABLE prefix)
//! [Messages: handoff, reminders, user/assistant/tool interleaved]  ← recent
//! ```

use indexmap::IndexMap;
use serde::{Deserialize, Serialize, ser::SerializeMap};

use aither_core::llm::Message;

/// The entire context window state. Fully serializable for session persistence.
///
/// `system_blocks` is an [`IndexMap`] to preserve insertion order, producing a
/// deterministic rendering and stable cache prefix.
///
/// `recent` contains all conversation messages, including system reminders
/// and handoff summaries. These are interleaved naturally with the conversation.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Context {
    /// Persistent system blocks, keyed by snake_case type name.
    /// Survive compaction. Concatenated into one system message (cacheable prefix).
    system_blocks: IndexMap<String, String>,

    /// All conversation messages: user, assistant, tool, system reminders, handoff.
    /// Everything that is NOT part of the stable system prefix lives here.
    recent: Vec<Message>,
}

// Custom Serialize so we can emit system_blocks as a map.
impl Serialize for Context {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("system_blocks", &self.system_blocks)?;
        map.serialize_entry("recent", &self.recent)?;
        map.end()
    }
}

impl Context {
    /// Creates a new empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── System blocks (stable, cacheable prefix) ──────────────────────

    /// Insert or replace a system block.
    ///
    /// Identity is derived from the snake_case of `T`'s struct name.
    /// The value is serialized to XML immediately.
    ///
    /// Replacing a block whose content has changed invalidates the cached
    /// prefix — this is expected and intentional (the content actually changed).
    pub fn insert_system<T: Serialize>(&mut self, value: &T) {
        let tag = snake_case_type_name::<T>();
        let xml = serialize_xml(&tag, value);
        self.system_blocks.insert(tag, xml);
    }

    /// Insert or replace a system block with an explicit tag name.
    ///
    /// Use this when you need a tag that differs from the struct name.
    pub fn insert_system_named(&mut self, tag: impl Into<String>, content: impl Into<String>) {
        let tag = tag.into();
        let content = content.into();
        let xml = format!("<{tag}>\n{content}\n</{tag}>");
        self.system_blocks.insert(tag, xml);
    }

    /// Remove a system block by type.
    pub fn remove_system<T>(&mut self) {
        let tag = snake_case_type_name::<T>();
        self.system_blocks.shift_remove(&tag);
    }

    /// Remove a system block by tag name.
    pub fn remove_system_named(&mut self, tag: &str) {
        self.system_blocks.shift_remove(tag);
    }

    /// Returns the number of system blocks.
    #[must_use]
    pub fn system_block_count(&self) -> usize {
        self.system_blocks.len()
    }

    /// Returns `true` if a system block with the given tag exists.
    #[must_use]
    pub fn has_system_block(&self, tag: &str) -> bool {
        self.system_blocks.contains_key(tag)
    }

    /// Returns a mutable reference to the system blocks map.
    ///
    /// Use this for bulk operations on system blocks (e.g., during session restore).
    #[must_use]
    pub fn system_blocks_mut(&mut self) -> &mut IndexMap<String, String> {
        &mut self.system_blocks
    }

    /// Returns a reference to the system blocks map.
    #[must_use]
    pub fn system_blocks(&self) -> &IndexMap<String, String> {
        &self.system_blocks
    }

    // ── Conversation messages (recent) ────────────────────────────────

    /// Append a message to the conversation.
    ///
    /// This includes all message types: user, assistant, tool results,
    /// system reminders, and handoff summaries.
    pub fn push(&mut self, message: Message) {
        self.recent.push(message);
    }

    /// Extend the conversation with multiple messages.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.recent.extend(messages);
    }

    /// Returns the number of recent messages.
    #[must_use]
    pub fn len_recent(&self) -> usize {
        self.recent.len()
    }

    /// Returns the recent messages.
    #[must_use]
    pub fn recent(&self) -> &[Message] {
        &self.recent
    }

    /// Returns a mutable reference to the recent messages.
    #[must_use]
    pub fn recent_mut(&mut self) -> &mut Vec<Message> {
        &mut self.recent
    }

    /// Returns the last message.
    #[must_use]
    pub fn last(&self) -> Option<&Message> {
        self.recent.last()
    }

    /// Returns `true` if there are no conversation messages.
    #[must_use]
    pub fn is_conversation_empty(&self) -> bool {
        self.recent.is_empty()
    }

    /// Drains the oldest recent messages, keeping only the specified count.
    /// Returns the drained (older) messages.
    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        if keep >= self.recent.len() {
            return Vec::new();
        }
        self.recent.drain(..self.recent.len() - keep).collect()
    }

    /// Returns all conversation messages (alias for `recent()` as a `Vec`).
    ///
    /// This is provided for backward compatibility with code that called
    /// `ConversationMemory::all()`.
    #[must_use]
    pub fn conversation_messages(&self) -> Vec<Message> {
        self.recent.clone()
    }

    // ── Build messages for LLM request ────────────────────────────────

    /// Build the full message array for the LLM request.
    ///
    /// Layout:
    /// 1. System blocks → one system message (cacheable prefix)
    /// 2. Recent conversation messages (including reminders, handoff, etc.)
    #[must_use]
    pub fn build_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        // 1. System blocks → one system message (cacheable prefix)
        if !self.system_blocks.is_empty() {
            let system_xml: String = self
                .system_blocks
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(Message::system(system_xml));
        }

        // 2. Recent conversation (includes reminders, handoff, everything)
        messages.extend(self.recent.iter().cloned());

        messages
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /// Clear all conversation history (keep system blocks).
    pub fn clear_conversation(&mut self) {
        self.recent.clear();
    }

    /// Clear everything.
    pub fn clear_all(&mut self) {
        self.system_blocks.clear();
        self.recent.clear();
    }

    /// Creates a fork (clone) of this context.
    #[must_use]
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Creates a checkpoint that can be restored later.
    #[must_use]
    pub fn checkpoint(&self) -> ContextCheckpoint {
        ContextCheckpoint {
            recent: self.recent.clone(),
        }
    }

    /// Restores conversation state from a checkpoint.
    /// System blocks are NOT restored (they are managed separately).
    pub fn restore(&mut self, checkpoint: ContextCheckpoint) {
        self.recent = checkpoint.recent;
    }
}

/// A snapshot of conversation state that can be restored.
#[derive(Debug, Clone)]
pub struct ContextCheckpoint {
    recent: Vec<Message>,
}

impl ContextCheckpoint {
    /// Returns the total number of messages in this checkpoint.
    #[must_use]
    pub fn len(&self) -> usize {
        self.recent.len()
    }

    /// Returns `true` if this checkpoint is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.recent.is_empty()
    }
}

// ── Backward compatibility: ConversationMemory facade ─────────────────

/// Conversation memory (legacy facade).
///
/// This type is preserved for backward compatibility. New code should
/// use [`Context`] directly.
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
    pub const fn len(&self) -> usize {
        self.summaries.len() + self.recent.len()
    }

    /// Returns all messages, combining summaries and recent messages.
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
    pub const fn len(&self) -> usize {
        self.summaries.len() + self.recent.len()
    }

    /// Returns `true` if this checkpoint is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.summaries.is_empty() && self.recent.is_empty()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Extracts the short struct name from `std::any::type_name::<T>()` and converts
/// it to snake_case.
///
/// For example:
/// - `my_crate::Memory` → `"memory"`
/// - `my_crate::ProfilePersona` → `"profile_persona"`
/// - `my_crate::nested::BrowserContext` → `"browser_context"`
fn snake_case_type_name<T>() -> String {
    let full = std::any::type_name::<T>();
    let short = full.rsplit("::").next().unwrap_or(full);
    heck::AsSnakeCase(short).to_string()
}

/// Serializes a value to XML with the given root element tag.
///
/// Uses quick-xml's serde integration. The resulting XML looks like:
/// ```xml
/// <tag>
///   <field1>value1</field1>
///   <field2>value2</field2>
/// </tag>
/// ```
///
/// For structs with `#[serde(rename = "$text")]`, the content is inlined:
/// ```xml
/// <tag>content here</tag>
/// ```
fn serialize_xml<T: Serialize>(tag: &str, value: &T) -> String {
    let mut buf = String::new();
    let result = quick_xml::se::Serializer::with_root(&mut buf, Some(tag))
        .and_then(|ser| value.serialize(ser).map(|_| ()));
    match result {
        Ok(()) => buf,
        Err(e) => {
            tracing::warn!(tag, error = %e, "XML serialization failed, using fallback");
            format!("<{tag}>serialization error: {e}</{tag}>")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_case_type_name() {
        assert_eq!(snake_case_type_name::<Context>(), "context");
    }

    #[derive(Serialize)]
    struct Memory {
        #[serde(rename = "$text")]
        content: String,
    }

    #[derive(Serialize)]
    struct ProfilePersona {
        #[serde(rename = "$text")]
        content: String,
    }

    #[derive(Serialize)]
    struct Workspace {
        sandbox: String,
        cwd: String,
    }

    #[test]
    fn test_serialize_xml_text_content() {
        let m = Memory {
            content: "hello world".to_string(),
        };
        let xml = serialize_xml("memory", &m);
        assert!(xml.contains("<memory>"));
        assert!(xml.contains("hello world"));
        assert!(xml.contains("</memory>"));
    }

    #[test]
    fn test_serialize_xml_struct_fields() {
        let w = Workspace {
            sandbox: "/tmp/sandbox".to_string(),
            cwd: "/home/user".to_string(),
        };
        let xml = serialize_xml("workspace", &w);
        assert!(xml.contains("<workspace>"));
        assert!(xml.contains("<sandbox>"));
        assert!(xml.contains("/tmp/sandbox"));
        assert!(xml.contains("</workspace>"));
    }

    #[test]
    fn test_insert_system() {
        let mut ctx = Context::new();
        let m = Memory {
            content: "remember this".to_string(),
        };
        ctx.insert_system(&m);
        assert!(ctx.has_system_block("memory"));
        assert_eq!(ctx.system_block_count(), 1);
    }

    #[test]
    fn test_insert_system_replaces() {
        let mut ctx = Context::new();
        ctx.insert_system(&Memory {
            content: "old".to_string(),
        });
        ctx.insert_system(&Memory {
            content: "new".to_string(),
        });
        assert_eq!(ctx.system_block_count(), 1);

        let messages = ctx.build_messages();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].content().contains("new"));
        assert!(!messages[0].content().contains("old"));
    }

    #[test]
    fn test_build_messages_layout() {
        let mut ctx = Context::new();

        // System block
        ctx.insert_system(&Memory {
            content: "system".to_string(),
        });

        // Conversation (reminders and handoff are just pushed as messages)
        ctx.push(Message::system("handoff summary"));
        ctx.push(Message::user("hello"));
        ctx.push(Message::assistant("hi"));

        let messages = ctx.build_messages();
        assert_eq!(messages.len(), 4);

        // 1. System blocks (one message)
        assert!(messages[0].content().contains("system"));
        // 2-4. Conversation (handoff + user + assistant)
        assert!(messages[1].content().contains("handoff summary"));
        assert!(messages[2].content().contains("hello"));
        assert!(messages[3].content().contains("hi"));
    }

    #[test]
    fn test_clear_conversation() {
        let mut ctx = Context::new();
        ctx.insert_system(&Memory {
            content: "persist".to_string(),
        });
        ctx.push(Message::user("hello"));
        ctx.push(Message::system("reminder"));

        ctx.clear_conversation();

        assert_eq!(ctx.len_recent(), 0);
        // System blocks survive
        assert_eq!(ctx.system_block_count(), 1);
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut ctx = Context::new();
        ctx.push(Message::user("hello"));
        ctx.push(Message::assistant("hi"));

        let cp = ctx.checkpoint();

        ctx.push(Message::user("more"));
        assert_eq!(ctx.len_recent(), 3);

        ctx.restore(cp);
        assert_eq!(ctx.len_recent(), 2);
    }

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
    fn test_memory_checkpoint_restore() {
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
