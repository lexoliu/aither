//! Unified context manager for agent conversation state.
//!
//! The [`Context`] struct owns the entire context window state: persistent
//! system blocks (stable cacheable prefix), ephemeral reminders, compaction
//! handoff, and recent conversation messages.

use std::hash::{Hash, Hasher};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use aither_core::llm::{Message, Role};

use crate::handoff::HandoffDocument;

#[derive(Serialize)]
struct TextBlock<'a> {
    #[serde(rename = "$text")]
    content: &'a str,
}

/// The entire context window state. Fully serializable for persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Context {
    /// Persistent system blocks keyed by `snake_case` type name.
    ///
    /// These survive compaction and are rendered as a single cacheable system
    /// prefix message.
    system_blocks: IndexMap<String, String>,

    /// Staged replacements for system blocks, applied on the next
    /// [`reassemble`](Self::reassemble).
    ///
    /// Rewriting a system block mid-session invalidates the KV-cache prefix.
    /// Hosts stage the new content here (surfacing the change to the model as
    /// an appended diff alert instead) and let reassembly fold it in once the
    /// cache is already going to be invalidated.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    staged_system_updates: IndexMap<String, String>,

    /// Compaction handoff document from previous context compaction.
    ///
    /// This is rendered immediately after the stable system prefix.
    handoff: Option<String>,

    /// Ordered runtime items rendered after the optional handoff anchor.
    runtime_items: Vec<ContextCheckpointItem>,
}

/// Outcome of a [`Context::reassemble`] pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReassemblyReport {
    /// Ephemeral reminders removed.
    pub reminders_cleared: usize,
    /// Recent system messages (system alerts) removed.
    pub system_messages_pruned: usize,
    /// Tool results replaced with a tombstone because a later identical call
    /// holds the current result.
    pub tool_results_deduped: usize,
    /// Staged system block updates folded into the stable prefix.
    pub staged_updates_applied: usize,
    /// Estimated bytes freed from the assembled window.
    pub bytes_freed: usize,
}

impl ReassemblyReport {
    /// Whether the pass changed anything.
    #[must_use]
    pub const fn changed(&self) -> bool {
        self.reminders_cleared > 0
            || self.system_messages_pruned > 0
            || self.tool_results_deduped > 0
            || self.staged_updates_applied > 0
    }
}

/// Tombstone text left in place of a superseded tool result.
const SUPERSEDED_TOOL_RESULT: &str =
    "(superseded: an identical later call in this conversation holds the current result)";

/// Ordered runtime items stored after the fixed handoff anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextCheckpointItem {
    /// A reminder rendered as a system message.
    Reminder(String),
    /// A regular conversation message.
    Message(Message),
}

impl Context {
    /// Creates a new empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── System blocks (stable, cacheable prefix) ──────────────────────

    /// Inserts or replaces a persistent system block.
    ///
    /// Identity is derived from the `snake_case` type name of `T`.
    pub fn insert_system<T: Serialize>(&mut self, value: &T) {
        let tag = snake_case_type_name::<T>();
        let xml = serialize_xml(&tag, value);
        self.system_blocks.insert(tag, xml);
    }

    /// Inserts or replaces a persistent system block with an explicit tag.
    pub fn insert_system_named(&mut self, tag: impl Into<String>, content: impl Into<String>) {
        let tag = tag.into();
        let content = content.into();
        let xml = serialize_xml(&tag, &TextBlock { content: &content });
        self.system_blocks.insert(tag, xml);
    }

    /// Removes a persistent system block by type.
    pub fn remove_system<T>(&mut self) {
        let tag = snake_case_type_name::<T>();
        self.system_blocks.shift_remove(&tag);
    }

    /// Removes a persistent system block by explicit tag.
    pub fn remove_system_named(&mut self, tag: &str) {
        self.system_blocks.shift_remove(tag);
    }

    /// Returns the number of persistent system blocks.
    #[must_use]
    pub fn system_block_count(&self) -> usize {
        self.system_blocks.len()
    }

    /// Returns whether a persistent system block with `tag` exists.
    #[must_use]
    pub fn has_system_block(&self, tag: &str) -> bool {
        self.system_blocks.contains_key(tag)
    }

    /// Returns an immutable view of persistent system blocks.
    #[must_use]
    pub const fn system_blocks(&self) -> &IndexMap<String, String> {
        &self.system_blocks
    }

    /// Returns a mutable view of persistent system blocks.
    #[must_use]
    pub const fn system_blocks_mut(&mut self) -> &mut IndexMap<String, String> {
        &mut self.system_blocks
    }

    /// Inserts or replaces a persistent system block with raw text.
    ///
    /// Unlike [`insert_system`](Self::insert_system) and
    /// [`insert_system_named`](Self::insert_system_named), the `content` is
    /// stored verbatim without any XML wrapping. This is the preferred entry
    /// point for prose system blocks such as workspace descriptions, runtime
    /// metadata, and environment hints, where XML structure adds tokens
    /// without providing semantic value.
    ///
    /// Use [`insert_system`](Self::insert_system) only when the block is a
    /// structured payload that genuinely needs XML delimiters to separate
    /// context from instructions (for example, attached documents).
    pub fn insert_system_text(&mut self, tag: impl Into<String>, content: impl Into<String>) {
        self.system_blocks.insert(tag.into(), content.into());
    }

    /// Returns a stable fingerprint of the current system blocks in order.
    ///
    /// The fingerprint changes if and only if the ordered set of
    /// `(tag, content)` pairs changes. Hosts can use this to detect
    /// cache-invalidating modifications to the stable system prefix
    /// (for example, to track KV-cache hit rate over time).
    #[must_use]
    pub fn stable_prefix_fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (tag, content) in &self.system_blocks {
            tag.hash(&mut hasher);
            content.hash(&mut hasher);
        }
        hasher.finish()
    }

    // ── Reminders (ephemeral) ─────────────────────────────────────────

    /// Inserts an ephemeral reminder.
    pub fn insert_reminder<T: Serialize>(&mut self, value: &T) {
        let tag = snake_case_type_name::<T>();
        let xml = serialize_xml(&tag, value);
        self.runtime_items
            .push(ContextCheckpointItem::Reminder(xml));
    }

    /// Clears all ephemeral reminders.
    pub fn clear_reminders(&mut self) {
        self.runtime_items
            .retain(|item| !matches!(item, ContextCheckpointItem::Reminder(_)));
    }

    /// Returns current reminders.
    #[must_use]
    pub fn reminders(&self) -> Vec<&str> {
        self.runtime_items
            .iter()
            .filter_map(|item| match item {
                ContextCheckpointItem::Reminder(content) => Some(content.as_str()),
                ContextCheckpointItem::Message(_) => None,
            })
            .collect()
    }

    // ── Handoff ───────────────────────────────────────────────────────

    /// Sets or replaces the compaction handoff document.
    pub fn set_handoff(&mut self, handoff: impl Into<String>) {
        let handoff = handoff.into();
        self.handoff = Some(serialize_xml("handoff", &TextBlock { content: &handoff }));
    }

    /// Sets the compaction handoff document from a structured handoff artifact.
    pub fn set_handoff_document(&mut self, handoff: &HandoffDocument) {
        self.handoff = Some(serialize_xml("handoff", handoff));
    }

    /// Clears the compaction handoff document.
    pub fn clear_handoff(&mut self) {
        self.handoff = None;
    }

    /// Returns the current compaction handoff document.
    #[must_use]
    pub fn handoff(&self) -> Option<&str> {
        self.handoff.as_deref()
    }

    // ── Recent conversation ───────────────────────────────────────────

    /// Appends a message to recent conversation.
    pub fn push(&mut self, message: Message) {
        self.runtime_items
            .push(ContextCheckpointItem::Message(message));
    }

    /// Extends recent conversation with multiple messages.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.runtime_items
            .extend(messages.into_iter().map(ContextCheckpointItem::Message));
    }

    /// Returns number of recent conversation messages.
    #[must_use]
    pub fn len_recent(&self) -> usize {
        self.runtime_items
            .iter()
            .filter(|item| matches!(item, ContextCheckpointItem::Message(_)))
            .count()
    }

    /// Returns the number of recent system messages.
    #[must_use]
    pub fn count_recent_system_messages(&self) -> usize {
        self.runtime_items
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    ContextCheckpointItem::Message(message) if message.role() == Role::System
                )
            })
            .count()
    }

    /// Returns recent conversation messages.
    #[must_use]
    pub fn recent(&self) -> Vec<&Message> {
        self.runtime_items
            .iter()
            .filter_map(|item| match item {
                ContextCheckpointItem::Reminder(_) => None,
                ContextCheckpointItem::Message(message) => Some(message),
            })
            .collect()
    }

    /// Returns the latest recent message.
    #[must_use]
    pub fn last(&self) -> Option<&Message> {
        self.runtime_items.iter().rev().find_map(|item| match item {
            ContextCheckpointItem::Reminder(_) => None,
            ContextCheckpointItem::Message(message) => Some(message),
        })
    }

    /// Returns whether recent conversation is empty.
    #[must_use]
    pub fn is_conversation_empty(&self) -> bool {
        self.len_recent() == 0
    }

    /// Drains oldest recent messages while keeping the latest `keep` messages.
    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        let remove_count = self.len_recent().saturating_sub(keep);
        if remove_count == 0 {
            return Vec::new();
        }
        let mut removed = Vec::with_capacity(remove_count);
        let mut removed_messages = 0usize;
        let mut retained = Vec::with_capacity(self.runtime_items.len());
        for item in self.runtime_items.drain(..) {
            match item {
                ContextCheckpointItem::Message(message) if removed_messages < remove_count => {
                    removed.push(message);
                    removed_messages += 1;
                }
                other => retained.push(other),
            }
        }
        self.runtime_items = retained;
        removed
    }

    /// Returns recent conversation messages cloned.
    #[must_use]
    pub fn conversation_messages(&self) -> Vec<Message> {
        self.runtime_items
            .iter()
            .filter_map(|item| match item {
                ContextCheckpointItem::Reminder(_) => None,
                ContextCheckpointItem::Message(message) => Some(message.clone()),
            })
            .collect()
    }

    // ── Message assembly ───────────────────────────────────────────────

    /// Builds the full LLM message list.
    ///
    /// Layout:
    /// 1. Persistent system blocks (single cacheable prefix)
    /// 2. Compaction handoff (optional system message)
    /// 3. Ordered runtime items (reminders and conversation messages)
    #[must_use]
    pub fn build_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        if !self.system_blocks.is_empty() {
            let system_xml = self
                .system_blocks
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(Message::system(system_xml));
        }

        if let Some(handoff) = &self.handoff {
            messages.push(Message::system(handoff));
        }

        messages.extend(self.runtime_items.iter().cloned().map(|item| match item {
            ContextCheckpointItem::Reminder(content) => Message::system(content),
            ContextCheckpointItem::Message(message) => message,
        }));
        messages
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /// Clears only recent conversation messages.
    pub fn clear_recent(&mut self) {
        self.runtime_items
            .retain(|item| !matches!(item, ContextCheckpointItem::Message(_)));
    }

    /// Clears all non-persistent context: reminders, handoff, recent conversation.
    pub fn clear_history(&mut self) {
        self.handoff = None;
        self.runtime_items.clear();
    }

    /// Clears everything, including persistent system blocks.
    pub fn clear_all(&mut self) {
        self.system_blocks.clear();
        self.clear_history();
    }

    /// Clones the context.
    #[must_use]
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Creates a checkpoint that can be restored later.
    #[must_use]
    pub fn checkpoint(&self) -> ContextCheckpoint {
        ContextCheckpoint {
            handoff: self.handoff.clone(),
            items: self.runtime_items.clone(),
        }
    }

    /// Restores reminders, handoff, and recent conversation from checkpoint.
    pub fn restore(&mut self, checkpoint: ContextCheckpoint) {
        self.handoff = checkpoint.handoff;
        self.runtime_items = checkpoint.items;
    }

    // ── Staged system block updates ───────────────────────────────────

    /// Stages a replacement for a system block, deferring the cache-invalidating
    /// rewrite until the next [`reassemble`](Self::reassemble).
    ///
    /// If no block with `tag` exists yet, the content is inserted immediately
    /// (nothing to preserve in the cache) and `false` is returned. Otherwise the
    /// update is staged and `true` is returned; the host should surface the
    /// change to the model as an appended diff alert.
    pub fn stage_system_text(
        &mut self,
        tag: impl Into<String>,
        content: impl Into<String>,
    ) -> bool {
        let tag = tag.into();
        let content = content.into();
        if let Some(existing) = self.system_blocks.get(&tag) {
            if *existing == content {
                self.staged_system_updates.shift_remove(&tag);
                return false;
            }
            self.staged_system_updates.insert(tag, content);
            true
        } else {
            self.system_blocks.insert(tag, content);
            false
        }
    }

    /// Returns the staged replacement for `tag`, if any.
    #[must_use]
    pub fn staged_system_update(&self, tag: &str) -> Option<&str> {
        self.staged_system_updates.get(tag).map(String::as_str)
    }

    // ── Reassembly (lightweight, non-LLM context rewrite) ────────────

    /// Performs a lightweight context reassembly.
    ///
    /// This rewrites the window without calling any LLM:
    /// 1. folds staged system block updates into the stable prefix,
    /// 2. clears all ephemeral reminders,
    /// 3. removes recent system messages (system alerts),
    /// 4. tombstones tool results superseded by a later identical call or an
    ///    identical later result.
    ///
    /// Reassembly invalidates the KV-cache prefix, so hosts should run it when
    /// the cache is already cold (idle gap) or right before compaction.
    pub fn reassemble(&mut self) -> ReassemblyReport {
        let mut report = ReassemblyReport::default();

        for (tag, content) in std::mem::take(&mut self.staged_system_updates) {
            if let Some(existing) = self.system_blocks.get_mut(&tag) {
                report.bytes_freed += existing.len().saturating_sub(content.len());
                *existing = content;
            } else {
                self.system_blocks.insert(tag, content);
            }
            report.staged_updates_applied += 1;
        }

        let mut removed_bytes = 0usize;
        let mut reminders_cleared = 0usize;
        let mut system_messages_pruned = 0usize;
        self.runtime_items.retain(|item| match item {
            ContextCheckpointItem::Reminder(content) => {
                reminders_cleared += 1;
                removed_bytes += content.len();
                false
            }
            ContextCheckpointItem::Message(message) if message.role() == Role::System => {
                system_messages_pruned += 1;
                removed_bytes += message.content().len();
                false
            }
            ContextCheckpointItem::Message(_) => true,
        });
        report.reminders_cleared = reminders_cleared;
        report.system_messages_pruned = system_messages_pruned;
        report.bytes_freed += removed_bytes;

        let (deduped, dedupe_bytes) = self.tombstone_superseded_tool_results();
        report.tool_results_deduped = deduped;
        report.bytes_freed += dedupe_bytes;

        report
    }

    /// Estimates how many bytes a [`reassemble`](Self::reassemble) would free,
    /// without mutating the context.
    #[must_use]
    pub fn estimate_reassembly_savings(&self) -> usize {
        let alerts: usize = self
            .runtime_items
            .iter()
            .map(|item| match item {
                ContextCheckpointItem::Reminder(content) => content.len(),
                ContextCheckpointItem::Message(message) if message.role() == Role::System => {
                    message.content().len()
                }
                ContextCheckpointItem::Message(_) => 0,
            })
            .sum();
        let (_, dedupe_bytes) = self.find_superseded_tool_results();
        alerts + dedupe_bytes
    }

    /// Identifies tool results superseded by a later identical call (same tool
    /// name and arguments) or an identical later result.
    ///
    /// Returns the indices to tombstone and the bytes that doing so would free.
    fn find_superseded_tool_results(&self) -> (Vec<usize>, usize) {
        use std::collections::HashMap;

        // Map call id → identity key of (tool name, canonical arguments).
        let mut call_identity: HashMap<&str, String> = HashMap::new();
        for item in &self.runtime_items {
            let ContextCheckpointItem::Message(message) = item else {
                continue;
            };
            for call in message.tool_calls() {
                call_identity.insert(
                    call.id.as_str(),
                    format!("{}\u{0}{}", call.name, call.arguments),
                );
            }
        }

        // Last occurrence per identity key and per exact result content.
        let mut last_by_call: HashMap<&str, usize> = HashMap::new();
        let mut last_by_content: HashMap<&str, usize> = HashMap::new();
        for (index, item) in self.runtime_items.iter().enumerate() {
            let ContextCheckpointItem::Message(Message::Tool {
                content,
                tool_call_id,
            }) = item
            else {
                continue;
            };
            if let Some(identity) = call_identity.get(tool_call_id.as_str()) {
                last_by_call.insert(identity.as_str(), index);
            }
            last_by_content.insert(content.as_str(), index);
        }

        let mut indices = Vec::new();
        let mut bytes = 0usize;
        for (index, item) in self.runtime_items.iter().enumerate() {
            let ContextCheckpointItem::Message(Message::Tool {
                content,
                tool_call_id,
            }) = item
            else {
                continue;
            };
            if content.len() <= SUPERSEDED_TOOL_RESULT.len() {
                continue;
            }
            let superseded_by_call = call_identity
                .get(tool_call_id.as_str())
                .and_then(|identity| last_by_call.get(identity.as_str()))
                .is_some_and(|&last| last > index);
            let superseded_by_content = last_by_content
                .get(content.as_str())
                .is_some_and(|&last| last > index);
            if superseded_by_call || superseded_by_content {
                indices.push(index);
                bytes += content.len() - SUPERSEDED_TOOL_RESULT.len();
            }
        }
        (indices, bytes)
    }

    /// Applies [`find_superseded_tool_results`](Self::find_superseded_tool_results).
    fn tombstone_superseded_tool_results(&mut self) -> (usize, usize) {
        let (indices, bytes) = self.find_superseded_tool_results();
        for &index in &indices {
            if let Some(ContextCheckpointItem::Message(Message::Tool { content, .. })) =
                self.runtime_items.get_mut(index)
            {
                *content = SUPERSEDED_TOOL_RESULT.to_string();
            }
        }
        (indices.len(), bytes)
    }

    /// Removes recent system messages and returns how many were pruned.
    pub fn prune_recent_system_messages(&mut self) -> usize {
        let before = self.len_recent();
        self.runtime_items.retain(|item| {
            !matches!(
                item,
                ContextCheckpointItem::Message(message) if message.role() == Role::System
            )
        });
        before - self.len_recent()
    }

    /// Restores runtime-managed state from another serialized context while
    /// preserving the current persistent system blocks.
    pub fn restore_runtime_state(&mut self, restored: Self) {
        self.handoff = restored.handoff;
        self.runtime_items = restored.runtime_items;
    }
}

/// A snapshot of non-persistent context state that can be restored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCheckpoint {
    handoff: Option<String>,
    items: Vec<ContextCheckpointItem>,
}

impl ContextCheckpoint {
    /// Creates a structured runtime checkpoint.
    #[must_use]
    pub const fn new(handoff: Option<String>, items: Vec<ContextCheckpointItem>) -> Self {
        Self { handoff, items }
    }

    /// Returns the number of recent messages in this checkpoint.
    #[must_use]
    pub fn len_recent(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item, ContextCheckpointItem::Message(_)))
            .count()
    }

    /// Returns the current handoff document, if present.
    #[must_use]
    pub fn handoff(&self) -> Option<&str> {
        self.handoff.as_deref()
    }

    /// Returns ordered runtime checkpoint items.
    #[must_use]
    pub fn items(&self) -> &[ContextCheckpointItem] {
        &self.items
    }

    /// Returns whether this checkpoint has no handoff and no runtime items.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.handoff.is_none() && self.items.is_empty()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn snake_case_type_name<T>() -> String {
    let full = std::any::type_name::<T>();
    let short = full.rsplit("::").next().unwrap_or(full);
    heck::AsSnakeCase(short).to_string()
}

pub fn serialize_xml<T: Serialize>(tag: &str, value: &T) -> String {
    let mut buffer = String::new();
    let serializer = quick_xml::se::Serializer::with_root(&mut buffer, Some(tag))
        .unwrap_or_else(|error| panic!("failed to create XML serializer for tag '{tag}': {error}"));

    value
        .serialize(serializer)
        .unwrap_or_else(|error| panic!("failed to serialize XML for tag '{tag}': {error}"));

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Memory {
        #[serde(rename = "$text")]
        content: String,
    }

    #[derive(Serialize)]
    struct Reminder {
        #[serde(rename = "$text")]
        content: String,
    }

    #[derive(Serialize)]
    struct Workspace {
        sandbox: String,
        cwd: String,
    }

    #[test]
    fn snake_case_type_name_uses_short_struct_name() {
        assert_eq!(snake_case_type_name::<Memory>(), "memory");
    }

    #[test]
    fn serialize_xml_renders_text_content() {
        let xml = serialize_xml(
            "memory",
            &Memory {
                content: "remember this".to_string(),
            },
        );
        assert!(xml.contains("<memory>"));
        assert!(xml.contains("remember this"));
        assert!(xml.contains("</memory>"));
    }

    #[test]
    fn serialize_xml_renders_struct_fields() {
        let xml = serialize_xml(
            "workspace",
            &Workspace {
                sandbox: "/tmp/sandbox".to_string(),
                cwd: "/home/user".to_string(),
            },
        );
        assert!(xml.contains("<workspace>"));
        assert!(xml.contains("<sandbox>"));
        assert!(xml.contains("/tmp/sandbox"));
        assert!(xml.contains("</workspace>"));
    }

    #[test]
    fn insert_system_text_stores_raw_content() {
        let mut context = Context::new();
        context.insert_system_text("workspace", "Environment: local\nCWD: /tmp");

        assert_eq!(context.system_block_count(), 1);
        let messages = context.build_messages();
        assert_eq!(messages.len(), 1);
        let rendered = messages[0].content();
        assert!(rendered.contains("Environment: local"));
        assert!(rendered.contains("CWD: /tmp"));
        assert!(
            !rendered.contains("<workspace>"),
            "insert_system_text must not add XML tags: {rendered}"
        );
    }

    #[test]
    fn insert_system_text_replaces_by_tag() {
        let mut context = Context::new();
        context.insert_system_text("workspace", "old");
        context.insert_system_text("workspace", "new");

        assert_eq!(context.system_block_count(), 1);
        let messages = context.build_messages();
        assert!(messages[0].content().contains("new"));
        assert!(!messages[0].content().contains("old"));
    }

    #[test]
    fn stable_prefix_fingerprint_reflects_block_changes() {
        let mut context = Context::new();
        let empty = context.stable_prefix_fingerprint();

        context.insert_system_text("workspace", "a");
        let after_insert = context.stable_prefix_fingerprint();
        assert_ne!(empty, after_insert);

        context.insert_system_text("workspace", "b");
        let after_edit = context.stable_prefix_fingerprint();
        assert_ne!(after_insert, after_edit);

        context.remove_system_named("workspace");
        assert_eq!(context.stable_prefix_fingerprint(), empty);
    }

    #[test]
    fn stable_prefix_fingerprint_is_order_sensitive() {
        let mut a = Context::new();
        a.insert_system_text("one", "1");
        a.insert_system_text("two", "2");

        let mut b = Context::new();
        b.insert_system_text("two", "2");
        b.insert_system_text("one", "1");

        assert_ne!(
            a.stable_prefix_fingerprint(),
            b.stable_prefix_fingerprint(),
            "insertion order must be part of the fingerprint to reflect KV-cache prefix stability"
        );
    }

    #[test]
    fn insert_system_replaces_existing_block() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "old".to_string(),
        });
        context.insert_system(&Memory {
            content: "new".to_string(),
        });

        assert_eq!(context.system_block_count(), 1);
        let messages = context.build_messages();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].content().contains("new"));
        assert!(!messages[0].content().contains("old"));
    }

    #[test]
    fn build_messages_has_expected_layout() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "base".to_string(),
        });
        context.insert_reminder(&Reminder {
            content: "todo".to_string(),
        });
        context.set_handoff("summary");
        context.push(Message::user("hello"));

        let messages = context.build_messages();
        assert_eq!(messages.len(), 4);
        assert!(messages[0].content().contains("base"));
        assert!(messages[1].content().contains("handoff"));
        assert!(messages[2].content().contains("todo"));
        assert!(messages[3].content().contains("hello"));
    }

    #[test]
    fn clear_history_preserves_system_blocks() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "persist".to_string(),
        });
        context.insert_reminder(&Reminder {
            content: "ephemeral".to_string(),
        });
        context.set_handoff("summary");
        context.push(Message::assistant("recent"));

        context.clear_history();

        assert_eq!(context.system_block_count(), 1);
        assert_eq!(context.reminders(), Vec::<&str>::new());
        assert!(context.handoff().is_none());
        assert_eq!(context.len_recent(), 0);
    }

    #[test]
    fn checkpoint_restore_roundtrip() {
        let mut context = Context::new();
        context.insert_reminder(&Reminder {
            content: "before".to_string(),
        });
        context.set_handoff("before");
        context.push(Message::user("first"));

        let checkpoint = context.checkpoint();

        context.clear_history();
        context.insert_reminder(&Reminder {
            content: "after".to_string(),
        });
        context.push(Message::user("second"));

        context.restore(checkpoint);

        assert_eq!(context.reminders().len(), 1);
        assert!(context.handoff().is_some());
        assert_eq!(context.len_recent(), 1);
        assert_eq!(context.recent()[0].content(), "first");
    }

    #[test]
    fn restore_runtime_state_preserves_system_blocks() {
        let mut current = Context::new();
        current.insert_system(&Memory {
            content: "fresh".to_string(),
        });

        let mut restored = Context::new();
        restored.insert_system(&Memory {
            content: "stale".to_string(),
        });
        restored.insert_reminder(&Reminder {
            content: "ephemeral".to_string(),
        });
        restored.set_handoff("summary");
        restored.push(Message::user("hello"));

        current.restore_runtime_state(restored);

        let messages = current.build_messages();
        assert!(messages[0].content().contains("fresh"));
        assert!(!messages[0].content().contains("stale"));
        assert_eq!(current.reminders().len(), 1);
        assert!(current.handoff().is_some());
        assert_eq!(current.len_recent(), 1);
    }

    #[test]
    fn prune_recent_system_messages_removes_only_system_entries() {
        let mut context = Context::new();
        context.push(Message::user("user"));
        context.push(Message::system("system"));
        context.push(Message::assistant("assistant"));
        context.push(Message::system("system-2"));

        let removed = context.prune_recent_system_messages();

        assert_eq!(removed, 2);
        assert_eq!(context.len_recent(), 2);
        assert_eq!(context.recent()[0].content(), "user");
        assert_eq!(context.recent()[1].content(), "assistant");
    }

    #[test]
    fn stage_system_text_defers_existing_block_rewrite() {
        let mut context = Context::new();
        context.insert_system_text("memory", "v1");

        assert!(context.stage_system_text("memory", "v2"));
        // The live block is untouched until reassembly (cache-friendly).
        assert!(context.build_messages()[0].content().contains("v1"));
        assert_eq!(context.staged_system_update("memory"), Some("v2"));

        let report = context.reassemble();
        assert_eq!(report.staged_updates_applied, 1);
        assert!(context.build_messages()[0].content().contains("v2"));
        assert_eq!(context.staged_system_update("memory"), None);
    }

    #[test]
    fn stage_system_text_inserts_directly_when_block_missing() {
        let mut context = Context::new();
        assert!(!context.stage_system_text("memory", "fresh"));
        assert!(context.build_messages()[0].content().contains("fresh"));
    }

    #[test]
    fn reassemble_clears_alerts_and_tombstones_superseded_results() {
        use aither_core::llm::ToolCall;

        let mut context = Context::new();
        context.insert_reminder(&Reminder {
            content: "alert".to_string(),
        });
        context.push(Message::system("system alert"));

        let args = serde_json::json!({"script": "cat skills/slide/SKILL.md"});
        let long_result = "the full skill document ".repeat(20);
        context.push(Message::assistant_with_tool_calls(
            "",
            vec![ToolCall::new("call_1", "terminal", args.clone())],
        ));
        context.push(Message::tool("call_1", long_result.clone()));
        context.push(Message::assistant_with_tool_calls(
            "",
            vec![ToolCall::new("call_2", "terminal", args)],
        ));
        context.push(Message::tool("call_2", long_result.clone()));

        let estimated = context.estimate_reassembly_savings();
        assert!(estimated > 0);

        let report = context.reassemble();
        assert_eq!(report.reminders_cleared, 1);
        assert_eq!(report.system_messages_pruned, 1);
        assert_eq!(report.tool_results_deduped, 1);
        assert_eq!(report.bytes_freed, estimated);

        let recent = context.recent();
        // First (superseded) result is tombstoned; the latest keeps full content.
        assert!(recent[1].content().contains("superseded"));
        assert_eq!(recent[3].content(), long_result);
    }

    #[test]
    fn reassemble_tombstones_identical_result_content_across_different_calls() {
        use aither_core::llm::ToolCall;

        let mut context = Context::new();
        let duplicated = "identical output body ".repeat(10);
        context.push(Message::assistant_with_tool_calls(
            "",
            vec![ToolCall::new(
                "call_1",
                "terminal",
                serde_json::json!({"script": "cat a.txt"}),
            )],
        ));
        context.push(Message::tool("call_1", duplicated.clone()));
        context.push(Message::assistant_with_tool_calls(
            "",
            vec![ToolCall::new(
                "call_2",
                "terminal",
                serde_json::json!({"script": "sed -n 1,200p a.txt"}),
            )],
        ));
        context.push(Message::tool("call_2", duplicated.clone()));

        let report = context.reassemble();
        assert_eq!(report.tool_results_deduped, 1);
        let recent = context.recent();
        assert!(recent[1].content().contains("superseded"));
        assert_eq!(recent[3].content(), duplicated);
    }

    #[test]
    fn serde_roundtrip_preserves_message_layout() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            content: "persist".to_string(),
        });
        context.insert_reminder(&Reminder {
            content: "ephemeral".to_string(),
        });
        context.set_handoff("summary");
        context.push(Message::user("hello"));
        context.push(Message::assistant("world"));

        let expected_messages = context.build_messages();
        let encoded = serde_json::to_string(&context)
            .unwrap_or_else(|error| panic!("failed to serialize context: {error}"));
        let decoded: Context = serde_json::from_str(&encoded)
            .unwrap_or_else(|error| panic!("failed to deserialize context: {error}"));
        let actual_messages = decoded.build_messages();

        assert_eq!(actual_messages, expected_messages);
    }
}
