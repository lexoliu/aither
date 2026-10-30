//! The context window and its prompt-cache-aware layout.
//!
//! Providers cache a prompt by its prefix: everything up to the first byte
//! that differs from the previous request is served from cache, and
//! everything after it is re-read and re-charged. That makes the layout of
//! the window the single most expensive decision the agent makes, so this
//! module enforces one rule — **the window grows only at the end**.
//!
//! ```text
//! ┌───────────────────────────────┐
//! │ system blocks (stable prefix) │  rewritten only by reassembly
//! ├───────────────────────────────┤
//! │ handoff document              │  replaced only by compaction
//! ├───────────────────────────────┤
//! │ conversation items (seq 1..n) │  append-only
//! ├───────────────────────────────┤
//! │ live tail (context sources)   │  rebuilt every request
//! └───────────────────────────────┘
//! ```
//!
//! Content that changes every turn — working documents, running jobs, window
//! pressure — lives in the tail, after the last cacheable byte, instead of
//! being interleaved with the conversation where it would invalidate the
//! prefix of every request that followed it. A rewrite of the stable part is
//! not forbidden, it is *scheduled*: [`Context::stage_system_text`] parks the
//! new content until [`Context::reassemble`] runs at a moment the cache is
//! already cold.
//!
//! Each item carries a monotonic sequence number. Compaction reads the window
//! at a request boundary and finishes several turns later, and `seq` is what
//! lets its result be applied to a window that has moved on: everything newer
//! than the fork point is kept verbatim.

use std::hash::{Hash, Hasher};

use askama::Template;
use indexmap::IndexMap;
use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use aither_core::llm::{Message, Role};

use super::compaction::{Handoff, HandoffDocument};
use super::source::Block;

/// A block of text wrapped in the one delimiting tag the model sees.
#[derive(Template)]
#[template(path = "tagged_block.txt", escape = "none")]
struct TaggedBlockTemplate<'a> {
    tag: &'a str,
    content: &'a str,
}

impl TaggedBlockTemplate<'_> {
    fn into_text(self) -> String {
        self.render().expect("tagged block template must render")
    }
}

/// Tombstone text left in place of a superseded tool result.
const SUPERSEDED_TOOL_RESULT: &str =
    "(superseded: an identical later call in this conversation holds the current result)";

/// A conversation message together with its position in the window.
///
/// The sequence number is assigned on insertion and never reused, so it
/// identifies an item across compaction and reassembly even though both
/// rewrite the surrounding window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextItem {
    /// Monotonic position of this item in the window.
    seq: u64,
    /// The message itself.
    message: Message,
    /// When the item was committed to the window. Bookkeeping only: it is
    /// never rendered into a request.
    committed_at: Timestamp,
}

impl ContextItem {
    /// The item's monotonic sequence number.
    #[must_use]
    pub const fn seq(&self) -> u64 {
        self.seq
    }

    /// The UTC time the item was committed to the window.
    #[must_use]
    pub const fn committed_at(&self) -> Timestamp {
        self.committed_at
    }

    /// The message this item carries.
    #[must_use]
    pub const fn message(&self) -> &Message {
        &self.message
    }

    /// Consumes the item, returning its message.
    #[must_use]
    pub fn into_message(self) -> Message {
        self.message
    }
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
    /// an appended alert instead) and let reassembly fold it in once the
    /// cache is already going to be invalidated.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    staged_system_updates: IndexMap<String, String>,

    /// Compaction handoff document from the previous compaction.
    ///
    /// Rendered immediately after the stable system prefix.
    handoff: Option<Handoff>,

    /// Append-only conversation items.
    items: Vec<ContextItem>,

    /// Sequence number the next item will receive.
    #[serde(default)]
    next_seq: u64,
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
    /// The block renders through `value`'s typed template; identity is
    /// derived from the `snake_case` type name of `T`.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render.
    pub fn insert_system<T: Template>(&mut self, value: &T) {
        let tag = snake_case_type_name::<T>();
        let rendered = value.render().expect("system block template must render");
        self.system_blocks.insert(tag, rendered);
    }

    /// Inserts or replaces a persistent system block with an explicit tag.
    pub fn insert_system_named(&mut self, tag: impl Into<String>, content: impl Into<String>) {
        let tag = tag.into();
        let content = content.into();
        let wrapped = TaggedBlockTemplate {
            tag: &tag,
            content: &content,
        }
        .into_text();
        self.system_blocks.insert(tag, wrapped);
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
    ///
    /// Mutating a block through this handle rewrites the cacheable prefix
    /// immediately; prefer [`stage_system_text`](Self::stage_system_text)
    /// mid-session.
    #[must_use]
    pub const fn system_blocks_mut(&mut self) -> &mut IndexMap<String, String> {
        &mut self.system_blocks
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

    // ── Staged system block updates ───────────────────────────────────

    /// Stages a replacement for a system block, deferring the
    /// cache-invalidating rewrite until the next
    /// [`reassemble`](Self::reassemble).
    ///
    /// If no block with `tag` exists yet, the content is inserted immediately
    /// (nothing to preserve in the cache) and `false` is returned. Otherwise
    /// the update is staged and `true` is returned; the host should surface
    /// the change to the model as an appended alert.
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

    /// Whether any system block rewrite is waiting for reassembly.
    #[must_use]
    pub fn has_staged_system_updates(&self) -> bool {
        !self.staged_system_updates.is_empty()
    }

    // ── Handoff ───────────────────────────────────────────────────────

    /// Sets or replaces the compaction handoff document.
    pub fn set_handoff(&mut self, handoff: impl Into<String>) {
        self.handoff = Some(Handoff::Text(handoff.into()));
    }

    /// Sets the compaction handoff document from a structured artifact.
    pub fn set_handoff_document(&mut self, handoff: &HandoffDocument) {
        self.handoff = Some(Handoff::Document(handoff.clone()));
    }

    /// Clears the compaction handoff document.
    pub fn clear_handoff(&mut self) {
        self.handoff = None;
    }

    /// Returns the current compaction handoff document.
    #[must_use]
    pub const fn handoff(&self) -> Option<&Handoff> {
        self.handoff.as_ref()
    }

    /// The file index of the current handoff, empty when there is none.
    #[must_use]
    pub fn handoff_file_index(&self) -> &[String] {
        self.handoff.as_ref().map_or(&[], Handoff::file_index)
    }

    // ── Conversation ──────────────────────────────────────────────────

    /// Appends a message to the window and returns its sequence number.
    pub fn push(&mut self, message: Message) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.items.push(ContextItem {
            seq,
            message,
            committed_at: Timestamp::now(),
        });
        seq
    }

    /// Appends several messages to the window.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        for message in messages {
            self.push(message);
        }
    }

    /// Appends a durable alert, rendered from its typed template as a system
    /// message.
    ///
    /// The alert enters the conversation and stays there until reassembly or
    /// compaction prunes it — the channel for content that must reach the
    /// *next* request rather than being re-rendered by a
    /// [`ContextSource`](super::ContextSource) every time.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render.
    pub fn push_alert<T: Template>(&mut self, value: &T) -> u64 {
        let rendered = value.render().expect("alert template must render");
        self.push(Message::system(rendered))
    }

    /// Appends a durable alert rendered by a context source.
    pub fn push_block_alert(&mut self, block: &Block) -> u64 {
        self.push(Message::system(block.content()))
    }

    /// Returns the sequence number a compaction may split the window at.
    ///
    /// This is the fork point handed to a background compaction: every item
    /// numbered at or above it arrived after the fork and must survive the
    /// handoff that replaces the rest.
    ///
    /// A tool round is never split: when the tail is an assistant turn whose
    /// tool calls still lack results, the point moves back to that turn's
    /// sequence number, so the kept side never starts with an orphaned
    /// result and the compacted side never ends inside an open round.
    #[must_use]
    pub fn fork_point(&self) -> u64 {
        let mut results = std::collections::HashSet::new();
        for item in self.items.iter().rev() {
            if let Some(id) = item.message.tool_call_id() {
                results.insert(id);
                continue;
            }
            let calls = item.message.tool_calls();
            if calls.is_empty() {
                break;
            }
            if calls.iter().any(|call| !results.contains(call.id.as_str())) {
                return item.seq;
            }
            break;
        }
        self.next_seq
    }

    /// Returns the items added at or after `seq`.
    #[must_use]
    pub fn items_since(&self, seq: u64) -> Vec<Message> {
        self.items
            .iter()
            .filter(|item| item.seq >= seq)
            .map(|item| item.message.clone())
            .collect()
    }

    /// Returns number of conversation messages.
    #[must_use]
    pub const fn len_recent(&self) -> usize {
        self.items.len()
    }

    /// Returns the number of system messages in the conversation.
    #[must_use]
    pub fn count_recent_system_messages(&self) -> usize {
        self.items
            .iter()
            .filter(|item| item.message.role() == Role::System)
            .count()
    }

    /// Returns the conversation messages.
    #[must_use]
    pub fn recent(&self) -> Vec<&Message> {
        self.items.iter().map(|item| &item.message).collect()
    }

    /// Returns the latest conversation message.
    #[must_use]
    pub fn last(&self) -> Option<&Message> {
        self.items.last().map(|item| &item.message)
    }

    /// Returns whether the conversation is empty.
    #[must_use]
    pub const fn is_conversation_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns the conversation messages, cloned.
    #[must_use]
    pub fn conversation_messages(&self) -> Vec<Message> {
        self.items.iter().map(|item| item.message.clone()).collect()
    }

    /// Drains the oldest messages while keeping the latest `keep` of them.
    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        let remove_count = self.items.len().saturating_sub(keep);
        if remove_count == 0 {
            return Vec::new();
        }
        self.items
            .drain(..remove_count)
            .map(ContextItem::into_message)
            .collect()
    }

    // ── Assembly ──────────────────────────────────────────────────────

    /// Builds the request message list, with `tail` appended after the
    /// conversation.
    ///
    /// Layout:
    /// 1. persistent system blocks, as one cacheable system message,
    /// 2. the compaction handoff, if any,
    /// 3. the append-only conversation,
    /// 4. one system message holding every live tail block.
    ///
    /// Steps 1-3 are byte-identical to the previous request whenever nothing
    /// was staged or compacted, which is exactly the condition a provider
    /// prefix cache needs.
    #[must_use]
    pub fn build_request(&self, tail: &[Block]) -> Vec<Message> {
        let mut messages = Vec::with_capacity(self.items.len() + 3);

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
            messages.push(Message::system(handoff.render()));
        }

        messages.extend(self.items.iter().map(|item| item.message.clone()));

        if let Some(tail_message) = render_tail(tail) {
            messages.push(tail_message);
        }

        messages
    }

    /// Appends the rendered live `tail`, if any, to the conversation as a
    /// message, so a context copied from a request carries exactly the
    /// messages that request had.
    pub(crate) fn push_tail(&mut self, tail: &[Block]) {
        if let Some(message) = render_tail(tail) {
            self.push(message);
        }
    }

    /// Builds the request message list with no live tail.
    ///
    /// Used for measurement and for the compaction fork, which must reproduce
    /// the prefix the main request already paid to cache.
    #[must_use]
    pub fn build_messages(&self) -> Vec<Message> {
        self.build_request(&[])
    }

    /// The number of characters the cacheable part of the window renders to.
    #[must_use]
    pub fn stable_chars(&self) -> usize {
        let blocks: usize = self
            .system_blocks
            .values()
            .map(|content| content.len() + 1)
            .sum();
        let handoff = self
            .handoff
            .as_ref()
            .map_or(0, |handoff| handoff.render().len());
        blocks + handoff
    }

    /// The number of characters the conversation renders to.
    #[must_use]
    pub fn conversation_chars(&self) -> usize {
        self.items
            .iter()
            .map(|item| message_chars(&item.message))
            .sum()
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /// Clears only the conversation, keeping system blocks and the handoff.
    pub fn clear_recent(&mut self) {
        self.items.clear();
    }

    /// Clears all non-persistent context: the handoff and the conversation.
    pub fn clear_history(&mut self) {
        self.handoff = None;
        self.items.clear();
    }

    /// Clears everything, including persistent system blocks.
    pub fn clear_all(&mut self) {
        self.system_blocks.clear();
        self.staged_system_updates.clear();
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
            items: self.items.clone(),
            next_seq: self.next_seq,
        }
    }

    /// Restores the handoff and conversation from a checkpoint.
    pub fn restore(&mut self, checkpoint: ContextCheckpoint) {
        self.handoff = checkpoint.handoff;
        self.next_seq = checkpoint
            .next_seq
            .max(checkpoint.items.last().map_or(0, |item| item.seq + 1));
        self.items = checkpoint.items;
    }

    /// Restores runtime-managed state from another serialized context while
    /// preserving the current persistent system blocks.
    pub fn restore_runtime_state(&mut self, restored: &Self) {
        self.restore(restored.checkpoint());
    }

    /// Replaces the conversation with a compaction result.
    ///
    /// `handoff` becomes the new anchor and only the items numbered at or
    /// above `fork_point` are kept: they arrived while the handoff was being
    /// written and are not described by it.
    ///
    /// Kept items lose reasoning bound to the old prefix: a provider whose
    /// thinking verifies the conversation around it rejects a block replayed
    /// under a rewritten prefix, so the strip happens here, where the rewrite
    /// happens. Text and tool calls stay; prefix-independent reasoning
    /// (`OpenAI`'s encrypted items, Gemini's signatures) stays valid and is
    /// kept.
    pub fn apply_handoff(&mut self, handoff: &HandoffDocument, fork_point: u64) {
        self.items.retain(|item| item.seq >= fork_point);
        for item in &mut self.items {
            item.message.strip_bound_reasoning();
        }
        self.set_handoff_document(handoff);
    }

    // ── Reassembly ────────────────────────────────────────────────────

    /// Performs a lightweight context reassembly.
    ///
    /// This rewrites the window without calling any LLM:
    /// 1. folds staged system block updates into the stable prefix,
    /// 2. removes durable system alerts,
    /// 3. tombstones tool results superseded by a later identical call or an
    ///    identical later result.
    ///
    /// Reassembly invalidates the cacheable prefix, so it runs as part of
    /// applying a compaction, which rewrites the prefix anyway.
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

        let mut system_messages_pruned = 0usize;
        let mut removed_bytes = 0usize;
        self.items.retain(|item| {
            if item.message.role() == Role::System {
                system_messages_pruned += 1;
                removed_bytes += item.message.content().len();
                false
            } else {
                true
            }
        });
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
            .items
            .iter()
            .filter(|item| item.message.role() == Role::System)
            .map(|item| item.message.content().len())
            .sum();
        let (_, dedupe_bytes) = self.find_superseded_tool_results();
        alerts + dedupe_bytes
    }

    /// Removes durable system alerts and returns how many were pruned.
    pub fn prune_recent_system_messages(&mut self) -> usize {
        let before = self.items.len();
        self.items
            .retain(|item| item.message.role() != Role::System);
        before - self.items.len()
    }

    /// Identifies tool results superseded by a later identical call (same
    /// tool name and arguments) or an identical later result.
    ///
    /// Returns the indices to tombstone and the bytes that doing so frees.
    fn find_superseded_tool_results(&self) -> (Vec<usize>, usize) {
        use std::collections::HashMap;

        // Map call id → identity key of (tool name, canonical arguments).
        let mut call_identity: HashMap<&str, String> = HashMap::new();
        for item in &self.items {
            for call in item.message.tool_calls() {
                call_identity.insert(
                    call.id.as_str(),
                    format!("{}\u{0}{}", call.name, call.arguments),
                );
            }
        }

        // Last occurrence per identity key and per exact result content.
        let mut last_by_call: HashMap<&str, usize> = HashMap::new();
        let mut last_by_content: HashMap<&str, usize> = HashMap::new();
        for (index, item) in self.items.iter().enumerate() {
            let Message::Tool {
                content,
                tool_call_id,
            } = &item.message
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
        for (index, item) in self.items.iter().enumerate() {
            let Message::Tool {
                content,
                tool_call_id,
            } = &item.message
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
            if let Some(item) = self.items.get_mut(index)
                && let Message::Tool { content, .. } = &mut item.message
            {
                *content = SUPERSEDED_TOOL_RESULT.to_string();
            }
        }
        (indices.len(), bytes)
    }
}

/// Outcome of a [`Context::reassemble`] pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReassemblyReport {
    /// Durable system alerts removed.
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
        self.system_messages_pruned > 0
            || self.tool_results_deduped > 0
            || self.staged_updates_applied > 0
    }
}

/// A snapshot of non-persistent context state that can be restored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCheckpoint {
    handoff: Option<Handoff>,
    items: Vec<ContextItem>,
    #[serde(default)]
    next_seq: u64,
}

impl ContextCheckpoint {
    /// Creates a structured runtime checkpoint.
    #[must_use]
    pub fn new(handoff: Option<Handoff>, items: Vec<ContextItem>) -> Self {
        let next_seq = items.last().map_or(0, |item| item.seq + 1);
        Self {
            handoff,
            items,
            next_seq,
        }
    }

    /// Returns the number of conversation messages in this checkpoint.
    #[must_use]
    pub const fn len_recent(&self) -> usize {
        self.items.len()
    }

    /// Returns the handoff document, if present.
    #[must_use]
    pub const fn handoff(&self) -> Option<&Handoff> {
        self.handoff.as_ref()
    }

    /// Returns the checkpointed conversation items.
    #[must_use]
    pub fn items(&self) -> &[ContextItem] {
        &self.items
    }

    /// Returns whether this checkpoint has no handoff and no items.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.handoff.is_none() && self.items.is_empty()
    }
}

impl Handoff {
    /// Renders this handoff as the system message body of a request.
    #[must_use]
    pub fn render(&self) -> String {
        let content = match self {
            Self::Text(text) => text.clone(),
            Self::Document(document) => document.render_markdown(),
        };
        TaggedBlockTemplate {
            tag: "handoff",
            content: &content,
        }
        .into_text()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Joins the live tail blocks into the single trailing system message.
fn render_tail(tail: &[Block]) -> Option<Message> {
    let mut rendered = String::new();
    for block in tail {
        if block.is_empty() {
            continue;
        }
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(block.content());
    }
    (!rendered.is_empty()).then(|| Message::system(rendered))
}

/// Approximate rendered size of a message, tool calls included.
fn message_chars(message: &Message) -> usize {
    let calls: usize = message
        .tool_calls()
        .iter()
        .map(|call| call.name.len() + call.arguments.to_string().len())
        .sum();
    message.content().len() + calls
}

fn snake_case_type_name<T>() -> String {
    let full = std::any::type_name::<T>();
    let short = full.rsplit("::").next().unwrap_or(full);
    heck::AsSnakeCase(short).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::ToolCall;

    #[derive(Template)]
    #[template(path = "tagged_block.txt", escape = "none")]
    struct Memory<'a> {
        tag: &'a str,
        content: &'a str,
    }

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
            reasoning_state: None,
        }
    }

    #[test]
    fn snake_case_type_name_uses_short_struct_name() {
        assert_eq!(snake_case_type_name::<Memory>(), "memory");
    }

    #[test]
    fn typed_system_blocks_render_from_their_template() {
        let mut context = Context::new();
        context.insert_system(&Memory {
            tag: "memory",
            content: "remember this",
        });
        context.insert_system_named("persona", "be brief");

        let messages = context.build_request(&[]);
        assert_eq!(
            messages[0].content(),
            "<memory>remember this</memory>\n<persona>be brief</persona>"
        );
    }

    #[test]
    fn alerts_render_from_their_template() {
        let mut context = Context::new();
        context.push_alert(&Memory {
            tag: "alert",
            content: "look here",
        });
        let messages = context.build_request(&[]);
        assert_eq!(messages[0].content(), "<alert>look here</alert>");
    }

    #[test]
    fn request_layout_puts_the_live_tail_last() {
        let mut context = Context::new();
        context.insert_system_named("base_system", "be useful");
        context.set_handoff("summary");
        context.push(Message::user("hello"));

        let messages = context.build_request(&[Block::new("tasks", "<tasks>todo</tasks>")]);

        assert_eq!(messages.len(), 4);
        assert!(messages[0].content().contains("be useful"));
        assert!(messages[1].content().contains("summary"));
        assert_eq!(messages[2].content(), "hello");
        assert_eq!(messages[3].role(), Role::System);
        assert!(messages[3].content().contains("todo"));
    }

    /// The invariant the whole layout exists to protect: appending to the
    /// conversation or changing the live tail must leave every earlier
    /// message byte-identical, or the provider re-reads the entire prompt.
    #[test]
    fn appending_never_perturbs_the_cacheable_prefix() {
        let mut context = Context::new();
        context.insert_system_named("base_system", "be useful");
        context.push(Message::user("first"));

        let before = context.build_request(&[Block::new("tasks", "<tasks>v1</tasks>")]);
        let prefix_len = before.len() - 1;

        context.push(Message::assistant("answer"));
        context.push(Message::user("second"));
        let after = context.build_request(&[Block::new("tasks", "<tasks>v2</tasks>")]);

        for index in 0..prefix_len {
            assert_eq!(
                before[index].content(),
                after[index].content(),
                "prefix message {index} changed"
            );
        }
        assert!(after.last().expect("tail present").content().contains("v2"));
    }

    #[test]
    fn staged_system_text_defers_the_prefix_rewrite() {
        let mut context = Context::new();
        context.insert_system_text("memory", "v1");

        assert!(context.stage_system_text("memory", "v2"));
        assert!(context.build_messages()[0].content().contains("v1"));
        assert_eq!(context.staged_system_update("memory"), Some("v2"));
        let fingerprint = context.stable_prefix_fingerprint();

        let report = context.reassemble();
        assert_eq!(report.staged_updates_applied, 1);
        assert!(context.build_messages()[0].content().contains("v2"));
        assert_ne!(context.stable_prefix_fingerprint(), fingerprint);
    }

    #[test]
    fn items_record_their_commit_time() {
        let mut context = Context::new();
        let before = Timestamp::now();
        context.push(Message::user("one"));
        let after = Timestamp::now();

        let committed = context.checkpoint().items()[0].committed_at();
        assert!(before <= committed && committed <= after);
    }

    #[test]
    fn checkpoint_round_trip_keeps_every_commit_time_and_the_request() {
        let mut context = Context::new();
        context.insert_system_named("base_system", "be useful");
        context.push(Message::user("first"));
        context.push(Message::assistant("answer"));
        context.push(Message::user("second"));
        let original_times: Vec<Timestamp> = context
            .checkpoint()
            .items()
            .iter()
            .map(ContextItem::committed_at)
            .collect();

        let json = serde_json::to_string(&context.checkpoint()).expect("serialize checkpoint");
        let checkpoint: ContextCheckpoint =
            serde_json::from_str(&json).expect("deserialize checkpoint");

        let mut restored = Context::new();
        restored.insert_system_named("base_system", "be useful");
        restored.restore(checkpoint);

        let restored_times: Vec<Timestamp> = restored
            .checkpoint()
            .items()
            .iter()
            .map(ContextItem::committed_at)
            .collect();
        assert_eq!(original_times, restored_times);

        let tail = [Block::new("tasks", "<tasks>v1</tasks>")];
        assert_eq!(context.build_request(&tail), restored.build_request(&tail));
    }

    #[test]
    fn sequence_numbers_are_monotonic_and_identify_the_fork_point() {
        let mut context = Context::new();
        assert_eq!(context.fork_point(), 0);
        assert_eq!(context.push(Message::user("one")), 0);
        let fork = context.fork_point();
        assert_eq!(context.push(Message::user("two")), 1);
        assert_eq!(context.items_since(fork).len(), 1);
    }

    #[test]
    fn applying_a_handoff_keeps_items_newer_than_the_fork() {
        let mut context = Context::new();
        context.push(Message::user("old work"));
        let fork = context.fork_point();
        context.push(Message::user("new request"));

        let handoff = HandoffDocument {
            current_task: "task".into(),
            ..HandoffDocument::default()
        };
        context.apply_handoff(&handoff, fork);

        assert_eq!(context.len_recent(), 1);
        assert_eq!(context.recent()[0].content(), "new request");
        assert!(
            context
                .handoff()
                .is_some_and(|handoff| handoff.render().contains("task"))
        );
    }

    #[test]
    fn applying_a_handoff_strips_prefix_bound_reasoning_from_kept_items() {
        let mut context = Context::new();
        context.push(Message::user("old work"));
        let fork = context.fork_point();
        context.push(Message::assistant_with_reasoning(
            "kept turn",
            vec![tool_call("c1", "terminal", serde_json::json!({}))],
            vec![
                aither_core::llm::ReasoningState::bound("claude", "sig-bound"),
                aither_core::llm::ReasoningState::new("openai", "enc-item"),
            ],
        ));
        context.push(Message::tool("c1", "done"));

        let handoff = HandoffDocument {
            current_task: "task".into(),
            ..HandoffDocument::default()
        };
        context.apply_handoff(&handoff, fork);

        let kept: Vec<&Message> = context.recent();
        assert_eq!(kept.len(), 2);
        let reasoning = kept[0].reasoning();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0].provider(), "openai");
        assert_eq!(reasoning[0].payload(), "enc-item");
        assert_eq!(kept[0].content(), "kept turn");
        assert_eq!(kept[0].tool_calls().len(), 1);
        assert_eq!(kept[1].content(), "done");
    }

    #[test]
    fn fork_point_never_splits_an_open_tool_round() {
        let mut context = Context::new();
        context.push(Message::user("closed round"));
        context.push(Message::assistant_with_tool_calls(
            "",
            vec![
                tool_call("a", "read", serde_json::json!({})),
                tool_call("b", "read", serde_json::json!({})),
            ],
        ));
        let round_start = context.fork_point();
        context.push(Message::tool("a", "one result"));

        // One result is still missing: the round is open, so the fork point
        // backs up to the assistant turn that opened it.
        assert_eq!(context.fork_point(), round_start);

        context.push(Message::tool("b", "two result"));
        assert_eq!(context.fork_point(), context.len_recent() as u64);
    }

    #[test]
    fn reassembly_prunes_alerts_and_tombstones_stale_tool_results() {
        let mut context = Context::new();
        context.push(Message::assistant_with_tool_calls(
            "",
            vec![tool_call("a", "read", serde_json::json!({"path": "x"}))],
        ));
        // Only results larger than the tombstone are worth replacing.
        context.push(Message::tool("a", "first read ".repeat(32)));
        context.push(Message::system("<alert>something happened</alert>"));
        context.push(Message::assistant_with_tool_calls(
            "",
            vec![tool_call("b", "read", serde_json::json!({"path": "x"}))],
        ));
        context.push(Message::tool("b", "second read ".repeat(32)));

        let estimate = context.estimate_reassembly_savings();
        let report = context.reassemble();

        assert_eq!(report.system_messages_pruned, 1);
        assert_eq!(report.tool_results_deduped, 1);
        assert!(report.changed());
        assert_eq!(estimate, report.bytes_freed);
        assert_eq!(context.recent()[1].content(), SUPERSEDED_TOOL_RESULT);
    }

    #[test]
    fn checkpoint_round_trip_preserves_sequence_numbers() {
        let mut context = Context::new();
        context.insert_system_named("base_system", "stable");
        context.push(Message::user("one"));
        let fork = context.fork_point();
        context.push(Message::user("two"));

        let checkpoint = context.checkpoint();
        let mut restored = Context::new();
        restored.insert_system_named("base_system", "stable");
        restored.restore(checkpoint);

        assert_eq!(restored.fork_point(), context.fork_point());
        assert_eq!(restored.items_since(fork).len(), 1);
        assert_eq!(restored.push(Message::user("three")), 2);
    }

    #[test]
    fn restore_runtime_state_preserves_system_blocks() {
        let mut current = Context::new();
        current.insert_system_named("base_system", "keep me");

        let mut restored = Context::new();
        restored.push(Message::user("hello"));
        restored.set_handoff("summary");

        current.restore_runtime_state(&restored);

        let messages = current.build_messages();
        assert!(messages[0].content().contains("keep me"));
        assert!(messages[1].content().contains("summary"));
        assert_eq!(current.len_recent(), 1);
    }

    #[test]
    fn drain_oldest_keeps_the_requested_tail() {
        let mut context = Context::new();
        for index in 0..5 {
            context.push(Message::user(format!("message {index}")));
        }
        let drained = context.drain_oldest(2);
        assert_eq!(drained.len(), 3);
        assert_eq!(context.len_recent(), 2);
        assert_eq!(context.recent()[0].content(), "message 3");
    }

    #[test]
    fn an_empty_tail_adds_no_message() {
        let mut context = Context::new();
        context.push(Message::user("hello"));
        assert_eq!(context.build_request(&[]).len(), 1);
        assert_eq!(
            context.build_request(&[Block::new("empty", "")]).len(),
            1,
            "a block that renders to nothing must not cost a message"
        );
    }
}
