//! Mem0-inspired long-term memory pipeline without the graph add-on.
//!
//! This crate implements the extraction/update workflow described in the
//! “Memo: Building Production-Ready AI Agents with Scalable Long-Term Memory”
//! paper. The pipeline keeps a rolling summary of the conversation, extracts
//! salient facts from each exchange, and updates a dense vector store by asking
//! an LLM to choose `ADD`, `UPDATE`, `DELETE`, or `NOOP` operations.
//!
//! The focus of this crate is the base Mem0 variant (no graph database). The
//! [`Mem0`] struct owns everything you need:
//! - Maintains a conversation summary refreshed every few exchanges.
//! - Runs the extraction phase with a configurable recency window.
//! - Evaluates candidates against similar memories and applies the correct
//!   operation selected by the LLM.
//! - Exposes a simple cosine-similarity store for downstream recall.

use std::{collections::BTreeMap, fmt::Write};

use aither_core::{
    EmbeddingModel, LanguageModel, Result,
    llm::{Message, oneshot},
};
use anyhow::{Context, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// Arbitrary metadata linked to each stored memory.
pub type Metadata = BTreeMap<String, String>;

/// Configures the Mem0 pipeline behaviour.
#[derive(Debug, Clone)]
pub struct Mem0Config {
    recency_window: usize,
    similar_memories: usize,
    summary_refresh_interval: usize,
    summary_instructions: String,
    extraction_instructions: String,
    update_instructions: String,
}

impl Default for Mem0Config {
    fn default() -> Self {
        Self {
            recency_window: 10,
            similar_memories: 10,
            summary_refresh_interval: 6,
            summary_instructions: "Summarize the whole dialogue tracking stable facts, \
                long-term preferences, and relationship details. Keep it under 200 words."
                .into(),
            extraction_instructions: "Extract compact memories from the new exchange. \
                Each memory must mention who said it, the concrete fact, and when it happened \
                if time information is available. Reject chit-chat."
                .into(),
            update_instructions: "ADD when the candidate fact is new. UPDATE when it \
                complements an existing memory (return the merged memory). DELETE when the \
                candidate contradicts an older memory. NOOP when the fact is irrelevant."
                .into(),
        }
    }
}

impl Mem0Config {
    /// Sets the extraction recency window `m`.
    #[must_use]
    pub fn with_recency_window(mut self, value: usize) -> Self {
        self.recency_window = value.max(2);
        self
    }

    /// Sets how many similar memories are inspected during the update phase.
    #[must_use]
    pub fn with_similar_memories(mut self, value: usize) -> Self {
        self.similar_memories = value.max(1);
        self
    }

    /// Sets how often (in processed exchanges) the summary is refreshed.
    #[must_use]
    pub fn with_summary_refresh_interval(mut self, value: usize) -> Self {
        self.summary_refresh_interval = value.max(1);
        self
    }

    /// Overrides the instructions used while refreshing the conversation summary.
    #[must_use]
    pub fn with_summary_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.summary_instructions = instructions.into();
        self
    }

    /// Overrides the memory extraction instructions.
    #[must_use]
    pub fn with_extraction_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.extraction_instructions = instructions.into();
        self
    }

    /// Overrides the update instructions.
    #[must_use]
    pub fn with_update_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.update_instructions = instructions.into();
        self
    }

    /// Returns the configured recency window.
    #[must_use]
    pub const fn recency_window(&self) -> usize {
        self.recency_window
    }

    /// Returns the configured summary refresh cadence.
    #[must_use]
    pub const fn summary_refresh_interval(&self) -> usize {
        self.summary_refresh_interval
    }

    /// Returns the `s` hyper-parameter (neighbor count).
    #[must_use]
    pub const fn similar_memories(&self) -> usize {
        self.similar_memories
    }
}

/// Importance level returned by the extraction step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryImportance {
    /// Critical persona or preference information.
    Critical,
    /// Useful for future conversations.
    Important,
    /// Lightweight background or situational context.
    Contextual,
}

impl Default for MemoryImportance {
    fn default() -> Self {
        Self::Important
    }
}

/// Structured payload returned by the extraction and update stages.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct MemoryPayload {
    /// Textual representation of the memory.
    pub content: String,
    /// Optional participant identifier.
    pub speaker: Option<String>,
    /// Free-form timestamp (the pipeline normalizes it later if desired).
    pub timestamp: Option<String>,
    /// How important the fact is.
    #[serde(default)]
    pub importance: MemoryImportance,
    /// Additional metadata, e.g., tags or external identifiers.
    #[serde(default)]
    pub metadata: Metadata,
}

impl MemoryPayload {
    fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }
}

/// Represents a stored memory fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Stable identifier generated by the store.
    pub id: String,
    /// Stored payload.
    pub payload: MemoryPayload,
    /// Creation timestamp.
    pub created_at: OffsetDateTime,
    /// Last update timestamp.
    pub updated_at: OffsetDateTime,
    embedding: Vec<f32>,
}

impl MemoryEntry {
    fn context(&self, score: f32) -> MemoryContext {
        MemoryContext {
            id: self.id.clone(),
            payload: self.payload.clone(),
            score,
        }
    }
}

/// Lightweight projection used to show similar memories to the LLM.
#[derive(Debug, Clone, Serialize)]
struct MemoryContext {
    id: String,
    payload: MemoryPayload,
    score: f32,
}

/// Cosine-similarity store keeping all memories in memory.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    entries: Vec<MemoryEntry>,
    next_id: u64,
    dimension: usize,
}

impl MemoryStore {
    /// Creates an empty store sized to the embedder dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            dimension,
        }
    }

    /// Returns all stored entries.
    #[must_use]
    pub fn entries(&self) -> &[MemoryEntry] {
        self.entries.as_slice()
    }

    fn insert(&mut self, payload: MemoryPayload, embedding: Vec<f32>) -> MemoryEntry {
        let now = OffsetDateTime::now_utc();
        let id = format!("mem-{}", self.next_id);
        self.next_id += 1;

        let entry = MemoryEntry {
            id,
            payload,
            created_at: now,
            updated_at: now,
            embedding,
        };
        self.entries.push(entry.clone());
        entry
    }

    fn update(
        &mut self,
        id: &str,
        payload: MemoryPayload,
        embedding: Vec<f32>,
    ) -> Option<MemoryEntry> {
        let now = OffsetDateTime::now_utc();
        let entry = self.entries.iter_mut().find(|entry| entry.id == id)?;
        entry.payload = payload;
        entry.embedding = embedding;
        entry.updated_at = now;
        Some(entry.clone())
    }

    fn remove(&mut self, id: &str) -> Option<MemoryEntry> {
        if let Some(pos) = self.entries.iter().position(|entry| entry.id == id) {
            Some(self.entries.remove(pos))
        } else {
            None
        }
    }

    fn top_similar(&self, vector: &[f32], top_k: usize) -> Vec<MemoryContext> {
        if vector.len() != self.dimension || self.entries.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let mut scored = self
            .entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(vector, &entry.embedding);
                entry.context(score)
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k);
        scored
    }

    fn recall(&self, vector: &[f32], top_k: usize) -> Vec<MemoryEntry> {
        if vector.len() != self.dimension {
            return Vec::new();
        }

        let mut scored = self
            .entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(vector, &entry.embedding);
                (entry, score)
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);
        scored.into_iter().map(|(entry, _)| entry.clone()).collect()
    }
}

/// Outcome produced after applying an update operation.
#[derive(Debug, Clone)]
pub struct MemoryChange {
    /// Operation that was executed.
    pub operation: OperationKind,
    /// Entry affected by the change (where applicable).
    pub entry: Option<MemoryEntry>,
    /// Raw target identifier returned by the update LLM.
    pub target: Option<String>,
    /// Optional reasoning string supplied by the model.
    pub reasoning: Option<String>,
}

/// High-level Mem0 orchestrator that maintains summaries and a memory store.
#[derive(Debug)]
pub struct Mem0<LLM, EMB> {
    llm: LLM,
    embedder: EMB,
    config: Mem0Config,
    store: MemoryStore,
    history: Vec<Message>,
    summary: Option<String>,
    processed_pairs: usize,
}

impl<LLM, EMB> Mem0<LLM, EMB>
where
    LLM: LanguageModel,
    EMB: EmbeddingModel,
{
    /// Creates a Mem0 instance with the default configuration.
    #[must_use]
    pub fn new(llm: LLM, embedder: EMB) -> Self {
        Self::with_config(llm, embedder, Mem0Config::default())
    }

    /// Creates a Mem0 instance with custom configuration.
    #[must_use]
    pub fn with_config(llm: LLM, embedder: EMB, config: Mem0Config) -> Self {
        let store = MemoryStore::new(embedder.dim());
        Self {
            llm,
            embedder,
            config,
            store,
            history: Vec::new(),
            summary: None,
            processed_pairs: 0,
        }
    }

    /// Returns the current configuration.
    #[must_use]
    pub const fn config(&self) -> &Mem0Config {
        &self.config
    }

    /// Returns the current long-term summary, if one exists.
    #[must_use]
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Returns all stored memories.
    #[must_use]
    pub fn memories(&self) -> &[MemoryEntry] {
        self.store.entries()
    }

    /// Ingests the latest message pair (assistant/user or user/assistant).
    ///
    /// This performs the extraction and update phases. The returned vector
    /// enumerates every change made to the memory store.
    pub async fn ingest_exchange(
        &mut self,
        previous: Message,
        current: Message,
    ) -> Result<Vec<MemoryChange>> {
        self.history.push(previous.clone());
        self.history.push(current.clone());
        self.processed_pairs = self.processed_pairs.saturating_add(1);
        self.refresh_summary_if_needed().await?;

        let summary = self
            .summary()
            .map_or_else(|| "No summary available".to_string(), ToString::to_string);
        let backlog = self.recent_backlog();

        let extracted = self
            .extract_memories(&summary, &backlog, &previous, &current)
            .await?;

        let mut changes = Vec::new();
        for mut payload in extracted {
            if payload.is_empty() {
                continue;
            }

            if payload.importance == MemoryImportance::Contextual {
                payload
                    .metadata
                    .entry("priority".into())
                    .or_insert_with(|| "contextual".into());
            }

            let embedding = self.embedder.embed(&payload.content).await?;
            let similar = self
                .store
                .top_similar(&embedding, self.config.similar_memories());

            let decision = self.decide_operation(&summary, &payload, &similar).await?;

            if let Some(change) = self.apply_decision(decision, payload, embedding).await? {
                changes.push(change);
            }
        }

        Ok(changes)
    }

    /// Performs a semantic recall across the stored memories.
    pub async fn recall(&self, query: &str, top_k: usize) -> Result<Vec<MemoryEntry>> {
        let vector = self.embedder.embed(query).await?;
        Ok(self.store.recall(&vector, top_k))
    }

    async fn refresh_summary_if_needed(&mut self) -> Result<()> {
        if self.history.is_empty() {
            return Ok(());
        }

        let needs_refresh = self.summary.is_none()
            || self.processed_pairs % self.config.summary_refresh_interval() == 0;
        if !needs_refresh {
            return Ok(());
        }

        let dialogue = format_messages(&self.history);
        let system = "You maintain a rolling summary of a conversation for a memory module.";
        let user = format!(
            "{instructions}\n\nConversation:\n{dialogue}",
            instructions = self.config.summary_instructions
        );

        let request = oneshot(system, user);
        let summary: String = self.llm.generate(request).await?;
        let trimmed = summary.trim();
        if !trimmed.is_empty() {
            self.summary = Some(trimmed.to_string());
        }
        Ok(())
    }

    fn recent_backlog(&self) -> Vec<Message> {
        if self.history.len() <= 2 {
            return Vec::new();
        }

        let total = self.history.len().saturating_sub(2);
        let keep = self.config.recency_window().min(total);
        self.history[total - keep..total].to_vec()
    }

    async fn extract_memories(
        &self,
        summary: &str,
        backlog: &[Message],
        previous: &Message,
        current: &Message,
    ) -> Result<Vec<MemoryPayload>> {
        let context = if backlog.is_empty() {
            "No additional backlog supplied.".to_string()
        } else {
            format_messages(backlog)
        };

        let prompt = format!(
            "# Summary\n{summary}\n\n# Recent backlog\n{context}\n\n# Latest exchange\n{}\n{}\n\n\
             Instructions: {}\nReturn JSON following the schema.",
            format_message(previous),
            format_message(current),
            self.config.extraction_instructions
        );
        let request = oneshot(
            "You are a precision memory extractor. Only emit factual memories.",
            prompt,
        );

        let ExtractionBatch { memories } = self.llm.generate(request).await?;
        Ok(memories)
    }

    async fn decide_operation(
        &self,
        summary: &str,
        candidate: &MemoryPayload,
        similar: &[MemoryContext],
    ) -> Result<OperationDecision> {
        let context_json =
            serde_json::to_string_pretty(similar).context("failed to encode similar memories")?;
        let candidate_json =
            serde_json::to_string_pretty(candidate).context("failed to encode candidate memory")?;

        let prompt = format!(
            "Summary:\n{summary}\n\nCandidate memory:\n{candidate_json}\n\n\
             Similar memories (descending similarity):\n{context_json}\n\n\
             {instructions}\nReturn a JSON object that contains the operation, optional \
             target id, optional merged memory payload, and concise reasoning.",
            instructions = self.config.update_instructions
        );

        let request = oneshot("You maintain a consistent memory database.", prompt);
        self.llm.generate(request).await
    }

    async fn apply_decision(
        &mut self,
        mut decision: OperationDecision,
        candidate: MemoryPayload,
        candidate_embedding: Vec<f32>,
    ) -> Result<Option<MemoryChange>> {
        match decision.action {
            OperationKind::Add => {
                let payload = decision
                    .merged_memory
                    .take()
                    .unwrap_or_else(|| candidate.clone());
                let entry = self.store.insert(payload, candidate_embedding);
                Ok(Some(MemoryChange {
                    operation: OperationKind::Add,
                    entry: Some(entry),
                    target: decision.target.take(),
                    reasoning: decision.reasoning.take(),
                }))
            }
            OperationKind::Update => {
                let target = decision
                    .target
                    .clone()
                    .ok_or_else(|| anyhow!("update operation missing target id"))?;
                let payload = decision
                    .merged_memory
                    .take()
                    .unwrap_or_else(|| candidate.clone());
                let embedding = if payload.content == candidate.content {
                    candidate_embedding
                } else {
                    self.embedder.embed(&payload.content).await?
                };
                let entry = self
                    .store
                    .update(&target, payload, embedding)
                    .ok_or_else(|| anyhow!("target {target} not found for update"))?;
                Ok(Some(MemoryChange {
                    operation: OperationKind::Update,
                    entry: Some(entry),
                    target: Some(target),
                    reasoning: decision.reasoning.take(),
                }))
            }
            OperationKind::Delete => {
                let target = decision
                    .target
                    .clone()
                    .ok_or_else(|| anyhow!("delete operation missing target id"))?;
                let entry = self
                    .store
                    .remove(&target)
                    .ok_or_else(|| anyhow!("target {target} not found for delete"))?;
                Ok(Some(MemoryChange {
                    operation: OperationKind::Delete,
                    entry: Some(entry),
                    target: Some(target),
                    reasoning: decision.reasoning.take(),
                }))
            }
            OperationKind::Noop => Ok(None),
        }
    }
}

/// Structured decision returned by the update model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OperationDecision {
    /// Operation to execute.
    pub action: OperationKind,
    /// Target memory identifier for `UPDATE` and `DELETE`.
    #[serde(default)]
    pub target: Option<String>,
    /// Optional merged payload for `UPDATE`/`ADD`.
    #[serde(default)]
    pub merged_memory: Option<MemoryPayload>,
    /// Optional explanation for auditing.
    #[serde(default)]
    pub reasoning: Option<String>,
}

/// Operations supported by the Mem0 update phase.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    /// Insert a new memory entry.
    Add,
    /// Merge the candidate into an existing memory.
    Update,
    /// Remove an outdated memory.
    Delete,
    /// Skip the candidate.
    Noop,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ExtractionBatch {
    memories: Vec<MemoryPayload>,
}

fn format_messages(messages: &[Message]) -> String {
    let mut out = String::new();
    for message in messages {
        let _ = writeln!(out, "{}", format_message(message));
    }
    out.trim().to_string()
}

fn format_message(message: &Message) -> String {
    format!(
        "{role:?}: {content}",
        role = message.role(),
        content = message.content()
    )
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut mag_a = 0.0;
    let mut mag_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a.sqrt() * mag_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_behaves() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!(cosine_similarity(&a, &b) > cosine_similarity(&a, &c));
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn memory_store_insert_update_delete() {
        let mut store = MemoryStore::new(3);
        let payload = MemoryPayload {
            content: "Alice loves hiking".into(),
            speaker: Some("alice".into()),
            importance: MemoryImportance::Important,
            ..MemoryPayload::default()
        };
        let embedding = vec![0.5, 0.1, 0.2];
        let entry = store.insert(payload.clone(), embedding.clone());
        assert_eq!(store.entries().len(), 1);

        let updated_payload = MemoryPayload {
            content: "Alice loves night hiking".into(),
            ..payload
        };
        let updated = store
            .update(&entry.id, updated_payload.clone(), embedding.clone())
            .unwrap();
        assert_eq!(updated.payload.content, updated_payload.content);
        assert_eq!(store.entries().len(), 1);

        let removed = store.remove(&entry.id).unwrap();
        assert_eq!(removed.id, entry.id);
        assert!(store.entries().is_empty());
    }
}
