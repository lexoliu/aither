use aither_core::embedding::Embedding;
use core::future::Future;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::error::Result;

/// A single memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Uuid,
    pub content: String,
    pub embedding: Embedding,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

impl Memory {
    pub fn new(content: impl Into<String>, embedding: Embedding) -> Self {
        let now = OffsetDateTime::now_utc();
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            embedding,
            user_id: None,
            agent_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }
}

/// Result of a vector search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f32,
}

/// Trait for storing and retrieving memories.
pub trait MemoryStore: Send + Sync {
    /// Add a new memory.
    fn add(&mut self, memory: Memory) -> impl Future<Output = Result<()>> + Send;

    /// Get a memory by ID.
    fn get(&self, id: Uuid) -> impl Future<Output = Result<Option<Memory>>> + Send;

    /// Update an existing memory.
    fn update(&mut self, memory: Memory) -> impl Future<Output = Result<()>> + Send;

    /// Delete a memory by ID.
    fn delete(&mut self, id: Uuid) -> impl Future<Output = Result<()>> + Send;

    /// Retrieve all stored memories.
    fn all(&self) -> impl Future<Output = Result<Vec<Memory>>> + Send;

    /// Search for memories similar to the query vector.
    ///
    /// `filters` can be used to filter by user_id or agent_id.
    /// `limit` is the maximum number of results to return.
    fn search(
        &self,
        query_embedding: &Embedding,
        limit: usize,
        filters: SearchFilters,
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;
}

#[derive(Debug, Default, Clone)]
pub struct SearchFilters {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
}

/// A simple in-memory store for testing and prototyping.
/// **Note**: This is O(N) for search and not persistent.
#[derive(Default)]
pub struct InMemoryStore {
    memories: Vec<Memory>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MemoryStore for InMemoryStore {
    async fn add(&mut self, memory: Memory) -> Result<()> {
        self.memories.push(memory);
        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<Memory>> {
        let memories = self.memories.clone();
        Ok(memories.iter().find(|m| m.id == id).cloned())
    }

    async fn update(&mut self, memory: Memory) -> Result<()> {
        if let Some(m) = self.memories.iter_mut().find(|m| m.id == memory.id) {
            *m = memory;
            Ok(())
        } else {
            Err(crate::error::Mem0Error::Store(format!(
                "Memory {} not found",
                memory.id
            )))
        }
    }

    async fn delete(&mut self, id: Uuid) -> Result<()> {
        self.memories.retain(|m| m.id != id);
        Ok(())
    }

    async fn all(&self) -> Result<Vec<Memory>> {
        let memories = self.memories.clone();
        Ok(memories)
    }

    async fn search(
        &self,
        query_embedding: &Embedding,
        limit: usize,
        filters: SearchFilters,
    ) -> Result<Vec<SearchResult>> {
        let memories = self.memories.clone();

        let mut scored_memories: Vec<SearchResult> = memories
            .iter()
            .filter(|m| {
                if let Some(uid) = &filters.user_id {
                    if m.user_id.as_ref() != Some(uid) {
                        return false;
                    }
                }
                if let Some(aid) = &filters.agent_id {
                    if m.agent_id.as_ref() != Some(aid) {
                        return false;
                    }
                }
                true
            })
            .map(|m| {
                let score = cosine_similarity(&m.embedding, query_embedding);
                SearchResult {
                    memory: m.clone(),
                    score,
                }
            })
            .collect();

        // Sort by score descending
        scored_memories.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        Ok(scored_memories.into_iter().take(limit).collect())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}
