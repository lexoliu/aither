//! HNSW-based vector index using instant-distance.

use instant_distance::{Builder, HnswMap, Point, Search};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use std::collections::HashMap;

use crate::error::{RagError, Result};
use crate::types::{Chunk, IndexEntry, SearchResult};

use super::VectorIndex;

/// A point wrapper for instant-distance that stores an embedding vector.
#[derive(Clone, Debug)]
struct EmbeddingPoint {
    embedding: Vec<f32>,
}

impl Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        // This gives smaller values for more similar vectors
        1.0 - cosine_similarity(&self.embedding, &other.embedding)
    }
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (lhs, rhs) in a.iter().zip(b) {
        dot += lhs * rhs;
        norm_a += lhs * lhs;
        norm_b += rhs * rhs;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Internal state for the HNSW index.
struct IndexState {
    /// All stored entries (chunks + embeddings).
    entries: Vec<IndexEntry>,
    /// Map from chunk ID to index in entries.
    id_to_index: HashMap<String, usize>,
    /// Set of content hashes for deduplication.
    content_hashes: HashMap<u64, String>,
    /// The HNSW index (rebuilt after modifications).
    hnsw: Option<HnswMap<EmbeddingPoint, usize>>,
    /// Whether the index needs rebuilding.
    dirty: bool,
}

impl IndexState {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            id_to_index: HashMap::new(),
            content_hashes: HashMap::new(),
            hnsw: None,
            dirty: false,
        }
    }

    fn rebuild_hnsw(&mut self) {
        if self.entries.is_empty() {
            self.hnsw = None;
            self.dirty = false;
            return;
        }

        let points: Vec<EmbeddingPoint> = self
            .entries
            .iter()
            .map(|e| EmbeddingPoint {
                embedding: e.embedding.clone(),
            })
            .collect();

        let indices: Vec<usize> = (0..self.entries.len()).collect();

        self.hnsw = Some(Builder::default().build(points, indices));
        self.dirty = false;
    }
}

/// HNSW-based vector index for approximate nearest neighbor search.
///
/// This index uses the `instant-distance` crate for efficient similarity search.
/// It maintains an internal list of entries and rebuilds the HNSW graph as needed.
///
/// # Example
///
/// ```rust
/// use aither_rag::index::{HnswIndex, VectorIndex};
/// use aither_rag::Chunk;
///
/// let index = HnswIndex::new(384); // 384-dimensional embeddings
/// // index.insert(chunk, embedding).unwrap();
/// // let results = index.search(&query_embedding, 5, 0.0).unwrap();
/// ```
pub struct HnswIndex {
    dimension: usize,
    state: RwLock<IndexState>,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read();
        f.debug_struct("HnswIndex")
            .field("dimension", &self.dimension)
            .field("len", &state.entries.len())
            .finish()
    }
}

impl HnswIndex {
    /// Creates a new HNSW index with the specified embedding dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            state: RwLock::new(IndexState::new()),
        }
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&self, chunk: Chunk, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(RagError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let mut state = self.state.write();

        // Check if this chunk ID already exists
        if let Some(&idx) = state.id_to_index.get(&chunk.id) {
            // Update existing entry
            let old_hash = state.entries[idx].chunk.content_hash;
            state.content_hashes.remove(&old_hash);
            state
                .content_hashes
                .insert(chunk.content_hash, chunk.id.clone());
            state.entries[idx] = IndexEntry::new(chunk, embedding);
        } else {
            // Insert new entry
            let idx = state.entries.len();
            state.id_to_index.insert(chunk.id.clone(), idx);
            state
                .content_hashes
                .insert(chunk.content_hash, chunk.id.clone());
            state.entries.push(IndexEntry::new(chunk, embedding));
        }

        state.dirty = true;
        Ok(())
    }

    fn remove(&self, chunk_id: &str) -> bool {
        let mut state = self.state.write();

        let Some(&idx) = state.id_to_index.get(chunk_id) else {
            return false;
        };

        // Remove from content hashes
        let hash = state.entries[idx].chunk.content_hash;
        state.content_hashes.remove(&hash);

        // Remove from entries (swap-remove for efficiency)
        let removed = state.entries.swap_remove(idx);
        state.id_to_index.remove(&removed.chunk.id);

        // Update index of swapped element if any
        if idx < state.entries.len() {
            let swapped_id = state.entries[idx].chunk.id.clone();
            state.id_to_index.insert(swapped_id, idx);
        }

        state.dirty = true;
        true
    }

    fn search(&self, query: &[f32], top_k: usize, threshold: f32) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(RagError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let mut state = self.state.write();

        if state.entries.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // Rebuild HNSW if dirty
        if state.dirty || state.hnsw.is_none() {
            state.rebuild_hnsw();
        }

        let Some(ref hnsw) = state.hnsw else {
            return Ok(Vec::new());
        };

        let query_point = EmbeddingPoint {
            embedding: query.to_vec(),
        };

        let mut search = Search::default();
        let mut results = Vec::new();

        for candidate in hnsw.search(&query_point, &mut search).take(top_k) {
            let idx = *candidate.value;
            let entry = &state.entries[idx];

            // Convert distance back to similarity
            let similarity = 1.0 - candidate.distance;

            if similarity >= threshold {
                results.push(SearchResult {
                    chunk: entry.chunk.clone(),
                    score: similarity,
                });
            }
        }

        // Sort by score descending
        results.sort_by_key(|r| std::cmp::Reverse(OrderedFloat(r.score)));

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.state.read().entries.len()
    }

    fn clear(&self) {
        let mut state = self.state.write();
        state.entries.clear();
        state.id_to_index.clear();
        state.content_hashes.clear();
        state.hnsw = None;
        state.dirty = false;
    }

    fn entries(&self) -> Vec<IndexEntry> {
        self.state.read().entries.clone()
    }

    fn load(&self, entries: Vec<IndexEntry>) -> Result<()> {
        let mut state = self.state.write();

        state.entries.clear();
        state.id_to_index.clear();
        state.content_hashes.clear();

        for (idx, entry) in entries.into_iter().enumerate() {
            if entry.embedding.len() != self.dimension {
                return Err(RagError::DimensionMismatch {
                    expected: self.dimension,
                    actual: entry.embedding.len(),
                });
            }
            state.id_to_index.insert(entry.chunk.id.clone(), idx);
            state
                .content_hashes
                .insert(entry.chunk.content_hash, entry.chunk.id.clone());
            state.entries.push(entry);
        }

        state.dirty = true;
        Ok(())
    }

    fn contains_hash(&self, hash: u64) -> bool {
        self.state.read().content_hashes.contains_key(&hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(id: &str, text: &str) -> Chunk {
        Chunk::new(id, text, "doc1", 0, crate::dedup::content_hash(text))
    }

    #[test]
    fn insert_and_search() {
        let index = HnswIndex::new(4);

        let chunk1 = make_chunk("c1", "hello");
        let chunk2 = make_chunk("c2", "world");

        index.insert(chunk1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(chunk2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "c1");
    }

    #[test]
    fn remove_entry() {
        let index = HnswIndex::new(4);

        let chunk = make_chunk("c1", "hello");
        index.insert(chunk, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 1);
        assert!(index.remove("c1"));
        assert_eq!(index.len(), 0);
        assert!(!index.remove("c1"));
    }

    #[test]
    fn update_existing() {
        let index = HnswIndex::new(4);

        let chunk1 = make_chunk("c1", "hello");
        index.insert(chunk1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        // Update with same ID
        let chunk2 = make_chunk("c1", "world");
        index.insert(chunk2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 1);

        let results = index.search(&[0.0, 1.0, 0.0, 0.0], 1, 0.0).unwrap();
        assert_eq!(results[0].chunk.text, "world");
    }

    #[test]
    fn dimension_mismatch() {
        let index = HnswIndex::new(4);
        let chunk = make_chunk("c1", "hello");

        let result = index.insert(chunk, vec![1.0, 0.0]); // Wrong dimension
        assert!(matches!(result, Err(RagError::DimensionMismatch { .. })));
    }

    #[test]
    fn threshold_filtering() {
        let index = HnswIndex::new(4);

        let chunk1 = make_chunk("c1", "hello");
        let chunk2 = make_chunk("c2", "world");

        index.insert(chunk1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(chunk2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        // High threshold should filter out low-similarity results
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10, 0.9).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "c1");
    }

    #[test]
    fn contains_hash() {
        let index = HnswIndex::new(4);

        let text = "hello world";
        let hash = crate::dedup::content_hash(text);
        let chunk = Chunk::new("c1", text, "doc1", 0, hash);

        assert!(!index.contains_hash(hash));
        index.insert(chunk, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(index.contains_hash(hash));
    }

    #[test]
    fn load_entries() {
        let index = HnswIndex::new(4);

        let entries = vec![
            IndexEntry::new(make_chunk("c1", "hello"), vec![1.0, 0.0, 0.0, 0.0]),
            IndexEntry::new(make_chunk("c2", "world"), vec![0.0, 1.0, 0.0, 0.0]),
        ];

        index.load(entries).unwrap();
        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 0.0).unwrap();
        assert_eq!(results[0].chunk.id, "c1");
    }

    #[test]
    fn clear_index() {
        let index = HnswIndex::new(4);

        let chunk = make_chunk("c1", "hello");
        index.insert(chunk, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 1);
        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }
}
