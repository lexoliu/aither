//! HNSW-based vector index using instant-distance.

use arc_swap::{ArcSwap, Guard};
use instant_distance::{Builder, HnswMap, Point, Search};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::sync::Arc;

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

    fn build_hnsw(entries: &[IndexEntry]) -> Option<HnswMap<EmbeddingPoint, usize>> {
        if entries.is_empty() {
            return None;
        }

        let points: Vec<EmbeddingPoint> = entries
            .iter()
            .map(|e| EmbeddingPoint {
                embedding: e.embedding.clone(),
            })
            .collect();

        let indices: Vec<usize> = (0..entries.len()).collect();
        Some(Builder::default().build(points, indices))
    }

    fn rebuilt(&self) -> Self {
        let mut next = self.clone_without_index();
        next.hnsw = Self::build_hnsw(&next.entries);
        next.dirty = false;
        next
    }

    fn clone_without_index(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            id_to_index: self.id_to_index.clone(),
            content_hashes: self.content_hashes.clone(),
            hnsw: None,
            dirty: true,
        }
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
    state: ArcSwap<IndexState>,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.load();
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
            state: ArcSwap::from_pointee(IndexState::new()),
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

        self.state.rcu(|state| {
            let mut next = state.clone_without_index();
            if let Some(&idx) = next.id_to_index.get(&chunk.id) {
                let old_hash = next.entries[idx].chunk.content_hash;
                next.content_hashes.remove(&old_hash);
                next.content_hashes
                    .insert(chunk.content_hash, chunk.id.clone());
                next.entries[idx] = IndexEntry::new(chunk.clone(), embedding.clone());
            } else {
                let idx = next.entries.len();
                next.id_to_index.insert(chunk.id.clone(), idx);
                next.content_hashes
                    .insert(chunk.content_hash, chunk.id.clone());
                next.entries
                    .push(IndexEntry::new(chunk.clone(), embedding.clone()));
            }
            next
        });
        Ok(())
    }

    fn remove(&self, chunk_id: &str) -> bool {
        let mut found = false;
        // The closure runs again on a newer state when another update wins,
        // so the entry is looked up in the state each attempt sees.
        self.state.rcu(|state| {
            let Some(&idx) = state.id_to_index.get(chunk_id) else {
                found = false;
                return Arc::clone(state);
            };
            found = true;
            let mut next = state.clone_without_index();
            let hash = next.entries[idx].chunk.content_hash;
            next.content_hashes.remove(&hash);

            let removed = next.entries.swap_remove(idx);
            next.id_to_index.remove(&removed.chunk.id);

            if idx < next.entries.len() {
                let swapped_id = next.entries[idx].chunk.id.clone();
                next.id_to_index.insert(swapped_id, idx);
            }

            Arc::new(next)
        });
        found
    }

    fn search(&self, query: &[f32], top_k: usize, threshold: f32) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(RagError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        if top_k == 0 {
            return Ok(Vec::new());
        }

        let state = loop {
            let current = self.state.load_full();
            if current.entries.is_empty() {
                return Ok(Vec::new());
            }
            if !current.dirty && current.hnsw.is_some() {
                break current;
            }

            let rebuilt = Arc::new(current.rebuilt());
            let previous =
                Guard::into_inner(self.state.compare_and_swap(&current, Arc::clone(&rebuilt)));
            if Arc::ptr_eq(&current, &previous) {
                break rebuilt;
            }
        };

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
        self.state.load().entries.len()
    }

    fn clear(&self) {
        self.state.store(Arc::new(IndexState::new()));
    }

    fn entries(&self) -> Vec<IndexEntry> {
        self.state.load().entries.clone()
    }

    fn load(&self, entries: Vec<IndexEntry>) -> Result<()> {
        let mut state = IndexState::new();

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
        self.state.store(Arc::new(state));
        Ok(())
    }

    fn contains_hash(&self, hash: u64) -> bool {
        self.state.load().content_hashes.contains_key(&hash)
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

    /// Two removals of one entry race on a shared index: exactly one removes
    /// it, and no other entry is lost. Each round starts both threads on a
    /// barrier so their lookups and updates overlap.
    #[test]
    fn concurrent_removals_of_one_entry_remove_it_once() {
        for round in 0..2000 {
            let index = HnswIndex::new(2);
            for (n, id) in ["c0", "c1", "c2"].into_iter().enumerate() {
                let hash = u64::try_from(n).unwrap();
                index
                    .insert(
                        Chunk::new(id, id, "doc", n, hash),
                        vec![1.0, f32::from(u8::try_from(n).unwrap())],
                    )
                    .unwrap();
            }

            let barrier = std::sync::Barrier::new(2);
            let removed = std::thread::scope(|scope| {
                let racers: Vec<_> = (0..2)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            index.remove("c0")
                        })
                    })
                    .collect();
                racers
                    .into_iter()
                    .map(|racer| racer.join().expect("racer"))
                    .filter(|removed| *removed)
                    .count()
            });
            assert_eq!(removed, 1, "round {round}: one removal wins");

            let mut left: Vec<_> = index.entries().into_iter().map(|e| e.chunk.id).collect();
            left.sort();
            assert_eq!(left, ["c1", "c2"], "round {round}");
        }
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
