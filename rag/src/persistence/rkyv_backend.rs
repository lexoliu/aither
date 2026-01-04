//! rkyv-based binary persistence.

use rkyv::rancor::Error as RkyvError;
use rkyv::{from_bytes, to_bytes};
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{RagError, Result};
use crate::types::IndexEntry;

use super::Persistence;

/// Wrapper for serialization with rkyv.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
struct EntriesWrapper {
    entries: Vec<EntryData>,
}

/// Internal entry data for rkyv serialization.
#[derive(Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
struct EntryData {
    chunk_id: String,
    chunk_text: String,
    chunk_source_id: String,
    chunk_index: u32,
    chunk_content_hash: u64,
    chunk_metadata: Vec<(String, String)>,
    embedding: Vec<f32>,
}

impl From<&IndexEntry> for EntryData {
    fn from(entry: &IndexEntry) -> Self {
        Self {
            chunk_id: entry.chunk.id.clone(),
            chunk_text: entry.chunk.text.clone(),
            chunk_source_id: entry.chunk.source_id.clone(),
            chunk_index: entry.chunk.index as u32,
            chunk_content_hash: entry.chunk.content_hash,
            chunk_metadata: entry
                .chunk
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            embedding: entry.embedding.clone(),
        }
    }
}

impl From<EntryData> for IndexEntry {
    fn from(data: EntryData) -> Self {
        use crate::types::{Chunk, Metadata};

        let metadata: Metadata = data.chunk_metadata.into_iter().collect();
        let chunk = Chunk::with_metadata(
            data.chunk_id,
            data.chunk_text,
            data.chunk_source_id,
            data.chunk_index as usize,
            data.chunk_content_hash,
            metadata,
        );
        IndexEntry::new(chunk, data.embedding)
    }
}

/// Binary persistence using rkyv for fast serialization.
///
/// This backend provides fast serialization and deserialization using
/// the rkyv library.
///
/// # Example
///
/// ```rust,no_run
/// use aither_rag::persistence::{Persistence, RkyvPersistence};
///
/// let persistence = RkyvPersistence::new("./index.rkyv");
/// // persistence.save(&entries)?;
/// // let loaded = persistence.load()?;
/// ```
#[derive(Debug)]
pub struct RkyvPersistence {
    path: PathBuf,
}

impl RkyvPersistence {
    /// Creates a new rkyv persistence backend.
    ///
    /// # Arguments
    /// * `path` - Path to the persistence file
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl Persistence for RkyvPersistence {
    fn save(&self, entries: &[IndexEntry]) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let wrapper = EntriesWrapper {
            entries: entries.iter().map(EntryData::from).collect(),
        };

        let bytes =
            to_bytes::<RkyvError>(&wrapper).map_err(|e| RagError::Serialization(e.to_string()))?;

        fs::write(&self.path, &bytes).map_err(|e| RagError::Persistence {
            path: self.path.clone(),
            source: e,
        })?;

        Ok(())
    }

    fn load(&self) -> Result<Vec<IndexEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let bytes = fs::read(&self.path).map_err(|e| RagError::Persistence {
            path: self.path.clone(),
            source: e,
        })?;

        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        let wrapper = from_bytes::<EntriesWrapper, RkyvError>(&bytes)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        Ok(wrapper.entries.into_iter().map(IndexEntry::from).collect())
    }

    fn extension(&self) -> &'static str {
        "rkyv"
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Chunk;
    use tempfile::tempdir;

    fn make_entry(id: &str, text: &str) -> IndexEntry {
        let chunk = Chunk::new(id, text, "doc1", 0, crate::dedup::content_hash(text));
        IndexEntry::new(chunk, vec![1.0, 2.0, 3.0, 4.0])
    }

    #[test]
    fn save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rkyv");
        let persistence = RkyvPersistence::new(&path);

        let entries = vec![make_entry("c1", "hello"), make_entry("c2", "world")];

        persistence.save(&entries).unwrap();
        assert!(path.exists());

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].chunk.id, "c1");
        assert_eq!(loaded[1].chunk.id, "c2");
    }

    #[test]
    fn load_nonexistent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.rkyv");
        let persistence = RkyvPersistence::new(&path);

        let loaded = persistence.load().unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn save_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.rkyv");
        let persistence = RkyvPersistence::new(&path);

        persistence.save(&[]).unwrap();
        let loaded = persistence.load().unwrap();
        assert!(loaded.is_empty());
    }
}
