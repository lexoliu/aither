//! Persistence backends for RAG indexes.
//!
//! This module provides the [`Persistence`] trait and implementations for
//! saving and loading index data.

mod redb_backend;
mod rkyv_backend;

pub use redb_backend::RedbPersistence;
pub use rkyv_backend::RkyvPersistence;

use crate::error::Result;
use crate::types::IndexEntry;
use std::path::Path;

/// Trait for persistence backends.
///
/// Persistence backends handle saving and loading index entries to/from storage.
pub trait Persistence: Send + Sync {
    /// Saves all index entries to storage.
    fn save(&self, entries: &[IndexEntry]) -> Result<()>;

    /// Loads all index entries from storage.
    ///
    /// Returns an empty vector if no data exists.
    fn load(&self) -> Result<Vec<IndexEntry>>;

    /// Returns the file extension used by this backend.
    fn extension(&self) -> &'static str;

    /// Returns the storage path.
    fn path(&self) -> &Path;
}
