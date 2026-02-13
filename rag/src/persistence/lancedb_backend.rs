//! LanceDB-based persistence backend.

use std::path::{Path, PathBuf};

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::ExecutableQuery;

use crate::error::{RagError, Result};
use crate::types::{Chunk, IndexEntry, Metadata};

use super::Persistence;

const TABLE_NAME: &str = "rag_entries";

/// LanceDB persistence backend.
#[derive(Debug)]
pub struct LanceDbPersistence {
    path: PathBuf,
}

impl LanceDbPersistence {
    /// Creates a new LanceDB backend rooted at the given directory.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    fn schema(embedding_dim: i32) -> Schema {
        Schema::new(vec![
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("chunk_text", DataType::Utf8, false),
            Field::new("chunk_source_id", DataType::Utf8, false),
            Field::new("chunk_index", DataType::UInt32, false),
            Field::new("chunk_content_hash", DataType::UInt64, false),
            Field::new("chunk_metadata", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
                    embedding_dim,
                ),
                false,
            ),
        ])
    }

    fn block_on<F, T>(&self, fut: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            Ok(tokio::task::block_in_place(|| handle.block_on(fut))?)
        } else {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| RagError::Database(e.to_string()))?;
            rt.block_on(fut)
        }
    }

    fn make_batches(
        entries: &[IndexEntry],
    ) -> Result<Box<dyn arrow_array::RecordBatchReader + Send>> {
        let embedding_dim = entries
            .first()
            .map_or(1_i32, |e| i32::try_from(e.embedding.len()).unwrap_or(1));

        let schema = std::sync::Arc::new(Self::schema(embedding_dim));

        let chunk_id = StringArray::from_iter_values(entries.iter().map(|e| e.chunk.id.as_str()));
        let chunk_text =
            StringArray::from_iter_values(entries.iter().map(|e| e.chunk.text.as_str()));
        let chunk_source_id =
            StringArray::from_iter_values(entries.iter().map(|e| e.chunk.source_id.as_str()));
        let chunk_index =
            UInt32Array::from_iter_values(entries.iter().map(|e| e.chunk.index as u32));
        let chunk_content_hash =
            UInt64Array::from_iter_values(entries.iter().map(|e| e.chunk.content_hash));
        let chunk_metadata = StringArray::from_iter_values(entries.iter().map(|e| {
            serde_json::to_string(&e.chunk.metadata).unwrap_or_else(|_| "{}".to_string())
        }));

        let embedding = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            entries
                .iter()
                .map(|e| Some(e.embedding.iter().copied().map(Some))),
            embedding_dim,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(chunk_id),
                std::sync::Arc::new(chunk_text),
                std::sync::Arc::new(chunk_source_id),
                std::sync::Arc::new(chunk_index),
                std::sync::Arc::new(chunk_content_hash),
                std::sync::Arc::new(chunk_metadata),
                std::sync::Arc::new(embedding),
            ],
        )
        .map_err(|e| RagError::Serialization(e.to_string()))?;

        Ok(Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema)))
    }
}

impl Persistence for LanceDbPersistence {
    fn save(&self, entries: &[IndexEntry]) -> Result<()> {
        self.block_on(async {
            let uri = self.path.to_string_lossy().to_string();
            let db = connect(&uri)
                .execute()
                .await
                .map_err(|e| RagError::Database(e.to_string()))?;

            let names = db
                .table_names()
                .execute()
                .await
                .map_err(|e| RagError::Database(e.to_string()))?;
            if names.iter().any(|n| n == TABLE_NAME) {
                db.drop_table(TABLE_NAME, &[])
                    .await
                    .map_err(|e| RagError::Database(e.to_string()))?;
            }

            if entries.is_empty() {
                let schema = std::sync::Arc::new(Self::schema(1));
                db.create_empty_table(TABLE_NAME, schema)
                    .execute()
                    .await
                    .map_err(|e| RagError::Database(e.to_string()))?;
                return Ok(());
            }

            let batches = Self::make_batches(entries)?;
            db.create_table(TABLE_NAME, batches)
                .execute()
                .await
                .map_err(|e| RagError::Database(e.to_string()))?;

            Ok(())
        })
    }

    fn load(&self) -> Result<Vec<IndexEntry>> {
        self.block_on(async {
            let uri = self.path.to_string_lossy().to_string();
            let db = connect(&uri)
                .execute()
                .await
                .map_err(|e| RagError::Database(e.to_string()))?;

            let names = db
                .table_names()
                .execute()
                .await
                .map_err(|e| RagError::Database(e.to_string()))?;
            if !names.iter().any(|n| n == TABLE_NAME) {
                return Ok(Vec::new());
            }

            let table = db
                .open_table(TABLE_NAME)
                .execute()
                .await
                .map_err(|e| RagError::Database(e.to_string()))?;

            let mut stream = table
                .query()
                .execute()
                .await
                .map_err(|e| RagError::Database(e.to_string()))?;

            let mut out = Vec::new();
            while let Some(batch_res) = stream.next().await {
                let batch = batch_res.map_err(|e| RagError::Database(e.to_string()))?;

                let ids = batch
                    .column_by_name("chunk_id")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| RagError::Serialization("missing chunk_id".to_string()))?;
                let texts = batch
                    .column_by_name("chunk_text")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| RagError::Serialization("missing chunk_text".to_string()))?;
                let source_ids = batch
                    .column_by_name("chunk_source_id")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| {
                        RagError::Serialization("missing chunk_source_id".to_string())
                    })?;
                let indices = batch
                    .column_by_name("chunk_index")
                    .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
                    .ok_or_else(|| RagError::Serialization("missing chunk_index".to_string()))?;
                let hashes = batch
                    .column_by_name("chunk_content_hash")
                    .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                    .ok_or_else(|| {
                        RagError::Serialization("missing chunk_content_hash".to_string())
                    })?;
                let metadata_json = batch
                    .column_by_name("chunk_metadata")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| RagError::Serialization("missing chunk_metadata".to_string()))?;
                let embedding_col = batch
                    .column_by_name("embedding")
                    .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                    .ok_or_else(|| RagError::Serialization("missing embedding".to_string()))?;

                for i in 0..batch.num_rows() {
                    let embedding_values = embedding_col.value(i);
                    let float_values = embedding_values
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| {
                            RagError::Serialization("invalid embedding type".to_string())
                        })?;

                    let embedding = (0..float_values.len())
                        .map(|j| float_values.value(j))
                        .collect::<Vec<_>>();

                    let metadata: Metadata = serde_json::from_str(metadata_json.value(i))
                        .map_err(|e| RagError::Serialization(e.to_string()))?;

                    let chunk = Chunk::with_metadata(
                        ids.value(i).to_string(),
                        texts.value(i).to_string(),
                        source_ids.value(i).to_string(),
                        indices.value(i) as usize,
                        hashes.value(i),
                        metadata,
                    );

                    out.push(IndexEntry::new(chunk, embedding));
                }
            }

            Ok(out)
        })
    }

    fn extension(&self) -> &'static str {
        "lancedb"
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, text: &str) -> IndexEntry {
        let chunk = Chunk::new(id, text, "doc1", 0, crate::dedup::content_hash(text));
        IndexEntry::new(chunk, vec![1.0, 2.0, 3.0, 4.0])
    }

    #[test]
    fn save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let persistence = LanceDbPersistence::new(dir.path().join("index.lancedb"));

        let entries = vec![make_entry("c1", "hello"), make_entry("c2", "world")];
        persistence.save(&entries).unwrap();

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 2);
    }
}
