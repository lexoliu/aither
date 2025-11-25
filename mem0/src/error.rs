use thiserror::Error;

#[derive(Debug, Error)]
pub enum Mem0Error {
    #[error("LLM error: {0}")]
    Llm(anyhow::Error),

    #[error("Embedding error: {0}")]
    Embedding(anyhow::Error),

    #[error("Store error: {0}")]
    Store(String),

    #[error("Extraction failed: {0}")]
    Extraction(String),
}

pub type Result<T> = core::result::Result<T, Mem0Error>;
