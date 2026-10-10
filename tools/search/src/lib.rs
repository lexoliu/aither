//! Search tool backed by `LanceDB` plus optional semantic embeddings.
//!
//! The tool indexes text/code files into chunks and supports text, full-text,
//! fuzzy, regex, hybrid, and semantic retrieval over the indexed workspace.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aither_core::llm::ToolContext;
use aither_core::{
    EmbeddingModel,
    llm::{Tool, ToolResult},
};
use aither_rag::{Document, Rag};
use arrow_array::{
    Array, ArrayRef, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures_lite::StreamExt;
use ignore::WalkBuilder;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::{
    Index,
    scalar::{BTreeIndexBuilder, InvertedIndexParams},
};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Table, connect};
use regex::RegexBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shell_words::ParseError as ShellParseError;
use sqlparser::ast::{BinaryOperator, Expr as SqlExpr, Ident, Value as SqlValue};

const TABLE_NAME: &str = "file_chunks";
const INDEX_META_FILE: &str = "search_index.meta.json";
const MAX_FILE_BYTES: usize = 1024 * 1024;
const CHUNK_TARGET_CHARS: usize = 2400;
const CHUNK_OVERLAP_CHARS: usize = 240;

/// Error returned when a shell-like search command cannot be split.
#[derive(Debug, thiserror::Error)]
#[error("{label} command contains invalid shell quoting: {source}")]
pub struct SplitShellWordsError {
    label: String,
    #[source]
    source: ShellParseError,
}

fn split_shell_words(input: &str, label: &str) -> Result<Vec<String>, SplitShellWordsError> {
    shell_words::split(input).map_err(|source| SplitShellWordsError {
        label: label.to_string(),
        source,
    })
}

fn push_unique<T: Eq>(items: &mut Vec<T>, value: T) {
    if !items.iter().any(|existing| existing == &value) {
        items.push(value);
    }
}

/// Embedding model interface required by semantic search.
pub trait SearchEmbeddingModel: EmbeddingModel + Clone {
    /// Stable selector describing the embedding backend and model.
    fn selector(&self) -> &str;
}

/// Runtime state of the optional embedding model used for semantic search.
#[derive(Debug, Clone)]
pub enum SearchEmbeddingResolution<E> {
    /// Semantic search is disabled.
    Disabled,
    /// Semantic search failed to initialize.
    Failed(String),
    /// Semantic search is ready.
    Ready(E),
}

impl<E> SearchEmbeddingResolution<E>
where
    E: SearchEmbeddingModel,
{
    #[must_use]
    /// Returns the ready embedding model when semantic search is available.
    pub const fn ready(&self) -> Option<&E> {
        match self {
            Self::Ready(model) => Some(model),
            Self::Disabled | Self::Failed(_) => None,
        }
    }

    #[must_use]
    /// Clones the ready embedding model when semantic search is available.
    pub fn clone_ready(&self) -> Option<E> {
        self.ready().cloned()
    }
}

/// Errors returned by the search tool.
#[derive(Debug, thiserror::Error)]
pub enum SearchToolError {
    /// The index command did not include a path.
    #[error("search index requires a non-empty path")]
    MissingIndexPath,
    /// The query command did not include a query.
    #[error("search query requires a non-empty query")]
    MissingQuery,
    /// The command string is empty.
    #[error("search command must not be empty")]
    EmptyCommand,
    /// The command string has no action or query body.
    #[error("search command must include an action or query")]
    MissingActionOrQuery,
    /// The index command received too many path arguments.
    #[error("search index accepts exactly one path argument")]
    InvalidIndexCommandArity,
    /// The query command received no prompt.
    #[error("search query requires a prompt")]
    MissingQueryPrompt,
    /// Failed to resolve a filesystem path.
    #[error("failed to resolve search path {path}: {source}")]
    ResolvePath {
        /// Path that failed to resolve.
        path: PathBuf,
        #[source]
        /// Source I/O error.
        source: std::io::Error,
    },
    /// The indexer found no text chunks.
    #[error("search index contains no text chunks; choose a text or code file/directory")]
    IndexContainsNoText,
    /// Failed to load index metadata.
    #[error("failed to read search index metadata {path}: {source}")]
    LoadIndexMetadata {
        /// Metadata path.
        path: PathBuf,
        #[source]
        /// Source I/O error.
        source: std::io::Error,
    },
    /// Failed to parse index metadata.
    #[error("failed to parse search index metadata {path}: {source}")]
    ParseIndexMetadata {
        /// Metadata path.
        path: PathBuf,
        #[source]
        /// Source JSON error.
        source: serde_json::Error,
    },
    /// Failed to remove a stale semantic index.
    #[error("failed to remove stale semantic index {path}: {source}")]
    RemoveStaleSemanticIndex {
        /// Semantic index path.
        path: PathBuf,
        #[source]
        /// Source I/O error.
        source: std::io::Error,
    },
    /// The metadata records no indexed roots.
    #[error("search index is empty; run `search index <path>` first")]
    EmptyIndex,
    /// The `LanceDB` table is missing.
    #[error("search index is missing; run `search index <path>` again")]
    MissingIndexTable,
    /// No embedding model is available.
    #[error(
        "semantic search requires a global models.embedding_model and at least one enabled provider that supports it"
    )]
    SemanticEmbeddingUnavailable,
    /// The semantic index was built with another embedding configuration.
    #[error(
        "semantic index was built for a different embedding configuration; run `search index <path>` again"
    )]
    SemanticEmbeddingMismatch,
    /// Regex compilation failed.
    #[error("invalid regex pattern: {0}")]
    InvalidRegexPattern(#[source] regex::Error),
    /// Semantic search cannot execute wildcard queries.
    #[error("semantic search does not support wildcard queries")]
    SemanticWildcardUnsupported,
    /// The semantic index file is missing.
    #[error("semantic index is missing; run `search index <path>` again")]
    MissingSemanticIndex,
    /// Failed to inspect a semantic index.
    #[error("failed to inspect semantic index {path}: {source}")]
    InspectSemanticIndex {
        /// Semantic index path.
        path: PathBuf,
        #[source]
        /// Source I/O error.
        source: std::io::Error,
    },
    /// Required semantic chunk metadata is missing.
    #[error("semantic index entry is missing '{key}' metadata")]
    MissingSemanticMetadata {
        /// Missing metadata key.
        key: &'static str,
    },
    /// A LanceDB/Arrow column has an unexpected type.
    #[error("missing or invalid '{name}' column; expected {expected}")]
    InvalidColumn {
        /// Column name.
        name: &'static str,
        /// Expected Arrow type.
        expected: &'static str,
    },
    /// Walking files failed.
    #[error("failed to walk searchable files under {root}: {source}")]
    WalkFiles {
        /// Walk root.
        root: PathBuf,
        #[source]
        /// Source walk error.
        source: ignore::Error,
    },
    /// Joining the file scan task failed.
    #[error("failed to join searchable file scan task: {0}")]
    JoinFileScan(#[source] tokio::task::JoinError),
    /// Filesystem I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// `LanceDB` failed.
    #[error("LanceDB error: {0}")]
    LanceDb(#[from] lancedb::Error),
    /// Arrow schema or batch construction failed.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// RAG semantic indexing failed.
    #[error("RAG error: {0}")]
    Rag(#[from] aither_rag::error::RagError),
    /// Lance full-text search failed outside `LanceDB`'s error type.
    #[error("Lance error: {0}")]
    Lance(String),
}

/// Indexing or query action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchAction {
    /// Build or rebuild the search index.
    Index,
    /// Query the existing search index.
    Query,
}

/// Search strategy used for querying the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Merge text, full-text, and fuzzy matches.
    Hybrid,
    /// Plain substring text search.
    Text,
    /// Regex search.
    Regex,
    /// Lance full-text search.
    Fulltext,
    /// Fuzzy full-text search.
    Fuzzy,
    /// Embedding-backed semantic search.
    Semantic,
}

/// Shape of search results returned to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchOutputMode {
    /// Collapse matches to file-level results.
    Files,
    /// Return individual matching chunks.
    Chunks,
}

/// Arguments accepted by the search tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// Shell-like command form, such as `search index src` or `search query foo`.
    #[serde(default)]
    pub command: Option<String>,
    /// Action to perform.
    #[serde(default = "default_action")]
    pub action: SearchAction,
    /// Path filter or path to index.
    #[serde(default)]
    pub path: Option<String>,
    /// Query text.
    #[serde(default)]
    pub query: Option<String>,
    /// Search strategy.
    #[serde(default = "default_search_mode")]
    pub search_mode: SearchMode,
    /// Output shape.
    #[serde(default = "default_output_mode")]
    pub output_mode: SearchOutputMode,
    /// Maximum number of results to return.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Number of results to skip.
    #[serde(default)]
    pub offset: usize,
    /// Maximum candidate rows to scan.
    #[serde(default = "default_scan_limit")]
    pub scan_limit: usize,
    /// Whether text and regex matching is case-sensitive.
    #[serde(default)]
    pub case_sensitive: bool,
    /// Regex pattern override.
    #[serde(default)]
    pub regex: Option<String>,
    /// Maximum fuzzy edit distance.
    #[serde(default = "default_fuzzy_distance")]
    pub fuzzy_distance: u32,
    /// Include matched chunk content in file-level results.
    #[serde(default)]
    pub include_content: bool,
}

const fn default_action() -> SearchAction {
    SearchAction::Query
}

const fn default_search_mode() -> SearchMode {
    SearchMode::Hybrid
}

const fn default_output_mode() -> SearchOutputMode {
    SearchOutputMode::Files
}

const fn default_limit() -> usize {
    20
}

const fn default_scan_limit() -> usize {
    2000
}

const fn default_fuzzy_distance() -> u32 {
    1
}

/// Search tool implementation.
#[derive(Debug, Clone)]
pub struct SearchTool<E>
where
    E: SearchEmbeddingModel + 'static,
{
    sandbox_dir: PathBuf,
    embedder: SearchEmbeddingResolution<E>,
}

impl<E> SearchTool<E>
where
    E: SearchEmbeddingModel + 'static,
{
    /// Create a search tool rooted in the sandbox data directory.
    #[must_use]
    pub const fn new(sandbox_dir: PathBuf, embedder: SearchEmbeddingResolution<E>) -> Self {
        Self {
            sandbox_dir,
            embedder,
        }
    }

    fn index_dir(&self) -> PathBuf {
        self.sandbox_dir.join("data").join("search")
    }

    fn db_path(&self) -> PathBuf {
        self.index_dir().join("index.lancedb")
    }

    fn meta_path(&self) -> PathBuf {
        self.index_dir().join(INDEX_META_FILE)
    }

    fn semantic_index_path(&self) -> PathBuf {
        self.index_dir().join("semantic.redb")
    }

    fn normalize_args(mut args: SearchArgs) -> Result<SearchArgs, SearchToolError> {
        if let Some(command) = args.command.take() {
            apply_command_form(&mut args, &command)?;
        }

        args.limit = args.limit.clamp(1, 200);
        args.scan_limit = args.scan_limit.clamp(args.limit, 50_000);
        args.fuzzy_distance = args.fuzzy_distance.min(3);

        if args.action == SearchAction::Index {
            let Some(path) = args
                .path
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return Err(SearchToolError::MissingIndexPath);
            };
            args.path = Some(path.to_string());
        } else {
            let Some(query) = args
                .query
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return Err(SearchToolError::MissingQuery);
            };
            args.query = Some(query.to_string());
        }

        Ok(args)
    }
}

pub(crate) async fn resolve_search_path(
    sandbox_dir: &Path,
    raw_path: &str,
) -> Result<PathBuf, SearchToolError> {
    let requested_path = PathBuf::from(raw_path);
    let resolved_path = if requested_path.is_absolute() {
        requested_path
    } else {
        sandbox_dir.join(requested_path)
    };
    async_fs::canonicalize(&resolved_path)
        .await
        .map_err(|source| SearchToolError::ResolvePath {
            path: resolved_path,
            source,
        })
}

impl<E> Tool for SearchTool<E>
where
    E: SearchEmbeddingModel + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("search")
    }

    type Arguments = SearchArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let result = async {
            let mut args = Self::normalize_args(arguments)?;
            if let Some(path) = args.path.as_deref() {
                args.path = Some(
                    resolve_search_path(self.sandbox_dir.as_path(), path)
                        .await?
                        .display()
                        .to_string(),
                );
            }
            if args.action == SearchAction::Index {
                return index_path(
                    self.sandbox_dir.clone(),
                    self.db_path(),
                    self.meta_path(),
                    self.semantic_index_path(),
                    self.embedder.clone(),
                    args.path.expect("validated"),
                )
                .await;
            }
            query_index(
                self.db_path(),
                self.meta_path(),
                self.semantic_index_path(),
                self.embedder.clone(),
                args,
            )
            .await
        }
        .await;

        match result {
            Ok(result) => Ok(result),
            Err(error) => Ok(ToolResult::error(error.to_string())),
        }
    }
}

/// Apply shell-like command text to structured search arguments.
///
/// # Errors
///
/// Returns an error when the command is empty, malformed, or contains invalid
/// shell quoting.
pub fn apply_command_form(args: &mut SearchArgs, command: &str) -> Result<(), SearchToolError> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err(SearchToolError::EmptyCommand);
    }
    let body = trimmed.strip_prefix("search ").unwrap_or(trimmed).trim();
    if body.is_empty() {
        return Err(SearchToolError::MissingActionOrQuery);
    }

    let tokens =
        split_shell_words(body, "search").map_err(|_| SearchToolError::MissingActionOrQuery)?;
    match tokens.as_slice() {
        [action, path] if action == "index" => {
            args.action = SearchAction::Index;
            args.path = Some(path.clone());
        }
        [action, ..] if action == "index" => {
            return Err(SearchToolError::InvalidIndexCommandArity);
        }
        [action, prompt_tokens @ ..] if action == "query" => {
            if prompt_tokens.is_empty() {
                return Err(SearchToolError::MissingQueryPrompt);
            }
            args.action = SearchAction::Query;
            args.query = Some(prompt_tokens.join(" "));
        }
        _ => {
            args.action = SearchAction::Query;
            args.query = Some(tokens.join(" "));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SearchIndexMeta {
    roots: Vec<String>,
    files_indexed: usize,
    chunks_indexed: usize,
    indexed_at: String,
    semantic_embedding_model: Option<String>,
    semantic_embedding_dimensions: Option<usize>,
}

#[derive(Debug, Clone)]
struct FileChunkRow {
    root_path: String,
    file_path: String,
    relative_path: String,
    file_name: String,
    modified_at: String,
    chunk_id: i64,
    content: String,
}

#[derive(Debug, Clone)]
struct SearchableFile {
    root_path: String,
    file_path: String,
    relative_path: String,
    file_name: String,
    modified_at: String,
    content: String,
}

#[derive(Debug, Clone)]
struct FileMatchRow {
    file_path: String,
    relative_path: String,
    modified_at: String,
    chunk_id: i64,
    content: String,
    score: f64,
    matched_by: Vec<SearchMode>,
}

#[derive(Debug, Clone, Serialize)]
struct ChunkResult {
    file_path: String,
    relative_path: String,
    modified_at: String,
    chunk_id: i64,
    score: f64,
    matched_by: Vec<SearchMode>,
    snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FileResult {
    file_path: String,
    relative_path: String,
    modified_at: String,
    match_count: usize,
    top_score: f64,
    highlights: Vec<ChunkResult>,
}

async fn index_path<E>(
    sandbox_dir: PathBuf,
    db_path: PathBuf,
    meta_path: PathBuf,
    semantic_index_path: PathBuf,
    embedder: SearchEmbeddingResolution<E>,
    raw_path: String,
) -> Result<ToolResult, SearchToolError>
where
    E: SearchEmbeddingModel + 'static,
{
    let canonical = resolve_search_path(sandbox_dir.as_path(), raw_path.as_str()).await?;

    let mut meta = load_meta(&meta_path).await?;
    let canonical_string = canonical.display().to_string();
    if !meta.roots.iter().any(|root| root == &canonical_string) {
        meta.roots.push(canonical_string.clone());
        meta.roots.sort_unstable();
    }

    let files = collect_searchable_files(&meta.roots).await?;
    let rows = build_rows(&files);
    if rows.is_empty() {
        return Err(SearchToolError::IndexContainsNoText);
    }

    async_fs::create_dir_all(
        meta_path
            .parent()
            .expect("search metadata path must have a parent"),
    )
    .await?;

    let conn = connect(db_path.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table_names = conn.table_names().execute().await?;
    if table_names.iter().any(|name| name == TABLE_NAME) {
        conn.drop_table(TABLE_NAME, &[]).await?;
    }

    let reader = rows_to_record_reader(&rows)?;
    let table = conn.create_table(TABLE_NAME, reader).execute().await?;
    for column in ["content", "relative_path", "file_name"] {
        table
            .create_index(&[column], Index::FTS(InvertedIndexParams::default()))
            .execute()
            .await?;
    }
    table
        .create_index(&["file_path"], Index::BTree(BTreeIndexBuilder::default()))
        .execute()
        .await?;

    meta.files_indexed = rows
        .iter()
        .map(|row| row.file_path.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    meta.chunks_indexed = rows.len();
    meta.indexed_at = jiff::Timestamp::now().to_string();
    meta.semantic_embedding_model = embedder.ready().map(|value| value.selector().to_string());
    meta.semantic_embedding_dimensions = embedder.ready().map(aither_core::EmbeddingModel::dim);
    async_fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?).await?;

    let semantic_indexed = if let Some(embedder) = embedder.clone_ready() {
        rebuild_semantic_index(&semantic_index_path, embedder, &files).await?
    } else {
        false
    };

    Ok(ToolResult::json_value(serde_json::json!({
        "action": "index",
        "indexed_root": canonical_string,
        "roots": meta.roots,
        "files_indexed": meta.files_indexed,
        "chunks_indexed": meta.chunks_indexed,
        "semantic_indexed": semantic_indexed,
        "indexed_at": meta.indexed_at,
    })))
}

async fn load_meta(meta_path: &Path) -> Result<SearchIndexMeta, SearchToolError> {
    match async_fs::read_to_string(meta_path).await {
        Ok(content) => {
            serde_json::from_str(&content).map_err(|source| SearchToolError::ParseIndexMetadata {
                path: meta_path.to_path_buf(),
                source,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(SearchIndexMeta::default())
        }
        Err(source) => Err(SearchToolError::LoadIndexMetadata {
            path: meta_path.to_path_buf(),
            source,
        }),
    }
}

async fn rebuild_semantic_index<E>(
    semantic_index_path: &Path,
    embedder: E,
    files: &[SearchableFile],
) -> Result<bool, SearchToolError>
where
    E: SearchEmbeddingModel + 'static,
{
    match async_fs::remove_file(semantic_index_path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(SearchToolError::RemoveStaleSemanticIndex {
                path: semantic_index_path.to_path_buf(),
                source,
            });
        }
    }

    let rag = Rag::builder(embedder)
        .index_path(semantic_index_path.to_path_buf())
        .code_chunking(CHUNK_TARGET_CHARS)
        .top_k(200)
        .deduplication(false)
        .auto_save(false)
        .build()?;

    let mut indexed_chunks = 0usize;
    for file in files {
        let mut metadata = aither_rag::Metadata::new();
        metadata.insert("path".to_string(), file.file_path.clone());
        metadata.insert("root_path".to_string(), file.root_path.clone());
        metadata.insert("file_path".to_string(), file.file_path.clone());
        metadata.insert("relative_path".to_string(), file.relative_path.clone());
        metadata.insert("modified_at".to_string(), file.modified_at.clone());
        metadata.insert("file_name".to_string(), file.file_name.clone());

        indexed_chunks += rag
            .insert(Document::with_metadata(
                file.file_path.clone(),
                file.content.clone(),
                metadata,
            ))
            .await?;
    }

    rag.save().await?;
    Ok(indexed_chunks > 0)
}

async fn query_index<E>(
    db_path: PathBuf,
    meta_path: PathBuf,
    semantic_index_path: PathBuf,
    embedder: SearchEmbeddingResolution<E>,
    args: SearchArgs,
) -> Result<ToolResult, SearchToolError>
where
    E: SearchEmbeddingModel + 'static,
{
    let meta = load_meta(&meta_path).await?;
    if meta.roots.is_empty() {
        return Err(SearchToolError::EmptyIndex);
    }

    let conn = connect(db_path.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = conn
        .open_table(TABLE_NAME)
        .execute()
        .await
        .map_err(|_| SearchToolError::MissingIndexTable)?;

    let query = args.query.clone().expect("validated");
    let query_enabled = query.trim() != "*" && !query.trim().is_empty();
    if args.search_mode == SearchMode::Semantic {
        let embedder = embedder
            .clone_ready()
            .ok_or(SearchToolError::SemanticEmbeddingUnavailable)?;
        if meta.semantic_embedding_model.as_deref() != Some(embedder.selector())
            || meta.semantic_embedding_dimensions != Some(embedder.dim())
        {
            return Err(SearchToolError::SemanticEmbeddingMismatch);
        }
        return run_semantic_query(semantic_index_path, embedder, args, meta).await;
    }

    let mut candidates = collect_query_candidates(&table, &args, query_enabled).await?;
    let regex = build_query_regex(&args, &query, query_enabled)?;
    score_query_candidates(
        &mut candidates,
        &args,
        &query,
        regex.as_ref(),
        query_enabled,
    );
    let rows = dedupe_and_sort_rows(candidates);
    let total_matches = rows.len();
    Ok(finalize_rows(&rows, total_matches, &args, &meta))
}

async fn collect_query_candidates(
    table: &Table,
    args: &SearchArgs,
    query_enabled: bool,
) -> Result<Vec<FileMatchRow>, SearchToolError> {
    if matches!(args.search_mode, SearchMode::Regex | SearchMode::Text)
        || (args.search_mode == SearchMode::Hybrid && !query_enabled)
    {
        return run_plain_query(table, args).await;
    }
    match args.search_mode {
        SearchMode::Fulltext => run_full_text_query(table, args, false).await,
        SearchMode::Fuzzy => run_full_text_query(table, args, true).await,
        SearchMode::Hybrid => {
            let ((mut fts, mut fuzzy), mut plain) = futures_lite::future::try_zip(
                futures_lite::future::try_zip(
                    run_full_text_query(table, args, false),
                    run_full_text_query(table, args, true),
                ),
                run_plain_query(table, args),
            )
            .await?;
            let mut merged = Vec::with_capacity(fts.len() + fuzzy.len() + plain.len());
            merged.append(&mut fts);
            merged.append(&mut fuzzy);
            merged.append(&mut plain);
            Ok(merged)
        }
        SearchMode::Regex | SearchMode::Text => unreachable!("plain search returns earlier"),
        SearchMode::Semantic => unreachable!("semantic search returns earlier"),
    }
}

fn build_query_regex(
    args: &SearchArgs,
    query: &str,
    query_enabled: bool,
) -> Result<Option<regex::Regex>, SearchToolError> {
    if args.search_mode != SearchMode::Regex || !(query_enabled || args.regex.is_some()) {
        return Ok(None);
    }
    let pattern = args.regex.as_deref().unwrap_or(query).to_string();
    RegexBuilder::new(&pattern)
        .case_insensitive(!args.case_sensitive)
        .build()
        .map(Some)
        .map_err(SearchToolError::InvalidRegexPattern)
}

fn score_query_candidates(
    candidates: &mut Vec<FileMatchRow>,
    args: &SearchArgs,
    query: &str,
    regex: Option<&regex::Regex>,
    query_enabled: bool,
) {
    let query_norm = if args.case_sensitive {
        Cow::Borrowed(query)
    } else {
        Cow::Owned(query.to_lowercase())
    };
    candidates.retain_mut(|row| {
        score_and_match_query_candidate(row, args, &query_norm, regex, query_enabled)
    });
}

fn score_and_match_query_candidate(
    row: &mut FileMatchRow,
    args: &SearchArgs,
    query_norm: &str,
    regex: Option<&regex::Regex>,
    query_enabled: bool,
) -> bool {
    if row.matched_by.is_empty() {
        push_unique(&mut row.matched_by, args.search_mode);
    }
    let haystack = [row.relative_path.as_str(), row.content.as_str()].join("\n");
    let haystack_norm = if args.case_sensitive {
        Cow::Borrowed(haystack.as_str())
    } else {
        Cow::Owned(haystack.to_lowercase())
    };
    if query_enabled
        && !matches!(args.search_mode, SearchMode::Fulltext | SearchMode::Fuzzy)
        && haystack_norm.contains(query_norm)
    {
        row.score += 10.0;
        push_unique(&mut row.matched_by, SearchMode::Text);
    }
    if let Some(re) = regex
        && re.is_match(&haystack)
    {
        row.score += 15.0;
        push_unique(&mut row.matched_by, SearchMode::Regex);
    }
    if query_enabled && args.search_mode == SearchMode::Text {
        return haystack_norm.contains(query_norm);
    }
    regex.is_none_or(|re| !matches!(args.search_mode, SearchMode::Regex) || re.is_match(&haystack))
}

fn dedupe_and_sort_rows(candidates: Vec<FileMatchRow>) -> Vec<FileMatchRow> {
    let mut deduped: HashMap<String, FileMatchRow> = HashMap::new();
    for row in candidates {
        let key = format!("{}:{}", row.file_path, row.chunk_id);
        match deduped.get_mut(&key) {
            Some(existing) if row.score > existing.score => *existing = row,
            Some(_) => {}
            None => {
                deduped.insert(key, row);
            }
        }
    }
    let mut rows: Vec<FileMatchRow> = deduped.into_values().collect();
    rows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.file_path.cmp(&b.file_path))
    });
    rows
}

fn require_semantic_metadata<'a>(
    metadata: &'a aither_rag::Metadata,
    key: &'static str,
) -> Result<&'a str, SearchToolError> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or(SearchToolError::MissingSemanticMetadata { key })
}

fn matches_path_prefix(path_filter: Option<&str>, file_path: &str) -> bool {
    path_filter.is_none_or(|prefix| file_path.starts_with(prefix))
}

async fn run_semantic_query<E>(
    semantic_index_path: PathBuf,
    embedder: E,
    args: SearchArgs,
    meta: SearchIndexMeta,
) -> Result<ToolResult, SearchToolError>
where
    E: SearchEmbeddingModel + 'static,
{
    let query = args.query.as_deref().expect("validated");
    if query == "*" {
        return Err(SearchToolError::SemanticWildcardUnsupported);
    }

    match async_fs::metadata(&semantic_index_path).await {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(SearchToolError::MissingSemanticIndex);
        }
        Err(source) => {
            return Err(SearchToolError::InspectSemanticIndex {
                path: semantic_index_path,
                source,
            });
        }
    }

    let rag = Rag::builder(embedder)
        .index_path(semantic_index_path)
        .code_chunking(CHUNK_TARGET_CHARS)
        .top_k(args.scan_limit)
        .deduplication(false)
        .auto_save(false)
        .build()?;
    rag.load().await?;

    let path_filter = args
        .path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let rows = rag
        .search_with_k(query, args.scan_limit)
        .await?
        .into_iter()
        .map(|result| -> Result<Option<FileMatchRow>, SearchToolError> {
            let file_path = require_semantic_metadata(&result.chunk.metadata, "file_path")?;
            if !matches_path_prefix(path_filter, file_path) {
                return Ok(None);
            }

            let relative_path = require_semantic_metadata(&result.chunk.metadata, "relative_path")?;
            let modified_at = require_semantic_metadata(&result.chunk.metadata, "modified_at")?;

            Ok(Some(FileMatchRow {
                file_path: file_path.to_string(),
                relative_path: relative_path.to_string(),
                modified_at: modified_at.to_string(),
                chunk_id: i64::try_from(result.chunk.index).map_err(|_| {
                    SearchToolError::Lance("semantic chunk index exceeds i64".into())
                })?,
                content: result.chunk.text,
                score: f64::from(result.score),
                matched_by: vec![SearchMode::Semantic],
            }))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let total_matches = rows.len();
    Ok(finalize_rows(&rows, total_matches, &args, &meta))
}

fn build_sql_filter(args: &SearchArgs) -> Option<String> {
    let mut filters = Vec::new();
    if let Some(path) = args
        .path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        filters.push(SqlExpr::Like {
            negated: false,
            any: false,
            expr: Box::new(identifier_sql_expr("file_path")),
            pattern: Box::new(string_sql_expr(prefix_match_pattern(path))),
            escape_char: None,
        });
    }
    filters
        .into_iter()
        .reduce(|left, right| SqlExpr::BinaryOp {
            left: Box::new(left),
            op: BinaryOperator::And,
            right: Box::new(right),
        })
        .map(|expr| expr.to_string())
}

fn identifier_sql_expr(column: &str) -> SqlExpr {
    SqlExpr::Identifier(Ident::new(column))
}

fn string_sql_expr(value: String) -> SqlExpr {
    SqlExpr::value(SqlValue::SingleQuotedString(value))
}

fn prefix_match_pattern(value: &str) -> String {
    let mut pattern = String::with_capacity(value.len() + 1);
    pattern.push_str(value);
    pattern.push('%');
    pattern
}

async fn run_plain_query(
    table: &Table,
    args: &SearchArgs,
) -> Result<Vec<FileMatchRow>, SearchToolError> {
    let mut query = table
        .query()
        .select(Select::columns(&[
            "file_path",
            "relative_path",
            "modified_at",
            "chunk_id",
            "content",
        ]))
        .limit(args.scan_limit);

    if let Some(filter) = build_sql_filter(args) {
        query = query.only_if(filter);
    }

    let mut stream = query.execute().await?;
    read_rows_from_stream(&mut stream).await
}

async fn run_full_text_query(
    table: &Table,
    args: &SearchArgs,
    fuzzy: bool,
) -> Result<Vec<FileMatchRow>, SearchToolError> {
    let query_text = args.query.clone().expect("validated");
    if query_text.trim() == "*" || query_text.trim().is_empty() {
        return run_plain_query(table, args).await;
    }

    let fts = if fuzzy {
        FullTextSearchQuery::new_fuzzy(query_text, Some(args.fuzzy_distance))
    } else {
        FullTextSearchQuery::new(query_text)
    }
    .with_columns(&[
        "content".to_string(),
        "relative_path".to_string(),
        "file_name".to_string(),
    ])
    .map_err(|error| SearchToolError::Lance(error.to_string()))?;

    let mut query = table
        .query()
        .full_text_search(fts)
        .select(Select::columns(&[
            "file_path",
            "relative_path",
            "modified_at",
            "chunk_id",
            "content",
            "_score",
        ]))
        .limit(args.scan_limit);

    if let Some(filter) = build_sql_filter(args) {
        query = query.only_if(filter);
    }

    let mut stream = query.execute().await?;
    let mut rows = read_rows_from_stream(&mut stream).await?;
    for row in &mut rows {
        push_unique(
            &mut row.matched_by,
            if fuzzy {
                SearchMode::Fuzzy
            } else {
                SearchMode::Fulltext
            },
        );
    }
    Ok(rows)
}

async fn read_rows_from_stream(
    stream: &mut lancedb::arrow::SendableRecordBatchStream,
) -> Result<Vec<FileMatchRow>, SearchToolError> {
    let mut rows = Vec::new();
    while let Some(batch_res) = stream.next().await {
        let batch = batch_res?;
        let file_path = as_string(batch.column_by_name("file_path"), "file_path")?;
        let relative_path = as_string(batch.column_by_name("relative_path"), "relative_path")?;
        let modified_at = as_string(batch.column_by_name("modified_at"), "modified_at")?;
        let chunk_id = as_i64(batch.column_by_name("chunk_id"), "chunk_id")?;
        let content = as_string(batch.column_by_name("content"), "content")?;

        let score_col = batch.column_by_name("_score");
        let scores = score_col
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
            .map_or_else(
                || vec![0.0; batch.num_rows()],
                |array| {
                    (0..array.len())
                        .map(|idx| {
                            if array.is_null(idx) {
                                0.0
                            } else {
                                f64::from(array.value(idx))
                            }
                        })
                        .collect::<Vec<_>>()
                },
            );

        for (idx, score) in scores.iter().copied().enumerate().take(batch.num_rows()) {
            rows.push(FileMatchRow {
                file_path: file_path.value(idx).to_string(),
                relative_path: relative_path.value(idx).to_string(),
                modified_at: modified_at.value(idx).to_string(),
                chunk_id: chunk_id.value(idx),
                content: content.value(idx).to_string(),
                score,
                matched_by: Vec::new(),
            });
        }
    }
    Ok(rows)
}

fn as_string<'a>(
    column: Option<&'a ArrayRef>,
    name: &'static str,
) -> Result<&'a StringArray, SearchToolError> {
    column
        .and_then(|value| value.as_any().downcast_ref::<StringArray>())
        .ok_or(SearchToolError::InvalidColumn {
            name,
            expected: "utf8",
        })
}

fn as_i64<'a>(
    column: Option<&'a ArrayRef>,
    name: &'static str,
) -> Result<&'a Int64Array, SearchToolError> {
    column
        .and_then(|value| value.as_any().downcast_ref::<Int64Array>())
        .ok_or(SearchToolError::InvalidColumn {
            name,
            expected: "int64",
        })
}

async fn collect_searchable_files(
    roots: &[String],
) -> Result<Vec<SearchableFile>, SearchToolError> {
    let mut files_out = Vec::new();
    for root in roots {
        let root_path = PathBuf::from(root);
        if async_fs::metadata(&root_path).await?.is_file() {
            if let Some(file) =
                read_searchable_file(root_path.as_path(), root_path.as_path()).await?
            {
                files_out.push(file);
            }
            continue;
        }

        let root_path_clone = root_path.clone();
        let files =
            tokio::task::spawn_blocking(move || -> Result<Vec<PathBuf>, SearchToolError> {
                let mut files = Vec::new();
                let mut walk = WalkBuilder::new(&root_path_clone);
                walk.standard_filters(true);
                walk.hidden(false);
                for entry in walk.build() {
                    let entry = entry.map_err(|source| SearchToolError::WalkFiles {
                        root: root_path_clone.clone(),
                        source,
                    })?;
                    if entry.file_type().is_some_and(|kind| kind.is_file()) {
                        files.push(entry.into_path());
                    }
                }
                Ok(files)
            })
            .await
            .map_err(SearchToolError::JoinFileScan)??;

        for file in files {
            if let Some(searchable) =
                read_searchable_file(root_path.as_path(), file.as_path()).await?
            {
                files_out.push(searchable);
            }
        }
    }
    Ok(files_out)
}

fn build_rows(files: &[SearchableFile]) -> Vec<FileChunkRow> {
    files
        .iter()
        .flat_map(|file| {
            chunk_text(file.content.as_str())
                .into_iter()
                .enumerate()
                .map(|(idx, chunk)| FileChunkRow {
                    root_path: file.root_path.clone(),
                    file_path: file.file_path.clone(),
                    relative_path: file.relative_path.clone(),
                    file_name: file.file_name.clone(),
                    modified_at: file.modified_at.clone(),
                    chunk_id: i64::try_from(idx)
                        .expect("chunk index should fit into i64 for indexed file"),
                    content: chunk,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn read_searchable_file(
    root_path: &Path,
    file_path: &Path,
) -> Result<Option<SearchableFile>, SearchToolError> {
    let bytes = async_fs::read(file_path).await?;
    if bytes.len() > MAX_FILE_BYTES || is_probably_binary(&bytes) {
        return Ok(None);
    }

    let content = String::from_utf8_lossy(&bytes).to_string();
    if content.trim().is_empty() {
        return Ok(None);
    }

    let metadata = async_fs::metadata(file_path).await?;
    let modified_at = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_default();
    let relative_path = file_path
        .strip_prefix(root_path)
        .ok()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(file_path)
        .display()
        .to_string();
    let file_name = file_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(relative_path.as_str())
        .to_string();

    Ok(Some(SearchableFile {
        root_path: root_path.display().to_string(),
        file_path: file_path.display().to_string(),
        relative_path,
        file_name,
        modified_at,
        content,
    }))
}

/// Split text into overlapping chunks used by the indexer.
#[must_use]
pub fn chunk_text(content: &str) -> Vec<String> {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= CHUNK_TARGET_CHARS {
        return vec![content.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + CHUNK_TARGET_CHARS).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(CHUNK_OVERLAP_CHARS);
    }
    chunks
}

fn is_probably_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(4096).any(|byte| *byte == 0)
}

fn rows_to_record_reader(
    rows: &[FileChunkRow],
) -> Result<Box<dyn RecordBatchReader + Send>, SearchToolError> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("root_path", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("relative_path", DataType::Utf8, false),
        Field::new("file_name", DataType::Utf8, false),
        Field::new("modified_at", DataType::Utf8, false),
        Field::new("chunk_id", DataType::Int64, false),
        Field::new("content", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.root_path.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.file_path.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.relative_path.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.file_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.modified_at.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.chunk_id),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.content.as_str()),
            )),
        ],
    )?;

    Ok(Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema)))
}

fn finalize_rows(
    rows: &[FileMatchRow],
    total_matches: usize,
    args: &SearchArgs,
    meta: &SearchIndexMeta,
) -> ToolResult {
    let start = args.offset.min(rows.len());
    let end = (start + args.limit).min(rows.len());
    let rows = &rows[start..end];

    let query = args.query.clone().expect("validated");
    let value = if args.output_mode == SearchOutputMode::Chunks {
        let items = rows
            .iter()
            .map(|row| ChunkResult {
                file_path: row.file_path.clone(),
                relative_path: row.relative_path.clone(),
                modified_at: row.modified_at.clone(),
                chunk_id: row.chunk_id,
                score: (row.score * 100.0).round() / 100.0,
                matched_by: row.matched_by.clone(),
                snippet: truncate_text(row.content.as_str(), 240),
                content: args.include_content.then_some(row.content.clone()),
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "action": "query",
            "query": query,
            "search_mode": args.search_mode,
            "output_mode": args.output_mode,
            "indexed_roots": meta.roots,
            "total_matches": total_matches,
            "returned": items.len(),
            "results": items,
        })
    } else {
        let mut grouped = HashMap::<String, FileResult>::new();
        for row in rows {
            let entry = grouped
                .entry(row.file_path.clone())
                .or_insert_with(|| FileResult {
                    file_path: row.file_path.clone(),
                    relative_path: row.relative_path.clone(),
                    modified_at: row.modified_at.clone(),
                    match_count: 0,
                    top_score: row.score,
                    highlights: Vec::new(),
                });
            entry.match_count += 1;
            if row.score > entry.top_score {
                entry.top_score = row.score;
            }
            if entry.highlights.len() < 3 {
                entry.highlights.push(ChunkResult {
                    file_path: row.file_path.clone(),
                    relative_path: row.relative_path.clone(),
                    modified_at: row.modified_at.clone(),
                    chunk_id: row.chunk_id,
                    score: (row.score * 100.0).round() / 100.0,
                    matched_by: row.matched_by.clone(),
                    snippet: truncate_text(row.content.as_str(), 240),
                    content: args.include_content.then_some(row.content.clone()),
                });
            }
        }
        let mut items = grouped.into_values().collect::<Vec<_>>();
        items.sort_by(|left, right| {
            right
                .top_score
                .partial_cmp(&left.top_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.file_path.cmp(&right.file_path))
        });
        serde_json::json!({
            "action": "query",
            "query": query,
            "search_mode": args.search_mode,
            "output_mode": args.output_mode,
            "indexed_roots": meta.roots,
            "total_matches": total_matches,
            "returned": items.len(),
            "results": items,
        })
    };

    ToolResult::json_value(value)
}

fn truncate_text(input: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for (idx, ch) in input.chars().enumerate() {
        if idx >= max_chars {
            output.push_str("...");
            break;
        }
        output.push(ch);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct NoopEmbedding;

    impl EmbeddingModel for NoopEmbedding {
        fn dim(&self) -> usize {
            3
        }

        fn embed(
            &self,
            _text: &str,
        ) -> impl std::future::Future<Output = aither_core::Result<Vec<f32>>> + Send {
            std::future::ready(Ok(vec![0.0, 0.0, 0.0]))
        }
    }

    impl SearchEmbeddingModel for NoopEmbedding {
        fn selector(&self) -> &'static str {
            "noop@embedding"
        }
    }

    fn base_args() -> SearchArgs {
        SearchArgs {
            command: None,
            action: SearchAction::Query,
            path: None,
            query: Some("hello".to_string()),
            search_mode: SearchMode::Hybrid,
            output_mode: SearchOutputMode::Files,
            limit: 20,
            offset: 0,
            scan_limit: 2000,
            case_sensitive: false,
            regex: None,
            fuzzy_distance: 1,
            include_content: false,
        }
    }

    #[test]
    fn parses_index_command_form() {
        let mut args = base_args();
        apply_command_form(&mut args, "search index ./docs").expect("index command should parse");
        assert_eq!(args.action, SearchAction::Index);
        assert_eq!(args.path.as_deref(), Some("./docs"));
    }

    #[test]
    fn parses_query_command_form() {
        let mut args = base_args();
        apply_command_form(&mut args, "search query concurrency bug")
            .expect("query command should parse");
        assert_eq!(args.action, SearchAction::Query);
        assert_eq!(args.query.as_deref(), Some("concurrency bug"));
    }

    #[test]
    fn parses_quoted_index_command_form() {
        let mut args = base_args();
        apply_command_form(&mut args, "search index \"./docs with spaces\"")
            .expect("quoted index command should parse");
        assert_eq!(args.action, SearchAction::Index);
        assert_eq!(args.path.as_deref(), Some("./docs with spaces"));
    }

    #[test]
    fn parses_quoted_query_command_form() {
        let mut args = base_args();
        apply_command_form(&mut args, "search query \"concurrency bug\"")
            .expect("quoted query command should parse");
        assert_eq!(args.action, SearchAction::Query);
        assert_eq!(args.query.as_deref(), Some("concurrency bug"));
    }

    #[test]
    fn chunks_long_text() {
        let content = "a".repeat(6000);
        let chunks = chunk_text(content.as_str());
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[test]
    fn search_mode_serializes_as_lowercase_string() {
        let value = serde_json::to_value(vec![SearchMode::Fulltext, SearchMode::Regex])
            .expect("search modes should serialize");
        assert_eq!(value, serde_json::json!(["fulltext", "regex"]));
    }

    #[tokio::test]
    async fn resolves_relative_query_path_against_sandbox() {
        let temp = tempfile::tempdir().expect("tempdir");
        let sandbox_dir = temp.path().join("session");
        let indexed_dir = sandbox_dir.join("data").join("semantic-test");
        std::fs::create_dir_all(&indexed_dir).expect("create indexed dir");
        std::fs::write(indexed_dir.join("note.md"), "hello").expect("write note");

        let resolved = resolve_search_path(sandbox_dir.as_path(), "data/semantic-test")
            .await
            .expect("resolve relative path");

        assert_eq!(
            resolved,
            std::fs::canonicalize(&indexed_dir).expect("canonical indexed dir")
        );
    }

    #[test]
    fn resolution_reports_ready_model() {
        let ready = SearchEmbeddingResolution::Ready(NoopEmbedding);
        assert!(ready.ready().is_some());
        assert!(ready.clone_ready().is_some());
    }
}
