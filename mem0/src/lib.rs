use std::str::FromStr;
use std::sync::Arc;

use aither_core::embedding::EmbeddingModel;
use aither_core::llm::{LLMRequest, LanguageModel, Message, Tool, ToolOutput};
use anyhow::Context;
use llm::{Action, ExtractedFacts, MemoryDecision};
use store::{MemoryStore, SearchFilters};
use tracing::debug;
use uuid::Uuid;

pub mod error;
pub mod llm;
pub mod store;

pub use error::{Mem0Error, Result};
pub use store::{InMemoryStore, Memory, SearchResult};

pub struct SearchTool<L, E, S> {
    inner: Mem0<L, E, S>,
}

impl<L, E, S> Tool for SearchTool<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    type Arguments = String;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "search_memories".into()
    }

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let result = self
            .inner
            .retrieve_formatted(&arguments, 50)
            .await
            .context("Fail to retrive memory")?;
        Ok(ToolOutput::text(result))
    }
}

pub struct AddFactTool<L, E, S> {
    inner: Mem0<L, E, S>,
}

impl<L, E, S> Tool for AddFactTool<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    type Arguments = Vec<String>;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "add_fact".into()
    }

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        self.inner
            .add_fact(arguments)
            .await
            .context("Fail to add fact")?;
        Ok(ToolOutput::Done)
    }
}

/// Configuration for Mem0.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of similar memories to retrieve for update context.
    pub retrieve_count: usize,
    /// User ID to associate with memories.
    pub user_id: Option<String>,
    /// Agent ID to associate with memories.
    pub agent_id: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            retrieve_count: 5,
            user_id: None,
            agent_id: None,
        }
    }
}

/// Mem0 memory manager.
struct Mem0Inner<L, E, S> {
    new_facts: async_lock::RwLock<Vec<String>>, // Store new facts temporarily
    extraction_in_progress: async_lock::Mutex<()>, // If this is locked, fact extraction is in progress
    llm: L,
    embedder: async_lock::Mutex<E>,
    store: async_lock::RwLock<S>,
    config: Config,
}

pub struct Mem0<L, E, S> {
    inner: Arc<Mem0Inner<L, E, S>>,
}

impl<L, E, S> Clone for Mem0<L, E, S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<L, E, S> Mem0<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    /// Create a new Mem0 instance.
    pub fn new(llm: L, embedder: E, store: S, config: Config) -> Self {
        Self {
            inner: Arc::new(Mem0Inner {
                new_facts: async_lock::RwLock::new(Vec::new()),
                llm,
                embedder: async_lock::Mutex::new(embedder),
                store: async_lock::RwLock::new(store),
                config,
                extraction_in_progress: async_lock::Mutex::new(()),
            }),
        }
    }

    /// Add a new interaction to memory.
    ///
    /// This triggers the extraction and update pipeline:
    /// 1. Extract facts from the messages.
    /// 2. For each fact, retrieve similar memories.
    /// 3. Decide on an operation (Add, Update, Delete, Noop).
    /// 4. Execute the operation.
    pub async fn add(&self, messages: &[Message]) -> Result<()> {
        // 1. Extract facts
        let facts = self.extract_facts(messages).await?;

        self.add_fact(facts).await?;

        Ok(())
    }

    /// Add new facts to memory.
    /// Tip: This method batches fact additions for efficiency and accuracy.
    /// Put it simply, only one fact extraction task is running at a time. Facts added at this moment will be queued and processed together.
    /// And the caller have to wait if you `.await` this method.
    ///
    /// So if you doesn't mind the result of adding facts, you can spawn a task to call this method.
    pub async fn add_fact(&self, facts: Vec<String>) -> Result<()> {
        self.inner.new_facts.write_blocking().extend(facts); // very fast operation

        // Waiting for any ongoing extraction to finish
        let lock = self.inner.extraction_in_progress.lock().await;
        // let's take all facts yet

        let facts = {
            let mut nf = self.inner.new_facts.write_blocking();
            std::mem::take(&mut *nf)
        };

        let store = &self.inner.store;

        for fact in facts {
            // 2. Embed the fact for search
            let embedding = self
                .inner
                .embedder
                .lock()
                .await
                .embed(&fact)
                .await
                .map_err(Mem0Error::Embedding)?;

            debug!("Embedding generated for fact: {}", fact);

            // 3. Retrieve similar memories
            let filters = SearchFilters {
                user_id: self.inner.config.user_id.clone(),
                agent_id: self.inner.config.agent_id.clone(),
            };
            let existing_memories = store
                .read_blocking()
                .search(&embedding, self.inner.config.retrieve_count, filters)
                .await?;

            debug!(
                "Found {} similar existing memories for fact.",
                existing_memories.len()
            );

            // 4. Decide operation
            let decision = self.decide_operation(&fact, &existing_memories).await?;

            // 5. Execute operation
            match decision.action {
                Action::Add => {
                    let mut memory = Memory::new(fact, embedding);
                    if let Some(uid) = &self.inner.config.user_id {
                        memory = memory.with_user_id(uid);
                    }
                    if let Some(aid) = &self.inner.config.agent_id {
                        memory = memory.with_agent_id(aid);
                    }

                    store.write_blocking().add(memory).await?;
                }
                Action::Update => {
                    if let (Some(id_str), Some(content)) =
                        (decision.memory_id, decision.new_content)
                    {
                        if let Ok(id) = Uuid::from_str(&id_str) {
                            // For update, we usually re-embed the new content.
                            let new_embedding = self
                                .inner
                                .embedder
                                .lock()
                                .await
                                .embed(&content)
                                .await
                                .map_err(Mem0Error::Llm)?;

                            // Fetch existing to check existence
                            if let Some(mut existing) = store.read_blocking().get(id).await? {
                                existing.content = content;
                                existing.embedding = new_embedding;
                                existing.updated_at = time::OffsetDateTime::now_utc();
                                store.write_blocking().update(existing).await?;
                            }
                        }
                    }
                }
                Action::Delete => {
                    if let Some(id_str) = decision.memory_id {
                        if let Ok(id) = Uuid::from_str(&id_str) {
                            store.write_blocking().delete(id).await?;
                        }
                    }
                }
                Action::Noop => {} // No operation needed
            }
        }

        drop(lock); // release the lock

        Ok(())
    }

    /// Search for relevant memories.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<store::SearchResult>> {
        let embedding = self
            .inner
            .embedder
            .lock()
            .await
            .embed(query)
            .await
            .map_err(Mem0Error::Llm)?;
        let filters = SearchFilters {
            user_id: self.inner.config.user_id.clone(),
            agent_id: self.inner.config.agent_id.clone(),
        };
        self.inner
            .store
            .read_blocking()
            .search(&embedding, limit, filters)
            .await
    }

    pub fn add_fact_tool(&self) -> AddFactTool<L, E, S> {
        AddFactTool {
            inner: self.clone(),
        }
    }

    pub fn search_tool(&self) -> SearchTool<L, E, S> {
        SearchTool {
            inner: self.clone(),
        }
    }

    /// Return all stored memories.
    pub async fn memories(&self) -> Result<Vec<Memory>> {
        self.inner.store.read_blocking().all().await
    }

    /// Retrieve relevant memories and format them for context injection.
    pub async fn retrieve_formatted(&self, query: &str, limit: usize) -> Result<String> {
        let results = self.search(query, limit).await?;
        if results.is_empty() {
            return Ok(String::new());
        }

        let formatted = results
            .into_iter()
            .map(|r| format!("- {}", r.memory.content))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!("Relevant Memories:\n{}", formatted))
    }

    async fn extract_facts(&self, messages: &[Message]) -> Result<Vec<String>> {
        // Format messages for the prompt
        let context = messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role(), m.content()))
            .collect::<Vec<_>>()
            .join("\n");

        let system_prompt = include_str!("../prompts/extractor.txt");

        let request = LLMRequest::new(vec![
            Message::system(system_prompt),
            Message::user(format!(
                "Extract facts from the following conversation:\n\n{}",
                context
            )),
        ]);

        let extracted: ExtractedFacts = self
            .inner
            .llm
            .generate(request)
            .await
            .map_err(Mem0Error::Llm)?;
        Ok(extracted.facts)
    }

    async fn decide_operation(
        &self,
        fact: &str,
        existing_memories: &[store::SearchResult],
    ) -> Result<MemoryDecision> {
        let memories_context = existing_memories
            .iter()
            .map(|r| format!("ID: {}\nContent: {}\n", r.memory.id, r.memory.content))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let system_prompt = include_str!("../prompts/manager.txt");

        let user_prompt = format!(
            "New Fact: {}\n\nExisting Memories:\n{}\n\nDecide the operation.",
            fact, memories_context
        );

        let request = LLMRequest::new(vec![
            Message::system(system_prompt),
            Message::user(user_prompt),
        ]);

        debug!("Deciding operation for fact: {}", fact);

        let decision: MemoryDecision = self
            .inner
            .llm
            .generate(request)
            .await
            .map_err(Mem0Error::Llm)?;

        Ok(decision)
    }
}
