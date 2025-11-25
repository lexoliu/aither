//! Terminal-based AI assistant backed by Gemini with RAG + Mem0.
//!
//! Features:
//! - Streamed responses from `gemini-2.0-flash`.
//! - Native Google Search + Code Execution enabled on every turn.
//! - `/file <path>` to index a local file into the RAG store.
//! - `/mem0` to inspect stored long-term memories.
//! - `/new` to clear the short-term conversation window.
//! - Press `Esc` at any time to quit.
//!
//! Requires `GEMINI_API_KEY` in your environment.
//! Run: `cargo run --example assistant`

use std::{
    fs,
    io::{self, Write},
    path::Path,
    sync::Arc,
    time::Duration,
};

use aither::llm::{LLMRequest, model::ReasoningEffort, tool::Tools};
use aither_core::{
    LanguageModel,
    llm::{Message, model::Parameters},
};
use aither_gemini::GeminiBackend;
use aither_mem0::{Config as Mem0Config, InMemoryStore, Mem0, Memory, SearchResult};
use aither_rag::{Document, Metadata, RagStore};
use anyhow::{Context, Result, anyhow};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use futures_lite::{StreamExt, pin};
use tracing_subscriber::EnvFilter;

type Backend = GeminiBackend;
type MemoryManager = Mem0<Backend, Backend, InMemoryStore>;
type Store = RagStore<Backend>;

const MODEL: &str = "gemini-2.5-pro";
const RAG_TOP_K: usize = 3;
const MAX_HISTORY_MESSAGES: usize = 20;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let api_key =
        std::env::var("GEMINI_API_KEY").context("set GEMINI_API_KEY in your environment")?;

    let gemini = GeminiBackend::new(api_key).with_text_model(MODEL);
    let embedder = gemini.clone();

    let mem0 = Mem0::new(
        gemini.clone().with_text_model("gemini-flash-lite-latest"), // super fast model for Mem0
        embedder.clone(),
        InMemoryStore::new(),
        Mem0Config::default(),
    );
    let rag = RagStore::new(embedder);

    run_loop(gemini, mem0, rag).await
}

async fn run_loop(gemini: Backend, mem0: MemoryManager, rag: Store) -> Result<()> {
    let mut short_history: Vec<Message> = Vec::new();
    println!("Terminal assistant ready. Type messages, /file <path>, /mem0, /new, or Esc to exit.");

    loop {
        let Some(input) = read_line("You> ")? else {
            break;
        };
        let user = input.trim();
        if user.is_empty() {
            continue;
        }

        if user.starts_with("/file ") {
            let path = user.trim_start_matches("/file").trim();
            let added = ingest_file(path, &rag).await?;
            println!("Indexed {added} chunk(s) from {path}");
            continue;
        }

        if user == "/mem0" {
            print_memories(&mem0).await?;
            continue;
        }

        if user == "/new" {
            short_history.clear();
            println!("Started a fresh conversation window (long-term memory retained).");
            continue;
        }

        let rag_hits = match rag.query(user, RAG_TOP_K).await {
            Ok(hits) => hits,
            Err(err) => {
                eprintln!("RAG lookup failed: {err}");
                Vec::new()
            }
        };

        let mut request_messages = vec![Message::system(system_prompt())];

        if let Some(rag_section) = format_rag_hits(&rag_hits) {
            request_messages.push(Message::system(rag_section));
        }
        request_messages.extend(short_history.clone());
        request_messages.push(Message::user(user));

        let params = Parameters::default().reasoning_effort(ReasoningEffort::Low);
        let mut tools = Tools::new();
        tools.register(mem0.add_fact_tool());
        tools.register(mem0.search_tool());
        let request = LLMRequest::new(request_messages)
            .with_parameters(params)
            .with_tools(&mut tools);

        println!("Gemini>");
        let mut response = String::new();
        let stream = gemini.respond(request);
        pin!(stream);
        while let Some(chunk) = stream.next().await {
            let text = chunk?;
            print!("{text}");
            io::stdout().flush().ok();
            response.push_str(&text);
        }
        println!();

        short_history.push(Message::user(user));
        short_history.push(Message::assistant(&response));
        if short_history.len() > MAX_HISTORY_MESSAGES {
            let start = short_history.len() - MAX_HISTORY_MESSAGES;
            short_history = short_history[start..].to_vec();
        }

        let mem_msgs = vec![Message::user(user), Message::assistant(&response)];
        let mem0_for_task = mem0.clone();
        let _ = tokio::spawn(async move {
            if let Err(err) = mem0_for_task.add(&mem_msgs).await {
                if !should_quiet_mem0_error(&err) {
                    eprintln!("Mem0 update failed: {err}");
                }
            }
        });
    }

    println!("Goodbye!");
    Ok(())
}

fn system_prompt() -> String {
    "You are a helpful terminal assistant. Use provided memories, retrieved documents, native web search, \
and code execution to ground your replies. Cite file paths when quoting retrieved context. Keep answers concise \
by default unless asked otherwise. Do not mention tool you use. You can use `add_fact` to remember important information, and `search_memories` to recall them later."
        .to_string()
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut truncated = String::with_capacity(limit + 3);
    for ch in text.chars().take(limit) {
        truncated.push(ch);
    }
    truncated.push_str("...");
    truncated
}

fn format_memories(hits: &[SearchResult]) -> Option<String> {
    if hits.is_empty() {
        return None;
    }
    let rows: Vec<String> = hits
        .iter()
        .map(|hit| {
            let content = truncate(&hit.memory.content, 220);
            format!("- {:.2}: {}", hit.score, content)
        })
        .collect();
    Some(format!("Relevant memories:\n{}", rows.join("\n")))
}

fn format_rag_hits(hits: &[aither_rag::RetrievedDocument]) -> Option<String> {
    if hits.is_empty() {
        return None;
    }
    let rows: Vec<String> = hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| {
            let origin = hit
                .document
                .metadata
                .get("path")
                .cloned()
                .unwrap_or_else(|| hit.document.id.clone());
            format!(
                "{}. [{}] {} :: {}",
                idx + 1,
                format!("{:.2}", hit.score),
                origin,
                truncate(&hit.document.text, 240)
            )
        })
        .collect();
    Some(format!("Retrieved context:\n{}", rows.join("\n")))
}

async fn print_memories(mem0: &MemoryManager) -> Result<()> {
    let entries = mem0.memories().await?;
    if entries.is_empty() {
        println!("No stored memories yet.");
        return Ok(());
    }

    println!("Stored memories:");
    for Memory {
        content,
        user_id,
        agent_id,
        ..
    } in entries
    {
        let who = user_id
            .as_deref()
            .or_else(|| agent_id.as_deref())
            .unwrap_or("unknown");
        println!("- {} ({who})", content.trim());
    }
    Ok(())
}

async fn ingest_file(path: &str, rag: &Store) -> Result<usize> {
    let file_path = Path::new(path);
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("failed to read file at {}", file_path.display()))?;

    let chunks = chunk_text(&content, 1200);
    if chunks.is_empty() {
        return Err(anyhow!("no content found in {}", file_path.display()));
    }

    for (idx, chunk) in chunks.iter().enumerate() {
        let mut metadata: Metadata = Metadata::new();
        metadata.insert("path".into(), file_path.display().to_string());
        metadata.insert("chunk".into(), (idx + 1).to_string());
        let id = format!("{}#{}", file_path.display(), idx + 1);
        let doc = Document::with_metadata(id, chunk.clone(), metadata);
        rag.insert(doc).await?;
    }

    Ok(chunks.len())
}

fn chunk_text(text: &str, size: usize) -> Vec<String> {
    text.as_bytes()
        .chunks(size)
        .map(|chunk| String::from_utf8_lossy(chunk).to_string())
        .collect()
}

fn should_quiet_mem0_error(err: &dyn std::fmt::Display) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("missing candidate") || msg.contains("blocked")
}

fn read_line(prompt: &str) -> Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;
    enable_raw_mode()?;
    let _guard = RawModeGuard;

    let mut buffer = String::new();
    loop {
        if event::poll(Duration::from_millis(250))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Esc => {
                        print!("\r\n");
                        io::stdout().flush().ok();
                        return Ok(None);
                    }
                    KeyCode::Enter => {
                        print!("\r\n");
                        io::stdout().flush().ok();
                        return Ok(Some(buffer));
                    }
                    KeyCode::Backspace => {
                        if buffer.pop().is_some() {
                            print!("\u{8} \u{8}");
                            io::stdout().flush().ok();
                        }
                    }
                    KeyCode::Char(c) => {
                        buffer.push(c);
                        print!("{c}");
                        io::stdout().flush().ok();
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
}

struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}
