//! Aither Agent Framework
//!
//! An ergonomic, minimal-core agent framework for building autonomous AI agents.
//! The agent handles the tool-use loop internally with a simple API.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use aither_agent::Agent;
//!
//! // Simple one-shot query
//! let agent = Agent::new(llm);
//! let response = agent.query("What is 2+2?").await?;
//!
//! // With tools and hooks
//! let agent = Agent::builder(llm)
//!     .tool(FileSystemTool::read_only("."))
//!     .hook(LoggingHook)
//!     .build();
//!
//! let response = agent.query("List all Rust files").await?;
//! ```
//!
//! # Features
//!
//! - **Ergonomic API**: Simple `query()` for one-shot tasks, `run()` for streaming
//! - **Internal Loop**: Tool execution handled automatically
//! - **Trait-based Hooks**: Intercept operations at compile time
//! - **Smart Compression**: Intelligent context management preserving critical info
//! - **Generic over LLM**: Works with any `LanguageModel` implementation

// Core modules
mod agent;
pub mod ask_user;
mod browser_backend;
mod builder;
mod cache_stats;
mod checkpoint;
mod compression;
mod config;
mod context;
mod context_window;
mod error;
mod event;
mod fs_util;
mod handoff;
mod hook;
mod model_group;
mod stream;
mod subagent_file;
mod terminal_agent;
pub mod tool_request;
mod tools;
pub mod transcript;
pub mod working_docs;
pub mod workspace_request;

// Specialized agents
pub mod specialized;

// File-based subagent definitions
pub use subagent_file::{SubagentDefinition, builtin_subagents};

// Re-export tool crates
#[cfg(feature = "command")]
pub use aither_command as command;
#[cfg(feature = "filesystem")]
pub use aither_fs as filesystem;
#[cfg(feature = "webfetch")]
pub use aither_webfetch as webfetch;
#[cfg(feature = "websearch")]
pub use aither_websearch as websearch;

// Sandbox is always available (terminal-first design)
pub use aither_sandbox as sandbox;

// Public API
pub use agent::{Agent, CompactResult};
pub use browser_backend::{BrowserBackend, BrowserBackendError, StaticCdpBrowser};
pub use builder::AgentBuilder;
pub use cache_stats::CacheStats;
pub use checkpoint::AgentCheckpoint;
pub use compression::{
    CompressionLevel, ContextStrategy, PreserveConfig, PreservedContent, SmartCompressionConfig,
};
pub use config::{AgentConfig, AgentKind, ContextAssemblerConfig};
pub use context::{Context, ContextCheckpoint, ReassemblyReport};
pub use context_window::{ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot};
pub use error::AgentError;
pub use event::AgentEvent;
pub use handoff::HandoffDocument;
pub use hook::{
    CheckpointContext, CheckpointReason, HCons, Hook, PostToolAction, PreToolAction, StopContext,
    StopReason, ToolResultContext, ToolUseContext, TurnBoundaryAction, TurnBoundaryContext,
};
pub use stream::AgentStream;
pub use terminal_agent::TerminalAgentBuilder;
pub use tools::AgentTools;

// Model groups for budget tracking and fallback
pub use model_group::{
    Budget, BudgetedModel, ModelGroup, ModelGroupError, ModelTier, TieredModels,
};

// Re-export core tool trait for convenience
pub use aither_attachments::{CacheEntry, FileCache};
pub use aither_core::llm::Tool;

/// Default system prompt for terminal-first agents.
///
/// This prompt teaches the LLM to use `terminal` as the primary tool interface,
/// with all capabilities exposed as CLI commands that can be piped and composed.
///
/// # Example
///
/// ```rust,ignore
/// use aither_agent::{Agent, TERMINAL_SYSTEM_PROMPT};
///
/// let agent = Agent::builder(llm)
///     .system_prompt(TERMINAL_SYSTEM_PROMPT)
///     .terminal(permission_handler, config, output_store)
///     .build();
/// ```
pub const TERMINAL_SYSTEM_PROMPT: &str = include_str!("prompts/system.md");
