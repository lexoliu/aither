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
//! - **Tool Search**: Auto-enabled when many tools are registered
//! - **Smart Compression**: Intelligent context management preserving critical info
//! - **Generic over LLM**: Works with any `LanguageModel` implementation

// Core modules
mod agent;
mod builder;
mod compression;
mod config;
mod context;
mod error;
mod event;
mod hook;
mod model_group;
mod search;
mod stream;
mod todo;
mod tools;

// Specialized agents
pub mod specialized;

// Re-export tool crates
#[cfg(feature = "command")]
pub use aither_command as command;
#[cfg(feature = "filesystem")]
pub use aither_fs as filesystem;
#[cfg(feature = "webfetch")]
pub use aither_webfetch as webfetch;
#[cfg(feature = "websearch")]
pub use aither_websearch as websearch;

// Public API
pub use agent::{Agent, CompactResult};
pub use builder::AgentBuilder;
pub use compression::{
    CompressionLevel, ContextStrategy, PreserveConfig, PreservedContent, SmartCompressionConfig,
};
pub use config::{AgentConfig, SearchStrategy, ToolSearchConfig};
pub use context::{ConversationMemory, MemoryCheckpoint};
pub use error::AgentError;
pub use event::AgentEvent;
pub use hook::{
    HCons, Hook, PostToolAction, PreToolAction, StopContext, StopReason, ToolResultContext,
    ToolUseContext,
};
pub use stream::AgentStream;
pub use todo::{TodoItem, TodoList, TodoStatus, TodoTool, TodoWriteArgs};
pub use tools::AgentTools;

// Model groups for budget tracking and fallback
pub use model_group::{Budget, BudgetedModel, ModelGroup, ModelGroupError, ModelTier, TieredModels};

// Re-export core tool trait for convenience
pub use aither_core::llm::Tool;
