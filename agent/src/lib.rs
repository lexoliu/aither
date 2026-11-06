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
//! - **Ergonomic API**: Simple `query()` for one-shot tasks, `run()` to handle each event as it happens
//! - **Internal Loop**: Tool execution handled automatically
//! - **Trait-based Hooks**: Intercept operations at compile time
//! - **Context Management**: One policy, a cache-aware window, async compaction
//! - **Generic over LLM**: Works with any `LanguageModel` implementation

// Core modules
mod agent;
pub mod ask_user;
mod browser_backend;
mod builder;
mod cache_stats;
#[cfg(test)]
mod call_identity_tests;
mod checkpoint;
mod clock;
mod command_catalog;
#[cfg(test)]
mod command_table_tests;
mod compact_tool;
mod compaction_state;
mod config;
mod connectivity;
pub mod context;
mod context_window;
mod error;
mod event;
pub mod external_hook;
mod fork;
mod fs_util;
mod hook;
mod host_load;
#[cfg(test)]
mod idle_compaction_tests;
#[cfg(feature = "languages")]
pub mod language;
mod model_group;
mod output_urls;
#[cfg(test)]
mod poll_park_tests;
mod pressure;
mod request_parameters;
mod runaway;
mod subagent_file;
mod terminal_agent;
#[cfg(test)]
mod test_support;
pub mod tool_request;
mod tools;
mod warmth;
pub mod working_docs;
pub mod workspace_request;

// Specialized agents
pub mod specialized;

// Subagent definitions
pub use subagent_file::{
    BUNDLED_SUBAGENTS, BundledSubagent, DefinitionError, LOCAL_SUBAGENTS_DIR,
    RESERVED_SUBAGENT_NAMES, ResolveError, SubagentAccess, SubagentDefinition,
    install_bundled_subagents, validate_subagent_name,
};

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
pub use agent::{Agent, BackgroundWake, CompactResult, Run};
pub use browser_backend::{BrowserBackend, BrowserBackendError, StaticCdpBrowser};
pub use builder::{AgentBuilder, AgentOptions, AgentSettings};
pub use cache_stats::CacheStats;
pub use checkpoint::AgentCheckpoint;
pub use clock::{AnyClock, Clock, Instant, ManualClock, SystemClock};
pub use compact_tool::{CompactArgs, CompactSignal, CompactTool};
pub use config::{AgentConfig, AgentKind, BackgroundJobPolicy};
#[cfg(feature = "skills")]
pub use context::ActiveSkills;
pub use context::{
    AnyContextSource, BackgroundJobs, Block, CompactionError, CompletedCompaction, Context,
    ContextBudget, ContextCheckpoint, ContextItem, ContextPolicy, ContextSource, ExternalContent,
    ExternalSlot, Handoff, HandoffDocument, InstructionSlot, LiveContent, LiveUpdate, PrefixBlock,
    PrefixChain, PrefixRewrites, PromptCache, PromptPrefix, ReassemblyReport, SystemLayout,
    TasksDocument,
};
pub use context_window::{ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot};
pub use error::AgentError;
pub use event::{AgentEvent, TimedEvent};
pub use external_hook::{
    ExternalHooks, HookTestTool, HooksConfig, LifecycleHookConfig, LiveHooks, ReviewedPermission,
    ReviewerConfig,
};
pub use fork::{ForkSnapshot, ForkSource};
pub use hook::{
    CheckpointContext, CheckpointReason, HCons, Hook, PostToolAction, PreToolAction, StopContext,
    StopReason, ToolResultContext, ToolUseContext, TurnBoundaryAction, TurnBoundaryContext,
};
pub use host_load::{
    AnyHostReader, HostLoadConfig, HostLoadEvent, HostLoadMonitor, HostReader, HostSample,
    MachineLoad, MemoryPressure, ReadError, SystemReader,
};
#[cfg(feature = "languages")]
pub use language::{Language, LanguageGuard, LanguageRefusal, Languages, WorkingText};
pub use output_urls::{extract_referenced_urls, is_valid_output_filename};
pub use runaway::RunawayGuard;
pub use terminal_agent::{Prepared, TerminalAgentBuilder, Unprepared};
pub use tools::AgentTools;
pub use warmth::{CacheState, CacheWarmth};
pub use working_docs::{AnyTaskBlockerResolver, TaskBlockerResolver};

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
