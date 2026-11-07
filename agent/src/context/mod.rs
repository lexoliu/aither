//! Context management: one window, one policy, one budget.
//!
//! Context is the agent's scarcest resource and the one it cannot measure
//! directly, so this module is organised around the few facts that are
//! actually knowable and derives everything else from them:
//!
//! - [`window`] owns the layout, and guarantees it grows only at the end so a
//!   provider's prefix cache survives the conversation growing;
//! - [`system`] orders the system prompt as the host lays it out and keeps
//!   external content inside boundaries it cannot close;
//! - [`budget`] turns the provider's own token counts into a usage figure,
//!   instead of trusting a character heuristic that is wrong by a different
//!   factor for prose, code and tool arguments;
//! - [`policy`] holds every threshold there is — when to compact, when to
//!   wait for it, when to rebuild, how much may be restored;
//! - [`cache`] translates the layout into per-provider cache controls;
//! - [`prefix`] hashes each request's prefix block by block, so a host can
//!   tell where the prefix changed when the cache misses;
//! - [`compaction`] forks a typed handoff at a request boundary and applies
//!   it at the next one, so the run never stops to summarize itself;
//! - [`reassembly`] re-reads what the handoff named, within budget;
//! - [`source`] is how everything ambient — working documents, running jobs,
//!   active skills — reaches the model without the agent loop knowing about
//!   any of it in particular, in the tail or, for live content, in the
//!   cacheable prefix;
//! - [`state`] tells the model of keyed conditions — memory pressure,
//!   machine load — only when their value differs from what it last saw;
//! - [`event`] tells it of each thing that happened once;
//! - [`live`] is how prefix content that changes mid-session — `AGENTS.md`,
//!   the command table — reaches the model as one diff while the prefix is
//!   cached and as its clean version after the next reassembly.

pub mod budget;
pub mod cache;
pub mod compaction;
pub mod event;
pub mod live;
pub mod policy;
pub mod prefix;
pub mod reassembly;
pub(crate) mod recovery;
pub mod source;
pub mod sources;
pub mod state;
pub mod system;
pub mod window;

pub use budget::ContextBudget;
pub use cache::{CacheLifetime, PromptCache};
pub use compaction::{
    CompactionError, CompactionFork, CompletedCompaction, ContextFile, Handoff, HandoffDocument,
    HandoffPrompt, HandoffSchema, InvalidHandoff, LineRange, PendingCompaction, Restoration,
    SkillCatalogEntry, build_file_index,
};
pub use event::{AnyEventSource, EventSource};
pub use live::{LiveContent, LiveUpdate};
pub use policy::ContextPolicy;
pub use prefix::{PrefixBlock, PrefixChain, PrefixRewrites, PromptPrefix};
pub use reassembly::restore_context;
pub use source::{AnyContextSource, AnyLiveSource, Block, ContextSource, LiveChange, LiveSource};
#[cfg(feature = "skills")]
pub use sources::ActiveSkills;
pub use sources::BackgroundJobs;
pub use state::{AnyStateSource, StateChange, StateSource};
pub use system::{
    BlockKind, External, ExternalContent, ExternalSlot, Instruction, InstructionSlot, Slot,
    Sourced, SourcedSlot, SystemLayout, SystemRegion,
};
pub use window::{Context, ContextCheckpoint, ContextItem, ReassemblyReport};
