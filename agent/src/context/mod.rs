//! Context management: one window, one policy, one budget.
//!
//! Context is the agent's scarcest resource and the one it cannot measure
//! directly, so this module is organised around the few facts that are
//! actually knowable and derives everything else from them:
//!
//! - [`window`] owns the layout, and guarantees it grows only at the end so a
//!   provider's prefix cache survives the conversation growing;
//! - [`budget`] turns the provider's own token counts into a usage figure,
//!   instead of trusting a character heuristic that is wrong by a different
//!   factor for prose, code and tool arguments;
//! - [`policy`] holds every threshold there is — when to compact, when to
//!   wait for it, when to rebuild, how much may be restored;
//! - [`cache`] translates the layout into per-provider cache controls;
//! - [`compaction`] forks a typed handoff at a request boundary and applies
//!   it at the next one, so the run never stops to summarize itself;
//! - [`reassembly`] re-reads what the handoff named, within budget;
//! - [`source`] is how everything ambient — working documents, running jobs,
//!   active skills — reaches the model without the agent loop knowing about
//!   any of it in particular.

pub mod budget;
pub mod cache;
pub mod compaction;
pub mod policy;
pub mod reassembly;
pub mod source;
pub mod sources;
pub mod window;

pub use budget::ContextBudget;
pub use cache::PromptCache;
pub use compaction::{
    CompactionError, CompactionFork, CompletedCompaction, HandoffDocument, PendingCompaction,
    build_file_index,
};
pub use policy::ContextPolicy;
pub use reassembly::restore_context;
pub use source::{AnyContextSource, Block, ContextSource};
#[cfg(feature = "skills")]
pub use sources::ActiveSkills;
pub use sources::{BackgroundJobs, TasksDocument};
pub use window::{Context, ContextCheckpoint, ContextItem, ReassemblyReport, serialize_xml};
