//! Subagent tool for spawning specialized subagents.
//!
//! Allows the main agent to delegate complex tasks to specialized subagents
//! that run autonomously and return results.
//!
//! Subagents can be defined in files or registered programmatically.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aither_core::llm::ToolContext;
use aither_core::{
    LanguageModel,
    llm::{Tool, ToolResult, oneshot},
};
use aither_sandbox::{
    CommandRole, CommitLedger, CommitLedgerError, CommitMark, SubagentCommit, SubagentGrant,
    TerminalBinding,
};
use askama::Template;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::AgentBuilder;
use crate::context::CacheLifetime;
use crate::external_hook::{self, ExternalHooks};
use crate::fork::{ForkSnapshot, ForkSource};
use crate::fs_util::path_exists;
use crate::model_group::ModelTier;
use crate::subagent_file::SubagentDefinition;

async fn checked_path_exists(path: &Path) -> anyhow::Result<bool> {
    path_exists(path)
        .await
        .map_err(|error| anyhow::anyhow!("failed to inspect path '{}': {error}", path.display()))
}

/// The hooks every subagent's agent carries: its own scope of the external
/// hooks, which holds none when the host configured none.
pub type SubagentHooks = crate::HCons<ExternalHooks, ()>;

/// Builder function type for configuring a subagent.
/// Returns an `AgentBuilder` so we can add hooks before building.
/// The builder returns a single-model agent (all tiers use the same LLM).
pub type SubagentBuilder<LLM> = Arc<dyn Fn(LLM) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync>;

/// Configuration for a subagent type.
pub struct SubagentType<LLM> {
    /// Description shown to the main agent.
    pub description: String,
    /// Model tier this subagent runs on when the caller does not choose one.
    pub model: Option<ModelTier>,
    /// How the subagent's agent is made.
    origin: Origin<LLM>,
}

/// How a subagent type makes its agent.
enum Origin<LLM> {
    /// A fresh agent from a configured builder.
    Builder(SubagentBuilder<LLM>),
    /// A copy of the main agent's request and model.
    Fork,
}

impl<LLM> Clone for Origin<LLM> {
    fn clone(&self) -> Self {
        match self {
            Self::Builder(builder) => Self::Builder(Arc::clone(builder)),
            Self::Fork => Self::Fork,
        }
    }
}

/// Maps a path the subagent sees onto the host directory it really refers to.
///
/// A subagent works inside a sandbox, so the paths in its prompts are virtual;
/// this records how to resolve them without letting one escape its root.
#[derive(Clone, Debug)]
pub struct SubagentFileMount {
    /// Path prefix as the subagent sees it.
    pub virtual_prefix: PathBuf,
    /// Directory on the host that prefix resolves to.
    pub host_root: PathBuf,
}

impl SubagentFileMount {
    /// Maps `virtual_prefix` onto `host_root`.
    #[must_use]
    pub fn new(virtual_prefix: impl Into<PathBuf>, host_root: impl Into<PathBuf>) -> Self {
        Self {
            virtual_prefix: virtual_prefix.into(),
            host_root: host_root.into(),
        }
    }
}

impl<LLM> Clone for SubagentType<LLM> {
    fn clone(&self) -> Self {
        Self {
            description: self.description.clone(),
            model: self.model,
            origin: self.origin.clone(),
        }
    }
}

impl<LLM> std::fmt::Debug for SubagentType<LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentType")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl<LLM: Clone> SubagentType<LLM> {
    /// Create a new subagent type with a builder function.
    ///
    /// The builder function should return an `AgentBuilder` (not a built `Agent`)
    /// so that hooks can be added before building.
    pub fn new<F>(description: impl Into<String>, builder: F) -> Self
    where
        F: Fn(LLM) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync + 'static,
    {
        Self {
            description: description.into(),
            model: None,
            origin: Origin::Builder(Arc::new(builder)),
        }
    }

    /// A subagent that continues the main agent's own context.
    ///
    /// It runs on exactly the main agent's model and its first request is the
    /// main agent's request followed by the prompt, so it reads the main
    /// agent's warm prompt cache. It runs with the authority of any other
    /// subagent: its own space, and what it cannot do there goes to the main
    /// agent. The main agent must publish its requests to the tool's
    /// [`fork_source`](SubagentTool::fork_source), and a start is refused
    /// while the main agent's context is within 10% of its compaction point.
    pub fn fork(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            model: None,
            origin: Origin::Fork,
        }
    }

    /// Declare the model tier this subagent runs on unless the caller
    /// overrides it with the `model` argument of a `start` call.
    #[must_use]
    pub const fn with_model(mut self, model: ModelTier) -> Self {
        self.model = Some(model);
        self
    }
}

/// Launch a specialized subagent for complex tasks.
///
/// Subagents are autonomous agents that run in their own context window.
/// Their work does not consume your context, and they return only their
/// final results. This enables:
///
/// - Context isolation for research-heavy tasks
/// - Parallel execution of independent investigations
/// - Specialized capabilities per subagent type
///
/// Each subagent type has focused tools and prompts optimized for its purpose.
/// Choose the subagent type that best matches the task requirements.
///
/// The task prompt is always read from a file: write the full prompt to a
/// file first, then pass its path in `prompt_file`. Inline prompt text is
/// not a file path and is rejected.
#[derive(Debug, Clone, JsonSchema, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum SubagentArgs {
    /// Start a new subagent.
    Start {
        /// Subagent type name (e.g., "explore", "plan") or path to a
        /// subagent file. If the value contains '/' or ends with '.md',
        /// it's treated as a file path, so custom subagents can be loaded
        /// from files like `./custom-agent.md` or `subagents/researcher.md`.
        subagent: String,
        /// Path to a file containing the detailed task prompt for the
        /// subagent. The file must exist and be non-empty.
        prompt_file: String,
        /// Model tier to run the subagent on. When omitted and the subagent
        /// definition declares none, the fast model reads the prompt and
        /// routes it to one of the configured tiers.
        model: Option<ModelTier>,
        /// Directories, usually linked git worktrees you created, the
        /// subagent may write in addition to its own space. Each must be
        /// inside what you can write. Everything else you can write is
        /// read-only to the subagent, and it cannot commit in these.
        #[serde(default)]
        worktrees: Vec<String>,
        /// Linked git worktrees the subagent may write and commit in. Each
        /// must be on a branch `subagent/<id>/<name>` whose namespace
        /// `subagent/<id>/` holds no other branch. Every commit it makes is
        /// listed in its result.
        #[serde(default)]
        commit_worktrees: Vec<String>,
    },
    /// Continue a finished subagent in its own context.
    ///
    /// Its earlier conversation stays its context and `prompt_file` is the
    /// follow-up prompt; the result comes back under the same id. Ids
    /// expire with the provider's prompt cache: continuing an expired or
    /// unknown id is an error.
    Resume {
        /// Id from a finished subagent's result.
        id: u64,
        /// Path to a file containing the follow-up prompt. The file must
        /// exist and be non-empty.
        prompt_file: String,
    },
}

impl SubagentArgs {
    /// What a start call grants beyond the subagent's own space, with
    /// relative paths resolved against the caller's working directory.
    fn grant(&self, working_dir: Option<&Path>) -> SubagentGrant {
        let Self::Start {
            worktrees,
            commit_worktrees,
            ..
        } = self
        else {
            return SubagentGrant::new();
        };
        let resolve = |path: &String| {
            let path = Path::new(path);
            match working_dir {
                Some(base) if path.is_relative() => base.join(path),
                _ => path.to_path_buf(),
            }
        };
        let grant = worktrees
            .iter()
            .map(resolve)
            .fold(SubagentGrant::new(), SubagentGrant::with_worktree);
        commit_worktrees
            .iter()
            .map(resolve)
            .fold(grant, SubagentGrant::with_committing_worktree)
    }

    /// The prompt file either call shape carries.
    fn prompt_file(&self) -> &str {
        match self {
            Self::Start { prompt_file, .. } | Self::Resume { prompt_file, .. } => prompt_file,
        }
    }
}

#[derive(Template)]
#[template(path = "subagent_result.txt", escape = "none")]
struct SubagentResultTemplate<'a> {
    subagent: &'a str,
    outcome: SubagentReport<'a>,
    /// Commits the subagent made in its granted worktrees during this run,
    /// or why they could not be listed; `None` for a subagent launched
    /// without a commit grant.
    commits: Option<&'a CommitListing>,
}

/// The commits listed for one run of a subagent holding a commit grant, or
/// why they could not be listed, and what could not be done to its
/// repositories before the run.
struct CommitListing {
    listed: Result<Vec<SubagentCommit>, CommitLedgerError>,
    /// Why a repository could not be refreshed or imported when the run
    /// started: its commits stay in it, out of the main repository.
    setbacks: Vec<String>,
}

/// What the result text reports about a subagent run.
enum SubagentReport<'a> {
    /// The user cancelled the run.
    Cancelled,
    /// The run produced a result.
    Completed {
        model: Option<&'a str>,
        result: &'a str,
        /// Id to continue this subagent with while its prompt cache is warm.
        id: u64,
        /// Set when the run crossed the frequent-idle threshold: the subagent
        /// kept needing prompt-cache keep-alives and this says why.
        alert: Option<&'a str>,
    },
}

struct HistoryEntry {
    role: aither_core::llm::Role,
    content: String,
    /// Names of the tools the message called.
    tool_names: Vec<String>,
}

#[derive(Template)]
#[template(path = "subagent_error.txt", escape = "none")]
struct SubagentErrorTemplate<'a> {
    subagent: &'a str,
    error: &'a str,
    /// Commits the subagent made before it failed; `None` for a subagent
    /// launched without a commit grant.
    commits: Option<&'a CommitListing>,
    history: &'a [HistoryEntry],
}

#[derive(Template)]
#[template(path = "fork_refused.txt", escape = "none")]
struct ForkRefusedTemplate<'a> {
    reason: &'a ForkRefusal,
}

/// Why a fork was not started.
enum ForkRefusal {
    /// The caller chose a model; a fork has the main agent's.
    ModelGiven,
    /// The caller is a subagent.
    NotMain,
    /// The tool is not bound to a terminal to launch from.
    NoTerminal,
    /// The main agent has published no request.
    NothingSent,
    /// The model the tool resolves the main agent's tier to is not the one
    /// the main agent runs on.
    OtherModel {
        /// The model the main agent ran its request on.
        main: String,
        /// The model the tool would run the fork on.
        fork: String,
    },
    /// The main agent's context is within 10% of its compaction point.
    NearCompaction {
        /// Window usage, in percent.
        usage: f32,
        /// Compaction point, in percent.
        compaction: f32,
    },
}

impl ForkRefusal {
    fn into_error(self) -> anyhow::Error {
        anyhow::anyhow!(
            ForkRefusedTemplate { reason: &self }
                .render()
                .expect("fork refusal template must render")
        )
    }
}

/// Structured response the fast model returns when routing a prompt.
#[derive(Debug, JsonSchema, Deserialize)]
struct SubagentRouting {
    /// The configured model tier that best fits the task.
    model: ModelTier,
}

/// One configured model offered to the router.
struct RouterModel<'a> {
    tier: &'static str,
    name: &'a str,
    description: &'a str,
}

/// Router prompt listing the configured models by tier.
#[derive(Template)]
#[template(path = "subagent_router.txt", escape = "none")]
struct SubagentRouter<'a> {
    models: [RouterModel<'a>; 3],
}

const fn tier_label(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Advanced => "advanced",
        ModelTier::Balanced => "balanced",
        ModelTier::Fast => "fast",
    }
}

/// Where a subagent request gets its definition and builder from.
enum Source<'a, LLM> {
    /// A `SubagentDefinition` loaded from a file.
    File(SubagentDefinition),
    /// A registered `SubagentType` on the tool.
    Registered(&'a SubagentType<LLM>),
}

/// Snapshot of one in-flight subagent run.
#[derive(Debug, Clone)]
pub struct SubagentRunInfo {
    /// Unique run id within the owning [`SubagentRuns`] registry.
    pub id: u64,
    /// Subagent type name (or file-derived id) that was launched.
    pub subagent: String,
    /// Task prompt the run was started with.
    pub prompt: String,
    /// When the run started.
    pub started: Instant,
}

/// Live run record: public info plus the sender that cancels it.
type RunSlot = (SubagentRunInfo, async_channel::Sender<()>);

/// Shared registry of in-flight [`SubagentTool`] runs.
///
/// The tool records every `call` while it is executing; a cloned handle
/// lets a host enumerate and cancel individual runs without aborting the
/// parent agent's turn. Cancellation drops the child query future and the
/// tool returns a `cancelled` result to the parent.
#[derive(Clone, Debug, Default)]
pub struct SubagentRuns {
    runs: Arc<Mutex<HashMap<u64, RunSlot>>>,
    next_id: Arc<AtomicU64>,
}

impl SubagentRuns {
    /// Snapshot of every in-flight run, ordered by run id.
    ///
    /// # Panics
    ///
    /// Panics if the internal registry lock is poisoned.
    #[must_use]
    pub fn list(&self) -> Vec<SubagentRunInfo> {
        let mut infos: Vec<_> = self
            .runs
            .lock()
            .expect("subagent runs lock poisoned")
            .values()
            .map(|(info, _)| info.clone())
            .collect();
        infos.sort_by_key(|info| info.id);
        infos
    }

    /// Cancel one in-flight run.
    ///
    /// Returns `false` only when `id` is unknown — a run that was already
    /// signalled but has not unwound yet still reports `true`. A cancelled
    /// run keeps appearing in [`list`](Self::list) until the tool
    /// observes the signal and unwinds.
    ///
    /// # Panics
    ///
    /// Panics if the internal registry lock is poisoned.
    #[must_use]
    pub fn cancel(&self, id: u64) -> bool {
        self.runs
            .lock()
            .expect("subagent runs lock poisoned")
            .get(&id)
            .is_some_and(|(_, cancel)| cancel.try_send(()).is_ok() || cancel.is_full())
    }

    /// Cancel every in-flight run; returns how many were signalled.
    ///
    /// # Panics
    ///
    /// Panics if the internal registry lock is poisoned.
    #[must_use]
    pub fn cancel_all(&self) -> usize {
        self.runs
            .lock()
            .expect("subagent runs lock poisoned")
            .values()
            .filter(|(_, cancel)| cancel.try_send(()).is_ok() || cancel.is_full())
            .count()
    }

    /// Register a run; the receiver resolves once when it is cancelled.
    fn begin(&self, subagent: &str, prompt: &str) -> (u64, async_channel::Receiver<()>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (cancel_tx, cancel_rx) = async_channel::bounded(1);
        let info = SubagentRunInfo {
            id,
            subagent: subagent.to_string(),
            prompt: prompt.to_string(),
            started: Instant::now(),
        };
        self.runs
            .lock()
            .expect("subagent runs lock poisoned")
            .insert(id, (info, cancel_tx));
        (id, cancel_rx)
    }

    /// Remove a run record when its tool call finishes.
    fn finish(&self, id: u64) {
        self.runs
            .lock()
            .expect("subagent runs lock poisoned")
            .remove(&id);
    }
}

enum SubagentOutcome<E> {
    Finished(Result<String, E>),
    Cancelled,
}

/// Await the child query or a cancellation signal, whichever lands first.
/// Dropping the query future aborts the run while the parent turn
/// continues.
async fn await_subagent_run<F, E>(
    query: F,
    cancel_rx: async_channel::Receiver<()>,
) -> SubagentOutcome<E>
where
    F: std::future::Future<Output = Result<String, E>>,
{
    futures_lite::future::or(async { SubagentOutcome::Finished(query.await) }, async {
        let _ = cancel_rx.recv().await;
        SubagentOutcome::Cancelled
    })
    .await
}

/// A finished subagent kept so the parent can continue it.
struct FinishedSubagent<LLM> {
    agent: crate::Agent<LLM, LLM, LLM, SubagentHooks>,
    subagent: String,
    model: Option<String>,
    child_job_registry: Option<aither_sandbox::JobRegistry>,
    /// Lists the commits the subagent makes; set when it was launched with a
    /// commit grant.
    commit_ledger: Option<CommitLedger>,
    /// How long the provider keeps this subagent's prompt cache warm — the
    /// [`PromptCache`](crate::context::PromptCache) lifetime the subagent
    /// ran with. The continuation id expires with the cache entry.
    lifetime: CacheLifetime,
}

impl<LLM: LanguageModel + 'static> FinishedSubagent<LLM> {
    /// Whether the cache entry's lifetime passed since it was last
    /// refreshed (the later of the last request and the last keep-alive).
    fn expired(&self) -> bool {
        self.agent
            .cache_entry_started_at()
            .is_none_or(|started| self.lifetime.is_cold(started.elapsed()))
    }
}

enum FinishedSlot<LLM> {
    /// Waiting to be continued.
    Idle(Box<FinishedSubagent<LLM>>),
    /// Checked out by a continuation in flight.
    Running,
    /// Its cache lifetime passed; the context was dropped. The lifetime
    /// stays so continuing the id names how long it lived.
    Expired(CacheLifetime),
}

struct FinishedInner<LLM> {
    next_id: u64,
    slots: HashMap<u64, FinishedSlot<LLM>>,
}

/// Finished subagents the parent may continue until their cache expires.
struct FinishedSubagents<LLM> {
    inner: Mutex<FinishedInner<LLM>>,
}

impl<LLM> FinishedSubagents<LLM> {
    fn new() -> Self {
        Self {
            inner: Mutex::new(FinishedInner {
                next_id: 1,
                slots: HashMap::new(),
            }),
        }
    }
}

impl<LLM: LanguageModel + 'static> FinishedSubagents<LLM> {
    /// Drops the context of every subagent whose cache lifetime passed.
    fn expire(inner: &mut FinishedInner<LLM>) {
        for slot in inner.slots.values_mut() {
            if let FinishedSlot::Idle(finished) = slot
                && finished.expired()
            {
                *slot = FinishedSlot::Expired(finished.lifetime);
            }
        }
    }

    /// Takes the subagent out for a continuation.
    fn checkout(&self, id: u64) -> anyhow::Result<Box<FinishedSubagent<LLM>>> {
        let mut inner = self.inner.lock().expect("finished subagents lock poisoned");
        Self::expire(&mut inner);
        match inner.slots.get_mut(&id) {
            None => anyhow::bail!("unknown subagent id {id}"),
            Some(FinishedSlot::Expired(lifetime)) => {
                let CacheLifetime::Limited(lifetime) = lifetime else {
                    unreachable!("a permanent cache lifetime never expires");
                };
                anyhow::bail!(
                    "subagent {id} expired: its prompt-cache entry lived {lifetime:?} and its context was released"
                );
            }
            Some(FinishedSlot::Running) => {
                anyhow::bail!("subagent {id} is still running a continuation")
            }
            Some(slot @ FinishedSlot::Idle(_)) => {
                let FinishedSlot::Idle(finished) = std::mem::replace(slot, FinishedSlot::Running)
                else {
                    unreachable!("slot matched as idle");
                };
                Ok(finished)
            }
        }
    }

    /// Keeps a finished subagent; `id` is the continued one's id, or `None`
    /// for a first run. Returns the id the result names.
    fn store(&self, id: Option<u64>, finished: FinishedSubagent<LLM>) -> u64 {
        let mut inner = self.inner.lock().expect("finished subagents lock poisoned");
        Self::expire(&mut inner);
        let id = id.unwrap_or_else(|| {
            let id = inner.next_id;
            inner.next_id += 1;
            id
        });
        inner
            .slots
            .insert(id, FinishedSlot::Idle(Box::new(finished)));
        id
    }

    /// Releases an id whose subagent can no longer be continued.
    fn release(&self, id: u64) {
        self.inner
            .lock()
            .expect("finished subagents lock poisoned")
            .slots
            .remove(&id);
    }
}

/// A tool that spawns specialized subagents to handle complex tasks.
///
/// Allows the main agent to delegate work to specialized subagents
/// that have different capabilities and run in isolated context windows.
///
/// # Example
///
/// ```rust,ignore
/// use aither_agent::specialized::task::{SubagentTool, SubagentType};
///
/// // Create subagent tool
/// let mut subagent_tool = SubagentTool::new(llm.clone());
///
/// // Register an "explore" subagent type
/// subagent_tool.register("explore", SubagentType::new(
///     "Codebase explorer for finding files and searching code",
///     |llm| Agent::builder(llm)
///         .system_prompt("You are a codebase explorer...")
///         .tool(FsTool::new())
///         .max_iterations(15)
///         .build()
/// ));
///
/// let agent = Agent::builder(llm)
///     .tool(subagent_tool)
///     .build();
/// ```
pub struct SubagentTool<LLM> {
    /// Model a subagent runs on when the advanced tier is selected.
    advanced: LLM,
    /// Model a subagent runs on when the balanced tier is selected.
    balanced: LLM,
    /// Model a subagent runs on when the fast tier is selected; also routes
    /// prompts that carry no model selector.
    fast: LLM,
    types: HashMap<String, SubagentType<LLM>>,
    /// Base directory for resolving relative paths (e.g., sandbox directory).
    base_dir: Option<PathBuf>,
    /// Ordered mount map used to resolve virtual sandbox paths into host paths.
    mounts: Vec<SubagentFileMount>,
    /// The terminal this tool is bound to: subagents launch from it, within
    /// its authority, and relative grants resolve against its working
    /// directory.
    binding: Option<TerminalBinding>,
    /// Registry the tool reports in-flight runs to; clone it before
    /// handing the tool to a builder to observe or cancel runs.
    runs: SubagentRuns,
    /// Prompt-cache keep-alive settings applied to every child agent; `None`
    /// disables keep-alive. The caller sets it only for the Claude client,
    /// which also puts the children on the five-minute cache TTL.
    cache_keep_alive: Option<crate::config::CacheKeepAlive>,
    /// Finished subagents the parent can continue; each lives until its
    /// own prompt-cache lifetime passes.
    finished: FinishedSubagents<LLM>,
    /// The main agent's latest request, which a fork starts from.
    fork_source: ForkSource,
    /// Whether this tool serves the main agent's terminal, the only one a
    /// fork can continue the context of.
    serves_main: bool,
    /// The external hooks of the agent this tool serves: told when a
    /// subagent starts and stops, and the parent of each subagent's own scope.
    hooks: Option<ExternalHooks>,
}

impl<LLM> std::fmt::Debug for SubagentTool<LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_names: Vec<_> = self.types.keys().cloned().collect();
        f.debug_struct("SubagentTool")
            .field("type_names", &type_names)
            .field("base_dir", &self.base_dir)
            .finish_non_exhaustive()
    }
}

impl<LLM: Clone + Sync> SubagentTool<LLM> {
    /// Creates a new `TaskTool` with the given LLM.
    ///
    /// All three model tiers (advanced/balanced/fast) use the same model;
    /// configure distinct ones with [`with_balanced_model`](Self::with_balanced_model)
    /// and [`with_fast_model`](Self::with_fast_model).
    pub fn new(llm: LLM) -> Self {
        Self {
            advanced: llm.clone(),
            balanced: llm.clone(),
            fast: llm,
            types: HashMap::new(),
            base_dir: None,
            mounts: Vec::new(),
            binding: None,
            runs: SubagentRuns::default(),
            cache_keep_alive: None,
            finished: FinishedSubagents::new(),
            fork_source: ForkSource::default(),
            serves_main: true,
            hooks: None,
        }
    }

    /// Sets the base directory for resolving relative paths.
    ///
    /// Paths like `subagents/...` and `skills/...` will be resolved
    /// relative to this directory.
    #[must_use]
    pub fn with_base_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    /// Sets explicit mount mappings for resolving subagent file paths.
    ///
    /// Mappings are checked in order and matched by virtual path prefix.
    #[must_use]
    pub fn with_file_mounts<I>(mut self, mounts: I) -> Self
    where
        I: IntoIterator<Item = SubagentFileMount>,
    {
        self.mounts = mounts.into_iter().collect();
        self
    }

    /// Sets the model that runs a subagent when the balanced tier is selected.
    #[must_use]
    pub fn with_balanced_model(mut self, model: LLM) -> Self {
        self.balanced = model;
        self
    }

    /// Sets the model that runs a subagent when the fast tier is selected and
    /// that routes prompts with no model selector.
    #[must_use]
    pub fn with_fast_model(mut self, model: LLM) -> Self {
        self.fast = model;
        self
    }

    /// This tool as the terminal `binding` describes serves it: subagents
    /// launch from that terminal, with at most its authority, and relative
    /// paths resolve against its working directory first.
    ///
    /// Register the tool with
    /// [`ToolRegistryBuilder::configure_terminal_bound_tool`](aither_sandbox::ToolRegistryBuilder::configure_terminal_bound_tool)
    /// so every terminal, a subagent's included, gets its own instance. The
    /// instance shares the run registry; the subagents it can continue are
    /// its own.
    #[must_use]
    pub fn for_terminal(&self, binding: &TerminalBinding) -> Self {
        Self {
            advanced: self.advanced.clone(),
            balanced: self.balanced.clone(),
            fast: self.fast.clone(),
            types: self.types.clone(),
            base_dir: self.base_dir.clone(),
            mounts: self.mounts.clone(),
            binding: Some(binding.clone()),
            runs: self.runs.clone(),
            cache_keep_alive: self.cache_keep_alive.clone(),
            finished: FinishedSubagents::new(),
            fork_source: self.fork_source.clone(),
            serves_main: binding.role() == CommandRole::Main,
            hooks: ExternalHooks::of_observer(binding.observer()).or_else(|| {
                self.hooks
                    .clone()
                    .filter(|_| binding.role() == CommandRole::Main)
            }),
        }
    }

    /// Tells `hooks` when a subagent starts and stops, and gives each
    /// subagent a scope of its own derived from it. A tool a terminal serves
    /// takes the hooks of that terminal's command registry instead.
    #[must_use]
    pub fn with_hooks(mut self, hooks: ExternalHooks) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Where the main agent publishes its requests for the `fork` subagent
    /// kind; give it to the main agent's builder with
    /// [`AgentBuilder::fork_source`](crate::AgentBuilder::fork_source) before
    /// handing the tool over, like [`runs`](Self::runs).
    #[must_use]
    pub fn fork_source(&self) -> ForkSource {
        self.fork_source.clone()
    }

    /// Attaches the shared run registry this tool reports to.
    ///
    /// Clone the [`SubagentRuns`] before building the agent — it is the
    /// only handle able to list or cancel in-flight subagent runs.
    #[must_use]
    pub fn with_runs(mut self, runs: SubagentRuns) -> Self {
        self.runs = runs;
        self
    }

    /// Enables prompt-cache keep-alive on child agents with these settings.
    ///
    /// Call it only when the tool's models are served through the Claude
    /// client. It also puts those children on the five-minute Claude cache
    /// TTL, so their cache lifetime is 300 s; without it they keep their
    /// provider's lifetime.
    #[must_use]
    pub const fn with_cache_keep_alive(
        mut self,
        keep_alive: crate::config::CacheKeepAlive,
    ) -> Self {
        self.cache_keep_alive = Some(keep_alive);
        self
    }

    /// Sets how many keep-alives in one run count as idling frequently.
    ///
    /// Crossing the threshold makes the call's result carry one alert
    /// naming the subagent and the reason for its most recent idle.
    #[must_use]
    pub fn with_frequent_idle_threshold(mut self, threshold: usize) -> Self {
        self.cache_keep_alive
            .get_or_insert_with(crate::config::CacheKeepAlive::default)
            .frequent_idle_threshold = threshold;
        self
    }

    /// The run registry this tool reports to.
    #[must_use]
    pub fn runs(&self) -> SubagentRuns {
        self.runs.clone()
    }

    /// Register a subagent type.
    pub fn register(&mut self, name: impl Into<String>, subagent: SubagentType<LLM>) {
        let name = name.into();
        self.types.insert(name, subagent);
    }

    /// Register a subagent type (builder pattern).
    #[must_use]
    pub fn with_type(mut self, name: impl Into<String>, subagent: SubagentType<LLM>) -> Self {
        self.register(name, subagent);
        self
    }

    /// Returns a list of registered subagent types with their descriptions.
    ///
    /// Useful for injecting into system prompts so the main agent knows
    /// what subagents are available.
    pub fn type_descriptions(&self) -> Vec<(&str, &str)> {
        self.types
            .iter()
            .map(|(name, t)| (name.as_str(), t.description.as_str()))
            .collect()
    }

    fn effective_mounts(&self) -> Vec<SubagentFileMount> {
        // A relative path names a file in the calling terminal's working
        // directory first: a subagent writes its prompts in its own space.
        let caller = self
            .binding
            .as_ref()
            .map(TerminalBinding::working_dir)
            .filter(|working_dir| self.base_dir.as_deref() != Some(*working_dir))
            .map(|working_dir| SubagentFileMount::new(PathBuf::from("."), working_dir));
        if !self.mounts.is_empty() {
            return caller.into_iter().chain(self.mounts.clone()).collect();
        }

        let base = self
            .base_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(Path::new(".")));
        caller
            .into_iter()
            .chain([
                SubagentFileMount::new(PathBuf::from("."), base.clone()),
                SubagentFileMount::new(PathBuf::from("subagents"), base.join("subagents")),
                SubagentFileMount::new(PathBuf::from("skills"), base.join("skills")),
            ])
            .collect()
    }

    fn strip_leading_curdir(path: &Path) -> PathBuf {
        let mut stripped = PathBuf::new();
        for component in path.components() {
            if matches!(component, Component::CurDir) {
                continue;
            }
            stripped.push(component.as_os_str());
        }
        stripped
    }

    fn to_mount_relative(path: &Path, virtual_prefix: &Path) -> Option<PathBuf> {
        let normalized_path = Self::strip_leading_curdir(path);
        if normalized_path.as_os_str().is_empty() {
            return None;
        }
        if normalized_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return None;
        }

        let normalized_prefix = Self::strip_leading_curdir(virtual_prefix);
        if normalized_prefix.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return None;
        }
        if normalized_prefix.as_os_str().is_empty() {
            return Some(normalized_path);
        }

        if normalized_path.starts_with(&normalized_prefix) {
            let Ok(stripped) = normalized_path.strip_prefix(&normalized_prefix) else {
                return None;
            };
            let relative = Self::strip_leading_curdir(stripped);
            if relative.as_os_str().is_empty() {
                None
            } else {
                Some(relative)
            }
        } else {
            None
        }
    }

    fn mount_specificity(prefix: &Path) -> usize {
        Self::strip_leading_curdir(prefix).components().count()
    }

    fn matching_mounts<'a>(
        mounts: &'a [SubagentFileMount],
        file_path: &Path,
    ) -> Vec<(usize, &'a SubagentFileMount, PathBuf)> {
        let mut matched = mounts
            .iter()
            .enumerate()
            .filter_map(|(index, mount)| {
                Self::to_mount_relative(file_path, &mount.virtual_prefix)
                    .map(|relative| (index, mount, relative))
            })
            .collect::<Vec<_>>();
        matched.sort_by(|(index_a, mount_a, _), (index_b, mount_b, _)| {
            Self::mount_specificity(&mount_b.virtual_prefix)
                .cmp(&Self::mount_specificity(&mount_a.virtual_prefix))
                .then(index_a.cmp(index_b))
        });
        matched
    }

    async fn resolve_relative_path_through_mounts(
        requested_display: &Path,
        file_path: &Path,
        mounts: &[SubagentFileMount],
    ) -> anyhow::Result<PathBuf> {
        let mut attempted = Vec::new();
        let matched = Self::matching_mounts(mounts, file_path);

        for (_, mount, relative) in matched {
            let candidate = mount.host_root.join(relative);
            let display_prefix = Self::strip_leading_curdir(&mount.virtual_prefix);
            let display_path = if display_prefix.as_os_str().is_empty() {
                candidate
                    .strip_prefix(&mount.host_root)
                    .map_or_else(|_| requested_display.to_path_buf(), Path::to_path_buf)
            } else {
                display_prefix.join(
                    candidate
                        .strip_prefix(&mount.host_root)
                        .map_or_else(|_| requested_display.to_path_buf(), Path::to_path_buf),
                )
            };
            attempted.push(display_path.display().to_string());
            if checked_path_exists(&candidate).await? {
                return Ok(candidate);
            }
        }

        if attempted.is_empty() {
            return Err(anyhow::anyhow!(
                "Subagent file '{}' is outside mounted paths or uses forbidden path components",
                requested_display.display()
            ));
        }

        Err(anyhow::anyhow!(
            "Subagent file '{}' not found. Attempted: {}",
            requested_display.display(),
            attempted.join(", ")
        ))
    }

    /// Resolves a path the subagent asked for against its mounts.
    ///
    /// Takes the mounts and base directory rather than `&self` so the returned
    /// future does not borrow the tool, and so stays `Send`.
    async fn resolve_subagent_path(
        mounts: &[SubagentFileMount],
        base_dir: Option<&Path>,
        file_path: &Path,
    ) -> anyhow::Result<PathBuf> {
        if file_path.is_absolute() {
            if let Some(base_dir) = base_dir
                && let Ok(stripped) = file_path.strip_prefix(base_dir)
            {
                let relative = Self::strip_leading_curdir(stripped);
                if !relative.as_os_str().is_empty() {
                    return Self::resolve_relative_path_through_mounts(
                        &relative, &relative, mounts,
                    )
                    .await;
                }
            }

            if checked_path_exists(file_path).await? {
                return Ok(file_path.to_path_buf());
            }
            return Err(anyhow::anyhow!(
                "Subagent file '{}' not found",
                file_path.display()
            ));
        }

        Self::resolve_relative_path_through_mounts(file_path, file_path, mounts).await
    }

    /// The configured model behind a tier.
    const fn model_for(&self, tier: ModelTier) -> &LLM {
        match tier {
            ModelTier::Advanced => &self.advanced,
            ModelTier::Balanced => &self.balanced,
            ModelTier::Fast => &self.fast,
        }
    }

    /// Reads the subagent prompt from the file `prompt_file` points to.
    ///
    /// The prompt is always a file: inline text fails path resolution, and a
    /// missing or empty file is an error.
    async fn read_prompt(&self, prompt_file: &str) -> anyhow::Result<String> {
        let resolved = Self::resolve_subagent_path(
            &self.effective_mounts(),
            self.base_dir.as_deref(),
            Path::new(prompt_file),
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!("subagent prompt must be a path to a prompt file: {error}")
        })?;
        let content = async_fs::read_to_string(&resolved).await.map_err(|error| {
            anyhow::anyhow!(
                "failed to read subagent prompt file '{}': {error}",
                resolved.display()
            )
        })?;
        let prompt = content.trim();
        if prompt.is_empty() {
            anyhow::bail!("subagent prompt file '{}' is empty", resolved.display());
        }
        Ok(prompt.to_string())
    }

    /// Resolves a `start` call's `subagent` into the source it builds its
    /// agent from.
    ///
    /// Returns the subagent id, the model the definition declares (if any),
    /// and the source. Values containing '/' or ending in '.md' load a
    /// `SubagentDefinition` from a file; anything else must name a
    /// registered subagent type.
    async fn resolve_source(
        &self,
        subagent: &str,
    ) -> anyhow::Result<(String, Option<ModelTier>, Source<'_, LLM>)> {
        let is_file_path = subagent.contains('/')
            || Path::new(subagent)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"));

        if is_file_path {
            // Resolve file paths using explicit search roots.
            let resolved_path = Self::resolve_subagent_path(
                &self.effective_mounts(),
                self.base_dir.as_deref(),
                &PathBuf::from(subagent),
            )
            .await?;
            let def = SubagentDefinition::from_file_async(&resolved_path)
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to read subagent file '{}': {e}",
                        resolved_path.display()
                    )
                })?;
            return Ok((def.id.clone(), def.model, Source::File(def)));
        }

        let subagent_type = self.types.get(subagent).ok_or_else(|| {
            let available: Vec<&str> = self.types.keys().map(std::string::String::as_str).collect();
            anyhow::anyhow!(
                "Unknown subagent type '{subagent}'. Available: {}",
                available.join(", ")
            )
        })?;
        Ok((
            subagent.to_string(),
            subagent_type.model,
            Source::Registered(subagent_type),
        ))
    }
}

impl<LLM: LanguageModel + Clone + Sync + 'static> SubagentTool<LLM> {
    /// Register a subagent from a definition (builder pattern).
    ///
    /// Creates a subagent type from a `SubagentDefinition` loaded from a file.
    /// The subagent will use the definition's system prompt and max iterations.
    #[must_use]
    pub fn with_definition(self, def: SubagentDefinition) -> Self {
        let mut subagent = SubagentType::new(def.description.clone(), move |llm| {
            crate::AgentBuilder::new(llm)
                .agent_kind(crate::AgentKind::Chatbot)
                .system_prompt(&def.system_prompt)
                .max_iterations(def.max_iterations)
        });
        subagent.model = def.model;
        self.with_type(def.id, subagent)
    }

    /// The main agent's request a fork starts from.
    ///
    /// # Errors
    ///
    /// Refuses, starting nothing, when the caller is not the main agent or
    /// chose a model, when no request was published, and when the main
    /// agent's context is within 10% of its compaction point.
    fn fork_snapshot(&self, chosen: Option<ModelTier>) -> anyhow::Result<Arc<ForkSnapshot>> {
        if chosen.is_some() {
            return Err(ForkRefusal::ModelGiven.into_error());
        }
        if !self.serves_main {
            return Err(ForkRefusal::NotMain.into_error());
        }
        if self.binding.is_none() {
            return Err(ForkRefusal::NoTerminal.into_error());
        }
        let snapshot = self
            .fork_source
            .latest()
            .ok_or_else(|| ForkRefusal::NothingSent.into_error())?;
        if !snapshot.policy().allows_fork(snapshot.usage()) {
            return Err(ForkRefusal::NearCompaction {
                usage: snapshot.usage() * 100.0,
                compaction: snapshot.policy().compact_at * 100.0,
            }
            .into_error());
        }
        Ok(snapshot)
    }

    /// Builds a fresh agent for `subagent`, selecting its model from `chosen`
    /// or by routing `prompt`.
    async fn start_agent(
        &self,
        subagent: &str,
        chosen: Option<ModelTier>,
        grant: SubagentGrant,
        prompt: &str,
    ) -> anyhow::Result<(
        crate::Agent<LLM, LLM, LLM, SubagentHooks>,
        String,
        String,
        Option<aither_sandbox::JobRegistry>,
        Option<CommitLedger>,
    )> {
        let (subagent_id, declared_model, source) = self.resolve_source(subagent).await?;

        // A fork is checked before anything is started, and runs on the
        // main agent's model.
        let fork = match &source {
            Source::Registered(SubagentType {
                origin: Origin::Fork,
                ..
            }) => Some(self.fork_snapshot(chosen)?),
            _ => None,
        };

        // A model given by the caller or the definition runs as given;
        // otherwise the fast model routes the prompt to a configured tier.
        let (tier, model) = match &fork {
            Some(fork) => {
                let resolved = self.select_model(prompt, Some(fork.tier())).await?;
                if resolved.1 != fork.model() {
                    return Err(ForkRefusal::OtherModel {
                        main: fork.model().to_string(),
                        fork: resolved.1,
                    }
                    .into_error());
                }
                resolved
            }
            None => self.select_model(prompt, chosen.or(declared_model)).await?,
        };
        let selected = self.model_for(tier).clone();

        let agent_builder = match source {
            Source::File(def) => crate::AgentBuilder::new(selected)
                .agent_kind(crate::AgentKind::Chatbot)
                .system_prompt(&def.system_prompt)
                .max_iterations(def.max_iterations),
            Source::Registered(SubagentType {
                origin: Origin::Builder(builder),
                ..
            }) => builder(selected),
            Source::Registered(SubagentType {
                origin: Origin::Fork,
                ..
            }) => crate::AgentBuilder::new(selected),
        };
        tracing::info!(subagent = %subagent_id, model = %model, "Starting subagent");
        let child_hooks = self
            .hooks
            .as_ref()
            .map_or_else(ExternalHooks::none, ExternalHooks::for_subagent);
        let command_observer = self.hooks.is_some().then(|| child_hooks.command_observer());
        let command_gate = self.hooks.is_some().then(|| child_hooks.command_gate());

        // Launch the subagent's terminal from the terminal this tool serves.
        let (agent_builder, child_job_registry, commit_ledger) = if let Some(binding) =
            &self.binding
        {
            let dyn_terminal = binding
                .launcher()
                .create(grant, command_observer, command_gate)
                .await
                .map_err(|e| anyhow::anyhow!("failed to create subagent terminal tool: {e}"))?;
            let job_registry = dyn_terminal.job_registry();
            let commit_ledger = dyn_terminal.commit_ledger().cloned();
            let agent_builder = match &fork {
                Some(fork) => agent_builder.forked_from(fork, dyn_terminal),
                None => agent_builder.dyn_terminal(dyn_terminal),
            };
            (agent_builder, Some(job_registry), commit_ledger)
        } else if grant == SubagentGrant::new() {
            (agent_builder, None, None)
        } else {
            anyhow::bail!("worktrees can only be granted to a subagent launched from a terminal");
        };

        // Keep-alive is configured only for the Claude client, whose
        // subagents run on the five-minute cache: in-run waits are covered
        // by the keep-alive and a finished subagent is continuable until
        // that entry expires. A fork applies it last, over the cache it
        // inherits from the main agent.
        let agent_builder = match &self.cache_keep_alive {
            Some(keep_alive) => agent_builder
                .cache_keep_alive(keep_alive.clone())
                .claude_cache_ttl(aither_core::llm::model::ClaudePromptCacheTtl::FiveMinutes),
            None => agent_builder,
        };

        Ok((
            agent_builder.hook(child_hooks).build(),
            subagent_id,
            model,
            child_job_registry,
            commit_ledger,
        ))
    }

    /// Returns the subagent a call runs: a fresh agent for
    /// [`SubagentArgs::Start`], or the finished one
    /// [`SubagentArgs::Resume`] names; the second value is the continued
    /// id.
    async fn acquire(
        &self,
        args: SubagentArgs,
        prompt: &str,
    ) -> anyhow::Result<(FinishedSubagent<LLM>, Option<u64>)> {
        match args {
            SubagentArgs::Start {
                ref subagent,
                model,
                ..
            } => {
                let grant = args.grant(
                    self.binding
                        .as_ref()
                        .map(TerminalBinding::working_dir)
                        .or(self.base_dir.as_deref()),
                );
                let (agent, subagent, model, child_job_registry, commit_ledger) =
                    self.start_agent(subagent, model, grant, prompt).await?;
                let lifetime = agent.cache_lifetime();
                Ok((
                    FinishedSubagent {
                        agent,
                        subagent,
                        model: Some(model),
                        child_job_registry,
                        commit_ledger,
                        lifetime,
                    },
                    None,
                ))
            }
            SubagentArgs::Resume { id, .. } => {
                let finished = self.finished.checkout(id)?;
                tracing::info!(subagent = %finished.subagent, id, "Continuing subagent");
                Ok((*finished, Some(id)))
            }
        }
    }

    /// Picks the model a subagent call runs on.
    ///
    /// A `chosen` tier — from a `start` call's `model` or the definition —
    /// is used as given. Without one, [`route_model`](Self::route_model)
    /// asks the fast model which configured tier fits the prompt.
    async fn select_model(
        &self,
        prompt: &str,
        chosen: Option<ModelTier>,
    ) -> anyhow::Result<(ModelTier, String)> {
        if let Some(tier) = chosen {
            let profile = self.model_for(tier).profile().await.map_err(|error| {
                anyhow::anyhow!(
                    "failed to fetch the {} model profile: {error}",
                    tier_label(tier)
                )
            })?;
            return Ok((tier, profile.name));
        }

        self.route_model(prompt).await
    }

    /// Asks the fast model which configured tier fits `prompt`.
    ///
    /// Returns the routed tier and the profile name of the model behind it.
    async fn route_model(&self, prompt: &str) -> anyhow::Result<(ModelTier, String)> {
        let (advanced, balanced, fast) = futures::join!(
            self.advanced.profile(),
            self.balanced.profile(),
            self.fast.profile(),
        );
        let profiles = [
            (ModelTier::Advanced, advanced),
            (ModelTier::Balanced, balanced),
            (ModelTier::Fast, fast),
        ]
        .map(|(tier, profile)| {
            profile.map_err(|error| {
                anyhow::anyhow!(
                    "failed to fetch the {} model profile for subagent routing: {error}",
                    tier_label(tier)
                )
            })
        });
        let [advanced, balanced, fast] = profiles;
        let (advanced, balanced, fast) = (advanced?, balanced?, fast?);

        let system = SubagentRouter {
            models: [
                RouterModel {
                    tier: tier_label(ModelTier::Advanced),
                    name: &advanced.name,
                    description: &advanced.description,
                },
                RouterModel {
                    tier: tier_label(ModelTier::Balanced),
                    name: &balanced.name,
                    description: &balanced.description,
                },
                RouterModel {
                    tier: tier_label(ModelTier::Fast),
                    name: &fast.name,
                    description: &fast.description,
                },
            ],
        }
        .render()
        .map_err(|error| anyhow::anyhow!("failed to render the subagent router prompt: {error}"))?;

        let routing: SubagentRouting = self
            .fast
            .generate(oneshot(system, prompt.to_string()))
            .await
            .map_err(|error| {
                anyhow::anyhow!("the fast model failed to route the subagent prompt: {error}")
            })?;

        let name = match routing.model {
            ModelTier::Advanced => advanced.name,
            ModelTier::Balanced => balanced.name,
            ModelTier::Fast => fast.name,
        };
        Ok((routing.model, name))
    }

    /// Register all builtin subagents.
    ///
    /// Loads subagents from embedded definition files (explore, plan, etc.).
    #[must_use]
    pub fn with_builtins(mut self) -> Self {
        for def in crate::builtin_subagents() {
            self = self.with_definition(def);
        }
        self.with_type(
            "fork",
            SubagentType::fork(
                "Continues your own context on your model; use it for work that depends on what you have read and decided",
            ),
        )
    }
}

impl<LLM: LanguageModel + Clone + Sync + 'static> SubagentTool<LLM> {
    /// Records where the subagent's committing worktrees stand before its run,
    /// since commits are listed per run. Nothing has run yet when that fails,
    /// so a continued subagent goes back to where it came from.
    async fn mark_commits(
        &self,
        finished: FinishedSubagent<LLM>,
        resumed: Option<u64>,
    ) -> anyhow::Result<(FinishedSubagent<LLM>, Option<CommitMark>)> {
        let Some(ledger) = &finished.commit_ledger else {
            return Ok((finished, None));
        };
        match ledger.mark().await {
            Ok(mark) => Ok((finished, Some(mark))),
            Err(error) => {
                if let Some(id) = resumed {
                    self.finished.store(Some(id), finished);
                }
                Err(anyhow::anyhow!(
                    "cannot record where the subagent's worktrees stand before its run: {error}"
                ))
            }
        }
    }
}

/// What the hooks are told about how a subagent's run ended.
fn subagent_stop_input<E: std::fmt::Display>(
    subagent: &str,
    outcome: &SubagentOutcome<E>,
) -> external_hook::SubagentStop {
    let (end, result) = match outcome {
        SubagentOutcome::Cancelled => (external_hook::SubagentEnd::Cancelled, String::new()),
        SubagentOutcome::Finished(Ok(result)) => {
            (external_hook::SubagentEnd::Completed, result.clone())
        }
        SubagentOutcome::Finished(Err(error)) => {
            (external_hook::SubagentEnd::Failed, error.to_string())
        }
    };
    external_hook::SubagentStop {
        subagent: subagent.to_string(),
        end,
        result,
    }
}

impl<LLM> Tool for SubagentTool<LLM>
where
    LLM: LanguageModel + Clone + Sync + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("subagent")
    }

    type Arguments = SubagentArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        // The task prompt comes from a file; inline prompt text is rejected.
        let prompt = self.read_prompt(args.prompt_file()).await?;

        let (finished, resumed) = self.acquire(args, &prompt).await?;
        let (
            FinishedSubagent {
                mut agent,
                subagent: subagent_id,
                model,
                child_job_registry,
                commit_ledger,
                lifetime,
            },
            commit_mark,
        ) = self.mark_commits(finished, resumed).await?;
        if let Some(hooks) = &self.hooks {
            hooks
                .fire(&external_hook::SubagentStart {
                    subagent: subagent_id.clone(),
                    prompt: prompt.clone(),
                })
                .await;
        }
        let (run_id, cancel_rx) = self.runs.begin(&subagent_id, &prompt);
        let outcome = await_subagent_run(agent.query(&prompt), cancel_rx).await;
        self.runs.finish(run_id);
        if let Some(hooks) = &self.hooks {
            hooks
                .fire(&subagent_stop_input(&subagent_id, &outcome))
                .await;
        }
        if let Some(job_registry) = &child_job_registry {
            let killed = job_registry.kill_running().await;
            if killed > 0 {
                tracing::info!(
                    subagent = %subagent_id,
                    killed,
                    "Stopped background terminal jobs left by subagent"
                );
            }
        }
        // Whatever way the run ended, the commits it made are reported; a
        // failure to list them is reported in their place and changes nothing
        // else about the run.
        let commits = match (&commit_ledger, &commit_mark) {
            (Some(ledger), Some(mark)) => Some(CommitListing {
                listed: ledger.since(mark).await,
                setbacks: mark.setbacks().iter().map(ToString::to_string).collect(),
            }),
            _ => None,
        };
        let result = match outcome {
            SubagentOutcome::Cancelled => {
                // The cancelled query left the agent mid-turn; it cannot be continued.
                if let Some(id) = resumed {
                    self.finished.release(id);
                }
                tracing::info!(subagent = %subagent_id, "Subagent cancelled");
                return Ok(ToolResult::text(
                    SubagentResultTemplate {
                        subagent: &subagent_id,
                        outcome: SubagentReport::Cancelled,
                        commits: commits.as_ref(),
                    }
                    .render()
                    .expect("subagent result template must render"),
                ));
            }
            SubagentOutcome::Finished(Ok(result)) => result,
            SubagentOutcome::Finished(Err(error)) => {
                if let Some(id) = resumed {
                    self.finished.release(id);
                }
                let error_text =
                    failure_text(&subagent_id, &error.to_string(), commits.as_ref(), &agent);
                return Err(anyhow::anyhow!(error_text));
            }
        };

        tracing::info!(subagent = %subagent_id, "Subagent completed");

        let alert = idle_alert(&subagent_id, agent.cache_keep_alive_state());

        // A finished subagent stays continuable on every provider until
        // its prompt-cache lifetime passes.
        let id = self.finished.store(
            resumed,
            FinishedSubagent {
                agent,
                subagent: subagent_id.clone(),
                model: model.clone(),
                child_job_registry,
                commit_ledger,
                lifetime,
            },
        );

        Ok(ToolResult::text(
            SubagentResultTemplate {
                subagent: &subagent_id,
                outcome: SubagentReport::Completed {
                    model: model.as_deref(),
                    result: &result,
                    id,
                    alert: alert.as_deref(),
                },
                commits: commits.as_ref(),
            }
            .render()
            .expect("subagent result template must render"),
        ))
    }
}

/// The error text for a failed run, with the tail of its history.
fn failure_text<LLM: LanguageModel + 'static>(
    subagent: &str,
    error: &str,
    commits: Option<&CommitListing>,
    agent: &crate::Agent<LLM, LLM, LLM, SubagentHooks>,
) -> String {
    let history = agent.history();
    let skipped = history.len().saturating_sub(12);
    let recent = history
        .iter()
        .skip(skipped)
        .map(|message| HistoryEntry {
            role: message.role(),
            content: message.content().to_string(),
            tool_names: message
                .tool_calls()
                .iter()
                .map(|call| call.name.clone())
                .collect(),
        })
        .collect::<Vec<_>>();
    SubagentErrorTemplate {
        subagent,
        error,
        commits,
        history: &recent,
    }
    .render()
    .expect("subagent error template must render")
}

/// The alert telling the parent a run kept needing prompt-cache keep-alives,
/// once it crossed the frequent-idle threshold.
fn idle_alert(subagent: &str, keep_alive: &crate::agent::CacheKeepAliveState) -> Option<String> {
    keep_alive.alerted.then(|| {
        let reason = keep_alive
            .last_reason
            .clone()
            .unwrap_or_else(|| "idle without a running background job".to_string());
        format!(
            "Subagent '{subagent}' needed {} prompt-cache keep-alives during its run — subagents should not idle frequently. Reason for its most recent idle: {reason}.",
            keep_alive.sent
        )
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use aither_core::llm::{Event, LLMRequest, model::Profile as ModelProfile};
    use futures_core::Stream;

    use super::*;

    #[test]
    fn subagent_args_schema() {
        let schema = schemars::schema_for!(SubagentArgs);
        let value = schema.to_value();
        let variants = value
            .get("oneOf")
            .and_then(serde_json::Value::as_array)
            .expect("start and resume are tagged-enum variants");
        assert_eq!(variants.len(), 2);

        let requires = |variant: &serde_json::Value, field: &str| {
            variant
                .get("required")
                .and_then(serde_json::Value::as_array)
                .expect("a variant lists its required fields")
                .contains(&serde_json::json!(field))
        };
        let start = variants
            .iter()
            .find(|variant| requires(variant, "subagent"))
            .expect("one variant starts a subagent");
        let resume = variants
            .iter()
            .find(|variant| requires(variant, "id"))
            .expect("one variant continues a subagent");

        assert!(requires(start, "prompt_file"));
        assert!(
            !requires(start, "model"),
            "`model` stays optional; the caller may leave routing to the tool"
        );
        assert!(requires(resume, "prompt_file"));
        let resume_props = resume
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("resume has properties");
        assert!(
            !resume_props.contains_key("model") && !resume_props.contains_key("subagent"),
            "a continuation keeps its model and its subagent type"
        );
    }

    #[test]
    fn subagent_error_history_names_tool_calls() {
        let history = [
            HistoryEntry {
                role: aither_core::llm::Role::User,
                content: "go".to_string(),
                tool_names: Vec::new(),
            },
            HistoryEntry {
                role: aither_core::llm::Role::Assistant,
                content: String::new(),
                tool_names: vec!["terminal".to_string(), "terminal_read".to_string()],
            },
        ];
        let rendered = SubagentErrorTemplate {
            subagent: "research",
            error: "boom",
            commits: None,
            history: &history,
        }
        .render()
        .expect("subagent error should render");
        assert_eq!(
            rendered,
            "Subagent 'research' error: boom\n\nRecent subagent history:\n- User: go\n- Assistant: [called terminal, terminal_read]\n"
        );
    }

    #[test]
    fn subagent_result_renders_as_prose() {
        let completed = SubagentResultTemplate {
            subagent: "research",
            outcome: SubagentReport::Completed {
                model: Some("test-model"),
                result: "done",
                id: 7,
                alert: None,
            },
            commits: None,
        }
        .render()
        .expect("subagent result should render");
        assert_eq!(
            completed,
            "Subagent 'research' completed (model: test-model).\n\ndone\n\nContinue this subagent with `subagent resume --id 7` while its prompt cache is still warm."
        );

        let without_model = SubagentResultTemplate {
            subagent: "research",
            outcome: SubagentReport::Completed {
                model: None,
                result: "done",
                id: 7,
                alert: Some("It idled too often."),
            },
            commits: None,
        }
        .render()
        .expect("subagent result should render");
        assert_eq!(
            without_model,
            "Subagent 'research' completed.\n\ndone\n\nContinue this subagent with `subagent resume --id 7` while its prompt cache is still warm.\n\nIt idled too often."
        );

        let cancelled = SubagentResultTemplate {
            subagent: "explore",
            outcome: SubagentReport::Cancelled,
            commits: None,
        }
        .render()
        .expect("cancelled result should render");
        assert_eq!(cancelled, "Subagent 'explore' was cancelled by the user.");
    }

    #[test]
    fn commit_listings_render_as_prose() {
        let listed = CommitListing {
            listed: Ok(vec![SubagentCommit {
                worktree: PathBuf::from("/work/tree"),
                commit: "abc123".to_string(),
                subject: "fix the parser".to_string(),
            }]),
            setbacks: Vec::new(),
        };
        let completed = SubagentResultTemplate {
            subagent: "worker",
            outcome: SubagentReport::Completed {
                model: None,
                result: "done",
                id: 3,
                alert: Some("It idled too often."),
            },
            commits: Some(&listed),
        }
        .render()
        .expect("subagent result should render");
        assert_eq!(
            completed,
            "Subagent 'worker' completed.\n\ndone\n\nContinue this subagent with `subagent resume --id 3` while its prompt cache is still warm.\n\nIt idled too often.\n\nCommits made in the granted worktrees:\n- abc123 fix the parser (in /work/tree)"
        );

        let none = CommitListing {
            listed: Ok(Vec::new()),
            setbacks: Vec::new(),
        };
        let cancelled = SubagentResultTemplate {
            subagent: "worker",
            outcome: SubagentReport::Cancelled,
            commits: Some(&none),
        }
        .render()
        .expect("cancelled result should render");
        assert_eq!(
            cancelled,
            "Subagent 'worker' was cancelled by the user.\n\nNo commits were made in the granted worktrees."
        );

        let failed = CommitListing {
            listed: Err(CommitLedgerError::Git {
                repository: PathBuf::from("/repo"),
                stderr: "boom".to_string(),
            }),
            setbacks: vec!["the repository at /repo is stuck".to_string()],
        };
        let rendered = SubagentErrorTemplate {
            subagent: "worker",
            error: "model failed",
            commits: Some(&failed),
            history: &[],
        }
        .render()
        .expect("subagent error should render");
        assert_eq!(
            rendered,
            "Subagent 'worker' error: model failed\n\nThe commits made in the granted worktrees could not be listed: git failed on /repo: boom\n\nBefore the run started: the repository at /repo is stuck\n\nRecent subagent history:\n"
        );
    }

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "aither-subagent-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create test temp dir");
        dir
    }

    #[tokio::test]
    async fn resolves_file_through_mount_prefix() {
        let root = unique_temp_dir("mount-resolve");
        let skills_root = root.join("skills");
        let skill_file = skills_root.join("slide").join("SKILL.md");
        std::fs::create_dir_all(skill_file.parent().expect("skill parent"))
            .expect("create skill parent");
        std::fs::write(&skill_file, "# slide").expect("write skill file");

        let tool = SubagentTool::new(()).with_file_mounts([
            SubagentFileMount::new(".", root.join("workspace")),
            SubagentFileMount::new(".skills", skills_root.clone()),
        ]);
        let resolved = SubagentTool::<()>::resolve_subagent_path(
            &tool.effective_mounts(),
            tool.base_dir.as_deref(),
            Path::new(".skills/slide/SKILL.md"),
        )
        .await
        .expect("resolve mounted skill path");
        assert_eq!(resolved, skill_file);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn rejects_parent_traversal_paths() {
        let root = unique_temp_dir("mount-traversal");
        let tool = SubagentTool::new(())
            .with_file_mounts([SubagentFileMount::new("skills", root.join("skills"))]);

        let error = SubagentTool::<()>::resolve_subagent_path(
            &tool.effective_mounts(),
            tool.base_dir.as_deref(),
            Path::new("../outside.md"),
        )
        .await
        .expect_err("parent traversal should fail");
        assert!(
            error
                .to_string()
                .contains("outside mounted paths or uses forbidden path components"),
            "unexpected error: {error}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn missing_file_error_does_not_leak_host_mount_path() {
        let root = unique_temp_dir("mount-errors");
        let skills_root = root.join("skills");
        std::fs::create_dir_all(&skills_root).expect("create skills root");
        let tool = SubagentTool::new(())
            .with_file_mounts([SubagentFileMount::new("skills", skills_root.clone())]);

        let error = SubagentTool::<()>::resolve_subagent_path(
            &tool.effective_mounts(),
            tool.base_dir.as_deref(),
            Path::new(".skills/missing.md"),
        )
        .await
        .expect_err("missing file should fail");
        let message = error.to_string();
        assert!(
            message.contains("skills/missing.md"),
            "missing virtual path in error: {message}"
        );
        assert!(
            !message.contains(skills_root.to_string_lossy().as_ref()),
            "host path leaked in error: {message}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[derive(Debug)]
    struct NeverError;

    impl std::fmt::Display for NeverError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("never resolves")
        }
    }

    impl std::error::Error for NeverError {}

    /// A model whose `respond` stream never yields, so the child query
    /// stays in flight until the run is cancelled.
    #[derive(Clone)]
    struct PendingLlm;

    impl LanguageModel for PendingLlm {
        type Error = NeverError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::stream::pending()
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "pending-model",
                "pending model",
                128_000,
            )))
        }
    }

    /// A model that answers with its own name, so a result identifies which
    /// configured model produced it. Structured-output requests get a fixed
    /// JSON body back, which lets the fast model play router.
    #[derive(Clone)]
    struct TextLlm {
        name: &'static str,
        structured: &'static str,
    }

    impl TextLlm {
        const fn new(name: &'static str) -> Self {
            Self {
                name,
                structured: "{}",
            }
        }

        const fn routing(name: &'static str, decision: &'static str) -> Self {
            Self {
                name,
                structured: decision,
            }
        }
    }

    impl LanguageModel for TextLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let text = if request.parameters().structured_outputs {
                self.structured
            } else {
                self.name
            };
            futures_lite::stream::iter([Ok(Event::Text(text.to_string()))])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                self.name,
                "test",
                self.name,
                "text mock",
                128_000,
            )))
        }
    }

    fn write_prompt_file(tag: &str, content: &str) -> (PathBuf, PathBuf) {
        let dir = unique_temp_dir(tag);
        let prompt_file = dir.join("prompt.txt");
        std::fs::write(&prompt_file, content).expect("write prompt file");
        (dir, prompt_file)
    }

    fn worker_tool<LLM: LanguageModel + Clone + Sync>(
        tool: SubagentTool<LLM>,
    ) -> SubagentTool<LLM> {
        tool.with_type(
            "worker",
            SubagentType::new("worker for tests", |llm| {
                crate::AgentBuilder::new(llm)
                    .agent_kind(crate::AgentKind::Chatbot)
                    .max_iterations(2)
            }),
        )
    }

    #[tokio::test]
    async fn cancelled_run_returns_cancelled_result() {
        let runs = SubagentRuns::default();
        let tool = SubagentTool::new(PendingLlm)
            .with_runs(runs.clone())
            .with_type(
                "explore",
                SubagentType::new("explorer", |llm| {
                    crate::AgentBuilder::new(llm)
                        .agent_kind(crate::AgentKind::Chatbot)
                        .max_iterations(5)
                }),
            );
        let (dir, prompt_file) = write_prompt_file("cancelled-run", "survey the codebase");

        let prompt_path = prompt_file.to_string_lossy().into_owned();
        let call = tokio::spawn(async move {
            Tool::call(
                &tool,
                SubagentArgs::Start {
                    worktrees: Vec::new(),
                    commit_worktrees: Vec::new(),
                    subagent: "explore".to_string(),
                    prompt_file: prompt_path,
                    model: Some(ModelTier::Fast),
                },
                ToolContext::new(),
            )
            .await
        });

        // Wait for the run to register before cancelling it.
        let run = loop {
            match runs.list().as_slice() {
                [run] => break run.clone(),
                [] => tokio::task::yield_now().await,
                other => panic!("expected one registered run, got {}", other.len()),
            }
        };
        assert_eq!(run.subagent, "explore");
        assert_eq!(run.prompt, "survey the codebase");
        assert!(runs.cancel(run.id));
        assert!(runs.cancel(run.id), "re-cancel should still report true");

        let result = call
            .await
            .expect("call task panicked")
            .expect("subagent call failed");
        let ToolResult::Text { text } = result else {
            panic!("expected text result, got {result:?}");
        };
        assert_eq!(text, "Subagent 'explore' was cancelled by the user.");
        assert!(runs.list().is_empty(), "finished run must unregister");
        assert!(!runs.cancel(run.id), "finished run cannot be cancelled");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Runs `git` with a fixed identity in `dir` and returns what it printed.
    fn git(dir: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .args(["-c", "user.name=test", "-c", "user.email=test@example.com"])
            .args(["-c", "commit.gpgSign=false"])
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// A main agent's terminal whose space holds a repository and a linked
    /// worktree on `subagent/work/main`, the worktree a subagent is granted
    /// commit in.
    struct CommitFixture {
        _dirs: [tempfile::TempDir; 3],
        terminal: aither_sandbox::TerminalTool<
            aither_sandbox::permission::NoopPermissionHandler,
            executor_core::tokio::TokioGlobal,
            aither_sandbox::Configured,
        >,
        repository: PathBuf,
        worktree: PathBuf,
    }

    impl CommitFixture {
        async fn new() -> Self {
            use aither_sandbox::permission::NoopPermissionHandler;
            use aither_sandbox::{TerminalTool, ToolRegistryBuilder};
            use executor_core::tokio::TokioGlobal;

            let work = tempfile::tempdir().unwrap();
            let source = tempfile::tempdir().unwrap();
            let cache = tempfile::tempdir().unwrap();
            let binary = crate::test_support::compile_baseline_fixture(source.path()).await;
            let distribution = crate::test_support::fixture_distribution(&binary, "1.0.0").await;

            let session = work.path().join("session");
            let repository = session.join("repository");
            std::fs::create_dir_all(&repository).unwrap();
            git(&repository, &["init", "--initial-branch=main"]);
            git(&repository, &["commit", "--allow-empty", "-m", "initial"]);
            let worktree = session.join("worktree");
            git(
                &repository,
                &[
                    "worktree",
                    "add",
                    "-b",
                    "subagent/work/main",
                    worktree.to_str().unwrap(),
                ],
            );

            let terminal = TerminalTool::new_exact(&session, NoopPermissionHandler, TokioGlobal)
                .await
                .expect("terminal tool should initialize")
                .initialize_tools(&distribution, cache.path())
                .await
                .expect("tools should prepare");
            let registry = Arc::new(ToolRegistryBuilder::new().build(terminal.outputs_dir()));
            let terminal = terminal.with_registry(registry);
            Self {
                _dirs: [work, source, cache],
                terminal,
                repository,
                worktree,
            }
        }

        /// A subagent tool bound to the fixture's terminal.
        fn tool<LLM: LanguageModel + Clone + Sync + 'static>(
            &self,
            llm: LLM,
            runs: &SubagentRuns,
        ) -> SubagentTool<LLM> {
            SubagentTool::new(llm)
                .with_runs(runs.clone())
                .with_type(
                    "worker",
                    SubagentType::new("worker", |llm| {
                        crate::AgentBuilder::new(llm)
                            .agent_kind(crate::AgentKind::Chatbot)
                            .max_iterations(5)
                    }),
                )
                .for_terminal(self.terminal.binding())
        }

        /// Starts a subagent granted commit in the fixture's worktree.
        fn start(&self, tag: &str) -> (PathBuf, SubagentArgs) {
            let (dir, prompt_file) = write_prompt_file(tag, "work in the worktree");
            (
                dir,
                SubagentArgs::Start {
                    worktrees: Vec::new(),
                    commit_worktrees: vec![self.worktree.to_string_lossy().into_owned()],
                    subagent: "worker".to_string(),
                    prompt_file: prompt_file.to_string_lossy().into_owned(),
                    model: Some(ModelTier::Fast),
                },
            )
        }
    }

    /// What [`CommittingLlm`] does in the worktree before it answers.
    #[derive(Clone, Copy)]
    enum CommitStep {
        /// Commits, then answers.
        Commit,
        /// Commits, then fails the request.
        CommitAndFail,
        /// Moves the repository's object store away, so no listing can read
        /// it, then answers.
        BreakRepository,
    }

    /// A model that acts in a worktree, standing in for a subagent's
    /// commands, then answers or fails.
    #[derive(Clone)]
    struct CommittingLlm {
        repository: PathBuf,
        worktree: PathBuf,
        step: CommitStep,
    }

    impl LanguageModel for CommittingLlm {
        type Error = NeverError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let answer = match self.step {
                CommitStep::Commit => {
                    git(
                        &self.worktree,
                        &["commit", "--allow-empty", "-m", "made by the subagent"],
                    );
                    Ok(Event::Text("done".to_string()))
                }
                CommitStep::CommitAndFail => {
                    git(
                        &self.worktree,
                        &["commit", "--allow-empty", "-m", "made before failing"],
                    );
                    Err(NeverError)
                }
                CommitStep::BreakRepository => {
                    let git_dir = self.repository.join(".git");
                    std::fs::rename(git_dir.join("objects"), git_dir.join("objects-gone")).unwrap();
                    Ok(Event::Text("done".to_string()))
                }
            };
            futures_lite::stream::iter([answer])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "test",
                "committing-model",
                "commits in a worktree",
                128_000,
            )))
        }
    }

    impl CommitFixture {
        fn llm(&self, step: CommitStep) -> CommittingLlm {
            CommittingLlm {
                repository: self.repository.clone(),
                worktree: self.worktree.clone(),
                step,
            }
        }
    }

    /// The subjects of the commits a result or failure text lists.
    fn listed_subjects(text: &str) -> Vec<&str> {
        let (_, listing) = text
            .split_once("Commits made in the granted worktrees:\n")
            .unwrap_or_else(|| panic!("the text lists commits: {text}"));
        listing
            .lines()
            .map_while(|line| line.strip_prefix("- "))
            .map(|line| {
                let (_, rest) = line.split_once(' ').expect("a commit line names its id");
                let (subject, _) = rest
                    .rsplit_once(" (in ")
                    .expect("a commit line names its worktree");
                subject
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_completed_run_lists_the_commits_made_in_its_granted_worktree() {
        let fixture = CommitFixture::new().await;
        let runs = SubagentRuns::default();
        let tool = fixture.tool(fixture.llm(CommitStep::Commit), &runs);
        let (dir, args) = fixture.start("completed-commits");

        let result = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect("the run completes");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");
        assert_eq!(listed_subjects(&text), ["made by the subagent"], "{text}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_run_lists_the_commits_made_in_its_granted_worktree() {
        let fixture = CommitFixture::new().await;
        let runs = SubagentRuns::default();
        let tool = fixture.tool(fixture.llm(CommitStep::CommitAndFail), &runs);
        let (dir, args) = fixture.start("failed-commits");

        let error = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect_err("the run fails");
        let failure = error.to_string();
        assert!(
            failure.starts_with("Subagent 'worker' error: "),
            "{failure}"
        );
        assert!(failure.contains("never resolves"), "{failure}");
        assert_eq!(
            listed_subjects(&failure),
            ["made before failing"],
            "{failure}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_listing_failure_keeps_the_runs_outcome() {
        let fixture = CommitFixture::new().await;
        let runs = SubagentRuns::default();
        let tool = fixture.tool(fixture.llm(CommitStep::BreakRepository), &runs);
        let (dir, args) = fixture.start("broken-listing");

        let result = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect("the run's outcome survives the listing failure");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");
        assert!(text.contains("\n\ndone\n\n"), "{text}");
        assert_eq!(
            continuation_id(&text),
            1,
            "the subagent stays continuable: {text}"
        );
        assert!(
            text.contains("The commits made in the granted worktrees could not be listed: "),
            "the listing failure is reported in place of the commits: {text}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancelled_run_lists_the_commits_made_in_its_granted_worktree() {
        let fixture = CommitFixture::new().await;
        let runs = SubagentRuns::default();
        let tool = fixture.tool(PendingLlm, &runs);
        let (dir, args) = fixture.start("cancelled-commits");
        let call = tokio::spawn(async move { Tool::call(&tool, args, ToolContext::new()).await });

        let run = loop {
            match runs.list().as_slice() {
                [run] => break run.clone(),
                [] => tokio::task::yield_now().await,
                other => panic!("expected one registered run, got {}", other.len()),
            }
        };
        // The commit lands after the run began and before it is cancelled.
        git(
            &fixture.worktree,
            &["commit", "--allow-empty", "-m", "made during the run"],
        );
        assert!(runs.cancel(run.id));

        let result = call
            .await
            .expect("call task panicked")
            .expect("subagent call failed");
        let text = result.render_for_model();
        assert!(
            text.starts_with("Subagent 'worker' was cancelled by the user."),
            "{text}"
        );
        assert_eq!(listed_subjects(&text), ["made during the run"], "{text}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_grant_without_a_terminal_is_refused() {
        let tool = worker_tool(SubagentTool::new(TextLlm::new("only-model")));
        let (dir, prompt_file) = write_prompt_file("grant-without-terminal", "do it");
        let error = Tool::call(
            &tool,
            SubagentArgs::Start {
                worktrees: vec!["somewhere".to_string()],
                commit_worktrees: Vec::new(),
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: Some(ModelTier::Fast),
            },
            ToolContext::new(),
        )
        .await
        .expect_err("a grant cannot be given without a terminal");
        assert!(
            error.to_string().contains("launched from a terminal"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn inline_prompt_text_is_rejected() {
        let tool = worker_tool(SubagentTool::new(TextLlm::new("only-model")));

        let error = Tool::call(
            &tool,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: "worker".to_string(),
                prompt_file: "summarize the repository layout".to_string(),
                model: Some(ModelTier::Fast),
            },
            ToolContext::new(),
        )
        .await
        .expect_err("inline prompt text must be rejected");
        assert!(
            error.to_string().contains("prompt file"),
            "error should name the file requirement: {error}"
        );
    }

    #[tokio::test]
    async fn empty_prompt_file_is_rejected() {
        let tool = worker_tool(SubagentTool::new(TextLlm::new("only-model")));
        let (dir, prompt_file) = write_prompt_file("empty-prompt", "   \n");

        let error = Tool::call(
            &tool,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: Some(ModelTier::Fast),
            },
            ToolContext::new(),
        )
        .await
        .expect_err("empty prompt file must be rejected");
        assert!(
            error.to_string().contains("is empty"),
            "error should report the empty file: {error}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn explicit_model_runs_as_given() {
        let tool = worker_tool(
            SubagentTool::new(TextLlm::new("advanced-model"))
                .with_balanced_model(TextLlm::new("balanced-model"))
                .with_fast_model(TextLlm::new("fast-model")),
        );
        let (dir, prompt_file) = write_prompt_file("explicit-model", "do the task");

        let result = Tool::call(
            &tool,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: Some(ModelTier::Fast),
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let ToolResult::Text { text } = result else {
            panic!("expected text result, got {result:?}");
        };
        assert_eq!(
            text,
            "Subagent 'worker' completed (model: fast-model).\n\nfast-model\n\nContinue this subagent with `subagent resume --id 1` while its prompt cache is still warm."
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn missing_model_is_routed_by_fast_model() {
        let tool = worker_tool(
            SubagentTool::new(TextLlm::new("advanced-model"))
                .with_balanced_model(TextLlm::new("balanced-model"))
                .with_fast_model(TextLlm::routing("fast-model", r#"{"model":"balanced"}"#)),
        );
        let (dir, prompt_file) = write_prompt_file("routed-model", "do the task");

        let result = Tool::call(
            &tool,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: None,
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let ToolResult::Text { text } = result else {
            panic!("expected text result, got {result:?}");
        };
        assert_eq!(
            text,
            "Subagent 'worker' completed (model: balanced-model).\n\nbalanced-model\n\nContinue this subagent with `subagent resume --id 1` while its prompt cache is still warm."
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn definition_model_is_used_when_caller_gives_none() {
        let root = unique_temp_dir("definition-model");
        let def_path = root.join("file-agent.md");
        std::fs::write(
            &def_path,
            r"---
version: 1
name: file-agent
description: agent loaded from a file
model: fast
---

You are a file-defined agent.
",
        )
        .expect("write subagent definition");
        let (dir, prompt_file) = write_prompt_file("definition-model-prompt", "do the task");

        let tool = SubagentTool::new(TextLlm::new("advanced-model"))
            .with_balanced_model(TextLlm::new("balanced-model"))
            .with_fast_model(TextLlm::new("fast-model"));

        let result = Tool::call(
            &tool,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: def_path.to_string_lossy().into_owned(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: None,
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let ToolResult::Text { text } = result else {
            panic!("expected text result, got {result:?}");
        };
        assert_eq!(
            text,
            "Subagent 'file-agent' completed (model: fast-model).\n\nfast-model\n\nContinue this subagent with `subagent resume --id 1` while its prompt cache is still warm."
        );

        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A model that records every request and answers with a scripted
    /// turn sequence, so tests can watch for `max_tokens: 0` keep-alives.
    #[derive(Clone)]
    struct KeepAliveMockLlm {
        author: &'static str,
        requests: async_channel::Sender<LLMRequest>,
        turns: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LanguageModel for KeepAliveMockLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests
                .try_send(request.clone())
                .expect("record model request");
            if request.parameters().max_tokens == Some(0) {
                // A keep-alive re-send: report it as a cache read and end.
                return futures_lite::stream::iter([Ok(Event::Usage(aither_core::llm::Usage {
                    prompt_tokens: Some(10),
                    completion_tokens: Some(0),
                    total_tokens: Some(10),
                    reasoning_tokens: None,
                    cache_read_tokens: Some(10),
                    cache_write_tokens: None,
                    cost_usd: None,
                    stop_reason: Some("max_tokens".to_string()),
                }))]);
            }
            let turn = self.turns.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let event = match turn {
                0 => Event::ToolCall(aither_core::llm::ToolCall::new(
                    "bg-1",
                    "terminal_background",
                    serde_json::json!({
                        "description": "hold the cache window",
                        "script": "sleep 2",
                        "timeout": 60
                    }),
                )),
                _ => Event::Text("done".to_string()),
            };
            futures_lite::stream::iter([Ok(event)])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            let author = self.author;
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                author,
                "mock-model",
                "mock model",
                100_000,
            )))
        }
    }

    /// Builds a subagent tool whose child agent waits on a background
    /// channel driven by the test — no terminal backend needed. Returns the
    /// tool plus the sender halves and the child's job registry, so the
    /// test can park the run and register the command it waits on.
    fn keep_alive_tool(
        author: &'static str,
        requests: async_channel::Sender<LLMRequest>,
        keep_alive: Option<crate::config::CacheKeepAlive>,
    ) -> (
        SubagentTool<KeepAliveMockLlm>,
        aither_sandbox::BackgroundTaskSender,
        aither_sandbox::JobRegistry,
        aither_sandbox::JobRegistryService,
    ) {
        let (sender, receiver) = aither_sandbox::background_task_channel();
        let (job_registry, job_service) = aither_sandbox::job_registry_channel();
        let child_jobs = job_registry.clone();
        let tool = SubagentTool::new(KeepAliveMockLlm {
            author,
            requests,
            turns: Arc::default(),
        });
        let tool = match keep_alive {
            Some(keep_alive) => tool.with_cache_keep_alive(keep_alive),
            None => tool,
        }
        .with_type(
            "worker",
            SubagentType::new("worker for keep-alive tests", move |llm| {
                crate::AgentBuilder::new(llm)
                    .agent_kind(crate::AgentKind::Chatbot)
                    .max_iterations(4)
                    .background_channels(receiver.clone(), child_jobs.clone())
            }),
        );
        (tool, sender, job_registry, job_service)
    }

    /// Registers a fake running background job carrying `script` so the
    /// run's idle reason names it.
    async fn register_running_job(
        job_registry: &aither_sandbox::JobRegistry,
        pid: u32,
        script: &str,
    ) {
        job_registry
            .register(
                pid,
                "amber-test-task",
                "exec-key",
                script,
                aither_sandbox::TerminalMode::Sandboxed,
                None,
            )
            .await;
    }

    fn keep_alive_requests(requests: &async_channel::Receiver<LLMRequest>) -> Vec<LLMRequest> {
        let mut drained = Vec::new();
        while let Ok(request) = requests.try_recv() {
            drained.push(request);
        }
        drained
            .into_iter()
            .filter(|request| request.parameters().max_tokens == Some(0))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn anthropic_subagent_waiting_on_background_command_gets_keep_alives() {
        let (dir, prompt_file) = write_prompt_file("keep-alive-warm-prompt", "hold a job");
        let (request_tx, request_rx) = async_channel::unbounded();
        let (tool, sender, job_registry, job_service) = keep_alive_tool(
            "Anthropic",
            request_tx,
            Some(crate::config::CacheKeepAlive {
                // The Claude-client subagent runs on the real five-minute
                // lifetime; sending this long before expiry makes the
                // first keep-alive due 250 ms into the wait.
                send_before_expiry: std::time::Duration::from_millis(299_750),
                frequent_idle_threshold: 2,
            }),
        );
        tokio::spawn(job_service.serve());
        register_running_job(&job_registry, 4242, "sleep 2").await;
        sender.task_started();

        let call = tokio::spawn(async move {
            Tool::call(
                &tool,
                SubagentArgs::Start {
                    worktrees: Vec::new(),
                    commit_worktrees: Vec::new(),
                    subagent: "worker".to_string(),
                    prompt_file: prompt_file.to_string_lossy().into_owned(),
                    model: Some(ModelTier::Fast),
                },
                ToolContext::new(),
            )
            .await
        });

        // Hold the job open until two keep-alives have gone out, then
        // complete the task so the wait unwinds.
        let mut keep_alives = Vec::new();
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            while keep_alives.len() < 2 {
                let request = request_rx.recv().await.expect("request channel closed");
                if request.parameters().max_tokens == Some(0) {
                    keep_alives.push(request);
                }
            }
        })
        .await
        .expect("two keep-alives while the job ran");
        sender.task_finished();
        sender
            .send_completed(aither_sandbox::CompletedTask {
                task_id: "amber-test-task".to_string(),
                script: "sleep 2".to_string(),
                result: Ok(aither_sandbox::CompletedCommand {
                    stdout: aither_core::llm::CommandStream::Inline {
                        text: String::new(),
                    },
                    stderr: None,
                    exit_code: 0,
                }),
            })
            .await;

        let result = call
            .await
            .expect("call task panicked")
            .expect("subagent call failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");

        keep_alives.extend(keep_alive_requests(&request_rx));
        for request in &keep_alives {
            let params = request.parameters();
            assert_eq!(params.max_tokens, Some(0));
            assert_eq!(
                params.stream,
                Some(false),
                "keep-alive must be non-streaming"
            );
            assert_eq!(
                params.reasoning_effort, None,
                "keep-alive must preserve the last request's effort"
            );
            assert!(
                !params.include_reasoning,
                "keep-alive must preserve the last request's thinking"
            );
        }

        // Two keep-alives crossed the configured threshold: the parent gets
        // one alert naming the subagent and the command it waited on.
        assert!(
            text.contains("Subagent 'worker' needed"),
            "alert names the subagent: {text}"
        );
        assert!(
            text.contains("sleep 2"),
            "alert gives the idle reason: {text}"
        );
        assert!(text.contains("should not idle frequently"), "{text}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn subagent_without_keep_alive_config_gets_no_keep_alives() {
        let (dir, prompt_file) = write_prompt_file("keep-alive-other-prompt", "hold a job");
        let (request_tx, request_rx) = async_channel::unbounded();
        let (tool, sender, job_registry, job_service) =
            keep_alive_tool("Anthropic", request_tx, None);
        tokio::spawn(job_service.serve());
        register_running_job(&job_registry, 4243, "sleep 2").await;
        sender.task_started();

        let call = tokio::spawn(async move {
            Tool::call(
                &tool,
                SubagentArgs::Start {
                    worktrees: Vec::new(),
                    commit_worktrees: Vec::new(),
                    subagent: "worker".to_string(),
                    prompt_file: prompt_file.to_string_lossy().into_owned(),
                    model: Some(ModelTier::Fast),
                },
                ToolContext::new(),
            )
            .await
        });

        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        sender.task_finished();
        sender
            .send_completed(aither_sandbox::CompletedTask {
                task_id: "amber-test-task".to_string(),
                script: "sleep 2".to_string(),
                result: Ok(aither_sandbox::CompletedCommand {
                    stdout: aither_core::llm::CommandStream::Inline {
                        text: String::new(),
                    },
                    stderr: None,
                    exit_code: 0,
                }),
            })
            .await;

        let result = call
            .await
            .expect("call task panicked")
            .expect("subagent call failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");
        assert!(
            keep_alive_requests(&request_rx).is_empty(),
            "no keep-alive config means no keep-alives"
        );
        assert!(
            !text.contains("keep-alives during its run"),
            "no alert without keep-alives: {text}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn finished_subagent_gets_no_keep_alives() {
        let (request_tx, request_rx) = async_channel::unbounded();
        let llm = KeepAliveMockLlm {
            author: "Anthropic",
            requests: request_tx,
            turns: Arc::new(std::sync::atomic::AtomicUsize::new(1)),
        };
        let tool = SubagentTool::new(llm).with_type(
            "worker",
            SubagentType::new("worker", |llm| {
                crate::AgentBuilder::new(llm)
                    .agent_kind(crate::AgentKind::Chatbot)
                    .max_iterations(2)
            }),
        );
        let (dir, prompt_file) = write_prompt_file("keep-alive-idle", "answer directly");

        let result = Tool::call(
            &tool,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: Some(ModelTier::Fast),
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");
        assert!(
            keep_alive_requests(&request_rx).is_empty(),
            "a run that never waits on a background job gets no keep-alive"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn absolute_workspace_skills_path_resolves_to_skills_mount() {
        let root = unique_temp_dir("absolute-workspace-skills");
        let workspace_root = root.join("workspace");
        let skills_root = root.join("skills");
        let session_skills_file = workspace_root.join("skills/slide/subagents/art_direction.md");
        let mounted_skill_file = skills_root.join("slide/subagents/art_direction.md");
        std::fs::create_dir_all(session_skills_file.parent().expect("session skill parent"))
            .expect("create session skill parent");
        std::fs::create_dir_all(mounted_skill_file.parent().expect("mounted skill parent"))
            .expect("create mounted skill parent");
        std::fs::write(&mounted_skill_file, "# art direction").expect("write mounted skill file");

        let tool = SubagentTool::new(())
            .with_base_dir(workspace_root.clone())
            .with_file_mounts([
                SubagentFileMount::new(".", workspace_root.clone()),
                SubagentFileMount::new("skills", skills_root.clone()),
            ]);

        let resolved = SubagentTool::<()>::resolve_subagent_path(
            &tool.effective_mounts(),
            tool.base_dir.as_deref(),
            &session_skills_file,
        )
        .await
        .expect("resolve absolute workspace .skills path");
        assert_eq!(resolved, mounted_skill_file);

        let _ = std::fs::remove_dir_all(root);
    }

    /// Records every request and answers each with a numbered reply, so a
    /// continuation's request can be checked for the earlier conversation.
    #[derive(Clone)]
    struct EchoLlm {
        requests: async_channel::Sender<LLMRequest>,
        replies: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LanguageModel for EchoLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.requests
                .try_send(request)
                .expect("record model request");
            let n = self
                .replies
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            futures_lite::stream::iter([Ok(Event::Text(format!("reply-{n}")))])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "mock",
                "Anthropic",
                "mock-model",
                "mock model",
                100_000,
            )))
        }
    }

    /// `keep_alive` marks the tool as serving the Claude client; `lifetime`
    /// overrides the cache lifetime the subagent runs with, and `None`
    /// leaves the one `SubagentTool` itself assigns.
    fn resume_tool(
        lifetime: Option<CacheLifetime>,
        keep_alive: Option<crate::config::CacheKeepAlive>,
    ) -> (SubagentTool<EchoLlm>, async_channel::Receiver<LLMRequest>) {
        let (request_tx, request_rx) = async_channel::unbounded();
        let tool = SubagentTool::new(EchoLlm {
            requests: request_tx,
            replies: Arc::default(),
        });
        let tool = match keep_alive {
            Some(keep_alive) => tool.with_cache_keep_alive(keep_alive),
            None => tool,
        }
        .with_type(
            "worker",
            SubagentType::new("worker for resume tests", move |llm| {
                let builder = crate::AgentBuilder::new(llm)
                    .agent_kind(crate::AgentKind::Chatbot)
                    .max_iterations(2);
                match lifetime {
                    Some(lifetime) => builder.prompt_cache(
                        crate::context::PromptCache::new("resume-test").with_lifetime(lifetime),
                    ),
                    None => builder,
                }
            }),
        );
        (tool, request_rx)
    }

    fn resume_args(prompt_file: &Path, resume: u64) -> SubagentArgs {
        SubagentArgs::Resume {
            id: resume,
            prompt_file: prompt_file.to_string_lossy().into_owned(),
        }
    }

    /// The continuation id a result text names.
    fn continuation_id(text: &str) -> u64 {
        let tail = text
            .split("subagent resume --id ")
            .nth(1)
            .unwrap_or_else(|| panic!("result names a continuation id: {text}"));
        tail.chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .expect("continuation id is a number")
    }

    /// Runs the first call and returns its result text and continuation id.
    async fn first_run(tool: &SubagentTool<EchoLlm>, prompt_file: &Path) -> (String, u64) {
        let result = Tool::call(
            tool,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: Some(ModelTier::Fast),
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let text = result.render_for_model();
        let id = continuation_id(&text);
        (text, id)
    }

    #[tokio::test]
    async fn finished_subagent_continues_with_its_earlier_conversation() {
        let (tool, requests) = resume_tool(
            None,
            Some(crate::config::CacheKeepAlive {
                send_before_expiry: std::time::Duration::ZERO,
                frequent_idle_threshold: 3,
            }),
        );
        let (dir, first_prompt) = write_prompt_file("resume-first", "first question marker");
        let (first, id) = first_run(&tool, &first_prompt).await;
        assert!(first.starts_with("Subagent 'worker' completed"), "{first}");
        assert!(first.contains("\n\nreply-0\n\n"), "{first}");
        while requests.try_recv().is_ok() {}

        let follow_up = dir.join("follow-up.txt");
        std::fs::write(&follow_up, "second question marker").expect("write follow-up");
        let result = Tool::call(&tool, resume_args(&follow_up, id), ToolContext::new())
            .await
            .expect("continuing failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");
        assert!(text.contains("\n\nreply-1\n\n"), "{text}");
        assert_eq!(
            continuation_id(&text),
            id,
            "a continued subagent keeps its id"
        );

        let request = requests.try_recv().expect("continuation sent a request");
        let sent = serde_json::to_string(request.messages()).expect("serialize messages");
        assert!(sent.contains("first question marker"), "{sent}");
        assert!(sent.contains("reply-0"), "{sent}");
        assert!(sent.contains("second question marker"), "{sent}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn continuing_an_unknown_id_is_an_error() {
        let (tool, _requests) = resume_tool(None, None);
        let (dir, prompt_file) = write_prompt_file("resume-unknown", "follow-up");
        let error = Tool::call(&tool, resume_args(&prompt_file, 99), ToolContext::new())
            .await
            .expect_err("unknown id must fail");
        assert!(
            error.to_string().contains("unknown subagent id 99"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Pretends `age` has passed since the finished subagent's last request,
    /// so expiry runs against its real cache lifetime without sleeping.
    fn age_finished_subagent(tool: &SubagentTool<EchoLlm>, id: u64, age: std::time::Duration) {
        let mut inner = tool
            .finished
            .inner
            .lock()
            .expect("finished subagents lock poisoned");
        let Some(FinishedSlot::Idle(finished)) = inner.slots.get_mut(&id) else {
            panic!("subagent {id} is not idle");
        };
        finished.agent.last_request_started_at = Instant::now().checked_sub(age);
        drop(inner);
    }

    #[tokio::test]
    async fn claude_subagent_expires_after_its_five_minute_lifetime() {
        let (tool, _requests) = resume_tool(None, Some(crate::config::CacheKeepAlive::default()));
        let (dir, prompt_file) = write_prompt_file("resume-expired", "first");
        let (_, id) = first_run(&tool, &prompt_file).await;

        age_finished_subagent(&tool, id, std::time::Duration::from_secs(301));
        let error = Tool::call(&tool, resume_args(&prompt_file, id), ToolContext::new())
            .await
            .expect_err("expired subagent must fail");
        let message = error.to_string();
        assert!(message.contains("expired"), "{message}");
        assert!(
            message.contains("300s"),
            "the error names the lifetime the id expired after: {message}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn claude_subagent_continues_just_inside_its_five_minute_lifetime() {
        let (tool, _requests) = resume_tool(None, Some(crate::config::CacheKeepAlive::default()));
        let (dir, prompt_file) = write_prompt_file("resume-inside", "first");
        let (_, id) = first_run(&tool, &prompt_file).await;

        age_finished_subagent(&tool, id, std::time::Duration::from_secs(290));
        Tool::call(&tool, resume_args(&prompt_file, id), ToolContext::new())
            .await
            .expect("a subagent inside its lifetime continues");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A subagent on another client keeps the lifetime of its provider, not
    /// the Claude five minutes.
    #[tokio::test]
    async fn non_claude_subagent_keeps_the_default_lifetime() {
        let (tool, _requests) = resume_tool(None, None);
        let (dir, prompt_file) = write_prompt_file("resume-default-lifetime", "first");
        let (_, id) = first_run(&tool, &prompt_file).await;

        age_finished_subagent(&tool, id, std::time::Duration::from_secs(301));
        Tool::call(&tool, resume_args(&prompt_file, id), ToolContext::new())
            .await
            .expect("past five minutes a non-Claude subagent still continues");

        age_finished_subagent(&tool, id, std::time::Duration::from_secs(3601));
        let error = Tool::call(&tool, resume_args(&prompt_file, id), ToolContext::new())
            .await
            .expect_err("expired subagent must fail");
        assert!(error.to_string().contains("3600s"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The request a Claude-client subagent sends asks for the five-minute
    /// cache (which the Claude client serializes without a `ttl` field),
    /// while the main agent's asks for one hour.
    #[tokio::test]
    async fn claude_subagent_requests_five_minute_cache_and_main_agent_one_hour() {
        use aither_core::llm::model::ClaudePromptCacheTtl;

        let claude_ttl = |request: &LLMRequest| {
            request
                .parameters()
                .cache
                .claude
                .as_ref()
                .expect("request carries Claude cache control")
                .ttl
        };

        let (tool, subagent_requests) =
            resume_tool(None, Some(crate::config::CacheKeepAlive::default()));
        let (dir, prompt_file) = write_prompt_file("cache-ttl-claude", "first");
        first_run(&tool, &prompt_file).await;
        let subagent_request = subagent_requests.try_recv().expect("subagent request");
        assert_eq!(
            claude_ttl(&subagent_request),
            ClaudePromptCacheTtl::FiveMinutes
        );

        let (other, other_requests) = resume_tool(None, None);
        first_run(&other, &prompt_file).await;
        let other_request = other_requests.try_recv().expect("other subagent request");
        assert_eq!(claude_ttl(&other_request), ClaudePromptCacheTtl::OneHour);

        let (main_tx, main_requests) = async_channel::unbounded();
        let mut main_agent = crate::AgentBuilder::new(EchoLlm {
            requests: main_tx,
            replies: Arc::default(),
        })
        .agent_kind(crate::AgentKind::Chatbot)
        .build();
        main_agent.query("hello").await.expect("main agent query");
        let main_request = main_requests.try_recv().expect("main agent request");
        assert_eq!(claude_ttl(&main_request), ClaudePromptCacheTtl::OneHour);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The `resume` shape carries no model or subagent selector at all, so
    /// an argument object that names one fails to deserialize.
    #[test]
    fn resume_args_carry_no_model_or_subagent() {
        for extra in ["model", "subagent"] {
            let error = serde_json::from_value::<SubagentArgs>(serde_json::json!({
                "command": "resume",
                "id": 7,
                "prompt_file": "prompt.txt",
                extra: "fast",
            }))
            .expect_err("a resume call cannot name a model or subagent");
            assert!(
                error.to_string().contains("unknown field"),
                "{extra} must be rejected as unknown: {error}"
            );
        }
    }

    /// Continuation does not depend on a keep-alive configuration: every
    /// finished subagent gets an id, and it stays continuable until the
    /// prompt-cache lifetime it ran with passes.
    #[tokio::test]
    async fn finished_subagent_without_keep_alive_still_continues() {
        let (tool, requests) = resume_tool(None, None);
        let (dir, first_prompt) = write_prompt_file("resume-no-keepalive", "first marker");
        let (first, id) = first_run(&tool, &first_prompt).await;
        assert!(first.starts_with("Subagent 'worker' completed"), "{first}");
        while requests.try_recv().is_ok() {}

        let follow_up = dir.join("follow-up.txt");
        std::fs::write(&follow_up, "second marker").expect("write follow-up");
        let result = Tool::call(&tool, resume_args(&follow_up, id), ToolContext::new())
            .await
            .expect("continuing failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");
        assert!(text.contains("\n\nreply-1\n\n"), "{text}");
        assert_eq!(continuation_id(&text), id);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A permanent prompt-cache lifetime never expires its continuation id.
    #[tokio::test]
    async fn permanent_cache_lifetime_never_expires() {
        let (tool, _requests) = resume_tool(Some(CacheLifetime::Permanent), None);
        let (dir, first_prompt) = write_prompt_file("resume-permanent", "first");
        let (_, id) = first_run(&tool, &first_prompt).await;

        // Well past the lifetime the expiry test uses; a permanent cache
        // never goes cold.
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let follow_up = dir.join("follow-up.txt");
        std::fs::write(&follow_up, "second").expect("write follow-up");
        let result = Tool::call(&tool, resume_args(&follow_up, id), ToolContext::new())
            .await
            .expect("a permanent lifetime never expires");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'worker' completed"), "{text}");
        assert_eq!(continuation_id(&text), id);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Handoff a scripted model answers a compaction request with.
    const FORK_HANDOFF_JSON: &str = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#;

    /// A model that records every request with its own name and answers with
    /// scripted turns, shared by its clones, so a test can tell which model a
    /// request went to and what it carried.
    #[derive(Clone)]
    struct ScriptedLlm {
        name: &'static str,
        seen: async_channel::Sender<(&'static str, LLMRequest)>,
        turns: Arc<Mutex<std::collections::VecDeque<Turn>>>,
    }

    /// A scripted answer, which may depend on the request it answers.
    type Turn = Box<dyn FnOnce(&LLMRequest) -> Vec<Event> + Send>;

    impl ScriptedLlm {
        fn new(
            name: &'static str,
            seen: &async_channel::Sender<(&'static str, LLMRequest)>,
        ) -> Self {
            Self {
                name,
                seen: seen.clone(),
                turns: Arc::default(),
            }
        }

        /// Queues the turns the model answers its next requests with.
        fn then(&self, turns: impl IntoIterator<Item = Vec<Event>>) {
            self.turns.lock().unwrap().extend(
                turns
                    .into_iter()
                    .map(|events| Box::new(move |_: &LLMRequest| events) as Turn),
            );
        }

        /// Queues a turn computed from the request it answers.
        fn then_with(&self, turn: impl FnOnce(&LLMRequest) -> Vec<Event> + Send + 'static) {
            self.turns.lock().unwrap().push_back(Box::new(turn));
        }
    }

    impl LanguageModel for ScriptedLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let compaction = request.parameters().structured_outputs;
            let events = if compaction {
                vec![Event::Text(FORK_HANDOFF_JSON.to_string())]
            } else {
                let turn = self.turns.lock().unwrap().pop_front();
                turn.map_or_else(
                    || vec![Event::Text("done".to_string())],
                    |turn| turn(&request),
                )
            };
            self.seen
                .try_send((self.name, request))
                .expect("record model request");
            futures_lite::stream::iter(events.into_iter().map(Ok))
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                self.name,
                "test",
                self.name,
                "scripted model",
                1_000_000,
            )))
        }
    }

    /// Records the permission requests that reach it and denies them.
    #[derive(Clone, Default)]
    struct RecordingPermission {
        asked: Arc<Mutex<Vec<(aither_sandbox::TerminalMode, String)>>>,
    }

    impl aither_sandbox::permission::PermissionHandler for RecordingPermission {
        fn check(
            &self,
            mode: aither_sandbox::TerminalMode,
            script: &str,
        ) -> impl std::future::Future<
            Output = Result<bool, aither_sandbox::permission::PermissionError>,
        > + Send {
            self.asked.lock().unwrap().push((mode, script.to_string()));
            std::future::ready(Ok(false))
        }
    }

    /// A main agent's terminal whose permission requests are recorded.
    struct ForkFixture {
        _hook_tools: crate::test_support::PreparedFixture,
        _dirs: [tempfile::TempDir; 3],
        terminal: aither_sandbox::TerminalTool<
            crate::ReviewedPermission<RecordingPermission>,
            executor_core::tokio::TokioGlobal,
            aither_sandbox::Configured,
        >,
        asked: Arc<Mutex<Vec<(aither_sandbox::TerminalMode, String)>>>,
    }

    impl ForkFixture {
        async fn new() -> Self {
            Self::reviewed(Vec::new()).await
        }

        /// A fixture whose permission requests go through `chain` first.
        async fn reviewed(reviewers: Vec<crate::external_hook::ReviewerConfig>) -> Self {
            use aither_sandbox::{TerminalTool, ToolRegistryBuilder};
            use executor_core::tokio::TokioGlobal;

            let session = tempfile::tempdir().unwrap();
            let source = tempfile::tempdir().unwrap();
            let cache = tempfile::tempdir().unwrap();
            let binary = crate::test_support::compile_baseline_fixture(source.path()).await;
            let distribution = crate::test_support::fixture_distribution(&binary, "1.0.0").await;
            let permission = RecordingPermission::default();
            let asked = permission.asked.clone();
            let hook_tools = crate::test_support::prepared_fixture().await;
            let permission = crate::ReviewedPermission::new(
                permission,
                crate::external_hook::ExternalHooks::new(
                    &crate::external_hook::LiveHooks::start(
                        crate::external_hook::HooksConfig {
                            reviewers,
                            ..crate::external_hook::HooksConfig::default()
                        },
                        &hook_tools.tools,
                        executor_core::AnyExecutor::new(TokioGlobal),
                    )
                    .await
                    .expect("the hooks start"),
                ),
            );
            let terminal = TerminalTool::new_exact(session.path(), permission, TokioGlobal)
                .await
                .expect("terminal tool should initialize")
                .initialize_tools(&distribution, cache.path())
                .await
                .expect("tools should prepare");
            let registry = Arc::new(ToolRegistryBuilder::new().build(terminal.outputs_dir()));
            Self {
                _dirs: [session, source, cache],
                _hook_tools: hook_tools,
                terminal: terminal.with_registry(registry),
                asked,
            }
        }

        /// The subagent tool of the fixture's terminal: the main model is the
        /// advanced tier, the other tiers answer under their own names.
        fn tool(
            &self,
            main: &ScriptedLlm,
            seen: &async_channel::Sender<(&'static str, LLMRequest)>,
        ) -> SubagentTool<ScriptedLlm> {
            SubagentTool::new(main.clone())
                .with_balanced_model(ScriptedLlm::new("balanced", seen))
                .with_fast_model(ScriptedLlm::new("fast", seen))
                .with_builtins()
                .with_type(
                    "worker",
                    SubagentType::new("worker", |llm| {
                        crate::AgentBuilder::new(llm)
                            .agent_kind(crate::AgentKind::Chatbot)
                            .max_iterations(5)
                    }),
                )
                .for_terminal(self.terminal.binding())
        }

        /// A main agent on `llm` with this terminal, publishing to `source`.
        fn main_agent(
            &self,
            llm: ScriptedLlm,
            source: ForkSource,
            policy: crate::ContextPolicy,
        ) -> crate::Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, ()> {
            let jobs = self.terminal.job_registry();
            crate::AgentBuilder::new(llm)
                .config(
                    crate::AgentConfig::new()
                        .with_system_prompt("You are the main agent.")
                        .with_context_policy(policy),
                )
                .agent_kind(crate::AgentKind::Chatbot)
                .dyn_terminal(self.terminal.clone().to_dyn())
                .tool(aither_sandbox::builtin::KillTerminalTool::new(jobs.clone()))
                .tool(aither_sandbox::builtin::InputTerminalTool::new(
                    jobs.clone(),
                ))
                .tool(aither_sandbox::builtin::ReadTerminalDeltaTool::new(jobs))
                .fork_source(source)
                .build()
        }
    }

    fn start_fork(tag: &str, subagent: &str, model: Option<ModelTier>) -> (PathBuf, SubagentArgs) {
        let (dir, prompt_file) = write_prompt_file(tag, "carry on from here");
        (
            dir,
            SubagentArgs::Start {
                worktrees: Vec::new(),
                commit_worktrees: Vec::new(),
                subagent: subagent.to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model,
            },
        )
    }

    fn unsafe_terminal_call(id: &str) -> Event {
        Event::ToolCall(aither_core::llm::ToolCall::new(
            id,
            "terminal",
            serde_json::json!({
                "description": "probe the permission path",
                "script": "echo forked",
                "mode": "unsafe",
            }),
        ))
    }

    fn drain(
        seen: &async_channel::Receiver<(&'static str, LLMRequest)>,
    ) -> Vec<(&'static str, LLMRequest)> {
        std::iter::from_fn(|| seen.try_recv().ok()).collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_forks_first_request_is_the_main_agents_request_then_the_fork_prompt() {
        let fixture = ForkFixture::new().await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        main.then([
            vec![
                Event::ReasoningState(aither_core::llm::ReasoningState::new("test", "opaque")),
                unsafe_terminal_call("call-1"),
            ],
            vec![Event::Text("main answer".to_string())],
            vec![Event::Text("fork answer".to_string())],
        ]);
        let tool = fixture.tool(&main, &seen_tx);
        let mut agent = fixture.main_agent(
            main.clone(),
            tool.fork_source(),
            crate::ContextPolicy::default(),
        );
        agent.query("look into it").await.expect("main run");
        let main_requests = drain(&seen_rx);
        let (_, last_main) = main_requests.last().expect("the main agent sent requests");
        assert_eq!(main_requests.len(), 2, "one tool round, then the answer");

        let (dir, args) = start_fork("fork-first-request", "fork", None);
        let result = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect("fork call");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'fork' completed"), "{text}");
        assert!(text.contains("fork answer"), "{text}");

        let forked = drain(&seen_rx);
        let [(model, first)] = forked.as_slice() else {
            panic!("the fork sent one request, got {}", forked.len());
        };
        assert_eq!(*model, "main", "a fork runs on the main agent's model");

        let shared = last_main.messages();
        assert!(
            shared.iter().any(|message| !message.reasoning().is_empty()),
            "the inherited history carries the main agent's thinking"
        );
        assert_eq!(
            &first.messages()[..shared.len()],
            shared,
            "the fork's request starts with the main agent's, unchanged"
        );
        assert_eq!(
            &first.messages()[shared.len()..],
            [aither_core::llm::Message::user("carry on from here")],
            "and continues with the fork prompt alone"
        );
        assert_eq!(first.tool_definitions(), last_main.tool_definitions());
        assert_eq!(
            first.parameters().cache.openai,
            last_main.parameters().cache.openai,
            "the fork reads the main agent's cache"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_forks_permission_request_reaches_the_main_agent_like_a_subagents() {
        let fixture = ForkFixture::new().await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        main.then([vec![Event::Text("main answer".to_string())]]);
        let tool = fixture.tool(&main, &seen_tx);
        let mut agent = fixture.main_agent(
            main.clone(),
            tool.fork_source(),
            crate::ContextPolicy::default(),
        );
        agent.query("look into it").await.expect("main run");
        assert!(fixture.asked.lock().unwrap().is_empty());

        let probe = || {
            [
                vec![unsafe_terminal_call("call-1")],
                vec![Event::Text("finished".to_string())],
            ]
        };
        main.then(probe());
        let (fork_dir, fork) = start_fork("fork-permission", "fork", None);
        Tool::call(&tool, fork, ToolContext::new())
            .await
            .expect("fork call");
        let asked_by_fork = std::mem::take(&mut *fixture.asked.lock().unwrap());

        main.then(probe());
        let (worker_dir, worker) =
            start_fork("worker-permission", "worker", Some(ModelTier::Advanced));
        Tool::call(&tool, worker, ToolContext::new())
            .await
            .expect("worker call");
        let asked_by_worker = std::mem::take(&mut *fixture.asked.lock().unwrap());

        assert_eq!(
            asked_by_fork,
            [(
                aither_sandbox::TerminalMode::Unsafe,
                "echo forked".to_string()
            )],
            "the main agent's permission handler was asked"
        );
        assert_eq!(asked_by_fork, asked_by_worker);
        drain(&seen_rx);

        let _ = std::fs::remove_dir_all(fork_dir);
        let _ = std::fs::remove_dir_all(worker_dir);
    }

    /// Runs a subagent whose model asks for an out-of-sandbox command, behind
    /// a chain of one reviewer printing `answer`. Returns what the existing
    /// route was asked, what the reviewer read, and the text the subagent's
    /// model got back for its command.
    #[cfg(unix)]
    async fn subagent_request_behind_reviewer(answer: &str) -> (usize, String, String) {
        use crate::external_hook::tests::{logged, reviewers::reviewer};

        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let fixture =
            ForkFixture::reviewed(vec![reviewer(dir.path(), "guard", &log, answer)]).await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        let tool = fixture.tool(&main, &seen_tx);
        main.then([
            vec![unsafe_terminal_call("call-1")],
            vec![Event::Text("finished".to_string())],
        ]);
        let (worker_dir, worker) =
            start_fork("worker-reviewed", "worker", Some(ModelTier::Advanced));
        Tool::call(&tool, worker, ToolContext::new())
            .await
            .expect("worker call");
        let requests = drain(&seen_rx);
        let (_, after_command) = requests.last().expect("the worker sent requests");
        let seen_by_model = after_command
            .messages()
            .iter()
            .map(|message| message.content().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let reviewed = logged(&log);
        assert_eq!(reviewed.len(), 1, "the reviewer saw the request once");
        assert_eq!(reviewed[0]["role"], "subagent");
        assert_eq!(reviewed[0]["script"], "echo forked");
        let asked = fixture.asked.lock().unwrap().len();
        let _ = std::fs::remove_dir_all(worker_dir);
        (asked, reviewed[0].to_string(), seen_by_model)
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagents_request_every_reviewer_passes_reaches_the_main_agent() {
        let (asked, _, _) = subagent_request_behind_reviewer(r#"{"decision":"pass"}"#).await;
        assert_eq!(asked, 1, "the main agent's permission handler was asked");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_reviewer_approving_a_subagents_request_grants_it_without_asking() {
        let (asked, _, seen) = subagent_request_behind_reviewer(r#"{"decision":"approve"}"#).await;
        assert_eq!(asked, 0, "the main agent was not asked");
        assert!(seen.contains("forked"), "the command ran: {seen}");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_reviewer_denying_a_subagents_request_tells_its_model_why() {
        let (asked, _, seen) =
            subagent_request_behind_reviewer(r#"{"decision":"deny","reason":"no escalation"}"#)
                .await;
        assert_eq!(asked, 0, "the main agent was not asked");
        assert!(
            seen.contains("The permission reviewer guard denied this request: no escalation"),
            "{seen}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_fork_near_the_compaction_point_is_refused_until_the_main_agent_compacts() {
        let fixture = ForkFixture::new().await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let big = "word ".repeat(40_000);

        // Measure the usage a main run reaches, then aim the compaction
        // point so that usage lands within 10% below it.
        let probe = ScriptedLlm::new("main", &seen_tx);
        let probe_tool = fixture.tool(&probe, &seen_tx);
        fixture
            .main_agent(
                probe.clone(),
                probe_tool.fork_source(),
                crate::ContextPolicy::default(),
            )
            .query(&big)
            .await
            .expect("measuring run");
        let usage = probe_tool
            .fork_source()
            .latest()
            .expect("the measuring run published its request")
            .usage();
        let compact_at = usage / 0.95;
        let policy = crate::ContextPolicy {
            compact_at,
            restore_budget: compact_at / 4.0,
            ..crate::ContextPolicy::default()
        };
        drain(&seen_rx);

        let main = ScriptedLlm::new("main", &seen_tx);
        main.then([
            vec![Event::Text("main answer".to_string())],
            vec![Event::Text("main answer again".to_string())],
            vec![Event::Text("fork answer".to_string())],
        ]);
        let tool = fixture.tool(&main, &seen_tx);
        let mut agent = fixture.main_agent(main.clone(), tool.fork_source(), policy);
        agent.query(&big).await.expect("main run");
        drain(&seen_rx);

        let spaces = || {
            std::fs::read_dir(fixture.terminal.working_dir())
                .unwrap()
                .count()
        };
        let before = spaces();
        let (dir, args) = start_fork("fork-refused", "fork", None);
        let error = Tool::call(&tool, args.clone(), ToolContext::new())
            .await
            .expect_err("a fork this close to compaction is refused");
        let text = error.to_string();
        assert!(text.contains("Run `compact` first, then fork"), "{text}");
        assert!(drain(&seen_rx).is_empty(), "nothing was sent to a model");
        assert_eq!(spaces(), before, "no subagent terminal was launched");
        assert!(tool.runs().list().is_empty());

        agent.compact(None).await.expect("compaction");
        agent
            .query("continue")
            .await
            .expect("main run after compaction");
        drain(&seen_rx);
        let result = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect("a fork after the compaction starts");
        assert!(
            result
                .render_for_model()
                .starts_with("Subagent 'fork' completed")
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_fork_is_refused_where_there_is_no_main_request_to_continue() {
        let (seen_tx, _seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        let tool = SubagentTool::new(main).with_builtins();
        let (dir, args) = start_fork("fork-unbound", "fork", None);
        let error = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect_err("a tool bound to no terminal cannot fork");
        assert!(error.to_string().contains("needs the terminal"), "{error}");

        let (dir_model, with_model) = start_fork("fork-model", "fork", Some(ModelTier::Fast));
        let error = Tool::call(&tool, with_model, ToolContext::new())
            .await
            .expect_err("a fork cannot choose a model");
        assert!(error.to_string().contains("main agent's model"), "{error}");

        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(dir_model);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_fork_on_another_model_than_the_main_agents_is_refused() {
        let fixture = ForkFixture::new().await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        main.then([vec![Event::Text("main answer".to_string())]]);
        // The tool resolves the main agent's tier to another model.
        let other = ScriptedLlm::new("other", &seen_tx);
        let tool = fixture.tool(&other, &seen_tx);
        let mut agent = fixture.main_agent(
            main.clone(),
            tool.fork_source(),
            crate::ContextPolicy::default(),
        );
        agent.query("look into it").await.expect("main run");
        drain(&seen_rx);

        let spaces = || {
            std::fs::read_dir(fixture.terminal.working_dir())
                .unwrap()
                .count()
        };
        let before = spaces();
        let (dir, args) = start_fork("fork-other-model", "fork", None);
        let error = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect_err("a fork on another model is refused");
        let text = error.to_string();
        assert!(
            text.contains("'main'") && text.contains("'other'"),
            "{text}"
        );
        assert!(drain(&seen_rx).is_empty(), "nothing was sent to a model");
        assert_eq!(spaces(), before, "no subagent terminal was launched");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    /// The task id the background-started reminder in `request` reports.
    fn reported_task_id(request: &LLMRequest) -> String {
        request
            .messages()
            .iter()
            .rev()
            .find_map(|message| {
                let (_, rest) = message.content().split_once("task_id=")?;
                Some(rest.split(')').next()?.to_string())
            })
            .unwrap_or_else(|| {
                let contents: Vec<&str> = request
                    .messages()
                    .iter()
                    .map(aither_core::llm::Message::content)
                    .collect();
                panic!("no terminal result reported a task id; messages: {contents:#?}")
            })
    }

    #[cfg(unix)]
    fn control_call(id: &str, name: &str, task_id: &str) -> Event {
        Event::ToolCall(aither_core::llm::ToolCall::new(
            id,
            name,
            serde_json::json!({ "task_id": task_id }),
        ))
    }

    // Windows runs terminal scripts through `cmd.exe`, which has no `sleep`
    // to keep the fork's job alive (#217).
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_forks_terminal_controls_reach_its_own_jobs_and_not_the_main_agents() {
        let fixture = ForkFixture::new().await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        main.then([vec![Event::Text("main answer".to_string())]]);
        let tool = fixture.tool(&main, &seen_tx);
        let mut agent = fixture.main_agent(
            main.clone(),
            tool.fork_source(),
            crate::ContextPolicy::default(),
        );
        agent.query("look into it").await.expect("main run");

        let main_jobs = fixture.terminal.job_registry();
        main_jobs
            .register(
                4242,
                "main-job",
                "exec-key",
                "long running main job",
                aither_sandbox::TerminalMode::Sandboxed,
                None,
            )
            .await;

        main.then([vec![Event::ToolCall(aither_core::llm::ToolCall::new(
            "bg-1",
            "terminal_background",
            serde_json::json!({
                "description": "start a job",
                "script": "sleep 30",
                "timeout": 60,
            }),
        ))]]);
        main.then_with(|request| {
            vec![control_call(
                "read-1",
                "terminal_read",
                &reported_task_id(request),
            )]
        });
        main.then_with(|request| {
            vec![control_call(
                "kill-1",
                "terminal_kill",
                &reported_task_id(request),
            )]
        });
        main.then([vec![control_call("kill-2", "terminal_kill", "main-job")]]);
        main.then([vec![Event::Text("finished".to_string())]]);

        let (dir, args) = start_fork("fork-controls", "fork", None);
        Tool::call(&tool, args, ToolContext::new())
            .await
            .expect("fork call");

        let requests = drain(&seen_rx);
        let (_, last) = requests.last().expect("the fork sent requests");
        let results: Vec<&str> = last
            .messages()
            .iter()
            .filter_map(|message| match message {
                aither_core::llm::Message::Tool { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            results.iter().any(|text| text.contains("next cursor")),
            "terminal_read read the fork's job: {results:?}"
        );
        assert!(
            results.iter().any(|text| text.contains("terminated")),
            "terminal_kill stopped the fork's job: {results:?}"
        );
        assert!(
            results
                .iter()
                .any(|text| text.contains("No running background task 'main-job'")),
            "the main agent's job is out of reach: {results:?}"
        );
        assert!(
            main_jobs
                .list()
                .await
                .iter()
                .any(|job| job.task_id == "main-job"
                    && job.status == aither_sandbox::JobStatus::Running),
            "the main agent's job still runs"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    // The hook fixtures are `sh` scripts (`external_hook::tests`).
    #[cfg(unix)]
    #[tokio::test]
    async fn a_subagents_actions_reach_hooks_with_the_subagent_role() {
        use crate::external_hook::LifecycleKind;
        use crate::external_hook::tests::{logged, recorder};

        let hooks_dir = unique_temp_dir("subagent-hooks");
        let tools = crate::test_support::prepared_fixture().await;
        let log = hooks_dir.join("log");
        let main = crate::external_hook::tests::lifecycle_hooks(
            vec![recorder(
                &hooks_dir,
                &log,
                &[
                    LifecycleKind::SessionStart,
                    LifecycleKind::UserMessage,
                    LifecycleKind::Stop,
                    LifecycleKind::SubagentStart,
                    LifecycleKind::SubagentStop,
                ],
            )],
            &tools,
        )
        .await;
        let tool =
            worker_tool(SubagentTool::new(TextLlm::new("only-model"))).with_hooks(main.clone());
        let (dir, prompt_file) = write_prompt_file("subagent-hooks", "do it");

        tokio::time::timeout(
            std::time::Duration::from_secs(60),
            Tool::call(
                &tool,
                SubagentArgs::Start {
                    worktrees: Vec::new(),
                    commit_worktrees: Vec::new(),
                    subagent: "worker".to_string(),
                    prompt_file: prompt_file.to_string_lossy().into_owned(),
                    model: Some(ModelTier::Fast),
                },
                ToolContext::new(),
            ),
        )
        .await
        .expect("the subagent finishes within its bound")
        .expect("the subagent completes");

        let seen: Vec<(String, String)> = logged(&log)
            .iter()
            .map(|input| {
                (
                    input["event"].as_str().unwrap().to_string(),
                    input["role"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let expected = [
            ("subagent_start", "main"),
            ("session_start", "subagent"),
            ("user_message", "subagent"),
            ("stop", "subagent"),
            ("subagent_stop", "main"),
        ]
        .map(|(event, role)| (event.to_string(), role.to_string()));
        assert_eq!(seen, expected);
        let inputs = logged(&log);
        assert_eq!(inputs[0]["subagent"], "worker");
        assert_eq!(inputs[0]["prompt"], "do it");
        assert_eq!(inputs[4]["end"], "completed");
        assert_eq!(inputs[4]["result"], "only-model");

        // Only what happened to the main agent reaches its inbox; the
        // subagent's context stayed in the subagent's own.
        let injected = crate::Hook::take_injected_context(&main);
        assert_eq!(injected.len(), 2);

        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(hooks_dir);
    }
}
