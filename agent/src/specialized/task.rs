//! The `subagent` command: runs a named subagent, forks the calling agent or
//! continues a finished subagent, each in a context window of its own.
//!
//! A named subagent is a TOML definition (see [`SubagentDefinition`]) looked
//! up first among the definitions the embedder supplies, then in
//! `agents/` under the calling terminal's working directory.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aither_core::llm::{AgentIdentity, CallIdentity, SubagentIdentity, ToolContext};
use aither_core::{
    AnyIdAllocator, IdLease, LanguageModel,
    llm::{Tool, ToolResult, model::Parameters},
};
use aither_sandbox::{
    CommandRole, CommandUsage, CommitLedger, CommitLedgerError, CommitMark, DynTerminalTool,
    SubagentCommit, SubagentGrant, TerminalBinding,
};
use askama::Template;
use executor_core::Executor;
use schemars::JsonSchema;
use serde::Deserialize;

use super::external::{AnyExternalSession, AnyExternalSubagent, ExternalStart, ExternalSubagent};
use crate::context::{CacheLifetime, HandoffDocument, HandoffSchema};
use crate::external_hook::{self, ExternalHooks};
use crate::fork::{ForkSnapshot, ForkSource};
use crate::fs_util::path_exists;
use crate::model_group::ModelTier;
use crate::request_parameters::TierParameters;
use crate::subagent_file::{
    Listed, SubagentAccess, SubagentDefinition, SubagentLookup, validate_subagent_name,
};
use crate::{AgentBuilder, AgentOptions};

async fn checked_path_exists(path: &Path) -> anyhow::Result<bool> {
    path_exists(path)
        .await
        .map_err(|error| anyhow::anyhow!("failed to inspect path '{}': {error}", path.display()))
}

/// The hooks every subagent's agent carries: its own scope of the external
/// hooks, which holds none when the host configured none.
pub type SubagentHooks = crate::HCons<ExternalHooks, ()>;

/// Adds to the agent a named subagent's definition makes, before its
/// terminal, cache keep-alive and hooks are attached.
///
/// See [`SubagentTool::with_agent_setup`]. The agent is single-model: all
/// tiers use the model the subagent runs on.
pub type SubagentSetup<LLM> =
    Arc<dyn Fn(AgentBuilder<LLM, LLM, LLM, ()>) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync>;

/// What a `fork` subagent is for, as the main agent reads it.
pub const FORK_DESCRIPTION: &str = "Continues your own context on your model; use it for work that depends on what you have read and decided";

/// A connected agent that runs outside aither, started with
/// `subagent external --agent <name>`.
#[derive(Clone)]
struct External {
    description: String,
    backend: Arc<AnyExternalSubagent>,
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

/// Run a subagent: an autonomous agent with a context window of its own that
/// returns only its final result.
///
/// `subagent <name> --prompt-file <path>` runs the subagent `<name>` names;
/// run `subagent` alone to list the names that resolve. Its definition fixes
/// its role prompt, its model and what it may write: every subagent reads the
/// whole workspace and writes its own working directory, and one whose
/// definition allows it also writes the worktrees you grant. Three names are
/// reserved: `fork` continues your own context, `resume` continues a
/// finished subagent, and `external` runs a connected agent.
///
/// The brief is always read from a file: write it to a file first, then pass
/// its path. Inline prompt text is not a file path and is rejected.
#[derive(Debug, Clone, JsonSchema, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubagentArgs {
    /// The subagent to run, or `fork`, `resume` or `external`.
    pub name: String,
    /// Path to a file holding the brief, or for `resume` the follow-up. The
    /// file must exist and be non-empty.
    pub prompt_file: String,
    /// `resume` only: the id from a finished subagent's result, a single
    /// word. Ids expire with the provider's prompt cache.
    #[serde(default)]
    pub id: Option<String>,
    /// `external` only: the connected agent to run.
    #[serde(default)]
    pub agent: Option<String>,
    /// Directories, usually linked git worktrees you created, the subagent
    /// may write besides its own space, when its definition lets it write
    /// worktrees. Each must be inside what you can write. It cannot commit
    /// in these.
    #[serde(default)]
    pub worktrees: Vec<String>,
    /// Linked git worktrees the subagent may write and commit in, when its
    /// definition lets it write worktrees. Each must be on a branch
    /// `subagent/<id>/<name>` whose namespace `subagent/<id>/` holds no other
    /// branch. Every commit it makes is listed in its result.
    #[serde(default)]
    pub commit_worktrees: Vec<String>,
}

/// What a call asks for, checked against the reserved names.
enum Call<'a> {
    /// Runs the subagent a name resolves to.
    Named { name: &'a str, grant: Requested<'a> },
    /// Continues the calling agent's context.
    Fork { grant: Requested<'a> },
    /// Runs a connected agent.
    External {
        agent: &'a str,
        grant: Requested<'a>,
    },
    /// Continues a finished subagent.
    Resume { id: &'a str },
}

/// The directories a call asks to grant, as written.
#[derive(Clone, Copy)]
struct Requested<'a> {
    worktrees: &'a [String],
    commit_worktrees: &'a [String],
}

impl Requested<'_> {
    const fn is_empty(&self) -> bool {
        self.worktrees.is_empty() && self.commit_worktrees.is_empty()
    }

    /// The writable and the committing directories, with relative paths
    /// resolved against the caller's working directory.
    fn paths(&self, working_dir: Option<&Path>) -> (Vec<PathBuf>, Vec<PathBuf>) {
        let resolve = |path: &String| {
            let path = Path::new(path);
            match working_dir {
                Some(base) if path.is_relative() => base.join(path),
                _ => path.to_path_buf(),
            }
        };
        (
            self.worktrees.iter().map(resolve).collect(),
            self.commit_worktrees.iter().map(resolve).collect(),
        )
    }

    /// The grant beyond the subagent's own space.
    fn grant(&self, working_dir: Option<&Path>) -> SubagentGrant {
        let (worktrees, commit_worktrees) = self.paths(working_dir);
        let grant = worktrees
            .into_iter()
            .fold(SubagentGrant::new(), SubagentGrant::with_worktree);
        commit_worktrees
            .into_iter()
            .fold(grant, SubagentGrant::with_committing_worktree)
    }
}

impl SubagentArgs {
    /// The subagent a call runs, by the name its result reports: the named
    /// subagent's, `fork` or the external agent's; `None` for `resume`.
    #[must_use]
    pub fn subagent(&self) -> Option<&str> {
        match self.name.as_str() {
            "resume" => None,
            "external" => self.agent.as_deref(),
            name => Some(name),
        }
    }

    /// The prompt file every call carries.
    #[must_use]
    pub fn prompt_file(&self) -> &str {
        &self.prompt_file
    }

    /// What the call asks for, refusing an option its name does not take.
    fn call(&self) -> anyhow::Result<Call<'_>> {
        let grant = Requested {
            worktrees: &self.worktrees,
            commit_worktrees: &self.commit_worktrees,
        };
        let name = self.name.as_str();
        if name != "resume" && self.id.is_some() {
            anyhow::bail!("`--id` continues a finished subagent: use it with `subagent resume`");
        }
        if name != "external" && self.agent.is_some() {
            anyhow::bail!("`--agent` names a connected agent: use it with `subagent external`");
        }
        Ok(match name {
            "resume" => {
                let Some(id) = self.id.as_deref() else {
                    anyhow::bail!(
                        "`subagent resume` takes the `--id` of the finished subagent to continue"
                    );
                };
                if !grant.is_empty() {
                    anyhow::bail!(
                        "a continued subagent keeps what it was granted: `subagent resume` takes no `--worktrees` or `--commit-worktrees`"
                    );
                }
                Call::Resume { id }
            }
            "fork" => Call::Fork { grant },
            "external" => {
                let Some(agent) = self.agent.as_deref() else {
                    anyhow::bail!("`subagent external` takes the `--agent` to run");
                };
                Call::External { agent, grant }
            }
            name => Call::Named { name, grant },
        })
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
    /// The working copies its space was prepared with, where its work stays.
    working_copies: &'a [PathBuf],
}

/// A fresh subagent's first prompt when its space was prepared with working
/// copies: where to work, then the brief.
#[derive(Template)]
#[template(path = "subagent_brief.txt", escape = "none")]
struct SubagentBriefTemplate<'a> {
    working_copies: &'a [PathBuf],
    brief: &'a str,
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

/// Lists the commits a run made on its granted repositories; `None` when the
/// run held no commit grant.
async fn list_commits(
    ledger: Option<&CommitLedger>,
    mark: Option<&CommitMark>,
) -> Option<CommitListing> {
    match (ledger, mark) {
        (Some(ledger), Some(mark)) => Some(CommitListing {
            listed: ledger.since(mark).await,
            setbacks: mark.setbacks().iter().map(ToString::to_string).collect(),
        }),
        _ => None,
    }
}

/// What the result text reports about a subagent run.
enum SubagentReport<'a> {
    /// The user cancelled the run.
    Cancelled,
    /// The run produced a result.
    Completed {
        /// The tier the subagent ran on; `None` for an external agent.
        tier: Option<&'static str>,
        model: Option<&'a str>,
        result: &'a str,
        /// Id to continue this subagent with.
        id: &'a str,
        /// Whether the id expires with the subagent's prompt cache; an
        /// external session's never does.
        cache_bound: bool,
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
    /// The working copies its space was prepared with, where its work stays.
    working_copies: &'a [PathBuf],
    history: &'a [HistoryEntry],
}

#[derive(Template)]
#[template(path = "fork_refused.txt", escape = "none")]
struct ForkRefusedTemplate<'a> {
    reason: &'a ForkRefusal,
}

/// Why a fork was not started, rendered by
/// [`Notices::fork_refusal`](crate::Notices::fork_refusal).
#[derive(Debug, Clone, PartialEq)]
pub enum ForkRefusal {
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

/// aither's wording of a fork refusal.
pub(crate) fn default_fork_refusal(refusal: &ForkRefusal) -> String {
    crate::notices::rendered(&ForkRefusedTemplate { reason: refusal })
}

use super::router::{
    AnySubagentRouter, JudgmentRouter, RouteRequest, SubagentRouter, TierOffer, tier_label,
};

/// How a start call makes its subagent's agent.
enum Source<'a, LLM> {
    /// From a named subagent's definition, with the setup the host added for
    /// that name.
    Definition {
        definition: SubagentDefinition,
        setup: Option<&'a SubagentSetup<LLM>>,
    },
    /// From the main agent's request and model.
    Fork,
}

/// The subagent a start call runs: its name, the tier it declares (`None`
/// leaves the tier to routing) and how its agent is made.
struct Resolved<'a, LLM> {
    name: String,
    tier: Option<ModelTier>,
    source: Source<'a, LLM>,
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
struct FinishedSubagent<LLM, Schema> {
    subagent: String,
    /// The tier it runs on; `None` for an external agent.
    tier: Option<ModelTier>,
    model: Option<String>,
    /// Lists the commits the subagent makes; set when it was launched with a
    /// commit grant.
    commit_ledger: Option<CommitLedger>,
    /// The working copies the launching terminal's spaces prepared for it.
    working_copies: Vec<PathBuf>,
    body: SubagentBody<LLM, Schema>,
}

/// What a run is prompted with: a fresh subagent with working copies learns
/// where they are before its brief; a continued one was told when it started.
fn run_prompt<'a>(prompt: &'a str, resumed: bool, working_copies: &[PathBuf]) -> Cow<'a, str> {
    if resumed || working_copies.is_empty() {
        return Cow::Borrowed(prompt);
    }
    Cow::Owned(
        SubagentBriefTemplate {
            working_copies,
            brief: prompt,
        }
        .render()
        .expect("subagent brief template must render"),
    )
}

/// The working copies the launcher's spaces prepared for the subagent
/// `terminal` serves.
fn prepared_working_copies(terminal: &DynTerminalTool) -> Vec<PathBuf> {
    terminal
        .prepared_space()
        .map(|space| space.working_copies().to_vec())
        .unwrap_or_default()
}

/// What runs a subagent's turns; an aither agent compacts into `Schema`.
enum SubagentBody<LLM, Schema> {
    /// An aither agent.
    Agent {
        agent: Box<crate::Agent<LLM, LLM, LLM, SubagentHooks, LLM, Schema>>,
        child_job_registry: Option<aither_sandbox::JobRegistry>,
        /// How long the provider keeps this subagent's prompt cache warm — the
        /// [`PromptCache`](crate::context::PromptCache) lifetime the subagent
        /// ran with. The continuation id expires with the cache entry.
        lifetime: CacheLifetime,
    },
    /// A session that runs outside aither; it lives until it is closed.
    External {
        session: AnyExternalSession,
        /// The terminal that holds the subagent's grant and commit ledger.
        _terminal: Option<Box<DynTerminalTool>>,
    },
}

impl<LLM: LanguageModel + 'static, Schema: HandoffSchema> FinishedSubagent<LLM, Schema> {
    /// The cache lifetime this subagent outlived, when it did: its cache is
    /// cold (never started, or past its lifetime). An
    /// external session never expires.
    fn expired(&self) -> Option<CacheLifetime> {
        let SubagentBody::Agent {
            agent, lifetime, ..
        } = &self.body
        else {
            return None;
        };
        (!agent.cache_warmth().is_warm()).then_some(*lifetime)
    }
}

enum FinishedSlot<LLM, Schema> {
    /// Waiting to be continued, under the lease of its id.
    Idle(IdLease, Box<FinishedSubagent<LLM, Schema>>),
    /// Checked out by a continuation in flight, which holds the lease.
    Running,
    /// Its cache lifetime passed; the context was dropped and its id
    /// released. The lifetime stays so continuing the id names how long it
    /// lived, until the word names another subagent.
    Expired(CacheLifetime),
}

/// The id a call's subagent is continued by: leased for a fresh subagent
/// when its call starts, or the one `resume` checked out.
struct CallId {
    lease: IdLease,
    /// Whether `resume` checked it out.
    resumed: bool,
}

/// Finished subagents the parent may continue until their cache expires,
/// each under a word leased from `ids` for as long as it can be continued.
struct FinishedSubagents<LLM, Schema> {
    ids: AnyIdAllocator,
    slots: Mutex<HashMap<String, FinishedSlot<LLM, Schema>>>,
}

impl<LLM, Schema> FinishedSubagents<LLM, Schema> {
    fn new(ids: AnyIdAllocator) -> Self {
        Self {
            ids,
            slots: Mutex::new(HashMap::new()),
        }
    }

    /// Leases the id of a fresh subagent.
    fn lease(&self) -> anyhow::Result<CallId> {
        Ok(CallId {
            lease: self.ids.lease()?,
            resumed: false,
        })
    }
}

impl<LLM: LanguageModel + 'static, Schema: HandoffSchema> FinishedSubagents<LLM, Schema> {
    /// Drops the context of every subagent whose cache lifetime passed,
    /// releasing its id.
    fn expire(slots: &mut HashMap<String, FinishedSlot<LLM, Schema>>) {
        for slot in slots.values_mut() {
            if let FinishedSlot::Idle(_, finished) = slot
                && let Some(lifetime) = finished.expired()
            {
                *slot = FinishedSlot::Expired(lifetime);
            }
        }
    }

    /// Takes the subagent out for a continuation, with the lease of its id.
    fn checkout(&self, id: &str) -> anyhow::Result<(CallId, Box<FinishedSubagent<LLM, Schema>>)> {
        let mut slots = self.slots.lock().expect("finished subagents lock poisoned");
        Self::expire(&mut slots);
        match slots.get_mut(id) {
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
            Some(slot @ FinishedSlot::Idle(..)) => {
                let FinishedSlot::Idle(lease, finished) =
                    std::mem::replace(slot, FinishedSlot::Running)
                else {
                    unreachable!("slot matched as idle");
                };
                Ok((
                    CallId {
                        lease,
                        resumed: true,
                    },
                    finished,
                ))
            }
        }
    }

    /// Keeps a finished subagent under `id`, the lease of its call.
    fn store(&self, id: IdLease, finished: FinishedSubagent<LLM, Schema>) {
        let mut slots = self.slots.lock().expect("finished subagents lock poisoned");
        Self::expire(&mut slots);
        slots.insert(id.to_string(), FinishedSlot::Idle(id, Box::new(finished)));
    }

    /// Releases an id whose subagent can no longer be continued.
    fn release(&self, id: &str) {
        self.slots
            .lock()
            .expect("finished subagents lock poisoned")
            .remove(id);
    }

    /// Releases `id` only while a continuation holds it checked out: its
    /// subagent went with the call that dropped it.
    fn release_checked_out(&self, id: &str) {
        let mut slots = self.slots.lock().expect("finished subagents lock poisoned");
        if matches!(slots.get(id), Some(FinishedSlot::Running)) {
            slots.remove(id);
        }
    }
}

/// Stops the terminal jobs an aither agent left running.
async fn stop_jobs(job_registry: Option<aither_sandbox::JobRegistry>, subagent: &str) {
    if let Some(job_registry) = job_registry {
        let killed = job_registry.kill_running().await;
        if killed > 0 {
            tracing::info!(
                subagent = %subagent,
                killed,
                "Stopped background terminal jobs left by subagent"
            );
        }
    }
}

/// A subagent's run as it is listed and owned while it executes.
struct LiveRun<LLM, Schema> {
    id: u64,
    subagent: String,
    /// The hooks told when the run stops.
    hooks: Option<ExternalHooks>,
    body: SubagentBody<LLM, Schema>,
}

impl<LLM: LanguageModel + 'static, Schema: HandoffSchema> LiveRun<LLM, Schema> {
    /// Ends a run whose call was dropped the way a cancelled run ends: the
    /// hooks hear it was cancelled, its background jobs are stopped and its
    /// session is closed.
    async fn abandon(self) {
        let Self {
            subagent,
            hooks,
            body,
            ..
        } = self;
        if let Some(hooks) = &hooks {
            hooks
                .fire(&subagent_stop_input::<String>(
                    &subagent,
                    &SubagentOutcome::Cancelled,
                ))
                .await;
        }
        stop_jobs(body.job_registry(), &subagent).await;
        body.end(true).await;
    }
}

/// What one `subagent` call holds in the tool's shared state: its entry in
/// the run registry, the id its subagent is continued by, and the run it owns
/// while it executes.
///
/// Dropping the hold, which is what dropping the call's future does, releases
/// all of it, so no exit from the call leaves a run listed or an id checked
/// out; the id stays leased only once its subagent is stored.
struct CallHold<LLM: LanguageModel + 'static, Schema: HandoffSchema> {
    runs: SubagentRuns,
    finished: Arc<FinishedSubagents<LLM, Schema>>,
    /// The terminal the tool serves, whose executor finishes an abandoned
    /// run's cleanup. A tool bound to none launches no terminal jobs, so
    /// there is none to stop.
    binding: Option<TerminalBinding>,
    id: CallId,
    run: Option<LiveRun<LLM, Schema>>,
}

impl<LLM: LanguageModel + 'static, Schema: HandoffSchema> CallHold<LLM, Schema> {
    const fn new(
        runs: SubagentRuns,
        finished: Arc<FinishedSubagents<LLM, Schema>>,
        binding: Option<TerminalBinding>,
        id: CallId,
    ) -> Self {
        Self {
            runs,
            finished,
            binding,
            id,
            run: None,
        }
    }

    /// Lists the run and takes `body` for its duration; the receiver resolves
    /// once when it is cancelled.
    fn begin(
        &mut self,
        subagent: &str,
        prompt: &str,
        hooks: Option<ExternalHooks>,
        body: SubagentBody<LLM, Schema>,
    ) -> async_channel::Receiver<()> {
        let (id, cancel_rx) = self.runs.begin(subagent, prompt);
        self.run = Some(LiveRun {
            id,
            subagent: subagent.to_string(),
            hooks,
            body,
        });
        cancel_rx
    }

    /// Runs one turn on `prompt` until it ends or `cancel_rx` fires.
    async fn run(
        &mut self,
        prompt: &str,
        cancel_rx: async_channel::Receiver<()>,
    ) -> SubagentOutcome<String> {
        self.run
            .as_mut()
            .expect("a run is begun before it is run")
            .body
            .run(prompt, cancel_rx)
            .await
    }

    /// Unlists the run and gives its subagent back.
    fn finish(&mut self) -> SubagentBody<LLM, Schema> {
        let run = self.run.take().expect("a run is begun before it finishes");
        self.runs.finish(run.id);
        run.body
    }
}

impl<LLM: LanguageModel + 'static, Schema: HandoffSchema> Drop for CallHold<LLM, Schema> {
    fn drop(&mut self) {
        if self.id.resumed {
            self.finished.release_checked_out(&self.id.lease);
        }
        if let Some(run) = self.run.take() {
            self.runs.finish(run.id);
            // The rest of the cleanup is asynchronous and a drop cannot wait
            // for it, so the terminal's executor finishes it.
            if let Some(binding) = &self.binding {
                binding.executor().spawn(run.abandon()).detach();
            }
        }
    }
}

/// The `subagent` command, which runs named subagents, forks the main agent
/// and continues finished subagents.
///
/// Only the main agent has it: register it with
/// [`TerminalAgentBuilder::subagents`](crate::TerminalAgentBuilder::subagents),
/// or with [`ToolRegistryBuilder::main_exclusive`](aither_sandbox::ToolRegistryBuilder::main_exclusive)
/// and [`configure_terminal_bound_usage_tool`](aither_sandbox::ToolRegistryBuilder::configure_terminal_bound_usage_tool),
/// so a subagent's terminal has no `subagent` command.
///
/// # Example
///
/// ```rust,ignore
/// use aither_agent::{SubagentDefinition, specialized::SubagentTool};
///
/// let explore = SubagentDefinition::parse(include_str!("explore.toml"))?;
/// let subagents = SubagentTool::new(llm.clone()).with_definitions([explore]);
/// let agent = TerminalAgentBuilder::new(llm, terminal)
///     .subagents(subagents, "Run a subagent")
///     .build();
/// ```
///
/// Every aither subagent it builds, a fork included, compacts into `Schema`;
/// [`TerminalAgentBuilder::subagents`](crate::TerminalAgentBuilder::subagents)
/// sets the main agent's.
pub struct SubagentTool<LLM, Schema = HandoffDocument> {
    /// Model a subagent runs on when the advanced tier is selected.
    advanced: LLM,
    /// Model a subagent runs on when the balanced tier is selected.
    balanced: LLM,
    /// Model a subagent runs on when the fast tier is selected; also routes
    /// the briefs of subagents that declare no tier.
    fast: LLM,
    /// The parameters of the requests sent on each tier's model: a
    /// subagent's, and the router's on the fast model.
    parameters: TierParameters,
    /// The embedder's definitions, by name, looked in before the calling
    /// terminal's `agents/`.
    definitions: BTreeMap<String, SubagentDefinition>,
    /// What the host adds to the agent of a named subagent, by name.
    setups: HashMap<String, SubagentSetup<LLM>>,
    /// Connected external agents, by name.
    externals: BTreeMap<String, External>,
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
    finished: Arc<FinishedSubagents<LLM, Schema>>,
    /// The main agent's latest request, which a fork starts from.
    fork_source: ForkSource,
    /// The external hooks of the agent this tool serves: told when a
    /// subagent starts and stops, and the parent of each subagent's own scope.
    hooks: Option<ExternalHooks>,
    /// The main agent's working-document inbox, told when a subagent starts
    /// and finishes.
    work_events: Option<crate::working_document::WorkEvents>,
    /// The working document every subagent keeps in its own space.
    document: Option<crate::working_document::DocumentSpec>,
    /// The stop rule every subagent's run follows.
    stop_hook: Option<crate::AnyStopHook>,
    /// Holds subagent prompts to the working language.
    #[cfg(feature = "languages")]
    languages: Option<crate::LanguageGuard>,
    /// Renders the tool's refusals and every notice of the subagents it
    /// builds.
    notices: crate::AnyNotices,
    /// Chooses the tier of a subagent whose definition states none; `None`
    /// asks the fast model with [`JudgmentRouter`].
    router: Option<AnySubagentRouter>,
}

impl<LLM, Schema> std::fmt::Debug for SubagentTool<LLM, Schema> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let setups: Vec<_> = self.setups.keys().collect();
        let externals: Vec<_> = self.externals.keys().collect();
        f.debug_struct("SubagentTool")
            .field("definitions", &self.definitions.keys().collect::<Vec<_>>())
            .field("setups", &setups)
            .field("externals", &externals)
            .field("base_dir", &self.base_dir)
            .finish_non_exhaustive()
    }
}

impl<LLM: Clone + Sync> SubagentTool<LLM> {
    /// Creates a subagent tool that runs subagents on `llm`. It has only the
    /// built-in forms (`fork`, `resume` and `external`) until
    /// [`with_definitions`](Self::with_definitions) adds the embedder's
    /// subagents. All three model tiers use `llm`, with the parameters
    /// [`with_request_parameters`](Self::with_request_parameters) sets;
    /// configure distinct ones with
    /// [`with_balanced_model`](Self::with_balanced_model) and
    /// [`with_fast_model`](Self::with_fast_model).
    pub fn new(llm: LLM) -> Self {
        Self {
            advanced: llm.clone(),
            balanced: llm.clone(),
            fast: llm,
            parameters: TierParameters::default(),
            definitions: BTreeMap::new(),
            setups: HashMap::new(),
            externals: BTreeMap::new(),
            base_dir: None,
            mounts: Vec::new(),
            binding: None,
            runs: SubagentRuns::default(),
            cache_keep_alive: None,
            finished: Arc::new(FinishedSubagents::new(AnyIdAllocator::default())),
            fork_source: ForkSource::default(),
            hooks: None,
            work_events: None,
            document: None,
            stop_hook: None,
            #[cfg(feature = "languages")]
            languages: None,
            notices: crate::AnyNotices::default(),
            router: None,
        }
    }
}

impl<LLM: Clone + Sync, Schema> SubagentTool<LLM, Schema> {
    /// This tool with its aither subagents compacting into `S2`.
    /// [`TerminalAgentBuilder::subagents`](crate::TerminalAgentBuilder::subagents)
    /// sets the main agent's schema.
    ///
    /// Finished subagents are typed by the schema they compact into, so the
    /// tool starts with none to continue, as an instance from
    /// [`for_terminal`](Self::for_terminal) does.
    #[must_use]
    pub fn with_handoff<S2: HandoffSchema>(self) -> SubagentTool<LLM, S2> {
        SubagentTool {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: self.fast,
            parameters: self.parameters,
            definitions: self.definitions,
            setups: self.setups,
            externals: self.externals,
            base_dir: self.base_dir,
            mounts: self.mounts,
            binding: self.binding,
            runs: self.runs,
            cache_keep_alive: self.cache_keep_alive,
            finished: Arc::new(FinishedSubagents::new(self.finished.ids.clone())),
            fork_source: self.fork_source,
            hooks: self.hooks,
            work_events: self.work_events,
            document: self.document,
            stop_hook: self.stop_hook,
            #[cfg(feature = "languages")]
            languages: self.languages,
            notices: self.notices,
            router: self.router,
        }
    }

    /// Adds the embedder's subagent definitions, which `subagent <name>` runs
    /// and `subagent` lists. A definition shadows a local `agents/<name>.toml`
    /// of the same name and replaces an earlier one of the same name.
    ///
    /// # Panics
    ///
    /// Panics when a definition's name cannot name a subagent, since the
    /// embedder wrote it.
    #[must_use]
    pub fn with_definitions(
        mut self,
        definitions: impl IntoIterator<Item = SubagentDefinition>,
    ) -> Self {
        for definition in definitions {
            if let Err(error) = validate_subagent_name(&definition.name) {
                panic!("subagent definition '{}': {error}", definition.name);
            }
            self.definitions.insert(definition.name.clone(), definition);
        }
        self
    }

    /// Leases the ids finished subagents are continued by from `ids`. A tool
    /// bound to a terminal with [`for_terminal`](Self::for_terminal) leases
    /// from that terminal's allocator instead, so its ids never share a word
    /// with the terminal's; one given none leases from its own
    /// [`AnyIdAllocator::default`].
    ///
    /// The tool starts with no finished subagents to continue.
    #[must_use]
    pub fn with_id_allocator(mut self, ids: AnyIdAllocator) -> Self {
        self.finished = Arc::new(FinishedSubagents::new(ids));
        self
    }

    /// Routes subagents whose definition states no tier through `router`
    /// instead of the default [`JudgmentRouter`] on the fast model. A
    /// definition that states a tier never consults it.
    #[must_use]
    pub fn with_router(mut self, router: impl SubagentRouter) -> Self {
        self.router = Some(AnySubagentRouter::new(router));
        self
    }

    /// Renders the tool's refusals, and every alert, notice and refusal of
    /// the subagents it builds, through `notices`.
    /// [`TerminalAgentBuilder::subagents`](crate::TerminalAgentBuilder::subagents)
    /// sets the main agent's.
    #[must_use]
    pub fn with_notices(mut self, notices: crate::AnyNotices) -> Self {
        self.notices = notices;
        self
    }

    /// Sets the directory relative prompt paths resolve against, and whose
    /// `agents/` holds the local subagents, when the tool is bound to no
    /// terminal.
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

    /// Sets the parameters of the requests sent on the model the tool was
    /// created with: those of every subagent on a tier that runs on it, and
    /// the router's while the fast tier does.
    ///
    /// Each subagent's agent sets the fields aither owns per request over
    /// them, as [`AgentOptions::request_parameters`] describes. A fork
    /// sends the main agent's parameters instead, since its requests start
    /// as the main agent's.
    #[must_use]
    pub fn with_request_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters.set_shared(parameters);
        self
    }

    /// Sets the model that runs a subagent when the balanced tier is
    /// selected, and the parameters of its requests.
    #[must_use]
    pub fn with_balanced_model(mut self, model: LLM, parameters: Parameters) -> Self {
        self.balanced = model;
        self.parameters.set_balanced(parameters);
        self
    }

    /// Sets the model that runs a subagent when the fast tier is selected and
    /// that routes prompts with no model selector, and the parameters of its
    /// requests.
    #[must_use]
    pub fn with_fast_model(mut self, model: LLM, parameters: Parameters) -> Self {
        self.fast = model;
        self.parameters.set_fast(parameters);
        self
    }

    /// This tool as the main agent's terminal `binding` describes serves it:
    /// subagents launch from that terminal, with at most its authority, and
    /// relative paths and `agents/` resolve against its working directory.
    ///
    /// Register the tool so only the main agent's terminal binds it (see
    /// [`SubagentTool`]). The instance shares the run registry; the
    /// subagents it can continue are its own.
    ///
    /// # Panics
    ///
    /// Panics when `binding` is a subagent's terminal: a subagent has no
    /// `subagent` command, so the tool was registered for it by mistake.
    #[must_use]
    pub fn for_terminal(&self, binding: &TerminalBinding) -> Self {
        assert_eq!(
            binding.role(),
            CommandRole::Main,
            "a subagent's terminal has no `subagent` command: register it with `main_exclusive`"
        );
        Self {
            advanced: self.advanced.clone(),
            balanced: self.balanced.clone(),
            fast: self.fast.clone(),
            parameters: self.parameters.clone(),
            definitions: self.definitions.clone(),
            setups: self.setups.clone(),
            externals: self.externals.clone(),
            base_dir: self.base_dir.clone(),
            mounts: self.mounts.clone(),
            binding: Some(binding.clone()),
            runs: self.runs.clone(),
            cache_keep_alive: self.cache_keep_alive.clone(),
            finished: Arc::new(FinishedSubagents::new(binding.id_allocator().clone())),
            fork_source: self.fork_source.clone(),
            hooks: ExternalHooks::of_observer(binding.observer()).or_else(|| self.hooks.clone()),
            work_events: self.work_events.clone(),
            #[cfg(feature = "languages")]
            languages: self.languages.clone(),
            notices: self.notices.clone(),
            router: self.router.clone(),
            document: self.document.clone(),
            stop_hook: self.stop_hook.clone(),
        }
    }

    /// Records each subagent's start and finish on `events`, the inbox of
    /// the main agent's working document, so a row waiting on a subagent
    /// follows it.
    #[must_use]
    pub fn with_work_events(mut self, events: crate::working_document::WorkEvents) -> Self {
        self.work_events = Some(events);
        self
    }

    /// Has every subagent keep the working document `spec` describes, in
    /// its own space, and decide through `stop_hook` whether its run ends.
    /// [`TerminalAgentBuilder::subagents`](crate::TerminalAgentBuilder::subagents)
    /// sets the main agent's.
    #[must_use]
    pub fn with_working_document(
        mut self,
        spec: Option<crate::working_document::DocumentSpec>,
        stop_hook: Option<crate::AnyStopHook>,
    ) -> Self {
        self.document = spec;
        self.stop_hook = stop_hook;
        self
    }

    /// Records `event` on the main agent's inbox, when there is one.
    fn record(&self, event: crate::working_document::WorkEvent) {
        if let Some(events) = &self.work_events {
            events.record(event);
        }
    }

    /// Tells the hooks and the main agent's inbox that `subagent` starts on `prompt`.
    async fn announce_start(&self, subagent: &str, prompt: &str) {
        if let Some(hooks) = &self.hooks {
            hooks
                .fire(&external_hook::SubagentStart {
                    subagent: subagent.to_string(),
                    prompt: prompt.to_string(),
                })
                .await;
        }
        self.record(crate::working_document::WorkEvent::SubagentStarted(
            subagent.to_string(),
        ));
    }

    /// Refuses a prompt file whose prose is not in the working language
    /// `guard` enforces. The user's own words may be quoted in it verbatim.
    #[cfg(feature = "languages")]
    #[must_use]
    pub fn with_languages(mut self, guard: crate::LanguageGuard) -> Self {
        self.languages = Some(guard);
        self
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
    /// [`AgentOptions::fork_source`] before
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

    /// Adds `setup` to the agent of the subagent `name` resolves to, on top
    /// of what its definition makes: its role prompt, the tier it runs on and
    /// what it may write stay the definition's.
    #[must_use]
    pub fn with_agent_setup<F>(mut self, name: impl Into<String>, setup: F) -> Self
    where
        F: Fn(AgentBuilder<LLM, LLM, LLM, ()>) -> AgentBuilder<LLM, LLM, LLM, ()>
            + Send
            + Sync
            + 'static,
    {
        self.setups.insert(name.into(), Arc::new(setup));
        self
    }

    /// Connects an agent whose turns run outside aither, on `backend`, as
    /// `subagent external --agent <name>`; a later one under the same name
    /// replaces it.
    ///
    /// The tool opens one [`ExternalSession`](super::ExternalSession) per
    /// start call and prompts it again on `resume`; a turn's final message is
    /// the result. The external agent picks its own model, and its sessions
    /// never expire.
    #[must_use]
    pub fn with_external(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        backend: impl ExternalSubagent,
    ) -> Self {
        self.externals.insert(
            name.into(),
            External {
                description: description.into(),
                backend: Arc::new(AnyExternalSubagent::new(backend)),
            },
        );
        self
    }

    /// The directory grants, prompt paths and `agents/` resolve against: the
    /// directory the calling process was in when it ran the command, else
    /// the starting directory of the calling terminal, else the base
    /// directory of a tool bound to none.
    fn caller_dir<'a>(&'a self, requested: Option<&'a Path>) -> Option<&'a Path> {
        requested
            .or_else(|| self.binding.as_ref().map(TerminalBinding::working_dir))
            .or(self.base_dir.as_deref())
    }

    /// Where a subagent name resolves for a call made from `requested`.
    fn lookup<'a>(&'a self, requested: Option<&'a Path>) -> anyhow::Result<SubagentLookup<'a>> {
        let working_dir = self.caller_dir(requested).ok_or_else(|| {
            anyhow::anyhow!(
                "the subagent tool is bound to no terminal and has no base directory, so it has no `{}/` to look subagents up in",
                crate::LOCAL_SUBAGENTS_DIR
            )
        })?;
        Ok(SubagentLookup {
            definitions: &self.definitions,
            working_dir,
        })
    }

    /// The names a call can run, as a call that does not parse lists them.
    async fn usage_text(&self) -> String {
        let lookup = match self.lookup(None) {
            Ok(lookup) => lookup,
            Err(error) => return error.to_string(),
        };
        let subagents = lookup.resolvable().await;
        SubagentUsageTemplate {
            subagents: &subagents,
            fork: FORK_DESCRIPTION,
            externals: self
                .externals
                .iter()
                .map(|(name, external)| (name.as_str(), external.description.as_str()))
                .collect(),
        }
        .render()
        .expect("subagent usage template must render")
    }

    fn effective_mounts(&self, requested: Option<&Path>) -> Vec<SubagentFileMount> {
        // A relative path names a file in the directory the calling process
        // was in first, else in the calling terminal's working directory: a
        // subagent writes its prompts in its own space.
        let caller = requested
            .or_else(|| self.binding.as_ref().map(TerminalBinding::working_dir))
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
            .chain([SubagentFileMount::new(PathBuf::from("."), base)])
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
    async fn read_prompt(
        &self,
        prompt_file: &str,
        requested: Option<&Path>,
    ) -> anyhow::Result<String> {
        let resolved = Self::resolve_subagent_path(
            &self.effective_mounts(requested),
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
        #[cfg(feature = "languages")]
        if let Some(guard) = &self.languages {
            guard
                .check_working(prompt, crate::WorkingText::Internal)
                .map_err(|refusal| {
                    aither_sandbox::Refused::rendered(refusal, |refusal| {
                        self.notices.language_refusal(refusal)
                    })
                })?;
        }
        Ok(prompt.to_string())
    }

    /// The subagent `name` resolves to, refusing a grant its definition does
    /// not let it take.
    async fn named_source(
        &self,
        name: &str,
        grant: Requested<'_>,
        requested: Option<&Path>,
    ) -> anyhow::Result<Resolved<'_, LLM>> {
        let definition = self.lookup(requested)?.resolve(name).await?;
        if definition.access == SubagentAccess::ReadOnly && !grant.is_empty() {
            anyhow::bail!(
                "subagent '{name}' is read-only: it writes only its own working directory and takes no `--worktrees` or `--commit-worktrees`"
            );
        }
        Ok(Resolved {
            name: definition.name.clone(),
            tier: definition.tier,
            source: Source::Definition {
                definition,
                setup: self.setups.get(name),
            },
        })
    }
}

impl<LLM: LanguageModel + Clone + Sync + 'static, Schema: HandoffSchema> SubagentTool<LLM, Schema> {
    /// The main agent's request a fork starts from.
    ///
    /// # Errors
    ///
    /// Refuses, starting nothing, when the tool is bound to no terminal,
    /// when no request was published, and when the main agent's context is
    /// within 10% of its compaction point.
    fn fork_snapshot(&self) -> anyhow::Result<Arc<ForkSnapshot>> {
        if self.binding.is_none() {
            return Err(self.refuse_fork(ForkRefusal::NoTerminal));
        }
        let snapshot = self
            .fork_source
            .latest()
            .ok_or_else(|| self.refuse_fork(ForkRefusal::NothingSent))?;
        if !snapshot.allows_fork() {
            return Err(self.refuse_fork(ForkRefusal::NearCompaction {
                usage: snapshot.usage() * 100.0,
                compaction: snapshot.compaction_usage() * 100.0,
            }));
        }
        Ok(snapshot)
    }

    /// The error refusing a fork for `refusal`, rendered for the model.
    fn refuse_fork(&self, refusal: ForkRefusal) -> anyhow::Error {
        aither_sandbox::Refused::rendered(refusal, |refusal| self.notices.fork_refusal(refusal))
            .into()
    }

    /// Launches the terminal a subagent holding `grant` works from, within the
    /// authority of the terminal this tool serves, for a call made from
    /// `caller`; `None` for a tool bound to no terminal when nothing is
    /// granted.
    async fn launch_terminal(
        &self,
        grant: SubagentGrant,
        observer: Option<aither_sandbox::AnyCommandObserver>,
        gate: Option<aither_sandbox::AnyCommandGate>,
        caller: Option<&Path>,
    ) -> anyhow::Result<Option<DynTerminalTool>> {
        if let Some(binding) = &self.binding {
            let terminal = binding
                .launcher()
                .create(grant, observer, gate, caller)
                .await
                .map_err(|e| anyhow::anyhow!("failed to create subagent terminal tool: {e}"))?;
            Ok(Some(terminal))
        } else if grant == SubagentGrant::new() {
            Ok(None)
        } else {
            anyhow::bail!("worktrees can only be granted to a subagent launched from a terminal");
        }
    }

    /// Opens a session of an external subagent.
    ///
    /// The grant is given through a terminal like any subagent's, which also
    /// gives the commit ledger; the session works in the directories granted.
    async fn start_external(
        &self,
        subagent_id: &str,
        requested: Requested<'_>,
        caller: Option<&Path>,
    ) -> anyhow::Result<FinishedSubagent<LLM, Schema>> {
        let backend = &self
            .externals
            .get(subagent_id)
            .ok_or_else(|| {
                let connected: Vec<&str> = self.externals.keys().map(String::as_str).collect();
                if connected.is_empty() {
                    anyhow::anyhow!("no external agent is connected, so '{subagent_id}' cannot run")
                } else {
                    anyhow::anyhow!(
                        "no external agent '{subagent_id}' is connected; connected: {}",
                        connected.join(", ")
                    )
                }
            })?
            .backend;
        let working_dir = self.caller_dir(caller).ok_or_else(|| {
            anyhow::anyhow!(
                "external agent '{subagent_id}' needs a working directory: the tool is bound to no terminal and has no base directory"
            )
        })?;
        let grant = requested.grant(Some(working_dir));
        let (worktrees, commit_worktrees) = requested.paths(Some(working_dir));
        let terminal = self
            .launch_terminal(grant, None, None, Some(working_dir))
            .await?
            .map(Box::new);
        let commit_ledger = terminal
            .as_ref()
            .and_then(|terminal| terminal.commit_ledger().cloned());
        let working_copies = terminal
            .as_deref()
            .map(prepared_working_copies)
            .unwrap_or_default();
        let session = backend
            .open(ExternalStart {
                working_dir,
                worktrees: &worktrees,
                commit_worktrees: &commit_worktrees,
                working_copies: &working_copies,
            })
            .await
            .map_err(|error| {
                anyhow::anyhow!("failed to open a '{subagent_id}' session: {error}")
            })?;
        tracing::info!(subagent = %subagent_id, "Starting external subagent");
        Ok(FinishedSubagent {
            subagent: subagent_id.to_string(),
            tier: None,
            model: session.model(),
            commit_ledger,
            working_copies,
            body: SubagentBody::External {
                session,
                _terminal: terminal,
            },
        })
    }

    /// Builds a fresh agent for the resolved subagent, on the tier it
    /// declares or the one the fast model routes `prompt` to.
    async fn start_agent(
        &self,
        Resolved {
            name: subagent_id,
            tier: declared,
            source,
        }: Resolved<'_, LLM>,
        grant: SubagentGrant,
        prompt: &str,
        started_by: Option<&CallIdentity>,
        caller: Option<&Path>,
    ) -> anyhow::Result<FinishedSubagent<LLM, Schema>> {
        // A fork is checked before anything is started, and runs on the
        // main agent's model.
        let fork = match &source {
            Source::Fork => Some(self.fork_snapshot()?),
            Source::Definition { .. } => None,
        };

        // A declared tier runs as given; otherwise the fast model routes the
        // prompt to a configured tier.
        let (tier, model) = match &fork {
            Some(fork) => {
                let resolved = self.select_model(prompt, Some(fork.tier())).await?;
                if resolved.1 != fork.model() {
                    return Err(self.refuse_fork(ForkRefusal::OtherModel {
                        main: fork.model().to_string(),
                        fork: resolved.1,
                    }));
                }
                resolved
            }
            None => self.select_model(prompt, declared).await?,
        };
        // A fork's builder takes the main agent's parameters over these when
        // it is made a fork of it.
        let mut selected = crate::AgentBuilder::new(self.model_for(tier).clone())
            .request_parameters(self.parameters.get(tier).clone())
            .notices(self.notices.clone());
        if let Some(spec) = &self.document {
            selected = selected.working_document(spec.clone());
        }
        if let Some(hook) = &self.stop_hook {
            selected = selected.stop_hook(hook.clone());
        }

        let agent_builder = match source {
            Source::Definition { definition, setup } => {
                let builder = selected
                    .agent_kind(crate::AgentKind::Chatbot)
                    .system_prompt(&definition.prompt);
                match setup {
                    Some(setup) => setup(builder),
                    None => builder,
                }
            }
            Source::Fork => selected,
        };
        tracing::info!(subagent = %subagent_id, model = %model, "Starting subagent");
        let child_hooks = self
            .hooks
            .as_ref()
            .map_or_else(ExternalHooks::none, ExternalHooks::for_subagent);
        let command_observer = self.hooks.is_some().then(|| child_hooks.command_observer());
        let command_gate = self.hooks.is_some().then(|| child_hooks.command_gate());

        // Launch the subagent's terminal from the terminal this tool serves.
        let (agent_builder, child_job_registry, commit_ledger, working_copies) = match self
            .launch_terminal(grant, command_observer, command_gate, caller)
            .await?
        {
            Some(dyn_terminal) => {
                let job_registry = dyn_terminal.job_registry();
                let commit_ledger = dyn_terminal.commit_ledger().cloned();
                let working_copies = prepared_working_copies(&dyn_terminal);
                let agent_builder = match &fork {
                    Some(fork) => agent_builder.forked_from(fork, dyn_terminal),
                    None => agent_builder.dyn_terminal(dyn_terminal),
                };
                (
                    agent_builder,
                    Some(job_registry),
                    commit_ledger,
                    working_copies,
                )
            }
            None => (agent_builder, None, None, Vec::new()),
        };

        let identity = AgentIdentity::Subagent(SubagentIdentity::new(
            subagent_id.clone(),
            started_by.cloned(),
        ));
        let agent = self
            .subagent_harness(agent_builder)
            .identity(identity)
            .hook(child_hooks)
            .build();
        let lifetime = agent.cache_lifetime();
        Ok(FinishedSubagent {
            subagent: subagent_id,
            tier: Some(tier),
            model: Some(model),
            commit_ledger,
            working_copies,
            body: SubagentBody::Agent {
                agent: Box::new(agent),
                child_job_registry,
                lifetime,
            },
        })
    }

    /// The harness mechanisms every subagent's agent carries, a named
    /// subagent's and a fork's alike, applied last over what its source and
    /// terminal set.
    ///
    /// Keep-alive is configured only for the Claude client, whose subagents
    /// run on the five-minute cache: in-run waits are covered by the
    /// keep-alive and a finished subagent is continuable until that entry
    /// expires. A fork gets it over the cache it inherits from the main agent.
    /// Every subagent compacts into the tool's handoff schema, the main
    /// agent's.
    fn subagent_harness(
        &self,
        agent_builder: AgentBuilder<LLM, LLM, LLM, ()>,
    ) -> AgentBuilder<LLM, LLM, LLM, (), LLM, Schema> {
        let agent_builder = agent_builder.handoff::<Schema>();
        match &self.cache_keep_alive {
            Some(keep_alive) => agent_builder
                .cache_keep_alive(keep_alive.clone())
                .claude_cache_ttl(aither_core::llm::model::ClaudePromptCacheTtl::FiveMinutes),
            None => agent_builder,
        }
    }

    /// Returns the subagent a call runs: a fresh one for a start call, or
    /// the finished one `resume` names; the second value is the continued
    /// id.
    async fn acquire(
        &self,
        call: Call<'_>,
        prompt: &str,
        started_by: Option<&CallIdentity>,
        caller: Option<&Path>,
    ) -> anyhow::Result<(FinishedSubagent<LLM, Schema>, CallId)> {
        let (resolved, grant) = match call {
            Call::Resume { id } => {
                let (id, finished) = self.finished.checkout(id)?;
                tracing::info!(subagent = %finished.subagent, id = %id.lease, "Continuing subagent");
                return Ok((*finished, id));
            }
            Call::External { agent, grant } => {
                // Leased before the session starts, so no work is lost to a
                // missing id.
                let id = self.finished.lease()?;
                return Ok((self.start_external(agent, grant, caller).await?, id));
            }
            Call::Fork { grant } => (
                Resolved {
                    name: "fork".to_string(),
                    tier: None,
                    source: Source::Fork,
                },
                grant,
            ),
            Call::Named { name, grant } => (self.named_source(name, grant, caller).await?, grant),
        };
        let caller = self.caller_dir(caller);
        let grant = grant.grant(caller);
        // Leased before the subagent is built, so no work is lost to a
        // missing id.
        let id = self.finished.lease()?;
        Ok((
            self.start_agent(resolved, grant, prompt, started_by, caller)
                .await?,
            id,
        ))
    }

    /// Picks the model a subagent call runs on.
    ///
    /// A `chosen` tier — the definition's or the main
    /// agent's for a fork — is used as given. Without one, [`route_model`](Self::route_model)
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

        let tiers = [
            TierOffer {
                tier: ModelTier::Advanced,
                name: &advanced.name,
                description: &advanced.description,
            },
            TierOffer {
                tier: ModelTier::Balanced,
                name: &balanced.name,
                description: &balanced.description,
            },
            TierOffer {
                tier: ModelTier::Fast,
                name: &fast.name,
                description: &fast.description,
            },
        ];
        let request = RouteRequest {
            brief: prompt,
            tiers: &tiers,
        };
        let tier = match &self.router {
            Some(router) => router.route(&request).await,
            None => {
                JudgmentRouter::new(
                    self.fast.clone(),
                    self.parameters.get(ModelTier::Fast).clone(),
                )
                .route(&request)
                .await
            }
        }?;

        let name = match tier {
            ModelTier::Advanced => advanced.name,
            ModelTier::Balanced => balanced.name,
            ModelTier::Fast => fast.name,
        };
        Ok((tier, name))
    }
}

impl<LLM: LanguageModel + Clone + Sync + 'static, Schema: HandoffSchema> SubagentTool<LLM, Schema> {
    /// Records where the subagent's committing worktrees stand before its run,
    /// since commits are listed per run. Nothing has run yet when that fails,
    /// so a continued subagent goes back to where it came from.
    async fn mark_commits(
        &self,
        finished: FinishedSubagent<LLM, Schema>,
        id: &CallId,
    ) -> anyhow::Result<(FinishedSubagent<LLM, Schema>, Option<CommitMark>)> {
        let Some(ledger) = &finished.commit_ledger else {
            return Ok((finished, None));
        };
        match ledger.mark().await {
            Ok(mark) => Ok((finished, Some(mark))),
            Err(error) => {
                if id.resumed {
                    self.finished.store(id.lease.clone(), finished);
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

impl<LLM, Schema> Tool for SubagentTool<LLM, Schema>
where
    LLM: LanguageModel + Clone + Sync + 'static,
    Schema: HandoffSchema,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("subagent")
    }

    type Arguments = SubagentArgs;
    type Res = ToolResult;

    #[allow(clippy::too_many_lines)]
    async fn call(&self, args: Self::Arguments, cx: ToolContext) -> aither_core::Result<Self::Res> {
        let call = args.call()?;
        // The task prompt comes from a file; inline prompt text is rejected.
        // Relative paths and `agents/` resolve where the calling process was.
        let caller = cx.working_dir().map(Path::new);
        let prompt = self.read_prompt(args.prompt_file(), caller).await?;

        let (finished, id) = self.acquire(call, &prompt, cx.identity(), caller).await?;
        let lease = id.lease.clone();
        let resumed = id.resumed;
        let mut hold = CallHold::new(
            self.runs.clone(),
            Arc::clone(&self.finished),
            self.binding.clone(),
            id,
        );
        let (
            FinishedSubagent {
                subagent: subagent_id,
                tier,
                model,
                commit_ledger,
                working_copies,
                body,
            },
            commit_mark,
        ) = self.mark_commits(finished, &hold.id).await?;
        self.announce_start(&subagent_id, &prompt).await;
        let cancel_rx = hold.begin(&subagent_id, &prompt, self.hooks.clone(), body);
        let outcome = hold
            .run(&run_prompt(&prompt, resumed, &working_copies), cancel_rx)
            .await;
        let body = hold.finish();
        self.record(crate::working_document::WorkEvent::SubagentFinished(
            subagent_id.clone(),
        ));
        if let Some(hooks) = &self.hooks {
            hooks
                .fire(&subagent_stop_input(&subagent_id, &outcome))
                .await;
        }
        stop_jobs(body.job_registry(), &subagent_id).await;
        // Whatever way the run ended, the commits it made are reported; a
        // failure to list them is reported in their place and changes nothing
        // else about the run.
        let commits = list_commits(commit_ledger.as_ref(), commit_mark.as_ref()).await;
        let result = match outcome {
            SubagentOutcome::Cancelled => {
                // The cancelled query left the agent mid-turn; it cannot be continued.
                if resumed {
                    self.finished.release(&lease);
                }
                body.end(true).await;
                tracing::info!(subagent = %subagent_id, "Subagent cancelled");
                return Ok(ToolResult::text(
                    SubagentResultTemplate {
                        subagent: &subagent_id,
                        outcome: SubagentReport::Cancelled,
                        commits: commits.as_ref(),
                        working_copies: &working_copies,
                    }
                    .render()
                    .expect("subagent result template must render"),
                ));
            }
            SubagentOutcome::Finished(Ok(result)) => result,
            SubagentOutcome::Finished(Err(error)) => {
                if resumed {
                    self.finished.release(&lease);
                }
                let error_text = SubagentErrorTemplate {
                    subagent: &subagent_id,
                    error: &error,
                    commits: commits.as_ref(),
                    working_copies: &working_copies,
                    history: &body.recent_history(),
                }
                .render()
                .expect("subagent error template must render");
                body.end(false).await;
                return Err(anyhow::anyhow!(error_text));
            }
        };

        tracing::info!(subagent = %subagent_id, "Subagent completed");

        let alert = body.idle_alert(&subagent_id);
        let cache_bound = matches!(body, SubagentBody::Agent { .. });

        // A finished subagent stays continuable on every provider until
        // its prompt-cache lifetime passes; an external session, until the
        // tool is dropped.
        self.finished.store(
            lease.clone(),
            FinishedSubagent {
                subagent: subagent_id.clone(),
                tier,
                model: model.clone(),
                commit_ledger,
                working_copies: working_copies.clone(),
                body,
            },
        );

        Ok(ToolResult::text(
            SubagentResultTemplate {
                subagent: &subagent_id,
                outcome: SubagentReport::Completed {
                    tier: tier.map(tier_label),
                    model: model.as_deref(),
                    result: &result,
                    id: &lease,
                    cache_bound,
                    alert: alert.as_deref(),
                },
                commits: commits.as_ref(),
                working_copies: &working_copies,
            }
            .render()
            .expect("subagent result template must render"),
        ))
    }
}

impl<LLM, Schema> CommandUsage for SubagentTool<LLM, Schema>
where
    LLM: LanguageModel + Clone + Sync + 'static,
    Schema: HandoffSchema,
{
    /// The names a call can run at the moment of the call.
    fn usage(&self) -> impl std::future::Future<Output = String> + Send {
        self.usage_text()
    }
}

/// What a call that does not parse is told it can run.
#[derive(Template)]
#[template(path = "subagent_usage.txt", escape = "none")]
struct SubagentUsageTemplate<'a> {
    /// The names that resolve, the embedder's shadowing local ones.
    subagents: &'a [Listed],
    fork: &'a str,
    /// The connected external agents, by name, with their descriptions.
    externals: Vec<(&'a str, &'a str)>,
}

impl<LLM: LanguageModel + 'static, Schema: HandoffSchema> SubagentBody<LLM, Schema> {
    /// Runs one turn on `prompt` until it ends or `cancel_rx` fires.
    async fn run(
        &mut self,
        prompt: &str,
        cancel_rx: async_channel::Receiver<()>,
    ) -> SubagentOutcome<String> {
        match self {
            Self::Agent { agent, .. } => {
                await_subagent_run(
                    async { agent.query(prompt).await.map_err(|error| error.to_string()) },
                    cancel_rx,
                )
                .await
            }
            Self::External { session, .. } => {
                await_subagent_run(
                    async {
                        session
                            .prompt(prompt)
                            .await
                            .map_err(|error| error.to_string())
                    },
                    cancel_rx,
                )
                .await
            }
        }
    }

    /// The registry of the terminal jobs an aither agent can leave running.
    fn job_registry(&self) -> Option<aither_sandbox::JobRegistry> {
        match self {
            Self::Agent {
                child_job_registry, ..
            } => child_job_registry.clone(),
            Self::External { .. } => None,
        }
    }

    /// Ends a subagent that cannot be continued: an external session is
    /// told to stop its turn first when `cancelled`, then closed.
    async fn end(self, cancelled: bool) {
        if let Self::External { mut session, .. } = self {
            if cancelled {
                session.cancel().await;
            }
            session.close().await;
        }
    }

    /// The alert for an aither agent that kept needing prompt-cache
    /// keep-alives; an external session has no prompt cache of ours.
    fn idle_alert(&self, subagent: &str) -> Option<String> {
        let Self::Agent { agent, .. } = self else {
            return None;
        };
        idle_alert(subagent, agent.cache_keep_alive_state())
    }

    /// The tail of the conversation, for a failed run's error text; an
    /// external session keeps its history to itself.
    fn recent_history(&self) -> Vec<HistoryEntry> {
        let Self::Agent { agent, .. } = self else {
            return Vec::new();
        };
        let history = agent.history();
        let skipped = history.len().saturating_sub(12);
        history
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
            .collect()
    }
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

    use crate::specialized::SubagentRouteError;
    use aither_core::llm::{Event, LLMRequest, model::Profile as ModelProfile};
    use futures_core::Stream;

    use super::*;

    /// The argument schema, as the terminal's command parser reads it.
    fn args_schema() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(SubagentArgs)).expect("schema serializes")
    }

    /// Parses `words` the way the `subagent` terminal command does.
    fn parse_cli(words: &[&str]) -> anyhow::Result<SubagentArgs> {
        let words: Vec<String> = words.iter().map(ToString::to_string).collect();
        let json = aither_sandbox::cli_to_json(&args_schema(), &words)?;
        Ok(serde_json::from_value(json)?)
    }

    #[test]
    fn the_name_and_the_brief_are_the_positional_arguments() {
        let args = parse_cli(&["explore", "p.md"]).expect("a name and a brief parse");
        assert_eq!(args.subagent(), Some("explore"));
        assert_eq!(args.prompt_file(), "p.md");
        assert!(matches!(
            args.call().unwrap(),
            Call::Named {
                name: "explore",
                ..
            }
        ));

        let error = parse_cli(&[]).expect_err("a bare `subagent` names no subagent");
        assert_eq!(error.to_string(), "missing required argument: name");
        let schema = args_schema();
        assert_eq!(
            schema["required"],
            serde_json::json!(["name", "prompt_file"]),
            "only the name and the brief are positional"
        );
        assert!(
            !schema["properties"]
                .as_object()
                .unwrap()
                .contains_key("model"),
            "no call takes a model"
        );
    }

    #[test]
    fn fork_resume_and_external_are_reserved_names_with_their_own_options() {
        let fork = parse_cli(&["fork", "p.md", "--worktrees", "wt"]).expect("fork parses");
        assert_eq!(fork.subagent(), Some("fork"));
        let Call::Fork { grant } = fork.call().unwrap() else {
            panic!("`fork` forks");
        };
        assert_eq!(
            grant.paths(Some(Path::new("/base"))),
            (vec![PathBuf::from("/base/wt")], vec![]),
            "a fork keeps its worktree grants"
        );

        let resume = parse_cli(&["resume", "--id", "river", "--prompt-file", "p.md"])
            .expect("resume parses");
        assert_eq!(resume.subagent(), None);
        assert!(matches!(
            resume.call().unwrap(),
            Call::Resume { id: "river" }
        ));

        let external = parse_cli(&["external", "p.md", "--agent", "codex"]).expect("parses");
        assert_eq!(external.subagent(), Some("codex"));
        assert!(matches!(
            external.call().unwrap(),
            Call::External { agent: "codex", .. }
        ));

        for (words, refused) in [
            (&["resume", "p.md"][..], "takes the `--id`"),
            (
                &["resume", "p.md", "--id", "river", "--worktrees", "wt"],
                "keeps what it was granted",
            ),
            (
                &["explore", "p.md", "--id", "3"],
                "use it with `subagent resume`",
            ),
            (
                &["fork", "p.md", "--agent", "codex"],
                "use it with `subagent external`",
            ),
            (&["external", "p.md"], "takes the `--agent`"),
        ] {
            let error = parse_cli(words).unwrap().call().map(|_| ()).unwrap_err();
            assert!(error.to_string().contains(refused), "{words:?}: {error}");
        }
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
            working_copies: &[],
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
                tier: Some("fast"),
                model: Some("test-model"),
                result: "done",
                id: "river",
                cache_bound: true,
                alert: None,
            },
            commits: None,
            working_copies: &[],
        }
        .render()
        .expect("subagent result should render");
        assert_eq!(
            completed,
            "Subagent 'research' completed (fast tier, model: test-model).\n\ndone\n\nContinue this subagent with `subagent resume --id river` while its prompt cache is still warm."
        );

        let external = SubagentResultTemplate {
            subagent: "codex",
            outcome: SubagentReport::Completed {
                tier: None,
                model: Some("its-model"),
                result: "done",
                id: "river",
                cache_bound: false,
                alert: None,
            },
            commits: None,
            working_copies: &[],
        }
        .render()
        .expect("subagent result should render");
        assert_eq!(
            external,
            "Subagent 'codex' completed (model: its-model).\n\ndone\n\nContinue this subagent with `subagent resume --id river`."
        );

        let without_model = SubagentResultTemplate {
            subagent: "research",
            outcome: SubagentReport::Completed {
                tier: None,
                model: None,
                result: "done",
                id: "river",
                cache_bound: true,
                alert: Some("It idled too often."),
            },
            commits: None,
            working_copies: &[],
        }
        .render()
        .expect("subagent result should render");
        assert_eq!(
            without_model,
            "Subagent 'research' completed.\n\ndone\n\nContinue this subagent with `subagent resume --id river` while its prompt cache is still warm.\n\nIt idled too often."
        );

        let cancelled = SubagentResultTemplate {
            subagent: "explore",
            outcome: SubagentReport::Cancelled,
            commits: None,
            working_copies: &[],
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
                tier: None,
                model: None,
                result: "done",
                id: "river",
                cache_bound: true,
                alert: Some("It idled too often."),
            },
            commits: Some(&listed),
            working_copies: &[],
        }
        .render()
        .expect("subagent result should render");
        assert_eq!(
            completed,
            "Subagent 'worker' completed.\n\ndone\n\nContinue this subagent with `subagent resume --id river` while its prompt cache is still warm.\n\nIt idled too often.\n\nCommits made in the granted worktrees:\n- abc123 fix the parser (in /work/tree)"
        );

        let none = CommitListing {
            listed: Ok(Vec::new()),
            setbacks: Vec::new(),
        };
        let cancelled = SubagentResultTemplate {
            subagent: "worker",
            outcome: SubagentReport::Cancelled,
            commits: Some(&none),
            working_copies: &[],
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
            working_copies: &[],
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

    /// The definitions of six test personas, beside a working directory
    /// whose `agents/` a test may fill.
    struct Personas {
        root: tempfile::TempDir,
    }

    /// A definition of `name` as an embedder would write it.
    fn persona(name: &str, tier: Option<&str>, access: &str) -> SubagentDefinition {
        defined(name, tier, access, &format!("You are the {name} persona."))
    }

    /// A definition of `name` with the role `prompt`.
    fn defined(name: &str, tier: Option<&str>, access: &str, prompt: &str) -> SubagentDefinition {
        let tier = tier.map_or_else(String::new, |tier| format!("tier = \"{tier}\"\n"));
        SubagentDefinition::parse(&format!(
            "name = \"{name}\"\ndescription = \"the {name} persona\"\n{tier}access = \"{access}\"\nprompt = '''\n{prompt}\n'''\n"
        ))
        .unwrap()
    }

    /// The test personas.
    fn persona_set() -> Vec<SubagentDefinition> {
        vec![
            persona("explore", Some("fast"), "read_only"),
            persona("research", Some("fast"), "read_only"),
            persona("audit", Some("advanced"), "read_only"),
            persona("review", Some("advanced"), "read_only"),
            persona("plan", Some("advanced"), "read_only"),
            persona("general", None, "worktrees"),
        ]
    }

    impl Personas {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(root.path().join("work")).unwrap();
            Self { root }
        }

        fn work(&self) -> PathBuf {
            self.root.path().join("work")
        }

        /// A tool on `llm` resolving names in these directories.
        fn tool<LLM: Clone + Sync>(&self, llm: LLM) -> SubagentTool<LLM> {
            SubagentTool::new(llm)
                .with_definitions(persona_set())
                .with_base_dir(self.work())
        }
    }

    /// Writes a definition of `name` into `dir`.
    fn write_definition(dir: &Path, name: &str, tier: Option<&str>, access: &str, prompt: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let tier = tier.map_or_else(String::new, |tier| format!("tier = \"{tier}\"\n"));
        std::fs::write(
            dir.join(format!("{name}.toml")),
            format!(
                "name = \"{name}\"\ndescription = \"the {name} subagent\"\n{tier}access = \"{access}\"\nprompt = '''\n{prompt}\n'''\n"
            ),
        )
        .unwrap();
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
            &tool.effective_mounts(None),
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
            &tool.effective_mounts(None),
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
            &tool.effective_mounts(None),
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

    impl aither_core::llm::Classify for NeverError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    /// Routes a structured-output request to the fast tier.
    const ROUTE_FAST: &str = r#"{"model":"fast"}"#;

    /// A model whose `respond` stream never yields, so the child query
    /// stays in flight until the run is cancelled. It routes every prompt
    /// to the fast tier.
    #[derive(Clone)]
    struct PendingLlm;

    impl LanguageModel for PendingLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            if request.parameters().structured_outputs {
                futures::future::Either::Left(futures_lite::stream::iter([Ok(Event::Text(
                    ROUTE_FAST.to_string(),
                ))]))
            } else {
                futures::future::Either::Right(futures_lite::stream::pending())
            }
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

    /// A call of the subagent `name` reading its brief from `prompt_file`.
    fn named(name: &str, prompt_file: &Path) -> SubagentArgs {
        SubagentArgs {
            name: name.to_string(),
            prompt_file: prompt_file.to_string_lossy().into_owned(),
            id: None,
            agent: None,
            worktrees: Vec::new(),
            commit_worktrees: Vec::new(),
        }
    }

    /// An `explore` call reading its prompt from `prompt_file`.
    fn explore(prompt_file: &Path) -> SubagentArgs {
        named("explore", prompt_file)
    }

    #[tokio::test]
    async fn cancelled_run_returns_cancelled_result() {
        let runs = SubagentRuns::default();
        let personas = Personas::new();
        let tool = personas.tool(PendingLlm).with_runs(runs.clone());
        let (dir, prompt_file) = write_prompt_file("cancelled-run", "survey the codebase");

        let call = tokio::spawn(async move {
            Tool::call(&tool, explore(&prompt_file), ToolContext::new()).await
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

    use aither_sandbox::test_git::git;

    /// A main agent's terminal whose space holds a repository and a linked
    /// worktree on `subagent/work/main`, the worktree a subagent is granted
    /// commit in.
    struct CommitFixture {
        _dirs: [tempfile::TempDir; 2],
        _cache: crate::test_support::TempDir,
        terminal: aither_sandbox::TerminalTool<
            aither_sandbox::permission::NoopPermissionHandler,
            executor_core::tokio::TokioGlobal,
            aither_sandbox::Configured,
        >,
        repository: PathBuf,
        worktree: PathBuf,
    }

    /// The fixture's terminal before its registry is bound.
    type UnboundTerminal = aither_sandbox::TerminalTool<
        aither_sandbox::permission::NoopPermissionHandler,
        executor_core::tokio::TokioGlobal,
    >;

    impl CommitFixture {
        async fn new() -> Self {
            Self::configured(|terminal| terminal).await
        }

        /// The fixture with `configure` applied to its terminal before the
        /// registry is bound.
        async fn configured(configure: impl FnOnce(UnboundTerminal) -> UnboundTerminal) -> Self {
            use aither_sandbox::permission::NoopPermissionHandler;
            use aither_sandbox::{TerminalTool, ToolRegistryBuilder};
            use executor_core::tokio::TokioGlobal;

            let work = tempfile::tempdir().unwrap();
            let source = tempfile::tempdir().unwrap();
            let cache = crate::test_support::TempDir::new();
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
            let terminal = configure(terminal);
            let registry =
                Arc::new(ToolRegistryBuilder::new().build(terminal.output_store().clone()));
            let terminal = terminal.with_registry(registry);
            Self {
                _dirs: [work, source],
                _cache: cache,
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
                .with_definitions(persona_set())
                .with_runs(runs.clone())
                .with_agent_setup("general", |builder| builder.max_iterations(5))
                .for_terminal(self.terminal.binding())
        }

        /// Starts a `general` subagent granted commit in the fixture's
        /// worktree.
        fn start(&self, tag: &str) -> (PathBuf, SubagentArgs) {
            let (dir, prompt_file) = write_prompt_file(tag, "work in the worktree");
            (
                dir,
                SubagentArgs {
                    commit_worktrees: vec![self.worktree.to_string_lossy().into_owned()],
                    ..named("general", &prompt_file)
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
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            if request.parameters().structured_outputs {
                return futures_lite::stream::iter([Ok(Event::Text(ROUTE_FAST.to_string()))]);
            }
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

    /// Clones the launcher's `repository` into each subagent space, as an
    /// embedder gives a run a clone of its session's repository.
    struct CloneSpaces;

    impl aither_sandbox::SubagentSpaces for CloneSpaces {
        type Error = std::io::Error;

        async fn prepare(
            &self,
            run: aither_sandbox::RunSpace<'_>,
        ) -> Result<aither_sandbox::PreparedSpace, std::io::Error> {
            let source = run.launcher.join("repository");
            let clone = run.space.join("repo");
            let target = clone.clone();
            blocking::unblock(move || {
                git(
                    &source,
                    &["clone", "-q", "--local", ".", target.to_str().unwrap()],
                )
            })
            .await;
            Ok(aither_sandbox::PreparedSpace::new().with_working_copy(clone))
        }
    }

    /// The last message of the last turn a [`RecordingTextLlm`] recorded.
    fn last_prompt(seen: &async_channel::Receiver<(&'static str, LLMRequest)>) -> String {
        std::iter::from_fn(|| seen.try_recv().ok())
            .map(|(_, request)| request)
            .filter(|request| !request.parameters().structured_outputs)
            .last()
            .expect("the subagent ran a turn")
            .messages()
            .last()
            .expect("a turn sends messages")
            .content()
            .to_string()
    }

    /// The working copies a result or failure text lists.
    fn listed_working_copies(text: &str) -> Vec<PathBuf> {
        let (_, listing) = text
            .split_once("Its work stays in its working copies until you merge it:\n")
            .unwrap_or_else(|| panic!("the text lists working copies: {text}"));
        listing
            .lines()
            .map_while(|line| line.strip_prefix("- "))
            .map(PathBuf::from)
            .collect()
    }

    /// A run whose space the terminal's spaces prepared is told its working
    /// copies in its first prompt, and its result names them to the main
    /// agent, on every run; they stay after the run for the main agent to
    /// merge.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_run_works_in_the_working_copies_its_space_was_prepared_with() {
        let fixture =
            CommitFixture::configured(|terminal| terminal.with_subagent_spaces(CloneSpaces)).await;
        let runs = SubagentRuns::default();
        let (seen, seen_rx) = async_channel::unbounded();
        let tool = fixture.tool(
            RecordingTextLlm {
                text: TextLlm::routing("model", ROUTE_FAST),
                seen,
            },
            &runs,
        );
        let (dir, prompt_file) = write_prompt_file("prepared-space", "fix the parser");

        let first = completed_text(&tool, named("general", &prompt_file)).await;
        let session = std::fs::canonicalize(fixture.terminal.working_dir()).unwrap();
        let [clone] = listed_working_copies(&first)
            .try_into()
            .unwrap_or_else(|copies| panic!("one working copy: {copies:?}"));
        assert!(
            clone.starts_with(session.join("subagent-spaces")),
            "{first}"
        );
        assert!(clone.join(".git").is_dir(), "the clone outlives the run");
        let prompt = last_prompt(&seen_rx);
        assert!(
            prompt.starts_with(&format!(
                "Your working copies, prepared for this run; commit your work in them, since only commits reach the agent that launched you:\n- {}\n\nfix the parser\n",
                clone.display()
            )),
            "{prompt}"
        );

        let follow_up = dir.join("follow-up.txt");
        std::fs::write(&follow_up, "and the lexer").unwrap();
        let resumed =
            completed_text(&tool, resume_args(&follow_up, &continuation_id(&first))).await;
        assert_eq!(listed_working_copies(&resumed), [clone]);
        let prompt = last_prompt(&seen_rx);
        assert!(
            prompt.starts_with("and the lexer\n"),
            "a continued run was told its working copies when it started: {prompt}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    // Sandboxed git runs only off Windows: a package's ancestors carry
    // traverse alone, and git resolves its paths through a read on each.
    #[cfg(unix)]
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
        assert!(text.starts_with("Subagent 'general' completed"), "{text}");
        assert_eq!(listed_subjects(&text), ["made by the subagent"], "{text}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
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
            failure.starts_with("Subagent 'general' error: "),
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

    // A commit grant is refused where no shell can carry its shim.
    #[cfg(unix)]
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
        assert!(text.starts_with("Subagent 'general' completed"), "{text}");
        assert!(text.contains("\n\ndone\n\n"), "{text}");
        let id = continuation_id(&text);
        assert!(
            matches!(
                tool.finished.slots.lock().unwrap().get(&id),
                Some(FinishedSlot::Idle(..))
            ),
            "the subagent stays continuable: {text}"
        );
        assert!(
            text.contains("The commits made in the granted worktrees could not be listed: "),
            "the listing failure is reported in place of the commits: {text}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
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
            text.starts_with("Subagent 'general' was cancelled by the user."),
            "{text}"
        );
        assert_eq!(listed_subjects(&text), ["made during the run"], "{text}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_grant_without_a_terminal_is_refused() {
        let personas = Personas::new();
        let tool = personas.tool(TextLlm::routing("only-model", ROUTE_FAST));
        let (dir, prompt_file) = write_prompt_file("grant-without-terminal", "do it");
        let error = Tool::call(
            &tool,
            SubagentArgs {
                worktrees: vec!["somewhere".to_string()],
                ..named("general", &prompt_file)
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
        let personas = Personas::new();
        let tool = personas.tool(TextLlm::new("only-model"));

        let error = Tool::call(
            &tool,
            explore(Path::new("summarize the repository layout")),
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
        let personas = Personas::new();
        let tool = personas.tool(TextLlm::new("only-model"));
        let (dir, prompt_file) = write_prompt_file("empty-prompt", "   \n");

        let error = Tool::call(&tool, explore(&prompt_file), ToolContext::new())
            .await
            .expect_err("empty prompt file must be rejected");
        assert!(
            error.to_string().contains("is empty"),
            "error should report the empty file: {error}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A tool whose tiers answer under their own names; the fast model
    /// routes every prompt to `routed`.
    fn tiered_tool(personas: &Personas, routed: &'static str) -> SubagentTool<TextLlm> {
        personas
            .tool(TextLlm::new("advanced-model"))
            .with_balanced_model(TextLlm::new("balanced-model"), Parameters::default())
            .with_fast_model(
                TextLlm::routing("fast-model", routed),
                Parameters::default(),
            )
    }

    /// Runs `args` and returns its result text.
    async fn completed_text<LLM: LanguageModel + Clone + Sync + 'static>(
        tool: &SubagentTool<LLM>,
        args: SubagentArgs,
    ) -> String {
        let result = Tool::call(tool, args, ToolContext::new())
            .await
            .expect("subagent call failed");
        let ToolResult::Text { text } = result else {
            panic!("expected text result, got {result:?}");
        };
        text
    }

    /// A [`TextLlm`] that records each request under its name.
    #[derive(Clone)]
    struct RecordingTextLlm {
        text: TextLlm,
        seen: async_channel::Sender<(&'static str, LLMRequest)>,
    }

    impl LanguageModel for RecordingTextLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.seen
                .try_send((self.text.name, request.clone()))
                .expect("record model request");
            self.text.respond(request)
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            self.text.profile()
        }
    }

    #[tokio::test]
    async fn each_tiers_requests_carry_its_models_parameters_under_aithers_fields() {
        use crate::request_parameters::test_support::{
            assert_cache_controls, host_parameters, host_part,
        };
        use aither_core::llm::model::ReasoningEffort;

        let personas = Personas::new();
        let local = personas.work().join(crate::LOCAL_SUBAGENTS_DIR);
        write_definition(&local, "routed", None, "worktrees", "ROUTED-ROLE");
        write_definition(&local, "deep", Some("advanced"), "worktrees", "DEEP-ROLE");
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let recording = |text| RecordingTextLlm {
            text,
            seen: seen_tx.clone(),
        };
        let shared = host_parameters(ReasoningEffort::High, 0.2);
        let balanced = host_parameters(ReasoningEffort::Medium, 0.5);
        let fast = host_parameters(ReasoningEffort::Low, 0.8);
        let tool = personas
            .tool(recording(TextLlm::new("advanced-model")))
            .with_request_parameters(shared.clone())
            .with_balanced_model(recording(TextLlm::new("balanced-model")), balanced.clone())
            .with_fast_model(
                recording(TextLlm::routing("fast-model", r#"{"model":"balanced"}"#)),
                fast.clone(),
            );
        let (dir, prompt_file) = write_prompt_file("tier-parameters", "do the task");

        completed_text(&tool, named("routed", &prompt_file)).await;
        let requests: Vec<_> = std::iter::from_fn(|| seen_rx.try_recv().ok()).collect();
        let [("fast-model", router), ("balanced-model", turn)] = requests.as_slice() else {
            panic!(
                "the router ran on the fast model, then the subagent on the balanced one: {:?}",
                requests.iter().map(|(name, _)| name).collect::<Vec<_>>()
            );
        };
        assert_eq!(
            host_part(router),
            fast,
            "the router carries the fast model's"
        );
        assert!(router.parameters().structured_outputs);
        assert_eq!(host_part(turn), balanced, "the subagent carries its tier's");
        assert_cache_controls(turn);
        assert!(!turn.parameters().structured_outputs);

        completed_text(&tool, named("deep", &prompt_file)).await;
        let requests: Vec<_> = std::iter::from_fn(|| seen_rx.try_recv().ok()).collect();
        let [("advanced-model", turn)] = requests.as_slice() else {
            panic!(
                "the declared tier ran without routing: {:?}",
                requests.iter().map(|(name, _)| name).collect::<Vec<_>>()
            );
        };
        assert_eq!(
            host_part(turn),
            shared,
            "a tier on the created-with model carries its parameters"
        );
        assert_cache_controls(turn);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn each_embedder_subagent_runs_on_its_stated_tier() {
        // The router would pick balanced, which no persona states.
        let personas = Personas::new();
        let tool = tiered_tool(&personas, r#"{"model":"balanced"}"#);
        let (dir, prompt_file) = write_prompt_file("stated-tier", "do the task");
        let calls = [
            ("explore", "fast"),
            ("research", "fast"),
            ("audit", "advanced"),
            ("review", "advanced"),
            ("plan", "advanced"),
        ];
        let mut ids = std::collections::HashSet::new();
        for (subagent, tier) in calls {
            let text = completed_text(&tool, named(subagent, &prompt_file)).await;
            let id = continuation_id(&text);
            assert_eq!(
                text,
                format!(
                    "Subagent '{subagent}' completed ({tier} tier, model: {tier}-model).\n\n{tier}-model\n\nContinue this subagent with `subagent resume --id {id}` while its prompt cache is still warm."
                )
            );
            assert!(ids.insert(id), "every finished subagent has its own id");
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A script that `cd`s into a skill runs `subagent art_direction
    /// --prompt-file brief.md`: the request carries that directory, and both
    /// the skill's `agents/` and its brief resolve there, not in the
    /// terminal's starting directory.
    #[tokio::test]
    async fn subagent_after_cd_resolves_agents_and_brief_in_the_callers_directory() {
        let personas = Personas::new();
        let skill = personas.root.path().join("work/skills/slide");
        write_definition(
            &skill.join("agents"),
            "art_direction",
            Some("fast"),
            "read_only",
            "direct the art",
        );
        std::fs::write(skill.join("brief.md"), "design the deck").unwrap();
        let tool = tiered_tool(&personas, r#"{"model":"fast"}"#);
        let args = |name: &str| named(name, Path::new("brief.md"));
        let in_skill = || ToolContext::new().with_working_dir(skill.to_string_lossy());

        let result = Tool::call(&tool, args("art_direction"), in_skill())
            .await
            .expect("agents/ and the brief resolve in the caller's directory");
        let ToolResult::Text { text } = result else {
            panic!("expected text result, got {result:?}");
        };
        assert!(
            text.starts_with("Subagent 'art_direction' completed"),
            "{text}"
        );

        // From the terminal's starting directory neither exists.
        let error = Tool::call(&tool, args("art_direction"), ToolContext::new())
            .await
            .expect_err("the starting directory holds neither the brief nor agents/");
        assert!(error.to_string().contains("brief.md"), "{error}");
    }

    #[tokio::test]
    async fn general_runs_on_and_reports_the_tier_routing_picked() {
        let personas = Personas::new();
        let (dir, prompt_file) = write_prompt_file("routed-tier", "do the task");
        for (routed, tier) in [
            (r#"{"model":"balanced"}"#, "balanced"),
            (r#"{"model":"advanced"}"#, "advanced"),
        ] {
            let text = completed_text(
                &tiered_tool(&personas, routed),
                named("general", &prompt_file),
            )
            .await;
            let id = continuation_id(&text);
            assert_eq!(
                text,
                format!(
                    "Subagent 'general' completed ({tier} tier, model: {tier}-model).\n\n{tier}-model\n\nContinue this subagent with `subagent resume --id {id}` while its prompt cache is still warm."
                )
            );
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A router that answers `tier` and counts how often it was asked.
    struct FixedRouter {
        tier: ModelTier,
        asked: Arc<AtomicU64>,
    }

    impl SubagentRouter for FixedRouter {
        fn route(
            &self,
            request: &RouteRequest<'_>,
        ) -> impl Future<Output = Result<ModelTier, SubagentRouteError>> + Send {
            assert_eq!(request.brief, "do the task");
            let offered: Vec<_> = request.tiers.iter().map(|offer| offer.tier).collect();
            assert_eq!(
                offered,
                [ModelTier::Advanced, ModelTier::Balanced, ModelTier::Fast]
            );
            self.asked.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok(self.tier))
        }
    }

    /// A router that fails.
    struct FailingRouter;

    impl SubagentRouter for FailingRouter {
        fn route(
            &self,
            _: &RouteRequest<'_>,
        ) -> impl Future<Output = Result<ModelTier, SubagentRouteError>> + Send {
            std::future::ready(Err(SubagentRouteError::new("no rule applies")))
        }
    }

    #[tokio::test]
    async fn a_custom_router_picks_the_tier_of_an_untiered_definition() {
        let personas = Personas::new();
        let (dir, prompt_file) = write_prompt_file("custom-router", "do the task");
        let asked = Arc::new(AtomicU64::new(0));
        // The default router would pick the balanced tier.
        let tool = tiered_tool(&personas, r#"{"model":"balanced"}"#).with_router(FixedRouter {
            tier: ModelTier::Advanced,
            asked: asked.clone(),
        });

        let text = completed_text(&tool, named("general", &prompt_file)).await;

        assert!(
            text.contains("(advanced tier, model: advanced-model)"),
            "{text}"
        );
        assert_eq!(asked.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_tiered_definition_does_not_consult_the_router() {
        let personas = Personas::new();
        let (dir, prompt_file) = write_prompt_file("tiered-router", "do the task");
        let asked = Arc::new(AtomicU64::new(0));
        let tool = tiered_tool(&personas, r#"{"model":"balanced"}"#).with_router(FixedRouter {
            tier: ModelTier::Advanced,
            asked: asked.clone(),
        });

        let text = completed_text(&tool, named("explore", &prompt_file)).await;

        assert!(text.contains("(fast tier, model: fast-model)"), "{text}");
        assert_eq!(asked.load(Ordering::SeqCst), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_router_failure_stops_the_start() {
        let personas = Personas::new();
        let (dir, prompt_file) = write_prompt_file("failing-router", "do the task");
        let tool = tiered_tool(&personas, r#"{"model":"balanced"}"#).with_router(FailingRouter);

        let error = Tool::call(&tool, named("general", &prompt_file), ToolContext::new())
            .await
            .expect_err("the router failed");

        assert!(error.to_string().contains("no rule applies"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A model that answers with its own name and the role marker its
    /// request carries, so a result shows which tier ran which role prompt.
    /// Structured-output requests get `structured` back, which lets the fast
    /// model play router.
    #[derive(Clone)]
    struct RoleLlm {
        name: &'static str,
        structured: &'static str,
    }

    /// The role prompts [`RoleLlm`] tells apart.
    const ROLE_MARKERS: [&str; 3] = ["EMBEDDER-ROLE", "LOCAL-ROLE", "ONLY-LOCAL-ROLE"];

    impl LanguageModel for RoleLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let text = if request.parameters().structured_outputs {
                self.structured.to_string()
            } else {
                let sent = serde_json::to_string(request.messages()).unwrap();
                let role = ROLE_MARKERS
                    .iter()
                    .rev()
                    .find(|marker| sent.contains(*marker))
                    .unwrap_or(&"no role");
                format!("{} ran {role}", self.name)
            };
            futures_lite::stream::iter([Ok(Event::Text(text))])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                self.name,
                "test",
                self.name,
                "role mock",
                128_000,
            )))
        }
    }

    #[tokio::test]
    async fn embedder_and_local_definitions_run_as_declared_and_an_embedder_name_wins() {
        let personas = Personas::new();
        let local = personas.work().join(crate::LOCAL_SUBAGENTS_DIR);
        write_definition(
            &local,
            "shared",
            Some("advanced"),
            "worktrees",
            "LOCAL-ROLE",
        );
        write_definition(&local, "only_local", None, "worktrees", "ONLY-LOCAL-ROLE");
        let role = |name| RoleLlm {
            name,
            structured: r#"{"model":"balanced"}"#,
        };
        let tool = personas
            .tool(role("advanced-model"))
            .with_definitions([defined(
                "shared",
                Some("fast"),
                "read_only",
                "EMBEDDER-ROLE",
            )])
            .with_balanced_model(role("balanced-model"), Parameters::default())
            .with_fast_model(role("fast-model"), Parameters::default());
        let (dir, prompt_file) = write_prompt_file("definitions", "do the task");

        let shared = completed_text(&tool, named("shared", &prompt_file)).await;
        assert!(
            shared.starts_with(
                "Subagent 'shared' completed (fast tier, model: fast-model).\n\nfast-model ran EMBEDDER-ROLE\n"
            ),
            "the embedder definition shadows the local one: {shared}"
        );
        let only_local = completed_text(&tool, named("only_local", &prompt_file)).await;
        assert!(
            only_local.starts_with(
                "Subagent 'only_local' completed (balanced tier, model: balanced-model).\n\nbalanced-model ran ONLY-LOCAL-ROLE\n"
            ),
            "a local definition without a tier is routed: {only_local}"
        );

        // The embedder definition's access applies: it takes no grant, while
        // the local one it shadows would.
        let error = Tool::call(
            &tool,
            SubagentArgs {
                worktrees: vec!["wt".to_string()],
                ..named("shared", &prompt_file)
            },
            ToolContext::new(),
        )
        .await
        .expect_err("a read-only subagent takes no grant");
        assert!(
            error.to_string().contains("subagent 'shared' is read-only"),
            "{error}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn an_unknown_name_says_where_it_was_looked_for() {
        let personas = Personas::new();
        let tool = personas.tool(TextLlm::new("only-model"));
        let (dir, prompt_file) = write_prompt_file("unknown-name", "do the task");
        let error = Tool::call(&tool, named("nowhere", &prompt_file), ToolContext::new())
            .await
            .expect_err("an unknown name is refused");
        assert_eq!(
            error.to_string(),
            format!(
                "no subagent 'nowhere': the embedder defines none of that name and {} holds no nowhere.toml",
                personas.work().join("agents").display()
            )
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_tool_with_no_definitions_has_only_the_built_in_routes() {
        let personas = Personas::new();
        let tool = SubagentTool::new(TextLlm::new("only-model")).with_base_dir(personas.work());

        let usage = tool.usage().await;
        let listed: Vec<&str> = usage
            .lines()
            .filter_map(|line| line.strip_prefix("- "))
            .filter_map(|line| line.split_once(':').map(|(name, _)| name))
            .collect();
        assert_eq!(listed, ["fork"], "{usage}");
        assert!(usage.contains("`subagent resume --id <id>"), "{usage}");

        let (dir, prompt_file) = write_prompt_file("no-definitions", "do the task");
        for name in ["explore", "research", "audit", "review", "plan", "general"] {
            let error = Tool::call(&tool, named(name, &prompt_file), ToolContext::new())
                .await
                .expect_err("no persona is bundled");
            assert!(
                error.to_string().contains(&format!("no subagent '{name}'")),
                "{error}"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn an_embedders_definitions_are_listed_and_spawnable() {
        let personas = Personas::new();
        let tool = SubagentTool::new(TextLlm::new("only-model"))
            .with_definitions([defined("curator", Some("fast"), "read_only", "CURATE")])
            .with_base_dir(personas.work());

        let usage = tool.usage().await;
        assert!(
            usage.contains("- curator: the curator persona\n"),
            "{usage}"
        );

        let (dir, prompt_file) = write_prompt_file("embedder-definition", "do the task");
        let result = completed_text(&tool, named("curator", &prompt_file)).await;
        assert!(
            result.starts_with("Subagent 'curator' completed (fast tier, model: only-model)."),
            "{result}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    #[should_panic(expected = "reserved for `subagent fork`")]
    fn a_definition_taking_a_built_in_name_is_refused() {
        let taking_fork = SubagentDefinition {
            name: "fork".to_string(),
            ..defined("other", None, "read_only", "x")
        };
        let _ = SubagentTool::new(()).with_definitions([taking_fork]);
    }

    #[tokio::test]
    async fn the_usage_lists_every_name_that_resolves() {
        let personas = Personas::new();
        let local = personas.work().join(crate::LOCAL_SUBAGENTS_DIR);
        write_definition(&local, "only_local", None, "read_only", "ONLY-LOCAL-ROLE");
        write_definition(&local, "explore", None, "read_only", "LOCAL-ROLE");
        std::fs::write(local.join("broken.toml"), "name = ").unwrap();
        let tool = personas.tool(PendingLlm).with_external(
            "outside",
            "runs outside aither",
            TestBackend {
                log: EventLog::default(),
                opened: Arc::default(),
            },
        );

        let usage = tool.usage().await;
        let listed: Vec<&str> = usage
            .lines()
            .filter_map(|line| line.strip_prefix("- "))
            .filter_map(|line| line.split_once(':').map(|(name, _)| name))
            .collect();
        assert_eq!(
            listed,
            [
                "audit",
                "broken",
                "explore",
                "general",
                "only_local",
                "plan",
                "research",
                "review",
                "fork",
                "outside"
            ],
            "{usage}"
        );
        assert!(
            usage.contains(&format!("- explore: {}\n", "the explore persona")),
            "the embedder explore shadows the local one: {usage}"
        );
        assert!(
            usage.contains("- broken: the subagent definition "),
            "an invalid definition is listed with why: {usage}"
        );
        assert!(usage.contains("`subagent resume --id <id>"), "{usage}");
        assert!(
            usage.contains("`subagent external --agent <agent>"),
            "{usage}"
        );
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
                    "terminal",
                    serde_json::json!({
                        "description": "hold the cache window",
                        "script": "sleep 2",
                        "timeout": 60,
                        "background": true
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

    /// The host's parameters of the keep-alive tool's model.
    fn keep_alive_host_parameters() -> Parameters {
        crate::request_parameters::test_support::host_parameters(
            aither_core::llm::model::ReasoningEffort::High,
            0.4,
        )
    }

    /// Builds a subagent tool whose child agent waits on a background
    /// channel driven by the test — no terminal backend needed. Returns the
    /// tool plus the sender halves and the child's job registry, so the
    /// test can park the run and register the command it waits on.
    fn keep_alive_tool(
        personas: &Personas,
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
        let tool = personas
            .tool(KeepAliveMockLlm {
                author,
                requests,
                turns: Arc::default(),
            })
            .with_request_parameters(keep_alive_host_parameters());
        let tool = match keep_alive {
            Some(keep_alive) => tool.with_cache_keep_alive(keep_alive),
            None => tool,
        }
        .with_agent_setup("explore", move |builder| {
            builder
                .max_iterations(4)
                .background_channels(receiver.clone(), child_jobs.clone())
        });
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
                &aither_sandbox::test_ids::lease("amber"),
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
        let personas = Personas::new();
        let (tool, sender, job_registry, job_service) = keep_alive_tool(
            &personas,
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
            Tool::call(&tool, explore(&prompt_file), ToolContext::new()).await
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
                task_id: "amber".to_string(),
                script: "sleep 2".to_string(),
                kill: None,
                timeout_kill: None,
                pending: None,
                result: Ok(aither_sandbox::CompletedCommand {
                    stdout: aither_core::llm::CommandStream::Inline {
                        text: String::new(),
                    },
                    stderr: None,
                    exit_code: 0,
                    interruptions: Vec::new(),
                }),
            })
            .await;

        let result = call
            .await
            .expect("call task panicked")
            .expect("subagent call failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'explore' completed"), "{text}");

        keep_alives.extend(keep_alive_requests(&request_rx));
        let host = keep_alive_host_parameters();
        for request in &keep_alives {
            let params = request.parameters();
            assert_eq!(params.max_tokens, Some(0));
            assert_eq!(
                params.stream,
                Some(false),
                "keep-alive must be non-streaming"
            );
            assert_eq!(
                params.reasoning_effort, host.reasoning_effort,
                "keep-alive must preserve the last request's effort"
            );
            assert!(
                params.include_reasoning,
                "keep-alive must preserve the last request's thinking"
            );
            assert_eq!(params.temperature, host.temperature);
            assert_eq!(params.websearch, host.websearch);
            crate::request_parameters::test_support::assert_cache_controls(request);
        }

        // Two keep-alives crossed the configured threshold: the parent gets
        // one alert naming the subagent and the command it waited on.
        assert!(
            text.contains("Subagent 'explore' needed"),
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
        let personas = Personas::new();
        let (tool, sender, job_registry, job_service) =
            keep_alive_tool(&personas, "Anthropic", request_tx, None);
        tokio::spawn(job_service.serve());
        register_running_job(&job_registry, 4243, "sleep 2").await;
        sender.task_started();

        let call = tokio::spawn(async move {
            Tool::call(&tool, explore(&prompt_file), ToolContext::new()).await
        });

        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        sender.task_finished();
        sender
            .send_completed(aither_sandbox::CompletedTask {
                task_id: "amber".to_string(),
                script: "sleep 2".to_string(),
                kill: None,
                timeout_kill: None,
                pending: None,
                result: Ok(aither_sandbox::CompletedCommand {
                    stdout: aither_core::llm::CommandStream::Inline {
                        text: String::new(),
                    },
                    stderr: None,
                    exit_code: 0,
                    interruptions: Vec::new(),
                }),
            })
            .await;

        let result = call
            .await
            .expect("call task panicked")
            .expect("subagent call failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'explore' completed"), "{text}");
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
        let personas = Personas::new();
        let tool = personas
            .tool(llm)
            .with_agent_setup("explore", |builder| builder.max_iterations(2));
        let (dir, prompt_file) = write_prompt_file("keep-alive-idle", "answer directly");

        let result = Tool::call(&tool, explore(&prompt_file), ToolContext::new())
            .await
            .expect("subagent call failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'explore' completed"), "{text}");
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
        let session_skills_file = workspace_root.join("skills/slide/references/layouts.md");
        let mounted_skill_file = skills_root.join("slide/references/layouts.md");
        std::fs::create_dir_all(session_skills_file.parent().expect("session skill parent"))
            .expect("create session skill parent");
        std::fs::create_dir_all(mounted_skill_file.parent().expect("mounted skill parent"))
            .expect("create mounted skill parent");
        std::fs::write(&mounted_skill_file, "# layouts").expect("write mounted skill file");

        let tool = SubagentTool::new(())
            .with_base_dir(workspace_root.clone())
            .with_file_mounts([
                SubagentFileMount::new(".", workspace_root.clone()),
                SubagentFileMount::new("skills", skills_root.clone()),
            ]);

        let resolved = SubagentTool::<()>::resolve_subagent_path(
            &tool.effective_mounts(None),
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
    ) -> (
        SubagentTool<EchoLlm>,
        async_channel::Receiver<LLMRequest>,
        Personas,
    ) {
        let (request_tx, request_rx) = async_channel::unbounded();
        let personas = Personas::new();
        let tool = personas.tool(EchoLlm {
            requests: request_tx,
            replies: Arc::default(),
        });
        let tool = match keep_alive {
            Some(keep_alive) => tool.with_cache_keep_alive(keep_alive),
            None => tool,
        }
        .with_agent_setup("explore", move |builder| {
            let builder = builder.max_iterations(2);
            match lifetime {
                Some(lifetime) => builder.prompt_cache(
                    crate::context::PromptCache::new("resume-test").with_lifetime(lifetime),
                ),
                None => builder,
            }
        });
        (tool, request_rx, personas)
    }

    fn resume_args(prompt_file: &Path, resume: &str) -> SubagentArgs {
        SubagentArgs {
            id: Some(resume.to_string()),
            ..named("resume", prompt_file)
        }
    }

    /// The continuation id a result text names: a single word.
    fn continuation_id(text: &str) -> String {
        let tail = text
            .split("subagent resume --id ")
            .nth(1)
            .unwrap_or_else(|| panic!("result names a continuation id: {text}"));
        let id = tail.split('`').next().expect("split yields a first part");
        assert!(
            !id.is_empty() && id.bytes().all(|byte| byte.is_ascii_lowercase()),
            "continuation id is one word: {text}"
        );
        id.to_string()
    }

    /// Runs the first call and returns its result text and continuation id.
    async fn first_run(tool: &SubagentTool<EchoLlm>, prompt_file: &Path) -> (String, String) {
        let result = Tool::call(tool, explore(prompt_file), ToolContext::new())
            .await
            .expect("subagent call failed");
        let text = result.render_for_model();
        let id = continuation_id(&text);
        (text, id)
    }

    #[tokio::test]
    async fn finished_subagent_continues_with_its_earlier_conversation() {
        let (tool, requests, _bundled) = resume_tool(
            None,
            Some(crate::config::CacheKeepAlive {
                send_before_expiry: std::time::Duration::ZERO,
                frequent_idle_threshold: 3,
            }),
        );
        let (dir, first_prompt) = write_prompt_file("resume-first", "first question marker");
        let (first, id) = first_run(&tool, &first_prompt).await;
        assert!(first.starts_with("Subagent 'explore' completed"), "{first}");
        assert!(first.contains("\n\nreply-0\n\n"), "{first}");
        while requests.try_recv().is_ok() {}

        let follow_up = dir.join("follow-up.txt");
        std::fs::write(&follow_up, "second question marker").expect("write follow-up");
        let result = Tool::call(&tool, resume_args(&follow_up, &id), ToolContext::new())
            .await
            .expect("continuing failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'explore' completed"), "{text}");
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
        let (tool, _requests, _bundled) = resume_tool(None, None);
        let (dir, prompt_file) = write_prompt_file("resume-unknown", "follow-up");
        let error = Tool::call(
            &tool,
            resume_args(&prompt_file, "zebra"),
            ToolContext::new(),
        )
        .await
        .expect_err("unknown id must fail");
        assert!(
            error.to_string().contains("unknown subagent id zebra"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Answers its first request and never answers another, so a
    /// continuation stays in flight.
    #[derive(Clone, Default)]
    struct HangsAfterFirst {
        replies: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LanguageModel for HangsAfterFirst {
        type Error = NeverError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let n = self
                .replies
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                futures::future::Either::Left(futures_lite::stream::iter([Ok(Event::Text(
                    "first".to_string(),
                ))]))
            } else {
                futures::future::Either::Right(futures_lite::stream::pending())
            }
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

    /// Waits until exactly one run is registered.
    async fn the_registered_run(runs: &SubagentRuns) {
        loop {
            match runs.list().len() {
                1 => return,
                0 => tokio::task::yield_now().await,
                other => panic!("expected one registered run, got {other}"),
            }
        }
    }

    #[tokio::test]
    async fn a_dropped_call_unregisters_its_run() {
        let runs = SubagentRuns::default();
        let personas = Personas::new();
        let tool = personas.tool(PendingLlm).with_runs(runs.clone());
        let (dir, prompt_file) = write_prompt_file("dropped-run", "survey the codebase");

        let call = tokio::spawn(async move {
            Tool::call(&tool, explore(&prompt_file), ToolContext::new()).await
        });
        the_registered_run(&runs).await;

        call.abort();
        assert!(call.await.expect_err("the call was dropped").is_cancelled());

        assert!(
            runs.list().is_empty(),
            "a dropped call must leave no run listed"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_dropped_continuation_releases_its_id() {
        let runs = SubagentRuns::default();
        let personas = Personas::new();
        let tool = Arc::new(
            personas
                .tool(HangsAfterFirst::default())
                .with_runs(runs.clone())
                .with_agent_setup("explore", |builder| builder.max_iterations(2)),
        );
        let (dir, prompt_file) = write_prompt_file("dropped-continuation", "first question");
        let result = Tool::call(&*tool, explore(&prompt_file), ToolContext::new())
            .await
            .expect("first call");
        let id = continuation_id(&result.render_for_model());
        assert!(runs.list().is_empty());

        let continuing = Arc::clone(&tool);
        let follow_up = prompt_file.clone();
        let continued = id.clone();
        let call = tokio::spawn(async move {
            let id = continued;
            Tool::call(
                &*continuing,
                resume_args(&follow_up, &id),
                ToolContext::new(),
            )
            .await
        });
        the_registered_run(&runs).await;

        call.abort();
        assert!(call.await.expect_err("the call was dropped").is_cancelled());

        assert!(runs.list().is_empty(), "the dropped run is unregistered");
        let error = Tool::call(&*tool, resume_args(&prompt_file, &id), ToolContext::new())
            .await
            .expect_err("a dropped continuation cannot be continued");
        assert!(
            !error.to_string().contains("still running"),
            "the id must not stay checked out: {error}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Pretends `age` has passed since the finished subagent's last request,
    /// so expiry runs against its real cache lifetime without sleeping.
    fn age_finished_subagent(tool: &SubagentTool<EchoLlm>, id: &str, age: std::time::Duration) {
        let mut slots = tool
            .finished
            .slots
            .lock()
            .expect("finished subagents lock poisoned");
        let Some(FinishedSlot::Idle(_, finished)) = slots.get_mut(id) else {
            panic!("subagent {id} is not idle");
        };
        let SubagentBody::Agent { agent, .. } = &mut finished.body else {
            panic!("subagent {id} is not an aither agent");
        };
        agent.backdate_cache_entry(age);
        drop(slots);
    }

    #[tokio::test]
    async fn claude_subagent_expires_after_its_five_minute_lifetime() {
        let (tool, _requests, _bundled) =
            resume_tool(None, Some(crate::config::CacheKeepAlive::default()));
        let (dir, prompt_file) = write_prompt_file("resume-expired", "first");
        let (_, id) = first_run(&tool, &prompt_file).await;

        age_finished_subagent(&tool, &id, std::time::Duration::from_secs(301));
        let error = Tool::call(&tool, resume_args(&prompt_file, &id), ToolContext::new())
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

    /// A finished subagent's continuation id is a word leased from the
    /// tool's allocator when its call starts. It stays leased while the
    /// subagent can be continued, a continuation keeping the same word, and
    /// is released once the subagent expires, or with the tool.
    #[tokio::test]
    async fn continuation_ids_are_leased_while_the_subagent_can_be_continued() {
        let ids = aither_sandbox::test_ids::RecordingIds::default();
        let (tool, _requests, _personas) =
            resume_tool(None, Some(crate::config::CacheKeepAlive::default()));
        let tool = tool.with_id_allocator(ids.allocator());
        let (dir, prompt_file) = write_prompt_file("resume-leased", "first");

        let (_, first) = first_run(&tool, &prompt_file).await;
        assert_eq!(first, "alfa");
        Tool::call(&tool, resume_args(&prompt_file, &first), ToolContext::new())
            .await
            .expect("the subagent continues");
        assert_eq!(ids.leased(), ["alfa"], "a continuation keeps its word");
        let (_, second) = first_run(&tool, &prompt_file).await;
        assert_eq!(second, "bravo");
        assert_eq!(ids.live(), ["alfa", "bravo"]);

        age_finished_subagent(&tool, &first, std::time::Duration::from_secs(301));
        Tool::call(&tool, resume_args(&prompt_file, &first), ToolContext::new())
            .await
            .expect_err("the expired subagent cannot be continued");
        assert_eq!(ids.released(), ["alfa"], "expiry releases its word");

        drop(tool);
        assert_eq!(ids.released(), ["alfa", "bravo"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn claude_subagent_continues_just_inside_its_five_minute_lifetime() {
        let (tool, _requests, _bundled) =
            resume_tool(None, Some(crate::config::CacheKeepAlive::default()));
        let (dir, prompt_file) = write_prompt_file("resume-inside", "first");
        let (_, id) = first_run(&tool, &prompt_file).await;

        age_finished_subagent(&tool, &id, std::time::Duration::from_secs(290));
        Tool::call(&tool, resume_args(&prompt_file, &id), ToolContext::new())
            .await
            .expect("a subagent inside its lifetime continues");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A subagent on another client keeps the lifetime of its provider, not
    /// the Claude five minutes.
    #[tokio::test]
    async fn non_claude_subagent_keeps_the_default_lifetime() {
        let (tool, _requests, _bundled) = resume_tool(None, None);
        let (dir, prompt_file) = write_prompt_file("resume-default-lifetime", "first");
        let (_, id) = first_run(&tool, &prompt_file).await;

        age_finished_subagent(&tool, &id, std::time::Duration::from_secs(301));
        Tool::call(&tool, resume_args(&prompt_file, &id), ToolContext::new())
            .await
            .expect("past five minutes a non-Claude subagent still continues");

        age_finished_subagent(&tool, &id, std::time::Duration::from_secs(3601));
        let error = Tool::call(&tool, resume_args(&prompt_file, &id), ToolContext::new())
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

        let (tool, subagent_requests, _bundled) =
            resume_tool(None, Some(crate::config::CacheKeepAlive::default()));
        let (dir, prompt_file) = write_prompt_file("cache-ttl-claude", "first");
        first_run(&tool, &prompt_file).await;
        let subagent_request = subagent_requests.try_recv().expect("subagent request");
        assert_eq!(
            claude_ttl(&subagent_request),
            ClaudePromptCacheTtl::FiveMinutes
        );

        let (other, other_requests, _other_bundled) = resume_tool(None, None);
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
    fn no_call_carries_a_model_or_another_subagent_selector() {
        for extra in ["model", "subagent", "command"] {
            let error = serde_json::from_value::<SubagentArgs>(serde_json::json!({
                "name": "resume",
                "id": "river",
                "prompt_file": "prompt.txt",
                extra: "fast",
            }))
            .expect_err("a call cannot name a model or another selector");
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
        let (tool, requests, _bundled) = resume_tool(None, None);
        let (dir, first_prompt) = write_prompt_file("resume-no-keepalive", "first marker");
        let (first, id) = first_run(&tool, &first_prompt).await;
        assert!(first.starts_with("Subagent 'explore' completed"), "{first}");
        while requests.try_recv().is_ok() {}

        let follow_up = dir.join("follow-up.txt");
        std::fs::write(&follow_up, "second marker").expect("write follow-up");
        let result = Tool::call(&tool, resume_args(&follow_up, &id), ToolContext::new())
            .await
            .expect("continuing failed");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'explore' completed"), "{text}");
        assert!(text.contains("\n\nreply-1\n\n"), "{text}");
        assert_eq!(continuation_id(&text), id);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A permanent prompt-cache lifetime never expires its continuation id.
    #[tokio::test]
    async fn permanent_cache_lifetime_never_expires() {
        let (tool, _requests, _bundled) = resume_tool(Some(CacheLifetime::Permanent), None);
        let (dir, first_prompt) = write_prompt_file("resume-permanent", "first");
        let (_, id) = first_run(&tool, &first_prompt).await;

        // Well past the lifetime the expiry test uses; a permanent cache
        // never goes cold.
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let follow_up = dir.join("follow-up.txt");
        std::fs::write(&follow_up, "second").expect("write follow-up");
        let result = Tool::call(&tool, resume_args(&follow_up, &id), ToolContext::new())
            .await
            .expect("a permanent lifetime never expires");
        let text = result.render_for_model();
        assert!(text.starts_with("Subagent 'explore' completed"), "{text}");
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
            _language: aither_sandbox::ScriptLanguage,
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
        _dirs: [tempfile::TempDir; 2],
        _cache: crate::test_support::TempDir,
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
            Self::build(reviewers, |_| {}).await
        }

        /// A fixture whose terminal serves the commands `commands` registers,
        async fn build(
            reviewers: Vec<crate::external_hook::ReviewerConfig>,
            commands: impl FnOnce(&mut aither_sandbox::ToolRegistryBuilder),
        ) -> Self {
            use aither_sandbox::{TerminalTool, ToolRegistryBuilder};
            use executor_core::tokio::TokioGlobal;

            let session = tempfile::tempdir().unwrap();
            let source = tempfile::tempdir().unwrap();
            let cache = crate::test_support::TempDir::new();
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
            let mut registry = ToolRegistryBuilder::new();
            commands(&mut registry);
            let registry = Arc::new(registry.build(terminal.output_store().clone()));
            Self {
                _dirs: [session, source],
                _cache: cache,
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
                .with_definitions(persona_set())
                .with_request_parameters(crate::request_parameters::test_support::host_parameters(
                    aither_core::llm::model::ReasoningEffort::Low,
                    0.9,
                ))
                .with_balanced_model(ScriptedLlm::new("balanced", seen), Parameters::default())
                .with_fast_model(ScriptedLlm::new("fast", seen), Parameters::default())
                .for_terminal(self.terminal.binding())
        }

        /// A main agent on `llm` with this terminal, publishing to `source`.
        fn main_agent<LLM: LanguageModel + Clone + 'static>(
            &self,
            llm: LLM,
            source: ForkSource,
            policy: crate::ContextPolicy,
        ) -> crate::Agent<LLM, LLM, LLM, ()> {
            let jobs = self.terminal.job_registry();
            crate::AgentBuilder::new(llm)
                .request_parameters(main_agent_parameters())
                .config(
                    crate::AgentConfig::new()
                        .with_system_prompt("You are the main agent.")
                        .with_context_policy(policy),
                )
                .agent_kind(crate::AgentKind::Chatbot)
                .working_document(tasks_document())
                .dyn_terminal(self.terminal.clone().to_dyn())
                .compact_tool()
                .tool(aither_sandbox::builtin::KillTerminalTool::new(jobs.clone()))
                .tool(aither_sandbox::builtin::InputTerminalTool::new(
                    jobs.clone(),
                ))
                .tool(aither_sandbox::builtin::ReadTerminalDeltaTool::new(
                    jobs,
                    aither_sandbox::AnySandboxNotices::default(),
                ))
                .fork_source(source)
                .build()
        }
    }

    /// The host's parameters of the fixture's main agent; the subagent tool's
    /// own tiers carry others.
    fn main_agent_parameters() -> Parameters {
        crate::request_parameters::test_support::host_parameters(
            aither_core::llm::model::ReasoningEffort::High,
            0.1,
        )
    }

    /// A `fork` call, with the directory holding its prompt file.
    fn start_fork(tag: &str) -> (PathBuf, SubagentArgs) {
        let (dir, prompt_file) = write_prompt_file(tag, "carry on from here");
        (dir, named("fork", &prompt_file))
    }

    /// An `audit` call, which runs on the advanced tier as the main agent
    /// does, with the directory holding its prompt file.
    fn start_audit(tag: &str) -> (PathBuf, SubagentArgs) {
        let (dir, prompt_file) = write_prompt_file(tag, "carry on from here");
        (dir, named("audit", &prompt_file))
    }

    fn unsafe_terminal_call(id: &str) -> Event {
        Event::ToolCall(aither_core::llm::ToolCall::new(
            id,
            "terminal",
            serde_json::json!({
                "description": "probe the permission path",
                "script": "echo forked",
                "mode": "unsafe",
                "timeout": 30,
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

        let (dir, args) = start_fork("fork-first-request");
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
        let tail = &first.messages()[shared.len()..];
        assert_eq!(tail.len(), 1, "and continues with the fork prompt alone");
        assert!(
            tail[0]
                .content()
                .starts_with("carry on from here\n\nSent at "),
            "the fork prompt carries its time: {}",
            tail[0].content()
        );
        assert_eq!(first.tool_definitions(), last_main.tool_definitions());
        assert_eq!(
            first.parameters().cache.openai,
            last_main.parameters().cache.openai,
            "the fork reads the main agent's cache"
        );
        let host_part = crate::request_parameters::test_support::host_part;
        assert_eq!(host_part(last_main), main_agent_parameters());
        assert_eq!(
            host_part(first),
            main_agent_parameters(),
            "the fork sends the main agent's parameters, not its tier's"
        );
        crate::request_parameters::test_support::assert_cache_controls(first);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_fork_calling_compact_compacts_itself_and_leaves_the_main_agent_alone() {
        use crate::compact_tool::CompactRequest;

        let fixture = ForkFixture::new().await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        main.then([vec![Event::Text("main answer".to_string())]]);
        let tool = fixture.tool(&main, &seen_tx);
        let mut agent = fixture.main_agent(
            main.clone(),
            tool.fork_source(),
            crate::ContextPolicy {
                compaction_floor: 0,
                ..crate::ContextPolicy::default()
            },
        );
        agent.query("look into it").await.expect("main run");
        drain(&seen_rx);

        main.then([
            vec![Event::ToolCall(aither_core::llm::ToolCall::new(
                "call-1",
                "compact",
                serde_json::json!({}),
            ))],
            vec![Event::ToolCall(aither_core::llm::ToolCall::new(
                "call-2",
                "compact",
                serde_json::json!({}),
            ))],
            vec![Event::Text("fork answer".to_string())],
        ]);
        let (dir, args) = start_fork("fork-compact");
        Tool::call(&tool, args, ToolContext::new())
            .await
            .expect("fork call");

        let requests = drain(&seen_rx);
        let turns: Vec<_> = requests
            .iter()
            .filter(|(_, request)| !request.parameters().structured_outputs)
            .collect();
        let [_, (_, after_first), (_, after_second)] = turns.as_slice() else {
            panic!("the fork sent three turns, got {}", turns.len());
        };
        let results = |request: &LLMRequest| {
            request
                .messages()
                .iter()
                .map(|message| message.content().to_string())
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(
            results(after_first).contains("Compaction started"),
            "the first call started a compaction, not refused: {}",
            results(after_first)
        );
        assert!(
            results(after_second).contains("Nothing was started")
                || results(after_second).contains("nothing was started"),
            "the second call found the fork's own compaction: {}",
            results(after_second)
        );

        let signal = agent.compact_signal.as_ref().expect("main has the tool");
        assert_eq!(
            signal.request(),
            CompactRequest::Started,
            "the main agent's compaction state is untouched"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// An embedder's handoff, filled from the fixture's compaction answer.
    #[derive(Deserialize, JsonSchema)]
    struct TaskOnlyHandoff {
        current_task: String,
    }

    impl HandoffSchema for TaskOnlyHandoff {
        fn instruction(_: &crate::HandoffPrompt<'_>) -> String {
            "Write the task-only handoff.".to_string()
        }

        fn retry(reason: &str) -> String {
            reason.to_owned()
        }

        fn render(&self, _: &[String]) -> String {
            self.current_task.clone()
        }

        fn into_restoration(self) -> crate::Restoration {
            crate::Restoration::default()
        }
    }

    /// A subagent compacts into its tool's handoff schema: the compaction a
    /// fork starts asks with that schema's instruction, not aither's default.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_forks_compaction_uses_the_tools_handoff_schema() {
        let fixture = ForkFixture::new().await;
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let main = ScriptedLlm::new("main", &seen_tx);
        main.then([vec![Event::Text("main answer".to_string())]]);
        let tool = fixture
            .tool(&main, &seen_tx)
            .with_handoff::<TaskOnlyHandoff>();
        let mut agent = fixture.main_agent(
            main.clone(),
            tool.fork_source(),
            crate::ContextPolicy {
                compaction_floor: 0,
                ..crate::ContextPolicy::default()
            },
        );
        agent.query("look into it").await.expect("main run");
        drain(&seen_rx);

        main.then([vec![Event::Text("fork answer".to_string())]]);
        let (dir, args) = start_fork("fork-custom-handoff");
        let result = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect("fork call");
        let id = continuation_id(&result.render_for_model());
        drain(&seen_rx);

        // The fork's own agent compacts, as its auto-compaction or `compact`
        // call would.
        let (_, mut finished) = tool.finished.checkout(&id).expect("the fork is kept");
        let SubagentBody::Agent { agent, .. } = &mut finished.body else {
            panic!("a fork is an aither agent");
        };
        let compacted = agent
            .compact(None)
            .await
            .expect("the fork compacts")
            .expect("the fork has a conversation");
        assert_eq!(compacted.summary, "keep going", "rendered by the schema");

        let requests = drain(&seen_rx);
        let compactions: Vec<_> = requests
            .iter()
            .filter(|(_, request)| request.parameters().structured_outputs)
            .collect();
        let [(_, compaction)] = compactions.as_slice() else {
            panic!("the fork compacted once, got {}", compactions.len());
        };
        let asked: Vec<&str> = compaction
            .messages()
            .iter()
            .map(aither_core::llm::Message::content)
            .collect();
        assert!(asked.contains(&"Write the task-only handoff."), "{asked:?}");
        assert!(
            !asked
                .iter()
                .any(|content| content.contains("Your context window is being compacted")),
            "aither's default instruction is not sent: {asked:?}"
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
        let (fork_dir, fork) = start_fork("fork-permission");
        Tool::call(&tool, fork, ToolContext::new())
            .await
            .expect("fork call");
        let asked_by_fork = std::mem::take(&mut *fixture.asked.lock().unwrap());

        main.then(probe());
        let (worker_dir, worker) = start_audit("worker-permission");
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
        let (worker_dir, worker) = start_audit("worker-reviewed");
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
            compact_ceiling: usize::MAX,
            start_distance: 0,
            safe_buffer: 0,
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
        let (dir, args) = start_fork("fork-refused");
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
        let personas = Personas::new();
        let tool = personas.tool(main);
        let (dir, args) = start_fork("fork-unbound");
        let error = Tool::call(&tool, args, ToolContext::new())
            .await
            .expect_err("a tool bound to no terminal cannot fork");
        assert!(error.to_string().contains("needs the terminal"), "{error}");

        let _ = std::fs::remove_dir_all(dir);
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
        let (dir, args) = start_fork("fork-other-model");
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
                &aither_sandbox::test_ids::lease("primary"),
                "exec-key",
                "long running main job",
                aither_sandbox::TerminalMode::Sandboxed,
                None,
            )
            .await;

        main.then([vec![Event::ToolCall(aither_core::llm::ToolCall::new(
            "bg-1",
            "terminal",
            serde_json::json!({
                "description": "start a job",
                "script": "sleep 30",
                "timeout": 60,
                "background": true,
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
        main.then([vec![control_call("kill-2", "terminal_kill", "primary")]]);
        main.then([vec![Event::Text("finished".to_string())]]);

        let (dir, args) = start_fork("fork-controls");
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
                .any(|text| text.contains("No running background task 'primary'")),
            "the main agent's job is out of reach: {results:?}"
        );
        assert!(
            main_jobs
                .list()
                .await
                .iter()
                .any(|job| job.task_id == "primary"
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
        let personas = Personas::new();
        let tool = personas
            .tool(TextLlm::new("only-model"))
            .with_hooks(main.clone());
        let (dir, prompt_file) = write_prompt_file("subagent-hooks", "do it");

        tokio::time::timeout(
            std::time::Duration::from_secs(60),
            Tool::call(&tool, explore(&prompt_file), ToolContext::new()),
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
        assert_eq!(inputs[0]["subagent"], "explore");
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

    type EventLog = Arc<Mutex<Vec<String>>>;

    #[derive(Debug, thiserror::Error)]
    #[error("backend refused: {0}")]
    struct BackendError(String);

    /// Opens sessions that log what they are asked to do. A prompt `hang`
    /// never ends and a prompt `fail` errors; any other is answered with the
    /// session's turn count.
    struct TestBackend {
        log: EventLog,
        opened: Arc<AtomicU64>,
    }

    struct TestSession {
        number: u64,
        turns: u64,
        log: EventLog,
    }

    impl TestSession {
        fn record(&self, event: String) {
            self.log.lock().expect("log lock").push(event);
        }
    }

    impl ExternalSubagent for TestBackend {
        type Session = TestSession;
        type Error = BackendError;

        fn open(
            &self,
            start: ExternalStart<'_>,
        ) -> impl Future<Output = Result<TestSession, BackendError>> + Send {
            let number = self.opened.fetch_add(1, Ordering::Relaxed);
            self.log
                .lock()
                .expect("log lock")
                .push(format!("open {number} in {}", start.working_dir.display()));
            std::future::ready(Ok(TestSession {
                number,
                turns: 0,
                log: self.log.clone(),
            }))
        }
    }

    impl crate::specialized::ExternalSession for TestSession {
        type Error = BackendError;

        async fn prompt(&mut self, prompt: &str) -> Result<String, BackendError> {
            self.turns += 1;
            self.record(format!("prompt {} {prompt}", self.number));
            match prompt {
                "hang" => std::future::pending().await,
                "fail" => Err(BackendError("fail".to_string())),
                _ => Ok(format!(
                    "turn {} of session {}: {prompt}",
                    self.turns, self.number
                )),
            }
        }

        async fn cancel(&mut self) {
            self.record(format!("cancel {}", self.number));
        }

        async fn close(self) {
            self.record(format!("close {}", self.number));
        }

        fn model(&self) -> Option<String> {
            Some("external-model".to_string())
        }
    }

    fn external_tool(dir: &Path) -> (SubagentTool<PendingLlm>, SubagentRuns, EventLog) {
        let log = EventLog::default();
        let runs = SubagentRuns::default();
        let tool = SubagentTool::new(PendingLlm)
            .with_base_dir(dir)
            .with_runs(runs.clone())
            .with_external(
                "outside",
                "runs outside aither",
                TestBackend {
                    log: log.clone(),
                    opened: Arc::default(),
                },
            );
        (tool, runs, log)
    }

    fn external_start(agent: &str, prompt_file: &Path) -> SubagentArgs {
        SubagentArgs {
            agent: Some(agent.to_string()),
            ..named("external", prompt_file)
        }
    }

    fn events(log: &EventLog) -> Vec<String> {
        log.lock().expect("log lock").clone()
    }

    async fn call_text(
        tool: &SubagentTool<PendingLlm>,
        args: SubagentArgs,
    ) -> aither_core::Result<String> {
        match Tool::call(tool, args, ToolContext::new()).await? {
            ToolResult::Text { text } => Ok(text),
            other => panic!("expected a text result, got {other:?}"),
        }
    }

    #[cfg(feature = "languages")]
    #[tokio::test]
    async fn a_prompt_not_in_the_working_language_is_refused() {
        let (dir, chinese) =
            write_prompt_file("language-refused", "请检查数据库迁移脚本是否正确。");
        let english = dir.join("english.txt");
        std::fs::write(
            &english,
            "Check that the database migration script is correct.",
        )
        .expect("write prompt file");
        let (tool, _runs, _log) = external_tool(&dir);
        let tool = tool.with_languages(crate::LanguageGuard::new(crate::Languages::new(
            "Chinese".parse().unwrap(),
        )));

        let error = call_text(&tool, external_start("outside", &chinese))
            .await
            .expect_err("a Chinese prompt is refused");
        assert!(error.to_string().contains("English"), "{error}");
        call_text(&tool, external_start("outside", &english))
            .await
            .expect("an English prompt starts");
    }

    #[tokio::test]
    async fn external_subagent_starts_and_resumes_the_same_session() {
        let (dir, first) = write_prompt_file("external-resume", "first");
        let second = dir.join("second.txt");
        std::fs::write(&second, "second").expect("write prompt file");
        let (tool, runs, log) = external_tool(&dir);

        let text = call_text(&tool, external_start("outside", &first))
            .await
            .expect("start succeeds");
        assert!(text.contains("turn 1 of session 0: first"), "{text}");
        assert!(text.contains("(model: external-model)"), "{text}");
        assert!(!text.contains("keep-alives"), "{text}");
        assert!(!text.contains("prompt cache"), "{text}");
        assert!(runs.list().is_empty());

        let id = continuation_id(&text);
        let text = call_text(&tool, resume_args(&second, &id))
            .await
            .expect("resume succeeds");
        assert!(text.contains("turn 2 of session 0: second"), "{text}");
        assert_eq!(continuation_id(&text), id);
        assert_eq!(
            events(&log),
            [
                format!("open 0 in {}", dir.display()),
                "prompt 0 first".to_string(),
                "prompt 0 second".to_string(),
            ]
        );
        assert!(
            tool.usage()
                .await
                .ends_with("--prompt-file <path>`:\n- outside: runs outside aither"),
            "the usage lists the connected agent"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn an_agent_that_is_not_connected_is_refused_with_the_connected_ones() {
        let (dir, prompt_file) = write_prompt_file("external-unknown", "first");
        let (tool, _runs, log) = external_tool(&dir);

        let error = call_text(&tool, external_start("elsewhere", &prompt_file))
            .await
            .expect_err("an unknown agent is refused");
        assert_eq!(
            error.to_string(),
            "no external agent 'elsewhere' is connected; connected: outside"
        );
        assert!(events(&log).is_empty(), "no session was opened");

        let error = call_text(
            &SubagentTool::new(PendingLlm).with_base_dir(&dir),
            external_start("outside", &prompt_file),
        )
        .await
        .expect_err("no agent is connected");
        assert!(
            error
                .to_string()
                .starts_with("no external agent is connected"),
            "{error}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn external_subagent_run_can_be_cancelled() {
        let (dir, prompt_file) = write_prompt_file("external-cancel", "hang");
        let (tool, runs, log) = external_tool(&dir);

        let call =
            tokio::spawn(
                async move { call_text(&tool, external_start("outside", &prompt_file)).await },
            );
        let run = loop {
            match runs.list().as_slice() {
                [run] => break run.clone(),
                [] => tokio::task::yield_now().await,
                other => panic!("expected one registered run, got {}", other.len()),
            }
        };
        assert_eq!(run.subagent, "outside");
        assert!(runs.cancel(run.id));

        let text = call
            .await
            .expect("call task panicked")
            .expect("a cancelled run is a result");
        assert!(text.contains("was cancelled"), "{text}");
        assert!(runs.list().is_empty());
        assert_eq!(
            events(&log),
            [
                format!("open 0 in {}", dir.display()),
                "prompt 0 hang".to_string(),
                "cancel 0".to_string(),
                "close 0".to_string(),
            ]
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn external_subagent_prompt_error_closes_the_session() {
        let (dir, prompt_file) = write_prompt_file("external-error", "fail");
        let (tool, runs, log) = external_tool(&dir);

        let error = call_text(&tool, external_start("outside", &prompt_file))
            .await
            .expect_err("a failed turn is an error");
        assert!(
            error.to_string().contains("backend refused: fail"),
            "{error}"
        );
        assert!(runs.list().is_empty());
        assert_eq!(
            events(&log),
            [
                format!("open 0 in {}", dir.display()),
                "prompt 0 fail".to_string(),
                "close 0".to_string(),
            ]
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn external_subagent_needs_a_working_directory() {
        let (dir, prompt_file) = write_prompt_file("external-no-dir", "first");
        let tool = SubagentTool::new(PendingLlm).with_external(
            "outside",
            "runs outside aither",
            TestBackend {
                log: EventLog::default(),
                opened: Arc::default(),
            },
        );

        let error = call_text(&tool, external_start("outside", &prompt_file))
            .await
            .expect_err("no working directory is an error");
        assert!(error.to_string().contains("working directory"), "{error}");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The prompt that makes [`WritingLlm`] write.
    const WRITE_PROBE: &str = "write probe";

    /// A model that, given the [`WRITE_PROBE`] prompt, runs one command
    /// writing its own working directory, the workspace and a directory in
    /// it, then answers. Any other prompt it answers at once, and it routes
    /// every prompt to the fast tier.
    #[derive(Clone)]
    struct WritingLlm {
        workspace: PathBuf,
    }

    impl WritingLlm {
        fn script(&self) -> String {
            [
                "touch scratch.txt && echo scratch-written".to_string(),
                format!(
                    "touch '{}' || echo workspace-refused",
                    self.workspace.join("probe.txt").display()
                ),
                format!(
                    "touch '{}' || echo granted-refused",
                    self.workspace.join("granted").join("probe.txt").display()
                ),
            ]
            .join("; ")
        }
    }

    impl LanguageModel for WritingLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let event = if request.parameters().structured_outputs {
                Event::Text(ROUTE_FAST.to_string())
            } else {
                match request.messages().last() {
                    Some(aither_core::llm::Message::Tool { content, .. }) => {
                        Event::Text(content.clone())
                    }
                    Some(message) if message.content().contains(WRITE_PROBE) => {
                        Event::ToolCall(aither_core::llm::ToolCall::new(
                            "write-1",
                            "terminal",
                            serde_json::json!({
                                "description": "write probe",
                                "script": self.script(),
                                "timeout": 30,
                            }),
                        ))
                    }
                    _ => Event::Text("ok".to_string()),
                }
            };
            futures_lite::stream::iter([Ok(event)])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "writing",
                "test",
                "writing-model",
                "writes the workspace",
                1_000_000,
            )))
        }
    }

    /// The spaces, under the main terminal's working directory, whose
    /// subagent wrote `scratch.txt` there.
    fn spaces_written(workspace: &Path) -> usize {
        std::fs::read_dir(workspace.join("subagent-spaces"))
            .expect("a subagent ran, so the spaces directory exists")
            .map(|entry| entry.expect("a space entry reads").path())
            .filter(|space| space.join("scratch.txt").is_file())
            .count()
    }

    // Sandboxed writes are confined only where heel's per-command sandbox
    // runs, as in the sandbox crate's own confinement tests.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn read_only_subagents_write_only_their_scratch_directory_and_writers_their_grants() {
        let fixture = ForkFixture::new().await;
        let workspace = std::fs::canonicalize(fixture.terminal.working_dir()).unwrap();
        let granted = workspace.join("granted");
        std::fs::create_dir_all(&granted).unwrap();
        let local = workspace.join(crate::LOCAL_SUBAGENTS_DIR);
        write_definition(&local, "helper", Some("fast"), "read_only", "You help.");
        write_definition(&local, "writer", Some("fast"), "worktrees", "You write.");
        let llm = WritingLlm {
            workspace: workspace.clone(),
        };
        let tool = SubagentTool::new(llm.clone())
            .with_definitions(persona_set())
            .for_terminal(fixture.terminal.binding());
        let mut main = fixture.main_agent(llm, tool.fork_source(), crate::ContextPolicy::default());
        main.query("look around").await.expect("main run");
        let (dir, prompt_file) = write_prompt_file("read-only-writes", WRITE_PROBE);

        let read_only = [
            "explore", "research", "audit", "review", "plan", "helper", "fork",
        ];
        let mut runs = 0;
        for subagent in read_only {
            runs += 1;
            let text = completed_text(&tool, named(subagent, &prompt_file)).await;
            assert!(text.contains("scratch-written"), "{subagent}: {text}");
            assert!(text.contains("workspace-refused"), "{subagent}: {text}");
            assert!(text.contains("granted-refused"), "{subagent}: {text}");
            assert!(
                !workspace.join("probe.txt").exists(),
                "{subagent} wrote the workspace"
            );
            assert!(
                !granted.join("probe.txt").exists(),
                "{subagent} wrote the workspace"
            );
            assert_eq!(
                spaces_written(&workspace),
                runs,
                "{subagent} wrote its scratch directory"
            );
        }

        for subagent in ["general", "writer", "fork"] {
            runs += 1;
            let text = completed_text(
                &tool,
                SubagentArgs {
                    worktrees: vec![granted.to_string_lossy().into_owned()],
                    ..named(subagent, &prompt_file)
                },
            )
            .await;
            assert!(text.contains("workspace-refused"), "{subagent}: {text}");
            assert!(!text.contains("granted-refused"), "{subagent}: {text}");
            assert!(
                !workspace.join("probe.txt").exists(),
                "{subagent} wrote ungranted workspace"
            );
            assert!(
                granted.join("probe.txt").exists(),
                "{subagent} writes its grant"
            );
            std::fs::remove_file(granted.join("probe.txt")).unwrap();
            assert_eq!(spaces_written(&workspace), runs);
        }
        assert!(
            fixture.asked.lock().unwrap().is_empty(),
            "nothing asked to escalate"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The brief that makes [`ProbingLlm`] run its probe.
    const PROBE: &str = "run the probe";

    /// A model that, given the [`PROBE`] brief, runs `script` in its
    /// terminal and answers with what it printed; it routes every brief to
    /// the fast tier and records each request's messages.
    #[derive(Clone)]
    struct ProbingLlm {
        script: &'static str,
        seen: async_channel::Sender<String>,
    }

    impl LanguageModel for ProbingLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            let event = if request.parameters().structured_outputs {
                Event::Text(ROUTE_FAST.to_string())
            } else {
                self.seen
                    .try_send(serde_json::to_string(request.messages()).unwrap())
                    .unwrap();
                // The latest tool result since the brief: a fork's inherited
                // context holds its parent's earlier results.
                let probe_result = request
                    .messages()
                    .iter()
                    .rev()
                    .take_while(|message| {
                        !matches!(message, aither_core::llm::Message::User { .. })
                    })
                    .find_map(|message| match message {
                        aither_core::llm::Message::Tool { content, .. } => Some(content.clone()),
                        _ => None,
                    });
                match (probe_result, request.messages().last()) {
                    // The harness may follow a failed command's result with a
                    // notice, so the probe's result need not be the last message.
                    (Some(content), _) => Event::Text(content),
                    (None, Some(message)) if message.content().contains(PROBE) => {
                        Event::ToolCall(aither_core::llm::ToolCall::new(
                            "probe-1",
                            "terminal",
                            serde_json::json!({
                                "description": "probe",
                                "script": self.script,
                                "timeout": 30,
                            }),
                        ))
                    }
                    _ => Event::Text("ok".to_string()),
                }
            };
            futures_lite::stream::iter([Ok(event)])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            std::future::ready(Ok(ModelProfile::new(
                "probing",
                "test",
                "probing-model",
                "runs a probe",
                1_000_000,
            )))
        }
    }

    /// Reports whether the terminal has a `subagent` command, then calls it
    /// bare.
    const COMMAND_PROBE: &str = "if command -v subagent >/dev/null 2>&1; then echo has-subagent; else echo no-subagent; fi; subagent 2>&1; true";

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagent_has_no_subagent_command_and_a_bare_call_lists_the_names() {
        let (seen_tx, _seen_rx) = async_channel::unbounded();
        let llm = ProbingLlm {
            script: COMMAND_PROBE,
            seen: seen_tx,
        };
        let shared = Arc::new(SubagentTool::new(llm.clone()).with_definitions(persona_set()));
        let bound = shared.clone();
        let fixture = ForkFixture::build(Vec::new(), move |registry| {
            registry.main_exclusive(|main| {
                main.configure_terminal_bound_usage_tool("subagent", move |binding| {
                    bound.for_terminal(binding)
                });
            });
        })
        .await;
        let tool = shared.for_terminal(fixture.terminal.binding());

        let mut main = fixture.main_agent(llm, tool.fork_source(), crate::ContextPolicy::default());
        let answer = main.query(PROBE).await.expect("main run");
        assert!(
            answer.contains("has-subagent"),
            "the main agent has the command: {answer}"
        );
        assert!(
            answer.contains("missing required argument: name\n\nName the subagent to run"),
            "a bare `subagent` lists the names: {answer}"
        );
        for name in ["explore", "general", "fork"] {
            assert!(answer.contains(&format!("\n- {name}: ")), "{answer}");
        }

        let (dir, prompt_file) = write_prompt_file("nesting", PROBE);
        for subagent in ["explore", "fork"] {
            let text = completed_text(&tool, named(subagent, &prompt_file)).await;
            assert!(
                text.contains("no-subagent"),
                "{subagent} has no `subagent` command: {text}"
            );
            assert!(
                !text.contains("Name the subagent to run"),
                "{subagent} reached no `subagent` command: {text}"
            );
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The working document the fixture's agents keep: `tasks.md`, read by
    /// the test schema.
    fn tasks_document() -> crate::DocumentSpec {
        crate::DocumentSpec::new("tasks.md", crate::working_document::test_schema::ListSchema)
    }

    /// Writes a `tasks.md` with no open row, so the subagent ends after its
    /// two requests.
    const TASKS_PROBE: &str =
        "printf '%s\\n' '## Notes' 'SUBAGENT-TASK' > tasks.md && echo written";
    const SUBAGENT_TASKS: &str = "## Notes\nSUBAGENT-TASK\n";
    const PARENT_TASKS: &str = "## Notes\nPARENT-TASK\n";

    /// The `tasks.md` the subagents under `workspace` wrote in their spaces.
    fn subagent_tasks(workspace: &Path) -> Vec<String> {
        std::fs::read_dir(workspace.join("subagent-spaces"))
            .unwrap()
            .filter_map(|entry| {
                std::fs::read_to_string(entry.unwrap().path().join("tasks.md")).ok()
            })
            .collect()
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagent_keeps_its_own_tasks_md_and_never_touches_its_parents() {
        let fixture = ForkFixture::new().await;
        let workspace = std::fs::canonicalize(fixture.terminal.working_dir()).unwrap();
        let parent_tasks = workspace.join("tasks.md");
        std::fs::write(&parent_tasks, PARENT_TASKS).unwrap();
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let llm = ProbingLlm {
            script: TASKS_PROBE,
            seen: seen_tx,
        };
        let tool = SubagentTool::new(llm.clone())
            .with_definitions(persona_set())
            .with_working_document(Some(tasks_document()), None)
            .for_terminal(fixture.terminal.binding());
        let mut main = fixture.main_agent(llm, tool.fork_source(), crate::ContextPolicy::default());
        main.query("look around").await.expect("main run");
        let main_requests: Vec<String> = std::iter::from_fn(|| seen_rx.try_recv().ok()).collect();
        let parent_marks = main_requests
            .last()
            .expect("the main agent sent a request")
            .matches("PARENT-TASK")
            .count();
        assert!(parent_marks > 0, "the main agent sees its own tasks.md");
        let (dir, prompt_file) = write_prompt_file("subagent-tasks", PROBE);

        for (run, subagent) in (1..).zip(["explore", "fork"]) {
            let text = completed_text(&tool, named(subagent, &prompt_file)).await;
            assert!(text.contains("written"), "{subagent}: {text}");
            let requests: Vec<String> = std::iter::from_fn(|| seen_rx.try_recv().ok()).collect();
            let [first, after_write] = requests.as_slice() else {
                panic!("{subagent} sent two requests, got {}", requests.len());
            };
            // A fork's context is the main agent's, which already carries the
            // main agent's tasks.md; its own source adds nothing of it.
            let inherited = if subagent == "fork" { parent_marks } else { 0 };
            assert_eq!(
                first.matches("PARENT-TASK").count(),
                inherited,
                "{subagent} does not read its parent's tasks.md"
            );
            assert!(
                after_write.contains("SUBAGENT-TASK"),
                "{subagent}'s own tasks.md reaches its context: {after_write}"
            );
            assert_eq!(
                std::fs::read_to_string(&parent_tasks).unwrap(),
                PARENT_TASKS,
                "{subagent} left its parent's tasks.md alone"
            );
            assert_eq!(
                subagent_tasks(&workspace),
                vec![SUBAGENT_TASKS.to_string(); run],
                "{subagent} wrote tasks.md in its own space"
            );
        }

        let _ = std::fs::remove_dir_all(dir);
    }
}
