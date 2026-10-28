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
use aither_sandbox::TerminalToolFactory;
use askama::Template;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::AgentBuilder;
use crate::fs_util::path_exists;
use crate::model_group::ModelTier;
use crate::subagent_file::SubagentDefinition;

async fn checked_path_exists(path: &Path) -> anyhow::Result<bool> {
    path_exists(path)
        .await
        .map_err(|error| anyhow::anyhow!("failed to inspect path '{}': {error}", path.display()))
}

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
    /// Builder function that creates the configured agent builder.
    builder: SubagentBuilder<LLM>,
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
            builder: Arc::new(builder),
        }
    }

    /// Declare the model tier this subagent runs on unless the caller
    /// overrides it in `SubagentArgs::model`.
    #[must_use]
    pub const fn with_model(mut self, model: ModelTier) -> Self {
        self.model = Some(model);
        self
    }

    /// Get the agent builder for this subagent type.
    pub fn builder(&self, llm: LLM) -> AgentBuilder<LLM, LLM, LLM, ()> {
        (self.builder)(llm)
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
/// You can also load custom subagents from `.md` files by specifying a path
/// (e.g., `./custom-agent.md` or `subagents/researcher.md`).
///
/// The task prompt is read from a file: write the full prompt to a file
/// first, then pass its path in `prompt_file`. Inline prompt text is not a
/// file path and is rejected.
#[derive(Debug, Clone, JsonSchema, Deserialize)]
pub struct SubagentArgs {
    /// Subagent type name (e.g., "explore", "plan") or path to a subagent file.
    /// If the value contains '/' or ends with '.md', it's treated as a file path.
    pub subagent: String,
    /// Path to a file containing the detailed task prompt for the subagent.
    /// The file must exist and be non-empty.
    pub prompt_file: String,
    /// Model tier to run the subagent on. When omitted and the subagent
    /// definition declares none, the fast model reads the prompt and routes
    /// it to one of the configured tiers.
    pub model: Option<ModelTier>,
}

/// How the model that ran a subagent was selected.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum ModelSource {
    /// The call's `model` argument or the subagent definition chose it.
    Chosen,
    /// The fast model routed the prompt to it.
    Routed,
}

/// The model that ran a subagent, reported back to the caller.
#[derive(Debug, Clone, Serialize)]
struct SubagentRunModel {
    /// Profile name of the model that ran.
    name: String,
    /// Whether the model was chosen or routed.
    source: ModelSource,
}

#[derive(Debug, Clone, Serialize)]
struct SubagentResult {
    subagent: String,
    status: &'static str,
    result: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<SubagentRunModel>,
}

impl SubagentResult {
    fn cancelled(subagent: String) -> Self {
        Self {
            subagent,
            status: "cancelled",
            result: "Cancelled by user".to_string(),
            model: None,
        }
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
    /// Factory for creating child terminal tools for subagents.
    terminal_tool_factory: Option<TerminalToolFactory>,
    /// Registry the tool reports in-flight runs to; clone it before
    /// handing the tool to a builder to observe or cancel runs.
    runs: SubagentRuns,
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
            terminal_tool_factory: None,
            runs: SubagentRuns::default(),
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

    /// Sets the terminal tool factory used for subagents.
    #[must_use]
    pub fn with_terminal_tool_factory(mut self, factory: TerminalToolFactory) -> Self {
        self.terminal_tool_factory = Some(factory);
        self
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
        if !self.mounts.is_empty() {
            return self.mounts.clone();
        }

        let base = self
            .base_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(Path::new(".")));
        vec![
            SubagentFileMount::new(PathBuf::from("."), base.clone()),
            SubagentFileMount::new(PathBuf::from("subagents"), base.join("subagents")),
            SubagentFileMount::new(PathBuf::from("skills"), base.join("skills")),
        ]
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

    /// Resolves `args.subagent` into the source a call builds its agent from.
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

impl<LLM: LanguageModel + Clone + Sync> SubagentTool<LLM> {
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

    /// Picks the model a subagent call runs on.
    ///
    /// A `chosen` tier — from `SubagentArgs::model` or the definition —
    /// is used as given. Without one, [`route_model`](Self::route_model)
    /// asks the fast model which configured tier fits the prompt.
    async fn select_model(
        &self,
        prompt: &str,
        chosen: Option<ModelTier>,
    ) -> anyhow::Result<(ModelTier, SubagentRunModel)> {
        if let Some(tier) = chosen {
            let profile = self.model_for(tier).profile().await.map_err(|error| {
                anyhow::anyhow!(
                    "failed to fetch the {} model profile: {error}",
                    tier_label(tier)
                )
            })?;
            return Ok((
                tier,
                SubagentRunModel {
                    name: profile.name,
                    source: ModelSource::Chosen,
                },
            ));
        }

        let (tier, name) = self.route_model(prompt).await?;
        Ok((
            tier,
            SubagentRunModel {
                name,
                source: ModelSource::Routed,
            },
        ))
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
        self
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
        let (subagent_id, declared_model, source) = self.resolve_source(&args.subagent).await?;

        // The task prompt comes from a file; inline prompt text is rejected.
        let prompt = self.read_prompt(&args.prompt_file).await?;

        // A model given by the caller or the definition runs as given;
        // otherwise the fast model routes the prompt to a configured tier.
        let (tier, model) = self
            .select_model(&prompt, args.model.or(declared_model))
            .await?;
        let selected = self.model_for(tier).clone();

        let agent_builder = match source {
            Source::File(def) => crate::AgentBuilder::new(selected)
                .agent_kind(crate::AgentKind::Chatbot)
                .system_prompt(&def.system_prompt)
                .max_iterations(def.max_iterations),
            Source::Registered(subagent_type) => subagent_type.builder(selected),
        };

        tracing::info!(subagent = %subagent_id, model = %model.name.as_str(), "Starting subagent");

        // Add child terminal tool if factory is configured
        let (agent_builder, child_job_registry) = if let Some(factory) = &self.terminal_tool_factory
        {
            let dyn_terminal = factory
                .create()
                .await
                .map_err(|e| anyhow::anyhow!("failed to create subagent terminal tool: {e}"))?;
            let job_registry = dyn_terminal.job_registry();
            (agent_builder.dyn_terminal(dyn_terminal), Some(job_registry))
        } else {
            (agent_builder, None)
        };

        let mut agent = agent_builder.build();

        let (run_id, cancel_rx) = self.runs.begin(&subagent_id, &prompt);
        let outcome = await_subagent_run(agent.query(&prompt), cancel_rx).await;
        self.runs.finish(run_id);
        if let Some(job_registry) = child_job_registry {
            let killed = job_registry.kill_running().await;
            if killed > 0 {
                tracing::info!(
                    subagent = %subagent_id,
                    killed,
                    "Stopped background terminal jobs left by subagent"
                );
            }
        }
        let result = match outcome {
            SubagentOutcome::Cancelled => {
                tracing::info!(subagent = %subagent_id, "Subagent cancelled");
                return ToolResult::json(&SubagentResult::cancelled(subagent_id));
            }
            SubagentOutcome::Finished(Ok(result)) => result,
            SubagentOutcome::Finished(Err(error)) => {
                let history = agent.history();
                let recent = history
                    .into_iter()
                    .rev()
                    .take(12)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                let recent_json = serde_json::to_string(&recent)
                    .unwrap_or_else(|serialize_error| serialize_error.to_string());
                return Err(anyhow::anyhow!(
                    "Subagent '{subagent_id}' error: {error}\nRecent subagent history: {recent_json}"
                ));
            }
        };

        tracing::info!(subagent = %subagent_id, "Subagent completed");

        ToolResult::json(&SubagentResult {
            subagent: subagent_id,
            status: "completed",
            result,
            model: Some(model),
        })
    }
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
        let props = value.get("properties").expect("should have properties");
        assert!(props.get("subagent").is_some());
        assert!(props.get("prompt_file").is_some());
        assert!(props.get("model").is_some());

        // `model` stays optional; the caller may leave routing to the tool.
        let required = value
            .get("required")
            .expect("should have required")
            .as_array()
            .expect("required should be an array");
        assert!(required.contains(&serde_json::json!("subagent")));
        assert!(required.contains(&serde_json::json!("prompt_file")));
        assert!(!required.contains(&serde_json::json!("model")));
    }

    #[test]
    fn subagent_result_serializes_as_structured_json() {
        let value = serde_json::to_value(SubagentResult {
            subagent: "research".to_string(),
            status: "completed",
            result: "done".to_string(),
            model: Some(SubagentRunModel {
                name: "test-model".to_string(),
                source: ModelSource::Routed,
            }),
        })
        .expect("subagent result should serialize");
        assert_eq!(
            value,
            serde_json::json!({
                "subagent": "research",
                "status": "completed",
                "result": "done",
                "model": {
                    "name": "test-model",
                    "source": "routed",
                },
            })
        );

        let cancelled = serde_json::to_value(SubagentResult::cancelled("explore".to_string()))
            .expect("cancelled result should serialize");
        assert_eq!(
            cancelled,
            serde_json::json!({
                "subagent": "explore",
                "status": "cancelled",
                "result": "Cancelled by user",
            })
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
                SubagentArgs {
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
        let ToolResult::Json { value } = result else {
            panic!("expected json result, got {result:?}");
        };
        assert_eq!(value["status"], "cancelled");
        assert_eq!(value["subagent"], "explore");
        assert!(runs.list().is_empty(), "finished run must unregister");
        assert!(!runs.cancel(run.id), "finished run cannot be cancelled");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn inline_prompt_text_is_rejected() {
        let tool = worker_tool(SubagentTool::new(TextLlm::new("only-model")));

        let error = Tool::call(
            &tool,
            SubagentArgs {
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
            SubagentArgs {
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
            SubagentArgs {
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: Some(ModelTier::Fast),
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let ToolResult::Json { value } = result else {
            panic!("expected json result, got {result:?}");
        };
        assert_eq!(value["status"], "completed");
        assert_eq!(value["result"], "fast-model");
        assert_eq!(value["model"]["name"], "fast-model");
        assert_eq!(value["model"]["source"], "chosen");

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
            SubagentArgs {
                subagent: "worker".to_string(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: None,
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let ToolResult::Json { value } = result else {
            panic!("expected json result, got {result:?}");
        };
        assert_eq!(value["status"], "completed");
        assert_eq!(value["result"], "balanced-model");
        assert_eq!(value["model"]["name"], "balanced-model");
        assert_eq!(value["model"]["source"], "routed");

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
            SubagentArgs {
                subagent: def_path.to_string_lossy().into_owned(),
                prompt_file: prompt_file.to_string_lossy().into_owned(),
                model: None,
            },
            ToolContext::new(),
        )
        .await
        .expect("subagent call failed");
        let ToolResult::Json { value } = result else {
            panic!("expected json result, got {result:?}");
        };
        assert_eq!(value["status"], "completed");
        assert_eq!(value["result"], "fast-model");
        assert_eq!(value["model"]["name"], "fast-model");
        assert_eq!(value["model"]["source"], "chosen");

        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(root);
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
}
