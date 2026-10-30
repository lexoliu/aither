//! Live hooks: the hook set a host's configuration and scripts keep current.
//!
//! [`LiveHooks`] owns the set of hooks every session of a host shares. The
//! set is an immutable snapshot behind an [`ArcSwap`]: a change builds the
//! next snapshot and swaps it in, and a running hook holds the snapshot it
//! started from, so no lock is held while a hook runs and a change applies to
//! every session at its next hook call.
//!
//! Every hook has a current version and, once it was changed, the previous
//! one. A version is the hook's command with its script read into the
//! command (see [`script`](super::script)), so a later edit of the file cannot
//! reach a version. Startup refuses a hook that does not pass its syntax
//! check; afterwards a change that does not pass it, and a command hook or
//! reviewer that fails to run, raise a [`HookWarning`] and leave the previous
//! version in use. When there is none, or it fails too, the user is asked to
//! fix the hook (see [`LiveHooks::run_hook`]).

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::Duration;

use aither_sandbox::PreparedTools;
use arc_swap::ArcSwap;
use askama::Template;
use executor_core::{AnyExecutor, Executor};
use futures::lock::{Mutex, MutexGuard};
use notify::Watcher;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::command::CommandHookConfig;
use super::lifecycle::LifecycleHookConfig;
use super::reviewer::ReviewerConfig;
use super::runner::{HookCommand, HookRunner};
use super::script::{self, HookCheckError, ScriptSnapshot};
use crate::ask_user::{AskUserArgs, AskUserBroker, AskUserRequestPayload};

/// How long after a file event the hooks wait for the burst of events one
/// save causes to settle before they are reloaded.
const SETTLE: Duration = Duration::from_millis(100);

/// How many warnings wait for the host to take them. Past it the oldest
/// unread warning is dropped for each new one; every warning is also written
/// to `tracing` when it is raised.
const WARNING_BOUND: usize = 64;

/// What the user is offered when a hook needs fixing.
const RETRY: &str = "Retry";

/// The three kinds of external hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookKind {
    /// Event-driven, adds context.
    Lifecycle,
    /// Gates every shell command.
    Command,
    /// Reviews permission requests.
    Reviewer,
}

impl std::fmt::Display for HookKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Lifecycle => "lifecycle",
            Self::Command => "command",
            Self::Reviewer => "reviewer",
        })
    }
}

/// Everything a host configures: the hooks of all three kinds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HooksConfig {
    /// Lifecycle hooks.
    pub lifecycle: Vec<LifecycleHookConfig>,
    /// Command hooks; each shell command is put to all of them.
    pub command: Vec<CommandHookConfig>,
    /// Reviewers, in the order they review.
    pub reviewers: Vec<ReviewerConfig>,
}

/// Why a hook configuration is refused.
#[derive(Debug, thiserror::Error)]
pub enum HooksConfigError {
    /// Two hooks of one kind share a name, which names a hook's versions.
    #[error("two {kind} hooks are named `{name}`")]
    DuplicateName {
        /// The kind of the hooks.
        kind: HookKind,
        /// The shared name.
        name: String,
    },
}

impl HooksConfig {
    fn validate(&self) -> Result<(), HooksConfigError> {
        fn unique<C: HookConfig>(configs: &[C]) -> Result<(), HooksConfigError> {
            let mut seen = HashSet::new();
            configs
                .iter()
                .find(|config| !seen.insert(config.name()))
                .map_or(Ok(()), |config| {
                    Err(HooksConfigError::DuplicateName {
                        kind: C::KIND,
                        name: config.name().to_string(),
                    })
                })
        }
        unique(&self.lifecycle)?;
        unique(&self.command)?;
        unique(&self.reviewers)
    }
}

/// Why hooks cannot start.
#[derive(Debug, thiserror::Error)]
pub enum LiveHooksError {
    /// A hook's script does not pass its startup check.
    #[error("the {kind} hook `{hook}` cannot start: {error}")]
    Hook {
        /// The kind of the hook.
        kind: HookKind,
        /// The hook's name.
        hook: String,
        /// What is wrong with its script.
        #[source]
        error: HookCheckError,
    },
    /// The configuration names a hook twice.
    #[error(transparent)]
    Config(#[from] HooksConfigError),
    /// The configuration file cannot be read.
    #[error("the hook configuration {path} cannot be read: {source}")]
    ConfigRead {
        /// The file.
        path: PathBuf,
        /// Why it cannot be read.
        #[source]
        source: std::io::Error,
    },
    /// The configuration file is not a hook configuration.
    #[error("the hook configuration {path} is not valid: {source}")]
    ConfigParse {
        /// The file.
        path: PathBuf,
        /// What is wrong with it.
        #[source]
        source: serde_json::Error,
    },
    /// Files cannot be watched.
    #[error("the hook files cannot be watched: {0}")]
    Watch(#[source] notify::Error),
}

/// Something a host should tell its user about the hooks.
///
/// Raised through [`LiveHooks::warnings`] and `tracing`. The model is not
/// told: it cannot fix a hook, and the user can. A refusal of one hook's
/// change is raised once, not again by every later reload that finds the same
/// fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookWarning {
    /// A change to a hook was not taken up because its script is broken.
    Rejected {
        /// The kind of the hook.
        kind: HookKind,
        /// The hook's name.
        hook: String,
        /// What is wrong with it.
        error: String,
        /// Whether the hook's previous version stays in use; otherwise the
        /// hook is not in use at all.
        kept_previous: bool,
    },
    /// A command hook or reviewer failed to run.
    RunFailed {
        /// The kind of the hook.
        kind: HookKind,
        /// The hook's name.
        hook: String,
        /// How it failed.
        failure: String,
        /// Whether its previous version took over.
        fell_back: bool,
        /// When no version could answer: whether the session pauses and asks
        /// the user to fix the hook; otherwise nobody can be asked and a
        /// command hook blocks its command, a reviewer passes its request on.
        asked: bool,
    },
    /// A changed configuration file was not taken up.
    ConfigInvalid {
        /// What is wrong with it.
        error: String,
    },
}

#[derive(Template)]
#[template(path = "hook_warning.txt", escape = "none")]
struct WarningTemplate<'a> {
    warning: &'a HookWarning,
}

impl std::fmt::Display for HookWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        WarningTemplate { warning: self }
            .render_into(f)
            .map_err(|_| std::fmt::Error)
    }
}

#[derive(Template)]
#[template(path = "hook_paused.txt", escape = "none")]
struct PausedTemplate<'a> {
    kind: HookKind,
    hook: &'a str,
    failure: &'a str,
}

/// A hook configuration of one kind.
pub(super) trait HookConfig: Clone + PartialEq + Send + Sync + 'static {
    const KIND: HookKind;
    fn name(&self) -> &str;
    fn command(&self) -> &HookCommand;
    fn slot(set: &HookSet) -> &Vec<Arc<Entry<Self>>>;
    fn slot_mut(set: &mut HookSet) -> &mut Vec<Arc<Entry<Self>>>;
}

macro_rules! hook_config {
    ($config:ty, $kind:ident, $field:ident) => {
        impl HookConfig for $config {
            const KIND: HookKind = HookKind::$kind;
            fn name(&self) -> &str {
                &self.name
            }
            fn command(&self) -> &HookCommand {
                &self.command
            }
            fn slot(set: &HookSet) -> &Vec<Arc<Entry<Self>>> {
                &set.$field
            }
            fn slot_mut(set: &mut HookSet) -> &mut Vec<Arc<Entry<Self>>> {
                &mut set.$field
            }
        }
    };
}

hook_config!(LifecycleHookConfig, Lifecycle, lifecycle);
hook_config!(CommandHookConfig, Command, command);
hook_config!(ReviewerConfig, Reviewer, reviewers);

/// One version of a hook: what runs.
#[derive(Debug)]
pub(super) struct Version {
    id: u64,
    /// The command to run; for a script, the shell running its text.
    pub command: HookCommand,
    script: Option<ScriptSnapshot>,
    /// Set when this version failed to run, so that a change landing before
    /// the fallback is recorded never keeps it as the previous version.
    failed: AtomicBool,
}

/// A configured hook and its versions.
#[derive(Debug)]
pub(super) struct Entry<C> {
    pub config: C,
    /// The version in use; none while the hook has never had a version that
    /// passed its check.
    pub current: Option<Arc<Version>>,
    /// The version before the last change, when there was one.
    previous: Option<Arc<Version>>,
    /// The version that failed to run and was replaced by `previous`; a
    /// reload that finds the same version again leaves it retired.
    retired: Option<Arc<Version>>,
    /// Why the newest text of the hook is not in use, while it is not.
    pub rejected: Option<String>,
}

impl<C> Entry<C> {
    /// Why the hook has no version in use.
    pub(super) fn unusable(&self) -> String {
        self.rejected
            .clone()
            .expect("a hook without a version in use has the reason it was refused")
    }
}

/// A snapshot of all hooks.
#[derive(Debug, Default)]
pub(super) struct HookSet {
    pub lifecycle: Vec<Arc<Entry<LifecycleHookConfig>>>,
    pub command: Vec<Arc<Entry<CommandHookConfig>>>,
    pub reviewers: Vec<Arc<Entry<ReviewerConfig>>>,
}

impl Clone for HookSet {
    fn clone(&self) -> Self {
        Self {
            lifecycle: self.lifecycle.clone(),
            command: self.command.clone(),
            reviewers: self.reviewers.clone(),
        }
    }
}

/// Where the user is asked to fix a failing hook.
///
/// Clones share the question that may be open: a session asks one question at
/// a time, whichever of its scopes (main agent or subagents) a failing hook
/// is met in, and the questions of other sessions do not wait for it.
#[derive(Debug, Clone)]
pub(super) struct UserPause {
    broker: AskUserBroker,
    session_id: String,
    open: Arc<Mutex<()>>,
}

impl UserPause {
    pub(super) fn new(broker: AskUserBroker, session_id: String) -> Self {
        Self {
            broker,
            session_id,
            open: Arc::new(Mutex::new(())),
        }
    }
}

/// How a hook run ended.
#[derive(Debug)]
pub(super) enum Run<R> {
    /// The hook answered.
    Answered(R),
    /// The hook was removed from the configuration while it was failing.
    Gone,
    /// The hook failed, no version could answer and nobody can be asked.
    Failed(String),
}

/// What is wrong with a hook, as far as a repeated warning is concerned: the
/// version that failed (none when no version passed its check) and the reason
/// the newest text was refused. A new version or new text is a new fault.
#[derive(Debug, PartialEq, Eq)]
struct Fault {
    version: Option<u64>,
    rejected: Option<String>,
}

#[derive(Debug)]
struct Inner {
    set: ArcSwap<HookSet>,
    /// The configuration the set is reconciled against, even where a hook of
    /// it was refused.
    desired: ArcSwap<HooksConfig>,
    config_file: Option<PathBuf>,
    runner: HookRunner,
    /// Runs what must not delay its caller; absent without hooks.
    executor: Option<AnyExecutor>,
    warnings: async_channel::Sender<HookWarning>,
    warning_inbox: async_channel::Receiver<HookWarning>,
    /// The fault last warned of for each hook that failed with nobody to ask,
    /// so a hook that stays broken is not warned of at every gated command.
    /// Never held across an await.
    unattended: StdMutex<HashMap<(HookKind, String), Fault>>,
    /// Serializes every change of the set. Never held while a hook runs.
    writer: Mutex<()>,
    versions: AtomicU64,
    /// Tells the watcher which files to follow. Nothing listens without
    /// hooks, where sending fails and is ignored.
    track: async_channel::Sender<Vec<PathBuf>>,
    /// Closes when the hooks are dropped, which ends the watcher.
    _shutdown: async_channel::Sender<()>,
}

/// The hooks of a host, followed live.
///
/// Clones share one set. See the [module](self) docs.
#[derive(Debug, Clone)]
pub struct LiveHooks {
    inner: Arc<Inner>,
}

impl LiveHooks {
    /// Starts `config`: every hook is syntax-checked, then its scripts are
    /// followed. `executor` runs the hooks of IPC calls in the background and
    /// the watcher; every hook runs with `tools`, the terminal's prepared
    /// baseline tools, on its `PATH` ahead of the host's.
    ///
    /// Changes reach the hooks through [`apply`](Self::apply) and through
    /// edits of the scripts.
    ///
    /// # Errors
    ///
    /// Refuses to start when a hook does not pass its syntax check or the
    /// configuration is invalid.
    pub async fn start(
        config: HooksConfig,
        tools: &PreparedTools<PathBuf>,
        executor: AnyExecutor,
    ) -> Result<Self, LiveHooksError> {
        Self::start_with(config, None, tools, executor).await
    }

    /// Starts the hooks of the JSON file at `path` and follows the file as
    /// well as the scripts: an edit of it applies at once, and a version of it
    /// that is not valid keeps the previous configuration.
    ///
    /// # Errors
    ///
    /// Refuses to start when the file cannot be read or is invalid, or when a
    /// hook does not pass its syntax check.
    pub async fn start_from_file(
        path: PathBuf,
        tools: &PreparedTools<PathBuf>,
        executor: AnyExecutor,
    ) -> Result<Self, LiveHooksError> {
        let config = read_config(&path).await?;
        Self::start_with(config, Some(path), tools, executor).await
    }

    async fn start_with(
        config: HooksConfig,
        config_file: Option<PathBuf>,
        tools: &PreparedTools<PathBuf>,
        executor: AnyExecutor,
    ) -> Result<Self, LiveHooksError> {
        config.validate()?;
        let (watching, events) = blocking::unblock(|| {
            let (event_tx, events) = async_channel::unbounded();
            let watcher = notify::recommended_watcher(move |event| {
                // The watcher outlives the receiver only while it shuts down.
                let _ = event_tx.try_send(event);
            })?;
            Ok((Watching::new(watcher), events))
        })
        .await
        .map_err(LiveHooksError::Watch)?;
        let (track, tracked) = async_channel::unbounded();
        let (shutdown, closed) = async_channel::bounded(1);
        let (warnings, warning_inbox) = async_channel::bounded(WARNING_BOUND);
        let inner = Arc::new(Inner {
            set: ArcSwap::from_pointee(HookSet::default()),
            desired: ArcSwap::from_pointee(config.clone()),
            config_file: config_file.clone(),
            runner: HookRunner::new(tools),
            executor: Some(executor),
            warnings,
            warning_inbox,
            unattended: StdMutex::default(),
            writer: Mutex::new(()),
            versions: AtomicU64::new(0),
            track,
            _shutdown: shutdown,
        });
        let set = inner.start_set(&config).await?;
        let mut paths = set.tracked.clone();
        paths.extend(config_file);
        inner.set.store(Arc::new(set.set));
        // Watching starts before the scripts are read once more, so that no
        // change falls between the startup check and the watch.
        let (watching, _) = watching.retarget(paths).await;
        let hooks = Self { inner };
        hooks.inner.reload().await;
        hooks
            .inner
            .executor
            .as_ref()
            .expect("started hooks have an executor")
            .spawn(follow(
                Arc::downgrade(&hooks.inner),
                watching,
                events,
                tracked,
                closed,
            ))
            .detach();
        Ok(hooks)
    }

    /// Hooks that do nothing, for an agent whose host configured none.
    #[must_use]
    pub fn none() -> Self {
        let (track, _) = async_channel::unbounded();
        let (shutdown, _) = async_channel::bounded(1);
        let (warnings, warning_inbox) = async_channel::bounded(WARNING_BOUND);
        Self {
            inner: Arc::new(Inner {
                set: ArcSwap::from_pointee(HookSet::default()),
                desired: ArcSwap::from_pointee(HooksConfig::default()),
                config_file: None,
                runner: HookRunner::without_hooks(),
                executor: None,
                warnings,
                warning_inbox,
                unattended: StdMutex::default(),
                writer: Mutex::new(()),
                versions: AtomicU64::new(0),
                track,
                _shutdown: shutdown,
            }),
        }
    }

    /// Replaces the configuration. A hook whose change does not pass its
    /// syntax check keeps its previous version and raises a warning; the rest
    /// of the configuration applies.
    ///
    /// # Errors
    ///
    /// Refuses a configuration that names two hooks of one kind alike, and
    /// keeps the current one.
    ///
    /// # Panics
    ///
    /// Panics on hooks made by [`none`](Self::none), which cannot run any.
    pub async fn apply(&self, config: HooksConfig) -> Result<(), HooksConfigError> {
        assert!(
            self.inner.runner.has_tools(),
            "hooks are applied to hooks made with `LiveHooks::start`"
        );
        config.validate()?;
        let writer = self.inner.writer.lock().await;
        self.inner.desired.store(Arc::new(config.clone()));
        self.inner.reconcile(&writer, &config).await;
        Ok(())
    }

    /// The warnings the hooks raise, for the host to show its user.
    ///
    /// Up to 64 unread warnings wait; past that the oldest is dropped for each
    /// new one (all of them are also logged through `tracing`), so a host that
    /// never reads them costs a bounded amount of memory. Clones of the
    /// receiver share the warnings: each is taken by one.
    #[must_use]
    pub fn warnings(&self) -> async_channel::Receiver<HookWarning> {
        self.inner.warning_inbox.clone()
    }

    pub(super) fn set(&self) -> Arc<HookSet> {
        self.inner.set.load_full()
    }

    pub(super) fn runner(&self) -> &HookRunner {
        &self.inner.runner
    }

    pub(super) fn executor(&self) -> Option<&AnyExecutor> {
        self.inner.executor.as_ref()
    }

    /// Runs `entry` with `input` and reads its answer with `interpret`.
    ///
    /// A version that fails to run (or whose answer `interpret` refuses, with
    /// the reason) raises a warning and hands the run to the hook's previous
    /// version, which then stays in use. A hook with no usable version, no
    /// previous version, or a previous version that fails too, asks `user` to
    /// fix the hook, one question at a time per session, and the run is
    /// retried with the hook as it is then, after each answer; a hook that
    /// stays broken asks again, so nothing spins. Without a `user` nobody can
    /// be asked and the run fails.
    pub(super) async fn run_hook<C, I, O, R>(
        &self,
        user: Option<&UserPause>,
        entry: &Arc<Entry<C>>,
        input: &I,
        interpret: impl Fn(Option<O>) -> Result<R, String> + Sync,
    ) -> Run<R>
    where
        C: HookConfig,
        I: Serialize + Sync,
        O: DeserializeOwned,
    {
        let name = entry.config.name();
        let attempt = async |version: &Version| -> Result<R, String> {
            let result = match self.inner.runner.run::<I, O>(&version.command, input).await {
                Ok(answer) => interpret(answer),
                Err(error) => Err(error.to_string()),
            };
            if result.is_err() {
                version.failed.store(true, Ordering::Relaxed);
            }
            result
        };
        let mut failure = match &entry.current {
            Some(current) => match attempt(current).await {
                Ok(answer) => return Run::Answered(answer),
                Err(failure) => failure,
            },
            None => entry.unusable(),
        };
        if let (Some(current), Some(previous)) = (&entry.current, &entry.previous) {
            match attempt(previous).await {
                Ok(answer) => {
                    self.inner.warn(HookWarning::RunFailed {
                        kind: C::KIND,
                        hook: name.to_string(),
                        failure,
                        fell_back: true,
                        asked: false,
                    });
                    self.retire::<C>(name, current.id).await;
                    return Run::Answered(answer);
                }
                Err(previous_failure) => failure = previous_failure,
            }
        }
        self.inner.warn_unanswered(entry, user.is_some(), &failure);
        let Some(user) = user else {
            return Run::Failed(failure);
        };
        let mut failed = entry.current.as_ref().map(|version| version.id);
        let mut fresh = true;
        loop {
            // A question that is already open covers this hook too: wait for
            // its answer, and try the hook as the user left it.
            let (_open, waited) = match user.open.try_lock() {
                Some(open) => (open, false),
                None => (user.open.lock().await, true),
            };
            let Some(latest) = C::slot(&self.set())
                .iter()
                .find(|latest| latest.config.name() == name)
                .cloned()
            else {
                return Run::Gone;
            };
            // The failure that opened this loop is not repeated before the
            // user was asked, unless the hook changed since.
            let latest_id = latest.current.as_ref().map(|version| version.id);
            if !(fresh && !waited && latest_id == failed) {
                failure = match &latest.current {
                    Some(current) => match attempt(current).await {
                        Ok(answer) => return Run::Answered(answer),
                        Err(latest_failure) => latest_failure,
                    },
                    None => latest.unusable(),
                };
                failed = latest_id;
            }
            fresh = false;
            let question = PausedTemplate {
                kind: C::KIND,
                hook: name,
                failure: &failure,
            }
            .render()
            .expect("hook paused template must render");
            let asked = user
                .broker
                .request(AskUserRequestPayload {
                    session_id: user.session_id.clone(),
                    request: AskUserArgs {
                        question: Some(question),
                        option: vec![RETRY.to_string()],
                        multi_select: false,
                        questions: Vec::new(),
                    },
                })
                .await;
            if let Err(error) = asked {
                tracing::warn!(hook = name, %error, "the user cannot be asked to fix a hook");
                return Run::Failed(failure);
            }
        }
    }

    /// Makes the previous version of a hook its current one, after the
    /// current one failed to run.
    async fn retire<C: HookConfig>(&self, name: &str, failed: u64) {
        let _writer = self.inner.writer.lock().await;
        let mut set = (*self.inner.set.load_full()).clone();
        let slot = C::slot_mut(&mut set);
        let Some(position) = slot.iter().position(|entry| {
            entry.config.name() == name
                && entry
                    .current
                    .as_ref()
                    .is_some_and(|current| current.id == failed)
        }) else {
            return;
        };
        let entry = &slot[position];
        let Some(previous) = &entry.previous else {
            return;
        };
        slot[position] = Arc::new(Entry {
            config: entry.config.clone(),
            current: Some(Arc::clone(previous)),
            previous: None,
            retired: entry.current.clone(),
            rejected: entry.rejected.clone(),
        });
        self.inner.set.store(Arc::new(set));
    }
}

/// The set a reconciliation built, with the script files it depends on.
struct Reconciled {
    set: HookSet,
    tracked: Vec<PathBuf>,
}

impl Inner {
    /// Warns that no version of the hook could answer. With nobody to ask, a
    /// hook that stays broken fails every gated command, so it is warned of
    /// once until its version or the reason its text was refused changes.
    fn warn_unanswered<C: HookConfig>(&self, entry: &Entry<C>, asked: bool, failure: &str) {
        if !asked {
            let fault = Fault {
                version: entry.current.as_ref().map(|version| version.id),
                rejected: entry.rejected.clone(),
            };
            let repeated = {
                let mut seen = self
                    .unattended
                    .lock()
                    .expect("the fault record is never held across a panic");
                let key = (C::KIND, entry.config.name().to_string());
                let repeated = seen.get(&key) == Some(&fault);
                seen.insert(key, fault);
                repeated
            };
            if repeated {
                return;
            }
        }
        self.warn(HookWarning::RunFailed {
            kind: C::KIND,
            hook: entry.config.name().to_string(),
            failure: failure.to_string(),
            fell_back: false,
            asked,
        });
    }

    fn warn(&self, warning: HookWarning) {
        tracing::warn!(%warning, "hook warning");
        // The inbox is held by this very value and never closes; a full one
        // gives up its oldest warning.
        self.warnings
            .force_send(warning)
            .expect("the warning inbox is open while the hooks exist");
    }

    fn next_version(&self, resolved: script::Resolved) -> Arc<Version> {
        Arc::new(Version {
            id: self.versions.fetch_add(1, Ordering::Relaxed),
            command: resolved.command,
            script: resolved.script,
            failed: AtomicBool::new(false),
        })
    }

    /// The first set of hooks; a hook that does not pass the check refuses it.
    async fn start_set(&self, config: &HooksConfig) -> Result<Reconciled, LiveHooksError> {
        async fn first<C: HookConfig>(
            inner: &Inner,
            configs: &[C],
        ) -> Result<(Vec<Arc<Entry<C>>>, Vec<PathBuf>), LiveHooksError> {
            let entries = futures::future::join_all(configs.iter().map(|config| async move {
                script::resolve(config.command(), None)
                    .await
                    .map(|resolved| {
                        let path = resolved.script.as_ref().map(|s| s.path.clone());
                        let entry = Arc::new(Entry {
                            config: config.clone(),
                            current: Some(inner.next_version(resolved)),
                            previous: None,
                            retired: None,
                            rejected: None,
                        });
                        (entry, path)
                    })
                    .map_err(|error| LiveHooksError::Hook {
                        kind: C::KIND,
                        hook: config.name().to_string(),
                        error,
                    })
            }))
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
            let (entries, paths): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
            Ok((entries, paths.into_iter().flatten().collect()))
        }
        let (lifecycle, command, reviewers) = futures::join!(
            first(self, &config.lifecycle),
            first(self, &config.command),
            first(self, &config.reviewers),
        );
        let (lifecycle, mut tracked) = lifecycle?;
        let (command, more) = command?;
        tracked.extend(more);
        let (reviewers, more) = reviewers?;
        tracked.extend(more);
        Ok(Reconciled {
            set: HookSet {
                lifecycle,
                command,
                reviewers,
            },
            tracked,
        })
    }

    /// Brings the set in line with `desired`. A hook whose new version does
    /// not pass the check keeps its old one. The writer lock is held by the
    /// caller.
    async fn reconcile(&self, _writer: &MutexGuard<'_, ()>, desired: &HooksConfig) {
        let old = self.set.load_full();
        let (lifecycle, command, reviewers) = futures::join!(
            self.reconcile_kind(&desired.lifecycle, &old.lifecycle),
            self.reconcile_kind(&desired.command, &old.command),
            self.reconcile_kind(&desired.reviewers, &old.reviewers),
        );
        let mut tracked = lifecycle.1;
        tracked.extend(command.1);
        tracked.extend(reviewers.1);
        tracked.extend(self.config_file.clone());
        self.set.store(Arc::new(HookSet {
            lifecycle: lifecycle.0,
            command: command.0,
            reviewers: reviewers.0,
        }));
        // Nothing listens without hooks.
        let _ = self.track.try_send(tracked);
    }

    async fn reconcile_kind<C: HookConfig>(
        &self,
        configs: &[C],
        old: &[Arc<Entry<C>>],
    ) -> (Vec<Arc<Entry<C>>>, Vec<PathBuf>) {
        let reconciled = futures::future::join_all(configs.iter().map(|config| async move {
            let old = old.iter().find(|old| old.config.name() == config.name());
            let known = old
                .and_then(|old| old.current.as_ref())
                .and_then(|current| current.script.as_ref());
            let resolved = match script::resolve(config.command(), known).await {
                Ok(resolved) => resolved,
                Err(error) => {
                    let reason = error.to_string();
                    let in_use = old.is_some_and(|old| old.current.is_some());
                    // The same fault is raised once, not at every reload.
                    if old.is_none_or(|old| old.rejected.as_ref() != Some(&reason)) {
                        self.warn(HookWarning::Rejected {
                            kind: C::KIND,
                            hook: config.name().to_string(),
                            error: reason.clone(),
                            kept_previous: in_use,
                        });
                    }
                    // The version in use stays, with the configuration it
                    // runs under; a hook with none has none to keep.
                    let entry = match old {
                        Some(old) if in_use => Entry {
                            config: old.config.clone(),
                            current: old.current.clone(),
                            previous: old.previous.clone(),
                            retired: old.retired.clone(),
                            rejected: Some(reason),
                        },
                        _ => Entry {
                            config: config.clone(),
                            current: None,
                            previous: None,
                            retired: None,
                            rejected: Some(reason),
                        },
                    };
                    return (Some(Arc::new(entry)), Some(error.path().to_path_buf()));
                }
            };
            let path = resolved.script.as_ref().map(|script| script.path.clone());
            let same_config = old.filter(|old| old.config == *config);
            let current = same_config.filter(|old| {
                old.current
                    .as_ref()
                    .is_some_and(|current| current.command == resolved.command)
            });
            let retired = same_config.filter(|old| {
                old.retired
                    .as_ref()
                    .is_some_and(|retired| retired.command == resolved.command)
            });
            let entry = match (retired, current) {
                (Some(retired), _) => Arc::clone(retired),
                (None, Some(current)) if current.rejected.is_none() => Arc::clone(current),
                // The text that was refused is back to what runs.
                (None, Some(current)) => Arc::new(Entry {
                    config: current.config.clone(),
                    current: current.current.clone(),
                    previous: current.previous.clone(),
                    retired: current.retired.clone(),
                    rejected: None,
                }),
                (None, None) => {
                    // A version that already failed to run is not the
                    // previous version of its successor: the one before it is.
                    let previous = old.and_then(|old| match &old.current {
                        Some(current) if !current.failed.load(Ordering::Relaxed) => {
                            Some(Arc::clone(current))
                        }
                        _ => old.previous.clone(),
                    });
                    Arc::new(Entry {
                        config: config.clone(),
                        current: Some(self.next_version(resolved)),
                        previous,
                        retired: None,
                        rejected: None,
                    })
                }
            };
            (Some(entry), path)
        }))
        .await;
        let mut entries = Vec::new();
        let mut paths = Vec::new();
        for (entry, path) in reconciled {
            entries.extend(entry);
            paths.extend(path);
        }
        (entries, paths)
    }

    /// Re-reads what the hooks follow and applies it.
    async fn reload(&self) {
        let writer = self.writer.lock().await;
        if let Some(file) = &self.config_file {
            match read_config(file).await {
                Ok(config) => self.desired.store(Arc::new(config)),
                Err(error) => self.warn(HookWarning::ConfigInvalid {
                    error: error.to_string(),
                }),
            }
        }
        let desired = self.desired.load_full();
        self.reconcile(&writer, &desired).await;
    }
}

async fn read_config(path: &Path) -> Result<HooksConfig, LiveHooksError> {
    let text = async_fs::read(path)
        .await
        .map_err(|source| LiveHooksError::ConfigRead {
            path: path.to_path_buf(),
            source,
        })?;
    let config: HooksConfig =
        serde_json::from_slice(&text).map_err(|source| LiveHooksError::ConfigParse {
            path: path.to_path_buf(),
            source,
        })?;
    config.validate()?;
    Ok(config)
}

/// What the watcher task wakes for.
enum Wake {
    Event(notify::Result<notify::Event>),
    Track(Vec<PathBuf>),
    Closed,
}

/// The files the watcher follows, and the watcher.
///
/// Watches the directories of the files, not the files, so that an editor
/// that saves by replacing a file is followed; events for other files are
/// ignored by name. A file reached through a symlink is followed in both
/// directories, the link's and its target's, and a directory that does not
/// exist yet is followed through its nearest existing ancestor until it does.
///
/// `notify` calls block (on macOS a watch restarts the event stream and joins
/// its thread), so every method that touches the watcher runs on a blocking
/// thread: the caller moves the value in and gets it back.
pub(super) struct Watching {
    watcher: notify::RecommendedWatcher,
    /// The directories the watcher accepted.
    pub(super) watched: HashSet<PathBuf>,
    /// Directories removed while watched. `notify` still holds them; the next
    /// retarget unwatches them, on a blocking thread, before any is watched
    /// afresh.
    pub(super) stale: HashSet<PathBuf>,
    /// The names an event must carry to concern a followed file.
    names: HashSet<OsString>,
}

impl Watching {
    pub(super) fn new(watcher: notify::RecommendedWatcher) -> Self {
        Self {
            watcher,
            watched: HashSet::new(),
            stale: HashSet::new(),
            names: HashSet::new(),
        }
    }

    /// Follows exactly `paths` from now on, on a blocking thread. Reports
    /// whether a directory was newly watched, in which case a change may have
    /// fallen between the last read of the files and the watch.
    async fn retarget(mut self, paths: Vec<PathBuf>) -> (Self, bool) {
        blocking::unblock(move || {
            let changed = self.follow_paths(&paths);
            (self, changed)
        })
        .await
    }

    /// Drops the watcher on a blocking thread.
    async fn stop(self) {
        blocking::unblock(move || drop(self)).await;
    }

    pub(super) fn follow_paths(&mut self, paths: &[PathBuf]) -> bool {
        let mut wanted = HashSet::new();
        self.names.clear();
        for path in paths {
            self.follow_one(path, &mut wanted);
            // The file may be a link: its target's directory is followed too.
            if let Ok(target) = std::fs::canonicalize(path)
                && target != *path
            {
                self.follow_one(&target, &mut wanted);
            }
        }
        let mut gone: Vec<PathBuf> = self.watched.difference(&wanted).cloned().collect();
        gone.extend(self.stale.drain());
        for directory in gone {
            if let Err(error) = self.watcher.unwatch(&directory) {
                tracing::warn!(directory = %directory.display(), %error, "a hook directory cannot be unwatched");
            }
            self.watched.remove(&directory);
        }
        let new: Vec<PathBuf> = wanted.difference(&self.watched).cloned().collect();
        let mut changed = false;
        for directory in new {
            // A directory the watcher refuses is not recorded, so the next
            // retarget tries it again.
            match self
                .watcher
                .watch(&directory, notify::RecursiveMode::NonRecursive)
            {
                Ok(()) => {
                    self.watched.insert(directory);
                    changed = true;
                }
                Err(error) => {
                    tracing::warn!(directory = %directory.display(), %error, "a hook directory cannot be watched");
                }
            }
        }
        changed
    }

    /// Wants the directory of `path`, or the nearest ancestor of it that
    /// exists, and the names that concern the file.
    fn follow_one(&mut self, path: &Path, wanted: &mut HashSet<PathBuf>) {
        self.names.extend(path.file_name().map(ToOwned::to_owned));
        let mut directory = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        while !directory.is_dir() {
            // The missing directory appearing is what concerns us.
            self.names
                .extend(directory.file_name().map(ToOwned::to_owned));
            let Some(parent) = directory.parent() else {
                break;
            };
            directory = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
        }
        wanted.insert(directory.to_path_buf());
    }

    /// Whether `event` concerns a followed file or removes a followed
    /// directory. A removed directory is forgotten, so the next retarget
    /// watches it again: a watch does not survive its directory being
    /// replaced on every platform.
    pub(super) fn concerns(&mut self, event: &notify::Event) -> bool {
        use notify::EventKind;
        use notify::event::ModifyKind;
        if event.kind.is_access() {
            return false;
        }
        let leaves = matches!(
            event.kind,
            EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(_))
        );
        let mut concerns = false;
        for path in &event.paths {
            concerns |= path
                .file_name()
                .is_some_and(|name| self.names.contains(name));
            if leaves && self.watched.remove(path) {
                self.stale.insert(path.clone());
                concerns = true;
            }
        }
        concerns
    }
}

/// Follows the files the hooks depend on and reloads them when one changes,
/// until the hooks are dropped.
async fn follow(
    inner: Weak<Inner>,
    mut watching: Watching,
    events: async_channel::Receiver<notify::Result<notify::Event>>,
    tracked: async_channel::Receiver<Vec<PathBuf>>,
    closed: async_channel::Receiver<()>,
) {
    use futures_lite::future::or;
    loop {
        let wake = or(
            or(
                async { events.recv().await.map_or(Wake::Closed, Wake::Event) },
                async { tracked.recv().await.map_or(Wake::Closed, Wake::Track) },
            ),
            async {
                // Closing is the only thing the channel ever does.
                let _ = closed.recv().await;
                Wake::Closed
            },
        )
        .await;
        let reload = match wake {
            Wake::Closed => break,
            Wake::Track(paths) => {
                let (followed, changed) = watching.retarget(paths).await;
                watching = followed;
                // Directories watched only now may have changed since the
                // files in them were read.
                changed
            }
            Wake::Event(Err(error)) => {
                tracing::warn!(%error, "the hook file watcher reported an error");
                false
            }
            Wake::Event(Ok(event)) => {
                let concerns = watching.concerns(&event);
                if concerns {
                    // One save is several events: let them pass, then reload once.
                    async_io::Timer::after(SETTLE).await;
                    while events.try_recv().is_ok() {}
                }
                concerns
            }
        };
        if reload {
            let Some(inner) = inner.upgrade() else {
                break;
            };
            inner.reload().await;
        }
    }
    watching.stop().await;
}
