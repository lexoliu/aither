//! Command policies: the embedder's rules on what a script may run.
//!
//! The sandbox decides where a script runs and what it can reach; whether a
//! script should run at all — a toolchain preference, a ban on killing by
//! name, a rule against polling — is the embedder's. Every
//! [`CommandPolicy`] registered on the [`ToolRegistryBuilder`] sees each
//! script before it runs, parsed when it is shell, with the call it came
//! from and the policy's own state for the agent that runs it, and either
//! lets it run or refuses it with the text the model reads. With no policy
//! the sandbox refuses nothing.
//!
//! [`ToolRegistryBuilder`]: crate::ToolRegistryBuilder

use std::fmt;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use aither_core::llm::tool::CallIdentity;

use crate::command::CommandRole;
use crate::script_language::ScriptLanguage;
use crate::shell_syntax::{ShellParseError, ShellScript};

/// How a script came to the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptOrigin {
    /// A `terminal` call the caller waits on.
    Foreground,
    /// A `terminal` call that detaches the script (`background: true`).
    Background,
    /// A [`TerminalRunner::run`](crate::TerminalRunner::run) from a
    /// terminal-bound command: the embedder's own script, waited on.
    Run,
}

/// A script about to run, as a [`CommandPolicy`] sees it.
#[derive(Debug, Clone, Copy)]
pub struct ScriptCheck<'a> {
    /// The id the script's task gets when it runs; the [`ScriptEnd`] that
    /// reports how it ended carries the same id.
    pub task_id: &'a str,
    /// The script as written.
    pub script: &'a str,
    /// The language it is written in.
    pub language: ScriptLanguage,
    /// The script parsed as shell: `None` for a script in another language,
    /// an error when it is not valid shell.
    pub shell: Option<Result<&'a ShellScript, &'a ShellParseError>>,
    /// How the script came to the terminal.
    pub origin: ScriptOrigin,
    /// The role of the agent whose terminal runs it.
    pub role: CommandRole,
    /// The directory the terminal's relative paths resolve against.
    pub working_dir: &'a Path,
    /// The tool call that carries the script and the agent that made it,
    /// when the caller identified one.
    pub call: Option<&'a CallIdentity>,
}

/// How a script a [`CommandPolicy`] was asked about ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptOutcome {
    /// It never started: a policy, the command gate or the permission path
    /// refused it, or its launch could not be prepared.
    NotRun,
    /// It started but could not be run to an exit status.
    Failed,
    /// It exited with `code`.
    Exited {
        /// The exit status.
        code: i32,
    },
}

/// The end of a script a [`CommandPolicy`] was asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptEnd<'a> {
    /// The task id the [`ScriptCheck`] carried.
    pub task_id: &'a str,
    /// How it ended.
    pub outcome: ScriptOutcome,
}

/// A script a [`CommandPolicy`] refused; the model reads [`Self::text`] as
/// the call's result.
///
/// Tools return it as their error, so an agent loop tells a policy's
/// refusal from any other failure by downcasting.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{text}")]
pub struct PolicyRefusal {
    text: String,
    unchanged_read: bool,
}

impl PolicyRefusal {
    /// A refusal the model reads as `text`.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            unchanged_read: false,
        }
    }

    /// A refusal of a script that would only re-read output that has not
    /// changed: it could teach the model nothing, so only a background event
    /// can. An agent loop whose every call in a turn is refused this way
    /// waits for one instead of asking the model again.
    #[must_use]
    pub fn unchanged_read(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            unchanged_read: true,
        }
    }

    /// The text the model reads.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether the refused script would only have re-read unchanged output.
    #[must_use]
    pub const fn is_unchanged_read(&self) -> bool {
        self.unchanged_read
    }
}

/// What a [`CommandPolicy`] decided about one script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// The script goes on to the command gate, the permission path and its
    /// run.
    Allow,
    /// The script does not run; the model reads the refusal in its place.
    Refuse(PolicyRefusal),
}

/// Decides whether a script may run, before it does.
///
/// Every script of the `terminal` tool, the main agent's and every
/// subagent's, and every [`TerminalRunner::run`](crate::TerminalRunner::run),
/// is put to every policy of its registry ahead of the command gate and the
/// permission path. The policies of one registry decide concurrently; the
/// first refusal in registration order is the one the model reads.
///
/// A policy keeps what it remembers in its [`State`](Self::State): every
/// agent's terminal, each subagent's included, starts with a fresh one and
/// shares it with nobody else. Calls run in parallel, so the state is shared
/// by reference and the policy chooses how to synchronise it.
pub trait CommandPolicy: Send + Sync + 'static {
    /// What the policy remembers for one agent.
    type State: Default + Send + Sync + 'static;

    /// Decides on `script`.
    fn check(
        &self,
        script: &ScriptCheck<'_>,
        state: &Self::State,
    ) -> impl Future<Output = PolicyDecision> + Send;

    /// Told once how every script put to [`Self::check`] ended, whichever
    /// policy decided what: refused, never started, failed or exited. The
    /// result reaches the caller only after this returns, so a state it
    /// records holds for the agent's next call.
    fn ended(&self, end: &ScriptEnd<'_>, state: &Self::State) -> impl Future<Output = ()> + Send {
        let _ = (end, state);
        std::future::ready(())
    }
}

/// Object-safe twin of [`CommandPolicy`]: the policy, before it is bound to
/// an agent.
trait CommandPolicyImpl: Send + Sync {
    fn bind(self: Arc<Self>) -> Arc<dyn BoundPolicyImpl>;
}

impl<T: CommandPolicy> CommandPolicyImpl for T {
    fn bind(self: Arc<Self>) -> Arc<dyn BoundPolicyImpl> {
        Arc::new(Bound {
            policy: self,
            state: T::State::default(),
        })
    }
}

/// A policy with one agent's state.
struct Bound<T: CommandPolicy> {
    policy: Arc<T>,
    state: T::State,
}

trait BoundPolicyImpl: Send + Sync {
    fn check<'a>(
        &'a self,
        script: &'a ScriptCheck<'a>,
    ) -> Pin<Box<dyn Future<Output = PolicyDecision> + Send + 'a>>;

    fn ended<'a>(&'a self, end: &'a ScriptEnd<'a>)
    -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

impl<T: CommandPolicy> BoundPolicyImpl for Bound<T> {
    fn check<'a>(
        &'a self,
        script: &'a ScriptCheck<'a>,
    ) -> Pin<Box<dyn Future<Output = PolicyDecision> + Send + 'a>> {
        Box::pin(self.policy.check(script, &self.state))
    }

    fn ended<'a>(
        &'a self,
        end: &'a ScriptEnd<'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(self.policy.ended(end, &self.state))
    }
}

/// A type-erased [`CommandPolicy`].
#[derive(Clone)]
pub struct AnyCommandPolicy(Arc<dyn CommandPolicyImpl>);

impl AnyCommandPolicy {
    /// Erases `policy`.
    #[must_use]
    pub fn new(policy: impl CommandPolicy) -> Self {
        Self(Arc::new(policy))
    }
}

impl fmt::Debug for AnyCommandPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyCommandPolicy")
    }
}

/// A registry's policies, each bound to the state of the agent the registry
/// serves.
#[derive(Clone, Default)]
pub struct CommandPolicies {
    policies: Arc<[AnyCommandPolicy]>,
    bound: Arc<[Arc<dyn BoundPolicyImpl>]>,
}

impl CommandPolicies {
    /// `policies`, bound to a fresh agent.
    pub fn new(policies: Vec<AnyCommandPolicy>) -> Self {
        let policies: Arc<[AnyCommandPolicy]> = policies.into();
        Self {
            bound: Self::bind(&policies),
            policies,
        }
    }

    fn bind(policies: &[AnyCommandPolicy]) -> Arc<[Arc<dyn BoundPolicyImpl>]> {
        policies
            .iter()
            .map(|policy| policy.0.clone().bind())
            .collect()
    }

    /// The same policies, bound to another agent with fresh state.
    #[must_use]
    pub fn for_agent(&self) -> Self {
        Self {
            policies: self.policies.clone(),
            bound: Self::bind(&self.policies),
        }
    }

    /// Whether no policy is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bound.is_empty()
    }

    /// Puts `script` to every policy; the first refusal in registration
    /// order wins.
    ///
    /// # Errors
    /// Returns the first refusal.
    pub async fn check(&self, script: &ScriptCheck<'_>) -> Result<(), PolicyRefusal> {
        let decisions =
            futures::future::join_all(self.bound.iter().map(|policy| policy.check(script))).await;
        decisions
            .into_iter()
            .find_map(|decision| match decision {
                PolicyDecision::Allow => None,
                PolicyDecision::Refuse(refusal) => Some(refusal),
            })
            .map_or(Ok(()), Err)
    }

    /// Tells every policy how a checked script ended.
    pub async fn ended(&self, task_id: &str, outcome: ScriptOutcome) {
        if self.is_empty() {
            return;
        }
        let end = ScriptEnd { task_id, outcome };
        futures::future::join_all(self.bound.iter().map(|policy| policy.ended(&end))).await;
    }
}

impl fmt::Debug for CommandPolicies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandPolicies")
            .field("policies", &self.policies.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Refuses every script after the first `allowed` its agent ran.
    struct Quota {
        allowed: usize,
    }

    impl CommandPolicy for Quota {
        type State = AtomicUsize;

        fn check(
            &self,
            _script: &ScriptCheck<'_>,
            state: &AtomicUsize,
        ) -> impl Future<Output = PolicyDecision> + Send {
            std::future::ready(if state.fetch_add(1, Ordering::SeqCst) < self.allowed {
                PolicyDecision::Allow
            } else {
                PolicyDecision::Refuse(PolicyRefusal::new("over quota"))
            })
        }
    }

    /// Refuses everything with its own text.
    struct Never(&'static str);

    impl CommandPolicy for Never {
        type State = ();

        fn check(
            &self,
            _script: &ScriptCheck<'_>,
            (): &(),
        ) -> impl Future<Output = PolicyDecision> + Send {
            std::future::ready(PolicyDecision::Refuse(PolicyRefusal::new(self.0)))
        }
    }

    fn check(task_id: &str) -> ScriptCheck<'_> {
        ScriptCheck {
            task_id,
            script: "true",
            language: ScriptLanguage::Shell,
            shell: None,
            origin: ScriptOrigin::Foreground,
            role: CommandRole::Main,
            working_dir: Path::new("/"),
            call: None,
        }
    }

    #[tokio::test]
    async fn each_agent_binds_fresh_state() {
        let main = CommandPolicies::new(vec![AnyCommandPolicy::new(Quota { allowed: 1 })]);
        let view = main.clone();
        let subagent = main.for_agent();
        assert_eq!(main.check(&check("a")).await, Ok(()));
        assert_eq!(
            main.check(&check("b")).await,
            Err(PolicyRefusal::new("over quota"))
        );
        assert!(
            view.check(&check("c")).await.is_err(),
            "a view of the main agent's registry shares its state"
        );
        assert_eq!(
            subagent.check(&check("d")).await,
            Ok(()),
            "the subagent's state is its own"
        );
    }

    #[tokio::test]
    async fn the_first_refusal_in_registration_order_is_read() {
        let policies = CommandPolicies::new(vec![
            AnyCommandPolicy::new(Quota { allowed: 1 }),
            AnyCommandPolicy::new(Never("first")),
            AnyCommandPolicy::new(Never("second")),
        ]);
        assert_eq!(
            policies.check(&check("a")).await,
            Err(PolicyRefusal::new("first"))
        );
        assert!(CommandPolicies::default().is_empty());
        assert_eq!(CommandPolicies::default().check(&check("b")).await, Ok(()));
    }
}
