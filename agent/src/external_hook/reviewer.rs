//! Reviewers: external hooks on the permission path.

use std::sync::Arc;

use aither_sandbox::permission::{
    PermissionError, PermissionHandler, PermissionRequest, Review, TerminalMode,
};
use askama::Template;
use serde::{Deserialize, Serialize};

use super::lifecycle::ExternalHooks;
use super::live::Run;
use super::runner::{HookCommand, HookRole};

/// A reviewer as the host configures it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewerConfig {
    /// Names the reviewer in the denials it raises and in diagnostics.
    pub name: String,
    /// What to run and for how long.
    #[serde(flatten)]
    pub command: HookCommand,
}

/// What a reviewer reads on stdin.
///
/// The role of the agent that asked, and the request, which is either a command escalation (`"kind":"command"` with
/// `mode` and `script`) or a network domain (`"kind":"domain"` with `domain`
/// and `port`).
#[derive(Debug, Serialize)]
pub struct ReviewRequest<'a> {
    /// The agent whose request it is: the main agent or a subagent.
    pub role: HookRole,
    /// What is being asked for.
    #[serde(flatten)]
    pub request: PermissionRequest<'a>,
}

/// What a reviewer prints on stdout. Printing nothing is a pass. Any field
/// beyond the decision's own makes the answer invalid (the variants are
/// struct variants so that serde enforces this).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReviewAnswer {
    /// Grant the request without asking anyone else.
    Approve {},
    /// Refuse the request; the reason reaches the model.
    Deny {
        /// Why the request is refused.
        reason: String,
    },
    /// Leave the request to the next reviewer.
    Pass {},
}

#[derive(Template)]
#[template(path = "reviewer_denial.txt", escape = "none")]
struct ReviewerDenialTemplate<'a> {
    reviewer: &'a str,
    reason: &'a str,
}

impl ExternalHooks {
    /// Runs the reviewers in order until one approves or denies.
    ///
    /// A reviewer that fails to run, runs past its deadline or prints an
    /// invalid answer is replaced by its previous version when it has one;
    /// when no working version is left the user is asked to fix it. Where the
    /// user cannot be asked the failure is reported through `tracing` and
    /// counts as a pass, so a failure never grants anything.
    async fn review(&self, request: PermissionRequest<'_>) -> Review {
        let request = ReviewRequest {
            role: self.role(),
            request,
        };
        let set = self.live().set();
        for reviewer in &set.reviewers {
            let answer = self
                .live()
                .run_hook(
                    self.user(),
                    reviewer,
                    &request,
                    |answer: Option<ReviewAnswer>| Ok(answer.unwrap_or(ReviewAnswer::Pass {})),
                )
                .await;
            match answer {
                Run::Answered(ReviewAnswer::Approve {}) => return Review::Approved,
                Run::Answered(ReviewAnswer::Deny { reason }) => {
                    return Review::Denied(
                        ReviewerDenialTemplate {
                            reviewer: &reviewer.config.name,
                            reason: &reason,
                        }
                        .render()
                        .expect("reviewer denial template must render"),
                    );
                }
                Run::Answered(ReviewAnswer::Pass {}) | Run::Gone => {}
                Run::Failed(failure) => tracing::warn!(
                    reviewer = %reviewer.config.name,
                    %failure,
                    "permission reviewer failed; it counts as a pass"
                ),
            }
        }
        Review::Passed
    }
}

/// A permission handler with the host's reviewers in front of the existing
/// route.
///
/// The reviewers are the handler's [`review`](PermissionHandler::review) step,
/// which the terminal runs before it asks `inner` anything: the first
/// reviewer to approve grants the request without reaching `inner`, the first
/// to deny refuses it, and a request every reviewer passes goes to `inner`
/// unchanged, including its wait query. A command that needs a mode beyond
/// the sandbox and a network domain that `inner` would have to ask about are
/// both reviewed; a domain `inner` allows silently is not, and a domain a
/// reviewer approves is remembered by `inner` as if its user had approved it.
/// A subagent's terminal gets [`for_subagent`](PermissionHandler::for_subagent),
/// the same reviewers labelled with the subagent role, over the same `inner`
/// route.
///
/// The reviewers are those of the [`ExternalHooks`] scope the handler is made
/// with, so they follow the host's live changes like every other hook.
#[derive(Debug)]
pub struct ReviewedPermission<P> {
    inner: Arc<P>,
    hooks: ExternalHooks,
}

impl<P> ReviewedPermission<P> {
    /// The main agent's handler: the reviewers of `hooks` in front of
    /// `inner`.
    #[must_use]
    pub fn new(inner: P, hooks: ExternalHooks) -> Self {
        Self {
            inner: Arc::new(inner),
            hooks,
        }
    }
}

impl<P: PermissionHandler + 'static> PermissionHandler for ReviewedPermission<P> {
    /// Runs the chain; the first reviewer to decide ends it.
    async fn review(&self, request: &PermissionRequest<'_>) -> Review {
        self.hooks.review(*request).await
    }

    async fn check(&self, mode: TerminalMode, script: &str) -> Result<bool, PermissionError> {
        self.inner.check(mode, script).await
    }

    fn will_wait_for_approval(
        &self,
        mode: TerminalMode,
        script: &str,
    ) -> impl Future<Output = bool> + Send {
        self.inner.will_wait_for_approval(mode, script)
    }

    fn for_subagent(&self) -> Option<Self> {
        Some(Self {
            inner: Arc::clone(&self.inner),
            hooks: self.hooks.for_subagent(),
        })
    }

    fn check_domain(&self, domain: &str, port: u16) -> impl Future<Output = bool> + Send {
        self.inner.check_domain(domain, port)
    }

    fn domain_needs_approval(&self, domain: &str, port: u16) -> impl Future<Output = bool> + Send {
        self.inner.domain_needs_approval(domain, port)
    }

    fn remember_domain(&self, domain: &str, port: u16) -> impl Future<Output = ()> + Send {
        self.inner.remember_domain(domain, port)
    }
}
