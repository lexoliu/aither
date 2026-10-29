//! Reviewers: external hooks on the permission path.

use std::sync::Arc;

use aither_sandbox::permission::{
    PermissionError, PermissionHandler, PermissionRequest, Review, TerminalMode,
};
use askama::Template;
use serde::{Deserialize, Serialize};

use super::runner::{HookCommand, HookRole, HookRunner};

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

/// An ordered chain of reviewers; the host's configuration of them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReviewerChain {
    reviewers: Vec<ReviewerConfig>,
}

impl ReviewerChain {
    /// A chain running `reviewers` in order.
    #[must_use]
    pub const fn new(reviewers: Vec<ReviewerConfig>) -> Self {
        Self { reviewers }
    }

    /// Runs the reviewers in order, through `runner`, until one approves or
    /// denies.
    ///
    /// A reviewer that fails to run, runs past its deadline or prints an
    /// invalid answer is reported through `tracing` and counts as a pass, so
    /// a failure never grants anything.
    async fn review(&self, runner: &HookRunner, request: &ReviewRequest<'_>) -> Review {
        for reviewer in &self.reviewers {
            match runner
                .run::<_, ReviewAnswer>(&reviewer.command, request)
                .await
            {
                Ok(Some(ReviewAnswer::Approve {})) => return Review::Approved,
                Ok(Some(ReviewAnswer::Deny { reason })) => {
                    return Review::Denied(
                        ReviewerDenialTemplate {
                            reviewer: &reviewer.name,
                            reason: &reason,
                        }
                        .render()
                        .expect("reviewer denial template must render"),
                    );
                }
                Ok(Some(ReviewAnswer::Pass {}) | None) => {}
                Err(error) => tracing::warn!(
                    reviewer = %reviewer.name,
                    %error,
                    "permission reviewer failed; it counts as a pass"
                ),
            }
        }
        Review::Passed
    }
}

/// A permission handler with the host's reviewer chain in front of the
/// existing route.
///
/// The chain is the handler's [`review`](PermissionHandler::review) step, which
/// the terminal runs before it asks `inner` anything: the first reviewer to
/// approve grants the request without reaching `inner`, the first to deny
/// refuses it, and a request every reviewer passes goes to `inner` unchanged,
/// including its wait query. A command that needs a mode beyond the sandbox
/// and a network domain that `inner` would have to ask about are both
/// reviewed; a domain `inner` allows silently is not, and a domain a reviewer
/// approves is remembered by `inner` as if its user had approved it. A subagent's terminal gets
/// [`for_subagent`](PermissionHandler::for_subagent), the same chain labelled
/// with the subagent role, over the same `inner` route and the same runner.
///
/// Reviewers run through a [`HookRunner`] like every other hook, so they find
/// the terminal's prepared baseline tools on their `PATH`.
#[derive(Debug)]
pub struct ReviewedPermission<P> {
    inner: Arc<P>,
    chain: Arc<ReviewerChain>,
    runner: Arc<HookRunner>,
    role: HookRole,
}

impl<P> ReviewedPermission<P> {
    /// The main agent's handler: `chain` in front of `inner`, its reviewers
    /// run by `runner`.
    #[must_use]
    pub fn new(inner: P, chain: ReviewerChain, runner: HookRunner) -> Self {
        Self {
            inner: Arc::new(inner),
            chain: Arc::new(chain),
            runner: Arc::new(runner),
            role: HookRole::Main,
        }
    }
}

impl<P: PermissionHandler + 'static> PermissionHandler for ReviewedPermission<P> {
    /// Runs the chain; the first reviewer to decide ends it.
    async fn review(&self, request: &PermissionRequest<'_>) -> Review {
        self.chain
            .review(
                &self.runner,
                &ReviewRequest {
                    role: self.role,
                    request: *request,
                },
            )
            .await
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
            chain: Arc::clone(&self.chain),
            runner: Arc::clone(&self.runner),
            role: HookRole::Subagent,
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
