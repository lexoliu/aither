//! Permission handling for terminal execution modes.
//!
//! Different terminal modes have different permission requirements:
//! - `Sandboxed`: No approval needed; outbound network is available by
//!   default, filtered per-domain via [`PermissionHandler::check_domain`]
//!   and audited by higher layers
//! - `Unsafe`: Per-script approval required

use std::future::Future;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Permission mode for terminal execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TerminalMode {
    /// Sandboxed execution without extra host privileges.
    ///
    /// Outbound network is available by default, filtered per-domain through
    /// the configured network policy and always audited.
    #[default]
    Sandboxed,

    /// Unsafe execution without sandbox.
    /// Full system access. Requires per-script approval.
    Unsafe,
}

impl TerminalMode {
    /// Returns whether this mode requires user approval.
    #[must_use]
    pub const fn requires_approval(self) -> bool {
        matches!(self, Self::Unsafe)
    }

    /// Returns whether approval is needed per-script (vs first-use only).
    #[must_use]
    pub const fn requires_per_script_approval(self) -> bool {
        matches!(self, Self::Unsafe)
    }

    /// Returns a human-readable description of this mode.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Sandboxed => "sandboxed",
            Self::Unsafe => "unsafe (no sandbox, full access)",
        }
    }
}

/// A request for a permission beyond the sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PermissionRequest<'a> {
    /// A command that needs a mode beyond the sandbox.
    Command {
        /// The mode the command needs.
        mode: TerminalMode,
        /// The script to run.
        script: &'a str,
    },
    /// A sandboxed command's connection to a network domain that is not
    /// already allowed.
    Domain {
        /// The host being connected to.
        domain: &'a str,
        /// The port being connected to.
        port: u16,
    },
}

/// How [`PermissionHandler::review`] decided a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Review {
    /// Granted; nothing else is asked.
    Approved,
    /// Refused, with the reason the model reads.
    Denied(String),
    /// Not decided; the request goes on to the handler's own route.
    Passed,
}

/// Trait for handling permission requests.
///
/// Implementors decide whether to allow terminal executions based on mode and script.
pub trait PermissionHandler: Send + Sync {
    /// Checks if the given mode and script are allowed.
    ///
    /// For `Sandboxed` mode, this should always return `Ok(true)`.
    /// For `Unsafe` mode, this should prompt for each script.
    fn check(
        &self,
        mode: TerminalMode,
        script: &str,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send;

    /// Returns whether `check` will suspend waiting for an external approval.
    ///
    /// This is used by higher-level runtimes to surface pause/resume lifecycle
    /// events only when a permission request will actually block.
    fn will_wait_for_approval(
        &self,
        mode: TerminalMode,
        script: &str,
    ) -> impl Future<Output = bool> + Send {
        let _ = (mode, script);
        async { false }
    }

    /// Decides `request` before the handler's own route is asked.
    ///
    /// The terminal calls this first for every command that needs a mode
    /// beyond the sandbox and for every domain for which
    /// [`domain_needs_approval`](Self::domain_needs_approval) holds. Only a
    /// request this passes reaches [`will_wait_for_approval`](Self::will_wait_for_approval)
    /// and [`check`](Self::check) or [`check_domain`](Self::check_domain), so
    /// a request decided here never reports waiting on anyone.
    ///
    /// The default passes every request.
    fn review(&self, request: &PermissionRequest<'_>) -> impl Future<Output = Review> + Send {
        let _ = request;
        async { Review::Passed }
    }

    /// The handler a subagent's terminal uses, when it must differ from the
    /// one its launcher uses (for example because it labels who is asking).
    ///
    /// `None`, the default, means the subagent shares this handler, so its
    /// requests reach exactly the route the launcher's do.
    fn for_subagent(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// Checks if a network domain is allowed.
    ///
    /// Called for each network connection attempt in `Sandboxed` mode.
    /// The implementation may prompt the user, check against a whitelist, etc.
    ///
    /// Default implementation denies all domains (fail-safe).
    fn check_domain(&self, domain: &str, port: u16) -> impl Future<Output = bool> + Send {
        let _ = (domain, port);
        async { false }
    }

    /// Whether [`check_domain`](Self::check_domain) would have to ask
    /// someone about this domain, as opposed to deciding it silently (a
    /// default allow-list, a domain already approved, a fixed policy).
    ///
    /// Only such a domain is put to [`review`](Self::review). The default
    /// is `false`.
    fn domain_needs_approval(&self, domain: &str, port: u16) -> impl Future<Output = bool> + Send {
        let _ = (domain, port);
        async { false }
    }

    /// Records that `domain` was approved by a reviewer, the way the handler
    /// records a domain its user approved, so that
    /// [`domain_needs_approval`](Self::domain_needs_approval) no longer holds
    /// for it. A handler whose `domain_needs_approval` can hold implements
    /// this; the default does nothing.
    fn remember_domain(&self, domain: &str, port: u16) -> impl Future<Output = ()> + Send {
        let _ = (domain, port);
        async {}
    }

    /// The decision for a connection to `domain`: the review step first, for
    /// a domain that needs approval, then [`check_domain`](Self::check_domain).
    ///
    /// The terminal's network policy calls this. A handler wrapping another
    /// forwards the individual methods and leaves this one provided.
    fn authorize_domain(&self, domain: &str, port: u16) -> impl Future<Output = bool> + Send {
        async move {
            if self.domain_needs_approval(domain, port).await {
                match self
                    .review(&PermissionRequest::Domain { domain, port })
                    .await
                {
                    Review::Approved => {
                        self.remember_domain(domain, port).await;
                        return true;
                    }
                    Review::Denied(reason) => {
                        tracing::info!(domain, port, %reason, "network domain denied by review");
                        return false;
                    }
                    Review::Passed => {}
                }
            }
            self.check_domain(domain, port).await
        }
    }
}

/// Error type for permission operations.
#[derive(Debug, thiserror::Error)]
pub enum PermissionError {
    /// Permission was denied by the user.
    #[error("permission denied: {0}")]
    Denied(String),

    /// Permission check was interrupted.
    #[error("permission check interrupted")]
    Interrupted,

    /// Internal error during permission check.
    #[error("permission error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// A simple permission handler that allows all sandboxed operations
/// and denies everything else.
#[derive(Debug, Clone, Copy, Default)]
pub struct DenyUnsafe;

impl PermissionHandler for DenyUnsafe {
    fn check(
        &self,
        mode: TerminalMode,
        _script: &str,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send {
        std::future::ready(match mode {
            TerminalMode::Sandboxed => Ok(true),
            TerminalMode::Unsafe => Err(PermissionError::Denied(format!(
                "sandbox blocked {} execution; approval is required to escalate this command, but no interactive approval handler is configured",
                mode.description()
            ))),
        })
    }

    fn will_wait_for_approval(
        &self,
        _mode: TerminalMode,
        _script: &str,
    ) -> impl Future<Output = bool> + Send {
        std::future::ready(false)
    }
}

/// A no-op permission handler that allows every execution mode and domain.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopPermissionHandler;

impl PermissionHandler for NoopPermissionHandler {
    fn check(
        &self,
        _mode: TerminalMode,
        _script: &str,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send {
        std::future::ready(Ok(true))
    }

    fn will_wait_for_approval(
        &self,
        _mode: TerminalMode,
        _script: &str,
    ) -> impl Future<Output = bool> + Send {
        std::future::ready(false)
    }

    fn check_domain(&self, _domain: &str, _port: u16) -> impl Future<Output = bool> + Send {
        std::future::ready(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deny_unsafe() {
        let handler = DenyUnsafe;

        // Sandboxed should be allowed
        assert!(handler.check(TerminalMode::Sandboxed, "ls").await.unwrap());

        // Unsafe should be denied
        assert!(
            handler
                .check(TerminalMode::Unsafe, "rm -rf /")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_noop_permission_handler() {
        let handler = NoopPermissionHandler;

        assert!(handler.check(TerminalMode::Sandboxed, "ls").await.unwrap());
        assert!(
            handler
                .check(TerminalMode::Unsafe, "rm -rf /")
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_deny_unsafe_check_domain() {
        let handler = DenyUnsafe;

        // Default implementation denies all domains
        assert!(!handler.check_domain("example.com", 443).await);
        assert!(!handler.check_domain("api.github.com", 80).await);
    }

    #[tokio::test]
    async fn test_noop_permission_handler_check_domain() {
        let handler = NoopPermissionHandler;

        // NoopPermissionHandler allows all domains
        assert!(handler.check_domain("example.com", 443).await);
        assert!(handler.check_domain("malicious.com", 80).await);
    }
}
