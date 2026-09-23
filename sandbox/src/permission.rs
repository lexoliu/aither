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
