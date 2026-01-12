//! Permission handling for bash execution modes.
//!
//! Different bash modes have different permission requirements:
//! - `Sandboxed`: No approval needed (read-only, no network)
//! - `Network`: First-use approval only
//! - `Unsafe`: Per-script approval required

use std::future::Future;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Permission mode for bash execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BashMode {
    /// Sandboxed execution with read-only filesystem and no network.
    /// IPC commands (websearch, webfetch) still work.
    /// No approval needed.
    #[default]
    Sandboxed,

    /// Sandboxed execution with network access enabled.
    /// Requires first-use approval.
    Network,

    /// Unsafe execution without sandbox.
    /// Full system access. Requires per-script approval.
    Unsafe,
}

impl BashMode {
    /// Returns whether this mode requires user approval.
    #[must_use]
    pub const fn requires_approval(self) -> bool {
        matches!(self, Self::Network | Self::Unsafe)
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
            Self::Sandboxed => "sandboxed (read-only, no network)",
            Self::Network => "network-enabled sandbox",
            Self::Unsafe => "unsafe (no sandbox, full access)",
        }
    }
}

/// Trait for handling permission requests.
///
/// Implementors decide whether to allow bash executions based on mode and script.
pub trait PermissionHandler: Send + Sync {
    /// Checks if the given mode and script are allowed.
    ///
    /// For `Sandboxed` mode, this should always return `Ok(true)`.
    /// For `Network` mode, this might return `Ok(true)` after first approval.
    /// For `Unsafe` mode, this should prompt for each script.
    fn check(
        &self,
        mode: BashMode,
        script: &str,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send;

    /// Checks if a network domain is allowed.
    ///
    /// Called for each network connection attempt in `Network` mode.
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
    async fn check(&self, mode: BashMode, _script: &str) -> Result<bool, PermissionError> {
        match mode {
            BashMode::Sandboxed => Ok(true),
            BashMode::Network | BashMode::Unsafe => Err(PermissionError::Denied(format!(
                "{} mode requires approval but no interactive handler is configured",
                mode.description()
            ))),
        }
    }
}

/// A permission handler that allows everything (for testing).
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAll;

impl PermissionHandler for AllowAll {
    async fn check(&self, _mode: BashMode, _script: &str) -> Result<bool, PermissionError> {
        Ok(true)
    }

    async fn check_domain(&self, _domain: &str, _port: u16) -> bool {
        true
    }
}

/// A permission handler that tracks network approval state.
#[derive(Debug, Default)]
pub struct StatefulPermissionHandler<Inner> {
    inner: Inner,
    network_approved: std::sync::atomic::AtomicBool,
}

impl<Inner> StatefulPermissionHandler<Inner> {
    /// Creates a new stateful handler wrapping the given inner handler.
    pub const fn new(inner: Inner) -> Self {
        Self {
            inner,
            network_approved: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Returns whether network mode has been approved.
    pub fn is_network_approved(&self) -> bool {
        self.network_approved
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Marks network mode as approved.
    pub fn approve_network(&self) {
        self.network_approved
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

impl<Inner: PermissionHandler> PermissionHandler for StatefulPermissionHandler<Inner> {
    async fn check(&self, mode: BashMode, script: &str) -> Result<bool, PermissionError> {
        match mode {
            BashMode::Sandboxed => Ok(true),
            BashMode::Network => {
                // Check if already approved
                if self.is_network_approved() {
                    return Ok(true);
                }
                // Ask inner handler
                let approved = self.inner.check(mode, script).await?;
                if approved {
                    self.approve_network();
                }
                Ok(approved)
            }
            BashMode::Unsafe => {
                // Always ask for unsafe
                self.inner.check(mode, script).await
            }
        }
    }

    async fn check_domain(&self, domain: &str, port: u16) -> bool {
        self.inner.check_domain(domain, port).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deny_unsafe() {
        let handler = DenyUnsafe;

        // Sandboxed should be allowed
        assert!(handler.check(BashMode::Sandboxed, "ls").await.unwrap());

        // Network should be denied
        assert!(handler.check(BashMode::Network, "curl").await.is_err());

        // Unsafe should be denied
        assert!(handler.check(BashMode::Unsafe, "rm -rf /").await.is_err());
    }

    #[tokio::test]
    async fn test_allow_all() {
        let handler = AllowAll;

        assert!(handler.check(BashMode::Sandboxed, "ls").await.unwrap());
        assert!(handler.check(BashMode::Network, "curl").await.unwrap());
        assert!(handler.check(BashMode::Unsafe, "rm -rf /").await.unwrap());
    }

    #[tokio::test]
    async fn test_stateful_handler() {
        let handler = StatefulPermissionHandler::new(AllowAll);

        // Network not yet approved
        assert!(!handler.is_network_approved());

        // First network check approves
        assert!(handler.check(BashMode::Network, "curl").await.unwrap());
        assert!(handler.is_network_approved());

        // Subsequent checks use cached approval
        assert!(handler.check(BashMode::Network, "wget").await.unwrap());
    }

    #[tokio::test]
    async fn test_deny_unsafe_check_domain() {
        let handler = DenyUnsafe;

        // Default implementation denies all domains
        assert!(!handler.check_domain("example.com", 443).await);
        assert!(!handler.check_domain("api.github.com", 80).await);
    }

    #[tokio::test]
    async fn test_allow_all_check_domain() {
        let handler = AllowAll;

        // AllowAll allows all domains
        assert!(handler.check_domain("example.com", 443).await);
        assert!(handler.check_domain("malicious.com", 80).await);
    }

    #[tokio::test]
    async fn test_stateful_handler_check_domain() {
        let handler = StatefulPermissionHandler::new(AllowAll);

        // Delegates to inner handler
        assert!(handler.check_domain("example.com", 443).await);
    }
}
