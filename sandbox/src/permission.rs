//! Permission handling for bash execution modes.
//!
//! Different bash modes have different permission requirements:
//! - `Sandboxed`: No approval needed (read-only, no network)
//! - `Network`: Trusted domains auto-allowed, others require approval
//! - `Unsafe`: Per-script approval required

use std::collections::HashSet;
use std::future::Future;
use std::sync::RwLock;

use leash::DomainRequest;
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

    /// Checks if a network domain access is allowed.
    ///
    /// Called for each outbound network connection in network mode.
    /// Default implementation allows all (for backwards compatibility).
    fn check_domain(
        &self,
        request: &DomainRequest,
    ) -> impl Future<Output = bool> + Send {
        let _ = request;
        async { true }
    }
}

/// Default trusted domains for common package registries and services.
pub const TRUSTED_DOMAINS: &[&str] = &[
    // Package registries
    "*.npmjs.org",
    "*.npmjs.com",
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "*.pypi.org",
    "pypi.org",
    "files.pythonhosted.org",
    "*.crates.io",
    "crates.io",
    "static.crates.io",
    "*.rubygems.org",
    // Code hosting
    "*.github.com",
    "github.com",
    "*.githubusercontent.com",
    "*.gitlab.com",
    "gitlab.com",
    "*.bitbucket.org",
    // CDNs commonly used by package managers
    "*.cloudflare.com",
    "*.fastly.net",
    "*.akamaized.net",
    // Language-specific
    "*.golang.org",
    "proxy.golang.org",
    "*.rust-lang.org",
    "*.docs.rs",
    // Common development tools
    "*.docker.io",
    "*.docker.com",
    "auth.docker.io",
    "registry-1.docker.io",
];

/// Check if a domain matches a pattern (exact or wildcard).
fn domain_matches(domain: &str, pattern: &str) -> bool {
    if domain == pattern {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix("*.") {
        if domain.ends_with(suffix) && domain.len() > suffix.len() {
            return true;
        }
    }
    false
}

/// Check if a domain is in the trusted list.
pub fn is_trusted_domain(domain: &str) -> bool {
    TRUSTED_DOMAINS
        .iter()
        .any(|pattern| domain_matches(domain, pattern))
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
            BashMode::Network | BashMode::Unsafe => {
                Err(PermissionError::Denied(format!(
                    "{} mode requires approval but no interactive handler is configured",
                    mode.description()
                )))
            }
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
}

/// A permission handler that tracks approval state for network mode and domains.
///
/// - Network mode: requires first-use approval
/// - Trusted domains: auto-allowed without prompting
/// - Unknown domains: prompts user, caches approval
pub struct StatefulPermissionHandler<Inner> {
    inner: Inner,
    network_approved: std::sync::atomic::AtomicBool,
    /// Domains that have been approved by the user (not in trusted list)
    approved_domains: RwLock<HashSet<String>>,
}

impl<Inner: std::fmt::Debug> std::fmt::Debug for StatefulPermissionHandler<Inner> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatefulPermissionHandler")
            .field("inner", &self.inner)
            .field("network_approved", &self.network_approved)
            .field(
                "approved_domains",
                &self.approved_domains.read().unwrap().len(),
            )
            .finish()
    }
}

impl<Inner: Default> Default for StatefulPermissionHandler<Inner> {
    fn default() -> Self {
        Self::new(Inner::default())
    }
}

impl<Inner> StatefulPermissionHandler<Inner> {
    /// Creates a new stateful handler wrapping the given inner handler.
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            network_approved: std::sync::atomic::AtomicBool::new(false),
            approved_domains: RwLock::new(HashSet::new()),
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

    /// Check if a domain has been approved (either trusted or user-approved).
    fn is_domain_approved(&self, domain: &str) -> bool {
        // Check trusted list first
        if is_trusted_domain(domain) {
            return true;
        }
        // Check user-approved domains
        self.approved_domains
            .read()
            .unwrap()
            .contains(domain)
    }

    /// Mark a domain as approved.
    fn approve_domain(&self, domain: &str) {
        self.approved_domains
            .write()
            .unwrap()
            .insert(domain.to_string());
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

    async fn check_domain(&self, request: &DomainRequest) -> bool {
        let domain = request.target().to_string();

        // Check if already approved (trusted or user-approved)
        if self.is_domain_approved(&domain) {
            tracing::debug!(domain = %domain, "domain auto-allowed (trusted or cached)");
            return true;
        }

        // Ask inner handler for unknown domain
        tracing::info!(domain = %domain, "unknown domain, prompting user");
        let approved = self.inner.check_domain(request).await;
        if approved {
            self.approve_domain(&domain);
            tracing::info!(domain = %domain, "domain approved by user");
        } else {
            tracing::warn!(domain = %domain, "domain denied by user");
        }
        approved
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
}
