//! Browser backend abstraction for interactive browser automation.

/// Error returned by browser backend operations.
#[derive(Debug, Clone)]
pub struct BrowserBackendError {
    message: String,
}

impl BrowserBackendError {
    /// Create a new browser backend error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for BrowserBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "browser backend error: {}", self.message)
    }
}

impl std::error::Error for BrowserBackendError {}

/// Backend abstraction for browser automation runtimes.
///
/// Implementations may connect to local Chromium, remote CDP endpoints, or
/// cloud browser providers.
pub trait BrowserBackend: Send + Sync {
    /// Returns a CDP endpoint URL for Playwright/browser clients.
    fn cdp_endpoint(&self) -> impl Future<Output = Result<String, BrowserBackendError>> + Send;
}

/// Fixed CDP backend backed by a static endpoint string.
#[derive(Debug, Clone)]
pub struct StaticCdpBrowser {
    endpoint: String,
}

impl StaticCdpBrowser {
    /// Create a static CDP backend.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }
}

impl BrowserBackend for StaticCdpBrowser {
    fn cdp_endpoint(&self) -> impl Future<Output = Result<String, BrowserBackendError>> {
        std::future::ready(Ok(self.endpoint.clone()))
    }
}
