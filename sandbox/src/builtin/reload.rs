//! The `reload` command for loading file content back into context.
//!
//! This command returns a special marker that the agent loop interprets
//! to inject file content into the context window.
//!
//! Only supports:
//! - Images (any size)
//! - Small text files (< 500 lines)
//!
//! Usage:
//! ```bash
//! reload outputs/abc123.txt
//! reload outputs/image.png
//! ```

use leash::IpcCommand;
use serde::{Deserialize, Serialize};

/// The reload command - requests to load file content into context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReloadCommand {
    /// The URL of the file to reload (relative path like "outputs/abc123.txt").
    pub url: String,
}

/// Response from the reload command.
///
/// This is a special marker that the agent loop interprets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReloadResponse {
    /// Action marker for the agent to interpret.
    pub action: String,
    /// The URL to reload.
    pub url: String,
}

impl ReloadCommand {
    /// Creates a new reload command.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }
}

impl Default for ReloadCommand {
    fn default() -> Self {
        Self { url: String::new() }
    }
}

impl ReloadResponse {
    /// Creates a new reload response.
    #[must_use]
    pub fn new(url: String) -> Self {
        Self {
            action: "reload".to_string(),
            url,
        }
    }
}

impl IpcCommand for ReloadCommand {
    type Response = ReloadResponse;

    fn name(&self) -> String {
        "reload".to_string()
    }

    async fn handle(&mut self) -> ReloadResponse {
        // Return the marker response - agent loop will interpret this
        ReloadResponse::new(self.url.clone())
    }
}

impl ReloadResponse {
    /// Checks if a string is a reload response and extracts the URL.
    ///
    /// The agent loop uses this to detect reload markers and inject content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use aither_sandbox::builtin::ReloadResponse;
    ///
    /// let json = r#"{"action":"reload","url":"outputs/abc123.txt"}"#;
    /// if let Some(url) = ReloadResponse::parse_reload_url(json) {
    ///     // Load content from url and inject into context
    ///     println!("Reload requested for: {}", url);
    /// }
    /// ```
    #[must_use]
    pub fn parse_reload_url(s: &str) -> Option<String> {
        let parsed: Result<Self, _> = serde_json::from_str(s);
        match parsed {
            Ok(resp) if resp.action == "reload" => Some(resp.url),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reload_command() {
        let mut cmd = ReloadCommand::new("outputs/abc123.txt");
        let response = cmd.handle().await;
        assert_eq!(response.action, "reload");
        assert_eq!(response.url, "outputs/abc123.txt");
    }

    #[test]
    fn test_parse_reload_url() {
        let json = r#"{"action":"reload","url":"outputs/test.txt"}"#;
        assert_eq!(
            ReloadResponse::parse_reload_url(json),
            Some("outputs/test.txt".to_string())
        );

        // Non-reload response
        let other = r#"{"message":"hello"}"#;
        assert_eq!(ReloadResponse::parse_reload_url(other), None);

        // Invalid action
        let wrong_action = r#"{"action":"other","url":"test.txt"}"#;
        assert_eq!(ReloadResponse::parse_reload_url(wrong_action), None);
    }
}
