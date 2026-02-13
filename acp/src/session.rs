//! ACP session management.
//!
//! Each session represents an active conversation with the agent.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use uuid::Uuid;

use crate::protocol::{McpServerSpec, SessionUpdate};

/// An active ACP session.
///
/// Each session wraps a conversation context and can process prompts.
#[derive(Debug)]
pub struct AcpSession {
    id: String,
    cwd: PathBuf,
    mcp_servers: Vec<McpServerSpec>,
    cancelled: Arc<AtomicBool>,
}

impl AcpSession {
    /// Create a new session.
    #[must_use]
    pub fn new(cwd: PathBuf, mcp_servers: Vec<McpServerSpec>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cwd,
            mcp_servers,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get the session ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the working directory.
    #[must_use]
    pub const fn cwd(&self) -> &PathBuf {
        &self.cwd
    }

    /// Get MCP server specifications.
    #[must_use]
    pub fn mcp_servers(&self) -> &[McpServerSpec] {
        &self.mcp_servers
    }

    /// Check if the session is cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Cancel the current operation.
    pub fn stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Reset cancellation flag.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }

    /// Process a prompt and emit updates via the callback.
    ///
    /// This is a placeholder implementation. The full implementation
    /// will integrate with the aither agent to process prompts and
    /// stream events as ACP session updates.
    pub fn prompt<F>(&mut self, prompt: &str, mut on_update: F) -> Result<(), String>
    where
        F: FnMut(SessionUpdate),
    {
        // Reset cancellation flag
        self.reset();

        // Placeholder: Echo back the prompt as an agent message
        // In the real implementation, this will run the agent loop
        use crate::protocol::{ContentBlock, ContentChunk, TextContent};

        // Send agent message chunk
        on_update(SessionUpdate::AgentMessageChunk(ContentChunk {
            content: ContentBlock::Text(TextContent {
                text: format!("Received prompt in {}: {}", self.cwd.display(), prompt),
                annotations: None,
            }),
        }));

        Ok(())
    }
}
