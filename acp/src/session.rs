//! ACP session management.
//!
//! Each session represents an active conversation with the agent.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aither_agent::{Agent, AgentEvent};
use aither_core::LanguageModel;
use futures_lite::StreamExt;
use uuid::Uuid;

use crate::protocol::{
    ContentBlock, ContentChunk, McpServerSpec, SessionUpdate, TextContent, ToolCall,
    ToolCallContent, ToolCallStatus, ToolCallUpdate,
};

/// An active ACP session.
///
/// Each session wraps a conversation context and can process prompts.
#[derive(Debug)]
pub struct AcpSession<LLM: LanguageModel> {
    id: String,
    cwd: PathBuf,
    mcp_servers: Vec<McpServerSpec>,
    cancelled: Arc<AtomicBool>,
    agent: Agent<LLM, LLM, LLM>,
}

impl<LLM: LanguageModel> AcpSession<LLM> {
    /// Create a new session that drives the given agent.
    pub fn new(cwd: PathBuf, mcp_servers: Vec<McpServerSpec>, agent: Agent<LLM, LLM, LLM>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cwd,
            mcp_servers,
            cancelled: Arc::new(AtomicBool::new(false)),
            agent,
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

    /// Run the agent over `prompt`, reporting progress through `on_update`.
    ///
    /// Each agent event is translated into the corresponding ACP session update
    /// as it arrives, so the client sees text and tool activity while the turn
    /// is still running. Returns when the turn completes, the client cancels
    /// via [`Self::stop`], or the agent errors.
    ///
    /// # Errors
    ///
    /// Returns the agent's error message if the turn could not be completed.
    pub async fn prompt<F>(&mut self, prompt: &str, mut on_update: F) -> Result<(), String>
    where
        F: FnMut(SessionUpdate),
    {
        self.reset();

        let mut stream = Box::pin(self.agent.run(prompt, []));

        while let Some(event) = stream.next().await {
            if self.cancelled.load(Ordering::SeqCst) {
                break;
            }

            match event.map_err(|err| err.to_string())? {
                AgentEvent::Text(text) => {
                    on_update(SessionUpdate::AgentMessageChunk(text_chunk(text)));
                }
                AgentEvent::Reasoning(text) => {
                    on_update(SessionUpdate::AgentThoughtChunk(text_chunk(text)));
                }
                AgentEvent::ToolCallStart {
                    id,
                    name,
                    arguments,
                } => {
                    on_update(SessionUpdate::ToolCall(ToolCall {
                        tool_call_id: id,
                        title: name,
                        kind: None,
                        status: Some(ToolCallStatus::InProgress),
                        content: Vec::new(),
                        locations: Vec::new(),
                        raw_input: serde_json::from_str(&arguments).ok(),
                        raw_output: None,
                    }));
                }
                AgentEvent::ToolCallEnd { id, result, .. } => {
                    // The agent reports a typed ToolResult; ACP wants a status
                    // and the text a reader should see.
                    let status = if result.is_error() {
                        ToolCallStatus::Error
                    } else {
                        ToolCallStatus::Completed
                    };
                    let text = result
                        .render_for_model()
                        .unwrap_or_else(|error| error.to_string());
                    on_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                        tool_call_id: id,
                        status: Some(status),
                        content: Some(vec![ToolCallContent::Content {
                            content: ContentBlock::Text(TextContent {
                                text,
                                annotations: None,
                            }),
                        }]),
                        title: None,
                        kind: None,
                        locations: None,
                        raw_input: None,
                        raw_output: None,
                    }));
                }
                // Remaining events carry no ACP equivalent.
                _ => {}
            }
        }

        Ok(())
    }
}

/// Wraps plain text in the content-chunk shape ACP expects.
const fn text_chunk(text: String) -> ContentChunk {
    ContentChunk {
        content: ContentBlock::Text(TextContent {
            text,
            annotations: None,
        }),
    }
}
