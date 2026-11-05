//! ACP session management.
//!
//! Each session represents an active conversation with the agent.

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aither_agent::{Agent, AgentEvent};
use aither_core::LanguageModel;
use uuid::Uuid;

use crate::protocol::{
    ContentBlock, ContentChunk, McpServer, SessionUpdate, TextContent, ToolCall, ToolCallContent,
    ToolCallStatus, ToolCallUpdate,
};

/// A session's cancellation request, shared between the session and the
/// server's `session/cancel` handler.
///
/// The flag records that the turn was cancelled; the channel wakes a turn
/// that is waiting, so cancellation does not depend on the turn producing
/// another event.
#[derive(Debug, Clone)]
pub struct Cancellation {
    flag: Arc<AtomicBool>,
    wake: (async_channel::Sender<()>, async_channel::Receiver<()>),
}

impl Cancellation {
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            wake: async_channel::bounded(1),
        }
    }

    /// Requests that the running turn end.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        // A full channel already holds a pending wake-up.
        let _ = self.wake.0.try_send(());
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Clears a previous request before a new turn starts.
    fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst);
        while self.wake.1.try_recv().is_ok() {}
    }

    /// Resolves once the turn is cancelled.
    async fn requested(&self) {
        let _ = self.wake.1.recv().await;
    }
}

/// An active ACP session.
///
/// Each session wraps a conversation context and can process prompts.
#[derive(Debug)]
pub struct AcpSession<LLM: LanguageModel + 'static> {
    id: String,
    cwd: PathBuf,
    mcp_servers: Vec<McpServer>,
    cancelled: Cancellation,
    agent: Agent<LLM, LLM, LLM>,
}

impl<LLM: LanguageModel + 'static> AcpSession<LLM> {
    /// Create a new session that drives the given agent.
    #[must_use]
    pub fn new(cwd: PathBuf, mcp_servers: Vec<McpServer>, agent: Agent<LLM, LLM, LLM>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cwd,
            mcp_servers,
            cancelled: Cancellation::new(),
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
    pub fn mcp_servers(&self) -> &[McpServer] {
        &self.mcp_servers
    }

    /// Check if the session is cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.is_cancelled()
    }

    /// Cancel the current operation.
    pub fn stop(&self) {
        self.cancelled.cancel();
    }

    /// Reset cancellation flag.
    pub fn reset(&self) {
        self.cancelled.reset();
    }

    /// The shared cancellation request for this session.
    ///
    /// The server tracks one per session so a `session/cancel` notification
    /// can reach a session even while its prompt turn holds it out of the
    /// session map.
    pub(crate) fn cancellation(&self) -> Cancellation {
        self.cancelled.clone()
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

        let cancelled = &self.cancelled;
        // A cancelled turn ends like a completed one.
        let _ = self
            .agent
            .run(prompt, [])
            .drive(
                &mut on_update,
                cancelled.requested(),
                |on_update, timed| async move {
                    report_event(timed.event, on_update);
                    ControlFlow::Continue(on_update)
                },
            )
            .await
            .map_err(|err| err.to_string())?;

        Ok(())
    }
}

/// Reports one agent event to the client as the matching ACP session update.
fn report_event(event: AgentEvent, on_update: &mut impl FnMut(SessionUpdate)) {
    match event {
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
                meta: None,
            }));
        }
        AgentEvent::ToolCallEnd { id, result, .. } => {
            // The agent reports a typed ToolResult; ACP wants a status
            // and the text a reader should see.
            let status = if result.is_error() {
                ToolCallStatus::Failed
            } else {
                ToolCallStatus::Completed
            };
            let text = result.render_for_model();
            on_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                tool_call_id: id,
                status: Some(status),
                content: Some(vec![ToolCallContent::Content {
                    content: ContentBlock::Text(TextContent {
                        text,
                        annotations: None,
                        meta: None,
                    }),
                }]),
                title: None,
                kind: None,
                locations: None,
                raw_input: None,
                raw_output: None,
                meta: None,
            }));
        }
        // Remaining events carry no ACP equivalent.
        _ => {}
    }
}

/// Wraps plain text in the content-chunk shape ACP expects.
const fn text_chunk(text: String) -> ContentChunk {
    ContentChunk {
        content: ContentBlock::Text(TextContent {
            text,
            annotations: None,
            meta: None,
        }),
        message_id: None,
        meta: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::{Event, LLMRequest, model::Profile};
    use futures_lite::Stream;

    #[derive(Debug)]
    struct NeverError;

    impl std::fmt::Display for NeverError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("never")
        }
    }

    impl std::error::Error for NeverError {}

    impl aither_core::llm::Classify for NeverError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    /// A provider whose stream never yields.
    #[derive(Clone)]
    struct HungModel;

    impl LanguageModel for HungModel {
        type Error = NeverError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, NeverError>> + Send {
            futures_lite::stream::pending()
        }

        fn profile(&self) -> impl Future<Output = Result<Profile, NeverError>> + Send {
            std::future::ready(Ok(Profile::new(
                "hung",
                "test",
                "hung",
                "Hung model",
                128_000,
            )))
        }
    }

    #[tokio::test]
    async fn cancelling_ends_a_turn_blocked_on_a_provider_that_never_yields() {
        let mut session = AcpSession::new(PathBuf::from("."), Vec::new(), Agent::new(HungModel));
        let cancellation = session.cancellation();

        let turn = tokio::spawn(async move {
            let result = session.prompt("hello", |_| {}).await;
            (session, result)
        });
        // Let the turn reach the provider before cancelling it.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        cancellation.cancel();

        let (session, result) = tokio::time::timeout(std::time::Duration::from_secs(5), turn)
            .await
            .expect("the cancelled turn must return")
            .expect("turn task");
        assert_eq!(result, Ok(()));
        assert!(session.is_cancelled());
    }
}
