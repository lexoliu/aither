//! A session's MCP servers are connected and their tools reach its agent.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use aither_acp::{
    AcpClient, AcpServer, ClientHandler, ContentBlock, McpServerStdio, PromptParams,
    RequestPermissionParams, RequestPermissionResult, SessionNewParams, SessionNotification,
    SessionUpdate, StopReason, TextContent, ToolCallContent, ToolCallStatus,
};
use aither_agent::Agent;
use aither_core::LanguageModel;
use aither_core::llm::{Event, LLMRequest, model::Profile};
use aither_mcp::protocol::JsonRpcError;
use aither_mcp::transport::DuplexTransport;
use futures_lite::Stream;

#[derive(Debug)]
struct ModelError;

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("model error")
    }
}

impl std::error::Error for ModelError {}

impl aither_core::llm::Classify for ModelError {
    fn failure(&self) -> aither_core::llm::Failure {
        aither_core::llm::Failure::permanent()
    }
}

/// Calls `ping` on its first request and ends the turn on the next.
#[derive(Clone, Default)]
struct PingsOnce {
    requests: std::sync::Arc<AtomicUsize>,
}

impl LanguageModel for PingsOnce {
    type Error = ModelError;

    fn respond(
        &self,
        _request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, ModelError>> + Send {
        let event = if self.requests.fetch_add(1, Ordering::SeqCst) == 0 {
            Event::tool_call("call-1", "ping", serde_json::json!({}))
        } else {
            Event::Text("done".to_string())
        };
        futures_lite::stream::once(Ok(event))
    }

    fn profile(&self) -> impl Future<Output = Result<Profile, ModelError>> + Send {
        std::future::ready(Ok(Profile::new("ping", "test", "ping", "Ping", 128_000)))
    }
}

#[derive(Default)]
struct Recorder {
    updates: Mutex<Vec<SessionNotification>>,
}

impl ClientHandler for Recorder {
    fn session_update(&self, notification: SessionNotification) -> impl Future<Output = ()> + Send {
        self.updates.lock().expect("updates").push(notification);
        std::future::ready(())
    }

    fn request_permission(
        &self,
        _params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        std::future::ready(Err(JsonRpcError::method_not_found("request_permission")))
    }
}

#[tokio::test]
async fn a_session_agent_can_call_the_tools_of_the_mcp_servers_it_was_opened_with() {
    let (client_end, server_end) = DuplexTransport::pair();
    let (client, connection) = AcpClient::connect(client_end, Recorder::default());
    let connection = tokio::spawn(connection);
    let mut server = AcpServer::new(
        server_end,
        "test",
        "0",
        executor_core::tokio::TokioGlobal,
        |setup| async move {
            let builder = setup.mcp.into_iter().fold(
                Agent::builder(PingsOnce::default()),
                aither_agent::AgentBuilder::mcp,
            );
            Ok(builder.build())
        },
    );
    let server = tokio::spawn(async move { server.run().await });

    client.initialize().await.expect("initialize");
    let script = format!(
        "{}/tests/fixtures/ping_mcp_server.sh",
        env!("CARGO_MANIFEST_DIR")
    );
    let session = client
        .new_session(
            SessionNewParams::new(std::env::temp_dir()).mcp_servers(vec![
                McpServerStdio::new("ping", "sh").args(vec![script]).into(),
            ]),
        )
        .await
        .expect("session/new");
    let result = client
        .prompt(PromptParams::new(
            session.session_id.clone(),
            vec![ContentBlock::Text(TextContent {
                text: "ping".to_string(),
                annotations: None,
                meta: None,
            })],
        ))
        .await
        .expect("prompt");
    assert_eq!(result.stop_reason, StopReason::EndTurn);

    let updates = client.handler().updates.lock().expect("updates");
    let finished = updates
        .iter()
        .find_map(|n| match &n.update {
            SessionUpdate::ToolCallUpdate(update) => Some(update),
            _ => None,
        })
        .expect("the agent called a tool");
    assert_eq!(finished.status, Some(ToolCallStatus::Completed));
    let Some(
        [
            ToolCallContent::Content {
                content: ContentBlock::Text(text),
            },
        ],
    ) = finished.content.as_deref()
    else {
        panic!("expected text content, got {:?}", finished.content);
    };
    assert_eq!(text.text, "pong");

    drop(updates);
    server.abort();
    connection.abort();
}
