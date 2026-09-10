//! Tests for `AcpClient::spawn` against `tests/fake_agent.py` running as a
//! child process.

use std::sync::Mutex;

use aither_acp::{
    AcpClient, ClientError, ClientHandler, ContentBlock, RequestPermissionOutcome,
    RequestPermissionParams, RequestPermissionResult, SessionNotification, SessionUpdate,
    StopReason, TextContent,
};
use aither_mcp::protocol::JsonRpcError;

/// Handler that records updates and auto-approves permission requests.
#[derive(Default)]
struct TestHandler {
    updates: Mutex<Vec<SessionNotification>>,
    permission_calls: Mutex<Vec<RequestPermissionParams>>,
}

impl ClientHandler for TestHandler {
    fn session_update(
        &self,
        notification: SessionNotification,
    ) -> impl std::future::Future<Output = ()> + Send {
        self.updates
            .lock()
            .expect("updates poisoned")
            .push(notification);
        std::future::ready(())
    }

    fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> impl std::future::Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send
    {
        self.permission_calls
            .lock()
            .expect("permission_calls poisoned")
            .push(params);
        async move {
            Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Selected {
                    option_id: "allow-1".to_string(),
                },
                meta: None,
            })
        }
    }
}

fn text(message: &str) -> ContentBlock {
    ContentBlock::Text(TextContent {
        text: message.to_string(),
        annotations: None,
    })
}

/// Spawn the Python fake agent; `None` (with a message) when `python3` or the
/// script cannot be launched.
fn spawn_agent() -> Option<(AcpClient<TestHandler>, tokio::task::JoinHandle<()>)> {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_agent.py");
    match AcpClient::spawn("python3", &[script], [], "/tmp", TestHandler::default()) {
        Ok((client, connection)) => Some((client, tokio::spawn(connection))),
        Err(error) => {
            eprintln!("skipping child-process test: cannot spawn python3: {error}");
            None
        }
    }
}

#[tokio::test]
async fn child_agent_streams_and_responds() {
    let Some((client, connection)) = spawn_agent() else {
        return;
    };

    let init = client.initialize().await.expect("initialize failed");
    assert_eq!(init.protocol_version, 1);
    assert!(init.agent_capabilities.load_session);

    let session = client.new_session("/tmp", vec![]).await.unwrap();
    assert_eq!(session.session_id, "sess-1");
    client
        .set_mode(&session.session_id, "bypass")
        .await
        .unwrap();
    client
        .set_config_option(&session.session_id, "model", "b")
        .await
        .unwrap();

    let result = client
        .prompt(&session.session_id, vec![text("hello")])
        .await
        .expect("prompt failed");
    assert_eq!(result.stop_reason, StopReason::EndTurn);

    let updates = client.handler().updates.lock().unwrap().clone();
    let text: String = updates
        .iter()
        .filter_map(|n| match &n.update {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello abworld");
    assert!(
        updates
            .iter()
            .any(|n| matches!(n.update, SessionUpdate::ToolCall(_)))
    );
    assert_eq!(client.handler().permission_calls.lock().unwrap().len(), 1);

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn child_agent_cancel() {
    let Some((client, connection)) = spawn_agent() else {
        return;
    };
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();

    let prompt = {
        let client = client.clone();
        let session_id = session.session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, vec![text("wait")]).await })
    };
    for _ in 0..10_000 {
        if !client.handler().updates.lock().unwrap().is_empty() {
            break;
        }
        tokio::task::yield_now().await;
    }
    client.cancel(&session.session_id).await.unwrap();
    let result = prompt.await.expect("prompt task panicked").unwrap();
    assert_eq!(result.stop_reason, StopReason::Cancelled);

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn child_agent_exit_fails_pending_request_with_status() {
    let Some((client, connection)) = spawn_agent() else {
        return;
    };
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();

    let result = client.prompt(&session.session_id, vec![text("die")]).await;
    match result {
        Err(ClientError::Closed {
            status: Some(status),
        }) => {
            assert_eq!(status.code(), Some(3));
        }
        other => panic!("expected Closed with exit status, got {other:?}"),
    }

    connection.await.expect("connection task panicked");
}
