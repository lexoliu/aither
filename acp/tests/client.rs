//! Tests for `AcpClient` against an in-process fake ACP agent.

use std::sync::{Arc, Mutex};

use aither_acp::{
    AcpClient, ClientError, ClientHandler, ConfigOptionValue, ContentBlock, CurrentModeUpdate,
    PROTOCOL_VERSION, ReadTextFileParams, ReadTextFileResult, RequestPermissionOutcome,
    RequestPermissionParams, RequestPermissionResult, SessionNotification, SessionUpdate,
    StopReason, TextContent,
};
use aither_mcp::protocol::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId,
};
use aither_mcp::transport::{BidirectionalTransport, DuplexTransport, Transport};
use serde_json::{Value, json};

/// Events the fake agent records about what the client sent it.
type Log = Arc<Mutex<Vec<String>>>;

/// A scripted ACP agent on the far end of a [`DuplexTransport`].
///
/// It answers `initialize`, `session/new`, `session/load`, `session/set_mode`
/// and `session/set_config_option`. On `session/prompt` it streams message
/// chunks plus a tool call, issues one `session/request_permission` and one
/// `fs/read_text_file` request, and answers the prompt once both responses
/// arrived — or `cancelled` if `session/cancel` arrives first. A prompt whose
/// text is `"wait"` skips the agent-initiated requests and only waits; a
/// prompt whose text is `"die"` closes the transport, simulating the agent
/// process exiting mid-turn.
struct FakeAgent {
    transport: DuplexTransport,
    log: Log,
    session_id: String,
    pending_prompt: Option<RequestId>,
    permission_id: Option<RequestId>,
    fs_id: Option<RequestId>,
    next_id: i64,
}

impl FakeAgent {
    fn spawn(transport: DuplexTransport) -> (tokio::task::JoinHandle<()>, Log) {
        let agent = Self {
            transport,
            log: Log::default(),
            session_id: String::new(),
            pending_prompt: None,
            permission_id: None,
            fs_id: None,
            next_id: 1,
        };
        let log = agent.log.clone();
        (tokio::spawn(agent.run()), log)
    }

    async fn run(mut self) {
        loop {
            let Ok(Some(message)) = self.transport.recv().await else {
                return;
            };
            match message {
                JsonRpcMessage::Request(request) => self.on_request(request).await,
                JsonRpcMessage::Notification(notification) => {
                    self.on_notification(notification).await;
                }
                JsonRpcMessage::Response(response) => self.on_response(response).await,
            }
        }
    }

    async fn on_request(&mut self, request: JsonRpcRequest) {
        let params = request.params.clone().unwrap_or_default();
        match request.method.as_str() {
            "initialize" => self.on_initialize(request.id).await,
            "session/new" => self.on_session_new(request.id).await,
            "session/load" => self.on_session_load(request.id, &params).await,
            "session/set_mode" => {
                self.log(format!("set_mode:{}", params["modeId"]));
                self.respond(request.id, json!({})).await;
            }
            "session/set_config_option" => {
                self.log(format!(
                    "set_config:{}={}",
                    params["configId"], params["value"]
                ));
                self.respond(
                    request.id,
                    json!({
                        "configOptions": [{
                            "id": params["configId"],
                            "name": "Model",
                            "type": "select",
                            "currentValue": params["value"],
                            "options": [],
                        }],
                    }),
                )
                .await;
            }
            "session/prompt" => self.on_prompt(request.id, &params).await,
            _ => {
                let _ = self
                    .transport
                    .respond(JsonRpcResponse::error(
                        request.id,
                        JsonRpcError::method_not_found(&request.method),
                    ))
                    .await;
            }
        }
    }

    async fn on_initialize(&mut self, id: RequestId) {
        self.respond(
            id,
            json!({
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": {"list": {}, "delete": {}},
                },
                "agentInfo": {"name": "fake-agent", "version": "0.1.0"},
                "authMethods": [{"id": "none", "name": "No auth"}],
            }),
        )
        .await;
    }

    async fn on_session_new(&mut self, id: RequestId) {
        self.session_id = "sess-1".to_string();
        self.respond(
            id,
            json!({
                "sessionId": "sess-1",
                "modes": {
                    "currentModeId": "smart",
                    "availableModes": [
                        {"id": "smart", "name": "Smart"},
                        {"id": "bypass", "name": "Bypass"},
                    ],
                },
                "configOptions": [{
                    "id": "model",
                    "name": "Model",
                    "category": "model",
                    "type": "select",
                    "currentValue": "a",
                    "options": [{"value": "a", "name": "A"}, {"value": "b", "name": "B"}],
                }],
            }),
        )
        .await;
    }

    async fn on_session_load(&mut self, id: RequestId, params: &Value) {
        let session_id = params["sessionId"].as_str().unwrap_or("sess-1").to_string();
        self.session_id.clone_from(&session_id);
        self.send_update(
            &session_id,
            json!({
                "sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": "loaded user message"},
            }),
        )
        .await;
        self.respond(id, json!({})).await;
    }

    async fn on_prompt(&mut self, id: RequestId, params: &Value) {
        let session_id = params["sessionId"].as_str().unwrap_or_default().to_string();
        self.session_id.clone_from(&session_id);
        let text = params["prompt"][0]["text"].as_str().unwrap_or_default();
        if text == "die" {
            // Simulate the agent process exiting mid-turn.
            let _ = self.transport.close().await;
            return;
        }
        self.pending_prompt = Some(id);
        self.send_update(
            &session_id,
            json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Hello "},
            }),
        )
        .await;
        self.send_update(
            &session_id,
            json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "tc-1", "title": "fake-tool", "status": "pending",
            }),
        )
        .await;
        self.send_update(
            &session_id,
            json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc-1", "status": "in_progress",
            }),
        )
        .await;
        self.send_update(
            &session_id,
            json!({"sessionUpdate": "acme.vendor_update", "payload": 1}),
        )
        .await;
        if text != "wait" {
            self.permission_id = Some(
                self.send_request(
                    "session/request_permission",
                    json!({
                        "sessionId": session_id,
                        "toolCall": {
                            "toolCallId": "tc-1",
                            "title": "fake-tool",
                            "status": "in_progress",
                        },
                        "options": [
                            {"optionId": "allow-1", "name": "Allow once", "kind": "allow_once"},
                            {"optionId": "deny-1", "name": "Deny", "kind": "reject_once"},
                        ],
                    }),
                )
                .await,
            );
            // Two more chunks stream while the permission request is still
            // outstanding, then a second agent-to-client request lands.
            self.send_update(
                &session_id,
                json!({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "a"},
                }),
            )
            .await;
            self.send_update(
                &session_id,
                json!({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "b"},
                }),
            )
            .await;
            self.fs_id = Some(
                self.send_request(
                    "fs/read_text_file",
                    json!({"sessionId": session_id, "path": "/etc/hostname"}),
                )
                .await,
            );
        }
    }

    async fn on_notification(&mut self, notification: JsonRpcNotification) {
        if notification.method == "session/cancel" {
            if self.permission_id.is_some() {
                self.log("cancel_pending_perm".to_string());
            } else {
                self.log("cancel".to_string());
            }
            if let Some(id) = self.pending_prompt.take() {
                self.respond(id, json!({"stopReason": "cancelled"})).await;
            }
        }
    }

    async fn on_response(&mut self, response: JsonRpcResponse) {
        if Some(&response.id) == self.permission_id.as_ref() {
            self.permission_id = None;
            self.log(format!(
                "permission:{}",
                response.result.as_ref().map_or_else(
                    || "error".to_string(),
                    |r| r["outcome"]["optionId"].to_string()
                )
            ));
        } else if Some(&response.id) == self.fs_id.as_ref() {
            self.fs_id = None;
            self.log(format!(
                "fs:{}",
                response
                    .error
                    .map_or_else(|| "ok".to_string(), |e| e.code.0.to_string())
            ));
        }
        // Finish the prompt once both agent-initiated requests were answered.
        if self.pending_prompt.is_some() && self.permission_id.is_none() && self.fs_id.is_none() {
            let session_id = self.session_id.clone();
            self.send_update(
                &session_id,
                json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-1", "status": "completed",
                }),
            )
            .await;
            self.send_update(
                &session_id,
                json!({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "world"},
                }),
            )
            .await;
            if let Some(id) = self.pending_prompt.take() {
                self.respond(id, json!({"stopReason": "end_turn"})).await;
            }
        }
    }

    async fn respond(&mut self, id: RequestId, result: Value) {
        self.transport
            .respond(JsonRpcResponse::success(id, result))
            .await
            .expect("fake agent respond failed");
    }

    async fn send_update(&mut self, session_id: &str, update: Value) {
        self.transport
            .notify(JsonRpcNotification::with_params(
                "session/update",
                json!({"sessionId": session_id, "update": update}),
            ))
            .await
            .expect("fake agent notify failed");
    }

    async fn send_request(&mut self, method: &str, params: Value) -> RequestId {
        let id = RequestId::String(format!("agent-{}", self.next_id));
        self.next_id += 1;
        self.transport
            .send_request(JsonRpcRequest::with_params(id.clone(), method, params))
            .await
            .expect("fake agent request failed");
        id
    }

    fn log(&self, event: String) {
        self.log.lock().expect("log poisoned").push(event);
    }
}

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
            .push(params.clone());
        let option_id = params
            .options
            .first()
            .map_or_else(|| "none".to_string(), |o| o.option_id.clone());
        async move {
            Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Selected { option_id },
                meta: None,
            })
        }
    }
}

/// Handler whose permission answer waits for a release signal, and
/// optionally for progress on the connection (three streamed message chunks
/// and a completed `fs/read_text_file` answer). Exercises concurrent
/// agent-to-client requests.
struct GatedHandler {
    updates: Mutex<Vec<SessionNotification>>,
    permission_calls: Mutex<Vec<RequestPermissionParams>>,
    fs_calls: Mutex<Vec<ReadTextFileParams>>,
    /// Releases the permission answer once the test allows it.
    release: async_channel::Receiver<()>,
    /// Wait for streamed chunks plus the answered fs read before answering.
    wait_for_progress: bool,
}

impl ClientHandler for GatedHandler {
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
        let option_id = params
            .options
            .first()
            .map_or_else(|| "none".to_string(), |o| o.option_id.clone());
        self.permission_calls
            .lock()
            .expect("permission_calls poisoned")
            .push(params);
        async move {
            if self.wait_for_progress {
                // Answer only once both extra chunks and the fs read have been
                // observed — impossible if the connection stalls on us.
                loop {
                    let ready = {
                        let chunks = self
                            .updates
                            .lock()
                            .expect("updates poisoned")
                            .iter()
                            .filter(|n| matches!(n.update, SessionUpdate::AgentMessageChunk(_)))
                            .count();
                        let fs = self.fs_calls.lock().expect("fs_calls poisoned").len();
                        chunks >= 3 && fs >= 1
                    };
                    if ready {
                        break;
                    }
                    futures_lite::future::yield_now().await;
                }
            }
            let _ = self.release.recv().await;
            Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Selected { option_id },
                meta: None,
            })
        }
    }

    fn read_text_file(
        &self,
        params: ReadTextFileParams,
    ) -> impl std::future::Future<Output = Result<ReadTextFileResult, JsonRpcError>> + Send {
        // Recorded when the handler produces its answer; a `wait_for_progress`
        // gate can only observe it on a poll after this request completed.
        self.fs_calls
            .lock()
            .expect("fs_calls poisoned")
            .push(params);
        std::future::ready(Ok(ReadTextFileResult {
            content: "file contents".to_string(),
            meta: None,
        }))
    }
}

fn text(message: &str) -> ContentBlock {
    ContentBlock::Text(TextContent {
        text: message.to_string(),
        annotations: None,
    })
}

/// Wire a client to a fake agent and drive both on the test executor.
fn connect() -> (
    AcpClient<TestHandler>,
    Log,
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<()>,
) {
    let (client_transport, agent_transport) = DuplexTransport::pair();
    let (client, connection) = AcpClient::connect(client_transport, TestHandler::default());
    let (agent, log) = FakeAgent::spawn(agent_transport);
    (client, log, tokio::spawn(connection), agent)
}

/// Wire a client with a [`GatedHandler`] to a fake agent; the returned sender
/// releases the pending permission answer.
fn connect_gated(
    wait_for_progress: bool,
) -> (
    AcpClient<GatedHandler>,
    Log,
    async_channel::Sender<()>,
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<()>,
) {
    let (release_tx, release_rx) = async_channel::bounded(1);
    let handler = GatedHandler {
        updates: Mutex::default(),
        permission_calls: Mutex::default(),
        fs_calls: Mutex::default(),
        release: release_rx,
        wait_for_progress,
    };
    let (client_transport, agent_transport) = DuplexTransport::pair();
    let (client, connection) = AcpClient::connect(client_transport, handler);
    let (agent, log) = FakeAgent::spawn(agent_transport);
    (client, log, release_tx, tokio::spawn(connection), agent)
}

/// Yield until `predicate` holds; panics after too many attempts.
async fn wait_until(mut predicate: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if predicate() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("condition not reached");
}

#[tokio::test]
async fn initialize_and_new_session() {
    let (client, log, connection, _agent) = connect();

    let init = client.initialize().await.expect("initialize failed");
    assert_eq!(init.protocol_version, PROTOCOL_VERSION);
    assert!(init.agent_capabilities.load_session);
    assert!(init.agent_capabilities.session_capabilities.list.is_some());
    assert_eq!(
        init.agent_info.as_ref().map(|i| i.name.as_str()),
        Some("fake-agent")
    );
    assert_eq!(init.auth_methods[0].id, "none");

    let session = client
        .new_session("/tmp", vec![])
        .await
        .expect("session/new failed");
    assert_eq!(session.session_id, "sess-1");
    let modes = session.modes.expect("agent should send modes");
    assert_eq!(modes.current_mode_id, "smart");
    assert_eq!(modes.available_modes.len(), 2);
    let options = session.config_options.expect("agent should send options");
    assert_eq!(options[0].id, "model");
    assert_eq!(options[0].kind.as_deref(), Some("select"));

    client
        .set_mode(&session.session_id, "bypass")
        .await
        .unwrap();
    let options = client
        .set_config_option(&session.session_id, "model", "b")
        .await
        .unwrap();
    assert_eq!(
        options[0].current_value,
        Some(ConfigOptionValue::Selected("b".to_string()))
    );

    let log = log.lock().expect("log poisoned").clone();
    assert!(log.iter().any(|e| e == "set_mode:\"bypass\""));
    assert!(log.iter().any(|e| e == "set_config:\"model\"=\"b\""));

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn prompt_streams_updates_and_returns_end_turn() {
    let (client, log, connection, _agent) = connect();
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();

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
    assert_eq!(
        updates
            .iter()
            .filter(|n| matches!(n.update, SessionUpdate::ToolCallUpdate(_)))
            .count(),
        2
    );
    // The vendor update tag lands in the catch-all.
    assert!(
        updates
            .iter()
            .any(|n| matches!(&n.update, SessionUpdate::Other(_)))
    );

    // The agent saw the permission answer and the fs method-not-found error.
    let permission_calls = client.handler().permission_calls.lock().unwrap().clone();
    assert_eq!(permission_calls.len(), 1);
    assert_eq!(permission_calls[0].options.len(), 2);
    let log = log.lock().unwrap().clone();
    assert!(log.iter().any(|e| e == "permission:\"allow-1\""));
    assert!(log.iter().any(|e| e == "fs:-32601"));

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn follow_up_prompt_on_same_session() {
    let (client, _log, connection, _agent) = connect();
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();

    for _ in 0..2 {
        let result = client
            .prompt(&session.session_id, vec![text("again")])
            .await
            .expect("prompt failed");
        assert_eq!(result.stop_reason, StopReason::EndTurn);
    }

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn cancel_mid_turn() {
    let (client, log, connection, _agent) = connect();
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();
    let session_id = session.session_id.clone();

    let prompt = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, vec![text("wait")]).await })
    };

    // Wait for the first streamed update, proving the turn is in flight.
    wait_until(|| !client.handler().updates.lock().unwrap().is_empty()).await;
    client.cancel(&session_id).await.unwrap();

    let result = prompt.await.expect("prompt task panicked").unwrap();
    assert_eq!(result.stop_reason, StopReason::Cancelled);
    assert!(log.lock().unwrap().iter().any(|e| e == "cancel"));

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn load_session_replays_history() {
    let (client, _log, connection, _agent) = connect();
    client.initialize().await.unwrap();

    client
        .load_session("sess-old", "/tmp", vec![])
        .await
        .expect("session/load failed");

    wait_until(|| !client.handler().updates.lock().unwrap().is_empty()).await;
    let updates = client.handler().updates.lock().unwrap().clone();
    assert!(updates.iter().any(|n| matches!(
        &n.update,
        SessionUpdate::UserMessageChunk(chunk)
            if matches!(&chunk.content, ContentBlock::Text(t) if t.text == "loaded user message")
    )));

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn agent_exit_fails_pending_requests() {
    let (client, _log, connection, agent) = connect();
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();

    let result = client.prompt(&session.session_id, vec![text("die")]).await;
    assert!(
        matches!(result, Err(ClientError::Closed { .. })),
        "expected Closed, got {result:?}"
    );

    // The connection task terminates as well.
    connection.await.expect("connection task panicked");
    let _ = agent.await;
}

#[tokio::test]
async fn unknown_session_update_tag_parses_as_other() {
    let notification: SessionNotification = serde_json::from_str(
        r#"{"sessionId":"s","update":{"sessionUpdate":"brand_new_kind","x":1}}"#,
    )
    .unwrap();
    let SessionUpdate::Other(raw) = notification.update else {
        panic!("expected Other, got {:?}", notification.update)
    };
    assert_eq!(raw["sessionUpdate"], "brand_new_kind");
    assert_eq!(raw["x"], 1);

    // A known tag with a malformed payload also lands in Other.
    let notification: SessionNotification =
        serde_json::from_str(r#"{"sessionId":"s","update":{"sessionUpdate":"tool_call"}}"#)
            .unwrap();
    assert!(matches!(notification.update, SessionUpdate::Other(_)));

    // A known update round-trips through the tagged shape.
    let update = SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
        current_mode_id: "plan".to_string(),
        meta: None,
    });
    let value = serde_json::to_value(&update).unwrap();
    assert_eq!(value["sessionUpdate"], "current_mode_update");
    let parsed: SessionUpdate = serde_json::from_value(value).unwrap();
    assert!(matches!(parsed, SessionUpdate::CurrentModeUpdate(_)));
}

#[tokio::test]
async fn permission_does_not_block_the_connection() {
    let (client, log, release, connection, _agent) = connect_gated(true);
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();
    // Arm the release; the progress gate still delays the permission answer
    // until the handler has observed both extra chunks and the fs answer.
    release.send(()).await.unwrap();

    let result = client
        .prompt(&session.session_id, vec![text("hello")])
        .await
        .expect("prompt failed");
    assert_eq!(result.stop_reason, StopReason::EndTurn);

    let updates = client.handler().updates.lock().unwrap().clone();
    let chunks = updates
        .iter()
        .filter(|n| matches!(n.update, SessionUpdate::AgentMessageChunk(_)))
        .count();
    assert!(chunks >= 3, "expected the extra chunks to stream through");
    assert_eq!(client.handler().fs_calls.lock().unwrap().len(), 1);
    assert_eq!(client.handler().permission_calls.lock().unwrap().len(), 1);

    // The fs answer must have reached the agent before the permission answer:
    // the gate only releases on the pass after the fs handler completed.
    let log = log.lock().unwrap().clone();
    let fs_pos = log
        .iter()
        .position(|e| e == "fs:ok")
        .expect("fs answer missing");
    let perm_pos = log
        .iter()
        .position(|e| e == "permission:\"allow-1\"")
        .expect("permission answer missing");
    assert!(fs_pos < perm_pos, "fs answer must arrive first: {log:?}");

    client.close();
    connection.await.expect("connection task panicked");
}

#[tokio::test]
async fn cancel_goes_out_while_permission_pending() {
    let (client, log, release, connection, _agent) = connect_gated(false);
    client.initialize().await.unwrap();
    let session = client.new_session("/tmp", vec![]).await.unwrap();
    let session_id = session.session_id.clone();

    let prompt = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, vec![text("hello")]).await })
    };

    // The permission request is dispatched but its answer is not released.
    wait_until(|| !client.handler().permission_calls.lock().unwrap().is_empty()).await;
    client.cancel(&session_id).await.unwrap();

    // The cancel reached the wire while the permission was still outstanding.
    wait_until(|| {
        log.lock()
            .unwrap()
            .iter()
            .any(|e| e == "cancel_pending_perm")
    })
    .await;
    let result = prompt.await.expect("prompt task panicked").unwrap();
    assert_eq!(result.stop_reason, StopReason::Cancelled);

    // Let the blocked handler finish; its late response is written to the
    // agent but the prompt is already answered.
    release.send(()).await.unwrap();
    wait_until(|| {
        log.lock()
            .unwrap()
            .iter()
            .any(|e| e == "permission:\"allow-1\"")
    })
    .await;

    client.close();
    connection.await.expect("connection task panicked");
}
