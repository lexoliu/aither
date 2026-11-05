//! An MCP server's tool list changing mid-session, through the public
//! `TerminalAgentBuilder` and a real MCP connection: the new tool runs at
//! once, the model hears of it in one diff that leaves the cached prefix as it
//! was, and after a compaction the clean command table lists it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aither_core::LanguageModel;
use aither_core::llm::{Event, LLMRequest, Message, ToolCall, model::Profile as ModelProfile};
use aither_mcp::McpConnection;
use aither_mcp::protocol::{JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use aither_mcp::transport::{BidirectionalTransport, DuplexTransport, Transport};
use async_channel::{Receiver, Sender};
use executor_core::tokio::TokioGlobal;
use futures_core::Stream;
use futures_lite::future;
use serde_json::{Value, json};
use tempfile::tempdir;

use crate::sandbox::TerminalTool;
use crate::sandbox::permission::NoopPermissionHandler;
use crate::{AgentOptions, TerminalAgentBuilder, test_support};

const HANDOFF_JSON: &str = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#;

/// The tag the command table's diff alert wraps its diff in.
const DIFF_TAG: &str = "<commands_diff boundary=";

/// Fails the test instead of hanging it when `wait` does not finish.
async fn bounded<T>(what: &str, wait: impl Future<Output = T>) -> T {
    future::or(wait, async {
        async_io::Timer::after(Duration::from_secs(20)).await;
        panic!("{what} did not happen within 20s")
    })
    .await
}

/// The server's description of the tool `name`.
fn tool(name: &str) -> Value {
    json!({
        "name": name,
        "description": format!("Runs {name} for the test. Ignores its arguments."),
        "inputSchema": {"type": "object"}
    })
}

/// An MCP server over `transport` offering `alpha`, whose tool list the test
/// replaces through `changes`.
///
/// After the client re-lists a changed list, the server pings it and reports
/// the answer on `relisted`: the client handles the re-listing before the
/// ping, so by then its tool list is the new one.
async fn serve(
    mut transport: DuplexTransport,
    changes: Receiver<Vec<&'static str>>,
    relisted: Sender<()>,
) {
    enum Step {
        Message(Option<JsonRpcMessage>),
        Change(Vec<&'static str>),
    }

    let mut tools = vec!["alpha"];
    let mut relist_pending = false;
    loop {
        let step = future::or(
            async { Step::Message(transport.recv().await.expect("the duplex never fails")) },
            async {
                match changes.recv().await {
                    Ok(names) => Step::Change(names),
                    Err(_) => std::future::pending().await,
                }
            },
        )
        .await;
        match step {
            Step::Change(names) => {
                tools = names;
                relist_pending = true;
                transport
                    .notify(JsonRpcNotification::new("notifications/tools/list_changed"))
                    .await
                    .unwrap();
            }
            Step::Message(None) => return,
            Step::Message(Some(JsonRpcMessage::Request(request))) => {
                let result = match request.method.as_str() {
                    "initialize" => json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": true}},
                        "serverInfo": {"name": "scripted"}
                    }),
                    "tools/list" => {
                        json!({"tools": tools.iter().map(|name| tool(name)).collect::<Vec<_>>()})
                    }
                    "tools/call" => {
                        let name = request.params.as_ref().unwrap()["name"].as_str().unwrap();
                        json!({"content": [{"type": "text", "text": format!("{name} ran on the server")}]})
                    }
                    other => panic!("the client sent an unexpected {other} request"),
                };
                let relist = request.method == "tools/list" && std::mem::take(&mut relist_pending);
                transport
                    .respond(JsonRpcResponse::success(request.id, result))
                    .await
                    .unwrap();
                if relist {
                    transport
                        .send_request(JsonRpcRequest::new("relisted", "ping"))
                        .await
                        .unwrap();
                }
            }
            Step::Message(Some(JsonRpcMessage::Response(_))) => relisted.send(()).await.unwrap(),
            Step::Message(Some(JsonRpcMessage::Notification(_))) => {}
        }
    }
}

#[derive(Debug)]
struct MockError;

impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("mock error")
    }
}

impl std::error::Error for MockError {}

impl aither_core::llm::Classify for MockError {
    fn failure(&self) -> aither_core::llm::Failure {
        aither_core::llm::Failure::permanent()
    }
}

/// Records every request. Its first turn has the server add `gamma` and,
/// once the connection has re-listed, runs `gamma` in the terminal; then it
/// answers, writes the handoff for the compaction, and answers again.
#[derive(Clone)]
struct ScriptedLlm {
    requests: Sender<LLMRequest>,
    turns: Arc<AtomicUsize>,
    changes: Sender<Vec<&'static str>>,
    relisted: Receiver<()>,
}

impl LanguageModel for ScriptedLlm {
    type Error = MockError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.requests.try_send(request).expect("record the request");
        let turn = self.turns.fetch_add(1, Ordering::SeqCst);
        let changes = self.changes.clone();
        let relisted = self.relisted.clone();
        futures_lite::stream::once_future(async move {
            let event = match turn {
                0 => {
                    changes.send(vec!["alpha", "gamma"]).await.unwrap();
                    bounded("the re-listing", relisted.recv()).await.unwrap();
                    Event::ToolCall(ToolCall::new(
                        "run_gamma",
                        "terminal",
                        json!({
                            "description": "Run the tool the server just added",
                            "script": "gamma",
                            "mode": "sandboxed",
                            "timeout": 30
                        }),
                    ))
                }
                2 => Event::Text(HANDOFF_JSON.to_string()),
                _ => Event::Text("done".to_string()),
            };
            Ok(event)
        })
    }

    fn profile(&self) -> impl Future<Output = Result<ModelProfile, Self::Error>> + Send {
        std::future::ready(Ok(ModelProfile::new(
            "mock",
            "test",
            "mock-model",
            "mock model",
            1_000_000,
        )))
    }
}

/// The messages of `request` that carry the command table's diff alert.
fn diff_alerts(request: &LLMRequest) -> Vec<&Message> {
    request
        .messages()
        .iter()
        .filter(|message| message.content().contains(DIFF_TAG))
        .collect()
}

/// The system prefix of `request`, which opens it.
fn prefix(request: &LLMRequest) -> &str {
    let first = &request.messages()[0];
    assert!(
        first.content().contains("<commands>\n"),
        "the prefix lists the commands"
    );
    first.content()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_mcp_tool_added_mid_session_runs_at_once_and_is_announced_once() {
    let (client_end, server_end) = DuplexTransport::pair();
    let (changes, server_changes) = async_channel::unbounded();
    let (server_relisted, relisted) = async_channel::unbounded();
    let server = tokio::spawn(serve(server_end, server_changes, server_relisted));
    let connection = McpConnection::connect(&TokioGlobal, client_end, ())
        .await
        .expect("the scripted server connects");

    let dir = tempdir().expect("temp dir");
    let terminal = TerminalTool::new_exact(dir.path(), NoopPermissionHandler, TokioGlobal)
        .await
        .expect("terminal tool initializes");
    let tools = tempdir().expect("tool dir");
    let cache = test_support::TempDir::new();
    let binary = test_support::compile_baseline_fixture(tools.path()).await;
    let distribution = test_support::fixture_distribution(&binary, "1.0.0").await;
    let (requests, captured) = async_channel::unbounded();
    let llm = ScriptedLlm {
        requests,
        turns: Arc::default(),
        changes,
        relisted,
    };
    let mut agent = TerminalAgentBuilder::new(llm, terminal)
        .max_iterations(2)
        .command_set(connection)
        .initialize_tools(&distribution, cache.path())
        .await
        .expect("tools prepare")
        .build();

    test_support::finish_run(agent.run("Use the server's tools", []))
        .await
        .expect("the run finishes");
    let first = captured.try_recv().expect("the first request");
    let second = captured
        .try_recv()
        .expect("the request after the terminal call");

    // The first request knew only `alpha`.
    assert!(prefix(&first).contains("- alpha: Runs alpha for the test\n"));
    assert!(!prefix(&first).contains("gamma"));
    assert_eq!(diff_alerts(&first).len(), 0);

    // `gamma` ran in the same session, before any rebuild.
    let transcript = second
        .messages()
        .iter()
        .map(Message::content)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains("gamma ran on the server"),
        "the added tool ran as a terminal command: {transcript}"
    );
    // The warm prefix is byte-identical, and the change arrived as one diff.
    assert_eq!(prefix(&second), prefix(&first));
    let alerts = diff_alerts(&second);
    assert_eq!(alerts.len(), 1, "exactly one diff alert");
    let alert = alerts[0].content();
    assert!(alert.starts_with("The table of terminal commands changed."));
    assert!(alert.contains("\n+- gamma: Runs gamma for the test\n"));
    assert!(
        !alert.contains("+- alpha"),
        "the diff holds only the change"
    );

    agent
        .compact(None)
        .await
        .expect("the compaction succeeds")
        .expect("the compaction ran");
    let _compaction = captured.try_recv().expect("the compaction request");
    test_support::finish_run(agent.run("Carry on", []))
        .await
        .expect("the second run finishes");
    let after = captured
        .try_recv()
        .expect("the request after the compaction");

    // The clean table lists `gamma`, with no diff and no repeated alert.
    assert!(
        prefix(&after)
            .contains("- alpha: Runs alpha for the test\n- gamma: Runs gamma for the test\n")
    );
    assert_eq!(diff_alerts(&after).len(), 0, "no diff after the compaction");
    assert!(captured.try_recv().is_err(), "nothing else was sent");
    server.abort();
}
