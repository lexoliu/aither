//! The client's receive loop over the child-process and HTTP transports.

use std::time::Duration;

use async_channel::{Receiver, Sender};
use async_process::Command;
use executor_core::tokio::TokioGlobal;
use serde_json::json;

use super::{McpClient, McpConnection, McpHandler, ServerNotification};
use crate::protocol::{ErrorCode, McpError};
use crate::test_server::{serve, serve_routes, sse};
use crate::transport::{ChildProcessTransport, HttpTransport};

/// Forwards every notification to the test.
struct Recorder(Sender<ServerNotification>);

impl McpHandler for Recorder {
    async fn notification(&self, notification: ServerNotification) {
        self.0.send(notification).await.unwrap();
    }
}

fn recorder() -> (Recorder, Receiver<ServerNotification>) {
    let (sender, receiver) = async_channel::unbounded();
    (Recorder(sender), receiver)
}

/// Wait for the next notification, bounded so a regression fails the test
/// instead of hanging it.
async fn next(notifications: &Receiver<ServerNotification>) -> ServerNotification {
    tokio::time::timeout(Duration::from_secs(10), notifications.recv())
        .await
        .expect("no notification within 10s")
        .unwrap()
}

/// The `params` of a `test/echo` notification.
fn echoed(notification: ServerNotification) -> serde_json::Value {
    let ServerNotification::Other(echo) = notification else {
        panic!("expected a test/echo notification, got {notification:?}");
    };
    assert_eq!(echo.method, "test/echo");
    echo.params.unwrap()
}

fn assert_ping_answered(response: &serde_json::Value) {
    assert_eq!(response["id"], "ping-1");
    assert_eq!(response["result"], json!({}));
}

fn assert_method_not_found(response: &serde_json::Value) {
    assert_eq!(response["id"], "sampling-1");
    assert_eq!(response["error"]["code"], ErrorCode::METHOD_NOT_FOUND.0);
}

fn script(name: &str) -> Command {
    let mut command = Command::new("sh");
    command.arg(format!(
        "{}/src/client/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ));
    command
}

#[tokio::test]
async fn child_process_server_messages_interleave_with_a_request() {
    let (handler, notifications) = recorder();
    let transport = ChildProcessTransport::from_command(&mut script("interleaving_server.sh"))
        .expect("spawn sh");
    let client = McpClient::connect(&TokioGlobal, transport, handler)
        .await
        .unwrap();
    assert_eq!(client.server_info().name, "script");

    // The request gets its response although the server pings, notifies and
    // sends an unadvertised request first.
    assert!(client.list_tools().await.unwrap().is_empty());

    assert!(matches!(
        next(&notifications).await,
        ServerNotification::ResourcesListChanged
    ));
    assert_ping_answered(&echoed(next(&notifications).await));
    assert_method_not_found(&echoed(next(&notifications).await));

    client.close().await;
}

#[tokio::test]
async fn child_process_tool_list_changed_refreshes_the_connection_tools() {
    let (handler, notifications) = recorder();
    let transport = ChildProcessTransport::from_command(&mut script("changing_tools_server.sh"))
        .expect("spawn sh");
    let conn = McpConnection::connect(&TokioGlobal, transport, handler)
        .await
        .unwrap();
    let names = |conn: &McpConnection| {
        conn.mcp_definitions()
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&conn), ["alpha", "beta"]);
    let before = conn.mcp_definitions();

    // The server answers this call, then announces a different tool list.
    conn.call("alpha", json!({})).await.unwrap();
    assert!(matches!(
        next(&notifications).await,
        ServerNotification::ToolsListChanged
    ));

    // The re-listing runs beside the handler; wait for it, bounded.
    tokio::time::timeout(Duration::from_secs(10), async {
        while !conn.has_tool("gamma") {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the tool list was not refreshed within 10s");

    assert_eq!(names(&conn), ["beta", "gamma"]);
    assert_eq!(conn.definitions().len(), 2);
    // A list taken before the refresh is unchanged, never a mixture.
    assert_eq!(before.len(), 2);
    assert_eq!(before[0].name, "alpha");

    // The removed tool is no longer offered or called.
    assert!(!conn.has_tool("alpha"));
    assert!(matches!(
        conn.call("alpha", json!({})).await,
        Err(McpError::UnknownTool(name)) if name == "alpha"
    ));

    conn.close().await;
}

#[tokio::test]
async fn a_tool_added_by_list_changed_is_a_terminal_command_at_once() {
    let transport = ChildProcessTransport::from_command(&mut script("changing_tools_server.sh"))
        .expect("spawn sh");
    let conn = McpConnection::connect(&TokioGlobal, transport, ())
        .await
        .unwrap();
    let mut builder = aither_sandbox::ToolRegistryBuilder::new();
    builder.command_set(conn);
    // The fixture's results are empty, so nothing is saved to the directory.
    let outputs = aither_sandbox::OutputStore::new(
        std::env::temp_dir(),
        aither_core::AnyIdAllocator::default(),
    )
    .await
    .unwrap();
    let registry = builder.build(outputs);
    let mut names = registry.registered_tool_names();
    names.sort();
    assert_eq!(names, ["alpha", "beta"]);

    // The server answers this call, then announces a different tool list.
    assert!(registry.query_tool_handler("alpha", &[]).await.ok);
    tokio::time::timeout(Duration::from_secs(10), async {
        while !registry.is_tool_configured("gamma") {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the tool list was not refreshed within 10s");

    // The same registry, never rebuilt, follows the server's list.
    let mut names = registry.registered_tool_names();
    names.sort();
    assert_eq!(names, ["beta", "gamma"]);
    assert!(registry.tool_help("gamma").is_some());
    let gone = registry.query_tool_handler("alpha", &[]).await;
    assert_eq!(
        gone.error.as_deref(),
        Some("unknown command 'alpha'. Run 'help' to see available commands.")
    );
}

#[tokio::test]
async fn child_process_exit_fails_the_pending_request() {
    let transport =
        ChildProcessTransport::from_command(&mut script("exiting_server.sh")).expect("spawn sh");
    let client = McpClient::connect(&TokioGlobal, transport, ())
        .await
        .unwrap();

    let error = client.list_tools().await.unwrap_err();
    let McpError::Transport(message) = &error else {
        panic!("expected a transport error naming the exit, got {error:?}");
    };
    assert!(message.contains("server process"), "{message}");

    // The loop is gone: later requests fail too, and closing is a no-op.
    assert!(client.list_tools().await.is_err());
    client.close().await;
}

#[tokio::test]
async fn http_slow_reply_does_not_stall_other_requests_or_server_messages() {
    use std::sync::Arc;

    use tokio::sync::Notify;

    let slow_arrived = Arc::new(Notify::new());
    let release_slow = Arc::new(Notify::new());
    let url = {
        let slow_arrived = slow_arrived.clone();
        let release_slow = release_slow.clone();
        serve_routes(move |request| {
            let slow_arrived = slow_arrived.clone();
            let release_slow = release_slow.clone();
            async move {
                let reply = |result: serde_json::Value| {
                    json!({"jsonrpc": "2.0", "id": request["id"], "result": result}).to_string()
                };
                match request["method"].as_str().unwrap() {
                    "initialize" => (
                        "Content-Type: application/json\r\n",
                        reply(json!({
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "serverInfo": {"name": "http-script"}
                        })),
                    ),
                    // Plain-JSON reply, held back until the test lets go.
                    "tools/call" => {
                        slow_arrived.notify_one();
                        release_slow.notified().await;
                        (
                            "Content-Type: application/json\r\n",
                            reply(json!({"content": [], "isError": false})),
                        )
                    }
                    // SSE reply carrying a server message before the result.
                    "tools/list" => (
                        "Content-Type: text/event-stream\r\n",
                        sse(&[
                            json!({"jsonrpc": "2.0", "method": "notifications/resources/list_changed"}),
                            serde_json::from_str(&reply(json!({"tools": []}))).unwrap(),
                        ]),
                    ),
                    _ => ("", String::new()),
                }
            }
        })
        .await
    };

    let (handler, notifications) = recorder();
    let client = McpClient::connect(&TokioGlobal, HttpTransport::new(url), handler)
        .await
        .unwrap();

    let both = async {
        tokio::join!(client.call_tool("slow", json!({})), async {
            // Only once the slow request is at the server, issue another.
            slow_arrived.notified().await;
            let tools = client.list_tools().await.unwrap();
            let first = next(&notifications).await;
            // The slow request is still outstanding; let it finish.
            release_slow.notify_one();
            (tools, first)
        })
    };
    let (slow, (tools, first)) = tokio::time::timeout(Duration::from_secs(10), both)
        .await
        .expect("a slow reply stalled the connection");

    assert!(tools.is_empty());
    assert!(matches!(first, ServerNotification::ResourcesListChanged));
    assert!(!slow.unwrap().is_error);
}

#[tokio::test]
async fn http_server_messages_interleave_with_a_request() {
    let initialize = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "serverInfo": {"name": "http-script"}
        }
    })
    .to_string();
    let list = sse(&[
        json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": 7, "progress": 1.0}}),
        json!({"jsonrpc": "2.0", "id": "ping-1", "method": "ping"}),
        json!({"jsonrpc": "2.0", "id": "sampling-1", "method": "sampling/createMessage", "params": {}}),
        json!({"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}),
    ]);
    let (url, seen) = serve(vec![
        ("Content-Type: application/json\r\n", initialize),
        // The `notifications/initialized` acknowledgement.
        ("", String::new()),
        ("Content-Type: text/event-stream\r\n", list),
        // Acknowledgements of the ping answer and of the rejection.
        ("", String::new()),
        ("", String::new()),
    ])
    .await;

    let (handler, notifications) = recorder();
    let client = McpClient::connect(&TokioGlobal, HttpTransport::new(url), handler)
        .await
        .unwrap();
    assert!(client.list_tools().await.unwrap().is_empty());

    let ServerNotification::Progress(progress) = next(&notifications).await else {
        panic!("expected the progress notification");
    };
    assert!((progress.progress - 1.0).abs() < f64::EPSILON);

    // The answers were POSTed back before the response was routed.
    let seen = seen.lock().unwrap();
    // The body may be chunk-framed; the JSON object is what lies between the
    // outermost braces.
    let body = |request: &str| {
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        let object = &body[body.find('{').unwrap()..=body.rfind('}').unwrap()];
        serde_json::from_str::<serde_json::Value>(object).unwrap()
    };
    assert_ping_answered(&body(&seen[3]));
    assert_method_not_found(&body(&seen[4]));
}
