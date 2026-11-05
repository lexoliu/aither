//! The client's receive loop over the child-process and HTTP transports.

use std::time::Duration;

use async_channel::{Receiver, Sender};
use async_process::Command;
use executor_core::tokio::TokioGlobal;
use serde_json::json;

use super::{McpClient, McpHandler, ServerNotification};
use crate::protocol::{ErrorCode, McpError};
use crate::test_server::{serve, sse};
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
        ServerNotification::ToolsListChanged
    ));
    assert_ping_answered(&echoed(next(&notifications).await));
    assert_method_not_found(&echoed(next(&notifications).await));

    client.close().await;
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
