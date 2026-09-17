//! Smoke test for the ACP client against a real agent.
//!
//! Spawns `devin acp`, initializes, opens a session in `/tmp`, sets mode
//! `bypass` and config option `model=swe-2-max`, then prompts the agent to
//! reply with a single word, printing streamed text and the stop reason.
//!
//! Requires an authenticated `devin` CLI on `PATH`.
//!
//! ```console
//! cargo run -p aither-acp --example devin_smoke
//! ```

use std::sync::Mutex;

use aither_acp::{
    AcpClient, ClientError, ClientHandler, ContentBlock, PromptParams, RequestPermissionOutcome,
    RequestPermissionParams, RequestPermissionResult, SessionNewParams, SessionNotification,
    SessionSetConfigOptionParams, SessionSetModeParams, SessionUpdate, TextContent, vendor,
};
use aither_mcp::protocol::{JsonRpcError, JsonRpcNotification};

/// Handler that prints streamed text and auto-approves permission requests.
#[derive(Default)]
struct SmokeHandler {
    /// Accumulated agent message text.
    text: Mutex<String>,
}

impl ClientHandler for SmokeHandler {
    fn session_update(
        &self,
        notification: SessionNotification,
    ) -> impl std::future::Future<Output = ()> + Send {
        if let SessionUpdate::AgentMessageChunk(chunk) = &notification.update
            && let ContentBlock::Text(content) = &chunk.content
        {
            print!("{}", content.text);
            self.text
                .lock()
                .expect("text poisoned")
                .push_str(&content.text);
        }
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
            .map_or_else(|| "deny".to_string(), |o| o.option_id.clone());
        eprintln!("[permission] {} -> {option_id}", params.tool_call.title);
        async move {
            Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Selected { option_id },
                meta: None,
            })
        }
    }

    fn notification(
        &self,
        notification: JsonRpcNotification,
    ) -> impl std::future::Future<Output = ()> + Send {
        // Decode devin's private `_cognition.ai/*` notifications.
        match vendor::devin::notification(&notification) {
            Some(vendor::devin::DevinNotification::Output(output)) => {
                eprintln!("[devin:{}] {}", output.channel, output.message);
            }
            Some(vendor::devin::DevinNotification::McpServersChanged(_)) => {
                eprintln!("[devin] MCP servers changed");
            }
            Some(vendor::devin::DevinNotification::Other { method, .. }) => {
                eprintln!("[devin] {method}");
            }
            None => eprintln!("[notification] {}", notification.method),
        }
        std::future::ready(())
    }
}

#[tokio::main]
async fn main() -> Result<(), ClientError> {
    let handler = SmokeHandler::default();
    let (client, connection) = AcpClient::spawn(
        "devin",
        &["acp"],
        std::iter::empty::<(String, String)>(),
        "/tmp",
        handler,
    )?;
    let connection = tokio::spawn(connection);

    let init = client.initialize().await?;
    eprintln!(
        "[initialize] agent={} protocol={} load_session={}",
        init.agent_info.as_ref().map_or("?", |i| i.name.as_str()),
        init.protocol_version,
        init.agent_capabilities.load_session,
    );
    if let Some(path) = vendor::devin::mcp_config_path(&init) {
        eprintln!("[initialize] mcpConfigPath={path}");
    }

    let session = client.new_session(SessionNewParams::new("/tmp")).await?;
    eprintln!("[session] {}", session.session_id);

    client
        .set_mode(SessionSetModeParams::new(&session.session_id, "bypass"))
        .await?;
    client
        .set_config_option(SessionSetConfigOptionParams::new(
            &session.session_id,
            "model",
            "swe-2-max",
        ))
        .await?;

    eprintln!("[prompt] streaming response:");
    let result = client
        .prompt(PromptParams::new(
            &session.session_id,
            vec![ContentBlock::Text(TextContent {
                text: "Reply with the single word pong.".to_string(),
                annotations: None,
                meta: None,
            })],
        ))
        .await?;

    eprintln!("\n[done] stop_reason={:?}", result.stop_reason);
    eprintln!(
        "[done] full text: {}",
        client.handler().text.lock().unwrap()
    );

    client.close();
    connection.await.expect("connection task panicked");
    Ok(())
}
