//! Test MCP client by connecting to Context7's remote MCP server.
//!
//! Run with: cargo run --example context7 -p aither-mcp

use aither_mcp::McpConnection;

#[tokio::main]
async fn main() {
    println!("Connecting to Context7 MCP server...");

    // Connect to Context7's remote HTTP server
    let mut conn = match McpConnection::http("https://mcp.context7.com/mcp").await {
        Ok(conn) => conn,
        Err(e) => {
            eprintln!("Failed to connect: {e}");
            return;
        }
    };

    println!("Connected! Server: {:?}", conn.server_name());
    println!("\nAvailable tools:");

    for def in conn.definitions() {
        println!("  - {}: {}", def.name(), def.description());
    }

    // Try calling the resolve-library-id tool
    println!("\n--- Testing resolve-library-id tool ---");
    let args = serde_json::json!({
        "libraryName": "tokio",
        "query": "async runtime"
    });

    match conn.call("resolve-library-id", args).await {
        Ok(result) => {
            println!("Result (is_error={}): ", result.is_error);
            for content in &result.content {
                println!("{content:?}");
            }
        }
        Err(e) => {
            eprintln!("Tool call failed: {e}");
        }
    }

    println!("\nDone!");
}
