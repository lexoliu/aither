//! MCP protocol types and JSON-RPC message definitions.

mod error;
mod message;
mod types;

pub use error::{ErrorCode, JsonRpcError, McpError};
pub use message::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId,
};
pub use types::{
    CallToolParams, CallToolResult, ClientCapabilities, Content, ImageContent, InitializeParams,
    InitializeResult, ListToolsResult, McpToolDefinition, PromptMessage, Resource,
    ResourceContents, ServerCapabilities, ServerInfo, TextContent, ToolsCapability,
    PROTOCOL_VERSION,
};
