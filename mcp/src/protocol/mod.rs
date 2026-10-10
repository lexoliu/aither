//! MCP protocol types and JSON-RPC message definitions.

mod error;
mod message;
mod types;

pub use error::{ErrorCode, JsonRpcError, McpError};
pub use message::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId,
};
pub use types::{
    CallToolParams, CallToolResult, CancelledParams, ClientCapabilities, Content, EmbeddedResource,
    ImageContent, InitializeParams, InitializeResult, ListToolsResult, McpToolDefinition,
    PROTOCOL_VERSION, ProgressNotificationParams, ProgressToken, PromptMessage, RequestMeta,
    Resource, ResourceContent, ResourceContents, ServerCapabilities, ServerInfo, TextContent,
    ToolsCapability,
};
