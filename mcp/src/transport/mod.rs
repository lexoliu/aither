//! Transport layer for MCP communication.
//!
//! This module provides transport abstractions for sending and receiving
//! JSON-RPC messages over various channels (stdio, HTTP, child processes).

mod child;
mod http;
mod stdio;
mod traits;

pub use child::ChildProcessTransport;
pub use http::HttpTransport;
pub use stdio::StdioTransport;
pub use traits::{BidirectionalTransport, Transport};
