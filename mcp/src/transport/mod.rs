//! Transport layer for MCP communication.
//!
//! This module provides transport abstractions for sending and receiving
//! JSON-RPC messages over various channels (stdio, HTTP, child processes).

#[cfg(feature = "client")]
mod child;
mod duplex;
#[cfg(feature = "http")]
mod http;
mod lines;
mod stdio;
mod traits;

#[cfg(feature = "client")]
pub use child::ChildProcessTransport;
pub use duplex::DuplexTransport;
#[cfg(feature = "http")]
pub use http::HttpTransport;
pub use stdio::StdioTransport;
pub use traits::{BidirectionalTransport, Transport};
