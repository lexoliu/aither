//! Newline-delimited JSON-RPC message reading shared by the transports.

use futures_lite::AsyncBufReadExt;
use futures_lite::io::AsyncBufRead;
use tracing::debug;

use super::traits::Result;
use crate::protocol::JsonRpcMessage;

/// Read one newline-delimited JSON-RPC message from `reader`.
///
/// Bytes are staged in `read_buf` between calls so a future dropped mid-line
/// loses nothing: the next call resumes the partial message, which makes the
/// transports' `recv` safe to race in a `select`-style loop. Returns `None`
/// on EOF, skips blank lines, and surfaces deserialization errors without
/// consuming the following message.
pub(super) async fn read_message(
    reader: &mut (impl AsyncBufRead + Unpin),
    read_buf: &mut Vec<u8>,
) -> Result<Option<JsonRpcMessage>> {
    loop {
        if let Some(end) = read_buf.iter().position(|b| *b == b'\n') {
            let mut line: Vec<u8> = read_buf.drain(..=end).collect();
            while matches!(line.last(), Some(b) if b.is_ascii_whitespace()) {
                line.pop();
            }
            if line.is_empty() {
                continue;
            }
            debug!("MCP RX: {}", String::from_utf8_lossy(&line));
            return serde_json::from_slice(&line).map(Some).map_err(Into::into);
        }

        let chunk = reader.fill_buf().await?;
        if chunk.is_empty() {
            if read_buf.is_empty() {
                return Ok(None);
            }
            // Final line without a trailing newline.
            let line = std::mem::take(read_buf);
            debug!("MCP RX: {}", String::from_utf8_lossy(&line));
            return serde_json::from_slice(&line).map(Some).map_err(Into::into);
        }
        let len = chunk.len();
        read_buf.extend_from_slice(chunk);
        reader.consume(len);
    }
}
