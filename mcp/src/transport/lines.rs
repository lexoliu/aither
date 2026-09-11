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
///
/// The line is assembled with `read_until`, which polls the reader once per
/// chunk. `fill_buf` is deliberately not used: it polls `poll_fill_buf` twice
/// and panics when the second poll is `Pending`, which is exactly what a
/// `BufReader<Unblock<Stdin>>` does after EOF, because every fresh poll hands
/// another read to the blocking pool.
pub(super) async fn read_message(
    reader: &mut (impl AsyncBufRead + Unpin),
    read_buf: &mut Vec<u8>,
) -> Result<Option<JsonRpcMessage>> {
    loop {
        let read = reader.read_until(b'\n', read_buf).await?;
        let complete = read_buf.last() == Some(&b'\n');
        if read == 0 && !complete {
            // EOF: whatever is staged is the final, unterminated line.
            if read_buf.iter().all(u8::is_ascii_whitespace) {
                read_buf.clear();
                return Ok(None);
            }
        } else if !complete {
            // The future was dropped mid-line and resumed; keep reading.
            continue;
        }
        let mut line = std::mem::take(read_buf);
        while matches!(line.last(), Some(b) if b.is_ascii_whitespace()) {
            line.pop();
        }
        if line.is_empty() {
            continue;
        }
        debug!("MCP RX: {}", String::from_utf8_lossy(&line));
        return serde_json::from_slice(&line).map(Some).map_err(Into::into);
    }
}
