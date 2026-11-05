//! A scripted local HTTP server for the HTTP transport's tests.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve one canned `(headers, body)` reply per connection, in order, and
/// record each raw request (lower-cased).
pub async fn serve(replies: Vec<(&'static str, String)>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    tokio::spawn(async move {
        for (extra_headers, body) in replies {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let head_end = loop {
                let mut chunk = [0u8; 1024];
                let n = stream.read(&mut chunk).await.unwrap();
                assert!(n > 0, "client closed before sending a full request");
                buf.extend_from_slice(&chunk[..n]);
                if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break at + 4;
                }
            };
            let head = String::from_utf8_lossy(&buf[..head_end]).to_lowercase();
            let length: usize = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .map_or(0, |v| v.trim().parse().unwrap());
            while buf.len() < head_end + length {
                let mut chunk = [0u8; 1024];
                let n = stream.read(&mut chunk).await.unwrap();
                buf.extend_from_slice(&chunk[..n]);
            }
            log.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&buf).to_lowercase());
            let reply = format!(
                "HTTP/1.1 200 OK\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(reply.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        }
    });
    (url, seen)
}

/// Render `messages` as an SSE body.
pub fn sse(messages: &[serde_json::Value]) -> String {
    let mut stream = String::new();
    for m in messages {
        write!(stream, "event: message\ndata: {m}\n\n").unwrap();
    }
    stream
}
