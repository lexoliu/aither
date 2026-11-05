//! Scripted local HTTP servers for the HTTP transport's tests.

use std::fmt::Write as _;
use std::future::Future;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Read one HTTP request off `stream`: the raw request and its body.
async fn read_request(stream: &mut TcpStream) -> (String, String) {
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
    let chunked = head.contains("transfer-encoding: chunked");
    while if chunked {
        !buf.ends_with(b"0\r\n\r\n")
    } else {
        buf.len() < head_end + length
    } {
        let mut chunk = [0u8; 1024];
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "client closed mid-body");
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = if chunked {
        dechunk(&buf[head_end..])
    } else {
        buf[head_end..].to_vec()
    };
    (
        String::from_utf8_lossy(&buf).into_owned(),
        String::from_utf8_lossy(&body).into_owned(),
    )
}

/// Decode a `Transfer-Encoding: chunked` body.
fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    loop {
        let line_end = raw.windows(2).position(|w| w == b"\r\n").unwrap();
        let size = usize::from_str_radix(std::str::from_utf8(&raw[..line_end]).unwrap().trim(), 16)
            .unwrap();
        if size == 0 {
            return body;
        }
        body.extend_from_slice(&raw[line_end + 2..line_end + 2 + size]);
        raw = &raw[line_end + 2 + size + 2..];
    }
}

/// Answer 200 with `extra_headers` and `body`, then close the connection.
async fn write_reply(mut stream: TcpStream, extra_headers: &str, body: &str) {
    let reply = format!(
        "HTTP/1.1 200 OK\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(reply.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
}

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
            let (request, _) = read_request(&mut stream).await;
            log.lock().unwrap().push(request.to_lowercase());
            write_reply(stream, extra_headers, &body).await;
        }
    });
    (url, seen)
}

/// Serve every connection concurrently: `route` gets the JSON body of the
/// request and returns the `(headers, body)` to answer with, possibly later.
pub async fn serve_routes<F, Fut>(route: F) -> String
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = (&'static str, String)> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let route = Arc::new(route);
    tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let route = Arc::clone(&route);
            tokio::spawn(async move {
                let (_, body) = read_request(&mut stream).await;
                let (headers, body) = route(serde_json::from_str(&body).unwrap()).await;
                write_reply(stream, headers, &body).await;
            });
        }
    });
    url
}

/// Render `messages` as an SSE body.
pub fn sse(messages: &[serde_json::Value]) -> String {
    let mut stream = String::new();
    for m in messages {
        write!(stream, "event: message\ndata: {m}\n\n").unwrap();
    }
    stream
}
