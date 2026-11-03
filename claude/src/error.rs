//! Error types for the Claude API client.

use core::fmt;

use aither_core::llm::{
    Classify, Failure, FailureKind,
    failure::zenwave::{body_failure, failure as http_failure, sse_failure},
};
use serde::Deserialize;
use zenwave::{BodyError, Error as ZenwaveError, sse::ParseError as SseParseError};

/// Errors that can arise when calling the Claude API.
#[derive(Debug)]
pub enum ClaudeError {
    /// HTTP layer errors.
    Http(ZenwaveError),
    /// Response body parsing failures.
    Body(BodyError),
    /// SSE parsing failures.
    Stream(SseParseError),
    /// JSON serialization/deserialization errors.
    Json(serde_json::Error),
    /// API contract violations or unsupported operations.
    Api(String),
    /// The API reported an error in an `error` event of an open stream.
    StreamEvent {
        /// The error's `type`, such as `overloaded_error`.
        error_type: String,
        /// The API's message.
        message: String,
    },
    /// The response stream sent nothing for the client's idle timeout.
    IdleTimeout {
        /// How long the stream stayed silent.
        idle: core::time::Duration,
    },
}

impl fmt::Display for ClaudeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "HTTP error: {err}"),
            Self::Body(err) => write!(f, "Body error: {err}"),
            Self::Stream(err) => write!(f, "SSE error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::Api(message) | Self::StreamEvent { message, .. } => write!(f, "{message}"),
            Self::IdleTimeout { idle } => {
                write!(f, "the response stream was silent for {idle:?}")
            }
        }
    }
}

impl std::error::Error for ClaudeError {}

/// The `error.type` of an Anthropic error body, `{"type":"error","error":{"type":...}}`.
fn body_error_type(body: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct Body {
        error: Detail,
    }
    #[derive(Deserialize)]
    struct Detail {
        #[serde(rename = "type")]
        error_type: String,
    }
    serde_json::from_str::<Body>(body)
        .ok()
        .map(|body| body.error.error_type)
}

/// Classifies by the API's `error.type`: `overloaded_error` and `api_error`
/// are transient, `billing_error` is an exhausted quota, the rest follow the
/// status.
fn kind_of_error_type(error_type: &str) -> Option<FailureKind> {
    match error_type {
        "overloaded_error" | "api_error" | "timeout_error" => Some(FailureKind::Transient),
        "billing_error" => Some(FailureKind::Quota),
        _ => None,
    }
}

impl Classify for ClaudeError {
    fn failure(&self) -> Failure {
        match self {
            Self::Http(err) => {
                let mut failure = http_failure(err);
                let by_type = err
                    .response_body()
                    .and_then(body_error_type)
                    .and_then(|error_type| kind_of_error_type(&error_type));
                let is_payment_required = err.response().is_some_and(|response| {
                    response.status() == zenwave::StatusCode::PAYMENT_REQUIRED
                });
                if let Some(kind) = by_type {
                    failure.kind = kind;
                } else if is_payment_required {
                    failure.kind = FailureKind::Quota;
                }
                failure
            }
            Self::Body(err) => body_failure(err),
            Self::Stream(err) => sse_failure(err),
            Self::StreamEvent { error_type, .. } => {
                Failure::new(kind_of_error_type(error_type).unwrap_or(FailureKind::Permanent))
            }
            Self::IdleTimeout { .. } => Failure::network(),
            Self::Json(_) | Self::Api(_) => Failure::permanent(),
        }
    }
}

impl From<ZenwaveError> for ClaudeError {
    fn from(value: ZenwaveError) -> Self {
        Self::Http(value)
    }
}

impl From<BodyError> for ClaudeError {
    fn from(value: BodyError) -> Self {
        Self::Body(value)
    }
}

impl From<SseParseError> for ClaudeError {
    fn from(value: SseParseError) -> Self {
        Self::Stream(value)
    }
}

impl From<serde_json::Error> for ClaudeError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use aither_core::{
        LanguageModel,
        llm::{Classify, FailureKind, failure::zenwave::http_error, oneshot},
    };
    use futures_lite::StreamExt;

    use super::ClaudeError;
    use crate::Claude;

    fn http(status: u16, headers: &[(&str, &str)], error_type: &str) -> ClaudeError {
        let body = format!(r#"{{"type":"error","error":{{"type":"{error_type}","message":"m"}}}}"#);
        ClaudeError::Http(http_error(status, headers, &body))
    }

    #[test]
    fn server_error_is_transient() {
        let failure = http(500, &[], "api_error").failure();
        assert_eq!(failure.kind, FailureKind::Transient);
    }

    #[test]
    fn overloaded_is_transient() {
        assert_eq!(
            http(529, &[], "overloaded_error").failure().kind,
            FailureKind::Transient
        );
        let event = ClaudeError::StreamEvent {
            error_type: "overloaded_error".into(),
            message: "Overloaded".into(),
        };
        assert_eq!(event.failure().kind, FailureKind::Transient);
    }

    #[test]
    fn rate_limit_is_transient_and_billing_is_quota() {
        assert_eq!(
            http(429, &[], "rate_limit_error").failure().kind,
            FailureKind::Transient
        );
        assert_eq!(
            http(402, &[], "billing_error").failure().kind,
            FailureKind::Quota
        );
    }

    #[test]
    fn auth_failure_is_permanent() {
        assert_eq!(
            http(401, &[], "authentication_error").failure().kind,
            FailureKind::Permanent
        );
        assert_eq!(
            http(400, &[], "invalid_request_error").failure().kind,
            FailureKind::Permanent
        );
    }

    #[test]
    fn retry_after_seconds() {
        let failure = http(429, &[("retry-after", "30")], "rate_limit_error").failure();
        assert_eq!(failure.retry_after, Some(Duration::from_secs(30)));
    }

    #[test]
    fn retry_after_http_date() {
        let failure = http(
            503,
            &[("retry-after", "Fri, 01 Jan 2100 00:00:00 GMT")],
            "api_error",
        )
        .failure();
        assert!(
            failure
                .retry_after
                .is_some_and(|delay| delay > Duration::from_secs(3600))
        );
    }

    #[tokio::test]
    async fn refused_connection_is_network() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let client = Claude::new("key").with_base_url(format!("http://127.0.0.1:{port}"));
        let stream = client.respond(oneshot("system", "hello"));
        futures_lite::pin!(stream);
        let error = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("connection is refused");
        assert_eq!(error.failure().kind, FailureKind::Network, "{error}");
    }

    /// A local server that opens an event stream, sends one `ping` and then
    /// goes silent while keeping the connection open.
    fn silent_sse_server() -> (u16, std::sync::mpsc::Sender<()>) {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (release, held) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut chunk = [0_u8; 4096];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = connection.read(&mut chunk).unwrap();
                if read == 0 {
                    return;
                }
                request.extend_from_slice(&chunk[..read]);
            }
            connection
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\nevent: ping\ndata: {\"type\": \"ping\"}\n\n",
                )
                .unwrap();
            // The connection stays open until the test drops `release`.
            let _ = held.recv();
        });
        (port, release)
    }

    #[tokio::test]
    async fn a_silent_stream_ends_with_a_network_failure() {
        let (port, _release) = silent_sse_server();
        let client = Claude::builder("key")
            .base_url(format!("http://127.0.0.1:{port}"))
            .idle_timeout(Duration::from_millis(300))
            .build();
        let stream = client.respond(oneshot("system", "hello"));
        futures_lite::pin!(stream);
        let error = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("the stream goes silent");
        assert!(matches!(error, ClaudeError::IdleTimeout { .. }), "{error}");
        assert_eq!(error.failure().kind, FailureKind::Network);
        assert!(stream.next().await.is_none(), "the stream ends");
    }

    #[test]
    fn the_idle_timeout_defaults_to_ninety_seconds() {
        assert_eq!(
            Claude::new("key").config().idle_timeout,
            Duration::from_secs(90)
        );
        assert_eq!(
            Claude::new("key")
                .with_idle_timeout(Duration::from_secs(5))
                .config()
                .idle_timeout,
            Duration::from_secs(5)
        );
    }
}
