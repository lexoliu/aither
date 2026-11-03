//! Error types for GitHub Copilot API integration.

use std::time::Duration;

use aither_core::llm::{
    Classify, Failure, FailureKind,
    failure::zenwave::{body_failure, failure as http_failure, sse_failure},
};

/// Errors that can arise when calling the GitHub Copilot API.
#[derive(Debug, thiserror::Error)]
pub enum CopilotError {
    /// HTTP layer errors from zenwave.
    #[error("HTTP error: {0}")]
    Http(#[from] zenwave::Error),

    /// Response body parsing failures.
    #[error("Body error: {0}")]
    Body(#[from] zenwave::BodyError),

    /// SSE parsing failures.
    #[error("SSE error: {0}")]
    Stream(#[from] zenwave::sse::ParseError),

    /// JSON serialization/deserialization errors.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// API contract violations or unsupported operations.
    #[error("{0}")]
    Api(String),

    /// OAuth authorization pending (user hasn't completed flow yet).
    #[error("Authorization pending")]
    AuthorizationPending,

    /// OAuth device code expired.
    #[error("Device code expired")]
    DeviceCodeExpired,

    /// OAuth access denied by user.
    #[error("Access denied")]
    AccessDenied,

    /// Rate limit exceeded (HTTP 429).
    #[error("Rate limit exceeded: {message}{}", retry_after.map(|d| format!(" (retry after {}s)", d.as_secs())).unwrap_or_default())]
    RateLimit {
        /// Message from the API.
        message: String,
        /// Suggested retry delay from Retry-After header.
        retry_after: Option<Duration>,
    },

    /// Server error (HTTP 5xx).
    #[error("Server error {status}: {message}")]
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Error message.
        message: String,
    },

    /// Request timed out.
    #[error("Request timed out")]
    Timeout,
}

impl CopilotError {
    /// HTTP status code carried by transport-level errors, if any.
    ///
    /// Keeps status-code inspection inside this crate so callers do not need
    /// to name the underlying HTTP client's error type.
    #[must_use]
    pub const fn status(&self) -> Option<u16> {
        match self {
            Self::Http(zenwave::Error::Http { status, .. }) => Some(status.as_u16()),
            Self::ServerError { status, .. } => Some(*status),
            _ => None,
        }
    }
}

/// The `error.code` of an OpenAI-style error body.
fn body_error_code(body: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Body {
        error: Detail,
    }
    #[derive(serde::Deserialize)]
    struct Detail {
        code: Option<String>,
    }
    serde_json::from_str::<Body>(body)
        .ok()
        .and_then(|body| body.error.code)
}

impl Classify for CopilotError {
    fn failure(&self) -> Failure {
        match self {
            Self::Http(err) => {
                let mut failure = http_failure(err);
                let payment_required = err.response().is_some_and(|response| {
                    response.status() == zenwave::StatusCode::PAYMENT_REQUIRED
                });
                let quota_code = err
                    .response_body()
                    .and_then(body_error_code)
                    .is_some_and(|code| code == "quota_exceeded" || code == "insufficient_quota");
                if payment_required || quota_code {
                    failure.kind = FailureKind::Quota;
                }
                failure
            }
            Self::Body(err) => body_failure(err),
            Self::Stream(err) => sse_failure(err),
            Self::RateLimit { retry_after, .. } => {
                Failure::transient().with_retry_after(*retry_after)
            }
            Self::ServerError { .. } => Failure::transient(),
            Self::Timeout => Failure::network(),
            Self::Json(_)
            | Self::Api(_)
            | Self::AuthorizationPending
            | Self::DeviceCodeExpired
            | Self::AccessDenied => Failure::permanent(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use aither_core::{
        LanguageModel,
        llm::{Classify, FailureKind, failure::zenwave::http_error, oneshot},
    };
    use futures_lite::StreamExt;

    use super::CopilotError;
    use crate::Copilot;

    fn http(status: u16, headers: &[(&str, &str)], body: &str) -> CopilotError {
        CopilotError::Http(http_error(status, headers, body))
    }

    #[test]
    fn server_errors_are_transient() {
        assert_eq!(http(500, &[], "").failure().kind, FailureKind::Transient);
        assert_eq!(http(529, &[], "").failure().kind, FailureKind::Transient);
    }

    #[test]
    fn quota_and_rate_limit() {
        let quota = r#"{"error":{"code":"quota_exceeded","message":"m"}}"#;
        assert_eq!(http(429, &[], quota).failure().kind, FailureKind::Quota);
        assert_eq!(http(402, &[], "").failure().kind, FailureKind::Quota);
        assert_eq!(http(429, &[], "").failure().kind, FailureKind::Transient);
    }

    #[test]
    fn unauthorized_is_permanent() {
        assert_eq!(http(401, &[], "").failure().kind, FailureKind::Permanent);
    }

    #[test]
    fn retry_after_seconds_and_date() {
        let failure = http(429, &[("retry-after", "12")], "").failure();
        assert_eq!(failure.retry_after, Some(Duration::from_secs(12)));
        let failure = http(503, &[("retry-after", "Fri, 01 Jan 2100 00:00:00 GMT")], "").failure();
        assert!(
            failure
                .retry_after
                .is_some_and(|d| d > Duration::from_secs(3600))
        );
    }

    #[tokio::test]
    async fn refused_connection_is_network() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let client = Copilot::new("token").with_base_url(format!("http://127.0.0.1:{port}"));
        let stream = client.respond(oneshot("system", "hello"));
        futures_lite::pin!(stream);
        let error = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("connection is refused");
        assert_eq!(error.failure().kind, FailureKind::Network, "{error}");
    }
}
