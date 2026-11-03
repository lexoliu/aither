//! Error type for the Codex provider.

use std::time::Duration;

use aither_core::llm::{
    Classify, Failure, FailureKind,
    failure::zenwave::{failure as http_failure, io_is_network},
};

/// Errors from Codex OAuth flows and API calls.
#[derive(Debug, thiserror::Error)]
pub enum CodexError {
    /// HTTP transport or status error.
    #[error("HTTP request failed: {0}")]
    Http(#[from] zenwave::Error),
    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Error body returned by an endpoint (`detail` field).
    #[error("Codex API error: {0}")]
    Api(String),
    /// The user has not completed device authorization yet.
    #[error("device authorization is still pending")]
    AuthorizationPending,
    /// The OAuth loopback listener failed or the callback did not arrive.
    #[error("OAuth callback error: {0}")]
    Callback(String),
    /// The callback `state` did not match the issued one.
    #[error("OAuth state mismatch")]
    StateMismatch,
    /// I/O error (e.g. binding the loopback listener).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Error from the underlying `OpenAI` client.
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] aither_openai::OpenAIError),
}

/// Marks `failure` as a quota exhaustion when the error body is the
/// `usage_limit_reached` error the `ChatGPT` backend answers with once the
/// subscription's usage window is spent, and takes its reset time as the delay.
fn apply_usage_limit(error: &zenwave::Error, failure: &mut Failure) {
    #[derive(serde::Deserialize)]
    struct Body {
        error: Detail,
    }
    #[derive(serde::Deserialize)]
    struct Detail {
        #[serde(rename = "type")]
        error_type: Option<String>,
        /// When the limit resets, in Unix seconds.
        resets_at: Option<u64>,
    }
    let Some(detail) = error
        .response_body()
        .and_then(|body| serde_json::from_str::<Body>(body).ok())
        .map(|body| body.error)
    else {
        return;
    };
    if detail.error_type.as_deref() == Some("usage_limit_reached") {
        failure.kind = FailureKind::Quota;
        failure.retry_after = failure
            .retry_after
            .or_else(|| detail.resets_at.map(until_unix_seconds));
    }
}

/// Time from now until `resets_at` (Unix seconds), zero once it has passed.
fn until_unix_seconds(resets_at: u64) -> Duration {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the system clock is after the Unix epoch");
    Duration::from_secs(resets_at).saturating_sub(now)
}

impl Classify for CodexError {
    fn failure(&self) -> Failure {
        match self {
            Self::Http(err) => {
                let mut failure = http_failure(err);
                apply_usage_limit(err, &mut failure);
                failure
            }
            Self::OpenAI(inner @ aither_openai::OpenAIError::Http(err)) => {
                let mut failure = inner.failure();
                apply_usage_limit(err, &mut failure);
                failure
            }
            Self::OpenAI(err) => err.failure(),
            Self::Io(err) if io_is_network(err.kind()) => Failure::network(),
            Self::Json(_)
            | Self::Api(_)
            | Self::AuthorizationPending
            | Self::Callback(_)
            | Self::StateMismatch
            | Self::Io(_) => Failure::permanent(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use aither_core::llm::{Classify, FailureKind, failure::zenwave::http_error};
    use aither_openai::OpenAIError;

    use super::CodexError;

    fn http(status: u16, headers: &[(&str, &str)], body: &str) -> CodexError {
        CodexError::OpenAI(OpenAIError::Http(http_error(status, headers, body)))
    }

    #[test]
    fn server_errors_are_transient() {
        assert_eq!(http(500, &[], "").failure().kind, FailureKind::Transient);
        assert_eq!(http(529, &[], "").failure().kind, FailureKind::Transient);
    }

    #[test]
    fn usage_limit_is_quota_with_reset() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let body = format!(
            r#"{{"error":{{"type":"usage_limit_reached","resets_at":{}}}}}"#,
            now + 3600
        );
        let body = body.as_str();
        let failure = http(429, &[], body).failure();
        assert_eq!(failure.kind, FailureKind::Quota);
        let retry_after = failure.retry_after.expect("the reset time is read");
        assert!(
            retry_after > Duration::from_secs(3590) && retry_after <= Duration::from_secs(3600)
        );
        let direct = CodexError::Http(http_error(429, &[], body)).failure();
        assert_eq!(direct.kind, FailureKind::Quota);
        assert_eq!(http(429, &[], "").failure().kind, FailureKind::Transient);
    }

    #[test]
    fn unauthorized_is_permanent() {
        assert_eq!(http(401, &[], "").failure().kind, FailureKind::Permanent);
    }

    #[test]
    fn retry_after_seconds_and_date() {
        let failure = http(429, &[("retry-after", "4")], "").failure();
        assert_eq!(failure.retry_after, Some(Duration::from_secs(4)));
        let failure = http(503, &[("retry-after", "Fri, 01 Jan 2100 00:00:00 GMT")], "").failure();
        assert!(
            failure
                .retry_after
                .is_some_and(|d| d > Duration::from_secs(3600))
        );
    }

    #[test]
    fn refused_connection_is_network() {
        use std::io::{Error, ErrorKind};
        let error = CodexError::Io(Error::from(ErrorKind::ConnectionRefused));
        assert_eq!(error.failure().kind, FailureKind::Network);
    }
}
