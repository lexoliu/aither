// These types mirror the provider's error payload. Some fields are declared
// so a response deserializes faithfully even where this crate only reads a
// few of them.
#![allow(dead_code)]

use std::{fmt, time::Duration};

use aither_core::llm::{
    Classify, Failure, FailureKind,
    failure::zenwave::{body_failure, failure as http_failure, sse_failure},
};
use base64::DecodeError;
use serde::Deserialize;
use zenwave::{BodyError, Error as ZenwaveError, sse::ParseError as SseParseError};

/// Errors raised by the Gemini backend.
#[derive(Debug)]
pub enum GeminiError {
    /// HTTP transport errors.
    Http(ZenwaveError),
    /// Request/response body encoding failures.
    Body(BodyError),
    /// JSON serialization/deserialization problems.
    Json(serde_json::Error),
    /// Base64 decode failure (inline images/audio).
    Decode(DecodeError),
    /// API level errors or unsupported operations.
    Api(String),
    /// Other parsing errors.
    Parse(String),
    /// SSE stream failures.
    Stream(SseParseError),
    /// Rate limit exceeded (includes retry delay if available).
    RateLimit {
        /// What the API said about the limit.
        message: String,
        /// How long the API asked us to wait, when it said.
        retry_after: Option<Duration>,
    },
    /// A quota that resets on a long horizon, such as the daily one, is
    /// exhausted.
    QuotaExceeded {
        /// What the API said about the quota.
        message: String,
    },
}

/// Gemini API error response structure.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    /// The error the API reported, when it reported one.
    pub error: Option<ApiErrorDetail>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ApiErrorDetail {
    /// Numeric status code.
    pub code: Option<i32>,
    /// Human-readable description.
    pub message: Option<String>,
    /// Canonical status name, such as `RESOURCE_EXHAUSTED`.
    pub status: Option<String>,
    /// Structured details, used to distinguish quota failures from retries.
    pub details: Option<Vec<ApiErrorInfo>>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "@type")]
pub enum ApiErrorInfo {
    /// A quota was exceeded; names which one.
    #[serde(rename = "type.googleapis.com/google.rpc.QuotaFailure")]
    QuotaFailure(QuotaFailureInfo),
    /// The API suggested when to retry.
    #[serde(rename = "type.googleapis.com/google.rpc.RetryInfo")]
    RetryInfo(RetryInfoDetail),
    /// Any other detail payload.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct QuotaFailureInfo {
    /// The specific quotas that were exceeded.
    pub violations: Option<Vec<QuotaViolation>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
// The `quota_` prefix comes from the wire format and must match it.
#[allow(clippy::struct_field_names)]
pub struct QuotaViolation {
    /// Metric the quota is measured in.
    pub quota_metric: Option<String>,
    /// Identifier of the exceeded quota.
    pub quota_id: Option<String>,
    /// The limit that was hit.
    pub quota_value: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct RetryInfoDetail {
    /// How long the API suggests waiting before retrying.
    #[serde(rename = "retryDelay")]
    /// Retry delay duration.
    pub retry_delay: Option<String>,
}

impl ApiErrorResponse {
    /// Extract a user-friendly message from the error response.
    pub fn friendly_message(&self) -> String {
        let Some(error) = &self.error else {
            return "Unknown API error".to_string();
        };

        let base_msg = error.message.clone().unwrap_or_else(|| {
            error
                .status
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string())
        });

        // Check for quota violations
        if let Some(details) = &error.details {
            for detail in details {
                if let ApiErrorInfo::QuotaFailure(quota) = detail
                    && let Some(violations) = &quota.violations
                    && let Some(v) = violations.first()
                {
                    let quota_id = v.quota_id.as_deref().unwrap_or("unknown");
                    let quota_value = v.quota_value.as_deref().unwrap_or("?");
                    return format!(
                        "Rate limit exceeded: {} requests/min (limit: {})",
                        quota_id
                            .split('-')
                            .next()
                            .unwrap_or(quota_id)
                            .replace("GenerateRequestsPerMinutePerProjectPerModel", "requests"),
                        quota_value
                    );
                }
            }
        }

        base_msg
    }

    /// The retry delay from the error response, in the API's `"20s"` format.
    #[must_use]
    pub fn retry_delay(&self) -> Option<Duration> {
        let details = self.error.as_ref()?.details.as_ref()?;
        for detail in details {
            if let ApiErrorInfo::RetryInfo(info) = detail
                && let Some(delay) = &info.retry_delay
            {
                let seconds: f64 = delay.trim_end_matches('s').parse().ok()?;
                return Duration::try_from_secs_f64(seconds).ok();
            }
        }
        None
    }

    /// Whether the error says a per-day quota is exhausted, which a retry
    /// within the day cannot cure.
    #[must_use]
    pub fn is_daily_quota(&self) -> bool {
        let Some(details) = self.error.as_ref().and_then(|error| error.details.as_ref()) else {
            return false;
        };
        details.iter().any(|detail| {
            matches!(detail, ApiErrorInfo::QuotaFailure(quota)
                if quota.violations.iter().flatten().any(|violation|
                    violation.quota_id.as_deref().is_some_and(|id| id.contains("PerDay"))))
        })
    }
}

impl From<std::io::Error> for GeminiError {
    fn from(value: std::io::Error) -> Self {
        Self::Api(format!("IO error: {value}"))
    }
}

impl fmt::Display for GeminiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "{}", parse_http_error_message(err)),
            Self::Body(err) => write!(f, "Request failed: {err}"),
            Self::Json(err) => write!(f, "Invalid response format: {err}"),
            Self::Decode(err) => write!(f, "Decode error: {err}"),
            Self::Api(message) | Self::QuotaExceeded { message } => write!(f, "{message}"),
            Self::Parse(message) => write!(f, "Parse error: {message}"),
            Self::Stream(err) => write!(f, "Parse error: {err}"),
            Self::RateLimit {
                message,
                retry_after,
            } => {
                if let Some(delay) = retry_after {
                    write!(f, "{message} (retry after {}s)", delay.as_secs())
                } else {
                    write!(f, "{message}")
                }
            }
        }
    }
}

/// Parse HTTP error to extract a user-friendly message.
fn parse_http_error_message(err: &ZenwaveError) -> String {
    // Try to parse as Gemini API error response
    if let Some(body) = err.response_body()
        && let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(body)
    {
        return api_error.friendly_message();
    }

    // Fall back to status-based messages
    if let zenwave::Error::Http { status, .. } = err {
        return match status.as_u16() {
            400 => "Invalid request".to_string(),
            401 => "Authentication failed - check your API key".to_string(),
            403 => "Access denied - check your API key permissions".to_string(),
            404 => "Model not found".to_string(),
            429 => "Rate limit exceeded - please wait before retrying".to_string(),
            500 => "Server error - please try again".to_string(),
            502..=504 => "Service temporarily unavailable - please try again".to_string(),
            _ => format!("HTTP error {status}"),
        };
    }

    // Network/transport errors
    if err.is_network_error() {
        return "Network connection failed - check your internet connection".to_string();
    }
    if err.is_timeout() {
        return "Request timed out - please try again".to_string();
    }

    err.to_string()
}

impl std::error::Error for GeminiError {}

impl GeminiError {
    /// Convert a zenwave error to a `GeminiError`, detecting rate limits and
    /// exhausted quotas.
    pub(crate) fn from_http(err: ZenwaveError) -> Self {
        if let zenwave::Error::Http { status, .. } = &err
            && status.as_u16() == 429
        {
            let header_delay = http_failure(&err).retry_after;
            if let Some(body) = err.response_body()
                && let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(body)
            {
                let message = api_error.friendly_message();
                if api_error.is_daily_quota() {
                    return Self::QuotaExceeded { message };
                }
                return Self::RateLimit {
                    message,
                    retry_after: api_error.retry_delay().or(header_delay),
                };
            }
            return Self::RateLimit {
                message: "Rate limit exceeded".to_string(),
                retry_after: header_delay,
            };
        }
        Self::Http(err)
    }
}

impl Classify for GeminiError {
    fn failure(&self) -> Failure {
        match self {
            Self::Http(err) => {
                let mut failure = http_failure(err);
                if let Some(body) = err.response_body()
                    && let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(body)
                {
                    if api_error.is_daily_quota() {
                        failure.kind = FailureKind::Quota;
                    }
                    failure.retry_after = failure.retry_after.or_else(|| api_error.retry_delay());
                }
                failure
            }
            Self::Body(err) => body_failure(err),
            Self::Stream(err) => sse_failure(err),
            Self::Json(_) | Self::Decode(_) | Self::Api(_) | Self::Parse(_) => Failure::permanent(),
            Self::RateLimit { retry_after, .. } => {
                Failure::transient().with_retry_after(*retry_after)
            }
            Self::QuotaExceeded { .. } => Failure::quota(),
        }
    }
}

impl From<BodyError> for GeminiError {
    fn from(value: BodyError) -> Self {
        Self::Body(value)
    }
}

impl From<serde_json::Error> for GeminiError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<DecodeError> for GeminiError {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
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

    use super::GeminiError;
    use crate::Gemini;

    const PER_MINUTE: &str = r#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"20s"}]}}"#;
    const PER_DAY: &str = r#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}"#;

    fn http(status: u16, headers: &[(&str, &str)], body: &str) -> GeminiError {
        GeminiError::Http(http_error(status, headers, body))
    }

    fn from_http(status: u16, headers: &[(&str, &str)], body: &str) -> GeminiError {
        GeminiError::from_http(http_error(status, headers, body))
    }

    #[test]
    fn server_errors_are_transient() {
        assert_eq!(http(500, &[], "").failure().kind, FailureKind::Transient);
        assert_eq!(http(503, &[], "").failure().kind, FailureKind::Transient);
        assert_eq!(http(529, &[], "").failure().kind, FailureKind::Transient);
    }

    #[test]
    fn per_minute_429_is_transient_with_body_delay() {
        for error in [http(429, &[], PER_MINUTE), from_http(429, &[], PER_MINUTE)] {
            let failure = error.failure();
            assert_eq!(failure.kind, FailureKind::Transient);
            assert_eq!(failure.retry_after, Some(Duration::from_secs(20)));
        }
    }

    #[test]
    fn daily_429_is_quota() {
        assert_eq!(http(429, &[], PER_DAY).failure().kind, FailureKind::Quota);
        assert_eq!(
            from_http(429, &[], PER_DAY).failure().kind,
            FailureKind::Quota
        );
    }

    #[test]
    fn unauthorized_is_permanent() {
        assert_eq!(http(401, &[], "").failure().kind, FailureKind::Permanent);
        assert_eq!(
            from_http(401, &[], "").failure().kind,
            FailureKind::Permanent
        );
    }

    #[test]
    fn retry_after_header_seconds_and_date() {
        let failure = from_http(429, &[("retry-after", "9")], "").failure();
        assert_eq!(failure.retry_after, Some(Duration::from_secs(9)));
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
        let client = Gemini::new("key").with_base_url(format!("http://127.0.0.1:{port}"));
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
