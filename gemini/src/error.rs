use std::fmt;

use base64::DecodeError;
use serde::Deserialize;
use zenwave::{BodyError, Error as ZenwaveError};

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
    /// SSE stream parsing errors.
    Parse(String),
    /// Rate limit exceeded (includes retry delay if available).
    RateLimit {
        message: String,
        retry_after_secs: Option<u64>,
    },
}

/// Gemini API error response structure.
#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
    pub error: Option<ApiErrorDetail>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorDetail {
    pub code: Option<i32>,
    pub message: Option<String>,
    pub status: Option<String>,
    pub details: Option<Vec<ApiErrorInfo>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiErrorInfo {
    QuotaFailure(QuotaFailureInfo),
    RetryInfo(RetryInfoDetail),
    Other(serde_json::Value),
}

#[derive(Debug, Deserialize)]
pub(crate) struct QuotaFailureInfo {
    #[serde(rename = "@type")]
    pub type_url: Option<String>,
    pub violations: Option<Vec<QuotaViolation>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct QuotaViolation {
    pub quota_metric: Option<String>,
    pub quota_id: Option<String>,
    pub quota_value: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RetryInfoDetail {
    #[serde(rename = "@type")]
    pub type_url: Option<String>,
    #[serde(rename = "retryDelay")]
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
                if let ApiErrorInfo::QuotaFailure(quota) = detail {
                    if quota
                        .type_url
                        .as_deref()
                        .map_or(false, |t| t.contains("QuotaFailure"))
                    {
                        if let Some(violations) = &quota.violations {
                            if let Some(v) = violations.first() {
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
                }
            }
        }

        base_msg
    }

    /// Extract retry delay in seconds from the error response.
    pub fn retry_delay_secs(&self) -> Option<u64> {
        let details = self.error.as_ref()?.details.as_ref()?;
        for detail in details {
            if let ApiErrorInfo::RetryInfo(info) = detail {
                if let Some(delay) = &info.retry_delay {
                    // Parse "20s" format
                    return delay.trim_end_matches('s').parse().ok();
                }
            }
        }
        None
    }
}

impl fmt::Display for GeminiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "{}", parse_http_error_message(err)),
            Self::Body(err) => write!(f, "Request failed: {err}"),
            Self::Json(err) => write!(f, "Invalid response format: {err}"),
            Self::Decode(err) => write!(f, "Decode error: {err}"),
            Self::Api(message) => write!(f, "{message}"),
            Self::Parse(message) => write!(f, "Parse error: {message}"),
            Self::RateLimit {
                message,
                retry_after_secs,
            } => {
                if let Some(secs) = retry_after_secs {
                    write!(f, "{message} (retry after {secs}s)")
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
    if let Some(body) = err.response_body() {
        if let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(body) {
            return api_error.friendly_message();
        }
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
            502 | 503 | 504 => "Service temporarily unavailable - please try again".to_string(),
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
    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http(err) => {
                // Retry on rate limit, server errors, network errors, and timeouts
                if let zenwave::Error::Http { status, .. } = err {
                    return status.as_u16() == 429
                        || status.as_u16() >= 500
                        || status.as_u16() == 408;
                }
                err.is_network_error() || err.is_timeout()
            }
            Self::RateLimit { .. } => true,
            _ => false,
        }
    }

    /// Get suggested retry delay in seconds.
    pub fn retry_delay_secs(&self) -> Option<u64> {
        match self {
            Self::RateLimit {
                retry_after_secs, ..
            } => *retry_after_secs,
            Self::Http(err) => {
                // Try to parse from response body
                if let Some(body) = err.response_body() {
                    if let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(body) {
                        return api_error.retry_delay_secs();
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Convert a zenwave error to a GeminiError, detecting rate limits.
    pub(crate) fn from_http(err: ZenwaveError) -> Self {
        // Check if it's a rate limit error
        if let zenwave::Error::Http { status, .. } = &err {
            if status.as_u16() == 429 {
                if let Some(body) = err.response_body() {
                    if let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(body) {
                        return Self::RateLimit {
                            message: api_error.friendly_message(),
                            retry_after_secs: api_error.retry_delay_secs(),
                        };
                    }
                }
                return Self::RateLimit {
                    message: "Rate limit exceeded".to_string(),
                    retry_after_secs: None,
                };
            }
        }
        Self::Http(err)
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
