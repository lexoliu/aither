//! Classification of `zenwave` HTTP errors, shared by every HTTP provider.

extern crate std;

use core::time::Duration;
use std::time::SystemTime;

use ::zenwave::{Error, sse::ParseError as SseParseError};

use super::{Failure, FailureKind};

/// Classifies an HTTP status the way every provider agrees on.
///
/// 408, 429 and every 5xx (529 included) are [`FailureKind::Transient`]; every
/// other status is [`FailureKind::Permanent`]. A 429 that is a quota
/// exhaustion is a provider-specific fact, so providers upgrade it to
/// [`FailureKind::Quota`] from their own error body.
#[must_use]
pub const fn status_kind(status: u16) -> FailureKind {
    match status {
        408 | 429 | 500..=599 => FailureKind::Transient,
        _ => FailureKind::Permanent,
    }
}

/// Parses a `Retry-After` header value: delay seconds or an HTTP-date.
///
/// A date in the past yields a zero delay. `now` is a parameter so callers and
/// tests control the clock.
#[must_use]
pub fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let date = httpdate::parse_http_date(value).ok()?;
    Some(date.duration_since(now).unwrap_or(Duration::ZERO))
}

/// The delay in the `Retry-After` header of the error's response, if any.
#[must_use]
pub fn retry_after(error: &Error) -> Option<Duration> {
    let value = error
        .response()?
        .headers()
        .get(::zenwave::header::RETRY_AFTER)?
        .to_str()
        .ok()?;
    parse_retry_after(value, SystemTime::now())
}

/// Classifies a `zenwave` error from its variant, status and headers.
///
/// - HTTP errors by status ([`status_kind`]) with the `Retry-After` header.
/// - Transport, TLS, proxy and timeout errors, and I/O errors of a
///   connection-level kind, as [`FailureKind::Network`].
/// - Everything else as [`FailureKind::Permanent`].
#[must_use]
pub fn failure(error: &Error) -> Failure {
    match error {
        Error::Http { status, .. } => {
            Failure::new(status_kind(status.as_u16())).with_retry_after(retry_after(error))
        }
        Error::Transport(_) | Error::Tls(_) | Error::Proxy(_) | Error::Timeout => {
            Failure::network()
        }
        Error::Io(error) if io_is_network(error.kind()) => Failure::network(),
        Error::BodyParse(error) => body_failure(error),
        _ => Failure::permanent(),
    }
}

/// Classifies a failure to read a response body.
///
/// An I/O error while reading means the connection broke mid-response.
#[must_use]
pub fn body_failure(error: &::zenwave::BodyError) -> Failure {
    match error {
        ::zenwave::BodyError::Io(error) if io_is_network(error.kind()) => Failure::network(),
        _ => Failure::permanent(),
    }
}

/// Classifies a failure of the SSE stream.
///
/// The stream's body ending abnormally is a lost connection; malformed
/// events are not.
#[must_use]
pub const fn sse_failure(error: &SseParseError) -> Failure {
    match error {
        SseParseError::BodyError(_) => Failure::network(),
        _ => Failure::permanent(),
    }
}

/// Whether an I/O error kind means the connection broke or never formed.
#[must_use]
pub const fn io_is_network(kind: std::io::ErrorKind) -> bool {
    use std::io::ErrorKind::{
        BrokenPipe, ConnectionAborted, ConnectionRefused, ConnectionReset, HostUnreachable,
        NetworkDown, NetworkUnreachable, NotConnected, TimedOut, UnexpectedEof,
    };
    matches!(
        kind,
        BrokenPipe
            | ConnectionAborted
            | ConnectionRefused
            | ConnectionReset
            | HostUnreachable
            | NetworkDown
            | NetworkUnreachable
            | NotConnected
            | TimedOut
            | UnexpectedEof
    )
}

/// Builds the error `zenwave` returns for an HTTP error response.
///
/// Lets provider crates test their classification of status, headers and error
/// bodies without a server.
///
/// # Panics
///
/// Panics when `status` or a header is not valid HTTP.
#[must_use]
pub fn http_error(status: u16, headers: &[(&str, &str)], body: &str) -> Error {
    use ::zenwave::{Response, StatusCode, error::HttpErrorResponse};

    let status = StatusCode::from_u16(status).expect("valid HTTP status");
    let mut response = Response::new(::zenwave::Body::empty());
    *response.status_mut() = status;
    for (name, value) in headers {
        response.headers_mut().insert(
            ::zenwave::header::HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
            ::zenwave::header::HeaderValue::from_str(value).expect("valid header value"),
        );
    }
    Error::Http {
        status,
        message: std::string::String::from(body),
        response: alloc::boxed::Box::new(HttpErrorResponse {
            response,
            body_text: Some(std::string::String::from(body)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses() {
        assert_eq!(status_kind(500), FailureKind::Transient);
        assert_eq!(status_kind(503), FailureKind::Transient);
        assert_eq!(status_kind(529), FailureKind::Transient);
        assert_eq!(status_kind(408), FailureKind::Transient);
        assert_eq!(status_kind(429), FailureKind::Transient);
        assert_eq!(status_kind(401), FailureKind::Permanent);
        assert_eq!(status_kind(400), FailureKind::Permanent);
    }

    #[test]
    fn retry_after_seconds_and_date() {
        // Sun, 06 Nov 1994 08:49:37 GMT.
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(784_111_777);
        assert_eq!(
            parse_retry_after("120", now),
            Some(Duration::from_secs(120))
        );
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:47 GMT", now),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:07 GMT", now),
            Some(Duration::ZERO)
        );
        assert_eq!(parse_retry_after("soon", now), None);
    }
}
