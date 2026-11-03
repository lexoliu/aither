//! Classification of language-model failures.
//!
//! A provider error says what went wrong in provider terms. [`Classify`] lets a
//! caller decide what to do about it without parsing a message: wait for the
//! network, retry after the upstream's requested delay, give up on a quota, or
//! surface a permanent fault.

use alloc::boxed::Box;
use core::{convert::Infallible, time::Duration};

#[cfg(feature = "zenwave")]
pub mod zenwave;

/// What kind of failure a provider error is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureKind {
    /// No response came back from the network: DNS failure, refused or reset
    /// connection, TLS failure, stream EOF or reset mid-response, request
    /// timeout.
    Network,
    /// The upstream answered with a temporary fault: 5xx, 529/overloaded, 408,
    /// or a 429 that is not a quota exhaustion.
    Transient,
    /// A quota or usage limit is exhausted; retrying will not help until it
    /// resets.
    Quota,
    /// Everything else: other 4xx, authentication, malformed requests, parse
    /// errors and local-model errors.
    Permanent,
}

/// A classified failure and, when the upstream asked for one, its retry delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Failure {
    /// What kind of failure this is.
    pub kind: FailureKind,
    /// The delay the upstream requested before the next attempt, from its
    /// `Retry-After` header or an equivalent field.
    pub retry_after: Option<Duration>,
}

impl Failure {
    /// A failure of `kind` with no requested retry delay.
    #[must_use]
    pub const fn new(kind: FailureKind) -> Self {
        Self {
            kind,
            retry_after: None,
        }
    }

    /// A [`FailureKind::Network`] failure.
    #[must_use]
    pub const fn network() -> Self {
        Self::new(FailureKind::Network)
    }

    /// A [`FailureKind::Transient`] failure.
    #[must_use]
    pub const fn transient() -> Self {
        Self::new(FailureKind::Transient)
    }

    /// A [`FailureKind::Quota`] failure.
    #[must_use]
    pub const fn quota() -> Self {
        Self::new(FailureKind::Quota)
    }

    /// A [`FailureKind::Permanent`] failure.
    #[must_use]
    pub const fn permanent() -> Self {
        Self::new(FailureKind::Permanent)
    }

    /// Sets the retry delay the upstream requested.
    #[must_use]
    pub const fn with_retry_after(mut self, retry_after: Option<Duration>) -> Self {
        self.retry_after = retry_after;
        self
    }

    /// Whether the same request may succeed if sent again later
    /// ([`FailureKind::Network`] or [`FailureKind::Transient`]).
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self.kind, FailureKind::Network | FailureKind::Transient)
    }
}

/// Errors that can say what kind of failure they are.
///
/// Implementations read the structured data behind the error — HTTP status,
/// the error type in the response body, the transport error's kind — and never
/// match on message text when a typed source exists.
pub trait Classify {
    /// Classifies this error.
    fn failure(&self) -> Failure;
}

impl<T: Classify + ?Sized> Classify for &T {
    fn failure(&self) -> Failure {
        T::failure(self)
    }
}

impl<T: Classify + ?Sized> Classify for Box<T> {
    fn failure(&self) -> Failure {
        T::failure(self)
    }
}

impl Classify for Infallible {
    fn failure(&self) -> Failure {
        // `Infallible` has no values, so this can never be called.
        Failure::permanent()
    }
}

impl<E: Classify> Classify for super::GenerateError<E> {
    fn failure(&self) -> Failure {
        match self {
            Self::Provider(error) => error.failure(),
            Self::Parse { .. } => Failure::permanent(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_kinds() {
        assert!(Failure::network().is_retryable());
        assert!(Failure::transient().is_retryable());
        assert!(!Failure::quota().is_retryable());
        assert!(!Failure::permanent().is_retryable());
    }

    #[test]
    fn retry_after_is_carried() {
        let failure = Failure::transient().with_retry_after(Some(Duration::from_secs(3)));
        assert_eq!(failure.retry_after, Some(Duration::from_secs(3)));
    }
}
