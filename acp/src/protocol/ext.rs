//! Extension method names.
//!
//! ACP reserves method names beginning with `_` for protocol extensions.
//! [`ExtMethod`] carries that invariant in the type: it can only be
//! constructed from a name that starts with `_`, so callers of
//! [`AcpClient::ext_request`](crate::AcpClient::ext_request) can never
//! accidentally shadow a standard method.

use std::borrow::Cow;
use std::fmt;

use thiserror::Error;

/// Error returned when an [`ExtMethod`] name does not start with `_`.
#[derive(Debug, Clone, Error)]
#[error("extension method name {0:?} must start with `_`")]
pub struct ExtMethodError(String);

/// A JSON-RPC method name for an ACP extension.
///
/// Per the protocol, custom methods must be prefixed with `_`; the prefix is
/// enforced at construction so an `ExtMethod` is always a valid extension
/// name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtMethod(Cow<'static, str>);

impl ExtMethod {
    /// Create an extension method from a static name.
    ///
    /// # Panics
    ///
    /// Panics if `name` does not start with `_`. For non-static names use
    /// [`try_new`](Self::try_new).
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        match name.as_bytes() {
            [b'_', ..] => Self(Cow::Borrowed(name)),
            _ => panic!("extension method names must start with `_`"),
        }
    }

    /// Create an extension method from any string, checking the `_` prefix
    /// at runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ExtMethodError`] if `name` does not start with `_`.
    pub fn try_new(name: impl Into<String>) -> Result<Self, ExtMethodError> {
        let name = name.into();
        if name.starts_with('_') {
            Ok(Self(Cow::Owned(name)))
        } else {
            Err(ExtMethodError(name))
        }
    }

    /// The method name as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ExtMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ExtMethod {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<ExtMethod> for String {
    fn from(method: ExtMethod) -> Self {
        method.0.into_owned()
    }
}

impl TryFrom<String> for ExtMethod {
    type Error = ExtMethodError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::try_new(name)
    }
}

impl TryFrom<&str> for ExtMethod {
    type Error = ExtMethodError;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        Self::try_new(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_underscore_names() {
        assert_eq!(ExtMethod::new("_session/goal").as_str(), "_session/goal");
        assert_eq!(ExtMethod::new("_").as_str(), "_");
    }

    #[test]
    #[should_panic(expected = "must start with `_`")]
    fn new_rejects_standard_names() {
        let _ = ExtMethod::new("session/new");
    }

    #[test]
    fn try_new_reports_invalid_names() {
        assert!(ExtMethod::try_new("_x").is_ok());
        let error = ExtMethod::try_new("session/new").expect_err("should reject");
        assert_eq!(
            error.to_string(),
            "extension method name \"session/new\" must start with `_`"
        );
    }
}
