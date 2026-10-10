//! The `AppleIntelligence` model and provider.

use crate::error::{AppleError, UnavailableReason};

/// Logical model id served by [`AppleIntelligenceProvider`].
pub const DEFAULT_MODEL_ID: &str = "apple-on-device";

/// Whether the on-device `SystemLanguageModel` is usable right now.
///
/// This is a point-in-time result of the native availability API; it can
/// change when the user toggles Apple Intelligence or when the model assets
/// finish downloading.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Availability {
    /// `SystemLanguageModel.default` reports `.available`.
    Available,
    /// The model cannot be used; the reason mirrors the native
    /// `UnavailableReason`.
    Unavailable(UnavailableReason),
}

impl Availability {
    /// Whether the model is available.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

/// A `LanguageModel` backed by Apple's on-device `SystemLanguageModel`.
///
/// The type is a handle, not a session: every `respond`/`generate` call
/// creates an independent `LanguageModelSession`, so requests can run
/// concurrently without shared mutable state.
///
/// # Platform support
///
/// The `LanguageModel` implementation exists only when compiled for macOS or
/// iOS against the Foundation Models SDK. On other platforms the type still
/// exists, [`Self::availability`] reports
/// [`UnavailableReason::UnsupportedOs`] and [`Self::new`] fails.
///
/// # Availability
///
/// ```no_run
/// # fn demo() -> Result<(), aither_apple::AppleError> {
/// use aither_apple::AppleIntelligence;
///
/// let model = AppleIntelligence::new()?;
/// # let _ = model;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AppleIntelligence;

impl AppleIntelligence {
    /// Creates a handle after validating that the on-device model is
    /// currently available.
    ///
    /// # Errors
    ///
    /// [`AppleError::Unavailable`] carrying the native
    /// [`UnavailableReason`] when the model cannot be used.
    pub fn new() -> Result<Self, AppleError> {
        match Self::availability() {
            Availability::Available => Ok(Self),
            Availability::Unavailable(reason) => Err(reason.into()),
        }
    }

    /// Queries `SystemLanguageModel.default.availability` (or the platform
    /// fallback on non-Apple targets).
    #[cfg(aither_apple_native)]
    #[must_use]
    pub fn availability() -> Availability {
        crate::ffi::native_availability()
    }

    /// Reports that Apple Intelligence is unsupported on this platform.
    #[cfg(not(aither_apple_native))]
    #[must_use]
    pub const fn availability() -> Availability {
        Availability::Unavailable(UnavailableReason::UnsupportedOs)
    }
}

/// Provider that vends [`AppleIntelligence`].
///
/// The provider list contains a single logical model,
/// [`DEFAULT_MODEL_ID`], because the device exposes exactly one
/// `SystemLanguageModel`. The `LanguageModelProvider` implementation exists
/// only on native Apple targets.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppleIntelligenceProvider;
