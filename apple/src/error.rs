//! Error types for the Apple Intelligence provider.

/// Why the on-device `SystemLanguageModel` is unavailable.
///
/// Mirrors `SystemLanguageModel.Availability.UnavailableReason`, with an
/// additional variant for operating systems the framework does not exist on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnavailableReason {
    /// The current OS has no Foundation Models framework (macOS/iOS < 26 or a
    /// non-Apple platform).
    UnsupportedOs,
    /// This device is not eligible for Apple Intelligence.
    DeviceNotEligible,
    /// Apple Intelligence is disabled in system settings.
    AppleIntelligenceNotEnabled,
    /// The on-device model assets are not ready (still downloading or not yet
    /// prepared). Retrying later may succeed.
    ModelNotReady,
    /// A reason the native framework reported that this crate does not know.
    /// Carries the native framework's description of the reason.
    Unknown(String),
}

impl core::fmt::Display for UnavailableReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedOs => {
                f.write_str("unsupported operating system (requires macOS or iOS 26.0 or newer)")
            }
            Self::DeviceNotEligible => f.write_str("device not eligible for Apple Intelligence"),
            Self::AppleIntelligenceNotEnabled => f.write_str("Apple Intelligence is not enabled"),
            Self::ModelNotReady => f.write_str("on-device model is not ready"),
            Self::Unknown(detail) => write!(f, "unknown native unavailability reason ({detail})"),
        }
    }
}

impl core::error::Error for UnavailableReason {}

/// A failure resolving an [`Attachment`](aither_core::llm::Attachment) into
/// image bytes for the native prompt.
#[derive(Debug, thiserror::Error)]
pub enum AttachmentError {
    /// The data URL could not be parsed or decoded.
    #[error("invalid data URL: {0}")]
    InvalidDataUrl(String),
    /// The URL scheme cannot be resolved on-device.
    #[error("unsupported attachment URL scheme '{0}'")]
    UnsupportedScheme(String),
    /// The declared media type is not an image the framework can decode.
    #[error("unsupported image media type '{0}'")]
    UnsupportedMediaType(String),
    /// Reading the attachment source failed.
    #[error("failed to read attachment: {0}")]
    Io(#[from] std::io::Error),
    /// Fetching the attachment over HTTP failed.
    #[error("failed to fetch attachment: {0}")]
    Http(String),
    /// The bytes did not decode to an image the framework accepts.
    #[error("attachment data could not be decoded as an image")]
    Undecodable,
}

/// All errors surfaced by the Apple Intelligence provider.
///
/// Native `LanguageModelSession.GenerationError` and `LanguageModelError`
/// cases map onto dedicated, matchable variants; anything the framework adds
/// later lands in [`AppleError::Native`] with its original code.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AppleError {
    /// The on-device model is not currently usable.
    #[error("Apple Intelligence unavailable: {0}")]
    Unavailable(#[from] UnavailableReason),

    /// Provider lookup was asked for a model id it does not serve.
    #[error("unknown Apple Intelligence model '{0}' (expected 'apple-on-device')")]
    UnknownModel(String),

    /// The request shape cannot be represented in a native transcript.
    #[error("unsupported request shape: {0}")]
    UnsupportedRequest(String),

    /// A request parameter has no native equivalent.
    #[error("unsupported parameter '{name}': {reason}")]
    UnsupportedParameter {
        /// The `Parameters` field that was rejected.
        name: &'static str,
        /// Why the parameter cannot be honored natively.
        reason: String,
    },

    /// A JSON schema constraint cannot be expressed with
    /// `DynamicGenerationSchema`.
    #[error("unsupported schema constraint at {path}: {reason}")]
    UnsupportedSchema {
        /// JSON-pointer-like location of the offending node.
        path: String,
        /// The constraint that has no native representation.
        reason: String,
    },

    /// The prompt plus response exceeded the model's context window.
    #[error("context size exceeded: {0}")]
    ContextExceeded(String),

    /// The on-device model rate-limited the request.
    #[error("rate limited: {0}")]
    RateLimited(String),

    /// Guardrails rejected the prompt or the generated content.
    #[error("guardrail violation: {0}")]
    GuardrailViolation(String),

    /// The model refused the request.
    #[error("model refused: {0}")]
    Refusal(String),

    /// The system language model does not support the requested language or
    /// locale.
    #[error("unsupported language or locale: {0}")]
    UnsupportedLanguageOrLocale(String),

    /// The capability the request needs (for example image attachments on
    /// macOS/iOS 26) is not supported by this OS or model.
    #[error("unsupported capability: {0}")]
    UnsupportedCapability(String),

    /// The stream or its driving future was dropped before completion.
    #[error("operation cancelled")]
    Cancelled,

    /// A natively dispatched tool call failed.
    #[error("tool '{tool}' failed: {message}")]
    Tool {
        /// Tool name as registered.
        tool: String,
        /// Failure detail returned by the tool or the framework.
        message: String,
    },

    /// A request attachment could not be resolved or decoded.
    #[error("attachment error: {0}")]
    Attachment(#[from] AttachmentError),

    /// Decoding the structured generation result failed inside the framework.
    #[error("response decoding failed: {0}")]
    DecodingFailure(String),

    /// Any other failure reported by the framework or the bridge. `code`
    /// preserves the machine-readable kind so new framework errors stay
    /// inspectable.
    #[error("native error ({code}): {message}")]
    Native {
        /// Stable code such as `concurrent_requests` or `timeout`.
        code: String,
        /// Human-readable detail from the framework.
        message: String,
    },
}
