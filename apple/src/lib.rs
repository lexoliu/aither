//! # aither-apple
//!
//! Apple Intelligence provider for `aither`: the on-device
//! `SystemLanguageModel` from the Foundation Models framework behind the
//! shared [`aither_core::llm::LanguageModel`] trait.
//!
//! ## Availability
//!
//! The provider works on macOS/iOS 26+ devices where Apple Intelligence is
//! enabled. [`AppleIntelligence::availability`] mirrors the native
//! `SystemLanguageModel.Availability`; [`AppleIntelligence::new`] fails with a
//! typed [`AppleError::Unavailable`] when the device is ineligible, Apple
//! Intelligence is off, or the model assets are not ready.
//!
//! ```no_run
//! # #[cfg(all(not(docsrs), any(target_os = "macos", target_os = "ios")))]
//! # async fn demo() -> Result<(), aither_apple::AppleError> {
//! use aither_apple::AppleIntelligence;
//! use aither_core::llm::{LanguageModel, LLMRequest, Message};
//! use futures_lite::StreamExt;
//!
//! let model = AppleIntelligence::new()?;
//! let request = LLMRequest::new(vec![Message::user("Hello!")]);
//! let mut stream = model.respond(request);
//! while let Some(event) = stream.next().await {
//!     # let _ = event;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Native bridge
//!
//! The crate compiles a small Swift C-ABI bridge (`swift/`) at build time via
//! `xcrun swiftc`. Sessions, tool dispatch and streaming use real Foundation
//! Models objects — `LanguageModelSession`, `Transcript`, `GenerationSchema`,
//! `Tool` — bridged through a bounded event channel with explicit
//! backpressure and cancellation.
//!
//! OS 27-only surfaces (image prompts, `toolCallingMode`, reasoning text,
//! usage, custom `LanguageModel`s) are gated at runtime and rejected with
//! typed errors on older systems rather than silently dropped.
//!
//! On non-Apple platforms the crate still compiles and documents the API;
//! [`Availability`] reports `UnsupportedOs` and no `LanguageModel`
//! implementation is provided.

mod error;
mod model;
#[cfg(aither_apple_native)]
mod params;
#[cfg(aither_apple_native)]
mod schema;
#[cfg(aither_apple_native)]
mod transcript;
#[cfg(aither_apple_native)]
mod wire;

#[cfg(aither_apple_native)]
mod attachments;
#[cfg(aither_apple_native)]
mod driver;
#[cfg(aither_apple_native)]
mod ffi;
#[cfg(aither_apple_native)]
mod llm;

pub use error::{AppleError, AttachmentError, UnavailableReason};
pub use model::{AppleIntelligence, AppleIntelligenceProvider, Availability, DEFAULT_MODEL_ID};
