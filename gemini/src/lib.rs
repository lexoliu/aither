//! Native Gemini provider for the `aither` trait ecosystem.
//!
//! This crate integrates Google’s **Gemini Developer API** with the shared abstractions from
//! `aither-core`. Each capability (chat, embeddings, images, audio, moderation) is implemented
//! as a thin wrapper around the corresponding REST endpoint so other providers can follow the same
//! patterns.
//!
//! # Quick start
//!
//! ```no_run
//! use aither_core::{
//!     LanguageModel,
//!     llm::{LLMRequest, Message, collect_text, model::Parameters},
//! };
//! use aither_gemini::Gemini;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::new(std::env::var("GEMINI_API_KEY")?);
//! let request = LLMRequest::new([
//!     Message::system("You are a concise assistant."),
//!     Message::user("Explain Tokio in two bullet points."),
//! ])
//! .with_parameters(Parameters::default());
//! let stream = gemini.respond(request);
//! let full = collect_text(stream).await?;
//! println!("{full}");
//! # Ok(()) }
//! ```

mod attachments;
mod audio;
mod client;
mod config;
mod embedding;
mod error;
pub mod files;
mod image;
mod llm;
#[cfg(not(target_arch = "wasm32"))]
mod mime;
mod moderation;
mod provider;
mod types;

pub use config::{AuthMode, GEMINI_API_BASE_URL, Gemini};
pub use error::GeminiError;
pub use provider::GeminiProvider;

/// Gemini model identifiers, for use with [`Gemini::with_text_model`].
///
/// These are plain constants rather than per-model constructors: a constructor
/// named after a model version has to be renamed every time Google ships one,
/// and callers who pin a model want the identifier, not a wrapper around it.
pub mod model {
    /// Latest and most capable Flash model; built for coding and agentic workflows.
    pub const GEMINI_3_7_FLASH: &str = "gemini-3.7-flash";
    /// Improved token efficiency and planning at a lower price than 3.5 Flash.
    pub const GEMINI_3_6_FLASH: &str = "gemini-3.6-flash";
    /// Previous-generation Flash.
    pub const GEMINI_3_5_FLASH: &str = "gemini-3.5-flash";
    /// Low-latency, cost-effective subagent model for high-volume automation.
    pub const GEMINI_3_5_FLASH_LITE: &str = "gemini-3.5-flash-lite";
    /// Earlier Flash-Lite release.
    pub const GEMINI_3_1_FLASH_LITE: &str = "gemini-3.1-flash-lite";
    /// Gemini 2.5 Pro. Superseded by the 3.x Flash line for most work.
    pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
    /// Gemini 2.5 Flash.
    pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
    /// Gemini 2.5 Flash-Lite.
    pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
}

/// Provider tag stamped on reasoning state produced by this crate.
///
/// Reasoning state is only ever replayed to the provider that signed it, so
/// this string must stay stable across releases; changing it silently
/// invalidates state stored in existing transcripts.
pub const PROVIDER_NAME: &str = "google";
