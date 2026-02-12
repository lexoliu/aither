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
mod moderation;
mod provider;
mod types;

pub use config::{AuthMode, GEMINI_API_BASE_URL, Gemini};
pub use error::GeminiError;
pub use provider::GeminiProvider;

/// Create a Gemini backend configured to use the `gemini-2.5-pro` model.
#[must_use]
pub fn gemini_2_5_pro(key: impl Into<String>) -> Gemini {
    Gemini::new(key).with_text_model("gemini-2.5-pro")
}

/// Create a Gemini backend configured to use the `gemini-2.5-flash` model.
#[must_use]
pub fn gemini_2_5_flash(key: impl Into<String>) -> Gemini {
    Gemini::new(key).with_text_model("gemini-2.5-flash")
}

/// Create a Gemini backend configured to use the `gemini-2.5-flash-lite` model.
#[must_use]
pub fn gemini_2_5_flash_lite(key: impl Into<String>) -> Gemini {
    Gemini::new(key).with_text_model("gemini-2.5-flash-lite")
}
