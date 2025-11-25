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
//!     llm::{Message, tool::Tools, model::Parameters},
//! };
//! use aither_gemini::GeminiBackend;
//! use futures_lite::StreamExt;
//!
//! # async fn run() -> anyhow::Result<()> {
//! let gemini = GeminiBackend::new(std::env::var("GEMINI_API_KEY")?);
//! let messages = [
//!     Message::system("You are a concise assistant."),
//!     Message::user("Explain Tokio in two bullet points."),
//! ];
//! let mut tools = Tools::new();
//! let params = Parameters::default();
//! let mut stream = gemini.respond(&messages, &mut tools, &params);
//! let mut full = String::new();
//! while let Some(chunk) = stream.next().await {
//!     full.push_str(&chunk?);
//! }
//! println!("{full}");
//! # Ok(()) }
//! ```

mod audio;
mod client;
mod config;
mod embedding;
mod error;
mod image;
mod llm;
mod moderation;
mod provider;
mod types;

pub use config::{AuthMode, GEMINI_API_BASE_URL, GeminiBackend};
pub use error::GeminiError;
pub use provider::GeminiProvider;

/// Create a Gemini backend configured to use the `gemini-2.5-pro` model.
#[must_use]
pub fn gemini_2_5_pro(key: impl Into<String>) -> GeminiBackend {
    GeminiBackend::new(key).with_text_model("gemini-2.5-pro")
}

/// Create a Gemini backend configured to use the `gemini-2.5-flash` model.
#[must_use]
pub fn gemini_2_5_flash(key: impl Into<String>) -> GeminiBackend {
    GeminiBackend::new(key).with_text_model("gemini-2.5-flash")
}

/// Create a Gemini backend configured to use the `gemini-2.5-flash-lite` model.
#[must_use]
pub fn gemini_2_5_flash_lite(key: impl Into<String>) -> GeminiBackend {
    GeminiBackend::new(key).with_text_model("gemini-2.5-flash-lite")
}
