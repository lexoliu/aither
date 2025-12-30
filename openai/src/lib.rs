//! `OpenAI` integration for the Aither framework built on top of the `zenwave`
//! HTTP client and the shared `aither-core` abstractions.
//!
//! By default this client uses the Responses API. The legacy `chat.completions`
//! endpoint is still available via [`OpenAI::builder`], but OpenAI has deprecated it.
//! If you must keep using `max_tokens`, enable the legacy compatibility flag on the builder.
//!
//! ```no_run
//! use aither_core::{LanguageModel, llm::{Message, Tools, model::Parameters}};
//! use aither_openai::OpenAI;
//! use futures_lite::StreamExt;
//!
//! # async fn demo() -> anyhow::Result<()> {
//! let model = OpenAI::new(std::env::var("OPENAI_API_KEY")?)
//!     .with_model("gpt-4o-mini");
//!
//! let messages = [
//!     Message::system("You are a concise assistant."),
//!     Message::user("Explain the Rust ownership model in one paragraph."),
//! ];
//! let mut tools = Tools::new();
//! let params = Parameters::default();
//! let mut stream = model.respond(&messages, &mut tools, &params);
//! let mut collected = String::new();
//! while let Some(chunk) = stream.next().await {
//!     collected.push_str(&chunk?);
//! }
//! println!("{collected}");
//! # Ok(()) }
//! ```

mod audio;
mod client;
mod embedding;
mod error;
mod image;
mod moderation;
mod provider;
mod request;
mod response;

pub use client::{ApiKind, Builder, OpenAI};
pub use error::OpenAIError;
pub use provider::OpenAIProvider;

mod constant;
pub use constant::*;

pub(crate) const DEFAULT_MODEL: &str = GPT5_2;
pub(crate) const DEFAULT_BASE_URL: &str = OPENAI_BASE_URL;
pub(crate) const DEFAULT_EMBEDDING_MODEL: &str = EMBEDDING_SMALL;
pub(crate) const DEFAULT_EMBEDDING_DIM: usize = 1536;
pub(crate) const DEFAULT_IMAGE_MODEL: &str = IMAGE_GPT_1_5;
pub(crate) const DEFAULT_AUDIO_MODEL: &str = TTS_GPT4O_MINI;
pub(crate) const DEFAULT_AUDIO_VOICE: &str = "alloy";
pub(crate) const DEFAULT_AUDIO_FORMAT: &str = "mp3";
pub(crate) const DEFAULT_TRANSCRIPTION_MODEL: &str = STT_GPT4O;
pub(crate) const DEFAULT_MODERATION_MODEL: &str = MODERATION_LATEST;
