#![no_std]
//! # aither
//!
//! High level façade crate that re-exports everything from [`aither_core`] plus the `#[tool]`
//! derive. Pull this crate into your binary to build portable AI applications that can talk to
//! `OpenAI`, Gemini, local models, or any future provider that implements the core traits.
//!
//! ## What's inside?
//!
//! - [`LanguageModel`] + [`llm::LLMResponse`] for streaming
//!   chat plus reasoning/thinking summaries.
//! - Audio, image, embeddings, and moderation traits that follow the same ergonomics.
//! - `#[tool]` macros for function-calling schemas, so you can share tool definitions across providers.
//!
//! ## Example
//!
//! ```rust,no_run
//! use aither::{LanguageModel, llm::{Message, Request, model::Parameters}};
//! use aither_openai::OpenAI;
//! use futures_lite::{StreamExt, future::poll_fn};
//! use core::pin::Pin;
//!
//! async fn demo(api_key: &str) -> aither::Result<String> {
//!     let model = OpenAI::new(api_key);
//!     let request = Request::new([
//!         Message::system("You are a creative assistant."),
//!         Message::user("Plan a day of food in Osaka.")
//!     ])
//!     .with_parameters(Parameters::default().include_reasoning(true));
//!
//!     let mut response = model.respond(request);
//!     while let Some(step) = poll_fn(|cx| Pin::new(&mut response).poll_reasoning_next(cx)).await {
//!         eprintln!("thinking: {}", step?);
//!     }
//!     let mut answer = String::new();
//!     while let Some(chunk) = response.next().await {
//!         answer.push_str(&chunk?);
//!     }
//!     Ok(answer)
//! }
//! ```
//!
//! ## Modules
//!
//! - [`aither_core::llm`] — language model requests, responses, tool registries, and reasoning streams.
//! - [`aither_core::embedding`] — convert text to vectors.
//! - [`aither_core::image`] — progressive image generation + editing.
//! - [`aither_core::audio`] — speech synthesis and transcription.
//! - [`aither_core::moderation`] — provider-neutral moderation scoring.

extern crate alloc;

pub use aither_core::*;
pub use aither_derive::tool;

// Provider integrations
#[cfg(feature = "openai")]
pub use aither_openai as openai;

#[cfg(feature = "claude")]
pub use aither_claude as claude;

#[cfg(feature = "gemini")]
pub use aither_gemini as gemini;

#[cfg(feature = "mistral")]
pub use aither_mistral as mistral;

#[cfg(feature = "llama")]
pub use aither_llama as llama;

#[cfg(feature = "ort")]
pub use aither_ort as ort;

// High-level features
#[cfg(feature = "agent")]
pub use aither_agent as agent;

#[cfg(feature = "rag")]
pub use aither_rag as rag;

#[cfg(feature = "mem0")]
pub use aither_mem0 as mem0;

#[cfg(feature = "pdf-process")]
pub use aither_pdf_process as pdf_process;

// Tools
#[cfg(feature = "websearch")]
pub use aither_websearch as websearch;

#[cfg(feature = "fs")]
pub use aither_fs as fs;

#[cfg(feature = "command")]
pub use aither_command as command;

#[doc(hidden)]
/// For internal use only.
pub mod __hidden {
    pub type CowStr = alloc::borrow::Cow<'static, str>;
}
