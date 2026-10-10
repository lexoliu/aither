//! `ChatGPT` Codex provider for aither.
//!
//! This crate integrates `ChatGPT` subscription accounts ("Codex") as a model
//! provider:
//!
//! - [`auth`] — the two OAuth login flows (browser PKCE loopback and device
//!   authorization), token refresh, and the persisted
//!   [`CodexCredentials`] type.
//! - [`Codex`] — a [`LanguageModel`] that wraps the `aither-openai` Responses
//!   client pointed at the `chatgpt.com/backend-api/codex` subscription
//!   endpoint, handling token refresh and rotation internally.
//! - [`CodexProvider`] — a [`LanguageModelProvider`] that lists the backend's
//!   own model catalog and hands out [`Codex`] clients.

pub mod auth;
mod client;
pub mod constant;
mod error;
mod provider;

pub use auth::CodexCredentials;
pub use client::Codex;
pub use error::CodexError;
pub use provider::CodexProvider;

pub use aither_core::llm::{LanguageModel, provider::LanguageModelProvider};
