//! GitHub Copilot integration for the Aither framework.
//!
//! This crate provides a `Copilot` client that implements the `LanguageModel` trait,
//! allowing you to use GitHub Copilot's chat API through the unified aither interface.
//!
//! GitHub Copilot uses OAuth device flow for authentication. The typical flow is:
//!
//! 1. Call [`auth::request_device_code`] to get a device code and verification URL
//! 2. Direct the user to the verification URL to enter the code
//! 3. Call [`auth::poll_for_token`] to wait for the user to complete authentication
//! 4. Use the token to create a [`Copilot`] client
//!
//! ```no_run
//! use aither_copilot::{Copilot, auth};
//! use aither_core::LanguageModel;
//!
//! # async fn demo() -> anyhow::Result<()> {
//! // Request device code
//! let device_code = auth::request_device_code().await?;
//! println!("Visit {} and enter code: {}", device_code.verification_uri, device_code.user_code);
//!
//! // Poll for token
//! let token = auth::poll_for_token(&device_code.device_code, device_code.interval).await?;
//!
//! // Create client
//! let copilot = Copilot::new(token.access_token);
//! # Ok(())
//! # }
//! ```

mod client;
mod constant;
mod error;
mod provider;

pub mod auth;

pub use client::{Builder, Copilot};
pub use constant::*;
pub use error::CopilotError;
pub use provider::CopilotProvider;
