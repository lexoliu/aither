//! Local llama.cpp provider integration for aither.

mod client;
mod error;
mod provider;

pub use client::{Builder, Llama};
pub use error::LlamaError;
pub use provider::LlamaProvider;
