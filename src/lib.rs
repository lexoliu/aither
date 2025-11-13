#![no_std]
//! Aither: AI Tooling Framework
//! Aither is a framework for building AI-powered applications with a focus on modularity and extensibility.
//! It provides unified trait abstractions for various AI capabilities, allowing developers to write code
//! once and switch between different AI providers without changing application logic.

extern crate alloc;

pub use aither_core::*;
pub use aither_derive::tool;

#[doc(hidden)]
/// For internal use only.
pub mod __hidden {
    pub type CowStr = alloc::borrow::Cow<'static, str>;
}
