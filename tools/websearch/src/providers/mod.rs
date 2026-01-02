//! Web search provider implementations.
//!
//! This module contains implementations for various web search APIs that can be
//! used with LLM agents to provide real-time internet search capabilities.

mod brave;
mod duckduckgo;
mod google;
mod searxng;
mod serper;
mod tavily;

pub use brave::BraveSearch;
pub use duckduckgo::DuckDuckGo;
pub use google::GoogleSearch;
pub use searxng::SearXNG;
pub use serper::Serper;
pub use tavily::Tavily;
