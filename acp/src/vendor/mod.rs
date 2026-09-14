//! Vendor-specific ACP surface.
//!
//! ACP's extension mechanism (`_` methods, `_meta` keys) lets providers
//! ship agent features before — or instead of — standardizing them. The
//! modules here model what a specific provider actually speaks on the wire.
//! Anything usable across providers belongs in the common
//! [`protocol`](crate::protocol) layer or the shared [`ext`](crate::ext)
//! conventions instead; nothing in the common layer references these
//! modules.

pub mod codex;
pub mod devin;
pub mod jetbrains;
