//! Shared ACP extension conventions.
//!
//! Extensions are ACP's escape hatches: `_`-prefixed JSON-RPC methods and
//! vendorable `_meta` keys. This module models the ones that are
//! provider-neutral conventions — usable with any agent that advertises
//! them — the way `std` carries more than `core`. Genuinely private
//! provider surface lives under [`crate::vendor`] instead.
//!
//! Each submodule exposes the extension's [`ExtMethod`](crate::protocol::ExtMethod)
//! name, its capability probe, and typed helpers over
//! [`AcpClient`](crate::AcpClient)'s generic `ext_request`/`ext_notify` channel.

pub mod async_tasks;
pub mod auth_status;
pub mod goal;
pub mod steering;
pub mod subagents;
