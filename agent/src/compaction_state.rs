//! The one compaction state of an agent.
//!
//! Automatic, idle, manual and tool-called compaction all reach the window
//! through this state, so there is only ever one handoff in the making: it is
//! either running in the background, landed and waiting for a request
//! boundary, or absent. Starting a compaction of any kind retires whatever
//! the state held, and a handoff forked from an older window is therefore
//! never applied on top of a newer one.

use crate::context::{CompletedCompaction, PendingCompaction};

/// Where the agent's compaction stands.
#[derive(Debug, Default)]
pub enum CompactionState {
    /// No compaction is running or waiting.
    #[default]
    Idle,
    /// A compaction forked alongside the run is still in flight.
    Running(PendingCompaction),
    /// A compaction finished and waits for the next request boundary.
    Landed(CompletedCompaction),
}

impl CompactionState {
    /// Whether a compaction is running or waiting to be applied.
    pub const fn is_active(&self) -> bool {
        !matches!(self, Self::Idle)
    }

    /// Whether a compaction is still in flight.
    pub const fn is_running(&self) -> bool {
        matches!(self, Self::Running(_))
    }

    /// Makes `pending` the compaction in flight, retiring whatever was there.
    pub fn start(&mut self, pending: PendingCompaction) {
        *self = Self::Running(pending);
    }

    /// Takes the compaction in flight out to be polled, leaving the state
    /// idle until it is [`resume`](Self::resume)d or [`land`](Self::land)ed.
    pub fn take_running(&mut self) -> Option<PendingCompaction> {
        match std::mem::take(self) {
            Self::Running(pending) => Some(pending),
            other => {
                *self = other;
                None
            }
        }
    }

    /// Puts a compaction taken with [`take_running`](Self::take_running) back.
    pub fn resume(&mut self, pending: Option<PendingCompaction>) {
        if let Some(pending) = pending {
            *self = Self::Running(pending);
        }
    }

    /// Records that a compaction finished.
    pub fn land(&mut self, completed: CompletedCompaction) {
        *self = Self::Landed(completed);
    }

    /// The finished compaction, if one waits, left in place.
    pub const fn landed(&self) -> Option<&CompletedCompaction> {
        match self {
            Self::Landed(completed) => Some(completed),
            _ => None,
        }
    }

    /// Takes the finished compaction, if one waits.
    pub fn take_landed(&mut self) -> Option<CompletedCompaction> {
        match std::mem::take(self) {
            Self::Landed(completed) => Some(completed),
            other => {
                *self = other;
                None
            }
        }
    }

    /// Drops any compaction in flight or waiting: it was forked from a window
    /// that is about to change under it.
    pub fn retire(&mut self) {
        *self = Self::Idle;
    }
}
