//! Warns the agent about the memory its jobs hold, and kills a runaway job
//! it left running as the last resort.
//!
//! Memory tightness is judged by the host's pressure level alone. A job is
//! killed in only two cases, each after a final warning went unanswered:
//!
//! 1. the agent's jobs together hold more than the machine's physical
//!    memory: its largest job goes;
//! 2. every agent's jobs together hold more than the physical memory while
//!    pressure is critical: the agent that contributes the most loses its
//!    largest job, and once that agent's total drops, the next contributor
//!    is first in line, while both conditions hold.
//!
//! Each agent runs its own guard and kills only its own jobs; the footprint
//! registry of [`JobMemory`] only totals and ranks.

use std::time::Duration;

use aither_sandbox::{JobMemory, MemorySize};

use crate::clock::Instant;
use crate::host_load::{HostLoadEvent, MemoryPressure};
use crate::pressure::{LevelTracker, Shift};

#[cfg(all(test, unix))]
mod tests;

/// The final warning an agent has not yet acted on.
#[derive(Debug)]
struct Warned {
    task_id: String,
    since: Instant,
    window: Duration,
}

/// The memory guard of one agent.
#[derive(Debug)]
pub struct RunawayGuard {
    memory: JobMemory,
    pressure: LevelTracker<MemoryPressure>,
    warned: Option<Warned>,
}

impl RunawayGuard {
    /// A guard over the jobs `memory` sees.
    #[must_use]
    pub const fn new(memory: JobMemory) -> Self {
        Self {
            memory,
            pressure: LevelTracker::new(MemoryPressure::Normal),
            warned: None,
        }
    }

    /// Takes the host's memory pressure at `now` (`None` when it could not
    /// be read, which leaves the last level in force) and returns the alerts
    /// due, killing the job a final warning named when its window passed
    /// unanswered.
    pub async fn observe(
        &mut self,
        now: Instant,
        pressure: Option<MemoryPressure>,
    ) -> Vec<HostLoadEvent> {
        let snapshot = match self.memory.snapshot().await {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tracing::warn!(%error, "the memory of the agent's jobs could not be read");
                return Vec::new();
            }
        };
        let pressure = pressure.unwrap_or_else(|| self.pressure.level());
        let mut events = Vec::new();
        match self.pressure.observe(pressure) {
            Some(Shift::Rose(MemoryPressure::Warning)) => {
                events.push(HostLoadEvent::MemoryYellow {
                    jobs: snapshot.jobs.clone(),
                });
            }
            Some(Shift::Rose(MemoryPressure::Critical)) => {
                events.push(HostLoadEvent::MemoryRed {
                    jobs: snapshot.jobs.clone(),
                });
            }
            Some(Shift::Fell(level)) => {
                events.push(HostLoadEvent::MemoryDroppedBack {
                    level,
                    jobs: snapshot.jobs.clone(),
                });
            }
            Some(Shift::Rose(MemoryPressure::Normal)) | None => {}
        }

        let physical = snapshot.physical_bytes;
        let alone = snapshot.own_bytes > physical;
        let together = snapshot.machine_bytes > physical
            && pressure == MemoryPressure::Critical
            && snapshot.largest_contributor;
        if !(alone || together) {
            self.warned = None;
            return events;
        }
        match self.warned.take() {
            Some(warned) => {
                // A job that is gone was acted on; the next tick warns about
                // the next one if the conditions still hold.
                if let Some(job) = snapshot
                    .jobs
                    .iter()
                    .find(|job| job.task_id == warned.task_id)
                {
                    if now.duration_since(warned.since) >= warned.window {
                        if self.memory.kill_runaway(job, physical).await {
                            events.push(HostLoadEvent::JobKilledForMemory {
                                task_id: job.task_id.clone(),
                                command: job.command.clone(),
                                footprint: MemorySize(job.footprint_bytes),
                                physical: MemorySize(physical),
                            });
                        }
                    } else {
                        self.warned = Some(warned);
                    }
                }
            }
            None => {
                if let Some(job) = snapshot.jobs.first().filter(|job| job.footprint_bytes > 0) {
                    let window = self.memory.response_window();
                    self.warned = Some(Warned {
                        task_id: job.task_id.clone(),
                        since: now,
                        window,
                    });
                    events.push(HostLoadEvent::MemoryFinalWarning {
                        job: job.clone(),
                        window,
                        physical: MemorySize(physical),
                    });
                }
            }
        }
        events
    }
}
