//! Warns the agent about the memory its jobs hold, and kills a runaway job
//! it left running as the last resort.
//!
//! The guard protects the home environment: the machine the embedder runs
//! on and, when the agent works in a container that has a memory limit, the
//! container. The two are judged apart, each by its own pressure level (the
//! host's, or the container's `memory.pressure`) against its own memory (the
//! machine's physical memory, or the container's limit). Jobs of a remote
//! execution environment are in neither.
//!
//! Within a domain, a job is killed in only two cases, each after a final
//! warning went unanswered:
//!
//! 1. the agent's jobs together hold more than the domain's memory: its
//!    largest job goes;
//! 2. every agent's jobs together hold more than the domain's memory while
//!    pressure is critical: the agent that contributes the most loses its
//!    largest job, and once that agent's total drops, the next contributor
//!    is first in line, while both conditions hold.
//!
//! Each agent runs its own guard and kills only its own jobs; the footprint
//! registry of [`JobMemory`] only totals and ranks.

use std::time::Duration;

use aither_sandbox::{JobMemory, MemorySize, MemorySnapshot};

use crate::clock::Instant;
use crate::host_load::{HostLoadEvent, MemoryPressure, psi_memory_pressure};
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

/// What the guard remembers of one domain between readings.
#[derive(Debug)]
struct Watch {
    pressure: LevelTracker<MemoryPressure>,
    warned: Option<Warned>,
}

impl Watch {
    const fn new() -> Self {
        Self {
            pressure: LevelTracker::new(MemoryPressure::Normal),
            warned: None,
        }
    }

    /// Takes the domain's pressure (`None` when it could not be read, which
    /// leaves the last level in force) and its memory at `now`, and returns
    /// the alerts due, killing the job a final warning named when its window
    /// passed unanswered.
    async fn observe(
        &mut self,
        memory: &JobMemory,
        now: Instant,
        pressure: Option<MemoryPressure>,
        snapshot: &MemorySnapshot,
    ) -> Vec<HostLoadEvent> {
        let domain = snapshot.domain;
        let pressure = pressure.unwrap_or_else(|| self.pressure.level());
        let mut events = Vec::new();
        match self.pressure.observe(pressure) {
            Some(Shift::Rose(MemoryPressure::Warning)) => {
                events.push(HostLoadEvent::MemoryYellow {
                    domain,
                    jobs: snapshot.jobs.clone(),
                });
            }
            Some(Shift::Rose(MemoryPressure::Critical)) => {
                events.push(HostLoadEvent::MemoryRed {
                    domain,
                    jobs: snapshot.jobs.clone(),
                });
            }
            Some(Shift::Fell(level)) => {
                events.push(HostLoadEvent::MemoryDroppedBack {
                    domain,
                    level,
                    jobs: snapshot.jobs.clone(),
                });
            }
            Some(Shift::Rose(MemoryPressure::Normal)) | None => {}
        }

        let capacity = snapshot.capacity_bytes;
        let alone = snapshot.exceeded_by(snapshot.own_bytes);
        let together = snapshot.exceeded_by(snapshot.total_bytes)
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
                        if memory.kill_runaway(job, snapshot).await {
                            events.push(HostLoadEvent::JobKilledForMemory {
                                domain,
                                task_id: job.task_id.clone(),
                                command: job.command.clone(),
                                footprint: MemorySize(job.footprint_bytes),
                                capacity: MemorySize(capacity),
                            });
                        }
                    } else {
                        self.warned = Some(warned);
                    }
                }
            }
            None => {
                if let Some(job) = snapshot.jobs.first().filter(|job| job.footprint_bytes > 0) {
                    let window = memory.response_window();
                    self.warned = Some(Warned {
                        task_id: job.task_id.clone(),
                        since: now,
                        window,
                    });
                    events.push(HostLoadEvent::MemoryFinalWarning {
                        domain,
                        job: job.clone(),
                        window,
                        capacity: MemorySize(capacity),
                    });
                }
            }
        }
        events
    }
}

/// The memory guard of one agent.
#[derive(Debug)]
pub struct RunawayGuard {
    memory: JobMemory,
    machine: Watch,
    container: Watch,
}

impl RunawayGuard {
    /// A guard over the jobs `memory` sees.
    #[must_use]
    pub const fn new(memory: JobMemory) -> Self {
        Self {
            memory,
            machine: Watch::new(),
            container: Watch::new(),
        }
    }

    /// Takes the host's memory pressure at `now` (`None` when it could not
    /// be read, which leaves the last level in force) and returns the alerts
    /// due, for the machine and for the container when the agent works in one
    /// that has a memory limit; kills the job a final warning named when its
    /// window passed unanswered.
    pub async fn observe(
        &mut self,
        now: Instant,
        pressure: Option<MemoryPressure>,
    ) -> Vec<HostLoadEvent> {
        let mut events = self.observe_machine(now, pressure).await;
        events.extend(self.observe_container(now).await);
        events
    }

    async fn observe_machine(
        &mut self,
        now: Instant,
        pressure: Option<MemoryPressure>,
    ) -> Vec<HostLoadEvent> {
        match self.memory.snapshot().await {
            Ok(snapshot) => {
                self.machine
                    .observe(&self.memory, now, pressure, &snapshot)
                    .await
            }
            Err(error) => {
                tracing::warn!(%error, "the memory of the agent's jobs could not be read");
                Vec::new()
            }
        }
    }

    async fn observe_container(&mut self, now: Instant) -> Vec<HostLoadEvent> {
        match self.memory.container_snapshot().await {
            Ok(Some(container)) => {
                // A kernel without PSI states no pressure, which is none.
                let pressure = if container.pressure.is_empty() {
                    Some(MemoryPressure::Normal)
                } else {
                    let level = psi_memory_pressure(&container.pressure);
                    if level.is_none() {
                        tracing::warn!(
                            pressure = container.pressure,
                            "the container's memory pressure is not understood"
                        );
                    }
                    level
                };
                self.container
                    .observe(&self.memory, now, pressure, &container.snapshot)
                    .await
            }
            // No container, or one with no memory limit: nothing to judge.
            Ok(None) => Vec::new(),
            Err(error) => {
                tracing::warn!(%error, "the memory of the container could not be read");
                Vec::new()
            }
        }
    }
}
