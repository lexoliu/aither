//! Alerts for high host memory pressure and high machine load.
//!
//! A [`HostLoadMonitor`] reads the host through a [`HostReader`] and turns
//! each of the two conditions into events: a condition *starts* when it stays
//! above its threshold for the start window, and *ends* when it stays below
//! for the end window, so a brief spike does not flap. The agent hears each
//! event once; nothing is injected while a condition persists.
//!
//! While high load lasts, the monitor also lengthens the terminal's
//! foreground window, because commands slowed by the load would otherwise
//! outlast the normal one and move to the background.
//!
//! The monitor never kills or pauses anything.

mod system;

use crate::disk::{AnyDiskReader, DiskPressure, DiskThresholds, DiskWatch};
use crate::runaway::RunawayGuard;

use std::fmt;
use std::time::Duration;

use aither_sandbox::{ForegroundWait, JobFootprint, MemorySize};
use askama::Template;
use async_channel::{Receiver, Sender};

use crate::clock::{AnyClock, Instant};

pub use system::SystemReader;

/// How hard the host's memory is pressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryPressure {
    /// Memory is plentiful.
    Normal,
    /// The host is reclaiming memory and tasks start to stall on it.
    Warning,
    /// The host is thrashing: nothing can make progress without memory.
    Critical,
}

impl fmt::Display for MemoryPressure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Critical => "critical",
        })
    }
}

/// How much of the machine's CPU capacity is in demand.
///
/// `1.0` is a machine whose cores are all busy. What exceeds it is platform
/// specific: where the host reports a load average (macOS) it is that load
/// divided by the number of cores, so a queue shows as more than `1.0`;
/// on Linux it is the share of the last ten seconds during which runnable
/// tasks waited for a CPU (PSI `cpu`), and on Windows the share of CPU time
/// not spent idle since the previous reading.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MachineLoad(pub f64);

impl MachineLoad {
    /// The load as a whole percentage of the machine's capacity.
    #[must_use]
    pub fn percent(self) -> u64 {
        // Truncation is the point: the percentage is for a human to read.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let percent = (self.0 * 100.0).round().max(0.0) as u64;
        percent
    }
}

/// Why a reading could not be taken.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    /// The host's source could not be read.
    #[error("reading {what} failed: {source}")]
    Io {
        /// The source that was read.
        what: &'static str,
        /// What the host answered.
        source: std::io::Error,
    },
    /// The host's source answered in a form not understood.
    #[error("{what} answered `{content}`, which is not understood")]
    Malformed {
        /// The source that was read.
        what: &'static str,
        /// What it answered.
        content: String,
    },
    /// The host offers no such source.
    #[error("{what} is not available on this host")]
    Unsupported {
        /// The source that is missing.
        what: &'static str,
    },
}

/// Reads the host's memory pressure and machine load.
///
/// Reads may block; the monitor runs them off the agent loop. A host
/// implements this for the platform it reads; tests inject their own.
pub trait HostReader: Send + 'static {
    /// The host's memory pressure now.
    ///
    /// # Errors
    ///
    /// Fails when the host's source cannot be read.
    fn memory_pressure(&mut self) -> Result<MemoryPressure, ReadError>;

    /// The machine's load now.
    ///
    /// # Errors
    ///
    /// Fails when the host's source cannot be read.
    fn machine_load(&mut self) -> Result<MachineLoad, ReadError>;
}

/// A type-erased [`HostReader`], what the builders store.
pub struct AnyHostReader(Box<dyn HostReader>);

impl AnyHostReader {
    /// Erases `reader`.
    #[must_use]
    pub fn new(reader: impl HostReader) -> Self {
        Self(Box::new(reader))
    }
}

impl HostReader for AnyHostReader {
    fn memory_pressure(&mut self) -> Result<MemoryPressure, ReadError> {
        self.0.memory_pressure()
    }

    fn machine_load(&mut self) -> Result<MachineLoad, ReadError> {
        self.0.machine_load()
    }
}

impl fmt::Debug for AnyHostReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyHostReader")
    }
}

/// When the monitor calls a condition high, and how long it must hold.
#[derive(Debug, Clone, PartialEq)]
pub struct HostLoadConfig {
    /// Memory pressure at or above this level is high.
    pub memory_threshold: MemoryPressure,
    /// Machine load at or above this is high; see [`MachineLoad`] for what
    /// the platform measures.
    pub load_threshold: MachineLoad,
    /// How long a condition must stay above its threshold to start.
    pub start_after: Duration,
    /// How long a condition must stay below its threshold to end.
    pub end_after: Duration,
    /// How often the host is read.
    pub poll_interval: Duration,
    /// The terminal's foreground window while high machine load lasts.
    pub high_load_foreground_wait: Duration,
}

impl HostLoadConfig {
    /// The defaults for the platform the agent runs on.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            memory_threshold: MemoryPressure::Warning,
            load_threshold: MachineLoad(SystemReader::DEFAULT_LOAD_THRESHOLD),
            start_after: Duration::from_secs(30),
            end_after: Duration::from_secs(60),
            poll_interval: Duration::from_secs(5),
            high_load_foreground_wait: Duration::from_secs(60),
        }
    }

    /// Sets the memory pressure at which the condition is high.
    #[must_use]
    pub const fn memory_threshold(mut self, threshold: MemoryPressure) -> Self {
        self.memory_threshold = threshold;
        self
    }

    /// Sets the machine load at which the condition is high.
    #[must_use]
    pub const fn load_threshold(mut self, threshold: MachineLoad) -> Self {
        self.load_threshold = threshold;
        self
    }

    /// Sets how long a condition must stay above its threshold to start.
    #[must_use]
    pub const fn start_after(mut self, window: Duration) -> Self {
        self.start_after = window;
        self
    }

    /// Sets how long a condition must stay below its threshold to end.
    #[must_use]
    pub const fn end_after(mut self, window: Duration) -> Self {
        self.end_after = window;
        self
    }

    /// Sets how often the host is read.
    #[must_use]
    pub const fn poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Sets the terminal's foreground window while high load lasts.
    #[must_use]
    pub const fn high_load_foreground_wait(mut self, window: Duration) -> Self {
        self.high_load_foreground_wait = window;
        self
    }
}

impl Default for HostLoadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A condition starting or ending, as the agent is told.
#[derive(Debug, Clone, PartialEq)]
pub enum HostLoadEvent {
    /// Memory pressure stayed high for the start window.
    MemoryPressureHigh {
        /// The pressure when the window passed.
        level: MemoryPressure,
    },
    /// Memory pressure stayed below the threshold for the end window.
    MemoryPressureEnded {
        /// The pressure when the window passed.
        level: MemoryPressure,
    },
    /// Machine load stayed high for the start window.
    LoadHigh {
        /// The load when the window passed.
        load: MachineLoad,
        /// The terminal's foreground window while the load lasts.
        foreground_wait: Duration,
    },
    /// Machine load stayed below the threshold for the end window.
    LoadEnded {
        /// The load when the window passed.
        load: MachineLoad,
        /// The terminal's foreground window again.
        foreground_wait: Duration,
    },
    /// Host memory pressure turned warning.
    MemoryYellow {
        /// The agent's running jobs, the largest first.
        jobs: Vec<JobFootprint>,
    },
    /// Host memory pressure turned critical.
    MemoryRed {
        /// The agent's running jobs, the largest first.
        jobs: Vec<JobFootprint>,
    },
    /// A job is killed unless the agent acts within the window.
    MemoryFinalWarning {
        /// The job that goes.
        job: JobFootprint,
        /// How long the agent has.
        window: Duration,
        /// The machine's physical memory.
        physical: MemorySize,
    },
    /// Host memory pressure fell from critical to warning, or to normal.
    MemoryDroppedBack {
        /// The level it fell to.
        level: MemoryPressure,
        /// The agent's running jobs, the largest first.
        jobs: Vec<JobFootprint>,
    },
    /// A job was killed for the memory it held.
    JobKilledForMemory {
        /// The job's task ID.
        task_id: String,
        /// The script the job ran.
        command: String,
        /// What its processes held.
        footprint: MemorySize,
        /// The machine's physical memory.
        physical: MemorySize,
    },
    /// The free space of the agent's volume fell to a lower level.
    DiskPressureRose {
        /// The volume.
        volume: String,
        /// The level it reached.
        level: DiskPressure,
        /// The free space.
        free: MemorySize,
    },
    /// The free space of the agent's volume rose to a higher level.
    DiskPressureDroppedBack {
        /// The volume.
        volume: String,
        /// The level it fell to.
        level: DiskPressure,
        /// The free space.
        free: MemorySize,
    },
}

/// The alert text for a host load event.
#[derive(Template)]
#[template(path = "host_load_alert.txt", escape = "none")]
struct HostLoadAlert<'a> {
    event: &'a HostLoadEvent,
}

impl HostLoadEvent {
    /// The event as the agent reads it.
    pub(crate) fn alert_text(&self) -> String {
        HostLoadAlert { event: self }
            .render()
            .expect("host load alert template must render")
    }
}

/// One condition's progress towards starting or ending.
#[derive(Debug, Default)]
struct Condition {
    active: bool,
    /// Since when readings have disagreed with the state: above the
    /// threshold while inactive, below it while active.
    contrary_since: Option<Instant>,
}

/// What a reading did to a condition.
#[derive(Debug, PartialEq, Eq)]
enum Transition {
    Started,
    Ended,
}

impl Condition {
    fn observe(
        &mut self,
        now: Instant,
        high: bool,
        start_after: Duration,
        end_after: Duration,
    ) -> Option<Transition> {
        if high == self.active {
            self.contrary_since = None;
            return None;
        }
        let since = *self.contrary_since.get_or_insert(now);
        let window = if self.active { end_after } else { start_after };
        if now.duration_since(since) < window {
            return None;
        }
        self.contrary_since = None;
        self.active = high;
        Some(if high {
            Transition::Started
        } else {
            Transition::Ended
        })
    }
}

/// One reading of the host; a part is missing when its source failed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostSample {
    memory: Option<MemoryPressure>,
    load: Option<MachineLoad>,
    disk_free: Option<u64>,
}

impl HostSample {
    /// A sample of the given readings.
    #[must_use]
    pub const fn new(memory: Option<MemoryPressure>, load: Option<MachineLoad>) -> Self {
        Self {
            memory,
            load,
            disk_free: None,
        }
    }

    /// The sample with the free space of the watched volume, in bytes.
    #[must_use]
    pub const fn with_disk_free(mut self, free: u64) -> Self {
        self.disk_free = Some(free);
        self
    }
}

/// Watches the host and turns sustained memory pressure and machine load
/// into [`HostLoadEvent`]s.
#[derive(Debug)]
pub struct HostLoadMonitor<R> {
    reader: R,
    config: HostLoadConfig,
    foreground_wait: ForegroundWait,
    memory: Condition,
    load: Condition,
    disk: Option<DiskWatch>,
}

impl<R: HostReader> HostLoadMonitor<R> {
    /// Also watches the free space of the volume `reader` reads, judged by
    /// `thresholds`: the agent hears once of each change of the disk
    /// pressure level, and nothing is ever killed for it. Each monitor
    /// watches its own volume.
    #[must_use]
    pub fn with_disk(mut self, reader: AnyDiskReader, thresholds: DiskThresholds) -> Self {
        self.disk = Some(DiskWatch::new(reader, thresholds));
        self
    }

    /// A monitor reading `reader`, lengthening `foreground_wait` while high
    /// load lasts.
    #[must_use]
    pub fn new(reader: R, config: HostLoadConfig, foreground_wait: ForegroundWait) -> Self {
        Self {
            reader,
            config,
            foreground_wait,
            memory: Condition::default(),
            load: Condition::default(),
            disk: None,
        }
    }

    /// Reads the host. A source that fails is logged and left out of the
    /// sample, which leaves its condition as it was.
    pub fn read(&mut self) -> HostSample {
        let memory = self
            .reader
            .memory_pressure()
            .inspect_err(|error| tracing::warn!(%error, "memory pressure could not be read"))
            .ok();
        let load = self
            .reader
            .machine_load()
            .inspect_err(|error| tracing::warn!(%error, "machine load could not be read"))
            .ok();
        let disk_free = self.disk.as_mut().and_then(DiskWatch::read);
        HostSample {
            memory,
            load,
            disk_free,
        }
    }

    /// Applies a sample taken at `now`; returns the events it completes.
    ///
    /// Starting high load lengthens the terminal's foreground window, and
    /// ending it restores the normal one.
    pub fn observe(&mut self, now: Instant, sample: HostSample) -> Vec<HostLoadEvent> {
        let config = &self.config;
        let mut events = Vec::new();
        if let Some(level) = sample.memory {
            let high = level >= config.memory_threshold;
            match self
                .memory
                .observe(now, high, config.start_after, config.end_after)
            {
                Some(Transition::Started) => {
                    events.push(HostLoadEvent::MemoryPressureHigh { level });
                }
                Some(Transition::Ended) => {
                    events.push(HostLoadEvent::MemoryPressureEnded { level });
                }
                None => {}
            }
        }
        if let Some(load) = sample.load {
            let high = load >= config.load_threshold;
            match self
                .load
                .observe(now, high, config.start_after, config.end_after)
            {
                Some(Transition::Started) => {
                    let window = config.high_load_foreground_wait;
                    self.foreground_wait.extend(window);
                    events.push(HostLoadEvent::LoadHigh {
                        load,
                        foreground_wait: window,
                    });
                }
                Some(Transition::Ended) => {
                    self.foreground_wait.restore();
                    events.push(HostLoadEvent::LoadEnded {
                        load,
                        foreground_wait: self.foreground_wait.normal(),
                    });
                }
                None => {}
            }
        }
        if let (Some(disk), Some(free)) = (self.disk.as_mut(), sample.disk_free) {
            events.extend(disk.observe(free));
        }
        events
    }

    /// Reads and observes in one step, as the monitor task does.
    pub fn tick(&mut self, clock: &AnyClock) -> Vec<HostLoadEvent> {
        let sample = self.read();
        self.observe(clock.now(), sample)
    }
}

/// Runs `monitor` on `executor` until the returned receiver is dropped.
///
/// The reads run on the blocking pool, so a slow host source never stalls
/// the agent loop or the executor.
///
/// With a `guard`, the same readings also drive the agent's memory guard.
pub fn spawn<R, E>(
    executor: &E,
    monitor: HostLoadMonitor<R>,
    guard: Option<RunawayGuard>,
    clock: AnyClock,
) -> Receiver<HostLoadEvent>
where
    R: HostReader,
    E: executor_core::Executor,
{
    use executor_core::Task;

    let (sender, receiver) = async_channel::unbounded();
    executor.spawn(run(monitor, guard, clock, sender)).detach();
    receiver
}

async fn run<R: HostReader>(
    mut monitor: HostLoadMonitor<R>,
    mut guard: Option<RunawayGuard>,
    clock: AnyClock,
    events: Sender<HostLoadEvent>,
) {
    let interval = monitor.config.poll_interval;
    while !events.is_closed() {
        let (back, sample) = blocking::unblock(move || {
            let sample = monitor.read();
            (monitor, sample)
        })
        .await;
        monitor = back;
        let now = clock.now();
        let mut raised = monitor.observe(now, sample);
        if let Some(guard) = guard.as_mut() {
            raised.extend(guard.observe(now, sample.memory).await);
        }
        for event in raised {
            tracing::info!(?event, "host load condition changed");
            if events.send(event).await.is_err() {
                return;
            }
        }
        async_io::Timer::after(interval).await;
    }
}

#[cfg(test)]
mod tests;
