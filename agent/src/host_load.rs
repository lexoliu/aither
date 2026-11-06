//! Alerts for high machine load.
//!
//! A [`HostLoadMonitor`] reads the host through a [`HostReader`] and turns
//! sustained machine load into events: the load *starts* when it stays
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

use crate::runaway::RunawayGuard;

use std::fmt;
use std::time::Duration;

use aither_sandbox::{ForegroundWait, JobFootprint, MemoryDomain, MemorySize};
use askama::Template;
use async_channel::{Receiver, Sender};

use crate::clock::{AnyClock, Instant};

pub use system::{SystemReader, psi_memory_pressure};

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

fn log_failed_read(error: &ReadError, what: &str) {
    if matches!(error, ReadError::Unsupported { .. }) {
        tracing::trace!(%error, "{what} is not available on this host");
    } else {
        tracing::warn!(%error, "{what} could not be read");
    }
}

/// A host that offers neither memory pressure nor machine load, such as a
/// Linux kernel built without PSI.
///
/// The monitor over it states nothing about the machine, yet still watches
/// what does not need the host's sources: a container's memory and the free
/// space of a volume.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnwatchedHost;

impl HostReader for UnwatchedHost {
    fn memory_pressure(&mut self) -> Result<MemoryPressure, ReadError> {
        Err(ReadError::Unsupported {
            what: "host memory pressure",
        })
    }

    fn machine_load(&mut self) -> Result<MachineLoad, ReadError> {
        Err(ReadError::Unsupported {
            what: "host machine load",
        })
    }
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
            load_threshold: MachineLoad(SystemReader::DEFAULT_LOAD_THRESHOLD),
            start_after: Duration::from_secs(30),
            end_after: Duration::from_secs(60),
            poll_interval: Duration::from_secs(5),
            high_load_foreground_wait: Duration::from_secs(60),
        }
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
    /// Memory pressure turned warning in the machine or the container.
    MemoryYellow {
        /// Whose memory it is.
        domain: MemoryDomain,
        /// The agent's running jobs in the domain, the largest first.
        jobs: Vec<JobFootprint>,
    },
    /// Memory pressure turned critical in the machine or the container.
    MemoryRed {
        /// Whose memory it is.
        domain: MemoryDomain,
        /// The agent's running jobs in the domain, the largest first.
        jobs: Vec<JobFootprint>,
    },
    /// A job is killed unless the agent acts within the window.
    MemoryFinalWarning {
        /// Whose memory it is.
        domain: MemoryDomain,
        /// The job that goes.
        job: JobFootprint,
        /// How long the agent has.
        window: Duration,
        /// The domain's memory: the machine's physical memory, or the
        /// container's limit.
        capacity: MemorySize,
    },
    /// Memory pressure fell from critical to warning, or to normal.
    MemoryDroppedBack {
        /// Whose memory it is.
        domain: MemoryDomain,
        /// The level it fell to.
        level: MemoryPressure,
        /// The agent's running jobs in the domain, the largest first.
        jobs: Vec<JobFootprint>,
    },
    /// A job was killed for the memory it held.
    JobKilledForMemory {
        /// Whose memory it is.
        domain: MemoryDomain,
        /// The job's task ID.
        task_id: String,
        /// The script the job ran.
        command: String,
        /// What its processes held.
        footprint: MemorySize,
        /// The domain's memory: the machine's physical memory, or the
        /// container's limit.
        capacity: MemorySize,
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
}

impl HostSample {
    /// A sample of the given readings.
    #[must_use]
    pub const fn new(memory: Option<MemoryPressure>, load: Option<MachineLoad>) -> Self {
        Self { memory, load }
    }
}

/// Watches the host and turns sustained machine load into [`HostLoadEvent`]s.
#[derive(Debug)]
pub struct HostLoadMonitor<R> {
    reader: R,
    config: HostLoadConfig,
    foreground_wait: ForegroundWait,
    load: Condition,
}

impl<R: HostReader> HostLoadMonitor<R> {
    /// A monitor reading `reader`, lengthening `foreground_wait` while high
    /// load lasts.
    #[must_use]
    pub fn new(reader: R, config: HostLoadConfig, foreground_wait: ForegroundWait) -> Self {
        Self {
            reader,
            config,
            foreground_wait,
            load: Condition::default(),
        }
    }

    /// Reads the host. A source that fails is logged and left out of the
    /// sample, which leaves its condition as it was.
    ///
    /// A host that has no such source at all ([`UnwatchedHost`]) is not an
    /// error each time it is read; the embedder was told once.
    pub fn read(&mut self) -> HostSample {
        let memory = self
            .reader
            .memory_pressure()
            .inspect_err(|error| log_failed_read(error, "memory pressure"))
            .ok();
        let load = self
            .reader
            .machine_load()
            .inspect_err(|error| log_failed_read(error, "machine load"))
            .ok();
        HostSample { memory, load }
    }

    /// Applies a sample taken at `now`; returns the events it completes.
    ///
    /// Starting high load lengthens the terminal's foreground window, and
    /// ending it restores the normal one.
    pub fn observe(&mut self, now: Instant, sample: HostSample) -> Vec<HostLoadEvent> {
        let config = &self.config;
        let mut events = Vec::new();
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
