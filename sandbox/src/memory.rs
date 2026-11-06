//! The memory a terminal's jobs hold, and what the host has of it.
//!
//! A job's footprint is the physical footprint of its process tree, unified
//! memory (GPU buffers) included, which resident size is not. A
//! [`FootprintReader`] reads it and the machine's physical memory, a
//! [`FootprintRegistry`] shares the total each agent's jobs hold with the
//! other agents on the machine, and [`JobMemory`] puts both to a terminal's
//! job registry. What to do about the numbers is the agent's business.
//!
//! The reader and the registry are traits so tests can inject them.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use crate::job_registry::{JobRegistry, JobStatus};

/// A number of bytes, shown the way a person reads memory: `36.2 GiB`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemorySize(pub u64);

impl fmt::Display for MemorySize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
        let mut unit = 0;
        let mut size = self.0;
        // Tenths of the unit, so one decimal needs no float.
        let mut tenths = 0;
        while size >= 1024 && unit + 1 < UNITS.len() {
            tenths = (size % 1024) * 10 / 1024;
            size /= 1024;
            unit += 1;
        }
        if unit == 0 {
            write!(f, "{size} B")
        } else {
            write!(f, "{size}.{tenths} {}", UNITS[unit])
        }
    }
}

/// Why a footprint or the physical memory could not be read.
#[derive(Debug, thiserror::Error)]
pub enum FootprintError {
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
    /// This platform has no reader.
    #[error("{what} is not available on this platform")]
    Unsupported {
        /// What is missing.
        what: &'static str,
    },
    /// The agent name cannot name a registry entry.
    #[error("`{0}` cannot name an agent's registry entry")]
    InvalidAgent(String),
}

/// Reads job footprints and the machine's physical memory.
///
/// Reads may block; [`JobMemory`] runs them off the agent loop.
pub trait FootprintReader: Send + Sync + 'static {
    /// The physical footprint in bytes of each process tree rooted at
    /// `roots`, in order, from one reading of the host. A tree whose root
    /// has exited holds nothing.
    ///
    /// # Errors
    ///
    /// Fails when the host's source cannot be read.
    fn tree_footprints(&self, roots: &[u32]) -> Result<Vec<u64>, FootprintError>;

    /// The machine's physical memory in bytes.
    ///
    /// # Errors
    ///
    /// Fails when the host's source cannot be read.
    fn physical_memory(&self) -> Result<u64, FootprintError>;
}

/// A type-erased [`FootprintReader`], what the builders store.
#[derive(Clone)]
pub struct AnyFootprintReader(Arc<dyn FootprintReader>);

impl AnyFootprintReader {
    /// Erases `reader`.
    #[must_use]
    pub fn new(reader: impl FootprintReader) -> Self {
        Self(Arc::new(reader))
    }
}

impl FootprintReader for AnyFootprintReader {
    fn tree_footprints(&self, roots: &[u32]) -> Result<Vec<u64>, FootprintError> {
        self.0.tree_footprints(roots)
    }

    fn physical_memory(&self) -> Result<u64, FootprintError> {
        self.0.physical_memory()
    }
}

impl fmt::Debug for AnyFootprintReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyFootprintReader")
    }
}

/// Reads the host this process runs on.
///
/// macOS reads `ri_phys_footprint` from `proc_pid_rusage`, Linux the
/// proportional set size of every process of the tree from `/proc`; Windows
/// has no reader yet.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemFootprints;

impl FootprintReader for SystemFootprints {
    fn tree_footprints(&self, roots: &[u32]) -> Result<Vec<u64>, FootprintError> {
        system::tree_footprints(roots)
    }

    fn physical_memory(&self) -> Result<u64, FootprintError> {
        system::physical_memory()
    }
}

#[cfg(unix)]
mod system {
    use super::FootprintError;

    pub(super) fn tree_footprints(roots: &[u32]) -> Result<Vec<u64>, FootprintError> {
        if roots.is_empty() {
            return Ok(Vec::new());
        }
        let trees = crate::process_tree::tree_pids(roots).map_err(|source| FootprintError::Io {
            what: "the process table",
            source,
        })?;
        Ok(trees
            .into_iter()
            .map(|pids| pids.into_iter().map(process_footprint).sum())
            .collect())
    }

    /// The footprint of one process; a process that is gone holds nothing.
    #[cfg(target_vendor = "apple")]
    fn process_footprint(pid: u32) -> u64 {
        let mut usage: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        // SAFETY: `usage` is a valid, writable `rusage_info_v2`, the layout
        // flavor `RUSAGE_INFO_V2` fills, for the whole call.
        let status = unsafe {
            libc::proc_pid_rusage(
                pid.cast_signed(),
                libc::RUSAGE_INFO_V2,
                (&raw mut usage).cast::<libc::rusage_info_t>(),
            )
        };
        if status == 0 {
            usage.ri_phys_footprint
        } else {
            0
        }
    }

    /// The footprint of one process; a process that is gone holds nothing.
    #[cfg(not(target_vendor = "apple"))]
    fn process_footprint(pid: u32) -> u64 {
        let Ok(rollup) = std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")) else {
            return 0;
        };
        rollup
            .lines()
            .find_map(|line| line.strip_prefix("Pss:"))
            .and_then(|rest| rest.trim().strip_suffix("kB"))
            .and_then(|kib| kib.trim().parse::<u64>().ok())
            .map_or(0, |kib| kib * 1024)
    }

    pub(super) fn physical_memory() -> Result<u64, FootprintError> {
        // SAFETY: `sysconf` takes a plain integer and touches no memory.
        let (pages, page) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        match (u64::try_from(pages), u64::try_from(page)) {
            (Ok(pages), Ok(page)) => Ok(pages * page),
            _ => Err(FootprintError::Io {
                what: "the physical memory size",
                source: std::io::Error::last_os_error(),
            }),
        }
    }
}

#[cfg(not(unix))]
mod system {
    use super::FootprintError;

    pub(super) fn tree_footprints(_: &[u32]) -> Result<Vec<u64>, FootprintError> {
        Err(FootprintError::Unsupported {
            what: "the footprint of a process tree",
        })
    }

    pub(super) fn physical_memory() -> Result<u64, FootprintError> {
        Err(FootprintError::Unsupported {
            what: "the physical memory size",
        })
    }
}

/// Shares the memory each agent's jobs hold between the agents on a machine,
/// so each sees the machine-wide total and who contributes the most. It is
/// used to total and to rank, never as a cap.
pub trait FootprintRegistry: Send + Sync + 'static {
    /// Publishes the bytes this agent's jobs hold; zero withdraws the entry.
    ///
    /// # Errors
    ///
    /// Fails when the registry cannot be written.
    fn publish(&self, total_bytes: u64) -> Result<(), FootprintError>;

    /// The bytes every other live agent's jobs hold.
    ///
    /// # Errors
    ///
    /// Fails when the registry cannot be read.
    fn others(&self) -> Result<Vec<u64>, FootprintError>;
}

/// A type-erased [`FootprintRegistry`], what the builders store.
#[derive(Clone)]
pub struct AnyFootprintRegistry(Arc<dyn FootprintRegistry>);

impl AnyFootprintRegistry {
    /// Erases `registry`.
    #[must_use]
    pub fn new(registry: impl FootprintRegistry) -> Self {
        Self(Arc::new(registry))
    }
}

impl FootprintRegistry for AnyFootprintRegistry {
    fn publish(&self, total_bytes: u64) -> Result<(), FootprintError> {
        self.0.publish(total_bytes)
    }

    fn others(&self) -> Result<Vec<u64>, FootprintError> {
        self.0.others()
    }
}

impl fmt::Debug for AnyFootprintRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyFootprintRegistry")
    }
}

/// A [`FootprintRegistry`] in a directory every agent of the machine shares.
///
/// It keeps one small file per agent, rewritten at each publish. An entry not
/// rewritten within the staleness window belongs to an agent that is gone
/// and is ignored.
#[derive(Debug, Clone)]
pub struct FileFootprintRegistry {
    dir: PathBuf,
    agent: String,
    stale_after: Duration,
}

impl FileFootprintRegistry {
    const EXTENSION: &'static str = "footprint";

    /// A registry in `dir` for the agent named `agent`, which must be unique
    /// on the machine and usable as a file name. Entries older than
    /// `stale_after` are ignored.
    ///
    /// # Errors
    ///
    /// Fails when `agent` cannot be a file name.
    pub fn new(
        dir: impl Into<PathBuf>,
        agent: impl Into<String>,
        stale_after: Duration,
    ) -> Result<Self, FootprintError> {
        let agent = agent.into();
        if agent.is_empty() || agent.contains(['/', '\\']) || agent.starts_with('.') {
            return Err(FootprintError::InvalidAgent(agent));
        }
        Ok(Self {
            dir: dir.into(),
            agent,
            stale_after,
        })
    }

    fn entry(&self) -> PathBuf {
        self.dir.join(format!("{}.{}", self.agent, Self::EXTENSION))
    }
}

impl FootprintRegistry for FileFootprintRegistry {
    fn publish(&self, total_bytes: u64) -> Result<(), FootprintError> {
        use std::io::Write as _;

        let io = |source| FootprintError::Io {
            what: "the footprint registry",
            source,
        };
        if total_bytes == 0 {
            return match std::fs::remove_file(self.entry()) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(io(error)),
                _ => Ok(()),
            };
        }
        std::fs::create_dir_all(&self.dir).map_err(io)?;
        let mut file = tempfile::NamedTempFile::new_in(&self.dir).map_err(io)?;
        write!(file, "{total_bytes}").map_err(io)?;
        file.persist(self.entry())
            .map_err(|error| io(error.error))?;
        Ok(())
    }

    fn others(&self) -> Result<Vec<u64>, FootprintError> {
        let io = |source| FootprintError::Io {
            what: "the footprint registry",
            source,
        };
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io(error)),
        };
        let own = self.entry();
        let now = SystemTime::now();
        let mut others = Vec::new();
        for entry in entries {
            let path = entry.map_err(io)?.path();
            if path == own || path.extension().is_none_or(|ext| ext != Self::EXTENSION) {
                continue;
            }
            // An entry withdrawn between the listing and the read is gone.
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let modified = std::fs::metadata(&path).and_then(|meta| meta.modified());
            if modified
                .is_ok_and(|at| now.duration_since(at).unwrap_or_default() > self.stale_after)
            {
                continue;
            }
            let bytes = content
                .trim()
                .parse()
                .map_err(|_| FootprintError::Malformed {
                    what: "a footprint registry entry",
                    content: content.clone(),
                })?;
            others.push(bytes);
        }
        Ok(others)
    }
}

/// How long the agent has to act on a final warning before its job is
/// killed, adjustable while the agent runs. Every clone shares it.
#[derive(Debug, Clone)]
pub struct ResponseWindow(Arc<AtomicU64>);

impl ResponseWindow {
    /// A window of `window`.
    #[must_use]
    pub fn new(window: Duration) -> Self {
        let window = Self::nanos(window);
        Self(Arc::new(AtomicU64::new(window)))
    }

    fn nanos(window: Duration) -> u64 {
        u64::try_from(window.as_nanos()).expect("a response window fits in u64 nanoseconds")
    }

    /// Changes the window for every handle; a final warning already sent
    /// keeps the window it stated.
    pub fn set(&self, window: Duration) {
        self.0.store(Self::nanos(window), Ordering::Relaxed);
    }

    /// The window now.
    #[must_use]
    pub fn get(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::Relaxed))
    }
}

/// One running job and the memory its process tree holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobFootprint {
    /// The job's task ID.
    pub task_id: String,
    /// The script the job runs.
    pub command: String,
    /// The job's root process.
    pub pid: u32,
    /// The physical footprint of the job's process tree, in bytes.
    pub footprint_bytes: u64,
}

impl JobFootprint {
    /// The footprint, as a person reads it.
    #[must_use]
    pub const fn footprint(&self) -> MemorySize {
        MemorySize(self.footprint_bytes)
    }
}

/// The memory state of the machine and of one agent's jobs at one moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// The machine's physical memory, in bytes.
    pub physical_bytes: u64,
    /// The agent's running jobs, the largest first.
    pub jobs: Vec<JobFootprint>,
    /// What the agent's jobs hold together.
    pub own_bytes: u64,
    /// What every agent's jobs hold together.
    pub machine_bytes: u64,
    /// Whether no other agent's jobs hold more than this agent's.
    pub largest_contributor: bool,
}

/// A terminal's view of the memory its jobs hold.
#[derive(Debug, Clone)]
pub struct JobMemory {
    jobs: JobRegistry,
    reader: AnyFootprintReader,
    registry: Option<AnyFootprintRegistry>,
    window: ResponseWindow,
}

impl JobMemory {
    /// The memory view of the jobs in `jobs`.
    #[must_use]
    pub const fn new(
        jobs: JobRegistry,
        reader: AnyFootprintReader,
        registry: Option<AnyFootprintRegistry>,
        window: ResponseWindow,
    ) -> Self {
        Self {
            jobs,
            reader,
            registry,
            window,
        }
    }

    /// The response window, as it stands now.
    #[must_use]
    pub fn response_window(&self) -> Duration {
        self.window.get()
    }

    /// Reads the memory every running job holds, publishes the total to the
    /// registry and reads the other agents' totals.
    ///
    /// # Errors
    ///
    /// Fails when the host or the registry cannot be read.
    pub async fn snapshot(&self) -> Result<MemorySnapshot, FootprintError> {
        let running: Vec<_> = self
            .jobs
            .list()
            .await
            .into_iter()
            .filter(|job| matches!(job.status, JobStatus::Running))
            .collect();
        let roots: Vec<u32> = running.iter().map(|job| job.pid).collect();
        let (reader, registry) = (self.reader.clone(), self.registry.clone());
        let (footprints, physical_bytes, own_bytes, others) = blocking::unblock(move || {
            let footprints = reader.tree_footprints(&roots)?;
            let own: u64 = footprints.iter().sum();
            let others = match registry {
                Some(registry) => {
                    registry.publish(own)?;
                    registry.others()?
                }
                None => Vec::new(),
            };
            Ok::<_, FootprintError>((footprints, reader.physical_memory()?, own, others))
        })
        .await?;
        let mut jobs: Vec<JobFootprint> = running
            .into_iter()
            .zip(footprints)
            .map(|(job, footprint_bytes)| JobFootprint {
                task_id: job.task_id,
                command: job.script,
                pid: job.pid,
                footprint_bytes,
            })
            .collect();
        jobs.sort_by_key(|job| std::cmp::Reverse(job.footprint_bytes));
        Ok(MemorySnapshot {
            physical_bytes,
            jobs,
            own_bytes,
            machine_bytes: own_bytes + others.iter().sum::<u64>(),
            largest_contributor: others.iter().all(|other| *other <= own_bytes),
        })
    }

    /// Kills the whole process tree of `job`, which reads
    /// [`KillCause::MemoryRunaway`](crate::KillCause::MemoryRunaway) afterwards.
    /// Returns whether the job was running and is killed.
    pub async fn kill_runaway(&self, job: &JobFootprint, physical_bytes: u64) -> bool {
        self.jobs
            .kill_runaway(job.pid, job.footprint_bytes, physical_bytes)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_in_binary_units() {
        assert_eq!(MemorySize(512).to_string(), "512 B");
        assert_eq!(MemorySize(1536).to_string(), "1.5 KiB");
        assert_eq!(MemorySize(36 * 1024 * 1024 * 1024).to_string(), "36.0 GiB");
    }

    #[test]
    fn the_file_registry_totals_other_agents_and_forgets_withdrawn_and_stale_ones() {
        let dir = tempfile::tempdir().unwrap();
        let long = Duration::from_secs(3600);
        let a = FileFootprintRegistry::new(dir.path(), "a", long).unwrap();
        let b = FileFootprintRegistry::new(dir.path(), "b", long).unwrap();
        a.publish(100).unwrap();
        b.publish(30).unwrap();
        assert_eq!(a.others().unwrap(), vec![30]);
        assert_eq!(b.others().unwrap(), vec![100]);
        a.publish(0).unwrap();
        assert_eq!(b.others().unwrap(), Vec::<u64>::new());
        let stale = FileFootprintRegistry::new(dir.path(), "c", Duration::ZERO).unwrap();
        b.publish(30).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(stale.others().unwrap(), Vec::<u64>::new());
        assert!(FileFootprintRegistry::new(dir.path(), "../x", long).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn this_process_tree_has_a_footprint_and_the_machine_has_memory() {
        let reader = SystemFootprints;
        let footprints = reader.tree_footprints(&[std::process::id()]).unwrap();
        assert!(footprints[0] > 0);
        assert!(reader.physical_memory().unwrap() > footprints[0]);
    }
}
