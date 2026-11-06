//! The memory a terminal's jobs hold, and what their home environment has of
//! it.
//!
//! A job's footprint is the physical footprint of its process tree, unified
//! memory (GPU buffers) included, which resident size is not. A
//! [`FootprintReader`] reads it and the machine's physical memory, a
//! [`FootprintRegistry`] shares the total each agent's jobs hold with the
//! other agents on the machine, and [`JobMemory`] puts both to a terminal's
//! job registry. What to do about the numbers is the agent's business.
//!
//! Only the home environment is measured. Its machine is one
//! [`MemoryDomain`]; a container the agent works in adds a second, read
//! through a [`ContainerMemory`] inside the container. Jobs of a remote
//! execution environment belong to neither and are never read.
//!
//! The readers and the registry are traits so tests can inject them.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use crate::job_registry::{JobPlace, JobRegistry, JobStatus};
use crate::shell_session::{ContainerProbe, ContainerProbeError};

/// The environment variable marking every process of a container job.
///
/// Its value is the job's task ID, so the container's reader finds the job's
/// processes without a pid the container does not share. (A local job's
/// processes carry [`JOB_MARKER_VAR`](crate::JOB_MARKER_VAR) instead, a marker
/// only the host reads.)
pub const JOB_MARKER_ENV: &str = "AITHER_CONTAINER_JOB";

/// Which memory a job competes for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    /// The memory of the machine the embedder runs on.
    Machine,
    /// The memory limit of the container the agent works in.
    Container,
}

impl fmt::Display for MemoryDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Machine => "machine",
            Self::Container => "container",
        })
    }
}

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
    /// The container could not be read.
    #[error(transparent)]
    Container(#[from] ContainerProbeError),
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
/// proportional set size of every process of the tree from `/proc`, and
/// Windows the private bytes (`PrivateUsage`, the memory committed to the
/// process alone) of every process of the tree.
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

#[cfg(windows)]
mod system {
    use std::collections::HashMap;
    use std::io;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
        TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
    };

    use super::FootprintError;

    /// A kernel handle closed on drop.
    struct Handle(HANDLE);

    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: the handle is open and closed exactly once, here.
            unsafe { CloseHandle(self.0) };
        }
    }

    fn io_error(what: &'static str) -> FootprintError {
        FootprintError::Io {
            what,
            source: io::Error::last_os_error(),
        }
    }

    /// Every process as `(pid, parent pid)`, from one snapshot of the
    /// process table.
    fn process_table() -> Result<Vec<(u32, u32)>, FootprintError> {
        const WHAT: &str = "the process table";
        // SAFETY: a snapshot of all processes takes no pointers; the
        // process id is ignored for `TH32CS_SNAPPROCESS`.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(io_error(WHAT));
        }
        let snapshot = Handle(snapshot);
        let mut entry = PROCESSENTRY32W {
            dwSize: size_of::<PROCESSENTRY32W>()
                .try_into()
                .expect("the entry size fits u32"),
            ..PROCESSENTRY32W::default()
        };
        let mut table = Vec::new();
        // SAFETY: `snapshot` is open and `entry` is valid for writes, with
        // `dwSize` set as the call requires.
        let mut more = unsafe { Process32FirstW(snapshot.0, &raw mut entry) } != 0;
        while more {
            table.push((entry.th32ProcessID, entry.th32ParentProcessID));
            // SAFETY: as above.
            more = unsafe { Process32NextW(snapshot.0, &raw mut entry) } != 0;
        }
        // The walk ends with `ERROR_NO_MORE_FILES`; any other end is a failure.
        let end = io::Error::last_os_error();
        if end.raw_os_error() == Some(18) {
            Ok(table)
        } else {
            Err(FootprintError::Io {
                what: WHAT,
                source: end,
            })
        }
    }

    /// The private bytes of one process; a process that is gone or that the
    /// caller may not query holds nothing to count.
    fn private_bytes(pid: u32) -> u64 {
        // SAFETY: opening a process takes no pointers.
        let process =
            unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid) };
        if process.is_null() {
            return 0;
        }
        let process = Handle(process);
        let mut counters = PROCESS_MEMORY_COUNTERS_EX {
            cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>()
                .try_into()
                .expect("the counters size fits u32"),
            PageFaultCount: 0,
            PeakWorkingSetSize: 0,
            WorkingSetSize: 0,
            QuotaPeakPagedPoolUsage: 0,
            QuotaPagedPoolUsage: 0,
            QuotaPeakNonPagedPoolUsage: 0,
            QuotaNonPagedPoolUsage: 0,
            PagefileUsage: 0,
            PeakPagefileUsage: 0,
            PrivateUsage: 0,
        };
        // SAFETY: `PROCESS_MEMORY_COUNTERS_EX` starts with a
        // `PROCESS_MEMORY_COUNTERS`, and `cb` tells the call to fill the
        // extended structure, which `counters` is valid for writes of.
        let ok = unsafe {
            GetProcessMemoryInfo(
                process.0,
                (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
                counters.cb,
            )
        };
        if ok == 0 {
            0
        } else {
            counters.PrivateUsage as u64
        }
    }

    pub(super) fn tree_footprints(roots: &[u32]) -> Result<Vec<u64>, FootprintError> {
        if roots.is_empty() {
            return Ok(Vec::new());
        }
        let table = process_table()?;
        let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(pid, parent) in &table {
            children.entry(parent).or_default().push(pid);
        }
        Ok(roots
            .iter()
            .map(|&root| {
                if !table.iter().any(|&(pid, _)| pid == root) {
                    return 0;
                }
                // The root and everything that descends from it.
                let mut seen = vec![root];
                let mut next = 0;
                while let Some(&pid) = seen.get(next) {
                    next += 1;
                    for &child in children.get(&pid).into_iter().flatten() {
                        if !seen.contains(&child) {
                            seen.push(child);
                        }
                    }
                }
                seen.into_iter().map(private_bytes).sum()
            })
            .collect())
    }

    pub(super) fn physical_memory() -> Result<u64, FootprintError> {
        let mut status = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>()
                .try_into()
                .expect("the status size fits u32"),
            dwMemoryLoad: 0,
            ullTotalPhys: 0,
            ullAvailPhys: 0,
            ullTotalPageFile: 0,
            ullAvailPageFile: 0,
            ullTotalVirtual: 0,
            ullAvailVirtual: 0,
            ullAvailExtendedVirtual: 0,
        };
        // SAFETY: `status` is valid for writes with `dwLength` set as the
        // call requires.
        if unsafe { GlobalMemoryStatusEx(&raw mut status) } == 0 {
            return Err(io_error("the physical memory size"));
        }
        Ok(status.ullTotalPhys)
    }
}

#[cfg(not(any(unix, windows)))]
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

/// What the container's own memory cgroup (v2) reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CgroupMemory {
    /// `memory.current`: what the container holds now, in bytes.
    pub current_bytes: u64,
    /// `memory.max`: the container's limit, in bytes; `None` when it has
    /// none.
    pub limit_bytes: Option<u64>,
    /// The text of `memory.pressure`, empty when the kernel offers none.
    pub pressure: String,
}

impl CgroupMemory {
    /// Parses the output of the cgroup probe: `memory.max`, `memory.current`,
    /// then the lines of `memory.pressure`.
    ///
    /// # Errors
    ///
    /// Fails when the first two lines are not those numbers.
    pub fn parse(output: &str) -> Result<Self, FootprintError> {
        let malformed = || FootprintError::Malformed {
            what: "the container's memory cgroup",
            content: output.to_string(),
        };
        let mut lines = output.lines();
        let limit = lines.next().ok_or_else(malformed)?.trim();
        let limit_bytes = if limit == "max" {
            None
        } else {
            Some(limit.parse().map_err(|_| malformed())?)
        };
        let current_bytes = lines
            .next()
            .ok_or_else(malformed)?
            .trim()
            .parse()
            .map_err(|_| malformed())?;
        let pressure = lines.collect::<Vec<_>>().join("\n");
        Ok(Self {
            current_bytes,
            limit_bytes,
            pressure,
        })
    }
}

/// Reads the memory of the container the agent works in, inside the
/// container: its cgroup, and the footprint of the jobs it runs.
pub trait ContainerMemory: Send + Sync + 'static {
    /// The container's cgroup now.
    fn cgroup(&self) -> impl Future<Output = Result<CgroupMemory, FootprintError>> + Send;

    /// The bytes held by each job's processes, in order, found by the
    /// [`JOB_MARKER_ENV`] their environment carries; a job with no process
    /// holds nothing.
    fn job_footprints(
        &self,
        task_ids: &[String],
    ) -> impl Future<Output = Result<Vec<u64>, FootprintError>> + Send;
}

/// Object-safe twin of [`ContainerMemory`], used only through
/// [`AnyContainerMemory`].
pub trait ContainerMemoryImpl: Send + Sync {
    /// [`ContainerMemory::cgroup`] through a boxed future.
    fn cgroup_boxed(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<CgroupMemory, FootprintError>> + Send + '_>>;

    /// [`ContainerMemory::job_footprints`] through a boxed future.
    fn job_footprints_boxed<'a>(
        &'a self,
        task_ids: &'a [String],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u64>, FootprintError>> + Send + 'a>>;
}

impl<T: ContainerMemory> ContainerMemoryImpl for T {
    fn cgroup_boxed(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<CgroupMemory, FootprintError>> + Send + '_>> {
        Box::pin(self.cgroup())
    }

    fn job_footprints_boxed<'a>(
        &'a self,
        task_ids: &'a [String],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u64>, FootprintError>> + Send + 'a>> {
        Box::pin(self.job_footprints(task_ids))
    }
}

/// A type-erased [`ContainerMemory`], what the builders store.
#[derive(Clone)]
pub struct AnyContainerMemory(Arc<dyn ContainerMemoryImpl>);

impl AnyContainerMemory {
    /// Erases `memory`.
    #[must_use]
    pub fn new(memory: impl ContainerMemory) -> Self {
        Self(Arc::new(memory))
    }
}

impl ContainerMemory for AnyContainerMemory {
    async fn cgroup(&self) -> Result<CgroupMemory, FootprintError> {
        self.0.cgroup_boxed().await
    }

    async fn job_footprints(&self, task_ids: &[String]) -> Result<Vec<u64>, FootprintError> {
        self.0.job_footprints_boxed(task_ids).await
    }
}

impl fmt::Debug for AnyContainerMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyContainerMemory")
    }
}

/// Reads a container's memory by running scripts in it through its exec path.
#[derive(Debug, Clone)]
pub struct ProbedContainerMemory {
    probe: ContainerProbe,
}

impl ProbedContainerMemory {
    /// A reader of the container `probe` runs scripts in.
    #[must_use]
    pub const fn new(probe: ContainerProbe) -> Self {
        Self { probe }
    }
}

impl ContainerMemory for ProbedContainerMemory {
    async fn cgroup(&self) -> Result<CgroupMemory, FootprintError> {
        let output = self
            .probe
            .read(include_str!("../templates/container_memory_cgroup.sh"))
            .await?;
        CgroupMemory::parse(&output)
    }

    async fn job_footprints(&self, task_ids: &[String]) -> Result<Vec<u64>, FootprintError> {
        if task_ids.is_empty() {
            return Ok(Vec::new());
        }
        let output = self
            .probe
            .read(include_str!("../templates/container_memory_jobs.sh"))
            .await?;
        let mut by_job: BTreeMap<&str, u64> = BTreeMap::new();
        for line in output.lines() {
            let malformed = || FootprintError::Malformed {
                what: "a container job's footprint",
                content: line.to_string(),
            };
            let (id, kib) = line.split_once(' ').ok_or_else(malformed)?;
            let kib: u64 = kib.trim().parse().map_err(|_| malformed())?;
            *by_job.entry(id).or_default() += kib * 1024;
        }
        Ok(task_ids
            .iter()
            .map(|id| by_job.get(id.as_str()).copied().unwrap_or(0))
            .collect())
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

/// The memory state of one [`MemoryDomain`] and of one agent's jobs in it at
/// one moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// The domain this is the state of.
    pub domain: MemoryDomain,
    /// The domain's memory, in bytes: the machine's physical memory, or the
    /// container's limit.
    pub capacity_bytes: u64,
    /// The agent's running jobs in the domain, the largest first.
    pub jobs: Vec<JobFootprint>,
    /// What the agent's jobs in the domain hold together.
    pub own_bytes: u64,
    /// What every agent's jobs in the domain hold together.
    pub total_bytes: u64,
    /// Whether no other agent's jobs hold more than this agent's.
    pub largest_contributor: bool,
}

impl MemorySnapshot {
    /// Whether `bytes` have used up the domain's memory. The machine's
    /// physical memory is exceeded by more bytes than it has; a container's
    /// cgroup never lets its holdings pass the limit, so reaching it is
    /// exceeding it.
    #[must_use]
    pub const fn exceeded_by(&self, bytes: u64) -> bool {
        match self.domain {
            MemoryDomain::Machine => bytes > self.capacity_bytes,
            MemoryDomain::Container => bytes >= self.capacity_bytes,
        }
    }
}

/// The state of the container's memory limit and of the agent's jobs in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerSnapshot {
    /// The jobs against the container's limit.
    pub snapshot: MemorySnapshot,
    /// The text of the container's `memory.pressure`, empty when the kernel
    /// offers none.
    pub pressure: String,
}

/// A terminal's view of the memory its jobs hold.
#[derive(Debug, Clone)]
pub struct JobMemory {
    jobs: JobRegistry,
    reader: AnyFootprintReader,
    registry: Option<AnyFootprintRegistry>,
    window: ResponseWindow,
    container: Option<AnyContainerMemory>,
}

impl JobMemory {
    /// The memory view of the jobs in `jobs`, of the machine the embedder
    /// runs on.
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
            container: None,
        }
    }

    /// Also judges the jobs that run in a container against the container's
    /// own memory limit, read through `container`.
    #[must_use]
    pub fn with_container(mut self, container: AnyContainerMemory) -> Self {
        self.container = Some(container);
        self
    }

    /// Whether the jobs run in a container whose memory is read.
    #[must_use]
    pub const fn has_container(&self) -> bool {
        self.container.is_some()
    }

    /// The response window, as it stands now.
    #[must_use]
    pub fn response_window(&self) -> Duration {
        self.window.get()
    }

    /// The running jobs that run at `place`.
    async fn running_at(&self, place: JobPlace) -> Vec<crate::JobInfo> {
        self.jobs
            .list()
            .await
            .into_iter()
            .filter(|job| job.place == place && matches!(job.status, JobStatus::Running))
            .collect()
    }

    /// Reads the memory of the container the agent works in: its jobs against
    /// the container's limit, and its pressure. `None` when the terminal
    /// runs in no container, or the container has no memory limit.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be read.
    pub async fn container_snapshot(&self) -> Result<Option<ContainerSnapshot>, FootprintError> {
        let Some(container) = &self.container else {
            return Ok(None);
        };
        let cgroup = container.cgroup().await?;
        let Some(capacity_bytes) = cgroup.limit_bytes else {
            return Ok(None);
        };
        let running = self.running_at(JobPlace::Container).await;
        let task_ids: Vec<String> = running.iter().map(|job| job.task_id.clone()).collect();
        let footprints = container.job_footprints(&task_ids).await?;
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
        let own_bytes = jobs.iter().map(|job| job.footprint_bytes).sum();
        Ok(Some(ContainerSnapshot {
            snapshot: MemorySnapshot {
                domain: MemoryDomain::Container,
                capacity_bytes,
                jobs,
                own_bytes,
                // A container belongs to one agent.
                total_bytes: own_bytes,
                largest_contributor: true,
            },
            pressure: cgroup.pressure,
        }))
    }

    /// Reads the memory the running jobs of this machine hold, publishes
    /// the total to the registry and reads the other agents' totals. Jobs in
    /// a container or on a remote host are not the machine's.
    ///
    /// # Errors
    ///
    /// Fails when the host or the registry cannot be read.
    pub async fn snapshot(&self) -> Result<MemorySnapshot, FootprintError> {
        let running = self.running_at(JobPlace::Host).await;
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
            domain: MemoryDomain::Machine,
            capacity_bytes: physical_bytes,
            jobs,
            own_bytes,
            total_bytes: own_bytes + others.iter().sum::<u64>(),
            largest_contributor: others.iter().all(|other| *other <= own_bytes),
        })
    }

    /// Kills the whole process tree of `job`, which reads
    /// [`KillCause::MemoryRunaway`](crate::KillCause::MemoryRunaway) afterwards.
    /// Returns whether the job was running and is killed.
    pub async fn kill_runaway(&self, job: &JobFootprint, snapshot: &MemorySnapshot) -> bool {
        self.jobs
            .kill_runaway(
                job.pid,
                job.footprint_bytes,
                snapshot.capacity_bytes,
                snapshot.domain,
            )
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

    #[cfg(any(unix, windows))]
    #[test]
    fn this_process_tree_has_a_footprint_and_the_machine_has_memory() {
        let reader = SystemFootprints;
        let footprints = reader.tree_footprints(&[std::process::id()]).unwrap();
        assert!(footprints[0] > 0);
        assert!(reader.physical_memory().unwrap() > footprints[0]);
    }

    #[test]
    fn the_cgroup_probe_reads_the_limit_the_usage_and_the_pressure() {
        let limited = CgroupMemory::parse(
            "4294967296\n1073741824\nsome avg10=1.00 avg60=0.00 avg300=0.00 total=1\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
        )
        .unwrap();
        assert_eq!(limited.limit_bytes, Some(4 * 1024 * 1024 * 1024));
        assert_eq!(limited.current_bytes, 1024 * 1024 * 1024);
        assert!(limited.pressure.starts_with("some avg10=1.00"));
        assert!(limited.pressure.contains("\nfull avg10=0.00"));

        let unlimited = CgroupMemory::parse("max\n5\n").unwrap();
        assert_eq!(unlimited.limit_bytes, None);
        assert_eq!(unlimited.pressure, "");

        assert!(CgroupMemory::parse("").is_err());
        assert!(CgroupMemory::parse("lots\n5\n").is_err());
        assert!(CgroupMemory::parse("max\n").is_err());
    }

    #[test]
    fn a_container_is_exceeded_at_its_limit_and_a_machine_above_its_memory() {
        let snapshot = |domain| MemorySnapshot {
            domain,
            capacity_bytes: 100,
            jobs: Vec::new(),
            own_bytes: 0,
            total_bytes: 0,
            largest_contributor: true,
        };
        let machine = snapshot(MemoryDomain::Machine);
        assert!(!machine.exceeded_by(100) && machine.exceeded_by(101));
        let container = snapshot(MemoryDomain::Container);
        assert!(!container.exceeded_by(99) && container.exceeded_by(100));
    }

    #[cfg(unix)]
    mod placed {
        use std::collections::HashMap;
        use std::os::unix::process::ExitStatusExt as _;
        use std::sync::Mutex;

        use async_channel::{Receiver, Sender};

        use super::*;
        use crate::{
            AnyContainerExec, ContainerExec, ContainerExecOutcome, ExecLifetime, TerminalMode,
            job_registry_channel,
        };

        const GIB: u64 = 1024 * 1024 * 1024;

        /// Answers the two probe scripts the way a container would.
        struct Container {
            cgroup: &'static str,
            jobs: &'static str,
            scripts: Mutex<Vec<String>>,
        }

        impl ContainerExec for Container {
            fn exec(
                &self,
                _container_id: &str,
                script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _lifetime: ExecLifetime,
            ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
                self.scripts.lock().unwrap().push(script.to_string());
                let stdout = if script.contains("memory.max") {
                    Ok(self.cgroup)
                } else if script.contains(JOB_MARKER_ENV) {
                    Ok(self.jobs)
                } else {
                    Err(format!("unexpected probe: {script}"))
                };
                std::future::ready(stdout.map(|stdout| ContainerExecOutcome::Completed {
                    output: std::process::Output {
                        status: std::process::ExitStatus::from_raw(0),
                        stdout: stdout.as_bytes().to_vec(),
                        stderr: Vec::new(),
                    },
                    killed: Vec::new(),
                }))
            }

            fn exec_streamed(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _stdout: Sender<Vec<u8>>,
            ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
                std::future::ready(Err("a probe is never streamed".to_string()))
            }
        }

        fn container_memory(
            cgroup: &'static str,
            jobs: &'static str,
        ) -> (Arc<Container>, AnyContainerMemory) {
            let container = Arc::new(Container {
                cgroup,
                jobs,
                scripts: Mutex::new(Vec::new()),
            });
            let exec = AnyContainerExec::new(Arc::clone(&container));
            let memory = ProbedContainerMemory::new(ContainerProbe::new(&exec, "box"));
            (container, AnyContainerMemory::new(memory))
        }

        /// Footprints the test says host processes hold.
        struct Held(HashMap<u32, u64>);

        impl FootprintReader for Held {
            fn tree_footprints(&self, roots: &[u32]) -> Result<Vec<u64>, FootprintError> {
                Ok(roots
                    .iter()
                    .map(|pid| self.0.get(pid).copied().unwrap_or(0))
                    .collect())
            }

            fn physical_memory(&self) -> Result<u64, FootprintError> {
                Ok(16 * GIB)
            }
        }

        async fn memory_with_jobs() -> JobMemory {
            let (jobs, service) = job_registry_channel();
            tokio::spawn(async move { service.serve().await });
            for (place, pid, task) in [
                (JobPlace::Host, 1, "host"),
                (JobPlace::Container, 2, "boxed"),
                (JobPlace::Remote, 3, "remote"),
            ] {
                jobs.register_at(place, pid, task, "exec", task, TerminalMode::Sandboxed)
                    .await;
            }
            // The remote and the container pid mean nothing to the host, yet
            // would read as huge if they were read there.
            let held = Held(HashMap::from([(1, 2 * GIB), (2, 90 * GIB), (3, 90 * GIB)]));
            JobMemory::new(
                jobs,
                AnyFootprintReader::new(held),
                None,
                ResponseWindow::new(Duration::from_secs(60)),
            )
        }

        #[tokio::test]
        async fn the_machine_counts_only_jobs_that_run_on_it() {
            let memory = memory_with_jobs().await;
            let snapshot = memory.snapshot().await.unwrap();
            let tasks: Vec<_> = snapshot
                .jobs
                .iter()
                .map(|job| job.task_id.as_str())
                .collect();
            assert_eq!(tasks, ["host"]);
            assert_eq!(snapshot.own_bytes, 2 * GIB);
            assert_eq!(snapshot.domain, MemoryDomain::Machine);
        }

        #[tokio::test]
        async fn a_container_with_a_limit_is_judged_against_it_with_its_own_jobs() {
            let (container, reader) = container_memory(
                "4294967296\n1073741824\nsome avg10=12.00 avg60=0.00 avg300=0.00 total=1\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
                "boxed 1048576\nboxed 524288\nother 99\n",
            );
            let memory = memory_with_jobs().await.with_container(reader);
            let snapshot = memory.container_snapshot().await.unwrap().unwrap();
            assert_eq!(snapshot.snapshot.domain, MemoryDomain::Container);
            assert_eq!(snapshot.snapshot.capacity_bytes, 4 * GIB);
            let jobs: Vec<_> = snapshot
                .snapshot
                .jobs
                .iter()
                .map(|job| (job.task_id.as_str(), job.footprint_bytes))
                .collect();
            // Both processes of the job are summed; the processes of a job
            // this terminal does not know are ignored.
            assert_eq!(jobs, [("boxed", GIB + GIB / 2)]);
            assert!(snapshot.pressure.starts_with("some avg10=12.00"));
            // Read through the container's exec path.
            assert_eq!(container.scripts.lock().unwrap().len(), 2);
        }

        #[tokio::test]
        async fn a_container_without_a_limit_or_a_terminal_without_a_container_adds_nothing() {
            let (_, unlimited) = container_memory("max\n123\n", "");
            let memory = memory_with_jobs().await.with_container(unlimited);
            assert_eq!(memory.container_snapshot().await.unwrap(), None);
            assert_eq!(
                memory_with_jobs().await.container_snapshot().await.unwrap(),
                None
            );
        }
    }
}
