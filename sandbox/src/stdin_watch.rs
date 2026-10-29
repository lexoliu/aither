//! Stdin-block detection for background jobs.
//!
//! A background job's stdin is a pipe the agent feeds through
//! `terminal_input`. A job is *blocked on stdin* when some process in its
//! tree is waiting in `read` on that pipe with nothing buffered. Detection is
//! per pipe, so a descendant's read counts the same as the job's own.
//!
//! Each platform answers one question per probe, [`Probe`]; the platform-free
//! [`EpisodeDetector`] turns the probes into at most one alert per blocking
//! episode.
//!
//! - macOS: the kernel sets `PIPE_WANTR` on a pipe's read side before a reader
//!   sleeps in `pipe_read` and clears it only on the write side, so the flag
//!   alone also stays set after a reader was interrupted by a signal or killed.
//!   A set flag therefore counts only once the job's process tree has also used
//!   no CPU between two probes.
//! - Linux: a process of the job's tree holds the pipe's inode on some
//!   descriptor and has a thread in state `S` whose
//!   `/proc/<pid>/task/<tid>/wchan` names a pipe-read wait function. `wchan`
//!   needs only ptrace *read* access, unlike `task/<tid>/syscall`, which
//!   needs *attach* — and it names the wait itself, so it finds a `cat`
//!   sleeping in `splice` as readily as a shell in `read`. The kernel
//!   wait-function names are data (`PIPE_READ_WCHAN`), covering current and
//!   older names: `pipe_read`, `anon_pipe_read`, `pipe_wait_readable`. Since
//!   `wchan` does not say which pipe the thread sleeps on, a process counts
//!   only when nothing else could feed any other pipe or FIFO it reads: each
//!   must be an anonymous pipe that only this single-threaded process holds
//!   open for writing. `out=$(cmd)` — whose shell holds the job's stdin and
//!   sleeps on the substitution's pipe — does not alert, nor does a reader of
//!   a FIFO or of a pipe a child or another thread can write, while a `cat`
//!   with a private splice pipe does. Kernels before 5.6 sleep
//!   in `pipe_wait`, shared with blocked writers, and stay undetected. The
//!   buffered byte count comes from `FIONREAD` on the write end, and the
//!   tree's CPU-idle confirmation works as on macOS.
//! - Containers: [`CONTAINER_JOB_SCRIPT`] runs the Linux check inside the
//!   container.
//!
//! A reader parked in `poll`/`select` reports wait channels like `do_select`
//! on Linux — and never sets `PIPE_WANTR` on macOS — so it stays undetected
//! on every platform.

use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

/// Poll interval of the local stdin probe.
pub const STDIN_WATCH_INTERVAL: Duration = Duration::from_millis(500);

/// Poll interval of the container stdin probe; every probe is a `docker exec`.
pub const CONTAINER_STDIN_WATCH_INTERVAL: Duration = Duration::from_secs(2);

/// Script run inside a container as `sh -c SCRIPT sh <action> <pid file>`,
/// where the pid file is the one the exec's wrapper records its own
/// in-container pid in. Action `probe` prints `blocked <ticks>` when a
/// thread of that process tree is blocked reading the job's stdin pipe, and
/// an error when it cannot probe ([`parse_container_probe`]); action `kill`
/// kills the whole tree, exiting 3 when no pid is recorded.
pub const CONTAINER_JOB_SCRIPT: &str = include_str!("../templates/container_job.sh");

/// Script every container exec runs under, as
/// `sh -c SCRIPT sh <pid file> <script>`: it records its own in-container
/// pid in the pid file for
/// [`CONTAINER_JOB_SCRIPT`], runs the script in a child shell and removes the
/// file when it exits.
pub const CONTAINER_EXEC_WRAPPER: &str = include_str!("../templates/container_exec_wrapper.sh");

/// The tick rate of the CPU times `/proc` reports, `USER_HZ`: the kernel's
/// user-space ABI fixes it at 100 on every architecture containers run on.
const USER_HZ: u64 = 100;

/// What a container exec's stdin watch reports over its `stdin_blocked`
/// channel.
#[derive(Debug)]
pub enum StdinWatchSignal {
    /// The job's stdin pipe has a reader waiting on it; one send per
    /// blocking episode.
    Blocked,
    /// The watch could not probe the job — carries the probe's error. It is
    /// the last signal the watch sends for this exec: detection cannot run.
    Failed(String),
}

/// What one probe saw on a job's stdin pipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// Nobody is waiting to read the pipe, or it holds buffered data.
    Clear,
    /// A reader is waiting on the empty pipe, but the alert also requires
    /// the job's process tree to have used no CPU since an earlier probe —
    /// on macOS because the kernel flag can be stale, elsewhere because a
    /// tree doing real work is not stuck; carries the tree's total CPU time.
    BlockedAtCpu(Duration),
}

/// Number of writes the agent has fed to a job's stdin.
///
/// The detector compares it between probes: a job that was fed has left its
/// blocking episode even when it blocks again before the next probe.
#[derive(Debug, Clone, Default)]
pub struct FeedCounter(Arc<AtomicU64>);

impl FeedCounter {
    /// Records one write to the job's stdin.
    pub fn record(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }

    /// Writes recorded so far.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}

/// Turns probes into one alert per blocking episode.
///
/// An episode ends when a probe sees the pipe clear or the job was fed since
/// the previous probe; the next blocked probe then alerts again.
#[derive(Debug, Default)]
pub struct EpisodeDetector {
    fed: u64,
    alerted: bool,
    cpu_at_first_sight: Option<Duration>,
}

impl EpisodeDetector {
    /// Feeds one probe; returns whether it opens an episode that must alert.
    pub fn observe(&mut self, fed: u64, probe: Probe) -> bool {
        if fed != self.fed {
            *self = Self {
                fed,
                ..Self::default()
            };
        }
        match probe {
            Probe::Clear => {
                self.alerted = false;
                self.cpu_at_first_sight = None;
                false
            }
            _ if self.alerted => false,
            Probe::BlockedAtCpu(cpu) => {
                if self.cpu_at_first_sight == Some(cpu) {
                    self.alerted = true;
                    true
                } else {
                    self.cpu_at_first_sight = Some(cpu);
                    false
                }
            }
        }
    }
}

/// Parses the probe script's output: empty for clear, `blocked <ticks>` for
/// a waiting reader with the tree's CPU time in `USER_HZ` ticks.
///
/// # Errors
/// Returns the unexpected output, which the script prints only when it
/// cannot probe (for example when the job's processes deny inspection).
pub fn parse_container_probe(output: &str) -> Result<Probe, String> {
    let line = output.trim();
    if line.is_empty() {
        return Ok(Probe::Clear);
    }
    let error = || format!("stdin probe failed: {line}");
    let ticks = line
        .strip_prefix("blocked ")
        .and_then(|ticks| ticks.parse::<u64>().ok())
        .ok_or_else(error)?;
    Ok(Probe::BlockedAtCpu(Duration::from_nanos(
        ticks.saturating_mul(1_000_000_000 / USER_HZ),
    )))
}

/// Probe of one local job's stdin pipe, built from the write end the broker
/// holds.
#[derive(Debug)]
pub struct LocalStdinProbe(platform::Pipe);

impl LocalStdinProbe {
    /// Identifies the pipe behind `write_end` for the process tree rooted at
    /// `root_pid`.
    ///
    /// # Errors
    /// Returns an error when the pipe cannot be inspected.
    pub fn new(root_pid: u32, write_end: &std::fs::File) -> io::Result<Self> {
        platform::Pipe::new(root_pid, write_end).map(Self)
    }

    /// Looks at the pipe once.
    ///
    /// # Errors
    /// Returns an error when the process tree cannot be inspected. Processes
    /// that exit while being inspected are not errors.
    pub fn probe(&self) -> io::Result<Probe> {
        self.0.probe()
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::{
        fs, io,
        os::{
            fd::AsRawFd,
            unix::fs::{FileTypeExt, MetadataExt},
        },
        path::{Path, PathBuf},
        time::Duration,
    };

    use super::Probe;

    /// Kernel wait functions a thread sleeping in a pipe read reports in
    /// `/proc/<pid>/task/<tid>/wchan`: the sleep is charged to the file
    /// operation (`pipe_read`, `anon_pipe_read`) or to the wait function
    /// (`pipe_wait_readable`) depending on the reader and kernel version.
    /// Kernels before 5.6 sleep in `pipe_wait`, which blocked writers and
    /// FIFO opens share, so they are left undetected rather than risk
    /// reporting a writer.
    pub(super) const PIPE_READ_WCHAN: &[&str] =
        &["anon_pipe_read", "pipe_read", "pipe_wait_readable"];

    /// `O_ACCMODE`, `O_RDONLY` and `O_WRONLY` of `<fcntl.h>`, the access mode
    /// `fdinfo` reports in its octal `flags` line.
    const ACCESS_MODE_MASK: u32 = 0o3;
    const O_RDONLY: u32 = 0;
    const O_WRONLY: u32 = 0o1;
    /// `O_NONBLOCK`, which `fdinfo` reports in the same octal `flags` line.
    #[allow(clippy::cast_sign_loss)]
    const O_NONBLOCK: u32 = libc::O_NONBLOCK as u32;

    #[derive(Debug)]
    pub(super) struct Pipe {
        root: u32,
        inode: u64,
        /// The broker's write end; `FIONREAD` on it reports what the pipe
        /// buffers right now.
        write_end: fs::File,
    }

    impl Pipe {
        pub(super) fn new(root: u32, write_end: &fs::File) -> io::Result<Self> {
            Ok(Self {
                root,
                inode: write_end.metadata()?.ino(),
                write_end: write_end.try_clone()?,
            })
        }

        /// A process of the job tree counts as blocked when one of its
        /// threads sleeps in a pipe read and the job's pipe is the only pipe
        /// that read can be waiting on ([`reader_of_job_pipe`]). The tree's
        /// CPU time rides along for the idle confirmation, as on macOS.
        pub(super) fn probe(&self) -> io::Result<Probe> {
            if self.buffered()? > 0 {
                return Ok(Probe::Clear);
            }
            let mut pending = vec![self.root];
            let mut tree = Vec::new();
            let mut ticks = 0_u64;
            while let Some(pid) = pending.pop() {
                if let Some(stat) = proc_stat(&proc_dir(pid))? {
                    ticks = ticks.saturating_add(stat.ticks);
                }
                let tasks = tasks(pid)?;
                for task in &tasks {
                    pending.extend(children(task)?);
                }
                tree.push((pid, tasks));
            }
            for (pid, tasks) in &tree {
                if reader_of_job_pipe(*pid, tasks, self.inode, &tree)? {
                    return Ok(Probe::BlockedAtCpu(ticks_to_duration(ticks)?));
                }
            }
            Ok(Probe::Clear)
        }

        /// Bytes buffered in the job's stdin pipe right now.
        fn buffered(&self) -> io::Result<libc::c_int> {
            let mut bytes: libc::c_int = 0;
            // SAFETY: `bytes` is a live `c_int`, exactly what `FIONREAD`
            // writes for a pipe descriptor.
            if unsafe { libc::ioctl(self.write_end.as_raw_fd(), libc::FIONREAD, &raw mut bytes) }
                < 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(bytes)
        }
    }

    fn proc_dir(pid: u32) -> PathBuf {
        PathBuf::from(format!("/proc/{pid}"))
    }

    /// The `/proc/<pid>/task/<tid>` directories of a process; empty when it
    /// has exited.
    fn tasks(pid: u32) -> io::Result<Vec<PathBuf>> {
        let entries = match fs::read_dir(proc_dir(pid).join("task")) {
            Ok(entries) => entries,
            Err(error) if gone(&error) => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        entries
            .map(|entry| entry.map(|entry| entry.path()))
            .filter(|path| !matches!(path, Err(error) if gone(error)))
            .collect()
    }

    fn children(task: &Path) -> io::Result<Vec<u32>> {
        match fs::read_to_string(task.join("children")) {
            Ok(children) => Ok(children
                .split_whitespace()
                .filter_map(|pid| pid.parse().ok())
                .collect()),
            Err(error) if gone(&error) => {
                // A `children` file dies with its thread; one missing under a
                // live thread means the kernel hides children entirely.
                match fs::metadata(task) {
                    Err(meta) if gone(&meta) => Ok(Vec::new()),
                    Ok(_) | Err(_) => Err(error),
                }
            }
            Err(error) => Err(error),
        }
    }

    /// Whether `pid`, whose threads are `tasks`, sleeps reading the job's
    /// stdin pipe (`inode`); `tree` is the whole job tree.
    ///
    /// `wchan` names the wait function, not the file, so a sleep in a pipe
    /// read proves a wait on the job's pipe only when nothing could feed any
    /// other pipe or FIFO the process holds open for reading. Each such pipe
    /// makes the wait ambiguous, and the process does not count, unless it
    /// is an anonymous pipe that only this single-threaded process holds open
    /// for writing — the private splice pipe of a `cat`. That excludes the
    /// shell of an `out=$(cmd)`, which sleeps on the substitution's pipe while
    /// holding the job's stdin; a FIFO, which any process may open and feed;
    /// a pipe whose write end a child also holds, as a jobserver's or a
    /// worker pool's; and a pipe another thread of the reader may feed. A
    /// reader of the job's pipe that also holds such a pipe goes undetected.
    ///
    /// An anonymous pipe the process reads only through `O_NONBLOCK`
    /// descriptors does not count: a read on those returns `EAGAIN` instead
    /// of sleeping, so the sleeping thread cannot be waiting on that pipe.
    fn reader_of_job_pipe(
        pid: u32,
        tasks: &[PathBuf],
        inode: u64,
        tree: &[(u32, Vec<PathBuf>)],
    ) -> io::Result<bool> {
        if !waits_on_pipe(pid, tasks)? {
            return Ok(false);
        }
        let Some(fds) = pipe_fds(pid)? else {
            return Ok(false);
        };
        if !fds
            .iter()
            .any(|fd| fd.mode.reads() && fd.inode == Some(inode))
        {
            return Ok(false);
        }
        let mut self_fed = Vec::new();
        for fd in fds.iter().filter(|fd| {
            fd.mode.reads() && fd.inode != Some(inode) && !only_read_nonblocking(&fds, fd.inode)
        }) {
            let Some(other) = fd.inode else {
                return Ok(false);
            };
            let fed_here = fds
                .iter()
                .any(|end| end.inode == Some(other) && end.mode.writes());
            if !fed_here || tasks.len() != 1 {
                return Ok(false);
            }
            self_fed.push(other);
        }
        if !self_fed.is_empty() {
            for (other_pid, _) in tree.iter().filter(|(other_pid, _)| *other_pid != pid) {
                let Some(other_fds) = pipe_fds(*other_pid)? else {
                    continue;
                };
                if other_fds.iter().any(|end| {
                    end.mode.writes() && end.inode.is_some_and(|inode| self_fed.contains(&inode))
                }) {
                    return Ok(false);
                }
            }
        }
        // `wchan` and the descriptors are read at different instants, so the
        // sleep must be seen on both sides of reading them: a process that
        // closed a pipe end and then went to sleep on it would otherwise look
        // like one that held both ends.
        waits_on_pipe(pid, tasks)
    }

    /// One pipe or FIFO descriptor of a process.
    struct PipeFd {
        /// The pipe's inode for an anonymous pipe, `None` for a FIFO.
        inode: Option<u64>,
        mode: AccessMode,
        /// Whether the open file description has `O_NONBLOCK` set.
        nonblocking: bool,
    }

    /// Whether `inode` names an anonymous pipe that every read-capable
    /// descriptor in `fds` holding it opened with `O_NONBLOCK`.
    fn only_read_nonblocking(fds: &[PipeFd], inode: Option<u64>) -> bool {
        inode.is_some()
            && fds
                .iter()
                .filter(|fd| fd.inode == inode && fd.mode.reads())
                .all(|fd| fd.nonblocking)
    }

    /// The pipe and FIFO descriptors of `pid`; `None` when it has exited.
    fn pipe_fds(pid: u32) -> io::Result<Option<Vec<PipeFd>>> {
        let fd_dir = proc_dir(pid).join("fd");
        let entries = match fs::read_dir(&fd_dir) {
            Ok(entries) => entries,
            Err(error) if gone(&error) => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut fds = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if gone(&error) => continue,
                Err(error) => return Err(error),
            };
            let path = entry.path();
            match fs::metadata(&path) {
                Ok(meta) if meta.file_type().is_fifo() => {}
                Ok(_) => continue,
                Err(error) if gone(&error) => continue,
                Err(error) => return Err(error),
            }
            let Some((mode, nonblocking)) = open_flags(pid, &entry.file_name())? else {
                continue;
            };
            // The link target names an anonymous pipe as `pipe:[<inode>]` and
            // a FIFO by its path.
            let target = match fs::read_link(&path) {
                Ok(target) => target,
                Err(error) if gone(&error) => continue,
                Err(error) => return Err(error),
            };
            fds.push(PipeFd {
                inode: pipe_inode(&target),
                mode,
                nonblocking,
            });
        }
        Ok(Some(fds))
    }

    /// How a descriptor is open.
    #[derive(Clone, Copy)]
    struct AccessMode(u32);

    impl AccessMode {
        const fn reads(self) -> bool {
            self.0 != O_WRONLY
        }

        const fn writes(self) -> bool {
            self.0 != O_RDONLY
        }
    }

    /// How descriptor `fd` of `pid` is open, from the access mode and
    /// `O_NONBLOCK` in its `fdinfo`; `None` for a descriptor that closed
    /// meanwhile.
    fn open_flags(pid: u32, fd: &std::ffi::OsStr) -> io::Result<Option<(AccessMode, bool)>> {
        let info = match fs::read_to_string(proc_dir(pid).join("fdinfo").join(fd)) {
            Ok(info) => info,
            Err(error) if gone(&error) => return Ok(None),
            Err(error) => return Err(error),
        };
        let flags = info
            .lines()
            .find_map(|line| line.strip_prefix("flags:"))
            .and_then(|flags| u32::from_str_radix(flags.trim(), 8).ok())
            .ok_or_else(|| {
                io::Error::other(format!("malformed fdinfo of fd {} of {pid}", fd.display()))
            })?;
        Ok(Some((
            AccessMode(flags & ACCESS_MODE_MASK),
            flags & O_NONBLOCK != 0,
        )))
    }

    /// Whether some thread of `pid`, whose threads are `tasks`, sleeps in a
    /// pipe read.
    ///
    /// The kernel prints `0` as the `wchan` of a thread the caller may not
    /// ptrace-read, the same as for one it cannot unwind, so a sleeping
    /// thread reporting `0` is checked for access: a denied process is an
    /// error, never one that is not waiting.
    fn waits_on_pipe(pid: u32, tasks: &[PathBuf]) -> io::Result<bool> {
        for task in tasks {
            let Some(stat) = proc_stat(task)? else {
                continue;
            };
            if stat.state != 'S' {
                continue;
            }
            let wchan = match fs::read_to_string(task.join("wchan")) {
                Ok(wchan) => wchan,
                Err(error) if gone(&error) => continue,
                Err(error) => return Err(error),
            };
            match wchan.trim() {
                "0" => match fs::read_link(proc_dir(pid).join("exe")) {
                    Err(error) if !gone(&error) => return Err(error),
                    Ok(_) | Err(_) => {}
                },
                wchan if PIPE_READ_WCHAN.contains(&wchan) => return Ok(true),
                _ => {}
            }
        }
        Ok(false)
    }

    /// A process or thread that exited between listing and reading it.
    fn gone(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
    }

    /// The fields of a `/proc` `stat` dump the probe needs.
    pub(super) struct ProcStat {
        /// Thread or process state letter.
        pub(super) state: char,
        /// User plus system time, in `CLK_TCK` ticks.
        pub(super) ticks: u64,
    }

    /// The `stat` of the process or thread at `dir`; `None` when it exited.
    fn proc_stat(dir: &Path) -> io::Result<Option<ProcStat>> {
        let line = match fs::read_to_string(dir.join("stat")) {
            Ok(line) => line,
            Err(error) if gone(&error) => return Ok(None),
            Err(error) => return Err(error),
        };
        parse_stat(&line)
            .map(Some)
            .ok_or_else(|| io::Error::other(format!("malformed proc stat at {}", dir.display())))
    }

    /// Fields 3 (`state`), 14 (`utime`) and 15 (`stime`) of a `stat` dump;
    /// the command name may contain spaces and parentheses, so the fields
    /// that matter follow the last `") "`.
    pub(super) fn parse_stat(line: &str) -> Option<ProcStat> {
        let mut fields = line.rsplit_once(") ")?.1.split_whitespace();
        let state = fields.next()?.chars().next()?;
        let utime = fields.nth(10)?.parse::<u64>().ok()?;
        let stime = fields.next()?.parse::<u64>().ok()?;
        Some(ProcStat {
            state,
            ticks: utime.checked_add(stime)?,
        })
    }

    /// The tree's CPU jiffies as a duration.
    fn ticks_to_duration(ticks: u64) -> io::Result<Duration> {
        // SAFETY: `sysconf` has no preconditions.
        let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        let hz = u64::try_from(hz)
            .ok()
            .filter(|hz| *hz > 0)
            .ok_or_else(io::Error::last_os_error)?;
        Ok(Duration::from_nanos(
            ticks.saturating_mul(1_000_000_000) / hz,
        ))
    }

    /// The inode in a `pipe:[<inode>]` descriptor target.
    pub(super) fn pipe_inode(target: &Path) -> Option<u64> {
        target
            .to_str()?
            .strip_prefix("pipe:[")?
            .strip_suffix(']')?
            .parse()
            .ok()
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::{
        cell::Cell,
        ffi::c_int,
        fs, io,
        mem::{size_of, zeroed},
        os::fd::AsRawFd,
        time::Duration,
    };

    use super::Probe;

    /// `PROC_PIDFDPIPEINFO` of `<sys/proc_info.h>`.
    const PROC_PIDFDPIPEINFO: c_int = 6;
    /// `PROC_PIDLISTTHREADS` of `<sys/proc_info.h>`.
    const PROC_PIDLISTTHREADS: c_int = 6;
    /// `PIPE_WANTR` of `<sys/pipe.h>`: a reader sleeps waiting for data.
    pub(super) const PIPE_WANTR: i32 = 0x8;

    /// `struct proc_fileinfo` of `<sys/proc_info.h>`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ProcFileInfo {
        openflags: u32,
        status: u32,
        offset: i64,
        r#type: i32,
        guardflags: u32,
    }

    /// `struct pipe_info` of `<sys/proc_info.h>`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PipeInfo {
        pipe_stat: libc::vinfo_stat,
        pipe_handle: u64,
        pipe_peerhandle: u64,
        pipe_status: i32,
        rfu_1: i32,
    }

    /// `struct pipe_fdinfo` of `<sys/proc_info.h>`, what `PROC_PIDFDPIPEINFO` fills.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PipeFdInfo {
        pfi: ProcFileInfo,
        pipeinfo: PipeInfo,
    }

    /// What the kernel reports about one pipe end.
    #[derive(Clone, Copy)]
    pub(super) struct PipeEnd {
        pub(super) handle: u64,
        pub(super) peer_handle: u64,
        pub(super) status: i32,
        pub(super) buffered: i64,
    }

    impl PipeEnd {
        pub(super) const fn reader_waiting(&self) -> bool {
            self.status & PIPE_WANTR != 0 && self.buffered == 0
        }
    }

    /// Reads the pipe end open as `fd` in `pid`; `None` when the process or
    /// descriptor is gone or is not a pipe.
    pub(super) fn pipe_end(pid: c_int, fd: c_int) -> io::Result<Option<PipeEnd>> {
        // SAFETY: `PipeFdInfo` is plain data for which all-zero bytes are valid.
        let mut info: PipeFdInfo = unsafe { zeroed() };
        let size = c_int::try_from(size_of::<PipeFdInfo>()).expect("struct size fits c_int");
        // SAFETY: `info` is a live buffer of exactly `size` bytes, which is
        // what `proc_pidfdinfo` is told it may write.
        let written = unsafe {
            libc::proc_pidfdinfo(pid, fd, PROC_PIDFDPIPEINFO, (&raw mut info).cast(), size)
        };
        if written == size {
            return Ok(Some(PipeEnd {
                handle: info.pipeinfo.pipe_handle,
                peer_handle: info.pipeinfo.pipe_peerhandle,
                status: info.pipeinfo.pipe_status,
                buffered: info.pipeinfo.pipe_stat.vst_size,
            }));
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ESRCH | libc::EBADF | libc::ENOENT | libc::EINVAL) if written <= 0 => {
                Ok(None)
            }
            _ => Err(error),
        }
    }

    /// Fills a vector from a `proc_*` call that reports the byte count it
    /// wrote. Some flavors also report the size they need for a null buffer,
    /// others report nothing, so the buffer grows until it is not full.
    fn list<T: Copy>(
        zero: T,
        call: impl Fn(*mut libc::c_void, c_int) -> c_int,
    ) -> io::Result<Vec<T>> {
        let item = size_of::<T>();
        let mut capacity = match call(std::ptr::null_mut(), 0) {
            needed if needed > 0 => usize::try_from(needed).expect("positive") / item + 16,
            _ => 64,
        };
        loop {
            let mut items = vec![zero; capacity];
            let bytes = c_int::try_from(capacity * item).expect("buffer fits c_int");
            // libproc reports failure as 0 with `errno` set, never a negative
            // count, so `errno` is meaningful only when cleared beforehand.
            // SAFETY: `__error` returns the calling thread's own errno slot.
            unsafe { *libc::__error() = 0 };
            let written = call(items.as_mut_ptr().cast(), bytes);
            if written < 0 {
                return gone_or(io::Error::last_os_error());
            }
            if written == 0 {
                let error = io::Error::last_os_error();
                return match error.raw_os_error() {
                    // The call succeeded with an empty list.
                    Some(0) | None => Ok(Vec::new()),
                    _ => gone_or(error),
                };
            }
            let count = usize::try_from(written).expect("non-negative") / item;
            if count < capacity {
                items.truncate(count);
                return Ok(items);
            }
            capacity *= 2;
        }
    }

    /// An empty list for a process that exited, the error otherwise.
    fn gone_or<T>(error: io::Error) -> io::Result<Vec<T>> {
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(Vec::new())
        } else {
            Err(error)
        }
    }

    fn child_pids(pid: c_int) -> io::Result<Vec<c_int>> {
        list(0, |buffer, size| {
            // SAFETY: the buffer is `size` bytes of live memory, or null with size 0.
            let pids = unsafe { libc::proc_listchildpids(pid, buffer, size) };
            // Unlike its siblings this call reports a count of pids, not bytes.
            pids.saturating_mul(c_int::try_from(size_of::<c_int>()).expect("fits c_int"))
        })
        .map(|pids| pids.into_iter().filter(|child| *child > 0).collect())
    }

    fn open_pipes(pid: c_int) -> io::Result<Vec<c_int>> {
        // SAFETY: `proc_fdinfo` is plain data for which all-zero bytes are valid.
        let zero: libc::proc_fdinfo = unsafe { zeroed() };
        list(zero, |buffer, size| {
            // SAFETY: the buffer is `size` bytes of live memory, or null with size 0.
            unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDLISTFDS, 0, buffer, size) }
        })
        .map(|fds| {
            fds.into_iter()
                .filter(|fd| i32::try_from(fd.proc_fdtype) == Ok(libc::PROX_FDTYPE_PIPE))
                .map(|fd| fd.proc_fd)
                .collect()
        })
    }

    /// User plus system CPU time of every thread in `pid` right now.
    fn cpu_time(pid: c_int) -> io::Result<Duration> {
        let handles = list(0_u64, |buffer, size| {
            // SAFETY: the buffer is `size` bytes of live memory, or null with size 0.
            unsafe { libc::proc_pidinfo(pid, PROC_PIDLISTTHREADS, 0, buffer, size) }
        })?;
        let size = c_int::try_from(size_of::<libc::proc_threadinfo>()).expect("fits c_int");
        let mut nanos = 0_u64;
        for handle in handles {
            // SAFETY: `proc_threadinfo` is plain data for which all-zero bytes are valid.
            let mut info: libc::proc_threadinfo = unsafe { zeroed() };
            // SAFETY: `info` is a live buffer of exactly `size` bytes.
            let written = unsafe {
                libc::proc_pidinfo(
                    pid,
                    libc::PROC_PIDTHREADINFO,
                    handle,
                    (&raw mut info).cast(),
                    size,
                )
            };
            // A thread that exited since it was listed reports nothing.
            if written == size {
                nanos =
                    nanos.saturating_add(info.pth_user_time.saturating_add(info.pth_system_time));
            }
        }
        Ok(Duration::from_nanos(nanos))
    }

    #[derive(Debug)]
    pub(super) struct Pipe {
        root: c_int,
        /// The handle the read end of the job's stdin pipe reports.
        reader_handle: u64,
        /// Where the read end was last found, tried before rescanning the tree.
        last_seen: Cell<Option<(c_int, c_int)>>,
    }

    impl Pipe {
        /// The sandbox hands the broker only the write end, so the read end
        /// is found in the job's process tree through the handle the write
        /// end reports for its peer.
        pub(super) fn new(root: u32, write_end: &fs::File) -> io::Result<Self> {
            let own = c_int::try_from(std::process::id()).map_err(io::Error::other)?;
            let end = pipe_end(own, write_end.as_raw_fd())?
                .ok_or_else(|| io::Error::other("stdin write end is not a pipe"))?;
            Ok(Self {
                root: c_int::try_from(root).map_err(io::Error::other)?,
                reader_handle: end.peer_handle,
                last_seen: Cell::new(None),
            })
        }

        pub(super) fn probe(&self) -> io::Result<Probe> {
            let Some((tree, reader)) = self.find_reader()? else {
                return Ok(Probe::Clear);
            };
            if !reader.reader_waiting() {
                return Ok(Probe::Clear);
            }
            let mut cpu = Duration::ZERO;
            for pid in tree {
                cpu = cpu.saturating_add(cpu_time(pid)?);
            }
            Ok(Probe::BlockedAtCpu(cpu))
        }

        /// The process tree and the job's stdin read end, when some process
        /// of the tree still holds it.
        fn find_reader(&self) -> io::Result<Option<(Vec<c_int>, PipeEnd)>> {
            let tree = self.tree()?;
            if let Some((pid, fd)) = self.last_seen.get()
                && let Some(end) = pipe_end(pid, fd)?
                && end.handle == self.reader_handle
            {
                return Ok(Some((tree, end)));
            }
            for pid in &tree {
                for fd in open_pipes(*pid)? {
                    if let Some(end) = pipe_end(*pid, fd)?
                        && end.handle == self.reader_handle
                    {
                        self.last_seen.set(Some((*pid, fd)));
                        return Ok(Some((tree, end)));
                    }
                }
            }
            self.last_seen.set(None);
            Ok(None)
        }

        /// The root and all its descendants.
        fn tree(&self) -> io::Result<Vec<c_int>> {
            let mut tree = vec![self.root];
            let mut next = 0;
            while let Some(pid) = tree.get(next).copied() {
                next += 1;
                tree.extend(child_pids(pid)?);
            }
            Ok(tree)
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use std::io;

    use super::Probe;

    /// This platform exposes no way to ask who reads a pipe, so a job's stdin
    /// is never reported blocked.
    #[derive(Debug)]
    pub(super) struct Pipe;

    impl Pipe {
        pub(super) fn new(_root: u32, _write_end: &std::fs::File) -> io::Result<Self> {
            Ok(Self)
        }

        pub(super) fn probe(&self) -> io::Result<Probe> {
            Ok(Probe::Clear)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EpisodeDetector, Probe, parse_container_probe};
    use std::time::Duration;

    #[test]
    fn detector_alerts_once_per_episode() {
        let mut detector = EpisodeDetector::default();
        let ms = Duration::from_millis;
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(!detector.observe(0, Probe::Clear));
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(detector.observe(0, Probe::BlockedAtCpu(ms(5))));
    }

    #[test]
    fn detector_realerts_after_feed_without_a_clear_probe() {
        let mut detector = EpisodeDetector::default();
        let ms = Duration::from_millis;
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(!detector.observe(1, Probe::BlockedAtCpu(ms(7))));
        assert!(detector.observe(1, Probe::BlockedAtCpu(ms(7))));
    }

    #[test]
    fn detector_needs_two_idle_probes_when_cpu_is_reported() {
        let mut detector = EpisodeDetector::default();
        let ms = Duration::from_millis;
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(9))));
        assert!(detector.observe(0, Probe::BlockedAtCpu(ms(9))));
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(9))));
    }

    #[test]
    fn detector_forgets_cpu_baseline_when_clear() {
        let mut detector = EpisodeDetector::default();
        let ms = Duration::from_millis;
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(5))));
        assert!(!detector.observe(0, Probe::Clear));
        assert!(!detector.observe(0, Probe::BlockedAtCpu(ms(5))));
    }

    #[test]
    fn container_probe_output_is_parsed() {
        assert_eq!(parse_container_probe(""), Ok(Probe::Clear));
        assert_eq!(
            parse_container_probe("blocked 150\n"),
            Ok(Probe::BlockedAtCpu(Duration::from_millis(1500)))
        );
        assert!(parse_container_probe("unsupported architecture").is_err());
        assert!(parse_container_probe("blocked").is_err());
        assert!(parse_container_probe("blocked 150 100").is_err());
    }

    /// The tests below spawn real `sh` jobs and read their `/proc` entries,
    /// so they run on Unix only.
    #[cfg(unix)]
    mod unix {
        use super::super::{EpisodeDetector, LocalStdinProbe, Probe};
        #[cfg(target_os = "macos")]
        use std::io::PipeReader;
        use std::{
            fs::File,
            io::Write,
            os::{fd::OwnedFd, unix::process::CommandExt},
            process::{Child, Command, Stdio},
            time::{Duration, Instant},
        };

        /// A job in its own process group, all of which is killed when the
        /// test ends.
        struct Job {
            child: Child,
            input: File,
            /// A second handle on the pipe's read end, through which the macOS
            /// tests observe the reader-waiting flag directly.
            #[cfg(target_os = "macos")]
            reader: PipeReader,
        }

        impl Drop for Job {
            fn drop(&mut self) {
                // SAFETY: signals the process group this test created.
                unsafe { libc::kill(-self.child.id().cast_signed(), libc::SIGKILL) };
                let _ = self.child.wait();
            }
        }

        impl Job {
            fn spawn(script: &str) -> Self {
                let mut command = Command::new("sh");
                command.args(["-c", script]);
                Self::spawn_command(command)
            }

            /// Spawns `command` with a pipe the test feeds as its stdin.
            fn spawn_command(mut command: Command) -> Self {
                let (reader, writer) = std::io::pipe().expect("pipe");
                #[cfg(target_os = "macos")]
                let held = reader.try_clone().expect("clone read end");
                let child = command
                    .process_group(0)
                    .stdin(Stdio::from(reader))
                    .stdout(Stdio::null())
                    .spawn()
                    .expect("the job should spawn");
                Self {
                    child,
                    input: OwnedFd::from(writer).into(),
                    #[cfg(target_os = "macos")]
                    reader: held,
                }
            }

            fn probe(&self) -> LocalStdinProbe {
                LocalStdinProbe::new(self.child.id(), &self.input).expect("probe the stdin pipe")
            }

            fn feed(&mut self, bytes: &[u8]) {
                self.input.write_all(bytes).expect("feed stdin");
            }

            /// What `/proc` shows of the job's tree, for a failure message.
            #[cfg(target_os = "linux")]
            fn dump(&self) -> String {
                use std::os::unix::fs::MetadataExt;
                proc_dump::render(
                    self.child.id(),
                    self.input.metadata().ok().map(|meta| meta.ino()),
                )
            }

            /// The dump is Linux-only; other platforms report the probe alone.
            #[cfg(not(target_os = "linux"))]
            #[allow(clippy::unused_self)]
            fn dump(&self) -> String {
                String::from("(no /proc on this platform)")
            }
        }

        /// A rendering of a job's process tree from `/proc`, so a probe that
        /// finds no reader says what the probe could see.
        #[cfg(target_os = "linux")]
        mod proc_dump {
            use std::{fmt::Write as _, fs, io, path::Path};

            /// `read_to_string` of `path`, trimmed, or the error as `<EACCES>`.
            fn text(path: impl AsRef<Path>) -> String {
                match fs::read_to_string(path) {
                    Ok(text) => text.trim().to_owned(),
                    Err(error) => failure(&error),
                }
            }

            fn failure(error: &io::Error) -> String {
                error.raw_os_error().map_or_else(
                    || format!("<{:?}>", error.kind()),
                    |errno| format!("<{}>", errno_name(errno)),
                )
            }

            fn errno_name(errno: i32) -> String {
                match errno {
                    libc::EACCES => "EACCES".to_owned(),
                    libc::EPERM => "EPERM".to_owned(),
                    libc::ENOENT => "ENOENT".to_owned(),
                    libc::ESRCH => "ESRCH".to_owned(),
                    libc::EIO => "EIO".to_owned(),
                    other => format!("errno {other}"),
                }
            }

            /// The state letter of `/proc/.../stat`, after the `(comm)` field.
            fn state_of(stat: &str) -> String {
                stat.rsplit_once(')')
                    .and_then(|(_, rest)| rest.split_whitespace().next())
                    .map_or_else(|| stat.to_owned(), str::to_owned)
            }

            /// The lines of `fdinfo` starting with one of `keys`, on one line.
            fn fdinfo(path: &Path, keys: &[&str]) -> String {
                match fs::read_to_string(path) {
                    Ok(info) => info
                        .lines()
                        .filter(|line| keys.iter().any(|key| line.starts_with(key)))
                        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
                        .collect::<Vec<_>>()
                        .join(" "),
                    Err(error) => failure(&error),
                }
            }

            /// The tree rooted at `root`, with the job pipe's inode when known.
            pub(super) fn render(root: u32, pipe_inode: Option<u64>) -> String {
                let mut out = String::new();
                let _ = writeln!(out, "job pipe inode: {pipe_inode:?}, root pid {root}");
                process(&mut out, root, 0);
                out
            }

            fn process(out: &mut String, pid: u32, depth: usize) {
                let indent = "  ".repeat(depth);
                let dir = format!("/proc/{pid}");
                let uid = fs::read_to_string(format!("{dir}/status")).map_or_else(
                    |error| failure(&error),
                    |status| {
                        status
                            .lines()
                            .find_map(|line| line.strip_prefix("Uid:"))
                            .map_or_else(|| "<none>".to_owned(), |uid| uid.trim().to_owned())
                    },
                );
                let exe = fs::read_link(format!("{dir}/exe"))
                    .map_or_else(|error| failure(&error), |path| path.display().to_string());
                let _ = writeln!(
                    out,
                    "{indent}pid {pid} comm={} exe={exe} uid={uid} state={}",
                    text(format!("{dir}/comm")),
                    state_of(&text(format!("{dir}/stat"))),
                );
                let mut tids = Vec::new();
                match fs::read_dir(format!("{dir}/task")) {
                    Ok(entries) => tids.extend(entries.flatten().map(|entry| entry.file_name())),
                    Err(error) => {
                        let _ = writeln!(out, "{indent}  tasks={}", failure(&error));
                    }
                }
                tids.sort();
                let mut children = Vec::new();
                for tid in &tids {
                    let task = format!("{dir}/task/{}", tid.to_string_lossy());
                    let _ = writeln!(
                        out,
                        "{indent}  tid {} state={} wchan={}",
                        tid.to_string_lossy(),
                        state_of(&text(format!("{task}/stat"))),
                        text(format!("{task}/wchan")),
                    );
                    for child in text(format!("{task}/children")).split_whitespace() {
                        match child.parse::<u32>() {
                            Ok(child) => children.push(child),
                            Err(_) => {
                                let _ = writeln!(out, "{indent}  children={child}");
                            }
                        }
                    }
                }
                match fs::read_dir(format!("{dir}/fd")) {
                    Ok(entries) => {
                        let mut fds: Vec<_> =
                            entries.flatten().map(|entry| entry.file_name()).collect();
                        fds.sort_by_key(|fd| {
                            fd.to_string_lossy().parse::<u32>().unwrap_or(u32::MAX)
                        });
                        for fd in fds {
                            let name = fd.to_string_lossy();
                            let target = fs::read_link(format!("{dir}/fd/{name}")).map_or_else(
                                |error| failure(&error),
                                |path| path.display().to_string(),
                            );
                            let info = fdinfo(
                                Path::new(&format!("{dir}/fdinfo/{name}")),
                                &["flags", "pos"],
                            );
                            let _ = writeln!(out, "{indent}  fd {name} -> {target} {info}");
                        }
                    }
                    Err(error) => {
                        let _ = writeln!(out, "{indent}  fds={}", failure(&error));
                    }
                }
                for child in children {
                    process(out, child, depth + 1);
                }
            }
        }

        /// Probes every 100 ms until `wanted` accepts a probe or 10 s pass.
        fn probe_until(probe: &LocalStdinProbe, wanted: impl Fn(Probe) -> bool) -> Probe {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let seen = probe.probe().expect("probe");
                if wanted(seen) || Instant::now() >= deadline {
                    return seen;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }

        /// A reader blocked in a grandchild of the job's root process is found.
        #[test]
        fn grandchild_reader_is_blocked() {
            // The root closes its stdin, so only the grandchild holds the pipe;
            // dash gives a background job /dev/null as stdin before applying
            // its redirections, hence the saved descriptor.
            let job =
                Job::spawn("exec 3<&0; sh -c 'cat >/dev/null; :' <&3 3<&- & exec 3<&- <&-; wait");
            let probe = job.probe();
            let seen = probe_until(&probe, |seen| seen != Probe::Clear);
            assert_ne!(
                seen,
                Probe::Clear,
                "a grandchild blocked in read is a reader\n{}",
                job.dump()
            );
        }

        #[test]
        fn job_that_never_reads_is_clear() {
            let job = Job::spawn("sleep 30");
            let probe = job.probe();
            for _ in 0..5 {
                assert_eq!(probe.probe().expect("probe"), Probe::Clear);
                std::thread::sleep(Duration::from_millis(100));
            }
        }

        #[test]
        fn unfed_reader_alerts_once_and_feeding_clears_it() {
            let mut job = Job::spawn("read line; sleep 30");
            let probe = job.probe();
            probe_until(&probe, |seen| seen != Probe::Clear);
            let mut detector = EpisodeDetector::default();
            let mut alerts = 0;
            for _ in 0..4 {
                std::thread::sleep(Duration::from_millis(100));
                alerts += usize::from(detector.observe(0, probe.probe().expect("probe")));
            }
            assert_eq!(alerts, 1, "a stuck reader alerts exactly once");

            job.feed(b"input\n");
            let seen = probe_until(&probe, |seen| seen == Probe::Clear);
            assert_eq!(seen, Probe::Clear, "a fed reader is not waiting");
        }

        /// The `PIPE_WANTR` flag as XNU reports it for the read end the test
        /// holds.
        #[cfg(target_os = "macos")]
        mod macos {
            use crate::stdin_watch::platform::{PIPE_WANTR, pipe_end};

            use super::*;
            use std::os::fd::AsRawFd;

            fn wants_read(job: &Job) -> bool {
                let own = i32::try_from(std::process::id()).expect("pid fits c_int");
                let end = pipe_end(own, job.reader.as_raw_fd())
                    .expect("query the pipe")
                    .expect("the held read end is a pipe");
                end.status & PIPE_WANTR != 0 && end.buffered == 0
            }

            fn wait_for(job: &Job, wanted: bool) {
                let deadline = Instant::now() + Duration::from_secs(10);
                while wants_read(job) != wanted {
                    assert!(Instant::now() < deadline, "flag never became {wanted}");
                    std::thread::sleep(Duration::from_millis(50));
                }
            }

            #[test]
            fn waiting_reader_flag_is_set_and_cleared_by_feeding() {
                let mut job = Job::spawn("read line; sleep 30");
                wait_for(&job, true);
                job.feed(b"input\n");
                wait_for(&job, false);
                std::thread::sleep(Duration::from_millis(300));
                assert!(
                    !wants_read(&job),
                    "a reader that took its data is not waiting"
                );
            }

            /// A read interrupted by a signal leaves the kernel flag set while the
            /// process does real work, so the detector must not report it.
            #[test]
            fn interrupted_reader_that_moved_on_is_not_reported() {
                let job =
                    Job::spawn("perl -e '$SIG{USR1} = sub {}; sysread(STDIN, $b, 1); 1 while 1'");
                wait_for(&job, true);
                let probe = job.probe();
                // SAFETY: signals the child this test spawned.
                unsafe { libc::kill(job.child.id().cast_signed(), libc::SIGUSR1) };
                let mut detector = EpisodeDetector::default();
                for _ in 0..8 {
                    std::thread::sleep(Duration::from_millis(200));
                    assert!(
                        !detector.observe(0, probe.probe().expect("probe")),
                        "a process spinning after an interrupted read is not blocked on stdin"
                    );
                }
                assert!(
                    wants_read(&job),
                    "the kernel leaves the flag set after an interrupted read; the CPU check is what rejects it"
                );
            }
        }

        #[cfg(target_os = "linux")]
        mod linux {
            use super::*;
            use crate::stdin_watch::platform::{PIPE_READ_WCHAN, parse_stat, pipe_inode};
            use crate::stdin_watch::{
                CONTAINER_EXEC_WRAPPER, CONTAINER_JOB_SCRIPT, parse_container_probe,
            };
            use std::os::fd::AsRawFd;
            use std::path::{Path, PathBuf};

            #[test]
            fn stats_yield_state_and_cpu() {
                let stat =
                    parse_stat("123 (sh) S 1 2 3 4 5 6 7 8 9 10 30 40").expect("stat parses");
                assert_eq!(stat.state, 'S');
                assert_eq!(stat.ticks, 70);
                // The command name may contain spaces and parentheses.
                let stat = parse_stat("124 (weird ) name) R 1 2 3 4 5 6 7 8 9 10 11 12")
                    .expect("stat parses");
                assert_eq!(stat.state, 'R');
                assert_eq!(stat.ticks, 23);
            }

            #[test]
            fn pipe_wait_names_cover_known_kernel_names() {
                for name in ["pipe_read", "anon_pipe_read", "pipe_wait_readable"] {
                    assert!(
                        PIPE_READ_WCHAN.contains(&name),
                        "{name} must be a pipe-read wait name"
                    );
                }
                // A poll/select waiter, a sleeper, or a blocked writer on a
                // kernel that shares `pipe_wait` must not count.
                assert!(!PIPE_READ_WCHAN.contains(&"pipe_wait"));
                assert!(!PIPE_READ_WCHAN.contains(&"do_select"));
                assert!(!PIPE_READ_WCHAN.contains(&"hrtimer_nanosleep"));
            }

            /// Jobs that hold the job's stdin but wait on something else — a
            /// timer, another pipe, a FIFO — are not blocked on stdin, however
            /// long they are watched: neither the in-process probe nor the
            /// container script may report them. Each runs under the container
            /// exec wrapper, as a container exec does.
            #[test]
            fn jobs_waiting_on_something_else_are_clear() {
                let mut failures = Vec::new();
                for script in [
                    "r=$(sleep 30); echo \"$r\"",
                    "out=$(sleep 30 | cat); echo \"$out\"",
                    "python3 -c 'import subprocess; subprocess.check_output([\"sleep\", \"30\"])'",
                    "sleep 30",
                    // A FIFO held open read-write, which any process may open
                    // and feed at any time.
                    "mkfifo f; exec 3<>f; read -r l <&3",
                    "mkfifo f; exec 3<>f; while read -r l; do :; done <&3",
                    // A FIFO that replaced the job's stdin: the job no longer
                    // reads the pipe the agent feeds.
                    "mkfifo f; exec <>f; read -r l",
                    // Worker pools: workers hold write ends of pipes the
                    // parent's threads read, under both start methods.
                    "python3 -c 'from multiprocessing import Pool; import time; Pool(2).map(time.sleep, [30, 30])'",
                    "python3 -c 'import multiprocessing as m, time; m.get_context(\"fork\").Pool(2).map(time.sleep, [30, 30])'",
                    // A pipe the reader holds both ends of while a child holds
                    // the write end too, as a jobserver's.
                    "python3 -c 'import os, time; r, w = os.pipe(); os.fork() or (time.sleep(30), os.write(w, b\"x\"), os._exit(0)); os.read(r, 1)'",
                    // A pipe another thread of the reader will feed.
                    "python3 -c 'import os, threading, time; r, w = os.pipe(); threading.Thread(target=os.read, args=(r, 1)).start(); time.sleep(30); os.write(w, b\"x\")'",
                ] {
                    let dir = tempfile::tempdir().expect("temp dir");
                    let (job, pid_file) = spawn_wrapped(script, dir.path());
                    // Let the job reach its steady wait before judging it.
                    std::thread::sleep(Duration::from_secs(1));
                    let probe = job.probe();
                    let mut alerts = (0, 0);
                    for _ in 0..8 {
                        alerts.0 += usize::from(probe.probe().expect("probe") != Probe::Clear);
                        alerts.1 += usize::from(run_container_probe(&pid_file) != Ok(Probe::Clear));
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    if alerts != (0, 0) {
                        failures.push(format!(
                            "{script}: {} of 8 local and {} of 8 container probes not clear",
                            alerts.0, alerts.1
                        ));
                    }
                }
                assert!(failures.is_empty(), "{failures:#?}");
            }

            /// Runs `cat` under the exec wrapper holding, as fd 3, the read end
            /// of an unrelated pipe whose write end stays with the test, outside
            /// the job. Returns the job, its pid file and that write end.
            fn spawn_with_foreign_pipe(dir: &Path, nonblocking: bool) -> (Job, PathBuf, File) {
                let (reader, writer) = std::io::pipe().expect("pipe");
                let reader = OwnedFd::from(reader);
                if nonblocking {
                    // SAFETY: sets a flag on a descriptor this test owns.
                    let set = unsafe {
                        let flags = libc::fcntl(reader.as_raw_fd(), libc::F_GETFL);
                        libc::fcntl(reader.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK)
                    };
                    assert_ne!(set, -1, "set O_NONBLOCK");
                }
                let pid_file = dir.join("job.pid");
                let mut command = Command::new("sh");
                command
                    .args(["-c", CONTAINER_EXEC_WRAPPER, "sh"])
                    .arg(&pid_file)
                    .arg("cat")
                    .current_dir(dir);
                let fd = reader.as_raw_fd();
                // SAFETY: only `dup2`, which is async-signal-safe, runs between
                // fork and exec; it clears close-on-exec on the copy.
                unsafe {
                    command.pre_exec(move || {
                        if libc::dup2(fd, 3) == -1 {
                            return Err(std::io::Error::last_os_error());
                        }
                        Ok(())
                    });
                }
                let job = Job::spawn_command(command);
                (job, pid_file, OwnedFd::from(writer).into())
            }

            /// A reader of the job's pipe that also inherited an `O_NONBLOCK`
            /// read end of an unrelated pipe, fed from outside the job, is
            /// blocked: a read on that end cannot be what it sleeps on.
            #[test]
            fn nonblocking_foreign_pipe_does_not_hide_the_reader() {
                let dir = tempfile::tempdir().expect("temp dir");
                let (job, pid_file, _foreign_writer) = spawn_with_foreign_pipe(dir.path(), true);
                let probe = job.probe();
                let seen = probe_until(&probe, |seen| seen != Probe::Clear);
                assert_ne!(seen, Probe::Clear, "local probe\n{}", job.dump());
                let deadline = Instant::now() + Duration::from_secs(10);
                let container = loop {
                    let seen = run_container_probe(&pid_file);
                    if seen != Ok(Probe::Clear) || Instant::now() >= deadline {
                        break seen;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                };
                assert!(
                    matches!(container, Ok(Probe::BlockedAtCpu(_))),
                    "container probe: {container:?}\n{}",
                    job.dump()
                );
            }

            /// The same reader holding a blocking read end of such a pipe stays
            /// ambiguous, so it is clear.
            #[test]
            fn blocking_foreign_pipe_keeps_the_reader_ambiguous() {
                let dir = tempfile::tempdir().expect("temp dir");
                let (job, pid_file, _foreign_writer) = spawn_with_foreign_pipe(dir.path(), false);
                // Let `cat` reach its steady wait before judging it.
                std::thread::sleep(Duration::from_secs(1));
                let probe = job.probe();
                for _ in 0..8 {
                    assert_eq!(
                        probe.probe().expect("probe"),
                        Probe::Clear,
                        "local probe\n{}",
                        job.dump()
                    );
                    assert_eq!(
                        run_container_probe(&pid_file),
                        Ok(Probe::Clear),
                        "container probe\n{}",
                        job.dump()
                    );
                    std::thread::sleep(Duration::from_millis(100));
                }
            }

            /// `cat` waits in `splice`, a shell in `read`, a grandchild in
            /// either; each is blocked on the job's pipe, for both probes.
            #[test]
            fn readers_of_the_job_pipe_are_blocked() {
                for script in [
                    "cat",
                    "read line; sleep 30",
                    "head -c1; sleep 30",
                    "sh -c 'cat >/dev/null; :'",
                ] {
                    let dir = tempfile::tempdir().expect("temp dir");
                    let (job, pid_file) = spawn_wrapped(script, dir.path());
                    let probe = job.probe();
                    let seen = probe_until(&probe, |seen| seen != Probe::Clear);
                    assert_ne!(seen, Probe::Clear, "probe of {script}\n{}", job.dump());
                    let deadline = Instant::now() + Duration::from_secs(10);
                    let container = loop {
                        let seen = run_container_probe(&pid_file);
                        if seen != Ok(Probe::Clear) || Instant::now() >= deadline {
                            break seen;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    };
                    assert!(
                        matches!(container, Ok(Probe::BlockedAtCpu(_))),
                        "container probe of {script}: {container:?}\n{}",
                        job.dump()
                    );
                }
            }

            /// A job that ended before or while it is probed is clear, not a
            /// failed probe: its wrapper removed the pid file, or the
            /// recorded process is gone.
            #[test]
            fn container_probe_of_an_ended_job_is_clear() {
                let dir = tempfile::tempdir().expect("temp dir");
                let (mut job, pid_file) = spawn_wrapped("exit 0", dir.path());
                job.child.wait().expect("the job ends");
                assert!(!pid_file.exists(), "the wrapper removes its pid file");
                assert_eq!(run_container_probe(&pid_file), Ok(Probe::Clear));

                let mut gone = Command::new("true").spawn().expect("true spawns");
                gone.wait().expect("true ends");
                std::fs::write(&pid_file, format!("{}\n", gone.id())).expect("write pid file");
                assert_eq!(run_container_probe(&pid_file), Ok(Probe::Clear));
            }

            /// The container kill stops the whole job tree, not only its root,
            /// and removes the pid file; a job that recorded no pid is exit 3,
            /// which the exec turns into an error unless the exec has ended.
            #[test]
            fn container_kill_ends_the_whole_job_tree() {
                let dir = tempfile::tempdir().expect("temp dir");
                let (mut job, pid_file) =
                    spawn_wrapped("sh -c 'sleep 300' & sleep 300; wait", dir.path());
                // The wrapper, its script shell, and the script's two sleeps.
                let deadline = Instant::now() + Duration::from_secs(10);
                let tree = loop {
                    let tree = job_tree(job.child.id());
                    if tree.len() >= 5 || Instant::now() >= deadline {
                        break tree;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                };
                assert!(tree.len() >= 5, "job tree {tree:?}");

                let run = run_job_script("kill", &pid_file);
                assert!(
                    run.status.success(),
                    "{}",
                    String::from_utf8_lossy(&run.stdout)
                );
                job.child.wait().expect("the killed wrapper is reaped");
                for pid in tree {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while alive(pid) {
                        assert!(
                            Instant::now() < deadline,
                            "process {pid} of the job survived"
                        );
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
                assert!(!pid_file.exists(), "the kill removes the pid file");
                assert_eq!(
                    run_job_script("kill", &pid_file).status.code(),
                    Some(3),
                    "no recorded pid"
                );
                std::fs::write(&pid_file, "not a pid").expect("write garbage");
                assert_eq!(run_job_script("kill", &pid_file).status.code(), Some(1));
            }

            /// Runs `script` under the container exec wrapper in `dir`, as a
            /// container exec runs it; returns the job and the pid file the
            /// wrapper records its pid in.
            fn spawn_wrapped(script: &str, dir: &Path) -> (Job, PathBuf) {
                let pid_file = dir.join("job.pid");
                let mut command = Command::new("sh");
                command
                    .args(["-c", CONTAINER_EXEC_WRAPPER, "sh"])
                    .arg(&pid_file)
                    .arg(script)
                    .current_dir(dir);
                (Job::spawn_command(command), pid_file)
            }

            /// Runs the container job script on the host, where its `/proc`
            /// reads see the test's jobs.
            fn run_job_script(action: &str, pid_file: &Path) -> std::process::Output {
                Command::new("sh")
                    .args(["-c", CONTAINER_JOB_SCRIPT, "sh", action])
                    .arg(pid_file)
                    .output()
                    .expect("run the job script")
            }

            fn run_container_probe(pid_file: &Path) -> Result<Probe, String> {
                let run = run_job_script("probe", pid_file);
                assert!(
                    run.status.success() || !run.stdout.is_empty(),
                    "probe script failed silently: {}",
                    String::from_utf8_lossy(&run.stderr)
                );
                parse_container_probe(&String::from_utf8_lossy(&run.stdout))
            }

            /// `root` and its descendants, from each thread's `children`.
            fn job_tree(root: u32) -> Vec<u32> {
                let mut tree = vec![root];
                let mut next = 0;
                while let Some(pid) = tree.get(next).copied() {
                    next += 1;
                    let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
                        continue;
                    };
                    for task in tasks.flatten() {
                        let children = std::fs::read_to_string(task.path().join("children"))
                            .unwrap_or_default();
                        tree.extend(
                            children
                                .split_whitespace()
                                .filter_map(|c| c.parse::<u32>().ok()),
                        );
                    }
                }
                tree
            }

            /// Whether `pid` exists and is not a zombie.
            fn alive(pid: u32) -> bool {
                std::fs::read_to_string(format!("/proc/{pid}/stat"))
                    .is_ok_and(|stat| parse_stat(&stat).is_some_and(|stat| stat.state != 'Z'))
            }

            #[test]
            fn only_pipe_targets_have_an_inode() {
                assert_eq!(pipe_inode(Path::new("pipe:[1234]")), Some(1234));
                assert_eq!(pipe_inode(Path::new("socket:[1234]")), None);
                assert_eq!(pipe_inode(Path::new("/dev/null")), None);
            }
        }
    }
}
