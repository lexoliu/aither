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
//! - Linux: a thread of the process tree is in a `read` syscall on a descriptor
//!   whose target is the pipe's inode.
//! - Containers: [`CONTAINER_STDIN_PROBE`] runs the Linux check inside the
//!   container.

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

/// Script run inside a container with the exec's root pid as its argument; it
/// prints `blocked` when a thread of that process tree is blocked reading the
/// pipe that is the root's standard input.
pub const CONTAINER_STDIN_PROBE: &str = include_str!("../templates/stdin_probe.sh");

/// What one probe saw on a job's stdin pipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// Nobody is waiting to read the pipe, or it holds buffered data.
    Clear,
    /// A reader is waiting on the empty pipe.
    Blocked,
    /// A reader is waiting on the empty pipe, but the platform cannot tell a
    /// stuck reader from a stale flag unless the process tree used no CPU
    /// since an earlier probe; carries the tree's total CPU time.
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
            Probe::Blocked => {
                self.alerted = true;
                true
            }
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

/// Parses the probe script's output.
///
/// # Errors
/// Returns the unexpected output, which the script prints only when it cannot
/// probe (for example on an unsupported architecture).
pub fn parse_container_probe(output: &str) -> Result<Probe, String> {
    match output.trim() {
        "" => Ok(Probe::Clear),
        "blocked" => Ok(Probe::Blocked),
        other => Err(format!("stdin probe failed: {other}")),
    }
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
    pub fn probe(&mut self) -> io::Result<Probe> {
        self.0.probe()
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::{
        fs, io,
        os::unix::fs::MetadataExt,
        path::{Path, PathBuf},
    };

    use super::Probe;

    #[derive(Debug)]
    pub(super) struct Pipe {
        root: u32,
        inode: u64,
    }

    impl Pipe {
        pub(super) fn new(root: u32, write_end: &fs::File) -> io::Result<Self> {
            Ok(Self {
                root,
                inode: write_end.metadata()?.ino(),
            })
        }

        pub(super) fn probe(&mut self) -> io::Result<Probe> {
            let mut pending = vec![self.root];
            while let Some(pid) = pending.pop() {
                for task in tasks(pid)? {
                    if blocked_on_pipe(pid, &task, self.inode)? {
                        return Ok(Probe::Blocked);
                    }
                    pending.extend(children(&task)?);
                }
            }
            Ok(Probe::Clear)
        }
    }

    /// The `/proc/<pid>/task/<tid>` directories of a process; empty when it
    /// has exited.
    fn tasks(pid: u32) -> io::Result<Vec<PathBuf>> {
        let entries = match fs::read_dir(format!("/proc/{pid}/task")) {
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
            Err(error) if gone(&error) => Ok(Vec::new()),
            Err(error) => Err(error),
        }
    }

    /// Whether the thread sits in `read`/`readv` on a descriptor that is the
    /// pipe with `inode`.
    fn blocked_on_pipe(pid: u32, task: &Path, inode: u64) -> io::Result<bool> {
        let syscall = match fs::read_to_string(task.join("syscall")) {
            Ok(syscall) => syscall,
            Err(error) if gone(&error) => return Ok(false),
            Err(error) => return Err(error),
        };
        let Some(fd) = read_descriptor(&syscall) else {
            return Ok(false);
        };
        match fs::read_link(format!("/proc/{pid}/fd/{fd}")) {
            Ok(target) => Ok(pipe_inode(&target) == Some(inode)),
            Err(error) if gone(&error) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// A process or thread that exited between listing and reading it.
    fn gone(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
    }

    /// The descriptor of a `read`/`readv` in a `/proc/<pid>/task/<tid>/syscall`
    /// dump (`<nr> <arg0> ... <sp> <pc>`, numbers in decimal or `0x` hex).
    pub(super) fn read_descriptor(dump: &str) -> Option<u64> {
        let mut fields = dump.split_whitespace();
        let number = parse_number(fields.next()?)?;
        if ![libc::SYS_read, libc::SYS_readv]
            .into_iter()
            .any(|read| u64::try_from(read) == Ok(number))
        {
            return None;
        }
        parse_number(fields.next()?)
    }

    fn parse_number(field: &str) -> Option<u64> {
        match field.strip_prefix("0x") {
            Some(hex) => u64::from_str_radix(hex, 16).ok(),
            None => field.parse().ok(),
        }
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
            let written = call(items.as_mut_ptr().cast(), bytes);
            if written < 0 {
                return gone_or(io::Error::last_os_error());
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
        last_seen: Option<(c_int, c_int)>,
    }

    impl Pipe {
        /// The sandbox hands the broker only the write end, so the read end
        /// is found in the job's process tree through the handle the write
        /// end reports for its peer.
        pub(super) fn new(root: u32, write_end: &fs::File) -> io::Result<Self> {
            // SAFETY: `getpid` has no preconditions.
            let own = unsafe { libc::getpid() };
            let end = pipe_end(own, write_end.as_raw_fd())?
                .ok_or_else(|| io::Error::other("stdin write end is not a pipe"))?;
            Ok(Self {
                root: c_int::try_from(root).map_err(io::Error::other)?,
                reader_handle: end.peer_handle,
                last_seen: None,
            })
        }

        pub(super) fn probe(&mut self) -> io::Result<Probe> {
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
        fn find_reader(&mut self) -> io::Result<Option<(Vec<c_int>, PipeEnd)>> {
            let tree = self.tree()?;
            if let Some((pid, fd)) = self.last_seen
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
                        self.last_seen = Some((*pid, fd));
                        return Ok(Some((tree, end)));
                    }
                }
            }
            self.last_seen = None;
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

        pub(super) fn probe(&mut self) -> io::Result<Probe> {
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
        assert!(detector.observe(0, Probe::Blocked));
        assert!(!detector.observe(0, Probe::Blocked));
        assert!(!detector.observe(0, Probe::Clear));
        assert!(detector.observe(0, Probe::Blocked));
    }

    #[test]
    fn detector_realerts_after_feed_without_a_clear_probe() {
        let mut detector = EpisodeDetector::default();
        assert!(detector.observe(0, Probe::Blocked));
        assert!(detector.observe(1, Probe::Blocked));
        assert!(!detector.observe(1, Probe::Blocked));
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
        assert_eq!(parse_container_probe("blocked\n"), Ok(Probe::Blocked));
        assert!(parse_container_probe("unsupported architecture").is_err());
    }

    use super::LocalStdinProbe;
    #[cfg(target_os = "macos")]
    use std::io::PipeReader;
    use std::{
        fs::File,
        io::Write,
        os::fd::OwnedFd,
        process::{Child, Command, Stdio},
        time::Instant,
    };

    /// A child process killed when the test ends.
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
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    impl Job {
        fn spawn(script: &str) -> Self {
            let (reader, writer) = std::io::pipe().expect("pipe");
            #[cfg(target_os = "macos")]
            let held = reader.try_clone().expect("clone read end");
            let child = Command::new("sh")
                .args(["-c", script])
                .stdin(Stdio::from(reader))
                .stdout(Stdio::null())
                .spawn()
                .expect("sh should spawn");
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
    }

    /// Probes every 100 ms until `wanted` accepts a probe or 10 s pass.
    fn probe_until(probe: &mut LocalStdinProbe, wanted: impl Fn(Probe) -> bool) -> Probe {
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
        // The root closes its stdin, so only the grandchild holds the pipe.
        let job = Job::spawn("{ sh -c 'cat >/dev/null; :'; } <&0 & exec <&-; wait");
        let mut probe = job.probe();
        let seen = probe_until(&mut probe, |seen| seen != Probe::Clear);
        assert_ne!(
            seen,
            Probe::Clear,
            "a grandchild blocked in read is a reader"
        );
    }

    #[test]
    fn job_that_never_reads_is_clear() {
        let job = Job::spawn("sleep 30");
        let mut probe = job.probe();
        for _ in 0..5 {
            assert_eq!(probe.probe().expect("probe"), Probe::Clear);
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    #[test]
    fn unfed_reader_alerts_once_and_feeding_clears_it() {
        let mut job = Job::spawn("read line; sleep 30");
        let mut probe = job.probe();
        probe_until(&mut probe, |seen| seen != Probe::Clear);
        let mut detector = EpisodeDetector::default();
        let mut alerts = 0;
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(100));
            alerts += usize::from(detector.observe(0, probe.probe().expect("probe")));
        }
        assert_eq!(alerts, 1, "a stuck reader alerts exactly once");

        job.feed(b"input\n");
        let seen = probe_until(&mut probe, |seen| seen == Probe::Clear);
        assert_eq!(seen, Probe::Clear, "a fed reader is not waiting");
    }

    /// The `PIPE_WANTR` flag as XNU reports it for the read end the test holds.
    #[cfg(target_os = "macos")]
    mod macos {
        use super::super::platform::{PIPE_WANTR, pipe_end};
        use super::*;
        use std::os::fd::AsRawFd;

        fn wants_read(job: &Job) -> bool {
            // SAFETY: `getpid` has no preconditions.
            let own = unsafe { libc::getpid() };
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
            let job = Job::spawn("perl -e '$SIG{USR1} = sub {}; sysread(STDIN, $b, 1); 1 while 1'");
            wait_for(&job, true);
            let mut probe = job.probe();
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
        use super::super::platform::{pipe_inode, read_descriptor};
        use std::path::Path;

        #[test]
        fn read_syscalls_yield_their_descriptor() {
            let read = libc::SYS_read;
            assert_eq!(read_descriptor(&format!("{read} 0x0 0x1 0x2")), Some(0));
            assert_eq!(read_descriptor(&format!("{read} 0x7 0x1 0x2")), Some(7));
            assert_eq!(
                read_descriptor(&format!("{} 3 0x1", libc::SYS_readv)),
                Some(3)
            );
        }

        #[test]
        fn other_syscalls_and_states_yield_nothing() {
            assert_eq!(read_descriptor("running"), None);
            assert_eq!(read_descriptor("-1 0x7ffc 0x55"), None);
            assert_eq!(read_descriptor(&format!("{} 0x0", libc::SYS_write)), None);
            assert_eq!(read_descriptor(""), None);
        }

        #[test]
        fn only_pipe_targets_have_an_inode() {
            assert_eq!(pipe_inode(Path::new("pipe:[1234]")), Some(1234));
            assert_eq!(pipe_inode(Path::new("socket:[1234]")), None);
            assert_eq!(pipe_inode(Path::new("/dev/null")), None);
        }
    }
}
