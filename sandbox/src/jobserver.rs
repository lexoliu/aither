//! Jobserver token brokering between a shared pool and per-command fifos.
//!
//! Commands never see the shared pool: each execution gets its own fifo,
//! exported through `MAKEFLAGS=--jobserver-auth=fifo:PATH`, and the broker
//! moves tokens between the pool and each command fifo on demand, driven by
//! fd readiness — never by timer polling.
//!
//! Token accounting is exact: `handed` counts the tokens moved into a
//! command's fifo minus the tokens the broker moved back out, and a command
//! can never return more than `handed` — bytes a command writes beyond its
//! outstanding count are discarded as a protocol violation, never minted
//! into the shared pool. Tokens count as returned only when the command's
//! fifo reports EOF on the broker's read end, which happens once every
//! process holding the fifo has exited, so a surviving grandchild keeps its
//! tokens accounted to its command and a killed tree returns them all.

#![cfg(all(unix, feature = "jobserver"))]

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io;
use std::ops::Bound;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_channel::{Receiver, Sender};
use async_io::Async;
use futures_lite::future;
use polling::{Event, Events, PollMode, Poller};
use rustix::io as rustix_io;

#[cfg(any(target_os = "linux", target_os = "android"))]
use rustix::fs::inotify;
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::collections::HashMap;
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::os::unix::io::OwnedFd;

/// A single jobserver token byte, as used by GNU make.
const TOKEN: u8 = b'+';

/// Poller key for the shared pool's read end; fifo ids start at one.
const POOL_KEY: usize = 0;

/// Poller key for the inotify object; fifo ids stay below it.
#[cfg(any(target_os = "linux", target_os = "android"))]
const INOTIFY_KEY: usize = usize::MAX - 1;

/// Errors returned by jobserver pool and broker setup.
#[derive(Debug, thiserror::Error)]
pub enum JobserverError {
    /// An I/O error from fifo or watcher operations.
    #[error("jobserver I/O error: {0}")]
    Io(#[from] io::Error),
    /// The broker service stopped unexpectedly, so its pool and command
    /// fifos are gone.
    #[error("jobserver broker stopped")]
    BrokerStopped,
    /// A path was not valid UTF-8/CStr-compatible.
    #[error("jobserver fifo path is not valid: {0}")]
    InvalidPath(PathBuf),
}

impl From<rustix_io::Errno> for JobserverError {
    fn from(error: rustix_io::Errno) -> Self {
        Self::Io(io::Error::from(error))
    }
}

/// Opens both ends of the fifo at `path` as independent nonblocking fds.
///
/// The read end is opened first (a nonblocking read-only open never waits for
/// a writer) and the write end second, so the fifo never sees EOF while the
/// pair is alive. Separate ends rather than one read-write fd, because
/// `FIONREAD` reports the buffered byte count only on a read-only fd on macOS.
fn open_ends(path: &Path) -> io::Result<(File, File)> {
    let read = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)?;
    let write = OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)?;
    Ok((read, write))
}

fn mkfifo(path: &Path) -> Result<(), JobserverError> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| JobserverError::InvalidPath(path.to_path_buf()))?;
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
    // call, and `mkfifo` does not retain the pointer. Neither `polling` nor
    // `rustix` covers `mkfifo` on Apple targets (rustix's `mknodat` is
    // compiled out there), so this stays a libc call.
    if unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) } == -1 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(())
}

/// Bytes currently buffered in a fifo fd (`FIONREAD`).
fn pending_tokens(file: &File) -> io::Result<usize> {
    let count = rustix_io::ioctl_fionread(file)?;
    Ok(usize::try_from(count).unwrap_or(usize::MAX))
}

/// The outcome of a nonblocking single-token read.
#[derive(Debug)]
enum TokenRead {
    /// A token byte was consumed.
    Token,
    /// The fifo was empty but writers remain, so more tokens may arrive.
    Empty,
    /// The fifo was empty and every writer is gone: the command's whole
    /// process tree has exited.
    Eof,
}

/// Nonblocking single-token read.
fn read_token(file: &File) -> io::Result<TokenRead> {
    let mut byte = [0u8; 1];
    match rustix_io::read(file, &mut byte) {
        Ok(0) => Ok(TokenRead::Eof),
        Ok(1) => Ok(TokenRead::Token),
        Ok(_) => unreachable!("read of a 1-byte buffer returned more"),
        Err(error) if error == rustix_io::Errno::AGAIN => Ok(TokenRead::Empty),
        Err(error) => Err(error.into()),
    }
}

/// Nonblocking single-token write.
fn write_token(file: &File) -> io::Result<()> {
    match rustix_io::write(file, &[TOKEN]) {
        Ok(1) => Ok(()),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "jobserver token write returned zero",
        )),
        Err(error) => Err(error.into()),
    }
}

/// A connection to a shared jobserver token pool.
///
/// The pool is the `fifo:` form of the GNU make jobserver protocol: a named
/// pipe seeded with tokens, shared by every agent that opens it.
pub struct JobPool {
    read: File,
    write: File,
}

impl std::fmt::Debug for JobPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobPool").finish_non_exhaustive()
    }
}

impl JobPool {
    /// Connects to an existing shared pool at `path`, which must already be a
    /// seeded jobserver fifo.
    ///
    /// # Errors
    /// Returns an error when the fifo cannot be opened.
    pub fn fifo(path: impl AsRef<Path>) -> Result<Self, JobserverError> {
        let path = path.as_ref();
        let (read, write) = open_ends(path)?;
        Ok(Self { read, write })
    }

    /// Creates a new pool: makes the fifo at `path` and seeds it with
    /// `tokens` tokens.
    ///
    /// This is the host side of [`JobPool::fifo`]; harnesses use it to lay a
    /// shared pool somewhere multiple agents can reach it.
    ///
    /// # Errors
    /// Returns an error when the fifo cannot be created or filled.
    pub fn create_fifo(path: impl AsRef<Path>, tokens: usize) -> Result<Self, JobserverError> {
        let path = path.as_ref();
        mkfifo(path)?;
        let pool = Self::fifo(path)?;
        for _ in 0..tokens {
            write_token(&pool.write)?;
        }
        Ok(pool)
    }
}

/// The `MAKEFLAGS` value exported to a command for its private fifo.
fn jobserver_makeflags(auth_path: &str) -> String {
    format!("-j --jobserver-auth=fifo:{auth_path}")
}

/// What woke the broker loop.
enum Wake {
    Command(Result<BrokerCommand, async_channel::RecvError>),
    Ready(io::Result<()>),
}

enum BrokerCommand {
    Register {
        reply: Sender<Result<CommandFifo, JobserverError>>,
    },
    Finish {
        id: usize,
    },
    /// A finishing fifo's drain thread hit EOF; only bookkeeping remains.
    Finished {
        id: usize,
    },
}

/// A per-command jobserver fifo under broker management.
///
/// While the lease is alive the broker feeds the fifo on demand and reclaims
/// spare tokens. When the command's root exits the lease is dropped; the
/// broker then stops feeding and returns the command's tokens only once the
/// fifo reports EOF, i.e. once every process still holding it — the whole
/// command tree — is gone.
pub struct CommandFifo {
    id: usize,
    tx: Sender<BrokerCommand>,
    /// Path of the fifo the command's `MAKEFLAGS` names.
    path: PathBuf,
}

impl std::fmt::Debug for CommandFifo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandFifo")
            .field("id", &self.id)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl CommandFifo {
    /// Path the command's `MAKEFLAGS` names, valid on the host.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The `MAKEFLAGS`/`CARGO_MAKEFLAGS`/`MFLAGS` values for the command.
    ///
    /// # Panics
    /// Panics if the broker's fifo directory was not UTF-8; it is validated
    /// when the broker is configured, so this cannot fire.
    #[must_use]
    pub fn environment(&self) -> Vec<(String, String)> {
        let auth_path = self
            .path
            .to_str()
            .expect("jobserver fifo dir is validated UTF-8 at broker setup");
        let flags = jobserver_makeflags(auth_path);
        ["MAKEFLAGS", "CARGO_MAKEFLAGS", "MFLAGS"]
            .iter()
            .map(|name| ((*name).to_string(), flags.clone()))
            .collect()
    }
}

impl Drop for CommandFifo {
    fn drop(&mut self) {
        let _ = self.tx.try_send(BrokerCommand::Finish { id: self.id });
    }
}

/// Handle to the broker service: clones share one brokered pool.
#[derive(Clone, Debug)]
pub struct JobBroker {
    tx: Sender<BrokerCommand>,
}

impl JobBroker {
    /// Registers a new command fifo under the broker.
    ///
    /// The fifo is created beneath the directory configured through
    /// [`job_broker_channel`] and named for the command's `MAKEFLAGS`.
    ///
    /// # Errors
    /// Returns an error when the fifo cannot be created or the broker has
    /// stopped.
    pub async fn open_fifo(&self) -> Result<CommandFifo, JobserverError> {
        let (reply, rx) = async_channel::bounded(1);
        self.tx
            .send(BrokerCommand::Register { reply })
            .await
            .map_err(|_| JobserverError::BrokerStopped)?;
        rx.recv().await.map_err(|_| JobserverError::BrokerStopped)?
    }
}

/// Creates a broker handle plus its service for a shared pool.
///
/// `fifo_dir` is the broker-owned directory per-command fifos are created
/// in; it must be UTF-8 (its paths go into `MAKEFLAGS`) and is created if
/// missing. Keep it out of command working directories: a `git clean` there
/// must not be able to unlink a live fifo.
///
/// # Errors
/// Returns an error when the directory cannot be created or the readiness
/// poller cannot be set up, so configuration fails before any command runs.
pub fn job_broker_channel(
    pool: JobPool,
    fifo_dir: impl AsRef<Path>,
) -> Result<(JobBroker, JobBrokerService), JobserverError> {
    let dir = fifo_dir.as_ref();
    if dir.to_str().is_none() {
        return Err(JobserverError::InvalidPath(dir.to_path_buf()));
    }
    std::fs::create_dir_all(dir)?;

    let poller = Poller::new()?;
    // SAFETY: `pool.read` stays open until the registration is gone: the
    // service deletes it in `Drop` before the pool drops, and on the `?`
    // error paths below the `Poller` (a local, dropped before the `pool`
    // parameter) closes its epoll/kqueue instance first.
    unsafe {
        poller.add_with_mode(&pool.read, Event::readable(POOL_KEY), PollMode::Edge)?;
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let inotify_fd = {
        // Each command fifo gets a watch of its own when it registers: a
        // watch on `dir` would never report `IN_ACCESS`/`IN_MODIFY` for a
        // fifo inside it.
        let fd = inotify::init(inotify::CreateFlags::NONBLOCK | inotify::CreateFlags::CLOEXEC)?;
        // SAFETY: `fd` is owned by the returned service and is deleted from
        // the poller in the service's `Drop`. If this call fails nothing is
        // registered; if the later `Async::new_nonblocking` fails, the poller
        // is consumed and dropped inside it before `fd` drops.
        unsafe { poller.add_with_mode(&fd, Event::readable(INOTIFY_KEY), PollMode::Edge)? };
        fd
    };
    let poller = Async::new_nonblocking(poller)?;

    let (tx, rx) = async_channel::unbounded();
    Ok((
        JobBroker { tx: tx.clone() },
        JobBrokerService {
            rx,
            // Weak, so the service stops once every broker handle and lease
            // is gone.
            lease_tx: tx.downgrade(),
            pool,
            poller,
            dir: dir.to_path_buf(),
            fifos: BTreeMap::new(),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            inotify: inotify_fd,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            watches: HashMap::new(),
            next_id: 1,
            cursor: None,
        },
    ))
}

/// Background service moving tokens between the pool and command fifos.
///
/// Dropping the service — cleanly or while unwinding a broker panic —
/// returns every outstanding token to the shared pool, so a broker failure
/// can never lose tokens other agents own.
pub struct JobBrokerService {
    rx: Receiver<BrokerCommand>,
    lease_tx: async_channel::WeakSender<BrokerCommand>,
    pool: JobPool,
    /// The readiness poller wrapped for async waiting; `wait` on
    /// `get_ref()` drains it.
    poller: Async<Poller>,
    /// Broker-owned directory the per-command fifos live in.
    dir: PathBuf,
    fifos: BTreeMap<usize, FifoState>,
    /// Linux: one inotify object carrying one watch per command fifo.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    inotify: OwnedFd,
    /// Linux: fifo watch descriptor to broker id, for inotify event
    /// lookup.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    watches: HashMap<i32, usize>,
    next_id: usize,
    /// Last fifo id the feed pass moved a token into, for round-robin.
    cursor: Option<usize>,
}

impl std::fmt::Debug for JobBrokerService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobBrokerService")
            .field("fifos", &self.fifos.len())
            .finish_non_exhaustive()
    }
}

struct FifoState {
    /// Broker-held read end: reports tokens the command wrote back and,
    /// after `write` is closed, the EOF meaning the last process holding
    /// the fifo has exited.
    read: File,
    /// Broker-held write end; `None` once the command finished and the
    /// broker is only draining toward EOF.
    write: Option<File>,
    path: PathBuf,
    /// Tokens moved into the fifo minus tokens the broker moved back out:
    /// the most the command may still return.
    handed: usize,
    /// `Finish` received: the command is done; the fifo drains toward EOF.
    finishing: bool,
}

impl JobBrokerService {
    /// Runs the broker until every broker handle and command lease is gone.
    ///
    /// Errors from nonblocking token I/O on fds the broker owns are
    /// invariant violations — `EAGAIN` is already handled and the broker
    /// holds both ends, so `EPIPE` cannot occur — and panic rather than
    /// degrade: the unwinding `Drop` still returns outstanding tokens, and
    /// waiting commands learn about it through
    /// [`JobserverError::BrokerStopped`].
    ///
    /// # Panics
    /// Panics on a readiness or token-I/O error, which can only come from a
    /// broker bug: the unwinding `Drop` still returns outstanding tokens.
    pub async fn serve(mut self) {
        let mut events = Events::new();
        loop {
            let wake = future::or(async { Wake::Command(self.rx.recv().await) }, async {
                Wake::Ready(self.poller.readable().await)
            })
            .await;
            match wake {
                Wake::Command(Ok(BrokerCommand::Register { reply })) => {
                    let _ = reply.send(self.register().await).await;
                }
                Wake::Command(Ok(BrokerCommand::Finish { id })) => self.begin_finish(id),
                Wake::Command(Ok(BrokerCommand::Finished { id })) => self.finished(id),
                Wake::Command(Err(_)) => break,
                Wake::Ready(Err(error)) => {
                    panic!("jobserver poller readiness wait failed: {error}");
                }
                Wake::Ready(Ok(())) => self.reconcile(&mut events),
            }
        }
    }

    async fn register(&mut self) -> Result<CommandFifo, JobserverError> {
        let dir = self.dir.clone();
        let (path, read, write) = blocking::unblock(move || {
            let path = dir.join(format!("{}.fifo", uuid::Uuid::new_v4()));
            mkfifo(&path)?;
            let (read, write) = open_ends(&path)?;
            Ok::<_, JobserverError>((path, read, write))
        })
        .await?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .filter(|&id| id < usize::MAX - 1)
            .expect("jobserver fifo ids exhausted");
        if let Err(error) = self.watch_fifo(&path, &read, &write, id) {
            let _ = async_fs::remove_file(&path).await;
            return Err(error);
        }
        self.fifos.insert(
            id,
            FifoState {
                read,
                write: Some(write),
                path: path.clone(),
                handed: 0,
                finishing: false,
            },
        );
        Ok(CommandFifo {
            id,
            // A `Register` came from a live `JobBroker`, so the channel is
            // open and the lease must hold a sender of its own.
            tx: self
                .lease_tx
                .upgrade()
                .expect("broker handle outlives its register request"),
            path,
        })
    }

    /// Registers a new command fifo's readiness sources.
    ///
    /// Everywhere, the read end reports tokens the command wrote back and,
    /// once the broker's write end is closed, the EOF that means the last
    /// process holding the fifo is gone. Token *consumption* differs per
    /// platform:
    ///
    /// - Apple and BSDs: the write end's `EVFILT_WRITE` edge fires on every
    ///   occupancy change — on XNU fifos are socketpairs, so a read by the
    ///   command frees send space. That is verified by experiment, not a
    ///   documented guarantee.
    /// - Linux and Android: `EPOLLOUT` fires only when the pipe was full,
    ///   which a one-token fifo never is, so a read produces no event.
    ///   inotify reports access and modify instead — but only on a watch
    ///   placed on the fifo itself: a watch on the fifo's directory never
    ///   reports `IN_ACCESS`/`IN_MODIFY` for a fifo inside it (verified on
    ///   Linux 7.2.9; regular files in the same directory do report them),
    ///   so each fifo gets a watch of its own and events key on the watch
    ///   descriptor.
    fn watch_fifo(
        &mut self,
        path: &Path,
        read: &File,
        write: &File,
        id: usize,
    ) -> Result<(), JobserverError> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            // SAFETY: `read` is owned by the `FifoState` the caller inserts
            // next; its registration is deleted in `finished` or the
            // service's `Drop`, always before the `File` drops.
            unsafe {
                self.poller
                    .get_ref()
                    .add_with_mode(read, Event::readable(id), PollMode::Edge)?;
            }
            let wd = inotify::add_watch(
                &self.inotify,
                path,
                inotify::WatchFlags::ACCESS | inotify::WatchFlags::MODIFY,
            )
            .inspect_err(|_| {
                let _ = self.poller.get_ref().delete(read);
            })?;
            self.watches.insert(wd, id);
            let _ = write;
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            // SAFETY: `read` and `write` are owned by the `FifoState` the
            // caller inserts next. `write` is deleted in `begin_finish` or
            // `Drop`, `read` in `finished` or `Drop`, each before its
            // `File` drops; a failed second registration deletes `read`.
            unsafe {
                self.poller
                    .get_ref()
                    .add_with_mode(read, Event::readable(id), PollMode::Edge)?;
            }
            let result = unsafe {
                self.poller
                    .get_ref()
                    .add_with_mode(write, Event::writable(id), PollMode::Edge)
            };
            if let Err(error) = result {
                let _ = self.poller.get_ref().delete(read);
                return Err(error.into());
            }
            let _ = path;
        }
        Ok(())
    }

    /// Drains every event the poller has queued, then feeds empty fifos.
    fn reconcile(&mut self, events: &mut Events) {
        loop {
            let count = self
                .poller
                .get_ref()
                .wait(events, Some(Duration::ZERO))
                .expect("jobserver poller wait failed");
            if count == 0 {
                break;
            }
            let keys: Vec<usize> = events.iter().map(|event| event.key).collect();
            events.clear();
            for key in keys {
                if key == POOL_KEY {
                    continue;
                }
                #[cfg(any(target_os = "linux", target_os = "android"))]
                if key == INOTIFY_KEY {
                    for id in self.inotify_fired() {
                        self.settle_or_drain(id);
                    }
                    continue;
                }
                self.settle_or_drain(key);
            }
        }
        self.feed();
    }

    /// Linux: reads pending inotify events and returns the fifo ids they
    /// touched. Both `IN_ACCESS` (a token consumed) and `IN_MODIFY` (a token
    /// returned) land here, keyed by the fifo's watch descriptor.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn inotify_fired(&mut self) -> Vec<usize> {
        let mut buf = [std::mem::MaybeUninit::<u8>::uninit(); 4096];
        let mut reader = inotify::Reader::new(&self.inotify, &mut buf);
        let mut ids = Vec::new();
        loop {
            match reader.next() {
                Ok(event) => {
                    if event.events().contains(inotify::ReadFlags::QUEUE_OVERFLOW) {
                        tracing::warn!("jobserver inotify queue overflow; reconciling every fifo");
                        ids.extend(self.fifos.keys().copied());
                        continue;
                    }
                    if event.events().contains(inotify::ReadFlags::IGNORED) {
                        match self.watches.remove(&event.wd()) {
                            // Expected: the broker's drain thread unlinked
                            // the fifo, which removes its watch. `None`
                            // means `finished` already dropped the mapping.
                            Some(id) if self.fifos[&id].finishing => {}
                            None => {}
                            Some(id) => panic!(
                                "jobserver fifo watch on {} was removed while the command was live; consumption can no longer wake the broker",
                                self.fifos[&id].path.display()
                            ),
                        }
                        continue;
                    }
                    if let Some(&id) = self.watches.get(&event.wd()) {
                        ids.push(id);
                    }
                }
                Err(error) if error == rustix_io::Errno::AGAIN => break,
                Err(error) => panic!("jobserver inotify read failed: {error}"),
            }
        }
        ids
    }

    fn settle_or_drain(&mut self, id: usize) {
        let Some(state) = self.fifos.get(&id) else {
            return;
        };
        if !state.finishing {
            self.settle(id);
        }
    }

    /// Returns surplus tokens an active command wrote back to the pool.
    ///
    /// Only `min(pending, handed)` bytes can be legitimate returns; anything
    /// beyond that is a protocol violation by the command — it owns a
    /// read-write fd on its fifo and can write any byte count — so the
    /// surplus is discarded with a warning and can never inflate the pool.
    fn settle(&mut self, id: usize) {
        let Some(state) = self.fifos.get_mut(&id) else {
            return;
        };
        let pending = pending_tokens(&state.read).expect("jobserver fifo FIONREAD failed");
        let legit = pending.min(state.handed);
        if pending > legit {
            let fabricated = pending - legit;
            tracing::warn!(
                id,
                fabricated,
                "jobserver command wrote tokens it was never handed; discarding them"
            );
            for _ in 0..fabricated {
                if !matches!(read_token(&state.read), Ok(TokenRead::Token)) {
                    break;
                }
            }
        }
        // Every legitimate token beyond one returns to the pool; one stays
        // buffered so the command keeps a token to work with.
        for _ in 1..legit {
            match read_token(&state.read) {
                Ok(TokenRead::Token) => {
                    write_token(&self.pool.write).expect("jobserver pool write failed");
                    state.handed -= 1;
                }
                _ => break,
            }
        }
    }

    /// Stops feeding a fifo: the command's root exited, so the broker's
    /// write end closes and the fifo drains until its read end reports EOF —
    /// the point where every process holding the fifo is gone.
    ///
    /// No platform readiness API reports last-writer-close on a fifo (kqueue
    /// posts no knote for it; `EPOLLOUT` covers nothing here), so the drain
    /// runs on a dedicated thread doing a *blocking* read — it blocks on the
    /// exact condition waited for, no timer polling. The thread owns all of
    /// this fifo's token accounting from here; the service keeps only the
    /// registration bookkeeping a `Finished` message cleans up.
    fn begin_finish(&mut self, id: usize) {
        let Some(state) = self.fifos.get_mut(&id) else {
            return;
        };
        if state.finishing {
            return;
        }
        state.finishing = true;
        if let Some(write) = state.write.take() {
            #[cfg(not(any(target_os = "linux", target_os = "android")))]
            self.poller
                .get_ref()
                .delete(&write)
                .expect("jobserver fifo write end deregistration failed");
            drop(write);
        }
        let tx = self
            .lease_tx
            .upgrade()
            .expect("a Finish came from a live lease");
        assert!(
            spawn_drain(&self.pool.write, state, id, Some(tx)),
            "jobserver fifo drain thread could not be equipped"
        );
    }

    /// The drain thread reached EOF and finished the token accounting
    /// itself; drop the fifo's registration bookkeeping.
    fn finished(&mut self, id: usize) {
        let Some(state) = self.fifos.remove(&id) else {
            return;
        };
        self.poller
            .get_ref()
            .delete(&state.read)
            .expect("jobserver fifo read end deregistration failed");
        #[cfg(any(target_os = "linux", target_os = "android"))]
        self.watches.retain(|_, fifo_id| *fifo_id != id);
    }

    /// Feeds every empty active fifo one token from the pool, round-robin
    /// from the last fifo served so greedy commands share.
    fn feed(&mut self) {
        let ids: Vec<usize> = match self.cursor {
            Some(cursor) => self
                .fifos
                .range((Bound::Excluded(cursor), Bound::Unbounded))
                .chain(self.fifos.range(..=cursor))
                .map(|(id, _)| *id)
                .collect(),
            None => self.fifos.keys().copied().collect(),
        };
        for id in ids {
            let Some(state) = self.fifos.get_mut(&id) else {
                continue;
            };
            if state.finishing
                || pending_tokens(&state.read).expect("jobserver fifo FIONREAD failed") != 0
            {
                continue;
            }
            match read_token(&self.pool.read) {
                Ok(TokenRead::Token) => {
                    let write = state
                        .write
                        .as_ref()
                        .expect("an active fifo keeps a broker write end");
                    if let Err(error) = write_token(write) {
                        // The token came out of the pool; put it back before
                        // the accounting can notice it missing.
                        let _ = write_token(&self.pool.write);
                        panic!("jobserver fifo write failed: {error}");
                    }
                    state.handed += 1;
                    self.cursor = Some(id);
                }
                Ok(TokenRead::Empty | TokenRead::Eof) => break,
                Err(error) => panic!("jobserver pool read failed: {error}"),
            }
        }
    }
}

/// Hands a fifo's remaining accounting to a drain thread. `state.handed`
/// is frozen by the `finishing` flag, so the thread can cap forwarding at
/// it and mint the rest at EOF without any further bookkeeping. Returns
/// false when the thread could not be equipped; callers that cannot
/// recover panic, but `Drop` only warns so an unwind cannot abort.
fn spawn_drain(
    pool_write_src: &File,
    state: &FifoState,
    id: usize,
    tx: Option<Sender<BrokerCommand>>,
) -> bool {
    // The dup shares the file description, so clearing `O_NONBLOCK` here
    // flips `state.read` to blocking too — only the drain thread reads it
    // from now on.
    let equipped = || -> Option<(File, File)> {
        let drain = state.read.try_clone().ok()?;
        rustix::fs::fcntl_setfl(&drain, rustix::fs::OFlags::empty()).ok()?;
        let pool_write = pool_write_src.try_clone().ok()?;
        Some((drain, pool_write))
    };
    let Some((drain, pool_write)) = equipped() else {
        return false;
    };
    let limit = state.handed;
    let path = state.path.clone();
    std::thread::spawn(move || {
        drain_fifo(id, limit, &drain, &pool_write);
        let _ = std::fs::remove_file(&path);
        if let Some(tx) = tx {
            let _ = tx.send_blocking(BrokerCommand::Finished { id });
        }
    });
    true
}

/// Blocks on a fifo's read end until EOF — every process holding it has
/// exited — and returns the `limit` tokens the command was handed. Reads
/// are blocking because last-writer-close reports through `read` as EOF
/// while no readiness API surfaces it. Bytes beyond `limit` are fabricated
/// returns: discarded and warned once, never written to the pool. `EINTR`
/// is retried; only `Ok(0)` is EOF, and any other read error is a broker
/// bug. At EOF the tokens the dead tree still owed are minted back.
fn drain_fifo(id: usize, limit: usize, read: &File, pool: &File) {
    let mut forwarded = 0usize;
    let mut discarded = 0usize;
    let mut byte = [0u8; 1];
    loop {
        match rustix_io::read(read, &mut byte) {
            Ok(1) if forwarded < limit => {
                rustix_io::write(pool, &byte).expect("jobserver pool token return failed");
                forwarded += 1;
            }
            Ok(1) => discarded += 1,
            Ok(0) => break,
            Ok(_) => unreachable!("a one-byte read buffer cannot return more"),
            Err(error) if error == rustix::io::Errno::INTR => {}
            Err(error) => panic!("jobserver fifo drain read failed: {error}"),
        }
    }
    if discarded > 0 {
        tracing::warn!(
            id,
            discarded,
            "jobserver command returned tokens it was never handed"
        );
    }
    for _ in forwarded..limit {
        rustix_io::write(pool, &[TOKEN]).expect("jobserver pool token mint failed");
    }
}

impl Drop for JobBrokerService {
    /// Never blocks, never returns a token twice, and never returns a token
    /// a running command still holds: finishing fifos keep their drain
    /// thread, and every still-active fifo is closed on the write side and
    /// handed to a drain thread of its own, so its tokens come back only
    /// when its whole process tree is gone.
    fn drop(&mut self) {
        let _ = self.poller.get_ref().delete(&self.pool.read);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let _ = self.poller.get_ref().delete(&self.inotify);
        for (id, mut state) in std::mem::take(&mut self.fifos) {
            if !state.finishing
                && let Some(write) = state.write.take()
            {
                #[cfg(not(any(target_os = "linux", target_os = "android")))]
                let _ = self.poller.get_ref().delete(&write);
                drop(write);
            }
            // This fifo's registrations are deleted here (or already in
            // `begin_finish`) before `state`'s fds drop.
            let _ = self.poller.get_ref().delete(&state.read);
            if !state.finishing && !spawn_drain(&self.pool.write, &state, id, None) {
                tracing::warn!(
                    handed = state.handed,
                    path = %state.path.display(),
                    "jobserver tokens leak: drain thread could not be equipped"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use executor_core::tokio::TokioGlobal;
    use executor_core::{Executor, Task};
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    /// A broker over a freshly seeded pool of `size` tokens, plus an
    /// independent observer handle on the same pool fifo.
    struct Harness {
        _dir: tempfile::TempDir,
        broker: JobBroker,
        observer: JobPool,
    }

    impl Harness {
        fn new(size: usize) -> Self {
            let dir = tempdir().unwrap();
            let path = dir.path().join("pool.fifo");
            let pool = JobPool::create_fifo(&path, size).unwrap();
            let observer = JobPool::fifo(&path).unwrap();
            let (broker, service) = job_broker_channel(pool, dir.path().join("commands")).unwrap();
            TokioGlobal
                .spawn(async move { service.serve().await })
                .detach();
            Self {
                _dir: dir,
                broker,
                observer,
            }
        }

        async fn command(&self) -> (CommandFifo, File) {
            let lease = self.broker.open_fifo().await.unwrap();
            // The command's processes open the fifo read-write, like make does.
            let child = OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(lease.path())
                .unwrap();
            (lease, child)
        }

        fn in_pool(&self) -> usize {
            pending_tokens(&self.observer.read).unwrap()
        }
    }

    /// Waits until `condition` holds. The broker is driven by fd readiness;
    /// the test only gives it time to react.
    async fn settle(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            async_io::Timer::after(Duration::from_millis(5)).await;
        }
    }

    /// Takes tokens like a job-hungry build tool until the fifo runs dry.
    fn take_all(child: &File, held: &mut usize) {
        while matches!(read_token(child), Ok(TokenRead::Token)) {
            *held += 1;
        }
    }

    #[tokio::test]
    async fn pool_create_and_take() {
        let harness = Harness::new(4);
        assert_eq!(harness.in_pool(), 4);
        assert!(matches!(
            read_token(&harness.observer.read),
            Ok(TokenRead::Token)
        ));
        assert_eq!(harness.in_pool(), 3);
    }

    #[tokio::test]
    async fn environment_names_the_command_fifo_not_the_pool() {
        let harness = Harness::new(2);
        let (lease, _child) = harness.command().await;
        let fifo = lease.path().to_string_lossy().into_owned();
        assert!(fifo.contains("/commands/"));
        let environment = lease.environment();
        assert_eq!(environment.len(), 3);
        for (name, value) in environment {
            assert!(["MAKEFLAGS", "CARGO_MAKEFLAGS", "MFLAGS"].contains(&name.as_str()));
            assert_eq!(value, format!("-j --jobserver-auth=fifo:{fifo}"));
            assert!(!value.contains("pool.fifo"));
        }
    }

    #[tokio::test]
    async fn two_greedy_commands_never_hold_more_than_the_pool() {
        const POOL: usize = 3;
        let harness = Harness::new(POOL);
        let (lease_a, child_a) = harness.command().await;
        let (lease_b, child_b) = harness.command().await;
        let (mut held_a, mut held_b) = (0, 0);

        // Both commands keep asking for more tokens than the pool holds.
        settle("both commands to drain the pool", || {
            take_all(&child_a, &mut held_a);
            take_all(&child_b, &mut held_b);
            held_a + held_b == POOL
        })
        .await;
        assert!(
            held_a >= 1 && held_b >= 1,
            "pool starved a command: {held_a}/{held_b}"
        );
        for _ in 0..20 {
            async_io::Timer::after(Duration::from_millis(5)).await;
            take_all(&child_a, &mut held_a);
            take_all(&child_b, &mut held_b);
            assert!(
                held_a + held_b <= POOL,
                "over-subscribed: {held_a}+{held_b}"
            );
        }
        assert_eq!(harness.in_pool(), 0);

        drop((lease_a, lease_b));
        // The processes still hold their fds: tokens stay accounted to the
        // commands until every process holding a fifo exits.
        async_io::Timer::after(Duration::from_millis(200)).await;
        assert_eq!(harness.in_pool(), 0, "tokens returned before EOF");
        drop((child_a, child_b));
        settle("pool to refill", || harness.in_pool() == POOL).await;
    }

    #[tokio::test]
    async fn killed_command_returns_every_token() {
        const POOL: usize = 4;
        let harness = Harness::new(POOL);
        let (lease, child) = harness.command().await;
        let mut held = 0;
        settle("command to hold all tokens", || {
            take_all(&child, &mut held);
            held == POOL
        })
        .await;
        assert_eq!(harness.in_pool(), 0);

        // The process tree dies holding every token; its fds close with it.
        drop(child);
        drop(lease);
        settle("pool to refill after kill", || harness.in_pool() == POOL).await;
        // No token was minted beyond the pool size.
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(harness.in_pool(), POOL);
    }

    #[tokio::test]
    async fn exiting_command_leaves_the_pool_full() {
        const POOL: usize = 3;
        let harness = Harness::new(POOL);
        let (lease, child) = harness.command().await;
        let mut held = 0;
        settle("command to take a token", || {
            take_all(&child, &mut held);
            held >= 1
        })
        .await;
        for _ in 0..held {
            write_token(&child).unwrap();
        }
        drop(child);
        drop(lease);
        settle("pool to be full", || harness.in_pool() == POOL).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(harness.in_pool(), POOL);
    }

    #[tokio::test]
    async fn surplus_returned_on_exit_never_inflates_the_pool() {
        const POOL: usize = 3;
        let harness = Harness::new(POOL);
        let (lease, child) = harness.command().await;
        let mut held = 0;
        settle("command to take tokens", || {
            take_all(&child, &mut held);
            held == 2
        })
        .await;

        // A buggy client also releases its implicit slot: it writes one
        // byte more than it was ever handed, then exits normally. The
        // drain thread must not forward the surplus into the pool.
        for _ in 0..=held {
            write_token(&child).unwrap();
        }
        drop(child);
        drop(lease);
        settle("pool to refill after exit", || harness.in_pool() == POOL).await;
        async_io::Timer::after(Duration::from_millis(200)).await;
        assert_eq!(harness.in_pool(), POOL, "surplus token minted into pool");
    }

    #[tokio::test]
    async fn finish_while_a_process_lives_keeps_tokens_out_of_the_pool() {
        const POOL: usize = 4;
        let harness = Harness::new(POOL);
        let (lease, child) = harness.command().await;
        let mut held = 0;
        settle("command to hold all tokens", || {
            take_all(&child, &mut held);
            held == POOL
        })
        .await;

        // The lease finishes while a surviving process still holds the
        // fifo open: the pool must not see the held tokens until EOF.
        drop(lease);
        async_io::Timer::after(Duration::from_millis(200)).await;
        assert_eq!(harness.in_pool(), 0, "tokens returned before EOF");

        // Tokens a live process writes back still reach the pool.
        for _ in 0..2 {
            write_token(&child).unwrap();
        }
        settle("returned tokens to reach the pool", || {
            harness.in_pool() == 2
        })
        .await;

        drop(child);
        settle("pool to refill after EOF", || harness.in_pool() == POOL).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(harness.in_pool(), POOL);
    }

    #[tokio::test]
    async fn fabricated_tokens_never_inflate_the_pool() {
        const POOL: usize = 3;
        let harness = Harness::new(POOL);
        let (lease, child) = harness.command().await;
        let mut held = 0;
        settle("command to take two tokens", || {
            take_all(&child, &mut held);
            held == 2
        })
        .await;
        assert_eq!(harness.in_pool(), 1);

        // The command writes five tokens it never took; the broker may only
        // move back what it handed (two, minus the one it leaves buffered).
        for _ in 0..5 {
            write_token(&child).unwrap();
        }
        settle("surplus to be discarded", || harness.in_pool() == 2).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(harness.in_pool(), 2, "fabricated tokens reached the pool");

        drop(child);
        drop(lease);
        settle("pool to refill", || harness.in_pool() == POOL).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(harness.in_pool(), POOL);
    }

    #[tokio::test]
    async fn spare_tokens_move_back_to_the_pool() {
        const POOL: usize = 3;
        let harness = Harness::new(POOL);
        let (_lease, child) = harness.command().await;
        let mut held = 0;
        settle("command to hold all tokens", || {
            take_all(&child, &mut held);
            held == POOL
        })
        .await;

        // Returning every token leaves a spare surplus in the command fifo;
        // the broker keeps one there and moves the rest back to the pool.
        for _ in 0..held {
            write_token(&child).unwrap();
        }
        settle("surplus to return to the pool", || {
            harness.in_pool() == POOL - 1
        })
        .await;
        let mut spare = 0;
        take_all(&child, &mut spare);
        assert_eq!(spare, 1);
    }

    #[tokio::test]
    async fn open_fifo_fails_when_the_broker_is_gone() {
        let dir = tempdir().unwrap();
        let pool = JobPool::create_fifo(dir.path().join("pool.fifo"), 1).unwrap();
        let (broker, service) = job_broker_channel(pool, dir.path().join("commands")).unwrap();
        drop(service);
        let error = broker.open_fifo().await.unwrap_err();
        assert!(matches!(error, JobserverError::BrokerStopped));
    }
}
