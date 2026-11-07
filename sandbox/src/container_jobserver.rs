//! Jobserver fifos for container commands.
//!
//! A host fifo cannot serve a command in a container on a VM-backed runtime
//! (`OrbStack`, Docker Desktop, a Podman machine): the container runs on
//! another kernel, where a bind-mounted fifo inode opens as an unrelated pipe.
//! So the command's fifo is made inside the container, by a relay — a small
//! program run with the image's managed `python3` in an exec of its own,
//! outside the command's job — that brokers it over the exec's stdin and
//! stdout against a lease of the shared broker on the host.
//!
//! The host side of the relay is an ordinary command of the broker: it holds
//! the lease's fifo open as a command's process would, takes one token from
//! it whenever the relay reports the container fifo empty, and writes back
//! every token the relay returns. The broker's accounting therefore covers the
//! container command unchanged: it never mints a token, and once the host side
//! closes its end the lease's fifo reports EOF and every token still out comes
//! back. The host side closes it only after the relay has exited, which it does
//! once the command's exec has ended and the last container process holding the
//! container fifo is gone, so a killed command returns every token it held.

#![cfg(all(unix, feature = "jobserver"))]

use std::fs::File;

use async_channel::{Receiver, Sender};
use async_io::Async;
use executor_core::{Executor, Task as _};
use futures_lite::future;

use crate::container::ContainerPath;
use crate::jobserver::{
    CommandFifo, JobBroker, JobserverError, jobserver_environment, open_client_end,
    return_client_token, take_client_token,
};
use crate::shell_session::{ContainerExecOutcome, ContainerShellRuntime};

/// The relay program, run as `python3 -I -c SOURCE FIFO`.
const RELAY_SOURCE: &str = include_str!("../templates/container_jobserver_relay.py");

/// Relay to host: the container fifo exists.
const READY: u8 = b'R';
/// Relay to host: the container fifo is empty and wants a token.
const WANT: u8 = b'T';
/// Either way: one token.
const TOKEN: u8 = b'+';
/// Host to relay: the command's exec has ended.
const FINISH: u8 = b'F';

/// A container command's jobserver fifo, brokered from the host pool.
///
/// Dropping it tells the relay the command's exec has ended: the relay stops
/// feeding the fifo and returns its tokens once every container process
/// holding it has exited.
pub struct ContainerJobFifo {
    path: ContainerPath,
    /// Never sent on; its drop is the finish signal.
    _finish: Sender<()>,
}

impl std::fmt::Debug for ContainerJobFifo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerJobFifo")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl ContainerJobFifo {
    /// The fifo's path inside the container.
    pub const fn path(&self) -> &ContainerPath {
        &self.path
    }

    /// The `MAKEFLAGS`/`CARGO_MAKEFLAGS`/`MFLAGS` values naming the fifo as
    /// the container sees it.
    pub fn environment(&self) -> Vec<(String, String)> {
        jobserver_environment(self.path.as_str())
    }
}

/// The command line that runs the relay for the container fifo at `fifo`.
fn relay_command(python: &ContainerPath, fifo: &ContainerPath) -> String {
    shell_words::join([python.as_str(), "-I", "-c", RELAY_SOURCE, fifo.as_str()])
}

/// A fresh container fifo path, in the container's `/tmp` beside the exec
/// pid files and out of every command working directory.
fn container_fifo_path() -> ContainerPath {
    ContainerPath::new(format!(
        "/tmp/aither-jobserver-{}.fifo",
        uuid::Uuid::new_v4().simple()
    ))
    .expect("a path under /tmp is absolute")
}

/// Leases a fifo from `broker` for one command in the container of
/// `runtime`, starts the relay that makes and brokers the fifo inside the
/// container with the image's `python`, and returns once the fifo exists.
///
/// # Errors
/// Returns an error when the broker has stopped, the lease's fifo cannot be
/// opened, or the relay fails before the container fifo exists.
pub async fn open_container_job_fifo<E: Executor + Clone + 'static>(
    broker: &JobBroker,
    runtime: &ContainerShellRuntime,
    python: &ContainerPath,
    executor: E,
) -> Result<ContainerJobFifo, JobserverError> {
    let lease = broker.open_fifo().await?;
    let lease_path = lease.path().to_path_buf();
    let host_end = Async::new(blocking::unblock(move || open_client_end(&lease_path)).await?)?;

    let path = container_fifo_path();
    let script = relay_command(python, &path);
    let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
    let (stdin_tx, stdin_rx) = async_channel::unbounded::<Vec<u8>>();
    let (stdout_tx, stdout_rx) = async_channel::bounded::<Vec<u8>>(16);
    let (outcome_tx, outcome_rx) = async_channel::bounded(1);
    let exec = runtime.exec();
    let container_id = runtime.container_id().to_string();
    executor
        .spawn(async move {
            let outcome = exec
                .exec_streamed_boxed(&container_id, &script, "/", kill_rx, stdin_rx, stdout_tx)
                .await;
            let _ = outcome_tx.send(outcome).await;
        })
        .detach();

    let mut relay = HostRelay {
        _lease: lease,
        host_end,
        to_relay: stdin_tx,
        from_relay: stdout_rx,
        outcome: outcome_rx,
        _kill: kill_tx,
        backlog: Vec::new(),
        outstanding: 0,
        wants: false,
        finishing: false,
    };
    relay.await_ready().await?;
    let (finish_tx, finish_rx) = async_channel::bounded(1);
    executor.spawn(relay.serve(finish_rx)).detach();
    Ok(ContainerJobFifo {
        path,
        _finish: finish_tx,
    })
}

/// What woke the host side of a relay.
enum Wake {
    Relay(Option<Vec<u8>>),
    Token(std::io::Result<()>),
    Finish,
}

/// The host side of one container command's relay.
///
/// Field order is drop order: the lease's fifo end closes before the lease
/// itself finishes, and the relay exec is killed last.
struct HostRelay {
    host_end: Async<File>,
    _lease: CommandFifo,
    to_relay: Sender<Vec<u8>>,
    from_relay: Receiver<Vec<u8>>,
    outcome: Receiver<Result<ContainerExecOutcome, String>>,
    _kill: Sender<()>,
    /// Relay bytes received after `R` and not yet handled.
    backlog: Vec<u8>,
    /// Tokens sent to the relay minus tokens it returned: the most it may
    /// still return.
    outstanding: usize,
    /// The relay asked for a token that has not been sent yet.
    wants: bool,
    /// The command's exec has ended and the relay was told so.
    finishing: bool,
}

impl HostRelay {
    /// Waits for the relay's `R`: the container fifo exists.
    async fn await_ready(&mut self) -> Result<(), JobserverError> {
        loop {
            let Ok(chunk) = self.from_relay.recv().await else {
                return Err(JobserverError::ContainerRelay(self.failure().await));
            };
            if let Some((&first, rest)) = chunk.split_first() {
                if first != READY {
                    return Err(JobserverError::ContainerRelay(format!(
                        "the relay's first message was {first:#04x}, not ready"
                    )));
                }
                self.backlog.extend_from_slice(rest);
                return Ok(());
            }
        }
    }

    /// Why the relay exec ended, once its stdout has closed.
    async fn failure(&self) -> String {
        match self.outcome.recv().await {
            Ok(Ok(ContainerExecOutcome::Completed { output, .. })) => format!(
                "the relay exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            Ok(Ok(ContainerExecOutcome::Killed)) => "the relay was killed".to_string(),
            Ok(Err(error)) => error,
            Err(_) => "the relay exec ended without a result".to_string(),
        }
    }

    /// Moves tokens between the lease's fifo and the relay until the relay
    /// exits, then closes the lease's fifo so the broker settles the rest.
    async fn serve(mut self, finish: Receiver<()>) {
        let backlog = std::mem::take(&mut self.backlog);
        self.handle(&backlog);
        loop {
            let wake = {
                let relay = async { Wake::Relay(self.from_relay.recv().await.ok()) };
                let finished = async {
                    if self.finishing {
                        future::pending::<()>().await;
                    }
                    let _ = finish.recv().await;
                    Wake::Finish
                };
                let token = async {
                    if !self.wants || self.finishing {
                        future::pending::<()>().await;
                    }
                    Wake::Token(self.host_end.read_with(take_client_token).await)
                };
                future::or(relay, future::or(finished, token)).await
            };
            match wake {
                Wake::Relay(Some(chunk)) => self.handle(&chunk),
                Wake::Relay(None) => break,
                Wake::Token(Ok(())) => {
                    self.wants = false;
                    self.outstanding += 1;
                    // A gone relay closes its stdout too, which ends the loop;
                    // the token is the lease's, settled at its EOF.
                    let _ = self.to_relay.try_send(vec![TOKEN]);
                }
                Wake::Token(Err(error)) => panic!("jobserver lease fifo read failed: {error}"),
                Wake::Finish => {
                    self.finishing = true;
                    self.wants = false;
                    let _ = self.to_relay.try_send(vec![FINISH]);
                }
            }
        }
        if !self.finishing {
            let failure = self.failure().await;
            tracing::error!(
                outstanding = self.outstanding,
                %failure,
                "jobserver container relay ended while its command ran; its tokens return to the pool now"
            );
        } else if let Ok(Ok(ContainerExecOutcome::Completed { output, .. })) =
            self.outcome.recv().await
            && !output.status.success()
        {
            tracing::error!(
                status = %output.status,
                stderr = %String::from_utf8_lossy(&output.stderr).trim(),
                "jobserver container relay failed"
            );
        }
    }

    /// Handles the relay's messages.
    fn handle(&mut self, messages: &[u8]) {
        let mut fabricated = 0usize;
        for &message in messages {
            match message {
                WANT => self.wants = true,
                TOKEN if self.outstanding > 0 => {
                    return_client_token(self.host_end.get_ref())
                        .expect("jobserver lease fifo write failed");
                    self.outstanding -= 1;
                }
                TOKEN => fabricated += 1,
                other => panic!("jobserver container relay sent unknown message {other:#04x}"),
            }
        }
        if fabricated > 0 {
            tracing::warn!(
                fabricated,
                "jobserver container command returned tokens it was never handed; discarding them"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use executor_core::tokio::TokioGlobal;

    use crate::jobserver::{JobPool, job_broker_channel};
    use crate::shell_session::ExecLifetime;

    /// A broker over a freshly seeded pool, with an observer counting the
    /// tokens left in it.
    struct Pool {
        dir: tempfile::TempDir,
        broker: JobBroker,
        /// Read-only, the only end `FIONREAD` counts on every platform.
        observer: File,
    }

    impl Pool {
        fn new(size: usize) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("pool.fifo");
            let pool = JobPool::create_fifo(&path, size).unwrap();
            let observer = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&path)
                .unwrap();
            let (broker, service) = job_broker_channel(pool, dir.path().join("commands")).unwrap();
            TokioGlobal
                .spawn(async move { service.serve().await })
                .detach();
            Self {
                dir,
                broker,
                observer,
            }
        }

        fn tokens(&self) -> usize {
            usize::try_from(rustix::io::ioctl_fionread(&self.observer).unwrap()).unwrap()
        }

        fn runtime(exec: impl crate::ContainerExec + 'static) -> ContainerShellRuntime {
            ContainerShellRuntime::new(
                "aither-test-jobserver",
                crate::ContainerIpcRoute::new(
                    "host.docker.internal",
                    crate::BridgeAddress::try_from(std::net::IpAddr::from([127, 0, 0, 1])).unwrap(),
                ),
                crate::AnyContainerExec::new(Arc::new(exec)),
            )
        }
    }

    /// Waits until `condition` holds; the broker and relay are driven by
    /// readiness, the test only gives them time to react.
    async fn settle(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            async_io::Timer::after(Duration::from_millis(5)).await;
        }
    }

    fn exited(code: i32, stderr: &str) -> ContainerExecOutcome {
        ContainerExecOutcome::Completed {
            output: std::process::Output {
                status: std::process::ExitStatus::from_raw(code << 8),
                stdout: Vec::new(),
                stderr: stderr.as_bytes().to_vec(),
            },
            killed: Vec::new(),
        }
    }

    /// The container side of a relay exec, played by the test.
    struct ScriptedRelay {
        from_host: Receiver<Vec<u8>>,
        to_host: Sender<Vec<u8>>,
        /// The exec's outcome; dropping it ends the exec with status 0.
        exit: Sender<Result<ContainerExecOutcome, String>>,
    }

    impl ScriptedRelay {
        async fn say(&self, messages: &[u8]) {
            self.to_host.send(messages.to_vec()).await.unwrap();
        }

        /// The next message from the host, if one arrives within `wait`.
        async fn heard(&self, wait: Duration) -> Option<u8> {
            future::or(
                async {
                    let chunk = self.from_host.recv().await.ok()?;
                    assert_eq!(chunk.len(), 1, "the host sends one message per chunk");
                    Some(chunk[0])
                },
                async {
                    async_io::Timer::after(wait).await;
                    None
                },
            )
            .await
        }

        /// Asks for a token and returns whether one came.
        async fn want(&self) -> bool {
            self.say(&[WANT]).await;
            match self.heard(Duration::from_millis(200)).await {
                Some(TOKEN) => true,
                None => false,
                Some(other) => panic!("unexpected message {other:#04x}"),
            }
        }
    }

    /// Hands each relay exec's channels to the test.
    struct ScriptedExec(Sender<ScriptedRelay>);

    impl crate::ContainerExec for ScriptedExec {
        fn exec(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            std::future::ready(Err("only the relay runs here".to_string()))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            stdin_rx: Receiver<Vec<u8>>,
            stdout: Sender<Vec<u8>>,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            let (exit, outcome) = async_channel::bounded(1);
            self.0
                .try_send(ScriptedRelay {
                    from_host: stdin_rx,
                    to_host: stdout,
                    exit,
                })
                .unwrap();
            async move { outcome.recv().await.unwrap_or_else(|_| Ok(exited(0, ""))) }
        }
    }

    /// Opens a container fifo whose relay the test plays, once it reports
    /// ready.
    async fn scripted(
        pool: &Pool,
        runtime: &ContainerShellRuntime,
        relays: &Receiver<ScriptedRelay>,
    ) -> (ContainerJobFifo, ScriptedRelay) {
        let python = ContainerPath::new("/opt/aither/tools/key/bin/python3").unwrap();
        let open = open_container_job_fifo(&pool.broker, runtime, &python, TokioGlobal);
        let relay = async {
            let relay = relays.recv().await.unwrap();
            relay.say(&[READY]).await;
            relay
        };
        let (fifo, relay) = future::zip(open, relay).await;
        (fifo.unwrap(), relay)
    }

    #[tokio::test]
    async fn relay_draws_from_the_pool_and_a_killed_command_returns_every_token() {
        const POOL: usize = 3;
        let pool = Pool::new(POOL);
        let (relays_tx, relays) = async_channel::unbounded();
        let runtime = Pool::runtime(ScriptedExec(relays_tx));
        let (fifo, relay) = scripted(&pool, &runtime, &relays).await;

        for _ in 0..POOL {
            assert!(relay.want().await, "the pool still had tokens");
        }
        assert!(
            !relay.want().await,
            "a token beyond the pool was handed out"
        );
        settle("the pool to run dry", || pool.tokens() == 0).await;

        // The command's exec ends with its whole tree killed: the relay is
        // told, finds every holder gone, and exits holding all three tokens.
        drop(fifo);
        assert_eq!(
            relay.heard(Duration::from_secs(5)).await,
            Some(FINISH),
            "the relay is told the command ended"
        );
        drop(relay);
        settle("the pool to refill", || pool.tokens() == POOL).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(pool.tokens(), POOL, "no token was minted beyond the pool");
    }

    #[tokio::test]
    async fn returned_tokens_go_back_and_fabricated_ones_are_discarded() {
        const POOL: usize = 3;
        let pool = Pool::new(POOL);
        let (relays_tx, relays) = async_channel::unbounded();
        let runtime = Pool::runtime(ScriptedExec(relays_tx));
        let (fifo, relay) = scripted(&pool, &runtime, &relays).await;

        assert!(relay.want().await);
        assert!(relay.want().await);
        // Two handed out; five come back. Only two may reach the pool.
        relay.say(&[TOKEN; 5]).await;
        drop(fifo);
        assert_eq!(relay.heard(Duration::from_secs(5)).await, Some(FINISH));
        drop(relay);
        settle("the pool to refill", || pool.tokens() == POOL).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(pool.tokens(), POOL, "fabricated tokens reached the pool");
    }

    #[tokio::test]
    async fn two_relays_never_hold_more_than_the_pool() {
        const POOL: usize = 4;
        let pool = Pool::new(POOL);
        let (relays_tx, relays) = async_channel::unbounded();
        let runtime = Pool::runtime(ScriptedExec(relays_tx));
        let (fifo_a, relay_a) = scripted(&pool, &runtime, &relays).await;
        let (fifo_b, relay_b) = scripted(&pool, &runtime, &relays).await;

        let (mut held_a, mut held_b) = (0, 0);
        for _ in 0..POOL {
            let (a, b) = future::zip(relay_a.want(), relay_b.want()).await;
            held_a += usize::from(a);
            held_b += usize::from(b);
            assert!(
                held_a + held_b <= POOL,
                "over-subscribed: {held_a}+{held_b}"
            );
        }
        assert_eq!(held_a + held_b, POOL);
        assert!(
            held_a >= 1 && held_b >= 1,
            "a relay starved: {held_a}/{held_b}"
        );

        drop((fifo_a, fifo_b));
        for relay in [relay_a, relay_b] {
            assert_eq!(relay.heard(Duration::from_secs(5)).await, Some(FINISH));
        }
        settle("the pool to refill", || pool.tokens() == POOL).await;
    }

    #[tokio::test]
    async fn a_relay_that_fails_to_start_fails_the_command() {
        let pool = Pool::new(1);
        let (relays_tx, relays) = async_channel::unbounded();
        let runtime = Pool::runtime(ScriptedExec(relays_tx));
        let python = ContainerPath::new("/opt/aither/tools/key/bin/python3").unwrap();
        let open = open_container_job_fifo(&pool.broker, &runtime, &python, TokioGlobal);
        let relay = async {
            let relay = relays.recv().await.unwrap();
            relay
                .exit
                .send(Ok(exited(127, "python3: not found")))
                .await
                .unwrap();
            drop(relay.to_host);
        };
        let (opened, ()) = future::zip(open, relay).await;
        let error = opened.unwrap_err();
        assert!(
            matches!(&error, JobserverError::ContainerRelay(message) if message.contains("python3: not found")),
            "{error}"
        );
        settle("the lease's token to return", || pool.tokens() == 1).await;
    }

    /// Runs each script on this machine, standing in for a container on the
    /// same kernel: the relay program itself runs, so the fifo it makes and
    /// brokers is real.
    struct HostShellExec;

    impl crate::ContainerExec for HostShellExec {
        fn exec(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            std::future::ready(Err("only the relay runs here".to_string()))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            script: &str,
            _working_dir: &str,
            kill_rx: Receiver<()>,
            stdin_rx: Receiver<Vec<u8>>,
            stdout: Sender<Vec<u8>>,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            use futures_lite::{AsyncReadExt as _, AsyncWriteExt as _};
            let script = script.to_string();
            async move {
                let mut child = async_process::Command::new("sh")
                    .arg("-c")
                    .arg(&script)
                    .stdin(async_process::Stdio::piped())
                    .stdout(async_process::Stdio::piped())
                    .stderr(async_process::Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .map_err(|error| error.to_string())?;
                let mut stdin = child.stdin.take().unwrap();
                let mut child_stdout = child.stdout.take().unwrap();
                let mut child_stderr = child.stderr.take().unwrap();
                // Feeds stdin until the host closes it, which closes the
                // pipe; it never ends the exec itself.
                let feed = async move {
                    while let Ok(bytes) = stdin_rx.recv().await {
                        if stdin.write_all(&bytes).await.is_err() {
                            break;
                        }
                    }
                    drop(stdin);
                    future::pending::<Result<ContainerExecOutcome, String>>().await
                };
                let pump = async move {
                    let mut buffer = [0u8; 4096];
                    while let Ok(count @ 1..) = child_stdout.read(&mut buffer).await {
                        if stdout.send(buffer[..count].to_vec()).await.is_err() {
                            break;
                        }
                    }
                };
                let run = async {
                    let mut stderr = Vec::new();
                    let _ = child_stderr.read_to_end(&mut stderr).await;
                    let status = child.status().await.map_err(|error| error.to_string())?;
                    Ok(ContainerExecOutcome::Completed {
                        output: std::process::Output {
                            status,
                            stdout: Vec::new(),
                            stderr,
                        },
                        killed: Vec::new(),
                    })
                };
                let ran = async { future::zip(run, pump).await.0 };
                let killed = async {
                    let _ = kill_rx.recv().await;
                    Ok(ContainerExecOutcome::Killed)
                };
                future::or(ran, future::or(feed, killed)).await
            }
        }
    }

    fn host_python() -> ContainerPath {
        let python = std::env::split_paths(&std::env::var_os("PATH").unwrap())
            .map(|dir| dir.join("python3"))
            .find(|path| path.is_file())
            .expect("the relay tests need python3 on PATH");
        ContainerPath::new(python.to_str().unwrap()).unwrap()
    }

    /// A container fifo brokered by the real relay, and a client end on it
    /// as a build tool holds it.
    async fn relayed(pool: &Pool, runtime: &ContainerShellRuntime) -> (ContainerJobFifo, File) {
        let fifo = open_container_job_fifo(&pool.broker, runtime, &host_python(), TokioGlobal)
            .await
            .unwrap();
        let client = open_client_end(std::path::Path::new(fifo.path().as_str())).unwrap();
        (fifo, client)
    }

    fn take_all(client: &File, held: &mut usize) {
        while take_client_token(client).is_ok() {
            *held += 1;
        }
    }

    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "the relay runs in Linux containers only; it watches the fifo with inotify"
    )]
    #[tokio::test]
    async fn container_command_draws_from_the_pool_through_its_own_fifo() {
        const POOL: usize = 3;
        let pool = Pool::new(POOL);
        let runtime = Pool::runtime(HostShellExec);
        let (fifo, client) = relayed(&pool, &runtime).await;
        let auth = format!("-j --jobserver-auth=fifo:{}", fifo.path());
        assert!(fifo.environment().iter().all(|(_, value)| *value == auth));
        assert!(!auth.contains(pool.dir.path().to_str().unwrap()));

        let mut held = 0;
        settle("the command to hold the pool", || {
            take_all(&client, &mut held);
            held == POOL
        })
        .await;
        for _ in 0..20 {
            async_io::Timer::after(Duration::from_millis(5)).await;
            take_all(&client, &mut held);
            assert!(held <= POOL, "the command holds {held} of {POOL}");
        }
        assert_eq!(pool.tokens(), 0);

        // An exiting command writes its tokens back before it goes.
        for _ in 0..held {
            return_client_token(&client).unwrap();
        }
        drop(client);
        drop(fifo);
        settle("the pool to be full", || pool.tokens() == POOL).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(pool.tokens(), POOL);
    }

    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "the relay runs in Linux containers only; it watches the fifo with inotify"
    )]
    #[tokio::test]
    async fn killed_container_command_returns_every_token() {
        const POOL: usize = 4;
        let pool = Pool::new(POOL);
        let runtime = Pool::runtime(HostShellExec);
        let (fifo, client) = relayed(&pool, &runtime).await;
        let path = fifo.path().as_str().to_string();
        let mut held = 0;
        settle("the command to hold the pool", || {
            take_all(&client, &mut held);
            held == POOL
        })
        .await;

        // The exec ends while a process still holds the fifo: its tokens
        // stay accounted to the command.
        drop(fifo);
        async_io::Timer::after(Duration::from_millis(200)).await;
        assert_eq!(pool.tokens(), 0, "tokens returned while a holder lives");

        // The holder is killed with every token: all come back, none extra.
        drop(client);
        settle("the pool to refill", || pool.tokens() == POOL).await;
        async_io::Timer::after(Duration::from_millis(50)).await;
        assert_eq!(pool.tokens(), POOL);
        settle("the relay to remove its fifo", || {
            !std::path::Path::new(&path).exists()
        })
        .await;
    }

    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "the relay runs in Linux containers only; it watches the fifo with inotify"
    )]
    #[tokio::test]
    async fn two_container_commands_never_hold_more_than_the_pool() {
        const POOL: usize = 3;
        let pool = Pool::new(POOL);
        let runtime = Pool::runtime(HostShellExec);
        let (fifo_a, client_a) = relayed(&pool, &runtime).await;
        let (fifo_b, client_b) = relayed(&pool, &runtime).await;
        let (mut held_a, mut held_b) = (0, 0);
        settle("both commands to drain the pool", || {
            take_all(&client_a, &mut held_a);
            take_all(&client_b, &mut held_b);
            held_a + held_b == POOL
        })
        .await;
        assert!(
            held_a >= 1 && held_b >= 1,
            "the pool starved a command: {held_a}/{held_b}"
        );
        for _ in 0..20 {
            async_io::Timer::after(Duration::from_millis(5)).await;
            take_all(&client_a, &mut held_a);
            take_all(&client_b, &mut held_b);
            assert!(
                held_a + held_b <= POOL,
                "over-subscribed: {held_a}+{held_b}"
            );
        }
        drop((client_a, client_b, fifo_a, fifo_b));
        settle("the pool to refill", || pool.tokens() == POOL).await;
    }

    #[test]
    fn environment_names_the_container_fifo() {
        let (finish, _) = async_channel::bounded(1);
        let fifo = ContainerJobFifo {
            path: ContainerPath::new("/tmp/aither-jobserver-1.fifo").unwrap(),
            _finish: finish,
        };
        let environment = fifo.environment();
        assert_eq!(environment.len(), 3);
        for (name, value) in environment {
            assert!(["MAKEFLAGS", "CARGO_MAKEFLAGS", "MFLAGS"].contains(&name.as_str()));
            assert_eq!(
                value,
                "-j --jobserver-auth=fifo:/tmp/aither-jobserver-1.fifo"
            );
        }
    }

    #[test]
    fn fifo_paths_are_fresh_and_in_the_container_tmp() {
        let first = container_fifo_path();
        let second = container_fifo_path();
        assert_ne!(first, second);
        assert_eq!(first.parent().unwrap().as_str(), "/tmp");
        assert_eq!(first.as_unix_path().extension(), Some("fifo"));
    }

    #[test]
    fn relay_command_runs_the_relay_with_the_image_python() {
        let python = ContainerPath::new("/opt/aither/tools/key/bin/python3").unwrap();
        let fifo = ContainerPath::new("/tmp/aither-jobserver-1.fifo").unwrap();
        let words = shell_words::split(&relay_command(&python, &fifo)).unwrap();
        assert_eq!(
            words,
            [
                "/opt/aither/tools/key/bin/python3",
                "-I",
                "-c",
                RELAY_SOURCE,
                "/tmp/aither-jobserver-1.fifo"
            ]
        );
    }
}
