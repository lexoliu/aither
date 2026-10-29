//! Container exec implementation backed by bollard.

use async_channel::{Receiver, Sender};
use bollard::Docker;
use bollard::container::LogOutput;
use bollard::exec::CreateExecOptions;
use futures_lite::{Stream, StreamExt};
use std::future::Future;
use std::pin::Pin;
use std::process::{ExitStatus, Output};
use std::sync::Arc;

use crate::shell_session::{ContainerExec, ContainerExecOutcome};
use crate::stdin_watch::{
    CONTAINER_EXEC_WRAPPER, CONTAINER_JOB_SCRIPT, CONTAINER_STDIN_WATCH_INTERVAL, EpisodeDetector,
    Probe, StdinWatchSignal, parse_container_probe,
};

/// Executes commands in a running container via Docker exec.
#[derive(Debug, Clone)]
pub struct BollardContainerExec {
    client: Arc<Docker>,
}

impl BollardContainerExec {
    /// Create a new exec adapter from a shared Docker client.
    #[must_use]
    pub const fn new(client: Arc<Docker>) -> Self {
        Self { client }
    }
}

/// Exit code of [`CONTAINER_JOB_SCRIPT`]'s `kill` when no pid is recorded.
const KILL_NO_PID: i64 = 3;

/// What a helper script run inside the container printed and returned.
struct ScriptRun {
    stdout: String,
    stderr: String,
    exit_code: Option<i64>,
}

impl ScriptRun {
    /// The output when the script exited 0.
    fn success(self, what: &str) -> Result<String, String> {
        if self.exit_code == Some(0) {
            return Ok(self.stdout);
        }
        Err(format!(
            "{what} exited with {:?}: {}{}",
            self.exit_code,
            self.stdout.trim(),
            self.stderr.trim()
        ))
    }
}

/// Runs `sh -c script sh args…` in the container and collects its output and
/// exit code.
async fn run_script_in_container(
    client: &Docker,
    container_id: &str,
    script: &str,
    args: &[&str],
    what: &str,
) -> Result<ScriptRun, String> {
    let mut cmd = vec!["sh", "-c", script, "sh"];
    cmd.extend_from_slice(args);
    let exec = client
        .create_exec(
            container_id,
            CreateExecOptions {
                cmd: Some(cmd),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| format!("failed to create {what} exec: {e}"))?;

    let started = client
        .start_exec(&exec.id, None)
        .await
        .map_err(|e| format!("failed to start {what} exec: {e}"))?;

    let mut run = ScriptRun {
        stdout: String::new(),
        stderr: String::new(),
        exit_code: None,
    };
    match started {
        bollard::exec::StartExecResults::Attached { mut output, .. } => {
            while let Some(chunk) = output.next().await {
                match chunk {
                    Ok(LogOutput::StdOut { message } | LogOutput::Console { message }) => {
                        run.stdout.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(LogOutput::StdErr { message }) => {
                        run.stderr.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(LogOutput::StdIn { .. }) => {}
                    Err(e) => return Err(format!("{what} stream error: {e}")),
                }
            }
        }
        bollard::exec::StartExecResults::Detached => {
            return Err(format!("{what} exec detached unexpectedly"));
        }
    }
    run.exit_code = client
        .inspect_exec(&exec.id)
        .await
        .map_err(|e| format!("failed to inspect {what} exec: {e}"))?
        .exit_code;
    Ok(run)
}

/// Runs the stdin probe script inside the container for the exec whose root
/// recorded its own in-container pid in `pid_file`.
async fn probe_stdin_inside_container(
    client: &Docker,
    container_id: &str,
    pid_file: &str,
) -> Result<Probe, String> {
    let output = run_script_in_container(
        client,
        container_id,
        CONTAINER_JOB_SCRIPT,
        &["probe", pid_file],
        "stdin probe",
    )
    .await?
    .success("stdin probe")?;
    parse_container_probe(&output)
}

/// Whether the exec is still running.
async fn exec_running(client: &Docker, exec_id: &str) -> Result<bool, String> {
    Ok(client
        .inspect_exec(exec_id)
        .await
        .map_err(|e| format!("failed to inspect exec for kill: {e}"))?
        .running
        .unwrap_or(false))
}

/// Kills the exec's whole process tree inside the container. Docker reports
/// exec pids in the host's namespace, so the kill goes through `pid_file`,
/// where the exec root recorded its own in-container pid. Succeeds only when
/// the exec is gone: a running exec that recorded no pid is an error.
async fn kill_exec_inside_container(
    client: &Docker,
    container_id: &str,
    exec_id: &str,
    pid_file: &str,
) -> Result<(), String> {
    if !exec_running(client, exec_id).await? {
        return Ok(());
    }
    let run = run_script_in_container(
        client,
        container_id,
        CONTAINER_JOB_SCRIPT,
        &["kill", pid_file],
        "in-container kill",
    )
    .await?;
    if run.exit_code == Some(KILL_NO_PID) {
        // The root removes its pid file when it exits, so a missing file is
        // an exec that ended meanwhile — or one that never recorded a pid.
        return if exec_running(client, exec_id).await? {
            Err("the exec is running but recorded no in-container pid".to_string())
        } else {
            Ok(())
        };
    }
    run.success("in-container kill").map(drop)
}

enum StreamEvent {
    Output(Option<Result<LogOutput, bollard::errors::Error>>),
    Stdin(Result<Vec<u8>, async_channel::RecvError>),
    WatchTick,
    Cancel,
}

fn append_log_output(output: LogOutput, stdout: &mut Vec<u8>, stderr: &mut Vec<u8>) {
    match output {
        LogOutput::StdOut { message } | LogOutput::Console { message } => {
            stdout.extend_from_slice(&message);
        }
        LogOutput::StdErr { message } => {
            stderr.extend_from_slice(&message);
        }
        LogOutput::StdIn { .. } => {}
    }
}

async fn write_container_stdin<W>(input: &mut W, bytes: &[u8]) -> Result<(), String>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;

    input
        .write_all(bytes)
        .await
        .map_err(|error| format!("container exec stdin write failed: {error}"))?;
    input
        .flush()
        .await
        .map_err(|error| format!("container exec stdin flush failed: {error}"))
}

async fn shutdown_container_stdin<W>(input: &mut W) -> Result<(), String>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;

    input
        .shutdown()
        .await
        .map_err(|error| format!("container exec stdin shutdown failed: {error}"))
}

/// Alerts for the stdin of one container exec.
struct StdinWatch {
    signal: Sender<StdinWatchSignal>,
    detector: EpisodeDetector,
    /// Writes fed to the exec's stdin so far.
    fed: u64,
}

impl StdinWatch {
    /// Reports a probe outcome through the alert channel.
    fn send(&self, signal: StdinWatchSignal) {
        let _ = self.signal.try_send(signal);
    }
}

struct WatchTick<'a, W> {
    client: &'a Docker,
    container_id: &'a str,
    exec_id: &'a str,
    pid_file: &'a str,
    input: &'a mut W,
    stdin_open: &'a mut bool,
    watch: &'a mut Option<StdinWatch>,
}

/// One watch tick. A probe-side failure — a Docker API error or the script
/// reporting it cannot read the job — is surfaced to the job's alerts and
/// ends detection for this exec; it must never fail the exec itself.
async fn handle_container_watch_tick<W>(tick: WatchTick<'_, W>)
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let Some(watch) = tick.watch.as_mut() else {
        return;
    };
    match tick.client.inspect_exec(tick.exec_id).await {
        Err(error) => {
            watch.send(StdinWatchSignal::Failed(format!(
                "failed to inspect running exec for stdin watch: {error}"
            )));
            *tick.watch = None;
        }
        Ok(inspect) if !inspect.running.unwrap_or(false) => {
            if *tick.stdin_open {
                let _ = shutdown_container_stdin(tick.input).await;
                *tick.stdin_open = false;
            }
            *tick.watch = None;
        }
        Ok(_) => {
            match probe_stdin_inside_container(tick.client, tick.container_id, tick.pid_file).await
            {
                Ok(probe) => {
                    if watch.detector.observe(watch.fed, probe) {
                        watch.send(StdinWatchSignal::Blocked);
                    }
                }
                Err(error) => {
                    watch.send(StdinWatchSignal::Failed(error));
                    *tick.watch = None;
                }
            }
        }
    }
}

async fn inspect_exec_exit_status(client: &Docker, exec_id: &str) -> Result<ExitStatus, String> {
    let inspect = client
        .inspect_exec(exec_id)
        .await
        .map_err(|e| format!("failed to inspect exec: {e}"))?;
    let exit_code = i32::try_from(inspect.exit_code.unwrap_or(-1))
        .map_err(|_| "container exec exit code is outside i32 range".to_string())?;
    Ok(ExitStatusExt::from_raw(exit_code))
}

/// The output stream of an attached container exec.
type ExecOutput = Pin<Box<dyn Stream<Item = Result<LogOutput, bollard::errors::Error>> + Send>>;

/// The writable stdin of an attached container exec.
type ExecInput = Pin<Box<dyn tokio::io::AsyncWrite + Send>>;

/// Creates the exec running `script` in `working_dir` under
/// [`CONTAINER_EXEC_WRAPPER`],
/// which records its pid in `pid_file`, and starts it with its output stream
/// and stdin attached.
async fn start_attached_exec(
    client: &Docker,
    container_id: &str,
    script: &str,
    pid_file: &str,
    working_dir: &str,
) -> Result<(String, ExecOutput, ExecInput), String> {
    let config = CreateExecOptions {
        cmd: Some(vec![
            "sh",
            "-c",
            CONTAINER_EXEC_WRAPPER,
            "sh",
            pid_file,
            script,
        ]),
        attach_stdin: Some(true),
        attach_stdout: Some(true),
        attach_stderr: Some(true),
        working_dir: Some(working_dir),
        ..Default::default()
    };

    let exec_id = client
        .create_exec(container_id, config)
        .await
        .map_err(|e| format!("failed to create exec: {e}"))?
        .id;

    match client
        .start_exec(&exec_id, None)
        .await
        .map_err(|e| format!("failed to start exec: {e}"))?
    {
        bollard::exec::StartExecResults::Attached { output, input } => Ok((exec_id, output, input)),
        bollard::exec::StartExecResults::Detached => {
            Err("exec started in detached mode unexpectedly".to_string())
        }
    }
}

impl ContainerExec for BollardContainerExec {
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdin_blocked: Option<Sender<StdinWatchSignal>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();
        let client = self.client.clone();

        async move {
            // Docker reports the exec's pid in the host's namespace, which a
            // probe inside the container cannot use: the exec root records
            // its own in-container pid for the stdin probe and the kill path.
            let pid_file = format!(
                "/tmp/aither-exec-stdin-{}.pid",
                uuid::Uuid::new_v4().simple()
            );
            let (exec_id, mut output, mut input) =
                start_attached_exec(&client, &container_id, &script, &pid_file, &working_dir)
                    .await?;

            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut watch = stdin_blocked.map(|signal| StdinWatch {
                signal,
                detector: EpisodeDetector::default(),
                fed: 0,
            });
            let mut ticker = async_io::Timer::interval(CONTAINER_STDIN_WATCH_INTERVAL);
            let mut stdin_open = true;

            loop {
                let event = tokio::select! {
                    output = async { output.next().await } => StreamEvent::Output(output),
                    () = async {
                        let _ = kill_rx.recv().await;
                    } => StreamEvent::Cancel,
                    bytes = async { stdin_rx.recv().await }, if stdin_open => StreamEvent::Stdin(bytes),
                    _ = async { ticker.next().await }, if watch.is_some() => StreamEvent::WatchTick,
                };

                match event {
                    StreamEvent::Output(Some(Ok(output))) => {
                        append_log_output(output, &mut stdout, &mut stderr);
                    }
                    StreamEvent::Output(Some(Err(e))) => {
                        return Err(format!("exec stream error: {e}"));
                    }
                    StreamEvent::Output(None) => {
                        break;
                    }
                    StreamEvent::Stdin(Ok(bytes)) => {
                        if let Some(watch) = watch.as_mut() {
                            watch.fed += 1;
                        }
                        write_container_stdin(&mut input, &bytes).await?;
                    }
                    StreamEvent::Stdin(Err(_)) => {
                        shutdown_container_stdin(&mut input).await?;
                        stdin_open = false;
                    }
                    StreamEvent::Cancel => {
                        if stdin_open {
                            let _ = shutdown_container_stdin(&mut input).await;
                        }
                        // A job the kill could not stop is still running, so
                        // it must not be reported killed.
                        kill_exec_inside_container(&client, &container_id, &exec_id, &pid_file)
                            .await
                            .map_err(|error| format!("failed to kill container exec: {error}"))?;
                        return Ok(ContainerExecOutcome::Killed);
                    }
                    StreamEvent::WatchTick => {
                        handle_container_watch_tick(WatchTick {
                            client: &client,
                            container_id: &container_id,
                            exec_id: &exec_id,
                            pid_file: &pid_file,
                            input: &mut input,
                            stdin_open: &mut stdin_open,
                            watch: &mut watch,
                        })
                        .await;
                    }
                }
            }

            let status = inspect_exec_exit_status(&client, &exec_id).await?;
            Ok(ContainerExecOutcome::Completed(Output {
                status,
                stdout,
                stderr,
            }))
        }
    }
}

#[cfg(unix)]
struct ExitStatusExt;

#[cfg(unix)]
impl ExitStatusExt {
    fn from_raw(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt as _;
        ExitStatus::from_raw(code << 8)
    }
}

#[cfg(not(unix))]
struct ExitStatusExt;

#[cfg(not(unix))]
impl ExitStatusExt {
    fn from_raw(code: i32) -> ExitStatus {
        use std::os::windows::process::ExitStatusExt as _;
        ExitStatus::from_raw(code as u32)
    }
}

/// The wrapper scripts run on the host here; the kill and the stdin probe
/// read `/proc` and are tested on Linux in `stdin_watch`.
#[cfg(all(test, unix))]
mod tests {
    use std::process::Command;

    use super::CONTAINER_EXEC_WRAPPER;

    fn run_wrapper(
        pid_file: &std::path::Path,
        script: &str,
        dir: &std::path::Path,
    ) -> std::process::Output {
        Command::new("sh")
            .args(["-c", CONTAINER_EXEC_WRAPPER, "sh"])
            .arg(pid_file)
            .arg(script)
            .current_dir(dir)
            .env("PID_FILE", pid_file)
            .output()
            .expect("run the exec wrapper")
    }

    /// The wrapper records its own pid — the parent of the shell running
    /// the script — keeps the script's exit status, and removes the file
    /// when it exits, also after a script that `exec`s.
    #[test]
    fn wrapper_records_its_pid_and_removes_the_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let pid_file = dir.path().join("exec.pid");
        let run = run_wrapper(
            &pid_file,
            "echo \"$PPID $(cat \"$PID_FILE\")\"; exit 3",
            dir.path(),
        );
        assert_eq!(run.status.code(), Some(3), "the script's status survives");
        let printed = String::from_utf8_lossy(&run.stdout);
        let mut pids = printed.split_whitespace();
        let parent = pids.next().expect("the script's parent pid");
        assert_eq!(
            pids.next(),
            Some(parent),
            "the file holds the wrapper's pid"
        );
        assert!(!pid_file.exists(), "the pid file is removed on exit");

        let run = run_wrapper(&pid_file, "exec true", dir.path());
        assert!(run.status.success());
        assert!(
            !pid_file.exists(),
            "a script that execs still gets cleaned up"
        );
    }

    /// The script sees none of the wrapper's arguments or variables, and a
    /// variable it sets cannot redirect the wrapper's cleanup.
    #[test]
    fn wrapper_hides_its_arguments_from_the_script() {
        let dir = tempfile::tempdir().expect("temp dir");
        let pid_file = dir.path().join("exec.pid");
        let run = run_wrapper(
            &pid_file,
            "echo \"$#[${1-}][${pid_file-}][${script-}]\"; pid_file=user.pid; : > \"$pid_file\"; trap - EXIT",
            dir.path(),
        );
        assert!(run.status.success());
        assert_eq!(String::from_utf8_lossy(&run.stdout), "0[][][]\n");
        assert!(
            dir.path().join("user.pid").exists(),
            "the script's own file stays"
        );
        assert!(!pid_file.exists(), "the wrapper removes its own file");
    }

    /// A wrapper that cannot record its pid fails before running the
    /// script, so the exec can never run without a pid to kill it by.
    #[test]
    fn wrapper_fails_fast_without_a_pid_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let pid_file = dir.path().join("missing").join("exec.pid");
        let run = run_wrapper(&pid_file, "echo ran", dir.path());
        assert_eq!(run.status.code(), Some(125));
        assert!(run.stdout.is_empty(), "the script must not run");
        assert!(
            String::from_utf8_lossy(&run.stderr).contains("cannot record the exec's pid"),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}
