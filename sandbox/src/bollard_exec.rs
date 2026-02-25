//! Container exec implementation backed by bollard.

use std::future::Future;
use std::process::{ExitStatus, Output};
use std::sync::Arc;
use std::time::Duration;

use async_channel::{Receiver, Sender};
use bollard::Docker;
use bollard::container::LogOutput;
use bollard::exec::CreateExecOptions;
use futures_lite::StreamExt;

use crate::shell_session::{ContainerExec, ContainerExecOutcome};

/// Foreground tasks are promoted when the process blocks on stdin.
pub const CONTAINER_STDIN_BLOCKED_NOTICE: &str = "Auto-promoted to background: container process appears blocked on stdin (detected via /proc/<pid>/syscall). Use input_terminal to continue.";

const STDIN_WATCH_INTERVAL: Duration = Duration::from_millis(500);

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

fn parse_kernel_u64(value: &str) -> Option<u64> {
    let raw = value.trim();
    if let Some(hex) = raw.strip_prefix("0x") {
        return u64::from_str_radix(hex, 16).ok();
    }
    raw.parse::<u64>().ok()
}

/// Detect whether `/proc/<pid>/syscall` indicates `read(0, ...)`.
#[must_use]
pub fn is_waiting_on_stdin(syscall_dump: &str) -> bool {
    let line = syscall_dump.trim();
    if line.is_empty() || line.eq_ignore_ascii_case("running") {
        return false;
    }

    let mut parts = line.split_whitespace();
    let Some(syscall_no) = parts.next() else {
        return false;
    };
    let Some(fd_raw) = parts.next() else {
        return false;
    };

    parse_kernel_u64(syscall_no).is_some_and(|syscall| syscall == 0)
        && parse_kernel_u64(fd_raw).is_some_and(|fd| fd == 0)
}

async fn detect_stdin_blocked_inside_container(
    client: &Docker,
    container_id: &str,
    pid: i64,
) -> Result<bool, String> {
    let probe_script = format!("cat /proc/{pid}/syscall 2>/dev/null || true");
    let probe = client
        .create_exec(
            container_id,
            CreateExecOptions {
                cmd: Some(vec!["bash", "-lc", probe_script.as_str()]),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| format!("failed to create syscall probe exec: {e}"))?;

    let started = client
        .start_exec(&probe.id, None)
        .await
        .map_err(|e| format!("failed to start syscall probe exec: {e}"))?;

    let mut stdout = String::new();
    match started {
        bollard::exec::StartExecResults::Attached { mut output, .. } => {
            while let Some(chunk) = output.next().await {
                match chunk {
                    Ok(LogOutput::StdOut { message }) | Ok(LogOutput::Console { message }) => {
                        stdout.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(LogOutput::StdErr { .. }) | Ok(LogOutput::StdIn { .. }) => {}
                    Err(e) => return Err(format!("syscall probe stream error: {e}")),
                }
            }
        }
        bollard::exec::StartExecResults::Detached => {
            return Err("syscall probe exec detached unexpectedly".to_string());
        }
    }

    Ok(is_waiting_on_stdin(&stdout))
}

async fn kill_exec_pid_inside_container(
    client: &Docker,
    container_id: &str,
    exec_id: &str,
) -> Result<(), String> {
    let inspect = client
        .inspect_exec(exec_id)
        .await
        .map_err(|e| format!("failed to inspect exec for kill: {e}"))?;
    if !inspect.running.unwrap_or(false) {
        return Ok(());
    }
    let Some(pid) = inspect.pid else {
        return Ok(());
    };
    if pid <= 0 {
        return Ok(());
    }

    let kill_script = format!("kill -KILL {pid} >/dev/null 2>&1 || true");
    let kill_exec = client
        .create_exec(
            container_id,
            CreateExecOptions {
                cmd: Some(vec!["bash", "-lc", kill_script.as_str()]),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| format!("failed to create in-container kill exec: {e}"))?;
    match client
        .start_exec(&kill_exec.id, None)
        .await
        .map_err(|e| format!("failed to start in-container kill exec: {e}"))?
    {
        bollard::exec::StartExecResults::Attached { mut output, .. } => {
            while let Some(chunk) = output.next().await {
                if let Err(error) = chunk {
                    return Err(format!("in-container kill stream error: {error}"));
                }
            }
        }
        bollard::exec::StartExecResults::Detached => {
            return Err("in-container kill exec detached unexpectedly".to_string());
        }
    }

    Ok(())
}

enum StreamEvent {
    Output(Option<Result<LogOutput, bollard::errors::Error>>),
    WatchdogTick,
    Cancel,
}

impl ContainerExec for BollardContainerExec {
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();
        let client = self.client.clone();

        async move {
            let config = CreateExecOptions {
                cmd: Some(vec!["bash", "-c", &script]),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                working_dir: Some(working_dir.as_str()),
                ..Default::default()
            };

            let exec_instance = client
                .create_exec(&container_id, config)
                .await
                .map_err(|e| format!("failed to create exec: {e}"))?;
            let exec_id = exec_instance.id.clone();

            let start = client
                .start_exec(&exec_id, None)
                .await
                .map_err(|e| format!("failed to start exec: {e}"))?;

            let mut output = match start {
                bollard::exec::StartExecResults::Attached { output, .. } => output,
                bollard::exec::StartExecResults::Detached => {
                    return Err("exec started in detached mode unexpectedly".to_string());
                }
            };

            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut watchdog_active = stdin_blocked_notice.is_some();
            let mut notice_sent = false;

            loop {
                let event = if watchdog_active && !notice_sent {
                    futures_lite::future::or(
                        futures_lite::future::or(
                            async { StreamEvent::Output(output.next().await) },
                            async {
                                let _ = kill_rx.recv().await;
                                StreamEvent::Cancel
                            },
                        ),
                        async {
                            async_io::Timer::after(STDIN_WATCH_INTERVAL).await;
                            StreamEvent::WatchdogTick
                        },
                    )
                    .await
                } else {
                    futures_lite::future::or(
                        async { StreamEvent::Output(output.next().await) },
                        async {
                            let _ = kill_rx.recv().await;
                            StreamEvent::Cancel
                        },
                    )
                    .await
                };

                match event {
                    StreamEvent::Output(Some(Ok(LogOutput::StdOut { message }))) => {
                        stdout.extend_from_slice(&message);
                    }
                    StreamEvent::Output(Some(Ok(LogOutput::StdErr { message }))) => {
                        stderr.extend_from_slice(&message);
                    }
                    StreamEvent::Output(Some(Ok(LogOutput::Console { message }))) => {
                        stdout.extend_from_slice(&message);
                    }
                    StreamEvent::Output(Some(Ok(LogOutput::StdIn { .. }))) => {}
                    StreamEvent::Output(Some(Err(e))) => {
                        return Err(format!("exec stream error: {e}"));
                    }
                    StreamEvent::Output(None) => {
                        break;
                    }
                    StreamEvent::Cancel => {
                        if let Err(error) =
                            kill_exec_pid_inside_container(&client, &container_id, &exec_id).await
                        {
                            tracing::warn!(
                                error = %error,
                                exec_id = %exec_id,
                                "failed to kill container exec after cancellation request"
                            );
                        }
                        return Ok(ContainerExecOutcome::Killed);
                    }
                    StreamEvent::WatchdogTick => {
                        let inspect = client.inspect_exec(&exec_id).await.map_err(|e| {
                            format!("failed to inspect running exec for stdin watchdog: {e}")
                        })?;

                        if !inspect.running.unwrap_or(false) {
                            watchdog_active = false;
                            continue;
                        }

                        let Some(pid) = inspect.pid else {
                            continue;
                        };
                        if pid <= 0 {
                            continue;
                        }

                        match detect_stdin_blocked_inside_container(&client, &container_id, pid)
                            .await
                        {
                            Ok(true) => {
                                if let Some(notice_tx) = stdin_blocked_notice.as_ref() {
                                    let _ = notice_tx
                                        .try_send(CONTAINER_STDIN_BLOCKED_NOTICE.to_string());
                                }
                                notice_sent = true;
                                watchdog_active = false;
                            }
                            Ok(false) => {}
                            Err(error) => {
                                tracing::debug!(error = %error, pid, "stdin watchdog probe failed");
                            }
                        }
                    }
                }
            }

            let inspect = client
                .inspect_exec(&exec_id)
                .await
                .map_err(|e| format!("failed to inspect exec: {e}"))?;

            let exit_code = inspect.exit_code.unwrap_or(-1) as i32;
            Ok(ContainerExecOutcome::Completed(Output {
                status: ExitStatusExt::from_raw(exit_code),
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

#[cfg(test)]
mod tests {
    use super::is_waiting_on_stdin;

    #[test]
    fn stdin_blocked_detection_handles_decimal_and_hex() {
        assert!(is_waiting_on_stdin("0 0 0 0 0 0"));
        assert!(is_waiting_on_stdin("0x0 0x0 0x0 0x0"));
    }

    #[test]
    fn stdin_blocked_detection_ignores_non_blocking_syscalls() {
        assert!(!is_waiting_on_stdin("running"));
        assert!(!is_waiting_on_stdin(""));
        assert!(!is_waiting_on_stdin("1 0 0 0"));
        assert!(!is_waiting_on_stdin("0 1 0 0"));
        assert!(!is_waiting_on_stdin("garbage"));
    }
}
