use std::time::Duration;

/// Notice injected into background promotion previews when stdin blocking is detected.
pub const TERMINAL_STDIN_BLOCKED_NOTICE: &str = "The process appears blocked on stdin (detected via /proc/<pid>/syscall). Rerun it as a background job with terminal_background and feed input with terminal_input.";

/// Poll interval for `/proc/<pid>/syscall` stdin-block detection.
pub const STDIN_WATCH_INTERVAL: Duration = Duration::from_millis(500);

fn parse_kernel_u64(value: &str) -> Option<u64> {
    let raw = value.trim();
    if let Some(hex) = raw.strip_prefix("0x") {
        return u64::from_str_radix(hex, 16).ok();
    }
    raw.parse::<u64>().ok()
}

const fn is_linux_read_syscall(syscall_no: u64) -> bool {
    matches!(syscall_no, 0 | 3 | 63)
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

    parse_kernel_u64(syscall_no).is_some_and(is_linux_read_syscall)
        && parse_kernel_u64(fd_raw).is_some_and(|fd| fd == 0)
}

#[cfg(target_os = "linux")]
/// Detect whether a local process is blocked on `read(0, ...)` using `/proc/<pid>/syscall`.
///
/// # Errors
/// Returns an I/O error when the process syscall state cannot be read.
pub async fn detect_stdin_blocked_for_local_pid(pid: u32) -> std::io::Result<bool> {
    let path = format!("/proc/{pid}/syscall");
    let syscall_dump = async_fs::read_to_string(path).await?;
    Ok(is_waiting_on_stdin(&syscall_dump))
}

#[cfg(not(target_os = "linux"))]
/// Non-Linux platforms do not expose the `/proc/<pid>/syscall` probe used by stdin detection.
///
/// # Errors
/// This implementation does not fail.
pub async fn detect_stdin_blocked_for_local_pid(_pid: u32) -> std::io::Result<bool> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::is_waiting_on_stdin;

    #[test]
    fn stdin_blocked_detection_handles_decimal_and_hex() {
        assert!(is_waiting_on_stdin("0 0 0 0 0 0"));
        assert!(is_waiting_on_stdin("3 0 0 0 0 0"));
        assert!(is_waiting_on_stdin("63 0 0 0 0 0"));
        assert!(is_waiting_on_stdin("0x0 0x0 0x0 0x0"));
        assert!(is_waiting_on_stdin("0x3 0x0 0x0 0x0"));
        assert!(is_waiting_on_stdin("0x3f 0x0 0x0 0x0"));
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
