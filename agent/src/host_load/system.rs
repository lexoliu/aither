//! The readers of the platform the agent runs on.
//!
//! macOS reads `kern.memorystatus_vm_pressure_level` and the load average;
//! Linux reads PSI (`/proc/pressure`); Windows queries the low-memory
//! resource notification and the system CPU times.

use super::{HostReader, MachineLoad, MemoryPressure, ReadError};

#[cfg(target_vendor = "apple")]
use apple as platform;
#[cfg(any(target_os = "linux", target_os = "android"))]
use linux as platform;
#[cfg(not(any(
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    windows
)))]
use unsupported as platform;
#[cfg(windows)]
use windows as platform;

/// Reads the host this process runs on.
#[derive(Debug)]
pub struct SystemReader(platform::State);

impl SystemReader {
    /// The machine load this platform's reading is high at by default; see
    /// [`MachineLoad`] for what it measures.
    pub const DEFAULT_LOAD_THRESHOLD: f64 = platform::DEFAULT_LOAD_THRESHOLD;

    /// A reader of this host.
    ///
    /// # Errors
    ///
    /// Fails when the host offers no source of either reading, such as a
    /// Linux kernel built without PSI.
    pub fn new() -> Result<Self, ReadError> {
        let mut state = platform::State::new()?;
        platform::memory_pressure(&state)?;
        platform::machine_load(&mut state)?;
        Ok(Self(state))
    }
}

impl HostReader for SystemReader {
    fn memory_pressure(&mut self) -> Result<MemoryPressure, ReadError> {
        platform::memory_pressure(&self.0)
    }

    fn machine_load(&mut self) -> Result<MachineLoad, ReadError> {
        platform::machine_load(&mut self.0)
    }
}

/// The 10-second average of a PSI line, in percent.
///
/// `content` is the text of a `/proc/pressure/*` file or of a cgroup's
/// `memory.pressure`, and `kind` the line to read, `some` or `full`.
fn psi_avg10(content: &str, kind: &str) -> Option<f64> {
    content
        .lines()
        .find_map(|line| line.strip_prefix(kind)?.strip_prefix(' '))?
        .split_whitespace()
        .find_map(|field| field.strip_prefix("avg10="))?
        .parse()
        .ok()
}

/// Percent of the last ten seconds some task stalled on memory at which
/// pressure is a warning.
const PSI_WARNING_PERCENT: f64 = 10.0;
/// Percent of the last ten seconds every task stalled on memory at which
/// pressure is critical.
const PSI_CRITICAL_PERCENT: f64 = 10.0;

/// The memory pressure the text of a PSI memory file (`/proc/pressure/memory`
/// or a cgroup's `memory.pressure`) states; `None` when the text is not PSI.
pub fn psi_memory_pressure(content: &str) -> Option<MemoryPressure> {
    Some(if psi_avg10(content, "full")? >= PSI_CRITICAL_PERCENT {
        MemoryPressure::Critical
    } else if psi_avg10(content, "some")? >= PSI_WARNING_PERCENT {
        MemoryPressure::Warning
    } else {
        MemoryPressure::Normal
    })
}

#[cfg(target_vendor = "apple")]
mod apple {
    use std::ffi::{CStr, c_int, c_void};
    use std::io;

    use super::{MachineLoad, MemoryPressure, ReadError};

    pub const DEFAULT_LOAD_THRESHOLD: f64 = 1.0;

    const PRESSURE_LEVEL: &CStr = c"kern.memorystatus_vm_pressure_level";

    #[derive(Debug)]
    pub struct State {
        cores: f64,
    }

    impl State {
        #[allow(clippy::unnecessary_wraps)]
        pub fn new() -> Result<Self, ReadError> {
            // The number of cores load is relative to; never a count of core
            // classes.
            let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
            #[allow(clippy::cast_precision_loss)]
            Ok(Self {
                cores: cores as f64,
            })
        }
    }

    pub fn memory_pressure(_: &State) -> Result<MemoryPressure, ReadError> {
        let mut level: c_int = 0;
        let mut size = size_of::<c_int>();
        // SAFETY: `level` and `size` are valid for writes of `size` bytes
        // for the whole call, and the name is NUL terminated.
        let status = unsafe {
            libc::sysctlbyname(
                PRESSURE_LEVEL.as_ptr(),
                (&raw mut level).cast::<c_void>(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 {
            return Err(ReadError::Io {
                what: "kern.memorystatus_vm_pressure_level",
                source: io::Error::last_os_error(),
            });
        }
        // `DISPATCH_MEMORYPRESSURE_*`: 1 normal, 2 warn, 4 critical.
        match level {
            1 => Ok(MemoryPressure::Normal),
            2 => Ok(MemoryPressure::Warning),
            4 => Ok(MemoryPressure::Critical),
            other => Err(ReadError::Malformed {
                what: "kern.memorystatus_vm_pressure_level",
                content: other.to_string(),
            }),
        }
    }

    // Windows keeps the previous reading in its state; the platforms share
    // one signature.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn machine_load(state: &mut State) -> Result<MachineLoad, ReadError> {
        let mut average = 0.0_f64;
        // SAFETY: one sample is requested and `average` holds one.
        let samples = unsafe { libc::getloadavg(&raw mut average, 1) };
        if samples != 1 {
            return Err(ReadError::Io {
                what: "the load average",
                source: io::Error::other("getloadavg returned no sample"),
            });
        }
        Ok(MachineLoad(average / state.cores))
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux {
    use std::fs;

    use super::{MachineLoad, MemoryPressure, ReadError, psi_avg10, psi_memory_pressure};

    /// Stall time of the last ten seconds at which tasks waiting for a CPU
    /// are a load worth telling the agent about.
    pub const DEFAULT_LOAD_THRESHOLD: f64 = 0.3;

    const MEMORY: &str = "/proc/pressure/memory";
    const CPU: &str = "/proc/pressure/cpu";

    #[derive(Debug)]
    pub struct State;

    impl State {
        #[allow(clippy::unnecessary_wraps)]
        pub const fn new() -> Result<Self, ReadError> {
            Ok(Self)
        }
    }

    fn read(path: &'static str) -> Result<String, ReadError> {
        fs::read_to_string(path).map_err(|source| ReadError::Io { what: path, source })
    }

    fn avg10(path: &'static str, content: &str, kind: &str) -> Result<f64, ReadError> {
        psi_avg10(content, kind).ok_or_else(|| ReadError::Malformed {
            what: path,
            content: content.to_owned(),
        })
    }

    pub fn memory_pressure(_: &State) -> Result<MemoryPressure, ReadError> {
        let content = read(MEMORY)?;
        psi_memory_pressure(&content).ok_or(ReadError::Malformed {
            what: MEMORY,
            content,
        })
    }

    pub fn machine_load(_: &mut State) -> Result<MachineLoad, ReadError> {
        let content = read(CPU)?;
        Ok(MachineLoad(avg10(CPU, &content, "some")? / 100.0))
    }
}

#[cfg(windows)]
mod windows {
    use std::io;

    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Memory::{
        CreateMemoryResourceNotification, LowMemoryResourceNotification,
        QueryMemoryResourceNotification,
    };
    use windows_sys::Win32::System::Threading::GetSystemTimes;

    use super::{MachineLoad, MemoryPressure, ReadError};

    /// The share of CPU time not spent idle at which load is worth telling
    /// the agent about.
    pub const DEFAULT_LOAD_THRESHOLD: f64 = 0.9;

    /// Cumulative CPU time in 100 ns ticks.
    #[derive(Debug, Clone, Copy)]
    struct Times {
        idle: u64,
        total: u64,
    }

    #[derive(Debug)]
    pub struct State {
        previous: Times,
    }

    fn ticks(time: FILETIME) -> u64 {
        (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime)
    }

    fn times() -> Result<Times, ReadError> {
        let (mut idle, mut kernel, mut user) = (
            FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            },
            FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            },
            FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            },
        );
        // SAFETY: the three pointers are valid for writes for the call.
        let ok = unsafe { GetSystemTimes(&raw mut idle, &raw mut kernel, &raw mut user) };
        if ok == 0 {
            return Err(ReadError::Io {
                what: "GetSystemTimes",
                source: io::Error::last_os_error(),
            });
        }
        // Kernel time includes the idle time.
        Ok(Times {
            idle: ticks(idle),
            total: ticks(kernel) + ticks(user),
        })
    }

    impl State {
        pub fn new() -> Result<Self, ReadError> {
            Ok(Self { previous: times()? })
        }
    }

    pub fn memory_pressure(_: &State) -> Result<MemoryPressure, ReadError> {
        // SAFETY: creating a notification takes no pointers.
        let handle = unsafe { CreateMemoryResourceNotification(LowMemoryResourceNotification) };
        if handle.is_null() {
            return Err(ReadError::Io {
                what: "the low-memory resource notification",
                source: io::Error::last_os_error(),
            });
        }
        let mut low = 0;
        // SAFETY: `handle` is the notification just created and `low` is
        // valid for writes for the call.
        let queried = unsafe { QueryMemoryResourceNotification(handle, &raw mut low) };
        let queried_error = io::Error::last_os_error();
        // SAFETY: `handle` is open and closed exactly once.
        unsafe { CloseHandle(handle) };
        if queried == 0 {
            return Err(ReadError::Io {
                what: "the low-memory resource notification",
                source: queried_error,
            });
        }
        // The notification has one threshold: the system is low on memory.
        Ok(if low == 0 {
            MemoryPressure::Normal
        } else {
            MemoryPressure::Warning
        })
    }

    pub fn machine_load(state: &mut State) -> Result<MachineLoad, ReadError> {
        let now = times()?;
        let total = now.total.saturating_sub(state.previous.total);
        let idle = now.idle.saturating_sub(state.previous.idle);
        state.previous = now;
        if total == 0 {
            return Ok(MachineLoad(0.0));
        }
        #[allow(clippy::cast_precision_loss)]
        let busy = 1.0 - idle as f64 / total as f64;
        Ok(MachineLoad(busy))
    }
}

#[cfg(not(any(
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    windows
)))]
mod unsupported {
    use super::{MachineLoad, MemoryPressure, ReadError};

    pub const DEFAULT_LOAD_THRESHOLD: f64 = 1.0;

    #[derive(Debug)]
    pub struct State;

    impl State {
        pub const fn new() -> Result<Self, ReadError> {
            Err(ReadError::Unsupported {
                what: "host memory pressure and load",
            })
        }
    }

    pub fn memory_pressure(_: &State) -> Result<MemoryPressure, ReadError> {
        unreachable!("an unsupported host has no reader")
    }

    pub fn machine_load(_: &mut State) -> Result<MachineLoad, ReadError> {
        unreachable!("an unsupported host has no reader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PSI_MEMORY: &str = "some avg10=12.50 avg60=3.00 avg300=1.00 total=123456\nfull avg10=0.40 avg60=0.10 avg300=0.00 total=789\n";

    #[test]
    fn psi_lines_are_read_by_kind() {
        assert_eq!(psi_avg10(PSI_MEMORY, "some"), Some(12.5));
        assert_eq!(psi_avg10(PSI_MEMORY, "full"), Some(0.4));
        assert_eq!(psi_avg10("some avg60=1.0", "some"), None);
        assert_eq!(psi_avg10("", "full"), None);
    }

    #[test]
    fn psi_text_states_a_memory_pressure_level() {
        let stall = |some, full| {
            format!(
                "some avg10={some} avg60=0.00 avg300=0.00 total=1\nfull avg10={full} avg60=0.00 avg300=0.00 total=1\n"
            )
        };
        assert_eq!(
            psi_memory_pressure(&stall("0.00", "0.00")),
            Some(MemoryPressure::Normal)
        );
        assert_eq!(
            psi_memory_pressure(&stall("12.00", "0.00")),
            Some(MemoryPressure::Warning)
        );
        assert_eq!(
            psi_memory_pressure(&stall("30.00", "10.00")),
            Some(MemoryPressure::Critical)
        );
        assert_eq!(psi_memory_pressure("not pressure"), None);
    }

    #[cfg(any(target_vendor = "apple", windows))]
    #[test]
    fn the_host_reader_reads_this_host() {
        let mut reader = SystemReader::new().expect("this host offers both readings");
        reader.memory_pressure().expect("memory pressure reads");
        let load = reader.machine_load().expect("machine load reads");
        assert!(load.0 >= 0.0, "load is not negative: {load:?}");
    }
}
