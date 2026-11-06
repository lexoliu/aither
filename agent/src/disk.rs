//! Disk pressure: how little free space the volume that hosts the agent has.
//!
//! The levels mirror memory pressure's and are judged by free space against
//! three thresholds. Each agent watches its own volume through its own
//! [`DiskReader`]; agents of one process can live on different volumes, so
//! nothing here is shared between them. Disk pressure never kills anything.

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::host_load::ReadError;
use crate::pressure::LevelTracker;

#[cfg(test)]
mod tests;

const GIB: u64 = 1024 * 1024 * 1024;

/// How little free space the agent's volume has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiskPressure {
    /// Space is plentiful.
    Normal,
    /// Free space is below the yellow threshold.
    Yellow,
    /// Free space is below the red threshold.
    Red,
    /// Free space is below the final threshold.
    Final,
}

impl fmt::Display for DiskPressure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Normal => "normal",
            Self::Yellow => "yellow",
            Self::Red => "red",
            Self::Final => "final",
        })
    }
}

/// The free space, in bytes, below which each disk pressure level starts.
///
/// A handle: every clone shares the thresholds, so an embedder that keeps one
/// changes them while the agent runs, and the next reading is judged by the
/// new ones.
#[derive(Debug, Clone)]
pub struct DiskThresholds(Arc<Shared>);

#[derive(Debug)]
struct Shared {
    yellow: AtomicU64,
    red: AtomicU64,
    last: AtomicU64,
}

impl DiskThresholds {
    /// Thresholds of `yellow`, `red` and `last` free bytes; `last` is the
    /// final level's.
    #[must_use]
    pub fn new(yellow: u64, red: u64, last: u64) -> Self {
        Self(Arc::new(Shared {
            yellow: AtomicU64::new(yellow),
            red: AtomicU64::new(red),
            last: AtomicU64::new(last),
        }))
    }

    /// Changes the thresholds for every handle.
    pub fn set(&self, yellow: u64, red: u64, last: u64) {
        self.0.yellow.store(yellow, Ordering::Relaxed);
        self.0.red.store(red, Ordering::Relaxed);
        self.0.last.store(last, Ordering::Relaxed);
    }

    /// The level `free` bytes of free space put the volume at.
    #[must_use]
    pub fn level(&self, free: u64) -> DiskPressure {
        if free < self.0.last.load(Ordering::Relaxed) {
            DiskPressure::Final
        } else if free < self.0.red.load(Ordering::Relaxed) {
            DiskPressure::Red
        } else if free < self.0.yellow.load(Ordering::Relaxed) {
            DiskPressure::Yellow
        } else {
            DiskPressure::Normal
        }
    }
}

impl Default for DiskThresholds {
    /// Yellow below 10 GiB free, red below 5 GiB and final below 3 GiB.
    fn default() -> Self {
        Self::new(10 * GIB, 5 * GIB, 3 * GIB)
    }
}

/// Reads the free space of one volume.
///
/// Reads may block; the monitor runs them off the agent loop. A host
/// implements this for the platform it reads; tests inject their own.
pub trait DiskReader: Send + 'static {
    /// The volume's name as the agent is told it.
    fn volume(&self) -> String;

    /// The free space of the volume now, in bytes.
    ///
    /// # Errors
    ///
    /// Fails when the host's source cannot be read.
    fn free_bytes(&mut self) -> Result<u64, ReadError>;
}

/// A type-erased [`DiskReader`], what the builders store.
pub struct AnyDiskReader(Box<dyn DiskReader>);

impl AnyDiskReader {
    /// Erases `reader`.
    #[must_use]
    pub fn new(reader: impl DiskReader) -> Self {
        Self(Box::new(reader))
    }
}

impl DiskReader for AnyDiskReader {
    fn volume(&self) -> String {
        self.0.volume()
    }

    fn free_bytes(&mut self) -> Result<u64, ReadError> {
        self.0.free_bytes()
    }
}

impl fmt::Debug for AnyDiskReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyDiskReader")
    }
}

/// Reads the free space of the volume holding a path (`statvfs`; Windows has
/// no reader yet).
#[derive(Debug, Clone)]
pub struct SystemDisk {
    path: std::path::PathBuf,
}

impl SystemDisk {
    /// A reader of the volume that holds `path`, which names it in alerts.
    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl DiskReader for SystemDisk {
    fn volume(&self) -> String {
        self.path.display().to_string()
    }

    #[cfg(unix)]
    fn free_bytes(&mut self) -> Result<u64, ReadError> {
        use std::os::unix::ffi::OsStrExt as _;

        let path = std::ffi::CString::new(self.path.as_os_str().as_bytes()).map_err(|_| {
            ReadError::Malformed {
                what: "the volume path",
                content: self.path.display().to_string(),
            }
        })?;
        let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: `path` is NUL terminated and `stat` is valid for a write of
        // one `statvfs`, which the call fills on success.
        let status = unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) };
        if status != 0 {
            return Err(ReadError::Io {
                what: "the free space of the volume",
                source: std::io::Error::last_os_error(),
            });
        }
        // SAFETY: `statvfs` returned success, so `stat` is initialized.
        let stat = unsafe { stat.assume_init() };
        // The widths of the fields differ between platforms.
        #[allow(clippy::useless_conversion, clippy::unnecessary_cast)]
        Ok(u64::from(stat.f_bavail) * stat.f_frsize as u64)
    }

    #[cfg(not(unix))]
    fn free_bytes(&mut self) -> Result<u64, ReadError> {
        Err(ReadError::Unsupported {
            what: "the free space of the volume",
        })
    }
}

/// An agent's watch over its volume.
#[derive(Debug)]
pub struct DiskWatch {
    reader: AnyDiskReader,
    thresholds: DiskThresholds,
    level: LevelTracker<DiskPressure>,
}

impl DiskWatch {
    pub(crate) const fn new(reader: AnyDiskReader, thresholds: DiskThresholds) -> Self {
        Self {
            reader,
            thresholds,
            level: LevelTracker::new(DiskPressure::Normal),
        }
    }

    /// The volume's free space now; a failed read is logged and leaves the
    /// level as it was.
    pub(crate) fn read(&mut self) -> Option<u64> {
        self.reader
            .free_bytes()
            .inspect_err(|error| tracing::warn!(%error, "disk free space could not be read"))
            .ok()
    }

    /// Takes a reading of `free` bytes; returns the event a change of level
    /// raises.
    pub(crate) fn observe(&mut self, free: u64) -> Option<crate::HostLoadEvent> {
        use crate::host_load::HostLoadEvent;
        use crate::pressure::Shift;

        let level = self.thresholds.level(free);
        let volume = self.reader.volume();
        let free = aither_sandbox::MemorySize(free);
        match self.level.observe(level)? {
            Shift::Rose(level) => Some(HostLoadEvent::DiskPressureRose {
                volume,
                level,
                free,
            }),
            Shift::Fell(level) => Some(HostLoadEvent::DiskPressureDroppedBack {
                volume,
                level,
                free,
            }),
        }
    }
}
