//! Disk pressure: how little free space the agent's home environment has.
//!
//! The levels mirror memory pressure's and are judged by free space against
//! three thresholds. Only the home environment is watched, the place the
//! agent works in: the volume the embedder names on the machine it runs on
//! (the one holding the working directory unless told otherwise) and, when
//! the agent works in a container, the filesystem of that container, which
//! has its own size limit when the container is given one. A remote
//! execution environment is never watched. Each volume has its own
//! [`DiskReader`] and level; agents of one process can live on different
//! volumes, so nothing here is shared between them. Disk pressure never
//! kills anything.

use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use aither_sandbox::ContainerProbe;

use crate::host_load::{HostLoadEvent, ReadError};
use crate::pressure::{LevelTracker, Shift};

#[cfg(test)]
mod tests;

const GIB: u64 = 1024 * 1024 * 1024;

/// How little free space a volume has.
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
/// A host implements this for the volume it reads; tests inject their own.
pub trait DiskReader: Send + Sync + 'static {
    /// The volume's name as the agent is told it.
    fn volume(&self) -> String;

    /// The free space of the volume now, in bytes.
    ///
    /// # Errors
    ///
    /// Fails when the volume's source cannot be read.
    fn free_bytes(&self) -> impl Future<Output = Result<u64, ReadError>> + Send;
}

/// Object-safe twin of [`DiskReader`], used only through [`AnyDiskReader`].
pub trait DiskReaderImpl: Send + Sync {
    /// [`DiskReader::volume`].
    fn volume_name(&self) -> String;

    /// [`DiskReader::free_bytes`] through a boxed future.
    fn free_bytes_boxed(&self)
    -> Pin<Box<dyn Future<Output = Result<u64, ReadError>> + Send + '_>>;
}

impl<T: DiskReader> DiskReaderImpl for T {
    fn volume_name(&self) -> String {
        self.volume()
    }

    fn free_bytes_boxed(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<u64, ReadError>> + Send + '_>> {
        Box::pin(self.free_bytes())
    }
}

/// A type-erased [`DiskReader`], what the builders store.
#[derive(Clone)]
pub struct AnyDiskReader(Arc<dyn DiskReaderImpl>);

impl AnyDiskReader {
    /// Erases `reader`.
    #[must_use]
    pub fn new(reader: impl DiskReader) -> Self {
        Self(Arc::new(reader))
    }
}

impl DiskReader for AnyDiskReader {
    fn volume(&self) -> String {
        self.0.volume_name()
    }

    async fn free_bytes(&self) -> Result<u64, ReadError> {
        self.0.free_bytes_boxed().await
    }
}

impl fmt::Debug for AnyDiskReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyDiskReader")
    }
}

/// Reads the free space of the volume holding a path on the machine this
/// process runs on: `statvfs` on Unix, `GetDiskFreeSpaceExW` on Windows (the
/// space available to the calling user, quotas honored).
#[derive(Debug, Clone)]
pub struct SystemDisk {
    path: PathBuf,
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

    async fn free_bytes(&self) -> Result<u64, ReadError> {
        let path = self.path.clone();
        blocking::unblock(move || system_free_bytes(&path)).await
    }
}

#[cfg(unix)]
fn system_free_bytes(path: &Path) -> Result<u64, ReadError> {
    use std::os::unix::ffi::OsStrExt as _;

    let path_c =
        std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|_| ReadError::Malformed {
            what: "the volume path",
            content: path.display().to_string(),
        })?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path_c` is NUL terminated and `stat` is valid for a write of
    // one `statvfs`, which the call fills on success.
    let status = unsafe { libc::statvfs(path_c.as_ptr(), stat.as_mut_ptr()) };
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

#[cfg(windows)]
fn system_free_bytes(path: &Path) -> Result<u64, ReadError> {
    use std::os::windows::ffi::OsStrExt as _;

    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut available = 0_u64;
    // SAFETY: `wide` is NUL terminated and outlives the call, `available` is
    // valid for a write of a `u64`, and the two totals the call may also
    // fill are optional, so null asks for neither.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &raw mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(ReadError::Io {
            what: "the free space of the volume",
            source: std::io::Error::last_os_error(),
        });
    }
    Ok(available)
}

#[cfg(not(any(unix, windows)))]
fn system_free_bytes(_: &Path) -> Result<u64, ReadError> {
    Err(ReadError::Unsupported {
        what: "the free space of the volume",
    })
}

/// Reads the free space of the filesystem that holds the agent's home
/// directory inside the container it works in.
///
/// The read goes through the container's exec path. That filesystem's size is
/// the container's own limit when it has one.
#[derive(Debug, Clone)]
pub struct ContainerDisk {
    probe: ContainerProbe,
}

impl ContainerDisk {
    /// A reader of the container `probe` runs scripts in.
    #[must_use]
    pub const fn new(probe: ContainerProbe) -> Self {
        Self { probe }
    }
}

impl DiskReader for ContainerDisk {
    fn volume(&self) -> String {
        format!("the container's filesystem ({})", self.probe.home())
    }

    async fn free_bytes(&self) -> Result<u64, ReadError> {
        Ok(self.probe.free_bytes().await?)
    }
}

/// The volumes of the home environment: the machine's `local` volume, and
/// the container's filesystem when the agent works in a `container`. A
/// remote execution environment has no volume here.
#[must_use]
pub fn home_volumes(local: AnyDiskReader, container: Option<ContainerProbe>) -> Vec<AnyDiskReader> {
    let mut volumes = vec![local];
    volumes.extend(container.map(|probe| AnyDiskReader::new(ContainerDisk::new(probe))));
    volumes
}

/// The free space of every volume of `readers` now, in order; the volumes
/// are read together. A failed read is logged and left out as `None`.
pub async fn read_free_bytes(readers: Vec<AnyDiskReader>) -> Vec<Option<u64>> {
    futures::future::join_all(readers.iter().map(|reader| async {
        reader
            .free_bytes()
            .await
            .inspect_err(|error| {
                tracing::warn!(%error, volume = reader.volume(), "disk free space could not be read");
            })
            .ok()
    }))
    .await
}

/// One watched volume and the level it was last at.
#[derive(Debug)]
struct Watched {
    reader: AnyDiskReader,
    level: LevelTracker<DiskPressure>,
}

/// An agent's watch over the volumes of its home environment.
#[derive(Debug)]
pub struct DiskWatch {
    thresholds: DiskThresholds,
    volumes: Vec<Watched>,
}

impl DiskWatch {
    pub(crate) fn new(
        thresholds: DiskThresholds,
        readers: impl IntoIterator<Item = AnyDiskReader>,
    ) -> Self {
        Self {
            thresholds,
            volumes: readers
                .into_iter()
                .map(|reader| Watched {
                    reader,
                    level: LevelTracker::new(DiskPressure::Normal),
                })
                .collect(),
        }
    }

    /// The readers of the watched volumes, in the order they are watched,
    /// for a read that borrows nothing of the watch.
    pub(crate) fn readers(&self) -> Vec<AnyDiskReader> {
        self.volumes
            .iter()
            .map(|volume| volume.reader.clone())
            .collect()
    }

    /// Takes a reading of each volume's free space (`None` where it could
    /// not be read); returns the events its changes of level raise.
    pub(crate) fn observe(&mut self, free: &[Option<u64>]) -> Vec<HostLoadEvent> {
        let thresholds = &self.thresholds;
        self.volumes
            .iter_mut()
            .zip(free)
            .filter_map(|(volume, free)| {
                let free = (*free)?;
                let level = thresholds.level(free);
                let volume_name = volume.reader.volume();
                let free = aither_sandbox::MemorySize(free);
                Some(match volume.level.observe(level)? {
                    Shift::Rose(level) => HostLoadEvent::DiskPressureRose {
                        volume: volume_name,
                        level,
                        free,
                    },
                    Shift::Fell(level) => HostLoadEvent::DiskPressureDroppedBack {
                        volume: volume_name,
                        level,
                        free,
                    },
                })
            })
            .collect()
    }
}
