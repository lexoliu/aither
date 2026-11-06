use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_sandbox::{ForegroundWait, MemorySize};

use super::*;
use crate::host_load::{
    HostLoadConfig, HostLoadEvent, HostLoadMonitor, HostReader, MachineLoad, MemoryPressure,
};

const GIB: u64 = 1024 * 1024 * 1024;

/// A host with nothing to report but its disk.
struct QuietHost;

impl HostReader for QuietHost {
    fn memory_pressure(&mut self) -> Result<MemoryPressure, ReadError> {
        Ok(MemoryPressure::Normal)
    }

    fn machine_load(&mut self) -> Result<MachineLoad, ReadError> {
        Ok(MachineLoad(0.0))
    }
}

/// A volume whose free space the test sets; `None` is a read that fails.
#[derive(Clone)]
struct ScriptedVolume {
    name: &'static str,
    free: Arc<Mutex<Option<u64>>>,
}

impl ScriptedVolume {
    fn new(name: &'static str, free: u64) -> Self {
        Self {
            name,
            free: Arc::new(Mutex::new(Some(free))),
        }
    }

    fn set_free(&self, free: Option<u64>) {
        *self.free.lock().unwrap() = free;
    }
}

impl DiskReader for ScriptedVolume {
    fn volume(&self) -> String {
        self.name.to_string()
    }

    fn free_bytes(&self) -> impl Future<Output = Result<u64, ReadError>> + Send {
        let free = *self.free.lock().unwrap();
        std::future::ready(free.ok_or(ReadError::Unsupported { what: "the volume" }))
    }
}

/// An agent's monitor over the volumes of its home environment.
struct Agent {
    volumes: Vec<ScriptedVolume>,
    monitor: HostLoadMonitor<QuietHost>,
}

impl Agent {
    fn new(name: &'static str, free: u64, thresholds: DiskThresholds) -> Self {
        Self::over(vec![ScriptedVolume::new(name, free)], thresholds)
    }

    fn over(volumes: Vec<ScriptedVolume>, thresholds: DiskThresholds) -> Self {
        Self::watching(
            volumes.clone(),
            volumes.into_iter().map(AnyDiskReader::new).collect(),
            thresholds,
        )
    }

    fn watching(
        volumes: Vec<ScriptedVolume>,
        readers: Vec<AnyDiskReader>,
        thresholds: DiskThresholds,
    ) -> Self {
        let monitor = HostLoadMonitor::new(
            QuietHost,
            HostLoadConfig::new(),
            ForegroundWait::new(Duration::from_secs(10)),
        )
        .with_disk(thresholds, readers);
        Self { volumes, monitor }
    }

    /// The first volume now has `free` bytes; returns the events the next
    /// reading raises.
    async fn free(&mut self, free: u64) -> Vec<HostLoadEvent> {
        self.volumes[0].set_free(Some(free));
        self.read().await
    }

    async fn read(&mut self) -> Vec<HostLoadEvent> {
        let free = self.monitor.read_disks().await;
        self.monitor.observe_disks(&free)
    }
}

fn rose(volume: &str, level: DiskPressure, free: u64) -> HostLoadEvent {
    HostLoadEvent::DiskPressureRose {
        volume: volume.to_string(),
        level,
        free: MemorySize(free),
    }
}

fn dropped(volume: &str, level: DiskPressure, free: u64) -> HostLoadEvent {
    HostLoadEvent::DiskPressureDroppedBack {
        volume: volume.to_string(),
        level,
        free: MemorySize(free),
    }
}

#[tokio::test]
async fn each_crossing_down_alerts_once_and_each_crossing_up_sends_the_drop_back_once() {
    let mut agent = Agent::new("/work", 50 * GIB, DiskThresholds::default());
    assert_eq!(agent.free(40 * GIB).await, []);

    assert_eq!(
        agent.free(9 * GIB).await,
        [rose("/work", DiskPressure::Yellow, 9 * GIB)]
    );
    assert_eq!(agent.free(8 * GIB).await, []);
    assert_eq!(
        agent.free(4 * GIB).await,
        [rose("/work", DiskPressure::Red, 4 * GIB)]
    );
    assert_eq!(agent.free(3 * GIB + 1).await, []);
    assert_eq!(
        agent.free(2 * GIB).await,
        [rose("/work", DiskPressure::Final, 2 * GIB)]
    );
    assert_eq!(agent.free(GIB).await, []);

    assert_eq!(
        agent.free(4 * GIB).await,
        [dropped("/work", DiskPressure::Red, 4 * GIB)]
    );
    assert_eq!(agent.free(4 * GIB).await, []);
    assert_eq!(
        agent.free(9 * GIB).await,
        [dropped("/work", DiskPressure::Yellow, 9 * GIB)]
    );
    assert_eq!(
        agent.free(30 * GIB).await,
        [dropped("/work", DiskPressure::Normal, 30 * GIB)]
    );
    assert_eq!(agent.free(30 * GIB).await, []);
}

#[tokio::test]
async fn a_fall_across_several_levels_is_one_alert_at_the_level_reached() {
    let mut agent = Agent::new("/work", 50 * GIB, DiskThresholds::default());
    assert_eq!(
        agent.free(GIB).await,
        [rose("/work", DiskPressure::Final, GIB)]
    );
}

#[tokio::test]
async fn changed_thresholds_apply_without_a_rebuild() {
    let thresholds = DiskThresholds::default();
    let mut agent = Agent::new("/work", 50 * GIB, thresholds.clone());
    assert_eq!(agent.free(20 * GIB).await, []);

    thresholds.set(30 * GIB, 25 * GIB, 22 * GIB);
    assert_eq!(
        agent.free(20 * GIB).await,
        [rose("/work", DiskPressure::Final, 20 * GIB)]
    );

    thresholds.set(10 * GIB, 5 * GIB, 3 * GIB);
    assert_eq!(
        agent.free(20 * GIB).await,
        [dropped("/work", DiskPressure::Normal, 20 * GIB)]
    );
}

#[tokio::test]
async fn alerts_follow_the_volume_the_home_environment_names() {
    let mut first = Agent::new("/volumes/fast", 50 * GIB, DiskThresholds::default());
    let mut second = Agent::new("/volumes/slow", 50 * GIB, DiskThresholds::default());

    assert_eq!(
        first.free(2 * GIB).await,
        [rose("/volumes/fast", DiskPressure::Final, 2 * GIB)]
    );
    assert_eq!(second.free(50 * GIB).await, []);

    assert_eq!(
        second.free(9 * GIB).await,
        [rose("/volumes/slow", DiskPressure::Yellow, 9 * GIB)]
    );
    assert_eq!(first.free(2 * GIB).await, []);
    assert_eq!(
        first.free(40 * GIB).await,
        [dropped("/volumes/fast", DiskPressure::Normal, 40 * GIB)]
    );
    assert_eq!(second.free(9 * GIB).await, []);
}

#[tokio::test]
async fn disk_pressure_only_ever_alerts() {
    let mut agent = Agent::new("/work", 50 * GIB, DiskThresholds::default());
    for free in [2 * GIB, GIB, 0, 40 * GIB] {
        for event in agent.free(free).await {
            assert!(
                matches!(
                    event,
                    HostLoadEvent::DiskPressureRose { .. }
                        | HostLoadEvent::DiskPressureDroppedBack { .. }
                ),
                "{event:?}"
            );
        }
    }
}

#[tokio::test]
async fn a_volume_that_cannot_be_read_keeps_its_level_and_leaves_the_others_alone() {
    let mut agent = Agent::over(
        vec![
            ScriptedVolume::new("/work", 50 * GIB),
            ScriptedVolume::new("/other", 50 * GIB),
        ],
        DiskThresholds::default(),
    );
    agent.volumes[0].set_free(Some(4 * GIB));
    assert_eq!(
        agent.read().await,
        [rose("/work", DiskPressure::Red, 4 * GIB)]
    );

    agent.volumes[0].set_free(None);
    agent.volumes[1].set_free(Some(2 * GIB));
    assert_eq!(
        agent.read().await,
        [rose("/other", DiskPressure::Final, 2 * GIB)]
    );

    agent.volumes[0].set_free(Some(4 * GIB));
    assert_eq!(agent.read().await, []);
}

#[test]
fn the_alerts_differ_by_level_and_name_the_volume_and_free_space() {
    let texts: Vec<String> = [
        rose("/work", DiskPressure::Yellow, 9 * GIB),
        rose("/work", DiskPressure::Red, 4 * GIB),
        rose("/work", DiskPressure::Final, 2 * GIB),
        dropped("/work", DiskPressure::Red, 4 * GIB),
        dropped("/work", DiskPressure::Normal, 30 * GIB),
    ]
    .iter()
    .map(HostLoadEvent::alert_text)
    .collect();
    for (text, free) in texts
        .iter()
        .zip(["9.0 GiB", "4.0 GiB", "2.0 GiB", "4.0 GiB", "30.0 GiB"])
    {
        assert!(text.contains("/work") && text.contains(free), "{text}");
    }
    let distinct: std::collections::HashSet<_> = texts.iter().collect();
    assert_eq!(distinct.len(), texts.len(), "{texts:?}");
}

#[cfg(any(unix, windows))]
#[tokio::test]
async fn the_system_reader_reports_the_free_space_of_a_real_volume() {
    let disk = SystemDisk::new(std::env::temp_dir());
    assert!(disk.free_bytes().await.unwrap() > 0);
}

#[cfg(any(unix, windows))]
#[tokio::test]
async fn the_system_reader_fails_for_a_path_that_is_not_there() {
    let disk = SystemDisk::new(std::env::temp_dir().join("aither-no-such-directory/inside"));
    assert!(matches!(disk.free_bytes().await, Err(ReadError::Io { .. })));
}

#[cfg(unix)]
mod home {
    use std::os::unix::process::ExitStatusExt as _;

    use aither_sandbox::{
        AnyContainerExec, ContainerExec, ContainerExecOutcome, ContainerProbe, ExecLifetime,
    };
    use async_channel::{Receiver, Sender};

    use super::*;

    /// A container whose filesystem has `kib` KiB available.
    struct Container {
        kib: Arc<Mutex<u64>>,
    }

    impl ContainerExec for Container {
        fn exec(
            &self,
            _container_id: &str,
            script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            assert!(script.starts_with("df -Pk"), "{script}");
            let kib = *self.kib.lock().unwrap();
            std::future::ready(Ok(ContainerExecOutcome::Completed {
                output: std::process::Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: format!(
                        "Filesystem 1024-blocks Used Available Capacity Mounted on\noverlay 8388608 0 {kib} 0% /workspace\n"
                    )
                    .into_bytes(),
                    stderr: Vec::new(),
                },
                killed: Vec::new(),
            }))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _stdout: Sender<Vec<u8>>,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            std::future::ready(Err("a probe is never streamed".to_string()))
        }
    }

    const CONTAINER: &str = "the container's filesystem (/workspace)";

    fn container(free: u64) -> (Arc<Mutex<u64>>, ContainerProbe) {
        let kib = Arc::new(Mutex::new(free / 1024));
        let exec = AnyContainerExec::new(Arc::new(Container {
            kib: Arc::clone(&kib),
        }));
        (kib, ContainerProbe::new(&exec, "box"))
    }

    #[tokio::test]
    async fn a_container_adds_its_own_filesystem_and_its_limit_raises_alerts() {
        let local = ScriptedVolume::new("/work", 500 * GIB);
        let (kib, probe) = container(20 * GIB);
        let readers = home_volumes(AnyDiskReader::new(local.clone()), Some(probe));
        assert_eq!(readers.len(), 2);
        let mut agent = Agent::watching(vec![local], readers, DiskThresholds::default());
        // The machine has plenty, the container's limit does not.
        assert_eq!(agent.read().await, []);

        *kib.lock().unwrap() = 4 * GIB / 1024;
        assert_eq!(
            agent.read().await,
            [rose(CONTAINER, DiskPressure::Red, 4 * GIB)]
        );
        assert_eq!(agent.read().await, []);

        *kib.lock().unwrap() = 2 * GIB / 1024;
        assert_eq!(
            agent.read().await,
            [rose(CONTAINER, DiskPressure::Final, 2 * GIB)]
        );

        *kib.lock().unwrap() = 40 * GIB / 1024;
        assert_eq!(
            agent.read().await,
            [dropped(CONTAINER, DiskPressure::Normal, 40 * GIB)]
        );
    }

    #[tokio::test]
    async fn the_machine_and_the_container_alert_apart() {
        let local = ScriptedVolume::new("/work", 500 * GIB);
        let (kib, probe) = container(20 * GIB);
        let readers = home_volumes(AnyDiskReader::new(local.clone()), Some(probe));
        let mut agent = Agent::watching(vec![local.clone()], readers, DiskThresholds::default());

        local.set_free(Some(9 * GIB));
        *kib.lock().unwrap() = 9 * GIB / 1024;
        assert_eq!(
            agent.read().await,
            [
                rose("/work", DiskPressure::Yellow, 9 * GIB),
                rose(CONTAINER, DiskPressure::Yellow, 9 * GIB),
            ]
        );
        local.set_free(Some(500 * GIB));
        assert_eq!(
            agent.read().await,
            [dropped("/work", DiskPressure::Normal, 500 * GIB)]
        );
    }

    #[tokio::test]
    async fn changed_thresholds_apply_to_the_container_too() {
        let thresholds = DiskThresholds::default();
        let local = ScriptedVolume::new("/work", 500 * GIB);
        let (_, probe) = container(20 * GIB);
        let readers = home_volumes(AnyDiskReader::new(local.clone()), Some(probe));
        let mut agent = Agent::watching(vec![local], readers, thresholds.clone());
        assert_eq!(agent.read().await, []);

        thresholds.set(30 * GIB, 25 * GIB, 22 * GIB);
        assert_eq!(
            agent.read().await,
            [rose(CONTAINER, DiskPressure::Final, 20 * GIB)]
        );
    }

    #[tokio::test]
    async fn a_container_that_cannot_run_df_keeps_the_machine_watched() {
        struct Broken;

        impl ContainerExec for Broken {
            fn exec(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _lifetime: ExecLifetime,
            ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
                std::future::ready(Err("the container is gone".to_string()))
            }

            fn exec_streamed(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _stdout: Sender<Vec<u8>>,
            ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
                std::future::ready(Err("the container is gone".to_string()))
            }
        }

        let local = ScriptedVolume::new("/work", 500 * GIB);
        let probe = ContainerProbe::new(&AnyContainerExec::new(Arc::new(Broken)), "box");
        let readers = home_volumes(AnyDiskReader::new(local.clone()), Some(probe));
        let mut agent = Agent::watching(vec![local.clone()], readers, DiskThresholds::default());
        assert_eq!(agent.monitor.read_disks().await, [Some(500 * GIB), None]);

        local.set_free(Some(2 * GIB));
        assert_eq!(
            agent.read().await,
            [rose("/work", DiskPressure::Final, 2 * GIB)]
        );
    }
}

#[tokio::test]
async fn a_remote_execution_environment_adds_no_volume() {
    // A remote host is never a home environment, so nothing but the
    // machine's own volume is handed to the watch.
    let local = ScriptedVolume::new("/work", 500 * GIB);
    let readers = home_volumes(AnyDiskReader::new(local.clone()), None);
    assert_eq!(readers.len(), 1);
    let mut agent = Agent::watching(vec![local], readers, DiskThresholds::default());
    assert_eq!(agent.monitor.read_disks().await, [Some(500 * GIB)]);
    // Whatever the remote host's disk does, no alert can come from it.
    assert_eq!(agent.read().await, []);
}
