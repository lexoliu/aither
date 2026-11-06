#![allow(clippy::assert_is_empty)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_sandbox::{ForegroundWait, MemorySize};

use super::*;
use crate::clock::{AnyClock, ManualClock};
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

/// A volume whose free space the test sets.
#[derive(Clone)]
struct ScriptedVolume {
    name: &'static str,
    free: Arc<Mutex<u64>>,
}

impl ScriptedVolume {
    fn new(name: &'static str, free: u64) -> Self {
        Self {
            name,
            free: Arc::new(Mutex::new(free)),
        }
    }

    fn set_free(&self, free: u64) {
        *self.free.lock().unwrap() = free;
    }
}

impl DiskReader for ScriptedVolume {
    fn volume(&self) -> String {
        self.name.to_string()
    }

    fn free_bytes(&mut self) -> Result<u64, ReadError> {
        Ok(*self.free.lock().unwrap())
    }
}

/// An agent's monitor over a scripted volume.
struct Agent {
    volume: ScriptedVolume,
    monitor: HostLoadMonitor<QuietHost>,
    clock: AnyClock,
}

impl Agent {
    fn new(name: &'static str, free: u64, thresholds: DiskThresholds) -> Self {
        let volume = ScriptedVolume::new(name, free);
        let monitor = HostLoadMonitor::new(
            QuietHost,
            HostLoadConfig::new(),
            ForegroundWait::new(Duration::from_secs(10)),
        )
        .with_disk(AnyDiskReader::new(volume.clone()), thresholds);
        Self {
            volume,
            monitor,
            clock: AnyClock::new(ManualClock::new()),
        }
    }

    /// The volume now has `free` bytes; returns the events the next reading
    /// raises.
    fn free(&mut self, free: u64) -> Vec<HostLoadEvent> {
        self.volume.set_free(free);
        self.monitor.tick(&self.clock)
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

#[test]
fn each_crossing_down_alerts_once_and_each_crossing_up_sends_the_drop_back_once() {
    let mut agent = Agent::new("/work", 50 * GIB, DiskThresholds::default());
    assert!(agent.free(40 * GIB).is_empty());

    assert_eq!(
        agent.free(9 * GIB),
        [rose("/work", DiskPressure::Yellow, 9 * GIB)]
    );
    assert!(agent.free(8 * GIB).is_empty());
    assert_eq!(
        agent.free(4 * GIB),
        [rose("/work", DiskPressure::Red, 4 * GIB)]
    );
    assert!(agent.free(3 * GIB + 1).is_empty());
    assert_eq!(
        agent.free(2 * GIB),
        [rose("/work", DiskPressure::Final, 2 * GIB)]
    );
    assert!(agent.free(GIB).is_empty());

    assert_eq!(
        agent.free(4 * GIB),
        [dropped("/work", DiskPressure::Red, 4 * GIB)]
    );
    assert!(agent.free(4 * GIB).is_empty());
    assert_eq!(
        agent.free(9 * GIB),
        [dropped("/work", DiskPressure::Yellow, 9 * GIB)]
    );
    assert_eq!(
        agent.free(30 * GIB),
        [dropped("/work", DiskPressure::Normal, 30 * GIB)]
    );
    assert!(agent.free(30 * GIB).is_empty());
}

#[test]
fn a_fall_across_several_levels_is_one_alert_at_the_level_reached() {
    let mut agent = Agent::new("/work", 50 * GIB, DiskThresholds::default());
    assert_eq!(agent.free(GIB), [rose("/work", DiskPressure::Final, GIB)]);
}

#[test]
fn changed_thresholds_apply_without_a_rebuild() {
    let thresholds = DiskThresholds::default();
    let mut agent = Agent::new("/work", 50 * GIB, thresholds.clone());
    assert!(agent.free(20 * GIB).is_empty());

    thresholds.set(30 * GIB, 25 * GIB, 22 * GIB);
    assert_eq!(
        agent.free(20 * GIB),
        [rose("/work", DiskPressure::Final, 20 * GIB)]
    );

    thresholds.set(10 * GIB, 5 * GIB, 3 * GIB);
    assert_eq!(
        agent.free(20 * GIB),
        [dropped("/work", DiskPressure::Normal, 20 * GIB)]
    );
}

#[test]
fn two_agents_on_different_volumes_are_alerted_for_their_own_volume_only() {
    let mut first = Agent::new("/volumes/fast", 50 * GIB, DiskThresholds::default());
    let mut second = Agent::new("/volumes/slow", 50 * GIB, DiskThresholds::default());

    assert_eq!(
        first.free(2 * GIB),
        [rose("/volumes/fast", DiskPressure::Final, 2 * GIB)]
    );
    assert!(second.free(50 * GIB).is_empty());

    assert_eq!(
        second.free(9 * GIB),
        [rose("/volumes/slow", DiskPressure::Yellow, 9 * GIB)]
    );
    assert!(first.free(2 * GIB).is_empty());
    assert_eq!(
        first.free(40 * GIB),
        [dropped("/volumes/fast", DiskPressure::Normal, 40 * GIB)]
    );
    assert!(second.free(9 * GIB).is_empty());
}

#[test]
fn disk_pressure_only_ever_alerts() {
    let mut agent = Agent::new("/work", 50 * GIB, DiskThresholds::default());
    for free in [2 * GIB, GIB, 0, 40 * GIB] {
        for event in agent.free(free) {
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

#[cfg(unix)]
#[test]
fn the_system_reader_reports_the_free_space_of_a_real_volume() {
    let mut disk = SystemDisk::new(std::env::temp_dir());
    assert!(disk.free_bytes().unwrap() > 0);
}
