//! The guard over scripted pressure, footprints and registry entries. The
//! jobs are `sleep` processes the tests start in their own process groups;
//! no test holds memory.

#![allow(clippy::assert_is_empty, clippy::significant_drop_tightening)]

use std::collections::HashMap;
use std::os::unix::process::CommandExt as _;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

use aither_sandbox::{
    AnyFootprintReader, AnyFootprintRegistry, FootprintError, FootprintReader, FootprintRegistry,
    JobMemory, JobRegistry, JobStatus, KillCause, ResponseWindow, TerminalMode,
    job_registry_channel,
};

use super::*;
use crate::clock::{AnyClock, ManualClock};

const GIB: u64 = 1024 * 1024 * 1024;
const WINDOW: Duration = Duration::from_secs(60);

/// The footprints the test says processes hold.
#[derive(Clone, Default)]
struct Footprints {
    by_pid: Arc<Mutex<HashMap<u32, u64>>>,
    physical: u64,
}

impl FootprintReader for Footprints {
    fn tree_footprints(&self, roots: &[u32]) -> Result<Vec<u64>, FootprintError> {
        let by_pid = self.by_pid.lock().unwrap();
        Ok(roots
            .iter()
            .map(|pid| by_pid.get(pid).copied().unwrap_or(0))
            .collect())
    }

    fn physical_memory(&self) -> Result<u64, FootprintError> {
        Ok(self.physical)
    }
}

/// What every agent of the machine has published.
#[derive(Clone, Default)]
struct Machine(Arc<Mutex<HashMap<&'static str, u64>>>);

struct Entry {
    machine: Machine,
    agent: &'static str,
}

impl FootprintRegistry for Entry {
    fn publish(&self, total_bytes: u64) -> Result<(), FootprintError> {
        let mut entries = self.machine.0.lock().unwrap();
        if total_bytes == 0 {
            entries.remove(self.agent);
        } else {
            entries.insert(self.agent, total_bytes);
        }
        Ok(())
    }

    fn others(&self) -> Result<Vec<u64>, FootprintError> {
        Ok(self
            .machine
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|(agent, _)| **agent != self.agent)
            .map(|(_, bytes)| *bytes)
            .collect())
    }
}

/// A harmless process standing for a job, in a group of its own.
struct Sleeper(Child);

impl Sleeper {
    fn start() -> Self {
        Self(
            Command::new("sleep")
                .arg("600")
                .process_group(0)
                .spawn()
                .unwrap(),
        )
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }
}

impl Drop for Sleeper {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// One agent: its jobs, and a guard over them.
struct Agent {
    registry: JobRegistry,
    footprints: Footprints,
    guard: RunawayGuard,
    sleepers: Vec<Sleeper>,
    window: ResponseWindow,
}

impl Agent {
    fn new(physical: u64, machine: Option<(&Machine, &'static str)>) -> Self {
        let (registry, service) = job_registry_channel();
        tokio::spawn(async move { service.serve().await });
        let footprints = Footprints {
            physical,
            ..Footprints::default()
        };
        let window = ResponseWindow::new(WINDOW);
        let memory = JobMemory::new(
            registry.clone(),
            AnyFootprintReader::new(footprints.clone()),
            machine.map(|(machine, agent)| {
                AnyFootprintRegistry::new(Entry {
                    machine: machine.clone(),
                    agent,
                })
            }),
            window.clone(),
        );
        Self {
            registry,
            footprints,
            guard: RunawayGuard::new(memory),
            sleepers: Vec::new(),
            window,
        }
    }

    /// Starts a job holding `bytes`.
    async fn job(&mut self, task_id: &str, bytes: u64) -> u32 {
        let sleeper = Sleeper::start();
        let pid = sleeper.pid();
        self.registry
            .register(pid, task_id, "exec", task_id, TerminalMode::Sandboxed, None)
            .await;
        self.footprints.by_pid.lock().unwrap().insert(pid, bytes);
        self.sleepers.push(sleeper);
        pid
    }

    async fn status(&self, task_id: &str) -> JobStatus {
        self.registry
            .get_by_task_id_settled(task_id)
            .await
            .expect("the job is registered")
            .status
    }
}

struct Clock {
    manual: ManualClock,
    any: AnyClock,
}

impl Clock {
    fn new() -> Self {
        let manual = ManualClock::new();
        Self {
            any: AnyClock::new(manual.clone()),
            manual,
        }
    }

    fn now(&self) -> Instant {
        self.any.now()
    }

    fn advance(&self, secs: u64) {
        self.manual.advance(Duration::from_secs(secs));
    }
}

fn task_ids(jobs: &[aither_sandbox::JobFootprint]) -> Vec<&str> {
    jobs.iter().map(|job| job.task_id.as_str()).collect()
}

#[tokio::test]
async fn yellow_red_and_dropped_back_alerts_fire_once_per_change_and_list_jobs_by_footprint() {
    let clock = Clock::new();
    let mut agent = Agent::new(64 * GIB, None);
    agent.job("small", GIB).await;
    agent.job("large", 4 * GIB).await;
    let mut observe = async |pressure| agent.guard.observe(clock.now(), Some(pressure)).await;

    assert!(observe(MemoryPressure::Normal).await.is_empty());
    let yellow = observe(MemoryPressure::Warning).await;
    let [HostLoadEvent::MemoryYellow { jobs }] = yellow.as_slice() else {
        panic!("one yellow alert: {yellow:?}");
    };
    assert_eq!(task_ids(jobs), ["large", "small"]);
    assert!(observe(MemoryPressure::Warning).await.is_empty());

    let red = observe(MemoryPressure::Critical).await;
    let [HostLoadEvent::MemoryRed { jobs }] = red.as_slice() else {
        panic!("one red alert: {red:?}");
    };
    assert_eq!(task_ids(jobs), ["large", "small"]);
    assert!(observe(MemoryPressure::Critical).await.is_empty());

    let dropped = observe(MemoryPressure::Warning).await;
    assert!(
        matches!(
            dropped.as_slice(),
            [HostLoadEvent::MemoryDroppedBack {
                level: MemoryPressure::Warning,
                ..
            }]
        ),
        "{dropped:?}"
    );
    let normal = observe(MemoryPressure::Normal).await;
    assert!(
        matches!(
            normal.as_slice(),
            [HostLoadEvent::MemoryDroppedBack {
                level: MemoryPressure::Normal,
                ..
            }]
        ),
        "{normal:?}"
    );
    assert!(observe(MemoryPressure::Normal).await.is_empty());
}

#[tokio::test]
async fn an_agent_over_physical_memory_loses_its_largest_job_when_the_window_ends_unanswered() {
    let clock = Clock::new();
    let mut agent = Agent::new(10 * GIB, None);
    agent.job("small", 3 * GIB).await;
    agent.job("large", 8 * GIB).await;

    let warned = agent
        .guard
        .observe(clock.now(), Some(MemoryPressure::Normal))
        .await;
    let [HostLoadEvent::MemoryFinalWarning { job, window, .. }] = warned.as_slice() else {
        panic!("the final warning: {warned:?}");
    };
    assert_eq!((job.task_id.as_str(), *window), ("large", WINDOW));

    clock.advance(59);
    assert!(
        agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Normal))
            .await
            .is_empty()
    );
    assert_eq!(agent.status("large").await, JobStatus::Running);

    clock.advance(1);
    let killed = agent
        .guard
        .observe(clock.now(), Some(MemoryPressure::Normal))
        .await;
    assert!(
        matches!(
            killed.as_slice(),
            [HostLoadEvent::JobKilledForMemory { task_id, .. }] if task_id == "large"
        ),
        "{killed:?}"
    );
    assert!(matches!(
        agent.status("large").await,
        JobStatus::Killed {
            cause: KillCause::MemoryRunaway {
                footprint_bytes,
                physical_bytes,
            },
            ..
        } if footprint_bytes == 8 * GIB && physical_bytes == 10 * GIB
    ));
    assert_eq!(agent.status("small").await, JobStatus::Running);
}

#[tokio::test]
async fn a_job_the_agent_stopped_in_time_is_not_killed_and_reports_a_user_kill() {
    let clock = Clock::new();
    let mut agent = Agent::new(10 * GIB, None);
    agent.job("small", 3 * GIB).await;
    let large = agent.job("large", 8 * GIB).await;

    let warned = agent
        .guard
        .observe(clock.now(), Some(MemoryPressure::Normal))
        .await;
    assert_eq!(warned.len(), 1, "{warned:?}");
    clock.advance(30);
    assert!(agent.registry.kill(large).await);
    // The processes are gone, so the agent's jobs fit the machine again.
    agent.footprints.by_pid.lock().unwrap().insert(large, 0);

    clock.advance(120);
    let after = agent
        .guard
        .observe(clock.now(), Some(MemoryPressure::Normal))
        .await;
    assert!(after.is_empty(), "{after:?}");
    assert!(matches!(
        agent.status("large").await,
        JobStatus::Killed {
            cause: KillCause::User,
            ..
        }
    ));
    assert_eq!(agent.status("small").await, JobStatus::Running);
}

#[tokio::test]
async fn jobs_under_physical_memory_are_never_killed_whatever_the_pressure() {
    let clock = Clock::new();
    let mut agent = Agent::new(10 * GIB, None);
    agent.job("a", 4 * GIB).await;
    agent.job("b", 5 * GIB).await;
    for pressure in [
        MemoryPressure::Critical,
        MemoryPressure::Critical,
        MemoryPressure::Warning,
    ] {
        clock.advance(600);
        let events = agent.guard.observe(clock.now(), Some(pressure)).await;
        assert!(
            events.iter().all(|event| !matches!(
                event,
                HostLoadEvent::MemoryFinalWarning { .. } | HostLoadEvent::JobKilledForMemory { .. }
            )),
            "{events:?}"
        );
    }
    assert_eq!(agent.status("a").await, JobStatus::Running);
    assert_eq!(agent.status("b").await, JobStatus::Running);
}

#[tokio::test]
async fn the_machine_over_physical_memory_costs_the_largest_contributor_only_while_critical() {
    let clock = Clock::new();
    let machine = Machine::default();
    let mut big = Agent::new(50 * GIB, Some((&machine, "big")));
    let mut small = Agent::new(50 * GIB, Some((&machine, "small")));
    big.job("big-1", 25 * GIB).await;
    big.job("big-2", 10 * GIB).await;
    small.job("small-1", 20 * GIB).await;
    // Both publish; 55 GiB together on a machine of 50.
    for agent in [&mut big, &mut small] {
        agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Warning))
            .await;
    }

    // Below critical nothing is warned about, however long it lasts.
    clock.advance(600);
    for agent in [&mut big, &mut small] {
        let events = agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Warning))
            .await;
        assert!(events.is_empty(), "{events:?}");
    }

    // Critical: only the agent contributing the most is warned, about its
    // largest job.
    let small_events = small
        .guard
        .observe(clock.now(), Some(MemoryPressure::Critical))
        .await;
    assert!(
        small_events
            .iter()
            .all(|event| !matches!(event, HostLoadEvent::MemoryFinalWarning { .. })),
        "{small_events:?}"
    );
    let big_events = big
        .guard
        .observe(clock.now(), Some(MemoryPressure::Critical))
        .await;
    assert!(
        big_events.iter().any(|event| matches!(
            event,
            HostLoadEvent::MemoryFinalWarning { job, .. } if job.task_id == "big-1"
        )),
        "{big_events:?}"
    );

    clock.advance(60);
    let killed = big
        .guard
        .observe(clock.now(), Some(MemoryPressure::Critical))
        .await;
    assert!(
        matches!(
            killed.as_slice(),
            [HostLoadEvent::JobKilledForMemory { task_id, .. }] if task_id == "big-1"
        ),
        "{killed:?}"
    );
    assert!(matches!(
        big.status("big-1").await,
        JobStatus::Killed {
            cause: KillCause::MemoryRunaway { .. },
            ..
        }
    ));
    assert_eq!(big.status("big-2").await, JobStatus::Running);
    // The killed processes hold nothing now: 30 GiB together fits.
    let killed_pid = big.registry.get_by_task_id("big-1").await.unwrap().pid;
    big.footprints.by_pid.lock().unwrap().insert(killed_pid, 0);
    clock.advance(600);
    for agent in [&mut big, &mut small] {
        let events = agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Critical))
            .await;
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, HostLoadEvent::MemoryFinalWarning { .. })),
            "the machine fits again: {events:?}"
        );
    }
    assert_eq!(small.status("small-1").await, JobStatus::Running);
}

#[tokio::test]
async fn kills_continue_down_the_contributors_while_both_conditions_hold() {
    let clock = Clock::new();
    let machine = Machine::default();
    let mut first = Agent::new(20 * GIB, Some((&machine, "first")));
    let mut second = Agent::new(20 * GIB, Some((&machine, "second")));
    let first_pid = first.job("first-1", 15 * GIB).await;
    second.job("second-1", 12 * GIB).await;
    let critical = Some(MemoryPressure::Critical);

    // 27 GiB together on 20: the first contributes the most.
    second.guard.observe(clock.now(), critical).await;
    let warned = first.guard.observe(clock.now(), critical).await;
    assert!(
        warned
            .iter()
            .any(|event| matches!(event, HostLoadEvent::MemoryFinalWarning { .. })),
        "{warned:?}"
    );
    let second_warned = second.guard.observe(clock.now(), critical).await;
    assert!(
        second_warned
            .iter()
            .all(|event| !matches!(event, HostLoadEvent::MemoryFinalWarning { .. })),
        "{second_warned:?}"
    );

    clock.advance(60);
    first.guard.observe(clock.now(), critical).await;
    first.footprints.by_pid.lock().unwrap().insert(first_pid, 0);
    assert!(matches!(
        first.status("first-1").await,
        JobStatus::Killed {
            cause: KillCause::MemoryRunaway { .. },
            ..
        }
    ));

    // 12 GiB is under 20: nothing more. Shrink the machine to keep both
    // conditions true, and the second contributor is next.
    let mut second_only = second;
    second_only.footprints.physical = 8 * GIB;
    second_only.guard = RunawayGuard::new(JobMemory::new(
        second_only.registry.clone(),
        AnyFootprintReader::new(second_only.footprints.clone()),
        Some(AnyFootprintRegistry::new(Entry {
            machine: machine.clone(),
            agent: "second",
        })),
        second_only.window.clone(),
    ));
    clock.advance(10);
    let next = second_only.guard.observe(clock.now(), critical).await;
    assert!(
        next.iter().any(|event| matches!(
            event,
            HostLoadEvent::MemoryFinalWarning { job, .. } if job.task_id == "second-1"
        )),
        "{next:?}"
    );
    clock.advance(60);
    second_only.guard.observe(clock.now(), critical).await;
    assert!(matches!(
        second_only.status("second-1").await,
        JobStatus::Killed {
            cause: KillCause::MemoryRunaway { .. },
            ..
        }
    ));
}

#[test]
fn the_alerts_read_as_plain_text() {
    let job = |task: &str, gib| aither_sandbox::JobFootprint {
        task_id: task.to_string(),
        command: format!("run {task}"),
        pid: 1,
        footprint_bytes: gib * GIB,
    };
    let red = HostLoadEvent::MemoryRed {
        jobs: vec![job("big", 36), job("small", 1)],
    };
    assert_eq!(
        red.alert_text(),
        "The machine's memory is critically tight (pressure: critical). Stop the jobs you can do without now; a job that keeps growing may be killed. Your running jobs by memory, largest first:\n- 'big' (`run big`): 36.0 GiB\n- 'small' (`run small`): 1.0 GiB"
    );
    let killed = HostLoadEvent::JobKilledForMemory {
        task_id: "big".to_string(),
        command: "run big".to_string(),
        footprint: MemorySize(36 * GIB),
        physical: MemorySize(32 * GIB),
    };
    assert_eq!(
        killed.alert_text(),
        "Job 'big' (`run big`) was killed with everything it started: it held 36.0 GiB of memory on a machine with 32.0 GiB, and the final warning went unanswered."
    );
}
