//! The guard over scripted pressure, footprints and registry entries. The
//! jobs are `sleep` processes the tests start in their own process groups;
//! no test holds memory.

#![allow(clippy::assert_is_empty, clippy::significant_drop_tightening)]

use std::collections::HashMap;
use std::os::unix::process::CommandExt as _;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

use aither_sandbox::{
    AnyContainerMemory, AnyFootprintReader, AnyFootprintRegistry, CgroupMemory, ContainerMemory,
    FootprintError, FootprintReader, FootprintRegistry, JobMemory, JobPlace, JobRegistry,
    JobStatus, KillCause, MemoryDomain, ResponseWindow, ScriptLanguage, TerminalMode,
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
    /// The kill switches of the container jobs, by task.
    switches: Vec<(String, async_channel::Receiver<()>)>,
}

/// What the container's cgroup says, scripted.
#[derive(Clone)]
struct Cgroup(Arc<Mutex<CgroupState>>);

struct CgroupState {
    limit: Option<u64>,
    pressure: &'static str,
    jobs: HashMap<String, u64>,
}

const CALM: &str = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0";
const STALLED: &str = "some avg10=12.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0";
const THRASHING: &str = "some avg10=40.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=15.00 avg60=0.00 avg300=0.00 total=0";

impl Cgroup {
    fn limited(limit: u64) -> Self {
        Self(Arc::new(Mutex::new(CgroupState {
            limit: Some(limit),
            pressure: CALM,
            jobs: HashMap::new(),
        })))
    }

    fn pressure(&self, pressure: &'static str) {
        self.0.lock().unwrap().pressure = pressure;
    }

    fn hold(&self, task_id: &str, bytes: u64) {
        self.0
            .lock()
            .unwrap()
            .jobs
            .insert(task_id.to_string(), bytes);
    }
}

impl ContainerMemory for Cgroup {
    fn cgroup(&self) -> impl Future<Output = Result<CgroupMemory, FootprintError>> + Send {
        let state = self.0.lock().unwrap();
        std::future::ready(Ok(CgroupMemory {
            current_bytes: state.jobs.values().sum(),
            limit_bytes: state.limit,
            pressure: state.pressure.to_string(),
        }))
    }

    fn job_footprints(
        &self,
        task_ids: &[String],
    ) -> impl Future<Output = Result<Vec<u64>, FootprintError>> + Send {
        let state = self.0.lock().unwrap();
        std::future::ready(Ok(task_ids
            .iter()
            .map(|id| state.jobs.get(id).copied().unwrap_or(0))
            .collect()))
    }
}

impl Agent {
    fn new(physical: u64, machine: Option<(&Machine, &'static str)>) -> Self {
        Self::build(physical, machine, None)
    }

    /// An agent that works in a container reading as `cgroup`.
    fn containerized(physical: u64, cgroup: &Cgroup) -> Self {
        Self::build(physical, None, Some(cgroup))
    }

    fn build(
        physical: u64,
        machine: Option<(&Machine, &'static str)>,
        cgroup: Option<&Cgroup>,
    ) -> Self {
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
        let memory = match cgroup {
            Some(cgroup) => memory.with_container(AnyContainerMemory::new(cgroup.clone())),
            None => memory,
        };
        Self {
            registry,
            footprints,
            guard: RunawayGuard::new(memory),
            sleepers: Vec::new(),
            window,
            switches: Vec::new(),
        }
    }

    /// Starts a job inside the container holding `bytes`. A container job
    /// is killed through its kill switch, which the test keeps.
    async fn container_job(&mut self, cgroup: &Cgroup, task_id: &str, bytes: u64) {
        let pid = self.next_pid();
        self.registry
            .register_at(
                JobPlace::Container,
                pid,
                &aither_sandbox::test_ids::lease(task_id),
                "exec",
                task_id,
                ScriptLanguage::Shell,
                TerminalMode::Sandboxed,
            )
            .await;
        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        self.registry.attach_kill_switch(pid, kill_tx).await;
        self.switches.push((task_id.to_string(), kill_rx));
        cgroup
            .0
            .lock()
            .unwrap()
            .jobs
            .insert(task_id.to_string(), bytes);
        // The container's pid means nothing to the host: if it were read
        // there, this is what it would find.
        self.footprints
            .by_pid
            .lock()
            .unwrap()
            .insert(pid, 500 * GIB);
    }

    /// Starts a job on a remote host holding `bytes` there. Its local `ssh`
    /// stand-in is a real process, which must never be killed.
    async fn remote_job(&mut self, task_id: &str, bytes: u64) -> u32 {
        let sleeper = Sleeper::start();
        let pid = sleeper.pid();
        self.registry
            .register_at(
                JobPlace::Remote,
                pid,
                &aither_sandbox::test_ids::lease(task_id),
                "exec",
                task_id,
                ScriptLanguage::Shell,
                TerminalMode::Sandboxed,
            )
            .await;
        self.footprints.by_pid.lock().unwrap().insert(pid, bytes);
        self.sleepers.push(sleeper);
        pid
    }

    fn next_pid(&self) -> u32 {
        4_000_000 + u32::try_from(self.switches.len()).unwrap()
    }

    /// Whether the kill switch of the container job `task_id` was pulled.
    fn container_job_killed(&self, task_id: &str) -> bool {
        self.switches
            .iter()
            .find(|(id, _)| id == task_id)
            .is_some_and(|(_, rx)| rx.try_recv().is_ok())
    }

    /// Starts a job holding `bytes`.
    async fn job(&mut self, task_id: &str, bytes: u64) -> u32 {
        let sleeper = Sleeper::start();
        let pid = sleeper.pid();
        self.registry
            .register(
                pid,
                &aither_sandbox::test_ids::lease(task_id),
                "exec",
                task_id,
                TerminalMode::Sandboxed,
                None,
            )
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
    let [HostLoadEvent::MemoryYellow { jobs, .. }] = yellow.as_slice() else {
        panic!("one yellow alert: {yellow:?}");
    };
    assert_eq!(task_ids(jobs), ["large", "small"]);
    assert!(observe(MemoryPressure::Warning).await.is_empty());

    let red = observe(MemoryPressure::Critical).await;
    let [HostLoadEvent::MemoryRed { jobs, .. }] = red.as_slice() else {
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
                capacity_bytes,
                domain: MemoryDomain::Machine,
            },
            ..
        } if footprint_bytes == 8 * GIB && capacity_bytes == 10 * GIB
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
    big.job("anchor", 25 * GIB).await;
    big.job("bridge", 10 * GIB).await;
    small.job("canyon", 20 * GIB).await;
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
            HostLoadEvent::MemoryFinalWarning { job, .. } if job.task_id == "anchor"
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
            [HostLoadEvent::JobKilledForMemory { task_id, .. }] if task_id == "anchor"
        ),
        "{killed:?}"
    );
    assert!(matches!(
        big.status("anchor").await,
        JobStatus::Killed {
            cause: KillCause::MemoryRunaway { .. },
            ..
        }
    ));
    assert_eq!(big.status("bridge").await, JobStatus::Running);
    // The killed processes hold nothing now: 30 GiB together fits.
    let killed_pid = big.registry.get_by_task_id("anchor").await.unwrap().pid;
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
    assert_eq!(small.status("canyon").await, JobStatus::Running);
}

#[tokio::test]
async fn kills_continue_down_the_contributors_while_both_conditions_hold() {
    let clock = Clock::new();
    let machine = Machine::default();
    let mut first = Agent::new(20 * GIB, Some((&machine, "first")));
    let mut second = Agent::new(20 * GIB, Some((&machine, "second")));
    let first_pid = first.job("first", 15 * GIB).await;
    second.job("second", 12 * GIB).await;
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
        first.status("first").await,
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
            HostLoadEvent::MemoryFinalWarning { job, .. } if job.task_id == "second"
        )),
        "{next:?}"
    );
    clock.advance(60);
    second_only.guard.observe(clock.now(), critical).await;
    assert!(matches!(
        second_only.status("second").await,
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
        domain: MemoryDomain::Machine,
        jobs: vec![job("big", 36), job("small", 1)],
    };
    assert_eq!(
        red.alert_text(),
        "The machine's memory is critically tight (pressure: critical). Stop the jobs you can do without now; a job that keeps growing may be killed. Your running jobs by memory, largest first:\n- 'big' (`run big`): 36.0 GiB\n- 'small' (`run small`): 1.0 GiB"
    );
    let killed = HostLoadEvent::JobKilledForMemory {
        domain: MemoryDomain::Machine,
        task_id: "big".to_string(),
        command: "run big".to_string(),
        footprint: MemorySize(36 * GIB),
        capacity: MemorySize(32 * GIB),
    };
    assert_eq!(
        killed.alert_text(),
        "Job 'big' (`run big`) was killed with everything it started: it held 36.0 GiB of the 32.0 GiB of memory the machine has, and the final warning went unanswered."
    );
    let container = HostLoadEvent::MemoryFinalWarning {
        domain: MemoryDomain::Container,
        job: job("big", 3),
        window: WINDOW,
        capacity: MemorySize(4 * GIB),
    };
    assert_eq!(
        container.alert_text(),
        "Final memory warning: job 'big' (`run big`) holds 3.0 GiB of memory and is killed in 60s, with everything it started, unless you stop it or free memory yourself. The container has 4.0 GiB of memory in all."
    );
}

const fn is_container(event: &HostLoadEvent) -> bool {
    matches!(
        event,
        HostLoadEvent::MemoryYellow {
            domain: MemoryDomain::Container,
            ..
        } | HostLoadEvent::MemoryRed {
            domain: MemoryDomain::Container,
            ..
        } | HostLoadEvent::MemoryDroppedBack {
            domain: MemoryDomain::Container,
            ..
        } | HostLoadEvent::MemoryFinalWarning {
            domain: MemoryDomain::Container,
            ..
        } | HostLoadEvent::JobKilledForMemory {
            domain: MemoryDomain::Container,
            ..
        }
    )
}

#[tokio::test]
async fn a_containers_own_pressure_raises_its_own_alerts_once_per_change() {
    let clock = Clock::new();
    let cgroup = Cgroup::limited(8 * GIB);
    let mut agent = Agent::containerized(64 * GIB, &cgroup);
    agent.container_job(&cgroup, "small", GIB).await;
    agent.container_job(&cgroup, "large", 3 * GIB).await;
    // The machine is calm throughout: only the container's pressure moves.
    let mut observe = async |pressure| {
        cgroup.pressure(pressure);
        agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Normal))
            .await
    };

    assert!(observe(CALM).await.is_empty());
    let yellow = observe(STALLED).await;
    let [HostLoadEvent::MemoryYellow { domain, jobs }] = yellow.as_slice() else {
        panic!("one yellow alert: {yellow:?}");
    };
    assert_eq!(*domain, MemoryDomain::Container);
    assert_eq!(task_ids(jobs), ["large", "small"]);
    assert!(observe(STALLED).await.is_empty());

    let red = observe(THRASHING).await;
    assert!(
        matches!(
            red.as_slice(),
            [HostLoadEvent::MemoryRed {
                domain: MemoryDomain::Container,
                ..
            }]
        ),
        "{red:?}"
    );
    assert!(observe(THRASHING).await.is_empty());

    let dropped = observe(STALLED).await;
    assert!(
        matches!(
            dropped.as_slice(),
            [HostLoadEvent::MemoryDroppedBack {
                domain: MemoryDomain::Container,
                level: MemoryPressure::Warning,
                ..
            }]
        ),
        "{dropped:?}"
    );
    let normal = observe(CALM).await;
    assert!(
        matches!(
            normal.as_slice(),
            [HostLoadEvent::MemoryDroppedBack {
                domain: MemoryDomain::Container,
                level: MemoryPressure::Normal,
                ..
            }]
        ),
        "{normal:?}"
    );
    assert!(observe(CALM).await.is_empty());
}

#[tokio::test]
async fn a_container_is_judged_against_its_own_limit_not_the_machines_memory() {
    let clock = Clock::new();
    let cgroup = Cgroup::limited(4 * GIB);
    // The machine has 64 GiB, so 4 GiB of jobs would be nothing to it.
    let mut agent = Agent::containerized(64 * GIB, &cgroup);
    agent.container_job(&cgroup, "small", GIB).await;
    agent.container_job(&cgroup, "large", 3 * GIB).await;
    let normal = Some(MemoryPressure::Normal);

    // Together the jobs fill the limit: the final warning names the largest.
    let warned = agent.guard.observe(clock.now(), normal).await;
    let [
        HostLoadEvent::MemoryFinalWarning {
            domain,
            job,
            window,
            capacity,
        },
    ] = warned.as_slice()
    else {
        panic!("the final warning: {warned:?}");
    };
    assert_eq!(*domain, MemoryDomain::Container);
    assert_eq!((job.task_id.as_str(), *window), ("large", WINDOW));
    assert_eq!(*capacity, MemorySize(4 * GIB));

    clock.advance(59);
    assert!(agent.guard.observe(clock.now(), normal).await.is_empty());
    assert!(!agent.container_job_killed("large"));

    clock.advance(1);
    let killed = agent.guard.observe(clock.now(), normal).await;
    assert!(
        matches!(
            killed.as_slice(),
            [HostLoadEvent::JobKilledForMemory {
                domain: MemoryDomain::Container,
                task_id,
                ..
            }] if task_id == "large"
        ),
        "{killed:?}"
    );
    assert!(agent.container_job_killed("large"));
    assert!(matches!(
        agent.status("large").await,
        JobStatus::Killed {
            cause: KillCause::MemoryRunaway {
                footprint_bytes,
                capacity_bytes,
                domain: MemoryDomain::Container,
            },
            ..
        } if footprint_bytes == 3 * GIB && capacity_bytes == 4 * GIB
    ));
    assert_eq!(agent.status("small").await, JobStatus::Running);
}

#[tokio::test]
async fn a_container_job_the_agent_stopped_in_time_is_not_killed() {
    let clock = Clock::new();
    let cgroup = Cgroup::limited(4 * GIB);
    let mut agent = Agent::containerized(64 * GIB, &cgroup);
    agent.container_job(&cgroup, "large", 4 * GIB).await;
    let normal = Some(MemoryPressure::Normal);

    assert_eq!(agent.guard.observe(clock.now(), normal).await.len(), 1);
    clock.advance(30);
    assert!(agent.registry.kill_by_task_id("large").await);
    cgroup.hold("large", 0);

    clock.advance(120);
    assert!(agent.guard.observe(clock.now(), normal).await.is_empty());
    assert!(matches!(
        agent.status("large").await,
        JobStatus::Killed {
            cause: KillCause::User,
            ..
        }
    ));
}

#[tokio::test]
async fn jobs_under_the_containers_limit_are_never_killed_whatever_the_pressure() {
    let clock = Clock::new();
    let cgroup = Cgroup::limited(8 * GIB);
    let mut agent = Agent::containerized(64 * GIB, &cgroup);
    agent.container_job(&cgroup, "a", 3 * GIB).await;
    agent.container_job(&cgroup, "b", 4 * GIB).await;
    for pressure in [THRASHING, THRASHING, STALLED] {
        clock.advance(600);
        cgroup.pressure(pressure);
        let events = agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Critical))
            .await;
        assert!(
            events.iter().all(|event| !matches!(
                event,
                HostLoadEvent::MemoryFinalWarning { .. } | HostLoadEvent::JobKilledForMemory { .. }
            )),
            "{events:?}"
        );
    }
    assert!(!agent.container_job_killed("a") && !agent.container_job_killed("b"));
}

#[tokio::test]
async fn a_container_without_a_memory_limit_adds_nothing_to_watch() {
    let clock = Clock::new();
    let cgroup = Cgroup::limited(4 * GIB);
    cgroup.0.lock().unwrap().limit = None;
    cgroup.pressure(THRASHING);
    let mut agent = Agent::containerized(64 * GIB, &cgroup);
    agent.container_job(&cgroup, "large", 100 * GIB).await;

    let events = agent
        .guard
        .observe(clock.now(), Some(MemoryPressure::Normal))
        .await;
    assert!(events.is_empty(), "{events:?}");
    clock.advance(600);
    assert!(
        agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Normal))
            .await
            .is_empty()
    );
    assert!(!agent.container_job_killed("large"));
}

#[tokio::test]
async fn the_machine_and_the_container_are_judged_apart() {
    let clock = Clock::new();
    let cgroup = Cgroup::limited(8 * GIB);
    let mut agent = Agent::containerized(10 * GIB, &cgroup);
    agent.container_job(&cgroup, "boxed", 2 * GIB).await;
    // A local job over the machine's memory is the machine's trouble alone.
    agent.job("local", 11 * GIB).await;

    let events = agent
        .guard
        .observe(clock.now(), Some(MemoryPressure::Normal))
        .await;
    let [
        HostLoadEvent::MemoryFinalWarning {
            domain: MemoryDomain::Machine,
            job,
            ..
        },
    ] = events.as_slice()
    else {
        panic!("only the machine warns: {events:?}");
    };
    assert_eq!(job.task_id, "local");
    assert!(!events.iter().any(is_container));
}

#[tokio::test]
async fn jobs_of_a_remote_environment_are_never_warned_about_or_killed() {
    let clock = Clock::new();
    let mut agent = Agent::new(10 * GIB, None);
    // The remote job's reading would be far past the machine's memory, if
    // anything ever read it.
    let remote = agent.remote_job("remote", 500 * GIB).await;
    agent.job("local", GIB).await;

    for pressure in [
        MemoryPressure::Warning,
        MemoryPressure::Critical,
        MemoryPressure::Critical,
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
        for event in &events {
            if let HostLoadEvent::MemoryYellow { jobs, .. }
            | HostLoadEvent::MemoryRed { jobs, .. } = event
            {
                assert_eq!(task_ids(jobs), ["local"], "the remote job is not listed");
            }
        }
    }
    assert_eq!(agent.status("remote").await, JobStatus::Running);
    // Its stand-in process is still alive.
    assert!(
        agent.sleepers.iter().any(|sleeper| sleeper.pid() == remote),
        "the remote job's process was left alone"
    );
    assert_eq!(agent.registry.kill_running().await, 2);
}

#[tokio::test]
async fn a_remote_job_in_a_container_terminal_is_still_never_judged() {
    let clock = Clock::new();
    let cgroup = Cgroup::limited(4 * GIB);
    let mut agent = Agent::containerized(10 * GIB, &cgroup);
    agent.remote_job("remote", 500 * GIB).await;
    // The container's cgroup knows nothing of the remote job either.
    cgroup.hold("remote", 500 * GIB);
    cgroup.pressure(THRASHING);

    let events = agent
        .guard
        .observe(clock.now(), Some(MemoryPressure::Critical))
        .await;
    assert!(
        events.iter().all(|event| !matches!(
            event,
            HostLoadEvent::MemoryFinalWarning { .. } | HostLoadEvent::JobKilledForMemory { .. }
        )),
        "{events:?}"
    );
    clock.advance(600);
    assert!(
        agent
            .guard
            .observe(clock.now(), Some(MemoryPressure::Critical))
            .await
            .iter()
            .all(|event| !matches!(event, HostLoadEvent::JobKilledForMemory { .. }))
    );
    assert_eq!(agent.status("remote").await, JobStatus::Running);
}
