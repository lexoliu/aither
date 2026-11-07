use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_sandbox::ForegroundWait;

use super::*;
use crate::clock::ManualClock;

const NORMAL_WAIT: Duration = Duration::from_secs(10);
const HIGH_LOAD_WAIT: Duration = Duration::from_secs(90);

/// A host whose readings the test sets.
#[derive(Clone)]
struct ScriptedHost {
    memory: Arc<Mutex<MemoryPressure>>,
    load: Arc<Mutex<f64>>,
}

impl ScriptedHost {
    fn set_memory(&self, level: MemoryPressure) {
        *self.memory.lock().unwrap() = level;
    }

    fn set_load(&self, load: f64) {
        *self.load.lock().unwrap() = load;
    }
}

impl HostReader for ScriptedHost {
    fn memory_pressure(&mut self) -> Result<MemoryPressure, ReadError> {
        Ok(*self.memory.lock().unwrap())
    }

    fn machine_load(&mut self) -> Result<MachineLoad, ReadError> {
        Ok(MachineLoad(*self.load.lock().unwrap()))
    }
}

/// A monitor over a scripted host and a clock the test advances.
struct Harness {
    host: ScriptedHost,
    clock: ManualClock,
    any_clock: AnyClock,
    wait: ForegroundWait,
    monitor: HostLoadMonitor<ScriptedHost>,
}

impl Harness {
    fn new() -> Self {
        let host = ScriptedHost {
            memory: Arc::new(Mutex::new(MemoryPressure::Normal)),
            load: Arc::new(Mutex::new(0.1)),
        };
        let clock = ManualClock::new();
        let wait = ForegroundWait::new(NORMAL_WAIT);
        let config = HostLoadConfig::new()
            .load_threshold(MachineLoad(1.0))
            .start_after(Duration::from_secs(30))
            .end_after(Duration::from_secs(60))
            .high_load_foreground_wait(HIGH_LOAD_WAIT);
        Self {
            monitor: HostLoadMonitor::new(host.clone(), config, wait.clone()),
            any_clock: AnyClock::new(clock.clone()),
            host,
            clock,
            wait,
        }
    }

    /// Reads the host now and every 10 seconds for `secs` seconds, and
    /// returns the events raised. A reading that follows another at the same
    /// instant is how a change of the host takes effect.
    fn pass(&mut self, secs: u64) -> Vec<HostLoadEvent> {
        let mut events = self.monitor.tick(&self.any_clock);
        for _ in 0..secs / 10 {
            self.clock.advance(Duration::from_secs(10));
            events.extend(self.monitor.tick(&self.any_clock));
        }
        events
    }
}

#[test]
fn sustained_machine_load_yields_one_start_and_one_end() {
    let mut host = Harness::new();
    host.host.set_load(1.8);

    let started = host.pass(30);
    assert_eq!(
        started,
        [HostLoadEvent::LoadHigh {
            load: MachineLoad(1.8),
            foreground_wait: HIGH_LOAD_WAIT
        }]
    );

    host.host.set_load(0.2);
    let ended = host.pass(60);
    assert_eq!(
        ended,
        [HostLoadEvent::LoadEnded {
            load: MachineLoad(0.2),
            foreground_wait: NORMAL_WAIT
        }]
    );
}

#[test]
fn memory_pressure_never_raises_a_load_event() {
    let mut host = Harness::new();
    host.host.set_memory(MemoryPressure::Critical);
    assert_eq!(host.pass(30), Vec::new());

    // Load alerts still come and go while the memory pressure persists.
    host.host.set_load(2.0);
    let events = host.pass(30);
    assert!(
        matches!(events.as_slice(), [HostLoadEvent::LoadHigh { .. }]),
        "{events:?}"
    );
    host.host.set_load(0.0);
    let events = host.pass(60);
    assert!(
        matches!(events.as_slice(), [HostLoadEvent::LoadEnded { .. }]),
        "{events:?}"
    );
}

#[test]
fn a_spike_shorter_than_the_start_window_yields_no_alert() {
    let mut host = Harness::new();
    host.host.set_memory(MemoryPressure::Critical);
    host.host.set_load(3.0);
    assert_eq!(host.pass(20), Vec::new());
    host.host.set_memory(MemoryPressure::Normal);
    host.host.set_load(0.1);
    assert_eq!(host.pass(120), Vec::new(), "the spike left nothing to end");
    assert_eq!(host.wait.get(), NORMAL_WAIT);

    // The spike's earlier seconds do not count towards a later one.
    host.host.set_load(3.0);
    assert_eq!(host.pass(20), Vec::new());
}

#[test]
fn a_dip_shorter_than_the_end_window_does_not_end_the_condition() {
    let mut host = Harness::new();
    host.host.set_load(2.0);
    assert_eq!(host.pass(30).len(), 1);

    host.host.set_load(0.1);
    assert_eq!(host.pass(40), Vec::new());
    host.host.set_load(2.0);
    assert_eq!(host.pass(120), Vec::new());
    assert_eq!(host.wait.get(), HIGH_LOAD_WAIT);
}

#[test]
fn nothing_is_raised_while_a_condition_persists() {
    let mut host = Harness::new();
    host.host.set_memory(MemoryPressure::Warning);
    host.host.set_load(2.0);
    assert_eq!(host.pass(30).len(), 1);
    assert_eq!(host.pass(3_600), Vec::new());
}

#[test]
fn the_foreground_window_is_longer_only_while_high_load_lasts() {
    let mut host = Harness::new();
    host.host.set_load(2.0);
    host.pass(20);
    assert_eq!(host.wait.get(), NORMAL_WAIT, "not before the load starts");
    host.pass(10);
    assert_eq!(host.wait.get(), HIGH_LOAD_WAIT);

    host.host.set_load(0.1);
    host.pass(50);
    assert_eq!(host.wait.get(), HIGH_LOAD_WAIT, "not before the load ends");
    host.pass(10);
    assert_eq!(host.wait.get(), NORMAL_WAIT);

    // Memory pressure never touches the window.
    host.host.set_memory(MemoryPressure::Critical);
    host.pass(60);
    assert_eq!(host.wait.get(), NORMAL_WAIT);
}

#[test]
fn a_failed_reading_leaves_the_condition_as_it_was() {
    struct Failing;
    impl HostReader for Failing {
        fn memory_pressure(&mut self) -> Result<MemoryPressure, ReadError> {
            Err(ReadError::Unsupported { what: "memory" })
        }
        fn machine_load(&mut self) -> Result<MachineLoad, ReadError> {
            Err(ReadError::Unsupported { what: "load" })
        }
    }
    let clock = ManualClock::new();
    let any_clock = AnyClock::new(clock.clone());
    let mut monitor = HostLoadMonitor::new(
        Failing,
        HostLoadConfig::new(),
        ForegroundWait::new(NORMAL_WAIT),
    );
    for _ in 0..20 {
        clock.advance(Duration::from_secs(10));
        assert_eq!(monitor.tick(&any_clock), Vec::new());
    }
}

#[tokio::test]
async fn the_task_delivers_events_once_and_stops_with_the_receiver() {
    let host = ScriptedHost {
        memory: Arc::new(Mutex::new(MemoryPressure::Normal)),
        load: Arc::new(Mutex::new(2.0)),
    };
    // Sustained from the first reading.
    let config = HostLoadConfig::new()
        .start_after(Duration::ZERO)
        .poll_interval(Duration::from_millis(5));
    let monitor = HostLoadMonitor::new(host, config, ForegroundWait::new(NORMAL_WAIT));
    let channels = spawn(
        &executor_core::tokio::TokioGlobal,
        monitor,
        None,
        AnyClock::default(),
    );

    let first = tokio::time::timeout(Duration::from_secs(10), channels.conditions.recv())
        .await
        .expect("the event arrives")
        .expect("the task is running");
    assert_eq!(
        first,
        HostLoadEvent::LoadHigh {
            load: MachineLoad(2.0),
            foreground_wait: Duration::from_secs(60)
        }
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        channels.conditions.is_empty(),
        "a persisting condition raises nothing"
    );
    assert!(channels.events.is_empty(), "a condition is no event");
}

/// A condition goes to the state the agent keeps; a final warning and a
/// kill are events.
#[tokio::test]
async fn conditions_and_events_go_through_their_own_channels() {
    let (senders, channels) = channels();
    let high = HostLoadEvent::LoadHigh {
        load: MachineLoad(2.0),
        foreground_wait: Duration::from_secs(60),
    };
    let killed = HostLoadEvent::JobKilledForMemory {
        domain: aither_sandbox::MemoryDomain::Machine,
        task_id: "amber-forest-thunder-pearl".to_string(),
        command: "cargo build".to_string(),
        footprint: aither_sandbox::MemorySize(8 << 30),
        capacity: aither_sandbox::MemorySize(32 << 30),
    };
    senders.send(high.clone()).await;
    senders.send(killed.clone()).await;
    assert_eq!(channels.conditions.try_recv(), Ok(high));
    assert_eq!(channels.events.try_recv(), Ok(killed));
    assert!(channels.conditions.is_empty() && channels.events.is_empty());
}

#[test]
fn the_alerts_name_the_condition_and_the_window() {
    let high = HostLoadEvent::LoadHigh {
        load: MachineLoad(1.84),
        foreground_wait: Duration::from_secs(90),
    }
    .alert_text();
    assert!(high.contains("load is high (184%"), "{high}");
    assert!(high.contains("up to 90s"), "{high}");

    let ended = HostLoadEvent::LoadEnded {
        load: MachineLoad(0.2),
        foreground_wait: Duration::from_secs(10),
    }
    .alert_text();
    assert!(
        ended.contains("has ended") && ended.contains("10s"),
        "{ended}"
    );
}

#[test]
fn a_host_without_sources_states_nothing_and_never_raises_load_events() {
    let clock = ManualClock::new();
    let any_clock = AnyClock::new(clock.clone());
    let mut monitor = HostLoadMonitor::new(
        UnwatchedHost,
        HostLoadConfig::new(),
        ForegroundWait::new(NORMAL_WAIT),
    );
    assert_eq!(monitor.read(), HostSample::new(None, None));
    for _ in 0..3 {
        clock.advance(Duration::from_secs(600));
        assert_eq!(monitor.tick(&any_clock), []);
    }
}
