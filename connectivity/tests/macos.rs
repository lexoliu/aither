//! Smoke test against the real operating system.
#![cfg(target_os = "macos")]

use aither_connectivity::Connectivity;

/// The development and CI machines have a network path.
#[test]
fn start_reports_online() {
    let connectivity = Connectivity::start().expect("the path and power monitors register");
    assert!(connectivity.is_online());
}
