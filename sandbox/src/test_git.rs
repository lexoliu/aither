//! Git for tests, isolated from the host configuration.
//!
//! Every test that runs `git` goes through this module, so none reads the
//! developer's global or system configuration (identity, signing, hooks).

#![allow(clippy::missing_panics_doc, clippy::must_use_candidate)]

use std::path::Path;
use std::process::Command;

/// A fixed moment for every commit, so that commits made in the same second
/// are still the same commits on every run.
const MOMENT: &str = "1700000000 +0000";

#[cfg(windows)]
const NULL_DEVICE: &str = "NUL";
#[cfg(not(windows))]
const NULL_DEVICE: &str = "/dev/null";

/// The environment that cuts git off from the system and global configuration.
pub const ISOLATION: [(&str, &str); 2] = [
    ("GIT_CONFIG_NOSYSTEM", "1"),
    ("GIT_CONFIG_GLOBAL", NULL_DEVICE),
];

/// A git command isolated from the host configuration, with a fixed identity,
/// fixed commit dates and signing off.
pub fn command() -> Command {
    let mut command = Command::new("git");
    command
        .envs(ISOLATION)
        .env("GIT_AUTHOR_DATE", MOMENT)
        .env("GIT_COMMITTER_DATE", MOMENT)
        .args(["-c", "user.name=test", "-c", "user.email=test@example.com"])
        .args(["-c", "commit.gpgSign=false"]);
    command
}

/// Runs git in `dir`, panics unless it succeeds, and returns its trimmed stdout.
pub fn git(dir: &Path, args: &[&str]) -> String {
    let output = command()
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git runs");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// Whether git, run in `dir`, exits successfully.
pub fn succeeds(dir: &Path, args: &[&str]) -> bool {
    command()
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git runs")
        .status
        .success()
}
