//! The patch an agent produced against abs-lang/abs `evaluator/evaluator.go`
//! (commit cb1b3b671d0e, MIT; see the origin comment in the fixture) that the
//! previous matcher rejected: every hunk matches the file contiguously.

use std::fs;

use tempfile::tempdir;

#[test]
fn applies_the_recorded_multi_hunk_patch() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let dir = tempdir().expect("tempdir");
    let target = dir.path().join("evaluator");
    fs::create_dir_all(&target).expect("mkdir");
    let source = fs::read_to_string(format!("{fixtures}/evaluator.go")).expect("fixture source");
    fs::write(target.join("evaluator.go"), &source).expect("write source");
    let patch =
        fs::read_to_string(format!("{fixtures}/abs-evaluator.patch")).expect("fixture patch");

    aither_patch::apply_patch_text(&patch, dir.path()).expect("recorded patch applies");

    let patched = fs::read_to_string(target.join("evaluator.go")).expect("read result");
    assert_ne!(patched, source);
    assert!(patched.contains("step := Eval(node.Step, env)"));
}
