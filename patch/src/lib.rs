//! Engine behind the `apply_patch` sandbox executable.
//!
//! The crate applies the compact contextual patch format (`*** Begin Patch` …
//! `*** End Patch`) used by terminal-first agents. Parsing and hunk matching
//! are implemented here; file-level semantics — literal path resolution,
//! unconditional deletes, moves, mode preservation, and all-or-nothing
//! validation — live here so the executable stays a pure filesystem utility.
//!
//! All work happens inside the calling process, so the binary inherits
//! whatever filesystem sandbox the launching shell enforces; there is no IPC
//! and no privileged helper.

mod error;
mod matcher;
mod parse;

use std::{
    collections::{HashMap, HashSet},
    fs, io,
    path::Path,
};

use matcher::{Whitespace, apply_hunks};
use parse::{Action, parse};

pub use error::PatchError;

/// One file-level change a successfully applied patch made on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOutcome {
    /// A new file was created at this path.
    Added(String),
    /// An existing file's contents changed in place.
    Modified(String),
    /// A file was removed.
    Deleted(String),
    /// A file moved to a new path; its contents may also have changed.
    Renamed {
        /// The original path.
        from: String,
        /// The destination path.
        to: String,
    },
}

impl FileOutcome {
    /// The single-letter summary line printed for this change.
    ///
    /// Modeled on `git status --short` letters: `A`, `M`, `D`, `R`.
    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::Added(path) => format!("A {path}"),
            Self::Modified(path) => format!("M {path}"),
            Self::Deleted(path) => format!("D {path}"),
            Self::Renamed { from, to } => format!("R {from} -> {to}"),
        }
    }
}

/// Applies `patch` to the filesystem under `base`, returning per-file outcomes.
///
/// Every referenced file is read and every hunk is matched in memory before
/// anything is written, so a malformed patch or a mismatched hunk leaves the
/// filesystem untouched — including in multi-file patches. This is a
/// validation guarantee, not a filesystem transaction: an I/O failure during
/// the write phase itself can still leave earlier writes in place.
///
/// Paths resolve literally against `base` — the patch path is always the file
/// it names, never a suffix-matched sibling. File modes are preserved on
/// update and move; updates through symlinks write the link's target while
/// deletes remove the link itself.
///
/// # Errors
///
/// Returns [`PatchError`] when the patch is empty or unparsable, a referenced
/// file is missing, unreadable, non-UTF-8, or not a regular file, an Add or
/// Move target already exists, a hunk fails to match or is ambiguous, or a
/// filesystem write fails.
pub fn apply_patch_text(patch: &str, base: &Path) -> Result<Vec<FileOutcome>, PatchError> {
    if patch.trim().is_empty() {
        return Err(PatchError::EmptyInput);
    }
    let actions = parse(patch)?;

    // Read every file the patch touches up front. Keys are the literal patch
    // paths, so a hunk can only ever land on the file it names.
    let mut files: HashMap<String, String> = HashMap::new();
    let mut modes: HashMap<String, fs::Permissions> = HashMap::new();
    // Paths an earlier action produces in memory: an Update or Delete that
    // follows them must not read the disk, and their existence check must
    // defer to the in-memory result.
    let mut produced: HashSet<String> = HashSet::new();
    for action in &actions {
        match action {
            Action::Add { path, .. } => {
                if !produced.contains(path) {
                    ensure_absent(base, path)?;
                }
                produced.insert(path.clone());
            }
            Action::Update { path, new_path, .. } => {
                if !produced.contains(path) {
                    load_file(base, path, &mut files, &mut modes)?;
                }
                if let Some(new_path) = new_path {
                    ensure_absent(base, new_path)?;
                    produced.insert(new_path.clone());
                }
            }
            Action::Delete { path, .. } => {
                if !produced.contains(path) {
                    load_file(base, path, &mut files, &mut modes)?;
                }
                produced.remove(path);
            }
        }
    }

    // Validate every hunk in memory first; nothing is written unless the whole
    // patch applies cleanly.
    let patched = apply_actions(&actions, &files)?;
    let outcomes = file_outcomes(&actions);
    commit(base, &patched, &modes, &outcomes)?;
    Ok(outcomes)
}

/// Fails when `path` already exists on disk (or its metadata is unreadable).
fn ensure_absent(base: &Path, path: &str) -> Result<(), PatchError> {
    match fs::symlink_metadata(base.join(path)) {
        Ok(_) => Err(PatchError::AlreadyExists(path.to_owned())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(PatchError::Io {
            path: path.to_owned(),
            source,
        }),
    }
}

/// Reads `path` into `files` and remembers its mode for later restores.
///
/// `fs::metadata`/`fs::read` follow symlinks, so patching a symlink updates
/// its target — the same behavior `sed -i` style editors exhibit when they
/// write in place.
fn load_file(
    base: &Path,
    path: &str,
    files: &mut HashMap<String, String>,
    modes: &mut HashMap<String, fs::Permissions>,
) -> Result<(), PatchError> {
    if files.contains_key(path) {
        return Ok(());
    }
    let full = base.join(path);
    let meta = fs::metadata(&full).map_err(|source| match source.kind() {
        io::ErrorKind::NotFound => PatchError::FileNotFound(path.to_owned()),
        _ => PatchError::Io {
            path: path.to_owned(),
            source,
        },
    })?;
    if !meta.is_file() {
        return Err(PatchError::NotAFile(path.to_owned()));
    }
    let bytes = fs::read(&full).map_err(|source| PatchError::Io {
        path: path.to_owned(),
        source,
    })?;
    let content = String::from_utf8(bytes).map_err(|_| PatchError::NotUtf8(path.to_owned()))?;
    files.insert(path.to_owned(), content);
    modes.insert(path.to_owned(), meta.permissions());
    Ok(())
}

/// Runs every action against an in-memory copy of the files.
///
/// Add and Delete semantics live here: a bare `*** Delete File:` directive
/// removes the file unconditionally, and every key is the literal patch path.
/// Update hunks are matched strictly first and leniently on failure; a hunk
/// matching more than one place is an error rather than a guess.
fn apply_actions(
    actions: &[Action],
    files: &HashMap<String, String>,
) -> Result<HashMap<String, String>, PatchError> {
    let mut out = files.clone();
    for action in actions {
        match action {
            Action::Add { path, lines } => {
                if out.contains_key(path) {
                    return Err(PatchError::AlreadyExists(path.clone()));
                }
                let mut content = String::new();
                for line in lines {
                    content.push_str(line);
                    content.push('\n');
                }
                out.insert(path.clone(), content);
            }
            Action::Delete { path, expected } => {
                let existing = out
                    .get(path)
                    .ok_or_else(|| PatchError::FileNotFound(path.clone()))?;
                // Content lines under a Delete directive are a safety check:
                // they must reproduce the file exactly when present.
                if !expected.is_empty() && !expected.iter().map(String::as_str).eq(existing.lines())
                {
                    return Err(PatchError::DeleteMismatch(path.clone()));
                }
                out.remove(path);
            }
            Action::Update {
                path,
                new_path,
                hunks,
            } => {
                let original = out
                    .get(path)
                    .ok_or_else(|| PatchError::FileNotFound(path.clone()))?
                    .clone();
                let lines: Vec<String> = original.lines().map(String::from).collect();
                let applied = match apply_hunks(path, &lines, hunks, Whitespace::Strict) {
                    Err(PatchError::Conflict { .. } | PatchError::Ambiguous { .. }) => {
                        apply_hunks(path, &lines, hunks, Whitespace::Lenient)?
                    }
                    other => other?,
                };
                let updated = rejoin(&original, &applied);
                if let Some(new_path) = new_path {
                    if out.contains_key(new_path) {
                        return Err(PatchError::AlreadyExists(new_path.clone()));
                    }
                    out.remove(path);
                    out.insert(new_path.clone(), updated);
                } else {
                    out.insert(path.clone(), updated);
                }
            }
        }
    }
    Ok(out)
}

/// Maps each action to the change it reports on success.
fn file_outcomes(actions: &[Action]) -> Vec<FileOutcome> {
    actions
        .iter()
        .map(|action| match action {
            Action::Add { path, .. } => FileOutcome::Added(path.clone()),
            Action::Delete { path, .. } => FileOutcome::Deleted(path.clone()),
            Action::Update { path, new_path, .. } => new_path.as_ref().map_or_else(
                || FileOutcome::Modified(path.clone()),
                |to| FileOutcome::Renamed {
                    from: path.clone(),
                    to: to.clone(),
                },
            ),
        })
        .collect()
}

/// Writes the validated results to disk.
///
/// Reached only after every hunk applied in memory. Update writes truncate in
/// place, which keeps the existing mode; moves restore the original file's
/// mode onto the new path; deletes remove the named path itself (a symlink
/// drops the link, not its target).
fn commit(
    base: &Path,
    patched: &HashMap<String, String>,
    modes: &HashMap<String, fs::Permissions>,
    outcomes: &[FileOutcome],
) -> Result<(), PatchError> {
    for outcome in outcomes {
        match outcome {
            FileOutcome::Added(path) | FileOutcome::Modified(path) => {
                let content = patched.get(path).map_or("", String::as_str);
                write_file(base, path, content)?;
            }
            FileOutcome::Deleted(path) => {
                fs::remove_file(base.join(path)).map_err(|source| PatchError::Io {
                    path: path.clone(),
                    source,
                })?;
            }
            FileOutcome::Renamed { from, to } => {
                let content = patched.get(to).map_or("", String::as_str);
                write_file(base, to, content)?;
                if let Some(mode) = modes.get(from) {
                    fs::set_permissions(base.join(to), mode.clone()).map_err(|source| {
                        PatchError::Io {
                            path: to.clone(),
                            source,
                        }
                    })?;
                }
                fs::remove_file(base.join(from)).map_err(|source| PatchError::Io {
                    path: from.clone(),
                    source,
                })?;
            }
        }
    }
    Ok(())
}

/// Writes `content` to `path`, creating parent directories as needed.
fn write_file(base: &Path, path: &str, content: &str) -> Result<(), PatchError> {
    let full = base.join(path);
    if let Some(parent) = full.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| PatchError::Io {
            path: path.to_owned(),
            source,
        })?;
    }
    fs::write(&full, content).map_err(|source| PatchError::Io {
        path: path.to_owned(),
        source,
    })
}

/// Re-joins patched lines with the file's dominant line ending and restores
/// its trailing newline, so a one-line patch does not rewrite every ending in
/// a CRLF file or strip the final newline.
fn rejoin(original: &str, applied: &[String]) -> String {
    let crlf = original.matches("\r\n").count();
    let lf = original.matches('\n').count() - crlf;
    let eol = if crlf > lf { "\r\n" } else { "\n" };
    let mut updated = applied.join(eol);
    if original.ends_with('\n') && !updated.is_empty() {
        updated.push_str(eol);
    }
    updated
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::tempdir;

    use super::{FileOutcome, PatchError, apply_patch_text};

    fn write(base: &Path, rel: &str, content: &str) {
        let path = base.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("mkdir");
        }
        fs::write(path, content).expect("write fixture");
    }

    fn read(base: &Path, rel: &str) -> String {
        fs::read_to_string(base.join(rel)).expect("read result")
    }

    #[test]
    fn adds_a_file_in_nested_directories() {
        let dir = tempdir().expect("tempdir");
        let patch = "*** Begin Patch\n*** Add File: sub/dir/new.txt\n+alpha\n+beta\n*** End Patch";
        let outcomes = apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(outcomes, [FileOutcome::Added("sub/dir/new.txt".to_owned())]);
        assert_eq!(read(dir.path(), "sub/dir/new.txt"), "alpha\nbeta\n");
    }

    #[test]
    fn updates_multiple_hunks_across_multiple_files() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "a.txt", "one\ntwo\nthree\nfour\n");
        write(dir.path(), "b.txt", "x\ny\n");
        let patch = "*** Begin Patch\n\
            *** Update File: a.txt\n@@\n two\n-three\n+THREE\n@@\n four\n+five\n\
            *** Update File: b.txt\n@@\n-x\n+w\n\
            *** End Patch";
        let outcomes = apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(outcomes.len(), 2);
        assert_eq!(read(dir.path(), "a.txt"), "one\ntwo\nTHREE\nfour\nfive\n");
        assert_eq!(read(dir.path(), "b.txt"), "w\ny\n");
    }

    #[test]
    fn deletes_a_file_without_content_lines() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "gone.txt", "anything\nat all\n");
        let patch = "*** Begin Patch\n*** Delete File: gone.txt\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert!(!dir.path().join("gone.txt").exists());
    }

    #[test]
    fn delete_with_content_verifies_the_whole_file() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "exact.txt", "a\nb\n");
        let good = "*** Begin Patch\n*** Delete File: exact.txt\n-a\n-b\n*** End Patch";
        apply_patch_text(good, dir.path()).expect("delete applies");
        assert!(!dir.path().join("exact.txt").exists());

        write(dir.path(), "mismatch.txt", "a\nchanged\n");
        let bad = "*** Begin Patch\n*** Delete File: mismatch.txt\n-a\n-b\n*** End Patch";
        assert!(apply_patch_text(bad, dir.path()).is_err());
        assert_eq!(read(dir.path(), "mismatch.txt"), "a\nchanged\n");
    }

    #[test]
    fn moves_a_file_and_applies_content_changes() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "old/place.rs", "keep\nchange me\n");
        let patch = "*** Begin Patch\n*** Update File: old/place.rs\n*** Move to: new/place.rs\n@@\n-change me\n+changed\n*** End Patch";
        let outcomes = apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(
            outcomes,
            [FileOutcome::Renamed {
                from: "old/place.rs".to_owned(),
                to: "new/place.rs".to_owned()
            }]
        );
        assert!(!dir.path().join("old/place.rs").exists());
        assert_eq!(read(dir.path(), "new/place.rs"), "keep\nchanged\n");
    }

    #[test]
    fn anchors_a_hunk_at_end_of_file() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "tail.txt", "first\nlast\n");
        let patch = "*** Begin Patch\n*** Update File: tail.txt\n@@\n last\n+appended\n*** End of File\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "tail.txt"), "first\nlast\nappended\n");
    }

    #[test]
    fn rejects_malformed_patches() {
        let dir = tempdir().expect("tempdir");
        for patch in [
            "",
            "   \n  ",
            "no markers at all",
            "*** Begin Patch\n*** Update File: a.txt\n", // truncated: no End
            "*** Update File: a.txt\n-x\n+y\n*** End Patch", // missing Begin
            "*** Begin Patch\n*** Frobnicate File: a.txt\n*** End Patch",
            "*** Begin Patch\n*** End Patch", // no operations
        ] {
            assert!(
                apply_patch_text(patch, dir.path()).is_err(),
                "expected error for {patch:?}"
            );
        }
    }

    #[test]
    fn rejects_a_context_mismatch_without_touching_the_file() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "a.txt", "real content\n");
        let patch =
            "*** Begin Patch\n*** Update File: a.txt\n@@\n absent context\n+new\n*** End Patch";
        assert!(apply_patch_text(patch, dir.path()).is_err());
        assert_eq!(read(dir.path(), "a.txt"), "real content\n");
    }

    #[test]
    fn rejects_ambiguous_context_instead_of_guessing() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "dup.txt", "same\nsame\nsame\n");
        // A single-line replacement matches three positions: ambiguous.
        let patch = "*** Begin Patch\n*** Update File: dup.txt\n@@\n-same\n+unique\n*** End Patch";
        let error = apply_patch_text(patch, dir.path()).expect_err("must not guess");
        assert!(
            error.to_string().contains("mbiguous") || error.to_string().contains("onflict"),
            "unexpected error: {error}"
        );
        assert_eq!(read(dir.path(), "dup.txt"), "same\nsame\nsame\n");
    }

    #[test]
    fn leading_context_makes_a_unique_match() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "dup.txt", "same\nkeep one\nsame\nsame\n");
        // Context before the change anchors the deletion at the one position
        // where "same" follows "keep one".
        let patch =
            "*** Begin Patch\n*** Update File: dup.txt\n@@\n keep one\n-same\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "dup.txt"), "same\nkeep one\nsame\n");
    }

    #[test]
    fn preserves_crlf_endings_and_a_missing_final_newline() {
        let dir = tempdir().expect("tempdir");
        fs::write(dir.path().join("win.txt"), "a\r\nb\r\nc").expect("write");
        let patch = "*** Begin Patch\n*** Update File: win.txt\n@@\n-b\n+B\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "win.txt"), "a\r\nB\r\nc");
    }

    #[test]
    fn handles_unicode_content_and_patch_lines() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "uni.txt", "héllo\nwörld\n");
        let patch = "*** Begin Patch\n*** Update File: uni.txt\n@@\n-héllo\n+你好\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "uni.txt"), "你好\nwörld\n");
    }

    #[test]
    fn applies_a_multi_file_patch_or_nothing_at_all() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "ok.txt", "before\n");
        write(dir.path(), "bad.txt", "stays\n");
        let patch = "*** Begin Patch\n\
            *** Update File: ok.txt\n@@\n-before\n+after\n\
            *** Update File: bad.txt\n@@\n-no such line\n+never\n\
            *** End Patch";
        assert!(apply_patch_text(patch, dir.path()).is_err());
        // The good hunk must not have leaked through: validation is total.
        assert_eq!(read(dir.path(), "ok.txt"), "before\n");
        assert_eq!(read(dir.path(), "bad.txt"), "stays\n");
    }

    #[test]
    fn rejects_add_and_move_targets_that_exist() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "exists.txt", "here\n");
        write(dir.path(), "src.txt", "body\n");
        let add = "*** Begin Patch\n*** Add File: exists.txt\n+x\n*** End Patch";
        assert!(matches!(
            apply_patch_text(add, dir.path()),
            Err(PatchError::AlreadyExists(_))
        ));
        let mv = "*** Begin Patch\n*** Update File: src.txt\n*** Move to: exists.txt\n@@\n-body\n+new\n*** End Patch";
        assert!(matches!(
            apply_patch_text(mv, dir.path()),
            Err(PatchError::AlreadyExists(_))
        ));
        assert_eq!(read(dir.path(), "src.txt"), "body\n");
    }

    #[test]
    fn reports_missing_and_non_file_targets() {
        let dir = tempdir().expect("tempdir");
        let missing = "*** Begin Patch\n*** Update File: absent.txt\n@@\n-a\n+b\n*** End Patch";
        assert!(matches!(
            apply_patch_text(missing, dir.path()),
            Err(PatchError::FileNotFound(_))
        ));
        fs::create_dir_all(dir.path().join("a_dir")).expect("mkdir");
        let on_dir = "*** Begin Patch\n*** Update File: a_dir\n@@\n-a\n+b\n*** End Patch";
        assert!(matches!(
            apply_patch_text(on_dir, dir.path()),
            Err(PatchError::NotAFile(_))
        ));
        let del_missing = "*** Begin Patch\n*** Delete File: absent.txt\n*** End Patch";
        assert!(matches!(
            apply_patch_text(del_missing, dir.path()),
            Err(PatchError::FileNotFound(_))
        ));
    }

    #[test]
    fn rejects_non_utf8_targets() {
        let dir = tempdir().expect("tempdir");
        fs::write(dir.path().join("bin.dat"), [0xff, 0xfe, 0x00]).expect("write");
        let patch = "*** Begin Patch\n*** Update File: bin.dat\n@@\n-a\n+b\n*** End Patch";
        assert!(matches!(
            apply_patch_text(patch, dir.path()),
            Err(PatchError::NotUtf8(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn preserves_file_modes_on_update_and_move() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "mode.txt", "a\nb\n");
        fs::set_permissions(
            dir.path().join("mode.txt"),
            fs::Permissions::from_mode(0o640),
        )
        .expect("chmod");
        let patch = "*** Begin Patch\n*** Update File: mode.txt\n@@\n-a\n+A\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(
            fs::metadata(dir.path().join("mode.txt"))
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o640
        );

        let mv = "*** Begin Patch\n*** Update File: mode.txt\n*** Move to: moved.txt\n@@\n-A\n+AA\n*** End Patch";
        apply_patch_text(mv, dir.path()).expect("patch applies");
        assert_eq!(
            fs::metadata(dir.path().join("moved.txt"))
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o640
        );
    }

    #[cfg(unix)]
    #[test]
    fn updates_through_a_symlink_and_deletes_the_link_itself() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "real.txt", "a\nb\n");
        std::os::unix::fs::symlink("real.txt", dir.path().join("link.txt")).expect("symlink");
        let patch = "*** Begin Patch\n*** Update File: link.txt\n@@\n-a\n+A\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "real.txt"), "A\nb\n");
        assert!(
            dir.path()
                .join("link.txt")
                .symlink_metadata()
                .expect("meta")
                .file_type()
                .is_symlink()
        );

        let del = "*** Begin Patch\n*** Delete File: link.txt\n*** End Patch";
        apply_patch_text(del, dir.path()).expect("patch applies");
        assert!(!dir.path().join("link.txt").exists());
        assert_eq!(read(dir.path(), "real.txt"), "A\nb\n");
    }

    #[test]
    fn supports_add_then_update_and_repeat_updates_in_one_patch() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "twice.txt", "1\n2\n");
        let patch = "*** Begin Patch\n\
            *** Update File: twice.txt\n@@\n-1\n+one\n\
            *** Update File: twice.txt\n@@\n-2\n+two\n\
            *** Add File: made.txt\n+draft\n\
            *** Update File: made.txt\n@@\n-draft\n+final\n\
            *** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "twice.txt"), "one\ntwo\n");
        assert_eq!(read(dir.path(), "made.txt"), "final\n");
    }

    #[test]
    fn accepts_a_patch_wrapped_in_a_markdown_fence() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "f.txt", "a\n");
        let patch =
            "```diff\n*** Begin Patch\n*** Update File: f.txt\n@@\n-a\n+b\n*** End Patch\n```";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "f.txt"), "b\n");
    }

    #[test]
    fn matches_leniently_when_whitespace_differs() {
        let dir = tempdir().expect("tempdir");
        write(dir.path(), "ws.txt", "line   \nnext\n");
        let patch = "*** Begin Patch\n*** Update File: ws.txt\n@@\n-line\n+LINE\n*** End Patch";
        apply_patch_text(patch, dir.path()).expect("patch applies");
        assert_eq!(read(dir.path(), "ws.txt"), "LINE\nnext\n");
    }
}
