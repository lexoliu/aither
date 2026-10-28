//! Restoring working material after a compaction.
//!
//! A handoff document names the files and skills the work depends on, but
//! naming them is not enough: the instance that reads the handoff has to
//! spend its first turns re-reading them one by one, and until it does it is
//! reasoning about code it cannot see. The restoration here front-runs that —
//! the material the handoff listed is gathered and placed in the window as a
//! single `<restored_context>` block, so the fresh instance starts with the
//! same context the compacted one had.
//!
//! Two rules keep this honest. Every path is validated against the session's
//! readable roots before it is read — the paths come from the model, so a
//! `..` segment, an absolute path outside a root, or a symlink escaping one
//! is reported rather than opened. And everything the model loses is said:
//! missing, outside-root, unknown-skill and over-budget entries are listed
//! in the block with the reason instead of disappearing silently.
//!
//! The budget is what keeps this from undoing the compaction it follows.
//! [`ContextPolicy::restore_budget`](super::ContextPolicy::restore_budget)
//! caps the block as a fraction of the window, charged on what the template
//! actually emits — headers, ranges and line numbers included, not only the
//! file bytes. Files are taken in the order the handoff listed them, the
//! first one that does not fit whole is truncated, and the rest are listed
//! as unavailable so the model knows to read them on demand.

use std::path::{Path, PathBuf};

use askama::Template;
use futures::StreamExt as _;

use super::compaction::{HandoffDocument, LineRange, SkillCatalogEntry};
use super::source::Block;

/// How many files are read from disk at once.
///
/// Restoration happens at a request boundary with the user waiting, so the
/// reads are overlapped; the bound keeps a handoff that lists a hundred paths
/// from opening a hundred file descriptors at once.
const CONCURRENT_READS: usize = 8;

/// One file range the handoff asked to be re-read.
#[derive(Debug, Clone)]
enum ReadRequest {
    /// Restore the whole file.
    Whole,
    /// Restore the inclusive 1-based line range, rendered with numbers.
    Range(LineRange),
}

/// A file re-read for the restored context block.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RestoredFile {
    /// Path as the handoff named it.
    path: String,
    /// Contents, possibly truncated to fit the budget.
    content: String,
    /// Whether `content` is a prefix of the file rather than all of it.
    truncated: bool,
    /// Why the handoff needed this file, when it said.
    why: Option<String>,
}

/// A skill whose instructions were restored.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RestoredSkill {
    /// The name the handoff named.
    name: String,
    /// The instructions restored into the block.
    instructions: String,
}

/// An entry the handoff named that could not be inlined, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
struct UnavailableEntry {
    /// Path or skill name as the handoff wrote it.
    name: String,
    /// Why it was not restored.
    reason: String,
}

#[derive(Template)]
#[template(path = "restored_context.md", escape = "none")]
struct RestoredContextTemplate<'a> {
    files: &'a [RestoredFile],
    skills: &'a [RestoredSkill],
    unavailable: &'a [UnavailableEntry],
}

/// Re-reads the file ranges and skills a handoff named, capped at `budget_chars`.
///
/// `roots` are the session's readable roots: a candidate resolves relative to
/// the first root when it is not absolute, and is restored only when its
/// canonical form stays inside one of them — `..` segments, absolute paths
/// outside a root and symlinks escaping one are listed with the reason
/// rather than opened. Skill names are resolved against `catalog`; an
/// unknown name is listed too.
///
/// Returns `None` only when the handoff names nothing at all; a block whose
/// files all failed still renders, listing each with its reason.
///
/// # Panics
///
/// Panics if the compiled-in restoration template fails to render, which
/// would mean the binary was built against a broken template.
pub async fn restore_context(
    handoff: &HandoffDocument,
    roots: &[PathBuf],
    skills: &[SkillCatalogEntry],
    budget_chars: usize,
) -> Option<Block> {
    if budget_chars == 0 {
        return None;
    }

    let candidates = restoration_candidates(handoff);
    if candidates.is_empty() && handoff.skills.is_empty() {
        return None;
    }

    // The fixed cost of emitting the block — prologue, section headers and
    // the list markup — is part of the budget, not free.
    let overhead = RestoredContextTemplate {
        files: &[],
        skills: &[],
        unavailable: &[],
    }
    .render()
    .expect("restored context template must render")
    .len();

    let contents = read_all(roots, &candidates).await;
    let (files, restored_skills, unavailable) =
        place_restorations(contents, &handoff.skills, skills, overhead, budget_chars);

    let rendered = RestoredContextTemplate {
        files: &files,
        skills: &restored_skills,
        unavailable: &unavailable,
    }
    .render()
    .expect("restored context template must render");
    Some(Block::new("restored_context", rendered))
}

/// Packs the read outcomes and the named skills into `budget_chars`.
///
/// Entries are taken in the order the handoff listed them — files first,
/// skills after — and charged against `used`, which already carries the
/// template's fixed overhead. A file that does not fit whole gets one
/// truncated attempt; whatever still does not fit, and whatever failed to
/// read or resolve, joins the unavailable list with its reason.
fn place_restorations(
    contents: Vec<(FileCandidate, Result<RestoredFile, String>)>,
    requested: &[String],
    catalog: &[SkillCatalogEntry],
    mut used: usize,
    budget_chars: usize,
) -> (Vec<RestoredFile>, Vec<RestoredSkill>, Vec<UnavailableEntry>) {
    let mut files: Vec<RestoredFile> = Vec::new();
    let mut restored_skills: Vec<RestoredSkill> = Vec::new();
    let mut unavailable: Vec<UnavailableEntry> = Vec::new();

    for (candidate, outcome) in contents {
        let entry = match outcome {
            Ok(content) => content,
            Err(reason) => {
                used += candidate.path.len() + reason.len() + 8;
                unavailable.push(UnavailableEntry {
                    name: candidate.path,
                    reason,
                });
                continue;
            }
        };
        let fragment = render_fragment(&entry.path, entry.why.as_deref(), &entry.content, false);
        if used + fragment.len() > budget_chars {
            // Try the entry truncated; if even a sliver does not fit, the
            // entry is reported rather than inlined. The marker and the
            // fragment's own markup are part of the allowance.
            let allowance = budget_chars
                .saturating_sub(used + fragment.len() - entry.content.len() + " (truncated)".len());
            if allowance == 0 {
                used += candidate.path.len() + 40;
                unavailable.push(UnavailableEntry {
                    name: candidate.path,
                    reason: "over the restoration budget".to_string(),
                });
                continue;
            }
            let truncated = truncate_on_char_boundary(&entry.content, allowance);
            let fragment = render_fragment(&entry.path, entry.why.as_deref(), &truncated, true);
            if used + fragment.len() > budget_chars {
                used += candidate.path.len() + 40;
                unavailable.push(UnavailableEntry {
                    name: candidate.path,
                    reason: "over the restoration budget".to_string(),
                });
                continue;
            }
            used += fragment.len();
            files.push(RestoredFile {
                path: entry.path,
                content: truncated,
                truncated: true,
                why: entry.why,
            });
            continue;
        }
        used += fragment.len();
        files.push(entry);
    }

    for name in requested {
        let name = name.trim();
        let Some(skill) = catalog.iter().find(|skill| skill.name == name) else {
            used += name.len() + 45;
            unavailable.push(UnavailableEntry {
                name: name.to_string(),
                reason: "not in the session's skill catalog".to_string(),
            });
            continue;
        };
        let fragment = render_fragment(
            &format!("Skill: {}", skill.name),
            None,
            &skill.instructions,
            false,
        );
        if used + fragment.len() > budget_chars {
            used += name.len() + 40;
            unavailable.push(UnavailableEntry {
                name: name.to_string(),
                reason: "over the restoration budget".to_string(),
            });
            continue;
        }
        used += fragment.len();
        restored_skills.push(RestoredSkill {
            name: skill.name.clone(),
            instructions: skill.instructions.clone(),
        });
    }

    (files, restored_skills, unavailable)
}

/// One file the handoff asked for: the name it used plus the range to read.
#[derive(Debug, Clone)]
struct FileCandidate {
    /// Path as the handoff named it.
    path: String,
    /// Which part of the file to restore.
    read: ReadRequest,
    /// Why the handoff needed it, when it said.
    why: Option<String>,
}

/// The paths worth re-reading, in handoff order and without duplicates.
///
/// `context_files` entries keep their ranges and reasons; `file_index`
/// entries are rendered as `path — description`, so only the path part is
/// usable, and they restore whole. Two entries differ only when both their
/// path and their range differ.
fn restoration_candidates(handoff: &HandoffDocument) -> Vec<FileCandidate> {
    let mut seen = std::collections::HashSet::new();
    let mut named = std::collections::HashSet::new();
    let mut candidates = Vec::new();
    for file in &handoff.context_files {
        let path = file.path.trim();
        if path.is_empty() || path.ends_with('/') {
            continue;
        }
        named.insert(path.to_string());
        let range = file.lines.filter(|lines| lines.start <= lines.end);
        if !seen.insert((path.to_string(), range)) {
            continue;
        }
        candidates.push(FileCandidate {
            path: path.to_string(),
            read: range.map_or(ReadRequest::Whole, ReadRequest::Range),
            why: (!file.why.trim().is_empty()).then(|| file.why.trim().to_string()),
        });
    }
    for entry in &handoff.file_index {
        let Some(path) = entry.split(" — ").next() else {
            continue;
        };
        let path = path.trim();
        // A path the handoff already ranged is restored through its
        // `context_files` entry; the mechanical index adds only new paths.
        if path.is_empty()
            || path.ends_with('/')
            || named.contains(path)
            || !seen.insert((path.to_string(), None))
        {
            continue;
        }
        candidates.push(FileCandidate {
            path: path.to_string(),
            read: ReadRequest::Whole,
            why: None,
        });
    }
    candidates
}

/// Reads every candidate concurrently, preserving the requested order.
///
/// A path is resolved, canonicalized and checked against the roots before a
/// byte is read; a candidate that fails validation is reported rather than
/// opened, so a model-written path can never reach outside the session's
/// readable area.
async fn read_all(
    roots: &[PathBuf],
    candidates: &[FileCandidate],
) -> Vec<(FileCandidate, Result<RestoredFile, String>)> {
    let canonical_roots: Vec<PathBuf> = roots
        .iter()
        .filter_map(|root| root.canonicalize().ok())
        .collect();
    futures::stream::iter(candidates.to_vec())
        .map(move |candidate| {
            let canonical_roots = canonical_roots.clone();
            async move {
                let outcome = read_candidate(&canonical_roots, roots.first(), &candidate).await;
                (candidate, outcome)
            }
        })
        .buffered(CONCURRENT_READS)
        .collect()
        .await
}

/// Resolves and reads one candidate, validating it against the roots.
async fn read_candidate(
    canonical_roots: &[PathBuf],
    first_root: Option<&PathBuf>,
    candidate: &FileCandidate,
) -> Result<RestoredFile, String> {
    let raw = Path::new(&candidate.path);
    let resolved = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        match first_root {
            Some(root) => root.join(raw),
            None => {
                return Err("relative path with no readable root".to_string());
            }
        }
    };
    let canonical = resolved
        .canonicalize()
        .map_err(|_| "does not exist or cannot be read".to_string())?;
    if !canonical.is_file() {
        return Err("not a file".to_string());
    }
    if !canonical_roots
        .iter()
        .any(|root| canonical.starts_with(root))
    {
        return Err("outside the session's readable roots".to_string());
    }
    let content = async_fs::read_to_string(&canonical)
        .await
        .map_err(|_| "does not exist or cannot be read".to_string())?;
    let content = match candidate.read {
        ReadRequest::Whole => content,
        ReadRequest::Range(range) => {
            let (numbered, in_range) = number_range(&content, range);
            if !in_range {
                return Err(format!(
                    "line range {}-{} is beyond the file's end",
                    range.start, range.end
                ));
            }
            numbered
        }
    };
    Ok(RestoredFile {
        path: candidate.path.clone(),
        content,
        truncated: false,
        why: candidate.why.clone(),
    })
}

/// Extracts the inclusive 1-based `range` of `content`, numbered.
///
/// The end is clamped to the file's length; the returned flag reports whether
/// the range reached anything at all, so a range entirely beyond the file's
/// end is reported rather than rendered as an empty block.
fn number_range(content: &str, range: LineRange) -> (String, bool) {
    let start = range.start as usize;
    let end = range.end as usize;
    let mut numbered = String::new();
    let mut any = false;
    for (index, line) in content.lines().enumerate() {
        let line_number = index + 1;
        if line_number < start {
            continue;
        }
        if line_number > end {
            break;
        }
        any = true;
        numbered.push_str(&line_number.to_string());
        numbered.push_str(" | ");
        numbered.push_str(line);
        numbered.push('\n');
    }
    (numbered, any)
}

/// Renders what a restored entry costs in the final block.
///
/// The budget is charged on emitted text — the heading, the `why` line and
/// the fences — so the allowance bookkeeping matches what the template
/// actually produces rather than only the file bytes.
fn render_fragment(path: &str, why: Option<&str>, content: &str, truncated: bool) -> String {
    let mut fragment = String::with_capacity(path.len() + content.len() + 64);
    fragment.push_str("\n\n### ");
    fragment.push_str(path);
    if truncated {
        fragment.push_str(" (truncated)");
    }
    fragment.push('\n');
    if let Some(why) = why {
        fragment.push('_');
        fragment.push_str(why);
        fragment.push_str("_\n");
    }
    fragment.push_str("\n```\n");
    fragment.push_str(content);
    if !content.ends_with('\n') {
        fragment.push('\n');
    }
    fragment.push_str("```");
    fragment
}

/// Truncates to at most `limit` bytes without splitting a character.
fn truncate_on_char_boundary(content: &str, limit: usize) -> String {
    if content.len() <= limit {
        return content.to_string();
    }
    let mut end = limit;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    content[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::super::compaction::ContextFile;
    use super::*;

    fn handoff(paths: &[&str]) -> HandoffDocument {
        HandoffDocument {
            current_task: "task".to_string(),
            context_files: paths
                .iter()
                .map(|path| ContextFile {
                    path: (*path).to_string(),
                    lines: None,
                    why: String::new(),
                })
                .collect(),
            ..HandoffDocument::default()
        }
    }

    fn roots(root: &tempfile::TempDir) -> Vec<PathBuf> {
        vec![root.path().to_path_buf()]
    }

    #[tokio::test]
    async fn restores_the_files_the_handoff_named() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("a.rs"), "fn a() {}")
            .await
            .expect("write a");
        async_fs::write(root.path().join("b.rs"), "fn b() {}")
            .await
            .expect("write b");

        let block = restore_context(&handoff(&["a.rs", "b.rs"]), &roots(&root), &[], 10_000)
            .await
            .expect("restoration produces a block");

        assert_eq!(block.tag(), "restored_context");
        assert!(block.content().contains("fn a() {}"));
        assert!(block.content().contains("fn b() {}"));
        assert!(!block.content().contains("not restored"));
    }

    #[tokio::test]
    async fn a_line_range_is_restored_with_line_numbers() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("main.rs"), "one\ntwo\nthree\nfour\n")
            .await
            .expect("write main");

        let document = HandoffDocument {
            current_task: "task".to_string(),
            context_files: vec![ContextFile {
                path: "main.rs".to_string(),
                lines: Some(LineRange { start: 2, end: 3 }),
                why: "the failing lines".to_string(),
            }],
            ..HandoffDocument::default()
        };
        let block = restore_context(&document, &roots(&root), &[], 10_000)
            .await
            .expect("restoration produces a block");

        assert!(block.content().contains("2 | two"));
        assert!(block.content().contains("3 | three"));
        assert!(!block.content().contains("four"), "the range is respected");
        assert!(block.content().contains("the failing lines"));
    }

    #[tokio::test]
    async fn a_named_skill_is_restored_and_an_unknown_one_is_listed() {
        let root = tempfile::tempdir().expect("tempdir");
        let catalog = vec![SkillCatalogEntry {
            name: "pdf".to_string(),
            description: "read PDFs".to_string(),
            instructions: "use the pdf tool first".to_string(),
        }];
        let document = HandoffDocument {
            current_task: "task".to_string(),
            skills: vec!["pdf".to_string(), "nonsense".to_string()],
            ..HandoffDocument::default()
        };
        let block = restore_context(&document, &roots(&root), &catalog, 10_000)
            .await
            .expect("restoration produces a block");

        assert!(block.content().contains("use the pdf tool first"));
        assert!(block.content().contains("nonsense"));
        assert!(
            block
                .content()
                .contains("not in the session's skill catalog")
        );
    }

    #[tokio::test]
    async fn paths_outside_the_roots_are_listed_not_read() {
        let root = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside dir");
        async_fs::write(outside.path().join("secret.txt"), "hidden")
            .await
            .expect("write outside");

        let document = HandoffDocument {
            current_task: "task".to_string(),
            context_files: vec![
                ContextFile {
                    path: "../outside.txt".to_string(),
                    lines: None,
                    why: String::new(),
                },
                ContextFile {
                    path: outside
                        .path()
                        .join("secret.txt")
                        .to_string_lossy()
                        .into_owned(),
                    lines: None,
                    why: String::new(),
                },
            ],
            ..HandoffDocument::default()
        };
        let block = restore_context(&document, &roots(&root), &[], 10_000)
            .await
            .expect("a block still lists what it could not restore");

        assert!(!block.content().contains("hidden"));
        assert!(block.content().contains("not restored"));
    }

    #[tokio::test]
    async fn a_symlink_escaping_the_root_is_listed_not_read() {
        let root = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside dir");
        async_fs::write(outside.path().join("real.txt"), "real contents")
            .await
            .expect("write outside");
        std::os::unix::fs::symlink(
            outside.path().join("real.txt"),
            root.path().join("link.txt"),
        )
        .expect("symlink");

        let block = restore_context(&handoff(&["link.txt"]), &roots(&root), &[], 10_000)
            .await
            .expect("a block still lists what it could not restore");

        assert!(!block.content().contains("real contents"));
        assert!(
            block
                .content()
                .contains("outside the session's readable roots")
        );
    }

    #[tokio::test]
    async fn the_budget_truncates_and_then_lists_the_rest() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("big.rs"), "x".repeat(500))
            .await
            .expect("write big");
        async_fs::write(root.path().join("next.rs"), "y".repeat(500))
            .await
            .expect("write next");

        let block = restore_context(&handoff(&["big.rs", "next.rs"]), &roots(&root), &[], 300)
            .await
            .expect("restoration produces a block");

        assert!(block.content().contains("(truncated)"));
        assert!(block.content().contains("next.rs"));
        assert!(block.content().contains("over the restoration budget"));
        assert!(
            !block.content().contains(&"y".repeat(10)),
            "the over-budget file must not be inlined"
        );
    }

    #[tokio::test]
    async fn missing_files_are_listed_with_the_reason() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("present.rs"), "fn present() {}")
            .await
            .expect("write present");

        let block = restore_context(
            &handoff(&["gone.rs", "present.rs"]),
            &roots(&root),
            &[],
            10_000,
        )
        .await
        .expect("restoration produces a block");

        assert!(block.content().contains("fn present() {}"));
        assert!(block.content().contains("gone.rs"));
        assert!(block.content().contains("does not exist or cannot be read"));
    }

    #[tokio::test]
    async fn nothing_named_yields_no_block() {
        let root = tempfile::tempdir().expect("tempdir");
        assert!(
            restore_context(&handoff(&[]), &roots(&root), &[], 10_000)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_zero_budget_restores_nothing() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("a.rs"), "fn a() {}")
            .await
            .expect("write a");
        assert!(
            restore_context(&handoff(&["a.rs"]), &roots(&root), &[], 0)
                .await
                .is_none()
        );
    }

    #[test]
    fn candidates_merge_named_ranges_and_the_file_index() {
        let document = HandoffDocument {
            current_task: "task".to_string(),
            context_files: vec![
                ContextFile {
                    path: "src/lib.rs".to_string(),
                    lines: Some(LineRange { start: 1, end: 5 }),
                    why: "the parser".to_string(),
                },
                ContextFile {
                    path: "src/lib.rs".to_string(),
                    lines: Some(LineRange { start: 1, end: 5 }),
                    why: "duplicate".to_string(),
                },
            ],
            file_index: vec![
                "src/lib.rs — edited".to_string(),
                "artifacts/report.md — written".to_string(),
                "outputs/".to_string(),
            ],
            ..HandoffDocument::default()
        };
        let candidates = restoration_candidates(&document);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].path, "src/lib.rs");
        assert_eq!(candidates[1].path, "artifacts/report.md");
    }

    #[test]
    fn number_range_clamps_the_end_to_the_file() {
        let (numbered, any) = number_range("a\nb\nc\n", LineRange { start: 2, end: 99 });
        assert!(any);
        assert!(numbered.contains("2 | b"));
        assert!(numbered.contains("3 | c"));
    }
}
