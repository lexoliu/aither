//! The script behind a hook command: finding it, reading it and checking its
//! syntax.
//!
//! A hook command runs a script when its program is a script with a shebang,
//! or when its program is a shell and its first operand is a file. Such a
//! script is read once and run from that text (`shell -c <text> <path>`, so
//! `$0` is still the script's path), which is what lets a hook keep running a
//! version of the script after the file on disk was edited or broken.
//!
//! "Syntax-checked" means the shell's no-exec parse (`shell -n -c <text>`).
//! Only shell scripts are checked and kept this way: a hook that runs any
//! other interpreter or a compiled program has no script to parse, so it is
//! started as configured.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use super::runner::HookCommand;

/// The shells whose scripts are checked and kept.
const SHELLS: [&str; 6] = ["sh", "bash", "dash", "zsh", "ksh", "ash"];

/// How long a shell may take to parse a script.
const CHECK_DEADLINE: Duration = Duration::from_secs(10);

/// Why a hook's script cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum HookCheckError {
    /// The script, or the program that is one, cannot be read.
    #[error("the script {path} cannot be read: {source}")]
    Unreadable {
        /// The script.
        path: PathBuf,
        /// Why it cannot be read.
        #[source]
        source: std::io::Error,
    },
    /// The script is not text.
    #[error("the script {path} is not UTF-8 text")]
    NotText {
        /// The script.
        path: PathBuf,
    },
    /// The shell refused to parse the script.
    #[error("the script {path} has a syntax error: {diagnostics}")]
    Syntax {
        /// The script.
        path: PathBuf,
        /// What the shell reported.
        diagnostics: String,
    },
    /// The shell that parses the script could not be run.
    #[error("the shell `{shell}` that parses {path} cannot be run: {source}")]
    ShellUnavailable {
        /// The script.
        path: PathBuf,
        /// The shell.
        shell: String,
        /// Why it could not run.
        #[source]
        source: std::io::Error,
    },
    /// The shell did not finish parsing the script in time.
    #[error("the shell did not finish parsing {path} within {deadline:?}")]
    CheckTimeout {
        /// The script.
        path: PathBuf,
        /// The bound that passed.
        deadline: Duration,
    },
}

impl HookCheckError {
    /// The script the error is about.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::Unreadable { path, .. }
            | Self::NotText { path }
            | Self::Syntax { path, .. }
            | Self::ShellUnavailable { path, .. }
            | Self::CheckTimeout { path, .. } => path,
        }
    }
}

/// A script as it was read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptSnapshot {
    /// Where the script lives.
    pub path: PathBuf,
    /// Its text.
    pub content: String,
}

/// A hook command resolved to the script text it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Resolved {
    /// What to run: the configured command, or the shell running the script's
    /// text.
    pub command: HookCommand,
    /// The script, when the command runs one.
    pub script: Option<ScriptSnapshot>,
}

/// A script found in a hook command, not yet checked.
struct Located {
    /// The shell and the flags it runs the script with.
    shell: Vec<String>,
    /// The arguments after the script.
    tail: Vec<String>,
    /// The script as the command spells it; the script's `$0`.
    spelled: String,
    snapshot: ScriptSnapshot,
}

fn is_shell(program: &str) -> bool {
    Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| SHELLS.contains(&name))
}

async fn read_text(path: &Path) -> Result<String, HookCheckError> {
    let bytes = async_fs::read(path)
        .await
        .map_err(|source| HookCheckError::Unreadable {
            path: path.to_path_buf(),
            source,
        })?;
    String::from_utf8(bytes).map_err(|_| HookCheckError::NotText {
        path: path.to_path_buf(),
    })
}

/// The shell a shebang line names, with the flags it gives it.
fn shebang_shell(content: &str) -> Option<Vec<String>> {
    let line = content.strip_prefix("#!")?.lines().next()?;
    let mut words = line.split_whitespace();
    let mut interpreter = words.next()?;
    if Path::new(interpreter)
        .file_name()
        .is_some_and(|n| n == "env")
    {
        interpreter = words.find(|word| !word.starts_with('-'))?;
    }
    is_shell(interpreter).then(|| {
        std::iter::once(interpreter)
            .chain(words)
            .map(ToString::to_string)
            .collect()
    })
}

async fn locate(command: &[String]) -> Result<Option<Located>, HookCheckError> {
    let Some((program, args)) = command.split_first() else {
        return Ok(None);
    };
    if is_shell(program) {
        // The operand after the shell's own flags is the script; `-c` means
        // the script is inline and there is no file.
        let mut index = 0;
        while let Some(arg) = args.get(index) {
            if !arg.starts_with('-') {
                break;
            }
            if arg.starts_with("--") {
                index += 1;
            } else if arg.contains('c') {
                return Ok(None);
            } else if arg == "-o" || arg == "-O" {
                index += 2;
            } else {
                index += 1;
            }
        }
        let Some(spelled) = args.get(index) else {
            return Ok(None);
        };
        let path = PathBuf::from(spelled);
        let content = read_text(&path).await?;
        return Ok(Some(Located {
            shell: command[..=index].to_vec(),
            tail: args[index + 1..].to_vec(),
            spelled: spelled.clone(),
            snapshot: ScriptSnapshot { path, content },
        }));
    }
    if !program.contains(['/', std::path::MAIN_SEPARATOR]) {
        return Ok(None);
    }
    let path = PathBuf::from(program);
    // A program that cannot be read is no script; running it fails with the
    // reason, as for any hook that cannot start.
    let Ok(bytes) = async_fs::read(&path).await else {
        return Ok(None);
    };
    // A program that is not a text with a shebang is a compiled program.
    let Some(content) = String::from_utf8(bytes)
        .ok()
        .filter(|text| text.starts_with("#!"))
    else {
        return Ok(None);
    };
    let Some(shell) = shebang_shell(&content) else {
        return Ok(None);
    };
    Ok(Some(Located {
        shell,
        tail: args.to_vec(),
        spelled: program.clone(),
        snapshot: ScriptSnapshot { path, content },
    }))
}

/// Parses `content` with `shell` without running it.
async fn check(shell: &[String], snapshot: &ScriptSnapshot) -> Result<(), HookCheckError> {
    let (program, flags) = shell.split_first().expect("a located shell has a program");
    let mut command = async_process::Command::new(program);
    command
        .args(flags)
        .arg("-n")
        .arg("-c")
        .arg(&snapshot.content)
        .stdin(Stdio::null())
        .kill_on_drop(true);
    let parse = async {
        command
            .output()
            .await
            .map_err(|source| HookCheckError::ShellUnavailable {
                path: snapshot.path.clone(),
                shell: program.clone(),
                source,
            })
    };
    let expiry = async {
        async_io::Timer::after(CHECK_DEADLINE).await;
        Err(HookCheckError::CheckTimeout {
            path: snapshot.path.clone(),
            deadline: CHECK_DEADLINE,
        })
    };
    let output = futures_lite::future::or(parse, expiry).await?;
    if output.status.success() {
        Ok(())
    } else {
        Err(HookCheckError::Syntax {
            path: snapshot.path.clone(),
            diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
    }
}

/// Resolves `command` to what it runs. The script is syntax-checked unless it
/// is the one `known` already passed the check with.
///
/// # Errors
///
/// Returns why the script cannot be used: it cannot be read, is not text, or
/// has a syntax error.
pub(super) async fn resolve(
    command: &HookCommand,
    known: Option<&ScriptSnapshot>,
) -> Result<Resolved, HookCheckError> {
    let Some(located) = locate(&command.command).await? else {
        return Ok(Resolved {
            command: command.clone(),
            script: None,
        });
    };
    if known != Some(&located.snapshot) {
        check(&located.shell, &located.snapshot).await?;
    }
    let mut argv = located.shell;
    argv.push("-c".to_string());
    argv.push(located.snapshot.content.clone());
    argv.push(located.spelled);
    argv.extend(located.tail);
    Ok(Resolved {
        command: HookCommand {
            command: argv,
            deadline_ms: command.deadline_ms,
        },
        script: Some(located.snapshot),
    })
}
