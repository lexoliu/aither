//! Subagent definitions.
//!
//! A subagent is a TOML file named after the subagent it defines:
//!
//! ```toml
//! name = "explore"
//! description = "Investigates a codebase and returns its conclusion."
//! tier = "fast"            # advanced, balanced or fast; absent: the fast model routes the brief
//! access = "read_only"     # or "worktrees": writes the worktrees its caller grants
//! prompt = '''
//! You investigate the workspace for the task in the brief.
//! '''
//! ```
//!
//! `subagent <name>` looks `<name>.toml` up in the global subagents
//! directory the host configures, then in `agents/` under the calling
//! terminal's working directory; a global definition shadows a local one of
//! the same name.

use std::path::{Path, PathBuf};

use async_fs as fs;
use futures_lite::stream::StreamExt;
use serde::Deserialize;

use crate::model_group::ModelTier;

/// The names `subagent` gives its own forms; no definition can take one.
pub const RESERVED_SUBAGENT_NAMES: [&str; 3] = ["fork", "resume", "external"];

/// The directory under a working directory that holds its subagents.
pub const LOCAL_SUBAGENTS_DIR: &str = "agents";

/// What a subagent may write besides its own working directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentAccess {
    /// Nothing: it reads the workspace and writes only its own space.
    ReadOnly,
    /// The worktrees its caller grants, within the caller's own authority.
    Worktrees,
}

/// A subagent definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentDefinition {
    /// The name `subagent <name>` runs it by, which its file is named after.
    pub name: String,
    /// When to use it, as the delegating model reads it.
    pub description: String,
    /// Its role prompt; the caller's brief is its first message.
    pub prompt: String,
    /// The model tier it runs on; `None` lets the fast model route each
    /// brief to a tier.
    pub tier: Option<ModelTier>,
    /// What it may write besides its own working directory.
    pub access: SubagentAccess,
}

/// The fields of a definition file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DefinitionFile {
    name: String,
    description: String,
    prompt: String,
    #[serde(default)]
    tier: Option<ModelTier>,
    access: SubagentAccess,
}

/// Why a definition is invalid.
#[derive(Debug, thiserror::Error)]
pub enum DefinitionError {
    /// The file is not a definition in TOML.
    #[error("{0}")]
    Toml(#[from] basic_toml::Error),
    /// A text field is blank.
    #[error("`{0}` must not be empty")]
    Empty(&'static str),
    /// The name is one `subagent` gives its own forms.
    #[error("the name '{0}' is reserved for `subagent {0}`")]
    Reserved(String),
    /// The name cannot be a file name.
    #[error("the name '{0}' may hold only ASCII letters, digits, `-` and `_`")]
    InvalidName(String),
}

/// Checks that `name` can name a subagent: a file name of ASCII letters,
/// digits, `-` and `_` that `subagent` does not reserve.
///
/// # Errors
///
/// Returns why `name` cannot name a subagent.
pub fn validate_subagent_name(name: &str) -> Result<(), DefinitionError> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(DefinitionError::InvalidName(name.to_string()));
    }
    if RESERVED_SUBAGENT_NAMES.contains(&name) {
        return Err(DefinitionError::Reserved(name.to_string()));
    }
    Ok(())
}

impl SubagentDefinition {
    /// Parses a definition from its TOML source.
    ///
    /// # Errors
    ///
    /// Returns an error when the source is not a definition, a field is
    /// blank, or the name cannot name a subagent.
    pub fn parse(source: &str) -> Result<Self, DefinitionError> {
        let file: DefinitionFile = basic_toml::from_str(source)?;
        validate_subagent_name(&file.name)?;
        for (field, value) in [("description", &file.description), ("prompt", &file.prompt)] {
            if value.trim().is_empty() {
                return Err(DefinitionError::Empty(field));
            }
        }
        Ok(Self {
            name: file.name,
            description: file.description.trim().to_string(),
            prompt: file.prompt.trim().to_string(),
            tier: file.tier,
            access: file.access,
        })
    }
}

/// A subagent shipped with aither, for the host to keep in its global
/// subagents directory.
#[derive(Debug, Clone, Copy)]
pub struct BundledSubagent {
    /// Its name, which its file is named after.
    pub name: &'static str,
    /// Its TOML definition.
    pub source: &'static str,
}

impl BundledSubagent {
    /// Its parsed definition.
    ///
    /// # Panics
    ///
    /// Panics if the bundled definition is invalid.
    #[must_use]
    pub fn definition(&self) -> SubagentDefinition {
        SubagentDefinition::parse(self.source)
            .unwrap_or_else(|error| panic!("bundled subagent '{}': {error}", self.name))
    }
}

/// The subagents shipped with aither.
pub const BUNDLED_SUBAGENTS: [BundledSubagent; 6] = [
    BundledSubagent {
        name: "explore",
        source: include_str!("../subagents/explore.toml"),
    },
    BundledSubagent {
        name: "research",
        source: include_str!("../subagents/research.toml"),
    },
    BundledSubagent {
        name: "audit",
        source: include_str!("../subagents/audit.toml"),
    },
    BundledSubagent {
        name: "review",
        source: include_str!("../subagents/review.toml"),
    },
    BundledSubagent {
        name: "plan",
        source: include_str!("../subagents/plan.toml"),
    },
    BundledSubagent {
        name: "general",
        source: include_str!("../subagents/general.toml"),
    },
];

/// Writes every bundled subagent into the global subagents directory `dir`,
/// creating it, and overwriting the bundled files so they match this build.
///
/// # Errors
///
/// Returns an error when the directory or a file cannot be written.
pub async fn install_bundled_subagents(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir).await?;
    futures::future::try_join_all(
        BUNDLED_SUBAGENTS
            .iter()
            .map(|bundled| fs::write(definition_path(dir, bundled.name), bundled.source)),
    )
    .await?;
    Ok(())
}

fn definition_path(dir: &Path, name: &str) -> PathBuf {
    let mut file_name = String::with_capacity(name.len() + 5);
    file_name.push_str(name);
    file_name.push_str(".toml");
    dir.join(file_name)
}

/// Why `subagent <name>` resolved no definition.
#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    /// The name cannot name a subagent.
    #[error(transparent)]
    Name(#[from] DefinitionError),
    /// Neither place holds the name.
    #[error(
        "no subagent '{name}': neither the global subagents directory {} nor {} holds {name}.toml",
        global.display(),
        local.display()
    )]
    NotFound {
        /// The name looked up.
        name: String,
        /// The global subagents directory.
        global: PathBuf,
        /// The working directory's `agents/`.
        local: PathBuf,
    },
    /// A definition file cannot be read.
    #[error("cannot read the subagent definition {}: {error}", path.display())]
    Read {
        /// The file.
        path: PathBuf,
        /// What failed.
        error: std::io::Error,
    },
    /// A definition file is invalid.
    #[error("the subagent definition {} is invalid: {error}", path.display())]
    Invalid {
        /// The file.
        path: PathBuf,
        /// What is wrong with it.
        error: DefinitionError,
    },
    /// A definition file names another subagent than its file name does.
    #[error("the subagent definition {} names '{declared}'; its file is named after the subagent it defines", path.display())]
    Misnamed {
        /// The file.
        path: PathBuf,
        /// The name the file declares.
        declared: String,
    },
}

/// The two places a subagent name resolves in.
#[derive(Debug, Clone, Copy)]
pub struct SubagentLookup<'a> {
    /// The global subagents directory.
    pub global: &'a Path,
    /// The working directory whose `agents/` is looked in second.
    pub working_dir: &'a Path,
}

/// A subagent a name resolves to, as the listing of resolvable names shows
/// it: its description, or why its definition is invalid.
pub struct Listed {
    /// The subagent's name.
    pub name: String,
    /// Its description, or why its definition is invalid.
    pub description: Result<String, String>,
}

impl SubagentLookup<'_> {
    fn local(&self) -> PathBuf {
        self.working_dir.join(LOCAL_SUBAGENTS_DIR)
    }

    /// The definition `name` resolves to: the global one, else the working
    /// directory's.
    pub async fn resolve(&self, name: &str) -> Result<SubagentDefinition, ResolveError> {
        validate_subagent_name(name)?;
        for dir in [self.global.to_path_buf(), self.local()] {
            if let Some(definition) = load(&definition_path(&dir, name), name).await? {
                return Ok(definition);
            }
        }
        Err(ResolveError::NotFound {
            name: name.to_string(),
            global: self.global.to_path_buf(),
            local: self.local(),
        })
    }

    /// Every name that resolves, by name, with what each resolves to.
    pub async fn resolvable(&self) -> Vec<Listed> {
        let local_dir = self.local();
        let (global, local) = futures::join!(names_in(self.global), names_in(&local_dir));
        let unshadowed: Vec<String> = local
            .into_iter()
            .filter(|name| !global.contains(name))
            .collect();
        let mut names = global;
        names.extend(unshadowed);
        names.sort();
        futures::future::join_all(names.into_iter().map(|name| async move {
            let description = self
                .resolve(&name)
                .await
                .map(|definition| definition.description)
                .map_err(|error| error.to_string());
            Listed { name, description }
        }))
        .await
    }
}

/// Loads the definition at `path`, which `name` was looked up as; `None`
/// when there is no such file.
async fn load(path: &Path, name: &str) -> Result<Option<SubagentDefinition>, ResolveError> {
    let source = match fs::read_to_string(path).await {
        Ok(source) => source,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(ResolveError::Read {
                path: path.to_path_buf(),
                error,
            });
        }
    };
    let definition = SubagentDefinition::parse(&source).map_err(|error| ResolveError::Invalid {
        path: path.to_path_buf(),
        error,
    })?;
    if definition.name != name {
        return Err(ResolveError::Misnamed {
            path: path.to_path_buf(),
            declared: definition.name,
        });
    }
    Ok(Some(definition))
}

/// The subagent names `dir` holds definitions for; none when it does not
/// exist or cannot be read.
async fn names_in(dir: &Path) -> Vec<String> {
    let Ok(mut entries) = fs::read_dir(dir).await else {
        return Vec::new();
    };
    let mut names = Vec::new();
    while let Some(Ok(entry)) = entries.next().await {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "toml")
            && let Some(name) = path.file_stem().and_then(|stem| stem.to_str())
            && validate_subagent_name(name).is_ok()
        {
            names.push(name.to_string());
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFINITION: &str = r#"
name = "reviewer"
description = "Reviews a change."
tier = "balanced"
access = "worktrees"
prompt = '''
You review.
'''
"#;

    #[test]
    fn a_definition_parses_with_its_tier_and_access() {
        let definition = SubagentDefinition::parse(DEFINITION).unwrap();
        assert_eq!(
            definition,
            SubagentDefinition {
                name: "reviewer".to_string(),
                description: "Reviews a change.".to_string(),
                prompt: "You review.".to_string(),
                tier: Some(ModelTier::Balanced),
                access: SubagentAccess::Worktrees,
            }
        );
        let routed =
            SubagentDefinition::parse(&DEFINITION.replace("tier = \"balanced\"\n", "")).unwrap();
        assert_eq!(routed.tier, None);
    }

    #[test]
    fn an_invalid_definition_is_refused_with_why() {
        let cases = [
            (DEFINITION.replace("access = \"worktrees\"\n", ""), "access"),
            (
                DEFINITION.replace("tier = \"balanced\"", "tier = \"huge\""),
                "huge",
            ),
            (format!("{DEFINITION}model = \"fast\"\n"), "model"),
            (
                DEFINITION.replace("Reviews a change.", "  "),
                "`description` must not be empty",
            ),
            (
                DEFINITION.replace("\"reviewer\"", "\"fork\""),
                "reserved for `subagent fork`",
            ),
            (
                DEFINITION.replace("\"reviewer\"", "\"../up\""),
                "may hold only ASCII letters",
            ),
        ];
        for (source, expected) in cases {
            let error = SubagentDefinition::parse(&source).expect_err(&source);
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn every_bundled_subagent_parses_with_its_stated_tier_and_access() {
        let stated = BUNDLED_SUBAGENTS.map(|bundled| {
            let definition = bundled.definition();
            assert_eq!(definition.name, bundled.name);
            (bundled.name, definition.tier, definition.access)
        });
        assert_eq!(
            stated,
            [
                ("explore", Some(ModelTier::Fast), SubagentAccess::ReadOnly),
                ("research", Some(ModelTier::Fast), SubagentAccess::ReadOnly),
                ("audit", Some(ModelTier::Advanced), SubagentAccess::ReadOnly),
                (
                    "review",
                    Some(ModelTier::Advanced),
                    SubagentAccess::ReadOnly
                ),
                ("plan", Some(ModelTier::Advanced), SubagentAccess::ReadOnly),
                ("general", None, SubagentAccess::Worktrees),
            ]
        );
    }

    #[test]
    fn explore_searches_cheaply_and_research_names_no_websearch_command() {
        let explore = BUNDLED_SUBAGENTS[0].definition().prompt;
        for tool in [
            "`search`",
            "`rg`",
            "`ast-grep`",
            "JSON",
            "transcripts",
            "databases",
        ] {
            assert!(explore.contains(tool), "explore's prompt names {tool}");
        }
        let research = BUNDLED_SUBAGENTS[1].definition().prompt;
        assert!(!research.contains("websearch"), "{research}");
    }

    fn write(dir: &Path, name: &str, description: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            definition_path(dir, name),
            DEFINITION
                .replace("\"reviewer\"", &format!("\"{name}\""))
                .replace("Reviews a change.", description),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn a_global_name_shadows_the_working_directorys_and_both_places_are_named_when_neither_has_it()
     {
        let global = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        let local = work.path().join(LOCAL_SUBAGENTS_DIR);
        write(global.path(), "shared", "the global one");
        write(&local, "shared", "the local one");
        write(&local, "only-local", "the local one");
        let lookup = SubagentLookup {
            global: global.path(),
            working_dir: work.path(),
        };

        assert_eq!(
            lookup.resolve("shared").await.unwrap().description,
            "the global one"
        );
        assert_eq!(
            lookup.resolve("only-local").await.unwrap().description,
            "the local one"
        );
        let missing = lookup.resolve("missing").await.unwrap_err().to_string();
        assert_eq!(
            missing,
            format!(
                "no subagent 'missing': neither the global subagents directory {} nor {} holds missing.toml",
                global.path().display(),
                local.display()
            )
        );
        assert!(matches!(
            lookup.resolve("../shared").await,
            Err(ResolveError::Name(DefinitionError::InvalidName(_)))
        ));

        let listed: Vec<(String, String)> = lookup
            .resolvable()
            .await
            .into_iter()
            .map(|listed| (listed.name, listed.description.unwrap()))
            .collect();
        assert_eq!(
            listed,
            [
                ("only-local".to_string(), "the local one".to_string()),
                ("shared".to_string(), "the global one".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn a_definition_named_unlike_its_file_is_refused() {
        let global = tempfile::tempdir().unwrap();
        write(global.path(), "inner", "described");
        std::fs::rename(
            definition_path(global.path(), "inner"),
            definition_path(global.path(), "outer"),
        )
        .unwrap();
        let work = tempfile::tempdir().unwrap();
        let lookup = SubagentLookup {
            global: global.path(),
            working_dir: work.path(),
        };
        let error = lookup.resolve("outer").await.unwrap_err();
        assert!(
            matches!(error, ResolveError::Misnamed { ref declared, .. } if declared == "inner")
        );
    }

    #[tokio::test]
    async fn installing_writes_every_bundled_subagent_over_a_stale_copy() {
        let dir = tempfile::tempdir().unwrap();
        let global = dir.path().join("subagents");
        std::fs::create_dir_all(&global).unwrap();
        std::fs::write(definition_path(&global, "explore"), "stale").unwrap();
        install_bundled_subagents(&global).await.unwrap();
        let work = tempfile::tempdir().unwrap();
        let lookup = SubagentLookup {
            global: &global,
            working_dir: work.path(),
        };
        for bundled in BUNDLED_SUBAGENTS {
            assert_eq!(
                lookup.resolve(bundled.name).await.unwrap(),
                bundled.definition()
            );
        }
    }
}
