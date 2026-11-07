//! Communication and working languages.
//!
//! A host sets the language the agent talks to the user in and the language it
//! writes in for others to read. This module holds the settings
//! ([`Languages`]), the detection of a text's language on its prose and the
//! enforcement ([`LanguageGuard`]) the agent loop and the harness commands
//! share.

mod guard;
mod prose;
mod watch;

use std::fmt;
use std::str::FromStr;

pub use guard::{LanguageAlert, LanguageGuard, LanguageRefusal, WorkingText};
pub(crate) use guard::{default_alerts, default_refusal, translate};
pub(crate) use watch::LanguageAlerts;
use whatlang::Lang;

/// A natural language, as the detector knows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Language(Lang);

/// A language name the detector does not know.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown language `{0}`; use an English name, a native name or an ISO 639-3 code")]
pub struct UnknownLanguage(String);

impl Language {
    /// English, the default working language.
    pub const ENGLISH: Self = Self(Lang::Eng);

    /// The language's English name, such as `Chinese` or `Spanish`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self.0 {
            Lang::Cmn => "Chinese",
            lang => lang.eng_name(),
        }
    }

    /// The language's ISO 639-3 code, such as `cmn` or `spa`.
    #[must_use]
    pub fn code(self) -> &'static str {
        self.0.code()
    }

    pub(crate) const fn lang(self) -> Lang {
        self.0
    }

    pub(crate) fn is_english(self) -> bool {
        self.0 == Lang::Eng
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Language {
    type Err = UnknownLanguage;

    /// Parses an English name (`Chinese`, `german`), a native name or an ISO
    /// 639-3 code (`cmn`, `deu`), ignoring case. `zh` is accepted for Chinese.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let wanted = text.trim().to_lowercase();
        if wanted == "chinese" || wanted == "zh" {
            return Ok(Self(Lang::Cmn));
        }
        Lang::all()
            .iter()
            .copied()
            .find(|lang| {
                wanted == lang.code()
                    || wanted == lang.eng_name().to_lowercase()
                    || wanted == lang.name().to_lowercase()
            })
            .map(Self)
            .ok_or_else(|| UnknownLanguage(text.to_string()))
    }
}

/// The languages a session works in.
///
/// The communication language is the one the agent talks to the user in:
/// replies and the questions it asks. The working language, English by
/// default, is the one it writes in for others to read: `tasks.md`, prompts
/// and messages to subagents and other agents, issue and pull request text.
/// Both are inactive while the communication language is unset or English,
/// and the working language is enforced only once the communication language
/// is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Languages {
    communication: Option<Language>,
    working: Language,
}

impl Languages {
    /// Talks to the user in `communication` and writes in English.
    #[must_use]
    pub const fn new(communication: Language) -> Self {
        Self {
            communication: Some(communication),
            working: Language::ENGLISH,
        }
    }

    /// No language is set.
    #[must_use]
    pub const fn unset() -> Self {
        Self {
            communication: None,
            working: Language::ENGLISH,
        }
    }

    /// Writes for others in `working` instead of English.
    #[must_use]
    pub const fn with_working(mut self, working: Language) -> Self {
        self.working = working;
        self
    }

    /// The language the user is talked to in, as set. It is also what a
    /// sidecar that writes to the user, such as a recap or a prompt
    /// suggestion, writes in; [`enforced`](Self::enforced) tells whether the
    /// agent itself is held to it.
    #[must_use]
    pub const fn communication(&self) -> Option<Language> {
        self.communication
    }

    /// The language written for others, as set.
    #[must_use]
    pub const fn working(&self) -> Language {
        self.working
    }

    /// Whether languages are enforced: the communication language is set
    /// and is not English.
    #[must_use]
    pub fn enforced(&self) -> bool {
        self.communication
            .is_some_and(|language| !language.is_english())
    }
}

impl Default for Languages {
    fn default() -> Self {
        Self::unset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_parse_by_name_and_code() {
        assert_eq!("Chinese".parse::<Language>().unwrap().code(), "cmn");
        assert_eq!("zh".parse::<Language>().unwrap().name(), "Chinese");
        assert_eq!("deu".parse::<Language>().unwrap().name(), "German");
        assert!("Klingonese".parse::<Language>().is_err());
    }

    #[test]
    fn languages_are_enforced_only_for_a_non_english_communication_language() {
        let chinese: Language = "Chinese".parse().unwrap();
        assert!(Languages::new(chinese).enforced());
        assert!(!Languages::unset().enforced());
        assert!(!Languages::new(Language::ENGLISH).enforced());
        assert!(
            !Languages::new(Language::ENGLISH)
                .with_working(chinese)
                .enforced()
        );
    }
}
