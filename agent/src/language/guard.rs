//! Enforcing the languages: the check every question, reply and record goes
//! through, and the alerts it raises.

use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{Event, model::Parameters, oneshot},
};
use arc_swap::ArcSwap;
use askama::Template;
use futures_lite::StreamExt;

use super::{Language, Languages, prose};
use crate::error::AgentError;

/// Who reads a text the agent writes in the working language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingText {
    /// A record, prompt or message for an agent or for the session's own
    /// files. It may quote the user's own words verbatim.
    Internal,
    /// Issue or pull request text. It never quotes the conversation, so no
    /// quote is exempt.
    Public,
}

/// Why a text was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageRefusal {
    /// A question to the user, or its options, not in the communication
    /// language.
    Question {
        /// The language the user is talked to in.
        communication: Language,
    },
    /// A text for others not in the working language.
    Working {
        /// The language such texts are written in.
        working: Language,
        /// Whether the text is public, which exempts no quote.
        public: bool,
    },
}

#[derive(Template)]
#[template(path = "language_refusal.txt", escape = "none")]
struct RefusalTemplate<'a> {
    refusal: &'a LanguageRefusal,
}

impl std::fmt::Display for LanguageRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = RefusalTemplate { refusal: self }
            .render()
            .expect("language refusal template must render");
        f.write_str(&text)
    }
}

impl std::error::Error for LanguageRefusal {}

/// A one-shot alert about a lapse of language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageAlert {
    /// A question was refused; ask it again.
    AskAgain { language: Language },
    /// A reply was not in the communication language.
    Reply { language: Language },
    /// A change to a record was not in the working language.
    Record {
        language: Language,
        document: String,
    },
}

#[derive(Template)]
#[template(path = "language_alert.txt", escape = "none")]
pub struct LanguageAlertsTemplate<'a> {
    pub alerts: &'a [LanguageAlert],
}

#[derive(Template)]
#[template(path = "language_translate.txt", escape = "none")]
struct TranslateTemplate {
    language: Language,
}

/// The language checks of one session, cloned into every place that writes.
///
/// The agent loop checks replies and `tasks.md` with it; a harness command
/// that takes text from the agent holds a clone and calls
/// [`check_working`](Self::check_working) before it acts, as the `issue`,
/// pull request and agent-message commands do. The clones share the user's
/// messages, which is what a quote is matched against, and the alerts the
/// checks raise, which the agent delivers to the model once.
#[derive(Debug, Clone)]
pub struct LanguageGuard {
    languages: Languages,
    /// The user's messages, whitespace-normalized.
    user: Arc<ArcSwap<Vec<Arc<str>>>>,
    alerts: async_channel::Sender<LanguageAlert>,
    raised: async_channel::Receiver<LanguageAlert>,
}

impl LanguageGuard {
    /// A guard enforcing `languages`.
    #[must_use]
    pub fn new(languages: Languages) -> Self {
        let (alerts, raised) = async_channel::unbounded();
        Self {
            languages,
            user: Arc::default(),
            alerts,
            raised,
        }
    }

    /// The languages this guard enforces. An application reads the session's
    /// communication language here.
    #[must_use]
    pub const fn languages(&self) -> &Languages {
        &self.languages
    }

    /// Records something the user wrote, so a verbatim quote of it is not
    /// judged as prose. The agent records every message of the user; a tool
    /// that returns the user's words records them too.
    pub fn record_user_message(&self, text: &str) {
        let text: Arc<str> = prose::normalize(text).into();
        self.user.rcu(|messages| {
            let mut messages = Vec::clone(messages);
            messages.push(Arc::clone(&text));
            messages
        });
    }

    /// Checks that a question to the user, given as its text and options, is
    /// in the communication language, and raises the alert that tells the
    /// agent to ask again when it is not.
    ///
    /// # Errors
    ///
    /// [`LanguageRefusal::Question`] when the prose is in another language.
    pub fn check_question<'a>(
        &self,
        parts: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), LanguageRefusal> {
        let Some(communication) = self.communication() else {
            return Ok(());
        };
        let text = parts.into_iter().collect::<Vec<_>>().join("\n");
        if prose::is_in(&text, communication, &[self.languages.working()]) {
            return Ok(());
        }
        self.raise(LanguageAlert::AskAgain {
            language: communication,
        });
        Err(LanguageRefusal::Question { communication })
    }

    /// Whether a reply to the user is in the communication language. Always
    /// true when no language is enforced.
    #[must_use]
    pub fn reply_in_language(&self, text: &str) -> bool {
        self.communication().is_none_or(|communication| {
            prose::is_in(text, communication, &[self.languages.working()])
        })
    }

    /// Checks that `text`, written for others to read, is in the working
    /// language. Nothing is checked while the working language is not
    /// enforced.
    ///
    /// # Errors
    ///
    /// [`LanguageRefusal::Working`] when the prose is in another language.
    pub fn check_working(&self, text: &str, kind: WorkingText) -> Result<(), LanguageRefusal> {
        let Some(communication) = self.communication() else {
            return Ok(());
        };
        let working = self.languages.working();
        let in_language = match kind {
            WorkingText::Internal => {
                let user = self.user.load();
                prose::is_in(
                    &prose::without_quotes(text, user.as_slice()),
                    working,
                    &[communication],
                )
            }
            WorkingText::Public => prose::is_in(text, working, &[communication]),
        };
        if in_language {
            Ok(())
        } else {
            Err(LanguageRefusal::Working {
                working,
                public: kind == WorkingText::Public,
            })
        }
    }

    /// The communication language, when languages are enforced.
    fn communication(&self) -> Option<Language> {
        self.languages
            .enforced()
            .then(|| self.languages.communication())
            .flatten()
    }

    pub(crate) fn raise(&self, alert: LanguageAlert) {
        // The channel is unbounded and the guard holds its receiver.
        let _ = self.alerts.try_send(alert);
    }

    /// The alerts raised since the last call.
    pub(crate) fn take_alerts(&self) -> Vec<LanguageAlert> {
        std::iter::from_fn(|| self.raised.try_recv().ok()).collect()
    }
}

/// Translates `text` into `language` on `model`, whose requests carry
/// `parameters`.
///
/// # Errors
///
/// The model's failure.
pub async fn translate<M: LanguageModel>(
    model: &M,
    parameters: &Parameters,
    text: &str,
    language: Language,
) -> Result<String, AgentError> {
    let system = TranslateTemplate { language }
        .render()
        .expect("language translate template must render");
    let request = oneshot(system, text).with_parameters(parameters.clone());
    // Boxed so the provider's large stream state does not inflate the run
    // future that awaits this (aither#143).
    let mut stream = Box::pin(model.respond(request));
    let mut translated = String::new();
    while let Some(event) = stream.next().await {
        if let Event::Text(chunk) = event.map_err(|error| AgentError::from_model_error(&error))? {
            translated.push_str(&chunk);
        }
    }
    Ok(translated.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> LanguageGuard {
        LanguageGuard::new(Languages::new("Chinese".parse().unwrap()))
    }

    #[test]
    fn an_english_question_is_refused_with_an_alert_and_a_chinese_one_with_code_passes() {
        let guard = guard();
        let refusal = guard
            .check_question(["Which database should we use?", "PostgreSQL", "MySQL"])
            .unwrap_err();
        assert!(refusal.to_string().contains("Chinese"), "{refusal}");
        assert_eq!(guard.take_alerts().len(), 1);

        guard
            .check_question([
                "应该用 `tokio::spawn` 还是 spawn_blocking 来运行 parse_config？",
                "使用 `tokio::spawn`",
                "使用 spawn_blocking",
                "PostgreSQL",
            ])
            .expect("a Chinese question with English code goes through");
        assert_eq!(guard.take_alerts().len(), 0);
    }

    #[test]
    fn nothing_is_enforced_without_a_non_english_communication_language() {
        for languages in [Languages::unset(), Languages::new(Language::ENGLISH)] {
            let guard = LanguageGuard::new(languages);
            guard.check_question(["Which database?"]).unwrap();
            guard
                .check_working("数据库迁移完成了，请检查结果。", WorkingText::Public)
                .unwrap();
            assert!(guard.reply_in_language("Anything at all."));
            assert_eq!(guard.take_alerts().len(), 0);
        }
    }

    #[test]
    fn working_text_in_the_communication_language_is_refused() {
        let guard = guard();
        let refusal = guard
            .check_working("数据库迁移完成了，请检查结果。", WorkingText::Public)
            .unwrap_err();
        assert!(refusal.to_string().contains("English"), "{refusal}");
        guard
            .check_working(
                "Migrate the database to `PostgreSQL` 16.",
                WorkingText::Public,
            )
            .unwrap();
    }

    #[test]
    fn a_public_text_never_gets_the_quote_exemption() {
        let guard = guard();
        guard.record_user_message("请把数据库换成 PostgreSQL，不要用 MySQL");
        let text = "Switch the database as requested: \u{201c}请把数据库换成 PostgreSQL，不要用 MySQL\u{201d}";
        guard.check_working(text, WorkingText::Internal).unwrap();
        guard.check_working(text, WorkingText::Public).unwrap_err();
    }
}
