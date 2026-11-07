//! The context source that delivers language alerts.

use super::guard::LanguageAlert;
use super::{LanguageGuard, WorkingText};
use crate::context::{Block, ContextSource};
use crate::working_document::file::{self, DocumentFile, Snapshot};

/// Delivers the alerts the guard's checks raised, and watches the working
/// document for text that is not in the working language.
///
/// The document is edited through the agent's ordinary file tools, where a
/// write cannot be intercepted, so a change is judged after it was made: the
/// lines it added are checked once, and a violation raises one alert.
#[derive(Debug)]
pub struct LanguageAlerts {
    guard: LanguageGuard,
    /// The working document, when the agent keeps one.
    document: Option<DocumentFile>,
    /// The document as last observed; `None` until the session start has.
    seen: Option<Snapshot>,
    /// Renders the alerts.
    notices: crate::AnyNotices,
}

impl LanguageAlerts {
    pub(crate) const fn new(
        guard: LanguageGuard,
        document: Option<DocumentFile>,
        notices: crate::AnyNotices,
    ) -> Self {
        Self {
            guard,
            document,
            seen: None,
            notices,
        }
    }

    async fn observe_document(&mut self) {
        let Some(document) = &self.document else {
            return;
        };
        let current = document.read().await;
        if let Some(previous) = &self.seen
            && let Some(added) = file::added(previous, &current)
            && self
                .guard
                .check_working(&added, WorkingText::Internal)
                .is_err()
        {
            self.guard.raise(LanguageAlert::Record {
                language: self.guard.languages().working(),
                document: document.name().to_string(),
            });
        }
        self.seen = Some(current);
    }
}

impl ContextSource for LanguageAlerts {
    async fn on_request(&mut self, _reassembled: bool) -> Option<Block> {
        if self.seen.is_none()
            && let Some(document) = &self.document
        {
            self.seen = Some(document.read().await);
        }
        None
    }

    async fn on_tool_turn_end(&mut self) -> Option<Block> {
        self.observe_document().await;
        let alerts = self.guard.take_alerts();
        if alerts.is_empty() {
            return None;
        }
        Some(Block::new(
            "system-reminder",
            self.notices.language_alerts(&alerts),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::Languages;

    fn source(dir: &std::path::Path) -> (LanguageAlerts, LanguageGuard) {
        let guard = LanguageGuard::new(Languages::new("Chinese".parse().unwrap()));
        (
            LanguageAlerts::new(
                guard.clone(),
                Some(DocumentFile::new(dir, "tasks.md")),
                crate::AnyNotices::default(),
            ),
            guard,
        )
    }

    async fn alerts_after_writing(
        source: &mut LanguageAlerts,
        dir: &std::path::Path,
        tasks: &str,
    ) -> Option<Block> {
        std::fs::write(dir.join("tasks.md"), tasks).expect("write tasks.md");
        source.on_tool_turn_end().await
    }

    #[tokio::test]
    async fn a_chinese_edit_to_tasks_md_yields_one_alert() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (mut source, _guard) = source(dir.path());
        assert!(source.on_request(false).await.is_none());

        let block = alerts_after_writing(
            &mut source,
            dir.path(),
            "- [ ] 把数据库迁移到新的服务器上\n",
        )
        .await
        .expect("a Chinese record must alert");
        assert!(block.content().contains("tasks.md"), "{}", block.content());
        assert!(block.content().contains("English"), "{}", block.content());
        assert!(
            source.on_tool_turn_end().await.is_none(),
            "the alert is one-shot"
        );
        assert!(
            alerts_after_writing(
                &mut source,
                dir.path(),
                "- [ ] 把数据库迁移到新的服务器上\n"
            )
            .await
            .is_none(),
            "an unchanged file does not alert again"
        );
        assert!(
            alerts_after_writing(
                &mut source,
                dir.path(),
                "- [ ] 把数据库迁移到新的服务器上\n- [ ] Migrate the database to `PostgreSQL` 16\n"
            )
            .await
            .is_none(),
            "an English line does not alert"
        );
    }

    #[tokio::test]
    async fn a_verbatim_user_quote_in_tasks_md_is_exempt_but_other_chinese_is_not() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (mut source, guard) = source(dir.path());
        guard.record_user_message("请把数据库换成 PostgreSQL，不要用 MySQL");
        source.on_request(false).await;

        let quoted = "- [ ] Switch the database. User: \u{201c}请把数据库换成   PostgreSQL，不要用\nMySQL\u{201d}\n";
        assert!(
            alerts_after_writing(&mut source, dir.path(), quoted)
                .await
                .is_none(),
            "a verbatim quote is exempt, whitespace aside"
        );

        let invented = format!(
            "{quoted}- [ ] Switch the database. User: \u{201c}他们想要完全不同的另一个方案\u{201d}\n"
        );
        assert!(
            alerts_after_writing(&mut source, dir.path(), &invented)
                .await
                .is_some(),
            "quoted Chinese the user never wrote is prose"
        );
    }
}
