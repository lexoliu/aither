//! The context source that delivers language alerts.

use std::path::PathBuf;

use askama::Template;

use super::guard::{LanguageAlert, LanguageAlertsTemplate};
use super::{LanguageGuard, WorkingText};
use crate::context::{Block, ContextSource};
use crate::working_docs;

/// Delivers the alerts the guard's checks raised, and watches `tasks.md` for
/// text that is not in the working language.
///
/// `tasks.md` is edited through the agent's ordinary file tools, where a
/// write cannot be intercepted, so a change is judged after it was made: the
/// lines it added are checked once, and a violation raises one alert.
#[derive(Debug)]
pub struct LanguageAlerts {
    guard: LanguageGuard,
    /// Sandbox root holding `tasks.md`.
    sandbox_dir: Option<PathBuf>,
    /// `tasks.md` as last observed; `None` until the session start has.
    seen: Option<working_docs::WorkingDocsSnapshot>,
}

impl LanguageAlerts {
    pub(crate) const fn new(guard: LanguageGuard, sandbox_dir: Option<PathBuf>) -> Self {
        Self {
            guard,
            sandbox_dir,
            seen: None,
        }
    }

    async fn observe_tasks(&mut self) {
        let Some(sandbox_dir) = &self.sandbox_dir else {
            return;
        };
        let current = working_docs::read_snapshot(sandbox_dir).await;
        if let Some(previous) = &self.seen
            && let Some(added) = working_docs::tasks_md_added(previous, &current)
            && self
                .guard
                .check_working(&added, WorkingText::Internal)
                .is_err()
        {
            self.guard.raise(LanguageAlert::Record {
                language: self.guard.languages().working(),
                document: "tasks.md".to_string(),
            });
        }
        self.seen = Some(current);
    }
}

impl ContextSource for LanguageAlerts {
    async fn on_request(&mut self, _reassembled: bool) -> Option<Block> {
        if self.seen.is_none()
            && let Some(sandbox_dir) = &self.sandbox_dir
        {
            self.seen = Some(working_docs::read_snapshot(sandbox_dir).await);
        }
        None
    }

    async fn on_tool_turn_end(&mut self) -> Option<Block> {
        self.observe_tasks().await;
        let alerts = self.guard.take_alerts();
        if alerts.is_empty() {
            return None;
        }
        let content = LanguageAlertsTemplate { alerts: &alerts }
            .render()
            .expect("language alert template must render");
        Some(Block::new("system-reminder", content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::Languages;

    fn source(dir: &std::path::Path) -> (LanguageAlerts, LanguageGuard) {
        let guard = LanguageGuard::new(Languages::new("Chinese".parse().unwrap()));
        (
            LanguageAlerts::new(guard.clone(), Some(dir.to_path_buf())),
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
