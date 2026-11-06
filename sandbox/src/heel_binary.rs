//! The verified `heel` binary a terminal's sandboxes carry.

use std::sync::Arc;

use async_lock::OnceCell;
use heel::HeelBinary;

/// A terminal's `heel` binary, verified on first use and shared by the
/// terminal, its clones and its subagents' terminals.
///
/// Verification spawns `heel --version` and blocks, so it runs off the
/// executor. A failure is returned to the command that needed the sandbox and
/// is not cached: the next command verifies again and reports what it finds.
#[derive(Clone, Default)]
pub struct SharedHeelBinary(Arc<OnceCell<HeelBinary>>);

impl SharedHeelBinary {
    /// The verified binary, located and verified by `HeelBinary::locate` the
    /// first time.
    pub(crate) async fn get(&self) -> heel::Result<HeelBinary> {
        self.get_with(HeelBinary::locate).await
    }

    /// The verified binary, produced by `verify` the first time.
    async fn get_with(
        &self,
        verify: impl FnOnce() -> heel::Result<HeelBinary> + Send + 'static,
    ) -> heel::Result<HeelBinary> {
        self.0
            .get_or_try_init(|| blocking::unblock(verify))
            .await
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn counted(calls: &Arc<AtomicUsize>) -> impl FnOnce() -> heel::Result<HeelBinary> + use<> {
        let calls = calls.clone();
        move || {
            calls.fetch_add(1, Ordering::SeqCst);
            HeelBinary::locate()
        }
    }

    #[tokio::test]
    async fn verifies_once_for_every_clone() {
        let calls = Arc::new(AtomicUsize::new(0));
        let shared = SharedHeelBinary::default();
        let first = shared.get_with(counted(&calls)).await.unwrap();
        let second = shared.clone().get_with(counted(&calls)).await.unwrap();
        let third = shared.get_with(counted(&calls)).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(first, second);
        assert_eq!(first, third);
    }

    #[tokio::test]
    async fn concurrent_first_use_verifies_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let shared = SharedHeelBinary::default();
        let (a, (b, c)) = futures_lite::future::zip(
            shared.get_with(counted(&calls)),
            futures_lite::future::zip(
                shared.clone().get_with(counted(&calls)),
                shared.clone().get_with(counted(&calls)),
            ),
        )
        .await;
        a.unwrap();
        b.unwrap();
        c.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failure_is_reported_and_not_cached() {
        let calls = Arc::new(AtomicUsize::new(0));
        let shared = SharedHeelBinary::default();
        let failing = |calls: Arc<AtomicUsize>| {
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                HeelBinary::verify("/nonexistent/heel")
            }
        };
        let error = shared.get_with(failing(calls.clone())).await.unwrap_err();
        assert!(error.to_string().contains("the heel binary is missing"));
        shared.get_with(failing(calls.clone())).await.unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        shared.get_with(counted(&calls)).await.unwrap();
        shared.get_with(counted(&calls)).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
}
