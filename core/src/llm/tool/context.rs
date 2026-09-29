//! The per-call context a [`Tool`](super::Tool) runs with.
//!
//! Every invocation of [`Tool::call`](super::Tool::call) receives its own
//! [`ToolContext`]. Whoever dispatches the call decides what the context is
//! connected to: an MCP server connects it to the request's progress token, an
//! agent that has nowhere to show progress passes [`ToolContext::new`]. The
//! tool reports the same way in both cases and never learns which it got.

use alloc::boxed::Box;
use alloc::string::String;
use core::fmt::{self, Debug, Display};
use core::future::Future;
use core::pin::Pin;

/// How far a running tool call has got.
///
/// `progress` is the amount of work done so far and `total`, when known, the
/// amount there is; both may be fractional and neither has a unit. `message`
/// is a human-readable line describing the current state.
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    done: f64,
    total: Option<f64>,
    message: Option<String>,
}

impl Progress {
    /// Progress with the given amount of work done, no total, and no message.
    #[must_use]
    pub const fn new(progress: f64) -> Self {
        Self {
            done: progress,
            total: None,
            message: None,
        }
    }

    /// Sets the total amount of work.
    #[must_use]
    pub const fn with_total(mut self, total: f64) -> Self {
        self.total = Some(total);
        self
    }

    /// Sets a human-readable description of the current state.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// The amount of work done so far.
    #[must_use]
    pub const fn progress(&self) -> f64 {
        self.done
    }

    /// The total amount of work, if known.
    #[must_use]
    pub const fn total(&self) -> Option<f64> {
        self.total
    }

    /// The human-readable description of the current state, if any.
    #[must_use]
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }
}

/// Where a [`ToolContext`] delivers the progress its tool reports.
///
/// Implemented by whatever dispatches tool calls and has somewhere to show
/// progress, such as an MCP server forwarding it to the client as
/// `notifications/progress`. Every report reaching a sink has already been
/// validated by [`ToolContext::report_progress`]: its values are finite and
/// its `progress` is greater than that of the report before it.
pub trait ProgressSink: Send + 'static {
    /// Delivers one progress report.
    ///
    /// The returned future may wait for room to deliver (backpressure) but
    /// must not block the thread. A sink whose destination is gone drops the
    /// report: progress is advisory, and the call it belongs to is being torn
    /// down anyway.
    fn report(&mut self, progress: Progress) -> impl Future<Output = ()> + Send;
}

/// Object-safe twin of [`ProgressSink`], so [`ToolContext`] can hold any sink.
trait ProgressSinkImpl: Send {
    fn report(&mut self, progress: Progress) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

impl<S: ProgressSink> ProgressSinkImpl for S {
    fn report(&mut self, progress: Progress) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(ProgressSink::report(self, progress))
    }
}

/// A progress report that no receiver could make sense of.
///
/// Both variants are bugs in the reporting tool, so they are returned to it
/// rather than papered over.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgressError {
    /// `progress` or `total` was NaN or infinite.
    NotFinite(Progress),
    /// `progress` did not exceed the previous report's. Progress must increase
    /// with every report, even when the total is unknown.
    NotIncreasing {
        /// The `progress` of the previous accepted report.
        previous: f64,
        /// The rejected report.
        reported: Progress,
    },
}

impl Display for ProgressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite(progress) => write!(
                f,
                "progress report is not finite: progress {}, total {:?}",
                progress.done, progress.total
            ),
            Self::NotIncreasing { previous, reported } => write!(
                f,
                "progress must increase with every report: {} does not exceed {previous}",
                reported.done
            ),
        }
    }
}

impl core::error::Error for ProgressError {}

/// The context of one tool call.
///
/// A tool receives it by value in [`Tool::call`](super::Tool::call) and may
/// move it wherever the work happens. It is deliberately not `Clone`: one
/// owner reports in sequence, which is what lets it guarantee that progress
/// increases report by report.
pub struct ToolContext {
    sink: Option<Box<dyn ProgressSinkImpl>>,
    last_progress: Option<f64>,
}

impl ToolContext {
    /// A context whose caller does not listen for progress.
    ///
    /// Reports are still validated, so a tool that misreports fails the same
    /// way whether or not anyone is listening, but they are delivered nowhere.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sink: None,
            last_progress: None,
        }
    }

    /// A context that delivers its tool's progress reports to `sink`.
    #[must_use]
    pub fn with_progress(sink: impl ProgressSink) -> Self {
        Self {
            sink: Some(Box::new(sink)),
            last_progress: None,
        }
    }

    /// Whether the caller listens for progress.
    ///
    /// A tool whose work can outlast its caller's idle limit decides from
    /// this whether reporting progress keeps the call alive: without a
    /// listener, reports go nowhere and the caller sees nothing until the
    /// call returns.
    #[must_use]
    pub const fn is_listening(&self) -> bool {
        self.sink.is_some()
    }

    /// Reports how far the call has got.
    ///
    /// Delivered to the caller when it listens for progress, and a no-op when
    /// it does not. Awaiting the returned future may wait for the caller to
    /// take earlier reports.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::NotFinite`] when `progress` or `total` is NaN
    /// or infinite, and [`ProgressError::NotIncreasing`] when `progress` does
    /// not exceed the previous report's. A rejected report is not delivered
    /// and does not count as the previous report.
    pub async fn report_progress(&mut self, progress: Progress) -> Result<(), ProgressError> {
        if !progress.done.is_finite() || progress.total.is_some_and(|total| !total.is_finite()) {
            return Err(ProgressError::NotFinite(progress));
        }
        if let Some(previous) = self.last_progress
            && progress.done <= previous
        {
            return Err(ProgressError::NotIncreasing {
                previous,
                reported: progress,
            });
        }
        self.last_progress = Some(progress.done);
        if let Some(sink) = &mut self.sink {
            sink.report(progress).await;
        }
        Ok(())
    }
}

impl Default for ToolContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for ToolContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("listening", &self.sink.is_some())
            .field("last_progress", &self.last_progress)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use alloc::vec::Vec;
    use std::sync::mpsc;

    /// Forwards every report it is handed to a channel the test drains.
    struct Collect(mpsc::Sender<Progress>);

    impl ProgressSink for Collect {
        async fn report(&mut self, progress: Progress) {
            self.0.send(progress).expect("test receiver alive");
        }
    }

    fn listening() -> (ToolContext, mpsc::Receiver<Progress>) {
        let (tx, rx) = mpsc::channel();
        (ToolContext::with_progress(Collect(tx)), rx)
    }

    #[test]
    fn reports_whether_the_caller_listens() {
        assert!(!ToolContext::new().is_listening());
        assert!(listening().0.is_listening());
    }

    #[tokio::test]
    async fn delivers_increasing_reports_in_order() {
        let (mut cx, seen) = listening();

        cx.report_progress(Progress::new(1.0)).await.unwrap();
        cx.report_progress(Progress::new(2.5).with_total(10.0).with_message("half"))
            .await
            .unwrap();

        assert_eq!(
            seen.try_iter().collect::<Vec<_>>(),
            [
                Progress::new(1.0),
                Progress::new(2.5).with_total(10.0).with_message("half"),
            ]
        );
    }

    #[tokio::test]
    async fn rejects_non_increasing_and_non_finite_reports() {
        let (mut cx, seen) = listening();

        cx.report_progress(Progress::new(3.0)).await.unwrap();
        assert_eq!(
            cx.report_progress(Progress::new(3.0)).await,
            Err(ProgressError::NotIncreasing {
                previous: 3.0,
                reported: Progress::new(3.0),
            })
        );
        assert!(matches!(
            cx.report_progress(Progress::new(f64::NAN)).await,
            Err(ProgressError::NotFinite(_))
        ));
        assert!(matches!(
            cx.report_progress(Progress::new(4.0).with_total(f64::INFINITY))
                .await,
            Err(ProgressError::NotFinite(_))
        ));
        // A rejected report is not the new baseline.
        cx.report_progress(Progress::new(3.5)).await.unwrap();

        assert_eq!(
            seen.try_iter().collect::<Vec<_>>(),
            [Progress::new(3.0), Progress::new(3.5)]
        );
    }

    #[tokio::test]
    async fn context_without_a_listener_still_validates() {
        let mut cx = ToolContext::new();
        cx.report_progress(Progress::new(1.0)).await.unwrap();
        assert!(cx.report_progress(Progress::new(0.5)).await.is_err());
    }
}
