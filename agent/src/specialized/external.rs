//! Subagents whose turns run outside aither.
//!
//! A harness implements [`ExternalSubagent`] for an agent that runs elsewhere,
//! such as a separate agent process, and registers it with
//! [`SubagentType::external`](super::SubagentType::external). The subagent
//! tool then launches, continues, tracks and cancels it through the same
//! paths it uses for aither agents.

use std::path::{Path, PathBuf};

use futures::future::BoxFuture;

/// What an external subagent starts with.
#[derive(Debug, Clone, Copy)]
pub struct ExternalStart<'a> {
    /// The caller's working directory: the terminal's working directory, else
    /// the tool's base directory. Relative grants were resolved against it.
    pub working_dir: &'a Path,
    /// Directories the caller granted the subagent to write, without the
    /// right to commit.
    pub worktrees: &'a [PathBuf],
    /// Linked git worktrees the caller granted the subagent to write and
    /// commit in. Their commits are listed in the subagent's result.
    pub commit_worktrees: &'a [PathBuf],
    /// The working copies the terminal's
    /// [`SubagentSpaces`](aither_sandbox::SubagentSpaces) prepared for the
    /// run in its space; empty when the terminal has none.
    pub working_copies: &'a [PathBuf],
}

/// A backend that opens subagent sessions running outside aither.
pub trait ExternalSubagent: Send + Sync + 'static {
    /// The session this backend opens.
    type Session: ExternalSession;
    /// Why a session could not be opened.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Opens a fresh session. The subagent tool opens one per `start` call.
    fn open(
        &self,
        start: ExternalStart<'_>,
    ) -> impl Future<Output = Result<Self::Session, Self::Error>> + Send;
}

/// One conversation with an external agent.
///
/// A session keeps its context between prompts, so `resume` prompts the same
/// session again. Dropping a session must end it: the subagent tool drops the
/// sessions it still holds, finished and unclaimed, with itself.
pub trait ExternalSession: Send + 'static {
    /// Why a turn failed.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Runs one turn and resolves with its final message.
    ///
    /// The tool drops this future when the run is cancelled, then calls
    /// [`cancel`](Self::cancel).
    fn prompt(&mut self, prompt: &str) -> impl Future<Output = Result<String, Self::Error>> + Send;

    /// Stops the turn that was in flight when its [`prompt`](Self::prompt)
    /// future was dropped.
    fn cancel(&mut self) -> impl Future<Output = ()> + Send;

    /// Ends the session.
    fn close(self) -> impl Future<Output = ()> + Send;

    /// The model the session runs on, as the subagent's result reports it.
    fn model(&self) -> Option<String>;
}

/// A backend failure, whatever its own error type.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct ExternalError(Box<dyn std::error::Error + Send + Sync>);

/// Object-safe twin of [`ExternalSubagent`].
trait ExternalSubagentImpl: Send + Sync {
    fn open<'a>(
        &'a self,
        start: ExternalStart<'a>,
    ) -> BoxFuture<'a, Result<AnyExternalSession, ExternalError>>;
}

impl<B: ExternalSubagent> ExternalSubagentImpl for B {
    fn open<'a>(
        &'a self,
        start: ExternalStart<'a>,
    ) -> BoxFuture<'a, Result<AnyExternalSession, ExternalError>> {
        Box::pin(async move {
            let session = ExternalSubagent::open(self, start)
                .await
                .map_err(|error| ExternalError(Box::new(error)))?;
            Ok(AnyExternalSession(Box::new(session)))
        })
    }
}

/// Object-safe twin of [`ExternalSession`].
trait ExternalSessionImpl: Send {
    fn prompt<'a>(&'a mut self, prompt: &'a str) -> BoxFuture<'a, Result<String, ExternalError>>;
    fn cancel(&mut self) -> BoxFuture<'_, ()>;
    fn close(self: Box<Self>) -> BoxFuture<'static, ()>;
    fn model(&self) -> Option<String>;
}

impl<S: ExternalSession> ExternalSessionImpl for S {
    fn prompt<'a>(&'a mut self, prompt: &'a str) -> BoxFuture<'a, Result<String, ExternalError>> {
        Box::pin(async move {
            ExternalSession::prompt(self, prompt)
                .await
                .map_err(|error| ExternalError(Box::new(error)))
        })
    }

    fn cancel(&mut self) -> BoxFuture<'_, ()> {
        Box::pin(ExternalSession::cancel(self))
    }

    fn close(self: Box<Self>) -> BoxFuture<'static, ()> {
        Box::pin(ExternalSession::close(*self))
    }

    fn model(&self) -> Option<String> {
        ExternalSession::model(self)
    }
}

/// A type-erased [`ExternalSubagent`].
pub struct AnyExternalSubagent(Box<dyn ExternalSubagentImpl>);

impl AnyExternalSubagent {
    pub fn new(backend: impl ExternalSubagent) -> Self {
        Self(Box::new(backend))
    }

    pub async fn open(
        &self,
        start: ExternalStart<'_>,
    ) -> Result<AnyExternalSession, ExternalError> {
        self.0.open(start).await
    }
}

/// A type-erased [`ExternalSession`].
pub struct AnyExternalSession(Box<dyn ExternalSessionImpl>);

impl AnyExternalSession {
    pub async fn prompt(&mut self, prompt: &str) -> Result<String, ExternalError> {
        self.0.prompt(prompt).await
    }

    pub async fn cancel(&mut self) {
        self.0.cancel().await;
    }

    pub async fn close(self) {
        self.0.close().await;
    }

    pub fn model(&self) -> Option<String> {
        self.0.model()
    }
}
