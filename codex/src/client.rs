//! [`LanguageModel`] implementation backed by the `ChatGPT` Codex Responses
//! backend.
//!
//! [`Codex`] wraps [`aither_openai::Client`] pointed at
//! `https://chatgpt.com/backend-api/codex` with subscription-auth request
//! headers. It owns the OAuth token lifecycle: expired access tokens are
//! refreshed lazily inside `respond`, a `401` mid-stream triggers one forced
//! refresh and retry, and rotated credentials are pushed to a persistence
//! callback so the refresh token (single-use) is never lost.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use aither_core::llm::model::{Ability, Profile as ModelProfile};
use aither_core::llm::{Event, LLMRequest, LanguageModel};
use aither_openai::OpenAI;
use async_lock::Mutex;
use futures_lite::StreamExt as _;

use crate::auth::{CodexCredentials, refresh_tokens};
use crate::constant::{ACCOUNT_ID_HEADER, CODEX_BASE_URL, ORIGINATOR, RESIDENCY_HEADER};
use crate::error::CodexError;

/// Refresh the access token this long before its recorded expiry.
const EXPIRY_SKEW: Duration = Duration::from_secs(60);

/// Residency sentinel that means "no constraint" — the header is omitted.
const NO_RESIDENCY_CONSTRAINT: &str = "no_constraint";

/// Callback invoked whenever the credential set changes (token rotation).
///
/// The implementation is expected to persist the credentials; it is invoked
/// synchronously while the token lock is held, so it should schedule any I/O
/// rather than perform it inline.
pub type OnRefresh = Arc<dyn Fn(&CodexCredentials) + Send + Sync>;

/// A [`LanguageModel`] for a single model on the Codex subscription backend.
#[derive(Clone)]
pub struct Codex {
    inner: Arc<Inner>,
}

struct Inner {
    model: String,
    base_url: String,
    originator: String,
    session_id: String,
    cell: Mutex<Cell>,
    on_refresh: Option<OnRefresh>,
}

struct Cell {
    credentials: CodexCredentials,
}

impl Codex {
    /// Create a client for `model` using the given credentials.
    #[must_use]
    pub fn new(credentials: CodexCredentials, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                model: model.into(),
                base_url: CODEX_BASE_URL.into(),
                originator: ORIGINATOR.into(),
                session_id: session_id(),
                cell: Mutex::new(Cell { credentials }),
                on_refresh: None,
            }),
        }
    }

    /// Override the backend base URL (defaults to the Codex subscription
    /// endpoint).
    ///
    /// # Panics
    ///
    /// Panics if the client has already been cloned/shared.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_base_url must be called before the client is shared")
            .base_url = base_url.into();
        self
    }

    /// Override the `originator` header (defaults to [`ORIGINATOR`]).
    ///
    /// # Panics
    ///
    /// Panics if the client has already been cloned/shared.
    #[must_use]
    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_originator must be called before the client is shared")
            .originator = originator.into();
        self
    }

    /// Register a callback fired whenever tokens rotate.
    ///
    /// # Panics
    ///
    /// Panics if the client has already been cloned/shared.
    #[must_use]
    pub fn on_refresh(
        mut self,
        callback: impl Fn(&CodexCredentials) + Send + Sync + 'static,
    ) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("on_refresh must be called before the client is shared")
            .on_refresh = Some(Arc::new(callback));
        self
    }

    /// A snapshot of the current credentials.
    pub async fn credentials(&self) -> CodexCredentials {
        self.inner.cell.lock().await.credentials.clone()
    }
}

impl Inner {
    /// Build the inner `OpenAI` Responses client for a credential set.
    fn openai_client(&self, credentials: &CodexCredentials) -> OpenAI {
        let mut builder = OpenAI::builder(&credentials.access_token)
            .base_url(&self.base_url)
            .model(&self.model)
            .store(false)
            .extra_header("originator", &self.originator)
            .extra_header("session-id", &self.session_id);
        if let Some(account_id) = &credentials.account_id {
            builder = builder.extra_header(ACCOUNT_ID_HEADER, account_id);
        }
        if let Some(residency) = credentials
            .residency
            .as_deref()
            .filter(|r| *r != NO_RESIDENCY_CONSTRAINT)
        {
            builder = builder.extra_header(RESIDENCY_HEADER, residency);
        }
        builder.build()
    }

    /// Return valid credentials, refreshing if expired (or if `force`).
    ///
    /// The lock is held across the refresh so concurrent callers share one
    /// refresh — the first to acquire performs it and the rest observe the
    /// fresh credentials.
    async fn valid_credentials(&self, force: bool) -> Result<CodexCredentials, CodexError> {
        let mut cell = self.cell.lock().await;
        if !force && !cell.credentials.is_expired(EXPIRY_SKEW) {
            return Ok(cell.credentials.clone());
        }
        let fresh = refresh_tokens(&cell.credentials.refresh_token).await?;
        tracing::debug!("refreshed ChatGPT access token");
        if let Some(callback) = &self.on_refresh {
            callback(&fresh);
        }
        cell.credentials = fresh.clone();
        drop(cell);
        Ok(fresh)
    }

    /// Build a request-ready client (fresh token + credential headers).
    async fn request_client(&self, force_refresh: bool) -> Result<OpenAI, CodexError> {
        let credentials = self.valid_credentials(force_refresh).await?;
        Ok(self.openai_client(&credentials))
    }
}

impl fmt::Debug for Codex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Codex")
            .field("model", &self.inner.model)
            .field("base_url", &self.inner.base_url)
            .field("session_id", &self.inner.session_id)
            .finish_non_exhaustive()
    }
}

impl LanguageModel for Codex {
    type Error = CodexError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl futures_core::Stream<Item = Result<Event, Self::Error>> + Send {
        let inner = Arc::clone(&self.inner);
        async_stream::stream! {
            let mut retried = false;
            loop {
                let client = match inner.request_client(retried).await {
                    Ok(client) => client,
                    Err(err) => {
                        yield Err(err);
                        return;
                    }
                };
                let mut stream = std::pin::pin!(client.respond(request.clone()));
                let mut unauthorized = false;
                while let Some(event) = stream.next().await {
                    match event {
                        Err(err) if is_unauthorized(&err) && !retried => {
                            unauthorized = true;
                            break;
                        }
                        event => yield event.map_err(CodexError::OpenAI),
                    }
                }
                if !unauthorized {
                    return;
                }
                tracing::warn!("Codex request unauthorized; refreshing token and retrying");
                retried = true;
            }
        }
    }

    fn profile(&self) -> impl core::future::Future<Output = ModelProfile> + Send {
        let model = self.inner.model.clone();
        async move {
            let entry = aither_models::lookup(&model);
            let mut profile = ModelProfile::new(
                model.clone(),
                "openai",
                model,
                "OpenAI model through a ChatGPT subscription",
                entry
                    .and_then(aither_models::ModelEntry::max_input_tokens)
                    .unwrap_or(0),
            )
            .with_ability(Ability::ToolUse);
            if let Some(entry) = entry {
                for ability in entry.abilities() {
                    if !profile.abilities.contains(ability) {
                        profile.abilities.push(*ability);
                    }
                }
            }
            profile
        }
    }
}

/// Whether an `OpenAIError` is an HTTP 401.
const fn is_unauthorized(err: &aither_openai::OpenAIError) -> bool {
    matches!(
        err,
        aither_openai::OpenAIError::Http(zenwave::Error::Http { status, .. })
            if status.as_u16() == 401
    )
}

/// Random session identifier for the `session-id` header.
fn session_id() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("secure random source unavailable");
    bytes.iter().fold(String::with_capacity(32), |mut s, b| {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
        s
    })
}
