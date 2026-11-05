//! Request interception for the headless browser.
//!
//! The browser follows redirects and loads subresources on its own, so
//! checking the first URL says nothing about the requests that follow. Every
//! request a page makes is therefore paused at the CDP `Fetch` domain's request
//! stage and judged by a [`RequestPolicy`] before it is sent: allowed ones
//! continue, the rest fail as `BlockedByClient`. A redirect produces a new
//! paused request for its target, so each hop is judged the same way.
//!
//! # What this does not stop
//!
//! The browser resolves host names itself *after* the policy has resolved and
//! approved them. A server that answers the policy's lookup with a public
//! address and the browser's with a private one (DNS rebinding) is therefore
//! not stopped by interception alone. Closing that gap needs the browser to
//! connect only to the address the policy approved, which means pinning
//! resolution (`--host-resolver-rules`, known only per host and so only for a
//! fixed set of hosts) or routing the browser through a local proxy that
//! connects to approved addresses itself.

use std::future::Future;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use chromiumoxide::Page;
use chromiumoxide::cdp::browser_protocol::fetch::{
    ContinueRequestParams, EnableParams, EventRequestPaused, FailRequestParams, RequestPattern,
    RequestStage,
};
use chromiumoxide::cdp::browser_protocol::network::{ErrorReason, ResourceType};
use futures_lite::StreamExt;
use tracing::{debug, warn};

use crate::guard::{UrlPolicyError, guard_url};

/// Decides whether the browser may send a request.
pub trait RequestPolicy: Clone + Send + Sync + 'static {
    /// Judges `url`, which is about to be requested.
    ///
    /// # Errors
    ///
    /// Returns why the request must not be sent.
    fn check(&self, url: &str) -> impl Future<Output = Result<(), UrlPolicyError>> + Send;
}

/// The policy of every fetch: only publicly routable destinations, as
/// [`guard_url`] decides.
#[derive(Debug, Clone, Copy, Default)]
pub struct PublicOnly;

impl RequestPolicy for PublicOnly {
    async fn check(&self, url: &str) -> Result<(), UrlPolicyError> {
        guard_url(url).await.map(|_| ())
    }
}

/// Whether a URL names something the browser reads without a network request.
fn is_local_resource(url: &str) -> bool {
    ["data:", "blob:", "about:"]
        .iter()
        .any(|scheme| url.starts_with(scheme))
}

/// Interception of one page's requests, until dropped.
#[derive(Debug)]
pub struct Interception {
    task: tokio::task::JoinHandle<()>,
    blocked_document: Arc<Mutex<Option<String>>>,
}

impl Interception {
    /// Pauses every request `page` makes from now on and judges it with
    /// `policy`.
    ///
    /// # Errors
    ///
    /// Returns an error if the `Fetch` domain cannot be enabled; nothing is
    /// then fetched unchecked.
    pub async fn start<P: RequestPolicy>(page: &Page, policy: P) -> Result<Self> {
        // Listen first so no request is paused before anyone answers it.
        let mut paused = page
            .event_listener::<EventRequestPaused>()
            .await
            .map_err(|e| anyhow!("Failed to listen for requests: {e}"))?;
        page.execute(EnableParams {
            patterns: Some(vec![RequestPattern {
                url_pattern: Some("*".to_string()),
                resource_type: None,
                request_stage: Some(RequestStage::Request),
            }]),
            handle_auth_requests: None,
        })
        .await
        .map_err(|e| anyhow!("Failed to enable request interception: {e}"))?;

        let blocked_document = Arc::new(Mutex::new(None));
        let blocked = blocked_document.clone();
        let page = page.clone();
        let task = tokio::spawn(async move {
            while let Some(event) = paused.next().await {
                let (page, policy, blocked) = (page.clone(), policy.clone(), blocked.clone());
                tokio::spawn(async move { answer(&page, &policy, &blocked, &event).await });
            }
        });
        Ok(Self {
            task,
            blocked_document,
        })
    }

    /// The reason the page's own navigation was refused, if it was.
    #[must_use]
    pub fn blocked_navigation(&self) -> Option<String> {
        self.blocked_document
            .lock()
            .expect("blocked navigation lock poisoned")
            .clone()
    }
}

impl Drop for Interception {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Continues or fails one paused request.
async fn answer<P: RequestPolicy>(
    page: &Page,
    policy: &P,
    blocked_document: &Mutex<Option<String>>,
    event: &EventRequestPaused,
) {
    let url = &event.request.url;
    let verdict = if is_local_resource(url) {
        Ok(())
    } else {
        policy.check(url).await
    };
    let sent = match verdict {
        Ok(()) => page
            .execute(ContinueRequestParams::new(event.request_id.clone()))
            .await
            .map(|_| ()),
        Err(refusal) => {
            warn!(%url, %refusal, "headless browser request refused");
            if event.resource_type == ResourceType::Document {
                blocked_document
                    .lock()
                    .expect("blocked navigation lock poisoned")
                    .get_or_insert_with(|| refusal.to_string());
            }
            page.execute(FailRequestParams::new(
                event.request_id.clone(),
                ErrorReason::BlockedByClient,
            ))
            .await
            .map(|_| ())
        }
    };
    if let Err(error) = sent {
        // The page is gone or the browser is closing; the request cannot be
        // answered and so is never sent.
        debug!(%url, %error, "could not answer a paused request");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_resources_need_no_check() {
        assert!(is_local_resource("data:image/png;base64,AAAA"));
        assert!(is_local_resource("blob:https://example.com/1"));
        assert!(is_local_resource("about:blank"));
    }

    #[test]
    fn everything_else_is_judged() {
        assert!(!is_local_resource("http://127.0.0.1/"));
        assert!(!is_local_resource("file:///etc/passwd"));
        assert!(!is_local_resource("https://example.com/data:"));
    }

    #[tokio::test]
    async fn the_public_policy_refuses_what_the_guard_refuses() {
        assert!(PublicOnly.check("http://169.254.169.254/").await.is_err());
        assert!(PublicOnly.check("file:///etc/passwd").await.is_err());
    }
}
