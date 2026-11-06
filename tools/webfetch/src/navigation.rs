//! Where the page's own navigation sent its requests.
//!
//! The [`crate::proxy`] refuses requests without knowing which of them were
//! the page's navigation and which a subresource. This watches the main
//! frame's document requests, each redirect hop included, so the fetch can
//! tell a refused navigation, which fails it, from a refused image, which
//! does not. It only observes: nothing here decides what the browser may
//! reach.

use anyhow::{Result, anyhow};
use chromiumoxide::Page;
use chromiumoxide::cdp::browser_protocol::network::{EventRequestWillBeSent, ResourceType};
use chromiumoxide::cdp::browser_protocol::page::FrameId;
use chromiumoxide::listeners::EventStream;
use futures_lite::StreamExt;
use futures_lite::future::poll_once;
use url::Url;

use crate::proxy::Authority;

/// The main frame's document requests, from when it was started.
#[derive(Debug)]
pub struct Navigation {
    requests: EventStream<EventRequestWillBeSent>,
    main_frame: FrameId,
}

impl Navigation {
    /// Watches `page`'s main frame from now on.
    ///
    /// # Errors
    ///
    /// Returns an error if the page has no main frame or its requests cannot
    /// be listened to.
    pub async fn watch(page: &Page) -> Result<Self> {
        let requests = page
            .event_listener::<EventRequestWillBeSent>()
            .await
            .map_err(|e| anyhow!("Failed to listen for requests: {e}"))?;
        let main_frame = page
            .mainframe()
            .await
            .map_err(|e| anyhow!("Failed to find the main frame: {e}"))?
            .ok_or_else(|| anyhow!("The page has no main frame"))?;
        Ok(Self {
            requests,
            main_frame,
        })
    }

    /// The authorities of the document requests seen since the last call.
    pub async fn requested(&mut self) -> Vec<Authority> {
        let mut requested = Vec::new();
        while let Some(Some(event)) = poll_once(self.requests.next()).await {
            if event.r#type == Some(ResourceType::Document)
                && event.frame_id.as_ref() == Some(&self.main_frame)
                && let Some(authority) = Url::parse(&event.request.url)
                    .ok()
                    .as_ref()
                    .and_then(Authority::of_url)
            {
                requested.push(authority);
            }
        }
        requested
    }
}
