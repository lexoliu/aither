//! Routing of subagents whose definition states no tier.
//!
//! [`SubagentTool`](super::SubagentTool) asks a [`SubagentRouter`] which
//! configured tier runs such a subagent. [`JudgmentRouter`] is the default;
//! an embedder replaces it with
//! [`SubagentTool::with_router`](super::SubagentTool::with_router).

use std::error::Error;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use aither_core::LanguageModel;
use aither_core::llm::model::Parameters;
use askama::Template;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::model_group::ModelTier;

/// One configured tier a router may choose.
#[derive(Debug, Clone, Copy)]
pub struct TierOffer<'a> {
    /// The tier.
    pub tier: ModelTier,
    /// The name of the model behind the tier.
    pub name: &'a str,
    /// What the model behind the tier is described as good for.
    pub description: &'a str,
}

/// What a router decides from.
#[derive(Debug, Clone, Copy)]
pub struct RouteRequest<'a> {
    /// The subagent's brief.
    pub brief: &'a str,
    /// The configured tiers, from most to least capable.
    pub tiers: &'a [TierOffer<'a>],
}

/// Why a router chose no tier.
#[derive(Debug, thiserror::Error)]
#[error("the subagent router chose no tier: {0}")]
pub struct SubagentRouteError(Box<dyn Error + Send + Sync>);

impl SubagentRouteError {
    /// A routing failure caused by `source`.
    #[must_use]
    pub fn new(source: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        Self(source.into())
    }
}

/// Chooses the tier a subagent whose definition states none runs on.
///
/// The tier must be one of [`RouteRequest::tiers`]. A failure stops the
/// subagent's start; nothing falls back to another tier.
pub trait SubagentRouter: Send + Sync + 'static {
    /// The tier for the brief `request` describes.
    fn route(
        &self,
        request: &RouteRequest<'_>,
    ) -> impl Future<Output = Result<ModelTier, SubagentRouteError>> + Send;
}

type Routed<'a> = Pin<Box<dyn Future<Output = Result<ModelTier, SubagentRouteError>> + Send + 'a>>;

/// Object-safe twin of [`SubagentRouter`]; boxing lives inside
/// [`AnySubagentRouter`].
trait SubagentRouterImpl: Send + Sync {
    fn route<'a>(&'a self, request: &'a RouteRequest<'a>) -> Routed<'a>;
}

impl<T: SubagentRouter> SubagentRouterImpl for T {
    fn route<'a>(&'a self, request: &'a RouteRequest<'a>) -> Routed<'a> {
        Box::pin(SubagentRouter::route(self, request))
    }
}

/// A shared [`SubagentRouter`].
#[derive(Clone)]
pub struct AnySubagentRouter(Arc<dyn SubagentRouterImpl>);

impl AnySubagentRouter {
    /// Shares `router`.
    #[must_use]
    pub fn new(router: impl SubagentRouter) -> Self {
        Self(Arc::new(router))
    }

    /// See [`SubagentRouter::route`].
    pub fn route<'a>(
        &'a self,
        request: &'a RouteRequest<'a>,
    ) -> impl Future<Output = Result<ModelTier, SubagentRouteError>> + Send + 'a {
        self.0.route(request)
    }
}

impl std::fmt::Debug for AnySubagentRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnySubagentRouter").finish_non_exhaustive()
    }
}

/// Structured response the model returns when routing a brief.
#[derive(Debug, JsonSchema, Deserialize)]
struct Routing {
    /// The configured model tier that best fits the task.
    model: ModelTier,
}

/// One configured model offered in the router prompt.
struct RouterModel<'a> {
    tier: &'static str,
    name: &'a str,
    description: &'a str,
}

/// Router prompt listing the configured models by tier.
#[derive(Template)]
#[template(path = "subagent_router.txt", escape = "none")]
struct RouterPrompt<'a> {
    models: Vec<RouterModel<'a>>,
}

pub const fn tier_label(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Advanced => "advanced",
        ModelTier::Balanced => "balanced",
        ModelTier::Fast => "fast",
    }
}

/// The default router: asks `model` which offered tier fits the brief, as a
/// [`judge`](crate::judge) over a fixed prompt that prefers the least
/// capable tier able to do the task.
#[derive(Debug, Clone)]
pub struct JudgmentRouter<M> {
    model: M,
    parameters: Parameters,
}

impl<M> JudgmentRouter<M> {
    /// A router asking `model`, with `parameters`.
    #[must_use]
    pub const fn new(model: M, parameters: Parameters) -> Self {
        Self { model, parameters }
    }
}

impl<M: LanguageModel + Sync + 'static> SubagentRouter for JudgmentRouter<M> {
    async fn route(&self, request: &RouteRequest<'_>) -> Result<ModelTier, SubagentRouteError> {
        let prompt = RouterPrompt {
            models: request
                .tiers
                .iter()
                .map(|offer| RouterModel {
                    tier: tier_label(offer.tier),
                    name: offer.name,
                    description: offer.description,
                })
                .collect(),
        };
        let routing: Routing =
            crate::judge(&self.model, self.parameters.clone(), &prompt, request.brief)
                .await
                .map_err(SubagentRouteError::new)?;
        Ok(routing.model)
    }
}
