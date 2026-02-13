//! Model groups with budget tracking and automatic fallback.
//!
//! This module provides a tiered model system where models are organized by
//! capability (advanced, balanced, fast) with automatic fallback when budgets
//! are exhausted.

use std::sync::atomic::{AtomicU64, Ordering};

use aither_core::llm::Usage;

/// Budget limit for a model.
#[derive(Debug, Clone)]
pub enum Budget {
    /// Unlimited usage (no budget tracking).
    Unlimited,
    /// Limit by total tokens.
    Tokens(u64),
    /// Limit by estimated cost in USD (stored as microdollars for precision).
    Cost(f64),
}

impl Budget {
    /// Creates a token-based budget.
    #[must_use]
    pub const fn tokens(limit: u64) -> Self {
        Self::Tokens(limit)
    }

    /// Creates a cost-based budget in USD.
    #[must_use]
    pub const fn usd(limit: f64) -> Self {
        Self::Cost(limit)
    }
}

/// A model with budget tracking.
#[derive(Debug)]
pub struct BudgetedModel<M> {
    /// The underlying model.
    pub model: M,
    /// Budget limit.
    pub budget: Budget,
    /// Tokens used (atomic for thread safety).
    tokens_used: AtomicU64,
    /// Cost used in microdollars (1 USD = `1_000_000` microdollars).
    cost_used_micro: AtomicU64,
    /// Whether this model's budget is exhausted.
    exhausted: std::sync::atomic::AtomicBool,
}

impl<M> BudgetedModel<M> {
    /// Creates a new budgeted model.
    #[must_use]
    pub const fn new(model: M, budget: Budget) -> Self {
        Self {
            model,
            budget,
            tokens_used: AtomicU64::new(0),
            cost_used_micro: AtomicU64::new(0),
            exhausted: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Creates a model with unlimited budget.
    #[must_use]
    pub const fn unlimited(model: M) -> Self {
        Self::new(model, Budget::Unlimited)
    }

    /// Records usage from a response.
    pub fn record_usage(&self, usage: &Usage) {
        if let Some(tokens) = usage.total_tokens {
            self.tokens_used
                .fetch_add(u64::from(tokens), Ordering::Relaxed);
        }
        if let Some(cost) = usage.cost_usd {
            // Convert to microdollars
            let micro = (cost * 1_000_000.0) as u64;
            self.cost_used_micro.fetch_add(micro, Ordering::Relaxed);
        }
    }

    /// Marks this model as exhausted (e.g., from API quota error).
    pub fn mark_exhausted(&self) {
        self.exhausted.store(true, Ordering::Release);
    }

    /// Checks if the budget is exhausted.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        if self.exhausted.load(Ordering::Acquire) {
            return true;
        }

        match &self.budget {
            Budget::Unlimited => false,
            Budget::Tokens(limit) => self.tokens_used.load(Ordering::Relaxed) >= *limit,
            Budget::Cost(limit) => {
                let micro_limit = (*limit * 1_000_000.0) as u64;
                self.cost_used_micro.load(Ordering::Relaxed) >= micro_limit
            }
        }
    }

    /// Returns the current token usage.
    #[must_use]
    pub fn tokens_used(&self) -> u64 {
        self.tokens_used.load(Ordering::Relaxed)
    }

    /// Returns the current cost usage in USD.
    #[must_use]
    pub fn cost_used(&self) -> f64 {
        self.cost_used_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Resets the usage counters (but not the exhausted flag).
    pub fn reset_usage(&self) {
        self.tokens_used.store(0, Ordering::Relaxed);
        self.cost_used_micro.store(0, Ordering::Relaxed);
    }

    /// Resets everything including the exhausted flag.
    pub fn reset_all(&self) {
        self.reset_usage();
        self.exhausted.store(false, Ordering::Release);
    }

    /// Returns a reference to the underlying model.
    #[must_use]
    pub const fn inner(&self) -> &M {
        &self.model
    }
}

/// A group of models with the same tier, ordered by preference.
#[derive(Debug)]
pub struct ModelGroup<M> {
    /// Models in preference order (first = preferred).
    models: Vec<BudgetedModel<M>>,
    /// Current active model index.
    current: std::sync::atomic::AtomicUsize,
}

impl<M> ModelGroup<M> {
    /// Creates a new model group with a single model.
    #[must_use]
    pub fn new(model: BudgetedModel<M>) -> Self {
        Self {
            models: vec![model],
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Creates a group from multiple budgeted models.
    #[must_use]
    pub const fn from_models(models: Vec<BudgetedModel<M>>) -> Self {
        Self {
            models,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Adds a fallback model to the group.
    pub fn with_fallback(mut self, model: BudgetedModel<M>) -> Self {
        self.models.push(model);
        self
    }

    /// Returns the current active model, or None if all are exhausted.
    #[must_use]
    pub fn current(&self) -> Option<&BudgetedModel<M>> {
        if !self.try_advance() {
            return None;
        }
        let idx = self.current.load(Ordering::Acquire);
        self.models.get(idx).filter(|m| !m.is_exhausted())
    }

    /// Advances to the next model if the current one is exhausted.
    /// Returns true if a new model is available, false if all are exhausted.
    pub fn try_advance(&self) -> bool {
        loop {
            let idx = self.current.load(Ordering::Acquire);
            if idx >= self.models.len() {
                return false;
            }

            if let Some(model) = self.models.get(idx) {
                if !model.is_exhausted() {
                    return true;
                }
            }

            // Try to advance
            let next = idx + 1;
            if next >= self.models.len() {
                return false;
            }

            // CAS to advance
            if self
                .current
                .compare_exchange(idx, next, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                // Another thread advanced, retry
                continue;
            }

            // Check if the new model is available
            if let Some(model) = self.models.get(next) {
                if !model.is_exhausted() {
                    return true;
                }
            }
        }
    }

    /// Records usage for the current model.
    pub fn record_usage(&self, usage: &Usage) {
        let idx = self.current.load(Ordering::Acquire);
        if let Some(model) = self.models.get(idx) {
            model.record_usage(usage);
            if model.is_exhausted() {
                let _ = self.try_advance();
            }
        }
    }

    /// Marks the current model as exhausted and tries to advance.
    pub fn mark_current_exhausted(&self) -> bool {
        let idx = self.current.load(Ordering::Acquire);
        if let Some(model) = self.models.get(idx) {
            model.mark_exhausted();
        }
        self.try_advance()
    }

    /// Returns true if all models in the group are exhausted.
    #[must_use]
    pub fn all_exhausted(&self) -> bool {
        self.models.iter().all(BudgetedModel::is_exhausted)
    }

    /// Resets all models in the group.
    pub fn reset_all(&self) {
        for model in &self.models {
            model.reset_all();
        }
        self.current.store(0, Ordering::Release);
    }
}

/// Model tier classification for organizing models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTier {
    /// High-capability models (e.g., GPT-4, Claude Opus)
    Advanced,
    /// Balanced capability/cost models (e.g., GPT-4o, Claude Sonnet)
    Balanced,
    /// Fast/cheap models (e.g., GPT-4o-mini, Claude Haiku)
    Fast,
}

/// Collection of model groups organized by tier.
#[derive(Debug)]
pub struct TieredModels<M> {
    /// Advanced tier models.
    pub advanced: Option<ModelGroup<M>>,
    /// Balanced tier models.
    pub balanced: Option<ModelGroup<M>>,
    /// Fast tier models.
    pub fast: Option<ModelGroup<M>>,
}

impl<M> Default for TieredModels<M> {
    fn default() -> Self {
        Self {
            advanced: None,
            balanced: None,
            fast: None,
        }
    }
}

impl<M> TieredModels<M> {
    /// Creates empty tiered models.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the advanced tier models.
    #[must_use]
    pub fn with_advanced(mut self, group: ModelGroup<M>) -> Self {
        self.advanced = Some(group);
        self
    }

    /// Sets the balanced tier models.
    #[must_use]
    pub fn with_balanced(mut self, group: ModelGroup<M>) -> Self {
        self.balanced = Some(group);
        self
    }

    /// Sets the fast tier models.
    #[must_use]
    pub fn with_fast(mut self, group: ModelGroup<M>) -> Self {
        self.fast = Some(group);
        self
    }

    /// Gets the model group for a specific tier.
    #[must_use]
    pub const fn get(&self, tier: ModelTier) -> Option<&ModelGroup<M>> {
        match tier {
            ModelTier::Advanced => self.advanced.as_ref(),
            ModelTier::Balanced => self.balanced.as_ref(),
            ModelTier::Fast => self.fast.as_ref(),
        }
    }

    /// Gets a mutable reference to the model group for a specific tier.
    pub const fn get_mut(&mut self, tier: ModelTier) -> Option<&mut ModelGroup<M>> {
        match tier {
            ModelTier::Advanced => self.advanced.as_mut(),
            ModelTier::Balanced => self.balanced.as_mut(),
            ModelTier::Fast => self.fast.as_mut(),
        }
    }
}

// ============================================================================
// LanguageModel implementation for ModelGroup
// ============================================================================

use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, model::Profile},
};
use futures_core::Stream;

/// Error type for model group operations.
#[derive(Debug)]
pub enum ModelGroupError<E> {
    /// All models in the group are exhausted.
    AllModelsExhausted,
    /// Underlying model error.
    Model(E),
}

impl<E: std::fmt::Display> std::fmt::Display for ModelGroupError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllModelsExhausted => write!(f, "all models in group are exhausted"),
            Self::Model(e) => write!(f, "{e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ModelGroupError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AllModelsExhausted => None,
            Self::Model(e) => Some(e),
        }
    }
}

/// Checks if an error message indicates budget/quota exhaustion.
fn is_quota_error(error_msg: &str) -> bool {
    let lower = error_msg.to_lowercase();
    lower.contains("insufficient_quota")
        || lower.contains("rate_limit")
        || lower.contains("quota")
        || lower.contains("resource_exhausted")
        || lower.contains("overloaded")
        || lower.contains("capacity")
        || lower.contains("billing")
        || lower.contains("exceeded")
}

impl<M> LanguageModel for ModelGroup<M>
where
    M: LanguageModel + Send + Sync,
    M::Error: Send + Sync + 'static,
{
    type Error = ModelGroupError<M::Error>;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        ModelGroupStream::new(self, request)
    }

    async fn profile(&self) -> Profile {
        if let Some(model) = self.current() {
            model.inner().profile().await
        } else if let Some(model) = self.models.first() {
            model.inner().profile().await
        } else {
            Profile::new("unknown", "unknown", "unknown", "Empty model group", 0)
        }
    }
}

use std::pin::Pin;
use std::task::{Context, Poll};

/// Stream wrapper for `ModelGroup` that tracks usage and detects quota errors.
struct ModelGroupStream<'a, M: LanguageModel> {
    group: &'a ModelGroup<M>,
    inner: Option<Pin<Box<dyn Stream<Item = Result<Event, M::Error>> + Send + 'a>>>,
}

impl<'a, M: LanguageModel> ModelGroupStream<'a, M> {
    fn new(group: &'a ModelGroup<M>, request: LLMRequest) -> Self {
        let inner = group.current().map(|m| {
            let stream = m.inner().respond(request);
            Box::pin(stream) as Pin<Box<dyn Stream<Item = Result<Event, M::Error>> + Send + 'a>>
        });
        Self { group, inner }
    }
}

impl<M: LanguageModel> Stream for ModelGroupStream<'_, M>
where
    M::Error: Send + 'static,
{
    type Item = Result<Event, ModelGroupError<M::Error>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(ref mut inner) = self.inner else {
            return Poll::Ready(Some(Err(ModelGroupError::AllModelsExhausted)));
        };

        match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => {
                // Track usage
                if let Event::Usage(ref usage) = event {
                    if let Some(model) = self.group.current() {
                        model.record_usage(usage);
                    }
                }
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(e))) => {
                // Check if this is a quota error
                let error_msg = e.to_string();
                if is_quota_error(&error_msg) {
                    let _ = self.group.mark_current_exhausted();
                    tracing::warn!("Model quota exhausted: {}", error_msg);
                }
                self.inner = None;
                Poll::Ready(Some(Err(ModelGroupError::Model(e))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::{
        LanguageModel,
        llm::{Event, LLMRequest, model::Profile},
    };
    use futures_lite::{StreamExt, stream};

    #[derive(Debug, Clone)]
    struct DummyModel {
        name: &'static str,
    }

    #[derive(Debug)]
    struct DummyError;

    impl std::fmt::Display for DummyError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("dummy error")
        }
    }

    impl std::error::Error for DummyError {}

    impl LanguageModel for DummyModel {
        type Error = DummyError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl futures_core::Stream<Item = Result<Event, Self::Error>> + Send {
            stream::iter([Ok(Event::Text(self.name.to_string()))])
        }

        async fn profile(&self) -> Profile {
            Profile::new("dummy", self.name, "test", "dummy", 0)
        }
    }

    #[test]
    fn test_budget_tokens() {
        let model = BudgetedModel::new("gpt-4", Budget::tokens(1000));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(500, 400));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(50, 51));
        assert!(model.is_exhausted());
    }

    #[test]
    fn test_budget_cost() {
        let model = BudgetedModel::new("gpt-4", Budget::usd(1.0));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(0, 0).with_cost(0.5));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(0, 0).with_cost(0.6));
        assert!(model.is_exhausted());
    }

    #[test]
    fn test_model_group_fallback() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::new("primary", Budget::tokens(100)),
            BudgetedModel::new("fallback", Budget::tokens(100)),
        ]);

        // Initially on primary
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"primary")
        );

        // Exhaust primary
        group.record_usage(&Usage::new(50, 51));
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"fallback")
        );

        // Exhaust fallback
        group.record_usage(&Usage::new(50, 51));
        assert!(!group.try_advance());
        assert!(group.all_exhausted());
    }

    #[test]
    fn test_mark_exhausted() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::unlimited("primary"),
            BudgetedModel::unlimited("fallback"),
        ]);

        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"primary")
        );

        // Mark as exhausted (e.g., from API quota error)
        assert!(group.mark_current_exhausted());
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"fallback")
        );
    }

    #[test]
    fn current_advances_past_exhausted_models() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::new("primary", Budget::tokens(1)),
            BudgetedModel::new("fallback", Budget::tokens(100)),
        ]);
        group.models[0].mark_exhausted();
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"fallback")
        );
    }

    #[test]
    fn stream_uses_fallback_after_primary_is_exhausted() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::new(DummyModel { name: "primary" }, Budget::tokens(1)),
            BudgetedModel::new(DummyModel { name: "fallback" }, Budget::tokens(100)),
        ]);
        group.models[0].mark_exhausted();

        let request = LLMRequest::new([aither_core::llm::Message::user("hello")]);
        let mut stream = group.respond(request);
        let first = futures_lite::future::block_on(async { stream.next().await });

        match first {
            Some(Ok(Event::Text(text))) => assert_eq!(text, "fallback"),
            other => panic!("expected fallback model text event, got {other:?}"),
        }
    }
}
