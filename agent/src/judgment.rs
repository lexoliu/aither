//! A one-shot structured judgment: one model request whose system prompt a
//! typed template renders, whose input is the text being judged, and whose
//! answer is decoded into the caller's type from the model's structured
//! output. Subagent routing asks the fast model this way, and an embedder's
//! own checks use the same path.

use aither_core::LanguageModel;
use aither_core::llm::{GenerateError, model::Parameters, oneshot};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

#[cfg(test)]
mod tests;

/// Why a judgment produced no answer.
#[derive(Debug, thiserror::Error)]
pub enum JudgmentError<E: core::error::Error + 'static> {
    /// The system prompt's template did not render.
    #[error("the judgment's system prompt did not render: {0}")]
    Render(#[from] askama::Error),
    /// The model failed, or answered something the answer type does not
    /// decode.
    #[error("the model gave no judgment: {0}")]
    Model(#[source] GenerateError<E>),
}

/// Asks `model` one question: `system` rendered as the system prompt,
/// `input` as the user message, sent with `parameters`, and answered as a
/// `T` from the model's structured output.
///
/// # Errors
///
/// Returns [`JudgmentError::Render`] when `system` does not render and
/// [`JudgmentError::Model`] when the model fails or its answer does not
/// decode into `T`.
pub fn judge<'a, T, M, S, I>(
    model: &'a M,
    parameters: Parameters,
    system: &S,
    input: I,
) -> impl Future<Output = Result<T, JudgmentError<M::Error>>> + Send + use<'a, T, M, S, I>
where
    T: JsonSchema + DeserializeOwned + 'static,
    M: LanguageModel,
    S: askama::Template,
    I: Into<String>,
{
    // Rendered before the request starts, so the future holds no template.
    let request = system
        .render()
        .map(|system| oneshot(system, input).with_parameters(parameters));
    async move { model.generate(request?).await.map_err(JudgmentError::Model) }
}
