use aither_core::{EmbeddingModel, Error as AitherError, Result as AitherResult};

use crate::{
    client::embed_content,
    config::Gemini,
    types::{EmbedContentRequest, GeminiContent},
};

impl EmbeddingModel for Gemini {
    fn dim(&self) -> usize {
        self.config().embedding_dimensions
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl core::future::Future<Output = AitherResult<Vec<f32>>> + Send {
        let cfg = self.config();
        let prompt = text.to_owned();
        async move {
            let request =
                EmbedContentRequest::new(&cfg.embedding_model, GeminiContent::text("user", prompt));
            let response = embed_content(cfg, request)
                .await
                .map_err(AitherError::from)?;
            Ok(response.embedding.values)
        }
    }
}
