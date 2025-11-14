use aither_core::image::{Data as ImageData, ImageGenerator, Prompt, Size};
use futures_core::Stream;
use futures_lite::StreamExt;

use crate::{
    client::call_generate,
    config::GeminiBackend,
    error::GeminiError,
    types::{GeminiContent, GenerateContentRequest, GenerationConfig, Part},
};

impl ImageGenerator for GeminiBackend {
    type Error = GeminiError;

    fn create(
        &self,
        prompt: Prompt,
        size: Size,
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        let cfg = self.config();
        let text = prompt.text().to_owned();
        let images = prompt.images().to_vec();
        let size_desc = format!(
            "Generate an image roughly {}x{} pixels.",
            size.width(),
            size.height()
        );

        futures_lite::stream::iter(vec![async move {
            let model = cfg.image_model.clone().ok_or_else(|| {
                GeminiError::Api("image generation is disabled for this Gemini backend".into())
            })?;
            let mut parts = vec![Part::text(text)];
            for image in images {
                parts.push(Part::inline_image(image));
            }
            parts.push(Part::text(size_desc));

            let request = GenerateContentRequest {
                system_instruction: None,
                contents: vec![GeminiContent::with_parts("user", parts)],
                generation_config: Some(GenerationConfig {
                    response_modalities: Some(vec!["IMAGE".into()]),
                    ..GenerationConfig::default()
                }),
                tools: Vec::new(),
                tool_config: None,
                thinking_config: None,
                safety_settings: Vec::new(),
            };
            call_generate(cfg, &model, request).await
        }])
        .then(|fut| fut)
        .filter_map(Result::ok)
        .flat_map(|response| {
            let parts = response
                .primary_candidate()
                .and_then(|candidate| candidate.content.as_ref())
                .map(|content| content.parts.clone())
                .unwrap_or_default();
            futures_lite::stream::iter(parts)
                .filter_map(|part| {
                    part.inline_data
                        .as_ref()
                        .and_then(|inline| inline.decode().ok())
                })
                .map(Ok)
        })
    }

    fn edit(
        &self,
        prompt: Prompt,
        mask: &[u8],
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        let cfg = self.config();
        let mask_bytes = mask.to_vec();
        let text = prompt.text().to_owned();
        let base_image = prompt.images().first().cloned();

        futures_lite::stream::iter(vec![async move {
            let model = cfg.image_model.clone().ok_or_else(|| {
                GeminiError::Api("image generation is disabled for this Gemini backend".into())
            })?;
            let base_image = base_image.ok_or_else(|| {
                GeminiError::Api(
                    "image editing requires Prompt::with_image to supply a base image".into(),
                )
            })?;
            let mut parts = vec![Part::inline_image(base_image)];
            if !mask_bytes.is_empty() {
                parts.push(Part::inline_mask(mask_bytes));
            }
            parts.push(Part::text(format!(
                "Apply the mask and follow these instructions: {text}"
            )));
            let request = GenerateContentRequest {
                system_instruction: None,
                contents: vec![GeminiContent::with_parts("user", parts)],
                generation_config: Some(GenerationConfig {
                    response_modalities: Some(vec!["IMAGE".into()]),
                    ..GenerationConfig::default()
                }),
                tools: Vec::new(),
                tool_config: None,
                thinking_config: None,
                safety_settings: Vec::new(),
            };
            call_generate(cfg, &model, request).await
        }])
        .then(|fut| fut)
        .filter_map(Result::ok)
        .flat_map(|response| {
            let parts = response
                .primary_candidate()
                .and_then(|candidate| candidate.content.as_ref())
                .map(|content| content.parts.clone())
                .unwrap_or_default();
            futures_lite::stream::iter(parts)
                .filter_map(|part| {
                    part.inline_data
                        .as_ref()
                        .and_then(|inline| inline.decode().ok())
                })
                .map(Ok)
        })
    }
}
