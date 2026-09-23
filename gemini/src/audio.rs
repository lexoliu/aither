use aither_core::audio::{AudioGenerator, AudioTranscriber, Data as AudioData};
use futures_lite::StreamExt;

use crate::{
    client::call_generate,
    config::{Gemini, GeminiConfig},
    error::GeminiError,
    types::{
        GeminiContent, GenerateContentRequest, GenerationConfig, Part, PrebuiltVoiceConfig,
        SpeechConfig, ThinkingConfig, VoiceConfig,
    },
};

impl AudioGenerator for Gemini {
    type Error = GeminiError;

    fn generate(
        &self,
        prompt: &str,
    ) -> impl futures_core::Stream<Item = Result<AudioData, Self::Error>> + Send {
        let cfg = self.config();
        let text = prompt.to_owned();
        futures_lite::stream::iter(vec![synthesize_audio(cfg, text)]).then(|fut| fut)
    }
}

impl AudioTranscriber for Gemini {
    type Error = GeminiError;

    fn transcribe(
        &self,
        audio: &[u8],
    ) -> impl futures_core::Stream<Item = Result<String, Self::Error>> + Send {
        let cfg = self.config();
        let payload = audio.to_vec();
        futures_lite::stream::iter(vec![transcribe_audio(cfg, payload)]).then(|fut| fut)
    }
}

async fn synthesize_audio(cfg: &GeminiConfig, text: String) -> Result<Vec<u8>, GeminiError> {
    let model = cfg.tts_model.clone().ok_or_else(|| {
        GeminiError::Api("audio generation is disabled for this Gemini backend".into())
    })?;
    let request = GenerateContentRequest {
        system_instruction: None,
        contents: vec![GeminiContent::text("user", text)],
        generation_config: Some(GenerationConfig {
            response_modalities: Some(vec!["AUDIO".into()]),
            speech_config: Some(SpeechConfig {
                voice_config: Some(VoiceConfig {
                    prebuilt: PrebuiltVoiceConfig {
                        voice_name: cfg.tts_voice.clone(),
                    },
                }),
            }),
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(false),
                thinking_level: None,
            }),
            ..GenerationConfig::default()
        }),
        tools: Vec::new(),
        tool_config: None,
        safety_settings: Vec::new(),
        cached_content: None,
    };
    let response = call_generate(cfg, &model, request).await?;
    if let Some(candidate) = response.primary_candidate()
        && let Some(content) = &candidate.content
    {
        for part in &content.parts {
            if let Some(inline) = &part.inline_data {
                let bytes = inline.decode()?;
                return Ok(bytes);
            }
        }
    }
    Err(GeminiError::Api(
        "Gemini did not return audio data in the response".into(),
    ))
}

async fn transcribe_audio(cfg: &GeminiConfig, audio: Vec<u8>) -> Result<String, GeminiError> {
    let mut parts = vec![Part::inline_audio(audio)];
    parts.push(Part::text(
        "Transcribe the audio verbatim in the original language.",
    ));
    let request = GenerateContentRequest {
        system_instruction: None,
        contents: vec![GeminiContent::with_parts("user", parts)],
        generation_config: Some(GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(false),
                thinking_level: None,
            }),
            ..GenerationConfig::default()
        }),
        tools: Vec::new(),
        tool_config: None,
        safety_settings: Vec::new(),
        cached_content: None,
    };
    let response = call_generate(cfg, &cfg.text_model, request).await?;
    if let Some(candidate) = response.primary_candidate()
        && let Some(content) = &candidate.content
    {
        let text = content.text_chunks().join("");
        return Ok(text);
    }
    Err(GeminiError::Api(
        "Gemini did not return transcription text".into(),
    ))
}
