use alloc::{string::String, vec::Vec};
use futures_core::Stream;

/// Audio data as bytes.
///
/// Type alias for [`Vec<u8>`] representing raw audio data.
pub type Data = Vec<u8>;

/// Generates audio from text prompts.
/// # Example
///
/// ```rust,ignore
/// use aither::AudioGenerator;
/// use futures_core::Stream;
///
/// struct MyAudioGen;
///
/// impl AudioGenerator for MyAudioGen {
///     fn generate(&self, prompt: &str) -> impl Stream<Item = aither::audio::Data> + Send {
///         futures_lite::stream::iter(Some(vec![0u8; 1024]))
///     }
/// }
/// ```
pub trait AudioGenerator {
    /// Generates audio from text prompt.
    ///
    /// Returns a [`Stream`] of [`Data`] chunks.
    fn generate(&self, prompt: &str) -> impl Stream<Item = Data> + Send;
}

/// Transcribes audio to text.
///
/// # Example
///
/// ```rust,ignore
/// use aither::AudioTranscriber;
/// use futures_core::Stream;
///
/// struct MyTranscriber;
///
/// impl AudioTranscriber for MyTranscriber {
///     fn transcribe(&self, audio: &[u8]) -> impl Stream<Item = String> + Send {
///         futures_lite::stream::iter(vec!["Hello world".to_string()])
///     }
/// }
/// ```
pub trait AudioTranscriber {
    /// Transcribes audio data to text.
    ///
    /// Returns a [`Stream`] of transcribed text chunks.
    fn transcribe(&self, audio: &[u8]) -> impl Stream<Item = String> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::ToString, vec};
    use futures_lite::StreamExt;

    struct MockAudioGenerator;

    impl AudioGenerator for MockAudioGenerator {
        fn generate(&self, prompt: &str) -> impl Stream<Item = Data> + Send {
            // Generate mock audio data based on prompt length
            let chunks = if prompt.is_empty() {
                vec![]
            } else if prompt.len() < 10 {
                vec![vec![0x01; 512]] // Short audio for short prompts
            } else {
                vec![
                    vec![0x01; 512],  // First chunk
                    vec![0x02; 1024], // Second chunk
                    vec![0x03; 256],  // Final chunk
                ]
            };

            futures_lite::stream::iter(chunks)
        }
    }

    struct MockAudioTranscriber;

    impl AudioTranscriber for MockAudioTranscriber {
        fn transcribe(&self, audio: &[u8]) -> impl Stream<Item = String> + Send {
            // Generate mock transcription based on audio length
            let text_chunks = if audio.is_empty() {
                vec![]
            } else if audio.len() < 100 {
                vec!["Short".to_string()]
            } else if audio.len() < 1000 {
                vec!["Hello".to_string(), " world".to_string()]
            } else {
                vec![
                    "This".to_string(),
                    " is".to_string(),
                    " a".to_string(),
                    " longer".to_string(),
                    " transcription".to_string(),
                ]
            };

            futures_lite::stream::iter(text_chunks)
        }
    }

    #[tokio::test]
    async fn audio_generator_short_prompt() {
        let generator = MockAudioGenerator;
        let mut stream = generator.generate("Hi");

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 512);
        assert_eq!(chunks[0][0], 0x01);
    }

    #[tokio::test]
    async fn audio_generator_long_prompt() {
        let generator = MockAudioGenerator;
        let mut stream = generator
            .generate("This is a longer prompt that should generate multiple audio chunks");

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 512);
        assert_eq!(chunks[1].len(), 1024);
        assert_eq!(chunks[2].len(), 256);

        assert_eq!(chunks[0][0], 0x01);
        assert_eq!(chunks[1][0], 0x02);
        assert_eq!(chunks[2][0], 0x03);
    }

    #[tokio::test]
    async fn audio_generator_empty_prompt() {
        let generator = MockAudioGenerator;
        let mut stream = generator.generate("");

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn audio_transcriber_short_audio() {
        let transcriber = MockAudioTranscriber;
        let audio_data = vec![0x01; 50]; // Short audio
        let mut stream = transcriber.transcribe(&audio_data);

        let mut text_chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            text_chunks.push(chunk);
        }

        assert_eq!(text_chunks.len(), 1);
        assert_eq!(text_chunks[0], "Short");
    }

    #[tokio::test]
    async fn audio_transcriber_medium_audio() {
        let transcriber = MockAudioTranscriber;
        let audio_data = vec![0x01; 500]; // Medium audio
        let mut stream = transcriber.transcribe(&audio_data);

        let mut text_chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            text_chunks.push(chunk);
        }

        assert_eq!(text_chunks.len(), 2);
        assert_eq!(text_chunks[0], "Hello");
        assert_eq!(text_chunks[1], " world");
    }

    #[tokio::test]
    async fn audio_transcriber_long_audio() {
        let transcriber = MockAudioTranscriber;
        let audio_data = vec![0x01; 2000]; // Long audio
        let mut stream = transcriber.transcribe(&audio_data);

        let mut text_chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            text_chunks.push(chunk);
        }

        assert_eq!(text_chunks.len(), 5);
        let full_text: String = text_chunks.join("");
        assert_eq!(full_text, "This is a longer transcription");
    }

    #[tokio::test]
    async fn audio_transcriber_empty_audio() {
        let transcriber = MockAudioTranscriber;
        let audio_data = vec![]; // Empty audio
        let mut stream = transcriber.transcribe(&audio_data);

        let mut text_chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            text_chunks.push(chunk);
        }

        assert!(text_chunks.is_empty());
    }

    #[test]
    fn data_type_alias() {
        let data: Data = vec![1, 2, 3, 4, 5];
        assert_eq!(data.len(), 5);
        assert_eq!(data[0], 1);
        assert_eq!(data[4], 5);
    }

    #[test]
    fn data_operations() {
        let mut data: Data = vec![0xFF; 1024];
        assert_eq!(data.len(), 1024);

        // Test push
        data.push(0x00);
        assert_eq!(data.len(), 1025);
        assert_eq!(data[1024], 0x00);

        // Test extend
        data.extend_from_slice(&[0x01, 0x02, 0x03]);
        assert_eq!(data.len(), 1028);
        assert_eq!(data[1025], 0x01);
        assert_eq!(data[1026], 0x02);
        assert_eq!(data[1027], 0x03);

        // Test clear
        data.clear();
        assert!(data.is_empty());
    }

    #[test]
    fn data_creation() {
        let empty_data: Data = Vec::new();
        assert!(empty_data.is_empty());

        let filled_data: Data = vec![42; 100];
        assert_eq!(filled_data.len(), 100);
        assert!(filled_data.iter().all(|&x| x == 42));
    }

    #[tokio::test]
    async fn audio_workflow() {
        let generator = MockAudioGenerator;
        let transcriber = MockAudioTranscriber;

        // Generate audio from text
        let prompt = "Hello world";
        let mut audio_stream = generator.generate(prompt);

        let mut all_audio_data = Vec::new();
        while let Some(chunk) = audio_stream.next().await {
            all_audio_data.extend_from_slice(&chunk);
        }

        // Transcribe the generated audio back to text
        let mut transcription_stream = transcriber.transcribe(&all_audio_data);

        let mut transcription_chunks = Vec::new();
        while let Some(chunk) = transcription_stream.next().await {
            transcription_chunks.push(chunk);
        }

        // Verify the workflow
        assert!(!all_audio_data.is_empty());
        assert!(!transcription_chunks.is_empty());

        let full_transcription: String = transcription_chunks.join("");
        assert_eq!(full_transcription, "This is a longer transcription");
    }
}
