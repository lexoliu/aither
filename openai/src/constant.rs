//! Model & endpoint constants
//!
//! These constants only cover **stable, recommended, non-snapshot** model names
//! plus well-known OpenAI-compatible API base URLs.
//! Users can always pass custom strings for newer snapshot models.

/// Default `OpenAI` API base URL (chat, embeddings, etc.).
pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// [`Deepseek`](https://api-docs.deepseek.com)'s OpenAI-compatible base URL.
pub const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";
/// [`OpenRouter`](https://openrouter.ai)'s OpenAI-compatible base URL.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

// ============================================================
// 1. GPT-5 FAMILY (Latest, reasoning-enhanced large models)
// ============================================================

/// Latest flagship GPT-5.2 model.
pub const GPT5_2: &str = "gpt-5.2";

/// Dynamic alias pointing to the latest GPT-5.2 chat model.
pub const GPT5_2_CHAT_LATEST: &str = "gpt-5.2-chat-latest";

/// Maximum-capability GPT-5.2 variant.
pub const GPT5_2_PRO: &str = "gpt-5.2-pro";

/// GPT-5.1 model (kept for compatibility).
pub const GPT5_1: &str = "gpt-5.1";

/// GPT-5.1 Codex Max model for specialized coding workflows.
pub const GPT5_1_CODEX_MAX: &str = "gpt-5.1-codex-max";

/// Original GPT-5 model identifier (kept for compatibility).
pub const GPT5: &str = "gpt-5";

/// Smaller, cheaper GPT-5 variant with excellent general performance.
pub const GPT5_MINI: &str = "gpt-5-mini";

/// Smallest GPT-5 variant optimized for speed & cost.
pub const GPT5_NANO: &str = "gpt-5-nano";

/// Dynamic alias pointing to the latest GPT-5 chat model.
pub const GPT5_CHAT_LATEST: &str = GPT5_2_CHAT_LATEST;

// ============================================================
// 2. GPT-4.1 FAMILY (High-quality non-reasoning models)
// ============================================================

/// Strong general model without deep reasoning overhead.
pub const GPT41: &str = "gpt-4.1";

/// Smaller, faster 4.1 model for chat & apps.
pub const GPT41_MINI: &str = "gpt-4.1-mini";

/// Lowest-cost 4.1 model.
pub const GPT41_NANO: &str = "gpt-4.1-nano";

// ============================================================
// 3. GPT-4o FAMILY (Multimodal, fast, balanced)
// ============================================================

/// Multimodal flagship (text+vision, fast and strong).
pub const GPT4O: &str = "gpt-4o";

/// Cheapest small multimodal model, great for apps.
pub const GPT4O_MINI: &str = "gpt-4o-mini";

/// Dynamic alias used by `ChatGPT` for “best available 4o”.
///
/// `OpenAI` recommends using non-ChatGPT model identifiers for API usage.
pub const CHATGPT_4O_LATEST: &str = "chatgpt-4o-latest";

// ============================================================
// 4. REASONING MODELS (Long-thinking / chain-of-thought)
// ============================================================

/// High-quality reasoning model (deep thinking).
pub const O1: &str = "o1";

/// Fast & light reasoning model.
pub const O1_MINI: &str = "o1-mini";

/// Maximum compute reasoning model (slow but extremely accurate).
pub const O3_PRO: &str = "o3-pro";

/// Cheaper, smaller o3 model.
pub const O3_MINI: &str = "o3-mini";

// ============================================================
// 5. DEEP RESEARCH MODELS (Autonomous research agents)
// ============================================================

/// Advanced deep research model (internet research).
pub const O3_DEEP_RESEARCH: &str = "o3-deep-research";

/// Smaller deep-research model for lightweight agent tasks.
pub const O4_MINI_DEEP_RESEARCH: &str = "o4-mini-deep-research";

// ============================================================
// 6. EMBEDDING MODELS
// ============================================================

/// Small + inexpensive embedding model (1536-dim).
pub const EMBEDDING_SMALL: &str = "text-embedding-3-small";

/// High-accuracy embedding model (3072-dim).
pub const EMBEDDING_LARGE: &str = "text-embedding-3-large";

/// Legacy embedding model (kept for backward compatibility).
pub const EMBEDDING_ADA002: &str = "text-embedding-ada-002";

// ============================================================
// 7. IMAGE GENERATION MODELS
// ============================================================

/// Latest `OpenAI` image model (highest quality).
pub const IMAGE_GPT_1_5: &str = "gpt-image-1.5";

/// Previous `OpenAI` image model (kept for compatibility).
pub const IMAGE_GPT: &str = "gpt-image-1";

/// Fast, smaller image model.
pub const IMAGE_GPT_MINI: &str = "gpt-image-1-mini";

/// DALL·E 3 model (still supported).
pub const DALLE3: &str = "dall-e-3";

// ============================================================
// 8. TEXT-TO-SPEECH (TTS) MODELS
// ============================================================

/// Recommended TTS model (fast, high quality).
pub const TTS_GPT4O_MINI: &str = "gpt-4o-mini-tts";

/// Higher-fidelity TTS model.
pub const TTS_1_HD: &str = "tts-1-hd";

/// Legacy TTS model identifier kept for compatibility.
pub const TTS_GPT4O: &str = "gpt-4o-tts";

/// Legacy Whisper TTS model.
pub const TTS_1: &str = "tts-1";

// ============================================================
// 9. SPEECH-TO-TEXT (Transcription)
// ============================================================

/// Main modern transcription model.
pub const STT_GPT4O: &str = "gpt-4o-transcribe";

/// Transcription + speaker diarization.
pub const STT_GPT4O_DIARIZE: &str = "gpt-4o-transcribe-diarize";

/// Fast small transcription model.
pub const STT_GPT4O_MINI: &str = "gpt-4o-mini-transcribe";

/// Legacy Whisper speech-to-text model.
pub const WHISPER_1: &str = "whisper-1";

// ============================================================
// 10. MODERATION (Safety Classification)
// ============================================================

/// Latest and recommended `OpenAI` moderation model.
pub const MODERATION_LATEST: &str = "omni-moderation-latest";

/// Legacy text-only moderation model.
pub const MODERATION_TEXT: &str = "text-moderation-latest";
