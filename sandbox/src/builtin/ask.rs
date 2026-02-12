//! Ask command - query a fast LLM about piped content.
//!
//! Supports various input formats: text, markdown, images, PDFs.
//!
//! # Usage
//!
//! ```bash
//! cat large_file.txt | ask "summarize this"
//! websearch "rust async" | ask "what are the key patterns?"
//! ```

use std::borrow::Cow;

use aither_core::LanguageModel;
use aither_core::llm::{Event, LLMRequest, Message, Tool, ToolOutput};
use futures_lite::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Query a fast LLM about piped content.
///
/// Reads content from stdin and processes it with a fast language model.
/// Only the model's response is returned to your context - the input content
/// is not included in the response, saving context space for large inputs.
///
/// This is useful when you need to analyze, summarize, or extract information
/// from data without loading the full content into your context window.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AskArgs {
    /// The question or instruction about the input content.
    pub prompt: String,

    /// Input content (from stdin/pipe).
    #[serde(default)]
    pub input: String,
}

/// The ask command tool.
#[derive(Debug)]
pub struct AskCommand<LLM> {
    llm: LLM,
}

impl<LLM> AskCommand<LLM> {
    /// Creates a new ask command with the given LLM.
    pub const fn new(llm: LLM) -> Self {
        Self { llm }
    }
}

impl<LLM: LanguageModel> Tool for AskCommand<LLM> {
    fn name(&self) -> Cow<'static, str> {
        "ask".into()
    }

    type Arguments = AskArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        if args.input.is_empty() {
            return Ok(ToolOutput::text("No input provided. Pipe content to ask."));
        }

        // Build the prompt with context
        let user_content = format!("<context>\n{}\n</context>\n\n{}", args.input, args.prompt);

        let request = LLMRequest::new(vec![Message::user(user_content)]);
        let response = self.llm.respond(request);

        // Collect the response text
        futures_lite::pin!(response);
        let mut result = String::new();
        while let Some(event) = response.next().await {
            let event = event.map_err(|e| anyhow::anyhow!("{e}"))?;
            if let Event::Text(text) = event {
                result.push_str(&text);
            }
        }

        Ok(ToolOutput::text(result))
    }
}
