use alloc::{string::String, vec::Vec};

use crate::{
    LanguageModel,
    llm::{LLMRequest, Message, Tool, tool::Tools, try_collect},
};

/// A struct representing an Assistant that interacts with a language model (LLM),
/// manages a collection of messages, and provides access to various tools.
///
/// # Type Parameters
/// - `LLM`: A type that implements the `LanguageModel` trait, representing the
///   underlying language model used by the Assistant.
///
/// # Fields
/// - `messages`: A vector of `Message` instances representing the conversation
///   history or context.
/// - `tools`: A collection of tools available to the Assistant for performing
///   various tasks.
/// - `llm`: The language model instance used by the Assistant for generating
///   responses or performing language-related tasks.
#[derive(Debug)]
pub struct Assistant<LLM: LanguageModel> {
    messages: Vec<Message>,
    tools: Tools,
    llm: LLM,
}

impl<LLM: LanguageModel> Assistant<LLM> {
    /// Creates a new `Assistant` instance with the provided language model.
    ///
    /// # Parameters
    /// - `llm`: The language model instance to be used by the assistant.
    ///
    /// # Returns
    /// Returns a new `Assistant` with empty messages and tools.
    #[must_use]
    pub const fn new(llm: LLM) -> Self {
        Self {
            messages: Vec::new(),
            tools: Tools::new(),
            llm,
        }
    }

    /// Adds a system message to the conversation history.
    ///
    /// # Parameters
    /// - `prompt`: The system prompt to add as a message.
    ///
    /// # Returns
    /// Returns the updated Assistant instance with the system message added.
    #[must_use]
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.messages.push(Message::system(prompt.into()));
        self
    }

    /// Registers a tool with the assistant, making it available for use in interactions.
    ///
    /// # Parameters
    /// - `tool`: The tool to register with the assistant.
    ///
    /// # Returns
    /// Returns the updated Assistant instance with the tool registered.
    #[must_use]
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    /// Sends a user message to the assistant, processes it with the language model, and appends the response to the conversation history.
    ///
    /// # Parameters
    /// - `message`: The user message to send to the assistant.
    ///
    /// # Errors
    /// Returns an error if the language model fails to generate a response or if message processing fails.
    pub async fn send(&mut self, message: impl Into<String>) -> anyhow::Result<()> {
        self.messages.push(Message::user(message));
        let request = LLMRequest::new(self.messages.as_slice()).with_tools(&mut self.tools);
        let stream = self.llm.respond(request);

        let response = try_collect(stream).await?;
        self.messages.push(Message::assistant(response));
        Ok(())
    }

    /// Returns a slice of all messages in the conversation history.
    pub const fn messages(&self) -> &[Message] {
        self.messages.as_slice()
    }
}
