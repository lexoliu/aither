use aither_core::{
    LanguageModel, Result,
    llm::{Message, oneshot, tool::Tools},
};
use anyhow::Context;

/// Rolling memory that keeps both long-term summaries and the most recent messages.
#[derive(Debug, Clone, Default)]
pub struct ConversationMemory {
    summaries: Vec<Message>,
    recent: Vec<Message>,
}

impl ConversationMemory {
    /// Adds a new message to the recent conversation history.
    pub fn push(&mut self, message: Message) {
        self.recent.push(message);
    }

    /// Extends the recent conversation history with multiple messages.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        for message in messages {
            self.push(message);
        }
    }

    /// Adds a new summary message to the long-term summaries.
    pub fn push_summary(&mut self, summary: Message) {
        self.summaries.push(summary);
    }

    /// Returns the number of recent messages stored.
    #[must_use]
    pub const fn len_recent(&self) -> usize {
        self.recent.len()
    }

    /// Returns all messages in the memory, combining summaries and recent messages.
    #[must_use]
    pub fn all(&self) -> Vec<Message> {
        self.summaries
            .iter()
            .cloned()
            .chain(self.recent.iter().cloned())
            .collect()
    }

    /// Returns the last message in the memory, prioritizing recent messages.
    #[must_use]
    pub fn last(&self) -> Option<&Message> {
        self.recent.last().or_else(|| self.summaries.last())
    }

    /// Checks if the memory is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.summaries.is_empty() && self.recent.is_empty()
    }

    /// Drains the oldest messages from recent history, keeping only the specified number.
    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        if keep >= self.recent.len() {
            return Vec::new();
        }
        self.recent.drain(..self.recent.len() - keep).collect()
    }

    /// Clears all messages from memory.
    pub fn clear(&mut self) {
        self.summaries.clear();
        self.recent.clear();
    }
}

/// Strategy describing how the conversation should be compressed once the context grows too large.
#[derive(Debug, Clone)]
pub enum ContextStrategy {
    /// Never compress. Good for short-lived conversations or models with very large context windows.
    Unlimited,
    /// Keep a hard sliding window of the most recent `max_messages` exchanges.
    SlidingWindow {
        /// Maximum number of messages to retain in the sliding window.
        max_messages: usize,
    },
    /// Summarize older context while keeping the last `retain_recent` messages verbatim.
    Summarize {
        /// Maximum number of messages before compression is triggered.
        max_messages: usize,
        /// Number of recent messages to keep verbatim after summarization.
        retain_recent: usize,
        /// Instructions for how to summarize the conversation.
        instructions: String,
    },
}

impl ContextStrategy {
    fn clamp_parameters(max_messages: usize, retain_recent: usize) -> (usize, usize) {
        let max_messages = max_messages.max(4);
        let retain_recent = retain_recent.min(max_messages.saturating_sub(1)).max(1);
        (max_messages, retain_recent)
    }

    /// Maintains the conversation memory according to the strategy, compressing if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if summarization fails when using the `Summarize` strategy.
    pub async fn maintain<LLM: LanguageModel>(
        &self,
        llm: &LLM,
        _tools: &mut Tools,
        memory: &mut ConversationMemory,
    ) -> Result<()> {
        match self {
            Self::Unlimited => Ok(()),
            Self::SlidingWindow { max_messages } => {
                memory.drain_oldest(*max_messages);
                Ok(())
            }
            Self::Summarize {
                max_messages,
                retain_recent,
                instructions,
            } => {
                let (max_messages, retain_recent) =
                    Self::clamp_parameters(*max_messages, *retain_recent);

                if memory.len_recent() <= max_messages {
                    return Ok(());
                }

                let retained = retain_recent.min(max_messages);
                let overflow = memory.drain_oldest(retained);
                if overflow.is_empty() {
                    return Ok(());
                }

                let dialogue = format_messages(&overflow);
                let summary_text = compress(llm, instructions, dialogue).await?;
                let summary_message =
                    Message::system(format!("Compressed context:\n{summary_text}"));
                memory.push_summary(summary_message);
                Ok(())
            }
        }
    }
}

fn format_messages(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|msg| format!("{:?}: {}", msg.role(), msg.content()))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn compress<LLM: LanguageModel>(
    llm: &LLM,
    instructions: &str,
    dialogue: String,
) -> Result<String> {
    let system = if instructions.is_empty() {
        "You are a memory compressor. Compress the conversation without losing key facts.".into()
    } else {
        format!(
            "You are a memory compressor. Compress the conversation without losing key facts. \
             Follow these extra instructions: {instructions}"
        )
    };

    let summary: String = llm
        .generate(oneshot(system, dialogue))
        .await
        .context("Context compression failed")?;

    Ok(summary.trim().to_string())
}
