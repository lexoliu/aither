use aither_core::{
    LanguageModel, Result,
    llm::{Message, model::Parameters, tool::Tools},
};
use anyhow::Context;
use futures_lite::StreamExt;

/// Rolling memory that keeps both long-term summaries and the most recent messages.
#[derive(Debug, Clone, Default)]
pub struct ConversationMemory {
    summaries: Vec<Message>,
    recent: Vec<Message>,
}

impl ConversationMemory {
    pub fn push(&mut self, message: Message) {
        self.recent.push(message);
    }

    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        for message in messages {
            self.push(message);
        }
    }

    pub fn push_summary(&mut self, summary: Message) {
        self.summaries.push(summary);
    }

    pub fn len_recent(&self) -> usize {
        self.recent.len()
    }

    pub fn all(&self) -> Vec<Message> {
        self.summaries
            .iter()
            .cloned()
            .chain(self.recent.iter().cloned())
            .collect()
    }

    pub fn last(&self) -> Option<&Message> {
        self.recent.last().or_else(|| self.summaries.last())
    }

    pub fn is_empty(&self) -> bool {
        self.summaries.is_empty() && self.recent.is_empty()
    }

    pub fn drain_oldest(&mut self, keep: usize) -> Vec<Message> {
        if keep >= self.recent.len() {
            return Vec::new();
        }
        self.recent.drain(..self.recent.len() - keep).collect()
    }

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
    SlidingWindow { max_messages: usize },
    /// Summarize older context while keeping the last `retain_recent` messages verbatim.
    Summarize {
        max_messages: usize,
        retain_recent: usize,
        instructions: String,
    },
}

impl ContextStrategy {
    fn clamp_parameters(max_messages: usize, retain_recent: usize) -> (usize, usize) {
        let max_messages = max_messages.max(4);
        let retain_recent = retain_recent.min(max_messages.saturating_sub(1)).max(1);
        (max_messages, retain_recent)
    }

    pub async fn maintain<LLM: LanguageModel>(
        &self,
        llm: &LLM,
        tools: &mut Tools,
        memory: &mut ConversationMemory,
    ) -> Result<()> {
        match self {
            ContextStrategy::Unlimited => Ok(()),
            ContextStrategy::SlidingWindow { max_messages } => {
                memory.drain_oldest(*max_messages);
                Ok(())
            }
            ContextStrategy::Summarize {
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
                let summary_text = summarize(llm, tools, instructions, dialogue).await?;
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

async fn summarize<LLM: LanguageModel>(
    llm: &LLM,
    tools: &mut Tools,
    instructions: &str,
    dialogue: String,
) -> Result<String> {
    let messages = [
        Message::system(
            "You are a memory compressor. Compress the conversation without losing key facts.",
        ),
        Message::user(format!(
            "Instructions: {instructions}\nConversation:\n{dialogue}"
        )),
    ];

    let summary: String = llm
        .generate(&messages, tools, &Parameters::default())
        .await
        .context("Context compression failed")?;

    Ok(summary.trim().to_string())
}
