//! The hash chain of a request's prompt prefix.
//!
//! A provider's prompt cache serves the longest prefix of a request it has
//! seen before. When a request reads less from cache than the previous one
//! had cached, the question is *where* the prefix changed — the tool
//! definitions, the system prompt, or a message in the conversation. A
//! [`PrefixChain`] answers it: one link per block, in the order providers
//! cache them, each link hashing every block up to and including its own.
//! Two chains agree on a link exactly when they agree on everything before
//! it, so the first differing link names the first changed block.
//!
//! The links are SHA-256 based so a host can persist a chain and compare it
//! against a request made by a later build of the program.

use std::io;

use aither_core::llm::tool::ToolDefinition;
use aither_core::llm::{Message, Role};
use sha2::{Digest, Sha256};

/// One block of a request's prompt prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PrefixBlock {
    /// The tool definitions, which providers cache ahead of everything else.
    Tools,
    /// A message of the request.
    Message {
        /// Position of the message in the request.
        index: usize,
        /// Who the message is from.
        role: Role,
    },
}

/// The hash chain of one request's prompt prefix.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefixChain {
    /// The block each link ends with.
    blocks: Vec<PrefixBlock>,
    /// The cumulative hash of every block up to and including the same index.
    links: Vec<u64>,
    /// How many leading blocks the request asks the provider to cache: all
    /// but the live tail, which is rebuilt for every request.
    cached: usize,
}

impl PrefixChain {
    /// The chain of a request carrying `definitions` and `messages`.
    ///
    /// # Panics
    ///
    /// Panics when a message or tool definition cannot be serialized, which
    /// would mean the request itself cannot be sent.
    #[must_use]
    pub fn of(definitions: &[ToolDefinition], messages: &[Message]) -> Self {
        let mut hasher = HashWriter(Sha256::new());
        let mut blocks = Vec::with_capacity(messages.len() + 1);
        let mut links = Vec::with_capacity(messages.len() + 1);
        serde_json::to_writer(&mut hasher, definitions).expect("tool definitions serialize");
        blocks.push(PrefixBlock::Tools);
        links.push(hasher.link());
        for (index, message) in messages.iter().enumerate() {
            serde_json::to_writer(&mut hasher, message).expect("messages serialize");
            blocks.push(PrefixBlock::Message {
                index,
                role: message.role(),
            });
            links.push(hasher.link());
        }
        let cached = 1 + super::cache::stable_conversation_end(messages).map_or(0, |end| end + 1);
        Self {
            blocks,
            links,
            cached,
        }
    }

    /// The first block of `previous`'s cached prefix that this request does
    /// not repeat byte for byte, or `None` when it repeats all of it.
    ///
    /// A block this request no longer has at all counts as changed.
    #[must_use]
    pub fn first_divergence(&self, previous: &Self) -> Option<PrefixBlock> {
        (0..previous.cached)
            .find(|&index| self.links.get(index) != previous.links.get(index))
            .map(|index| {
                self.blocks
                    .get(index)
                    .copied()
                    .unwrap_or(previous.blocks[index])
            })
    }

    /// The number of blocks in the chain, the tool definitions included.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.links.len()
    }

    /// Whether the chain has no blocks; never true for a chain built by
    /// [`PrefixChain::of`], which always has the tool definitions block.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// How many leading blocks the request asks the provider to cache.
    #[must_use]
    pub const fn cached_blocks(&self) -> usize {
        self.cached
    }
}

/// What rewrote the prompt prefix since the agent's previous request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefixRewrites {
    /// A compaction replaced the conversation with its handoff.
    pub compacted: bool,
    /// The window was reassembled: staged system-block updates and
    /// superseded tool results folded in, or alerts cleaned for recovery.
    pub reassembled: bool,
}

/// The prompt prefix of a request about to be sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptPrefix {
    /// The hash chain of the request's prefix.
    pub chain: PrefixChain,
    /// What rewrote the prefix since this agent's previous request.
    pub rewrites: PrefixRewrites,
    /// Whether this is the agent's first request, so whatever the host sent
    /// before it came from another agent.
    pub first_request: bool,
}

/// Feeds serialized bytes straight into the hasher, without a buffer.
struct HashWriter(Sha256);

impl HashWriter {
    /// The chain link for everything written so far.
    fn link(&self) -> u64 {
        let digest = self.0.clone().finalize();
        let head = digest
            .first_chunk::<8>()
            .expect("a SHA-256 digest is 32 bytes");
        u64::from_le_bytes(*head)
    }
}

impl io::Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation() -> Vec<Message> {
        vec![
            Message::system("prefix"),
            Message::user("hello"),
            Message::assistant("hi"),
            Message::system("live tail"),
        ]
    }

    fn tool(name: &'static str) -> ToolDefinition {
        ToolDefinition::from_parts(
            name.into(),
            "runs a command".into(),
            serde_json::json!({"type": "object"}),
        )
        .expect("an object schema is valid")
    }

    #[test]
    fn an_unchanged_prefix_with_a_new_tail_does_not_diverge() {
        let tools = [tool("terminal")];
        let first = PrefixChain::of(&tools, &conversation());
        let mut next = conversation();
        next.pop();
        next.extend([Message::user("again"), Message::system("new tail")]);
        let second = PrefixChain::of(&tools, &next);
        assert_eq!(second.first_divergence(&first), None);
        assert_eq!(
            first.cached_blocks(),
            4,
            "tools plus every message but the tail"
        );
    }

    #[test]
    fn a_changed_tool_set_diverges_at_the_tools() {
        let first = PrefixChain::of(&[tool("terminal")], &conversation());
        let second = PrefixChain::of(&[tool("terminal"), tool("monitor")], &conversation());
        assert_eq!(second.first_divergence(&first), Some(PrefixBlock::Tools));
    }

    #[test]
    fn a_changed_message_names_its_index_and_role() {
        let tools = [tool("terminal")];
        let first = PrefixChain::of(&tools, &conversation());
        let mut edited = conversation();
        edited[1] = Message::user("hello there");
        let second = PrefixChain::of(&tools, &edited);
        assert_eq!(
            second.first_divergence(&first),
            Some(PrefixBlock::Message {
                index: 1,
                role: Role::User
            })
        );
    }

    #[test]
    fn a_shorter_request_diverges_at_the_first_missing_block() {
        let tools = [tool("terminal")];
        let first = PrefixChain::of(&tools, &conversation());
        let second = PrefixChain::of(&tools, &[Message::system("prefix")]);
        assert_eq!(
            second.first_divergence(&first),
            Some(PrefixBlock::Message {
                index: 1,
                role: Role::User
            })
        );
    }

    #[test]
    fn a_chain_survives_serialization() {
        let chain = PrefixChain::of(&[tool("terminal")], &conversation());
        let text = serde_json::to_string(&chain).expect("serializes");
        let back: PrefixChain = serde_json::from_str(&text).expect("deserializes");
        assert_eq!(back, chain);
        assert_eq!(back.len(), 5);
    }
}
