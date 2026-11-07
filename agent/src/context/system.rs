//! The system prompt's layout: which blocks it holds, in what order, and how
//! external content is kept from posing as instructions.
//!
//! The system prompt has two regions, rendered in this order:
//!
//! 1. **instructions** — the agent's own prompt: the base system prompt, the
//!    persona, the built-in rules and whatever the host adds;
//! 2. **external content** — text the host did not write as instructions
//!    (memory, a repository's `AGENTS.md`, fetched pages, files), each block
//!    inside a provenance-labelled boundary its content cannot close, behind
//!    one notice that such content is data.
//!
//! The host chooses the order of each region with a [`SystemLayout`]; a block
//! renders at its slot's position whenever it was inserted, so the prefix is
//! a function of the layout and the contents alone. The region a block
//! belongs to is part of its slot's type, which is what keeps external
//! content after every instruction.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::marker::PhantomData;

use askama::Template;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A block of text wrapped in the one delimiting tag the model sees.
#[derive(Template)]
#[template(path = "tagged_block.txt", escape = "none")]
pub(crate) struct TaggedBlockTemplate<'a> {
    pub(crate) tag: &'a str,
    pub(crate) content: &'a str,
}

impl TaggedBlockTemplate<'_> {
    pub(crate) fn into_text(self) -> String {
        self.render().expect("tagged block template must render")
    }
}

mod sealed {
    pub trait Sealed {}
}

/// The kind of a system block, which fixes the region it renders in and the
/// content it holds.
pub trait BlockKind: sealed::Sealed + Debug + Clone + PartialEq + Eq + Ord {
    /// What a block of this kind holds.
    type Content: Debug + Clone + PartialEq + Serialize + DeserializeOwned;

    /// The region's name, for diagnostics.
    const REGION: &'static str;

    /// The text of `content` the model reads, which a change is diffed on.
    fn text(content: &Self::Content) -> &str;
}

/// Instruction blocks: the agent's own prompt, rendered first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instruction;

impl sealed::Sealed for Instruction {}

impl BlockKind for Instruction {
    type Content = String;
    const REGION: &'static str = "instruction";

    fn text(content: &Self::Content) -> &str {
        content
    }
}

/// External-content blocks: bounded data, rendered after every instruction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct External;

impl sealed::Sealed for External {}

impl BlockKind for External {
    type Content = ExternalContent;
    const REGION: &'static str = "external";

    fn text(content: &Self::Content) -> &str {
        &content.content
    }
}

/// Blocks live sources render: placed after the system prompt and before the
/// handoff, in the cacheable prefix (see [`LiveSource`](crate::LiveSource)).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sourced;

impl sealed::Sealed for Sourced {}

impl BlockKind for Sourced {
    type Content = String;
    const REGION: &'static str = "live source";

    fn text(content: &Self::Content) -> &str {
        content
    }
}

/// The identity of a system block within its region.
///
/// The slot's type names its region, so a block can only be inserted where
/// the layout places blocks of its kind.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Slot<K> {
    name: Cow<'static, str>,
    #[serde(skip)]
    kind: PhantomData<fn() -> K>,
}

/// The slot of an instruction block.
pub type InstructionSlot = Slot<Instruction>;

/// The slot of an external-content block.
pub type ExternalSlot = Slot<External>;

/// The slot of a live source's block, named by its
/// [`TAG`](crate::LiveSource::TAG).
pub type SourcedSlot = Slot<Sourced>;

impl<K> Slot<K> {
    /// A slot named `name`. The name is also the tag a tagged block renders
    /// with.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            kind: PhantomData,
        }
    }

    /// The slot's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl InstructionSlot {
    /// The base system prompt ([`AgentConfig::system_prompt`](crate::AgentConfig::system_prompt)).
    pub const BASE_SYSTEM: Self = Self::new("base_system");
    /// The persona overlay ([`AgentConfig::persona_prompt`](crate::AgentConfig::persona_prompt)).
    pub const PERSONA: Self = Self::new("persona");
    /// The catalog of commands prepared for the terminal.
    pub const PREPARED_TOOLS: Self = Self::new("prepared_tools");
    /// The shell runtime the terminal executes on.
    pub const SHELL_RUNTIME: Self = Self::new("shell_runtime");
    /// How a coding agent discovers repository guidance.
    pub const WORKSPACE_FACTS: Self = Self::new("workspace_facts");
    /// The knowledge cutoff and how to check the current time.
    pub const KNOWLEDGE_AND_TIME: Self = Self::new("knowledge_and_time");
    /// The filesystem permission model.
    pub const PERMISSIONS: Self = Self::new("permissions");
    /// The table of the CLI commands `terminal` runs, which a terminal agent
    /// keeps current as live content (see
    /// [`Context::update_live`](crate::Context::update_live)).
    pub const COMMANDS: Self = Self::new("commands");
    /// The communication and working languages, present only while a
    /// communication language is enforced.
    #[cfg(feature = "languages")]
    pub const LANGUAGES: Self = Self::new("languages");

    /// Every built-in instruction slot, in the default layout's order.
    #[cfg(not(feature = "languages"))]
    pub const BUILTIN: [Self; 8] = [
        Self::BASE_SYSTEM,
        Self::PERSONA,
        Self::PREPARED_TOOLS,
        Self::SHELL_RUNTIME,
        Self::WORKSPACE_FACTS,
        Self::KNOWLEDGE_AND_TIME,
        Self::PERMISSIONS,
        Self::COMMANDS,
    ];

    /// Every built-in instruction slot, in the default layout's order.
    #[cfg(feature = "languages")]
    pub const BUILTIN: [Self; 9] = [
        Self::BASE_SYSTEM,
        Self::PERSONA,
        Self::PREPARED_TOOLS,
        Self::SHELL_RUNTIME,
        Self::WORKSPACE_FACTS,
        Self::KNOWLEDGE_AND_TIME,
        Self::PERMISSIONS,
        Self::COMMANDS,
        Self::LANGUAGES,
    ];
}

/// The order the host gives the system prompt's blocks.
///
/// Every block the prompt holds, the built-in ones included, has a slot here;
/// inserting a block whose slot is not laid out is a mistake in the host and
/// panics. Instructions render first in their listed order, then the
/// external-content notice and the external blocks in theirs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemLayout {
    instructions: Vec<InstructionSlot>,
    external: Vec<ExternalSlot>,
}

impl SystemLayout {
    /// A layout rendering `instructions`, then `external`, each in the order
    /// given.
    ///
    /// # Panics
    ///
    /// Panics if a slot is listed twice in its region.
    #[must_use]
    pub fn new(
        instructions: impl IntoIterator<Item = InstructionSlot>,
        external: impl IntoIterator<Item = ExternalSlot>,
    ) -> Self {
        Self {
            instructions: unique_slots(instructions),
            external: unique_slots(external),
        }
    }

    /// The instruction slots in render order.
    #[must_use]
    pub fn instructions(&self) -> &[InstructionSlot] {
        &self.instructions
    }

    /// The external-content slots in render order.
    #[must_use]
    pub fn external(&self) -> &[ExternalSlot] {
        &self.external
    }

    /// The instruction and external-content orders.
    pub(crate) fn into_regions(self) -> (Vec<InstructionSlot>, Vec<ExternalSlot>) {
        (self.instructions, self.external)
    }
}

impl Default for SystemLayout {
    /// The built-in instruction slots and no external content.
    fn default() -> Self {
        Self::new(InstructionSlot::BUILTIN, [])
    }
}

fn unique_slots<K: BlockKind>(slots: impl IntoIterator<Item = Slot<K>>) -> Vec<Slot<K>> {
    let slots: Vec<Slot<K>> = slots.into_iter().collect();
    for (index, slot) in slots.iter().enumerate() {
        assert!(
            !slots[..index].contains(slot),
            "{} slot `{}` is laid out twice",
            K::REGION,
            slot.name()
        );
    }
    slots
}

/// One region of the system prompt: the blocks of one kind, in the order the
/// layout gives them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "K::Content: Serialize",
    deserialize = "K::Content: DeserializeOwned"
))]
pub struct SystemRegion<K: BlockKind> {
    order: Vec<Slot<K>>,
    blocks: BTreeMap<Slot<K>, K::Content>,
    /// Live-content changes applied on the next reassembly: the new content,
    /// or `None` for a block that goes away.
    ///
    /// Rewriting a block while the prefix is cached invalidates it, so a
    /// change is parked here (and announced to the model as an appended diff
    /// alert instead) until reassembly, which rewrites the prefix anyway.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    staged: BTreeMap<Slot<K>, Option<K::Content>>,
}

/// One live block's change, as the model is told it: the text it knew and
/// the text it is to know.
#[derive(Debug)]
pub(crate) struct TextChange {
    pub(crate) before: String,
    pub(crate) after: String,
}

impl<K: BlockKind> SystemRegion<K> {
    pub(crate) const fn with_order(order: Vec<Slot<K>>) -> Self {
        Self {
            order,
            blocks: BTreeMap::new(),
            staged: BTreeMap::new(),
        }
    }

    /// Replaces the region's order.
    ///
    /// # Panics
    ///
    /// Panics if a block already present, or staged, has no slot in `order`.
    pub(crate) fn set_order(&mut self, order: Vec<Slot<K>>) {
        self.order = order;
        for slot in self.blocks.keys().chain(self.staged.keys()) {
            self.assert_placed(slot);
        }
    }

    /// Appends `slot` to the region's order unless it is already placed.
    pub(crate) fn place(&mut self, slot: Slot<K>) {
        if !self.order.contains(&slot) {
            self.order.push(slot);
        }
    }

    fn assert_placed(&self, slot: &Slot<K>) {
        assert!(
            self.order.contains(slot),
            "{} block `{}` has no slot in the system layout",
            K::REGION,
            slot.name()
        );
    }

    /// The region's slots in render order.
    #[must_use]
    pub fn order(&self) -> &[Slot<K>] {
        &self.order
    }

    /// Inserts or replaces the block in `slot`.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    pub fn insert(&mut self, slot: &Slot<K>, content: impl Into<K::Content>) {
        self.assert_placed(slot);
        self.blocks.insert(slot.clone(), content.into());
    }

    /// Removes the block in `slot`, together with any staged replacement.
    pub fn remove(&mut self, slot: &Slot<K>) -> Option<K::Content> {
        self.staged.remove(slot);
        self.blocks.remove(slot)
    }

    /// The block in `slot`.
    #[must_use]
    pub fn get(&self, slot: &Slot<K>) -> Option<&K::Content> {
        self.blocks.get(slot)
    }

    /// Whether `slot` holds a block.
    #[must_use]
    pub fn contains(&self, slot: &Slot<K>) -> bool {
        self.blocks.contains_key(slot)
    }

    /// The number of blocks held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the region holds no block.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// The blocks in render order.
    pub fn iter(&self) -> impl Iterator<Item = (&Slot<K>, &K::Content)> {
        self.order
            .iter()
            .filter_map(|slot| self.blocks.get(slot).map(|content| (slot, content)))
    }

    /// The content the model knows `slot` to hold: the prefix's block with
    /// every announced change applied.
    fn known(&self, slot: &Slot<K>) -> Option<&K::Content> {
        self.staged
            .get(slot)
            .map_or_else(|| self.blocks.get(slot), Option::as_ref)
    }

    /// Brings the live block in `slot` to `desired` (`None` removes it).
    ///
    /// While the prefix is `warm` the block is left as the cache holds it and
    /// the change is staged for the next reassembly; the change is returned
    /// when the text the model reads differs from what it knows, so the
    /// caller announces it. A cold prefix is rewritten at once and nothing
    /// is returned: no request has carried the old version.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    pub(crate) fn change(
        &mut self,
        slot: &Slot<K>,
        desired: Option<K::Content>,
        warm: bool,
    ) -> Option<TextChange> {
        self.assert_placed(slot);
        let known = self.known(slot);
        if known == desired.as_ref() {
            return None;
        }
        if warm {
            let change = TextChange {
                before: known.map_or("", K::text).to_owned(),
                after: desired.as_ref().map_or("", K::text).to_owned(),
            };
            if self.blocks.get(slot) == desired.as_ref() {
                self.staged.remove(slot);
            } else {
                self.staged.insert(slot.clone(), desired);
            }
            return (change.before != change.after).then_some(change);
        }
        self.staged.remove(slot);
        match desired {
            Some(content) => self.blocks.insert(slot.clone(), content),
            None => self.blocks.remove(slot),
        };
        None
    }

    /// Whether any change is waiting for reassembly.
    #[must_use]
    pub fn has_staged(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Folds the staged changes in, returning how many were applied and how
    /// many bytes they freed.
    pub(crate) fn apply_staged(&mut self) -> (usize, usize) {
        let mut applied = 0;
        let mut freed = 0;
        for (slot, content) in std::mem::take(&mut self.staged) {
            let before = self
                .blocks
                .get(&slot)
                .map_or(0, |block| K::text(block).len());
            let after = content.as_ref().map_or(0, |block| K::text(block).len());
            freed += before.saturating_sub(after);
            match content {
                Some(content) => self.blocks.insert(slot, content),
                None => self.blocks.remove(&slot),
            };
            applied += 1;
        }
        (applied, freed)
    }

    /// Drops every block and staged replacement, keeping the order.
    pub(crate) fn clear(&mut self) {
        self.blocks.clear();
        self.staged.clear();
    }
}

impl SystemRegion<Instruction> {
    /// Inserts or replaces the block in `slot`, wrapped in a tag named after
    /// the slot.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    pub fn insert_tagged(&mut self, slot: &InstructionSlot, content: &str) {
        let wrapped = TaggedBlockTemplate {
            tag: slot.name(),
            content,
        }
        .into_text();
        self.insert(slot, wrapped);
    }
}

/// How the model is to treat a block of external content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExternalKind {
    /// Data the model may read and use, never follow.
    Data,
    /// A repository's own rules (`AGENTS.md` and the like): followed as
    /// project rules, subordinate to the system prompt and the user.
    RepositoryInstructions,
}

impl ExternalKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::RepositoryInstructions => "repository-instructions",
        }
    }
}

/// Text the host places in the system prompt without writing it as
/// instructions: memory, a repository's instructions, a fetched page, a file.
///
/// It renders inside a boundary derived from its own bytes, which therefore
/// never occurs in it: nothing it contains can close the block or pose as a
/// block of its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalContent {
    kind: ExternalKind,
    source: String,
    content: String,
}

impl ExternalContent {
    /// Data from `source` (a path, a URL, or a name such as `memory`).
    #[must_use]
    pub fn data(source: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            kind: ExternalKind::Data,
            source: source.into(),
            content: content.into(),
        }
    }

    /// A repository's instructions, read from `source`: followed as project
    /// rules, subordinate to the system prompt and the user.
    #[must_use]
    pub fn repository_instructions(source: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            kind: ExternalKind::RepositoryInstructions,
            source: source.into(),
            content: content.into(),
        }
    }

    /// Where the content came from.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The content itself.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Whether the content is a repository's instructions.
    #[must_use]
    pub const fn is_repository_instructions(&self) -> bool {
        matches!(self.kind, ExternalKind::RepositoryInstructions)
    }

    /// The boundary this block renders inside.
    ///
    /// A hash of the block, so the same block always renders the same bytes
    /// and the prompt-cache prefix survives a rebuild; a candidate that
    /// occurs in the label or the content is skipped, so the boundary never
    /// does.
    fn boundary(&self) -> String {
        boundary(&[self.kind.label(), &self.source, &self.content])
    }
}

/// A boundary that occurs in none of `parts`: hex of a hash of them, rehashed
/// until it is absent, so the same parts always get the same boundary.
pub(crate) fn boundary(parts: &[&str]) -> String {
    let mut attempt: u64 = 0;
    loop {
        let mut hasher = Sha256::new();
        for (index, part) in parts.iter().enumerate() {
            if index > 0 {
                hasher.update([0]);
            }
            hasher.update(part);
        }
        hasher.update(attempt.to_le_bytes());
        let candidate = crate::agent::hex_encode(&hasher.finalize()[..8]);
        if parts.iter().all(|part| !part.contains(&candidate)) {
            return candidate;
        }
        attempt += 1;
    }
}

impl ExternalContent {
    /// The block as the templates read it.
    fn view(&self) -> ExternalBlockView<'_> {
        ExternalBlockView {
            boundary: self.boundary(),
            kind: self.kind.label(),
            source: &self.source,
            content: &self.content,
        }
    }

    /// The content inside its provenance-labelled boundary, for a host that
    /// places it in text of its own — an alert's template, say.
    ///
    /// The closing tag carries a boundary derived from the block's own
    /// bytes, which therefore never occurs in it, exactly as in the system
    /// prompt. The same block always renders the same text. The text does
    /// not repeat the notice that external content is data; see
    /// [`Block::external`](super::Block::external) for a block that does.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render.
    #[must_use]
    pub fn render(&self) -> String {
        ExternalBlockTemplate { block: self.view() }
            .render()
            .expect("external block template must render")
    }

    /// The block with the notice that external content is data before it,
    /// for an alert that arrives where the system prompt has no such notice.
    pub(crate) fn render_with_notice(&self) -> String {
        ExternalAlertTemplate { block: self.view() }
            .render()
            .expect("external alert template must render")
    }
}

/// One external block as the templates read it.
struct ExternalBlockView<'a> {
    boundary: String,
    kind: &'static str,
    source: &'a str,
    content: &'a str,
}

/// One external block, bounded.
#[derive(Template)]
#[template(path = "external_block.txt", escape = "none")]
struct ExternalBlockTemplate<'a> {
    block: ExternalBlockView<'a>,
}

/// One external block, bounded and preceded by the notice that it is data.
#[derive(Template)]
#[template(path = "external_alert.txt", escape = "none")]
struct ExternalAlertTemplate<'a> {
    block: ExternalBlockView<'a>,
}

/// The whole cacheable system prefix.
#[derive(Template)]
#[template(path = "system_prefix.txt", escape = "none")]
struct SystemPrefixTemplate<'a> {
    instructions: Vec<&'a str>,
    external: Vec<ExternalBlockView<'a>>,
}

/// Renders the system prefix: the instructions, then the external notice and
/// blocks. `None` when both regions are empty.
pub(crate) fn render_prefix(
    instructions: &SystemRegion<Instruction>,
    external: &SystemRegion<External>,
) -> Option<String> {
    if instructions.is_empty() && external.is_empty() {
        return None;
    }
    let template = SystemPrefixTemplate {
        instructions: instructions.iter().map(|(_, text)| text.as_str()).collect(),
        external: external.iter().map(|(_, block)| block.view()).collect(),
    };
    Some(
        template
            .render()
            .expect("system prefix template must render"),
    )
}
