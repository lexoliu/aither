//! Model-facing ids: the words a model reads and types back to name what it
//! can address, such as a background job, a stored output or a finished
//! subagent.
//!
//! aither mints none of them itself: each is leased from an [`IdAllocator`]
//! the embedder supplies, and the lease lasts as long as what the word names
//! can still be addressed. An embedder that also names its own things for the
//! model leases their words from the same allocator, so no two live ids share
//! a word. [`WordAllocator`] is the allocator used when none is supplied.

use alloc::{
    boxed::Box,
    collections::BTreeMap,
    string::String,
    sync::{Arc, Weak},
};
use core::{
    fmt,
    ops::Deref,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// Hands out the words that name what a model can address.
///
/// [`lease`](Self::lease) hands out a word no live lease holds, and
/// [`release`](Self::release) takes it back once what it names can no longer
/// be addressed. Callers lease through [`AnyIdAllocator::lease`], whose
/// [`IdLease`] releases the word when its last clone drops, so every lease is
/// released exactly once.
///
/// A word is one or more lowercase ASCII letters: it names files, is written
/// into commands and is recognized in text the model wrote.
pub trait IdAllocator: Send + Sync + 'static {
    /// Leases a word that no live lease holds.
    ///
    /// # Errors
    ///
    /// Returns [`IdsExhausted`] when every word the allocator can hand out is
    /// leased.
    fn lease(&self) -> Result<String, IdsExhausted>;

    /// Ends the lease of `word`, which [`lease`](Self::lease) handed out and
    /// which has not been released since.
    fn release(&self, word: &str);
}

/// Every word an [`IdAllocator`] can hand out is leased.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("every model-facing id is leased")]
pub struct IdsExhausted;

/// A shared [`IdAllocator`].
///
/// Clones lease from the same allocator. The default is a [`WordAllocator`]
/// over its built-in word list.
#[derive(Clone)]
pub struct AnyIdAllocator(Arc<dyn IdAllocator>);

impl AnyIdAllocator {
    /// Shares `allocator`.
    pub fn new(allocator: impl IdAllocator) -> Self {
        Self(Arc::new(allocator))
    }

    /// Leases a word that no live lease holds; it is released when the last
    /// clone of the returned lease drops.
    ///
    /// # Errors
    ///
    /// Returns [`IdsExhausted`] when every word is leased.
    ///
    /// # Panics
    ///
    /// Panics when the allocator hands out a word that is not one or more
    /// lowercase ASCII letters.
    pub fn lease(&self) -> Result<IdLease, IdsExhausted> {
        let word = self.0.lease()?;
        assert!(
            is_word(&word),
            "the id allocator handed out '{word}', which is not one or more lowercase ASCII letters"
        );
        Ok(IdLease(Arc::new(Held {
            word,
            allocator: self.clone(),
        })))
    }
}

/// Whether `word` has the shape every model-facing id has.
fn is_word(word: &str) -> bool {
    !word.is_empty() && word.bytes().all(|byte| byte.is_ascii_lowercase())
}

impl Default for AnyIdAllocator {
    fn default() -> Self {
        Self::new(WordAllocator::default())
    }
}

impl fmt::Debug for AnyIdAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyIdAllocator").finish_non_exhaustive()
    }
}

/// A word leased from an [`AnyIdAllocator`].
///
/// Clones share the lease, which ends when the last of them drops: whatever
/// can still address the named thing holds a clone.
#[derive(Clone)]
pub struct IdLease(Arc<Held>);

/// The word of a lease and the allocator it goes back to.
struct Held {
    word: String,
    allocator: AnyIdAllocator,
}

impl Drop for Held {
    fn drop(&mut self) {
        self.allocator.0.release(&self.word);
    }
}

impl IdLease {
    /// The leased word.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0.word
    }

    /// Whether `other` is a clone of this lease, not merely a lease of the
    /// same word: a word released and leased again names something else.
    #[must_use]
    pub fn same_lease(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// A reference to this lease that does not keep the word leased.
    #[must_use]
    pub fn downgrade(&self) -> WeakIdLease {
        WeakIdLease(Arc::downgrade(&self.0))
    }
}

/// A reference to an [`IdLease`] that does not keep its word leased, for
/// what must still tell whether the word names the same thing later.
#[derive(Clone)]
pub struct WeakIdLease(Weak<Held>);

impl WeakIdLease {
    /// The lease, while a clone of it still holds the word; `None` once the
    /// word went back to its allocator, after which it may name something
    /// else.
    #[must_use]
    pub fn upgrade(&self) -> Option<IdLease> {
        self.0.upgrade().map(IdLease)
    }

    /// Whether this refers to `lease` itself, not merely to a lease of the
    /// same word.
    #[must_use]
    pub fn refers_to(&self, lease: &IdLease) -> bool {
        // The weak reference keeps the lease's allocation, so no later lease
        // can share its address.
        core::ptr::eq(self.0.as_ptr(), Arc::as_ptr(&lease.0))
    }
}

impl fmt::Debug for WeakIdLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WeakIdLease")
    }
}

impl Deref for IdLease {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for IdLease {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for IdLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for IdLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IdLease").field(&self.as_str()).finish()
    }
}

/// The words of the default [`WordAllocator`]: the BIP-39 English word list
/// (CC0), 2048 common English words, SHA-256
/// `2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda`.
const ENGLISH_WORDS: &str = include_str!("ids/english.txt");

/// An [`IdAllocator`] over a fixed list of words.
///
/// It hands the words out in list order, moving on past each one it leases,
/// so a word just released is the last to come back. [`Default`] uses a
/// built-in list of 2048 common English words.
pub struct WordAllocator {
    words: Box<[Box<str>]>,
    /// Each word's position in `words`.
    positions: BTreeMap<Box<str>, usize>,
    leased: Box<[AtomicBool]>,
    /// Where the next lease starts looking.
    cursor: AtomicUsize,
}

impl WordAllocator {
    /// An allocator handing out `words`.
    ///
    /// # Panics
    ///
    /// Panics when `words` is empty, holds one word twice, or holds a word
    /// that is not one or more lowercase ASCII letters.
    pub fn new<W: Into<Box<str>>>(words: impl IntoIterator<Item = W>) -> Self {
        let words: Box<[Box<str>]> = words.into_iter().map(Into::into).collect();
        assert!(!words.is_empty(), "a word allocator needs words");
        let mut positions = BTreeMap::new();
        for (position, word) in words.iter().enumerate() {
            assert!(
                is_word(word),
                "'{word}' is not one or more lowercase ASCII letters"
            );
            assert!(
                positions.insert(word.clone(), position).is_none(),
                "the word '{word}' is listed twice"
            );
        }
        let leased = words.iter().map(|_| AtomicBool::new(false)).collect();
        Self {
            words,
            positions,
            leased,
            cursor: AtomicUsize::new(0),
        }
    }
}

impl Default for WordAllocator {
    fn default() -> Self {
        Self::new(ENGLISH_WORDS.lines())
    }
}

impl fmt::Debug for WordAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WordAllocator")
            .field("words", &self.words.len())
            .finish_non_exhaustive()
    }
}

impl IdAllocator for WordAllocator {
    fn lease(&self) -> Result<String, IdsExhausted> {
        let count = self.words.len();
        for _ in 0..count {
            let position = self.cursor.fetch_add(1, Ordering::Relaxed) % count;
            if !self.leased[position].swap(true, Ordering::AcqRel) {
                return Ok(String::from(&*self.words[position]));
            }
        }
        Err(IdsExhausted)
    }

    fn release(&self, word: &str) {
        let position = *self
            .positions
            .get(word)
            .unwrap_or_else(|| panic!("released '{word}', which this allocator never leased"));
        assert!(
            self.leased[position].swap(false, Ordering::AcqRel),
            "released '{word}', which is not leased"
        );
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn the_default_list_is_single_lowercase_words() {
        assert_eq!(ENGLISH_WORDS.lines().count(), 2048);
        // Building it checks every word's shape and that none is listed twice.
        let _ = WordAllocator::default();
    }

    #[test]
    #[should_panic(expected = "is not one or more lowercase ASCII letters")]
    fn a_word_of_another_shape_is_refused() {
        let _ = WordAllocator::new(["river", "task-1"]);
    }

    /// An allocator that hands out a word of the wrong shape.
    struct Numbered;

    impl IdAllocator for Numbered {
        fn lease(&self) -> Result<String, IdsExhausted> {
            Ok("7".to_string())
        }

        fn release(&self, _word: &str) {}
    }

    #[test]
    #[should_panic(expected = "handed out '7'")]
    fn an_embedder_word_of_another_shape_fails_the_lease() {
        let _ = AnyIdAllocator::new(Numbered).lease();
    }

    #[test]
    fn live_leases_never_share_a_word_and_a_released_word_comes_back() {
        let ids = AnyIdAllocator::new(WordAllocator::new(["river", "stone"]));
        let river = ids.lease().expect("a free word");
        let stone = ids.lease().expect("a free word");
        assert_eq!((river.as_str(), stone.as_str()), ("river", "stone"));
        assert_eq!(ids.lease().unwrap_err(), IdsExhausted);

        drop(river);
        assert_eq!(
            ids.lease().expect("river is free again").to_string(),
            "river"
        );
    }

    #[test]
    fn a_lease_ends_with_its_last_clone() {
        let ids = AnyIdAllocator::new(WordAllocator::new(["river"]));
        let lease = ids.lease().expect("a free word");
        let held = lease.clone();
        drop(lease);
        assert_eq!(
            ids.lease().unwrap_err(),
            IdsExhausted,
            "a clone still holds it"
        );
        drop(held);
        assert_eq!(
            &*ids.lease().expect("released with the last clone"),
            "river"
        );
    }

    #[test]
    fn only_clones_are_the_same_lease() {
        let ids = AnyIdAllocator::new(WordAllocator::new(["river"]));
        let lease = ids.lease().expect("a free word");
        assert!(lease.same_lease(&lease.clone()));
        drop(lease);
        let again = ids.lease().expect("river is free again");
        let other = AnyIdAllocator::new(WordAllocator::new(["river"]))
            .lease()
            .expect("a free word");
        assert_eq!(again.as_str(), other.as_str());
        assert!(!again.same_lease(&other), "a word alone is not the lease");

        let weak = again.downgrade();
        assert!(weak.refers_to(&again));
        assert!(!weak.refers_to(&other));
        drop(again);
        let after = ids.lease().expect("river is free again");
        assert!(
            !weak.refers_to(&after),
            "a released lease is not its successor"
        );
    }

    #[test]
    fn a_weak_lease_upgrades_only_while_the_word_is_held() {
        let ids = AnyIdAllocator::new(WordAllocator::new(["river"]));
        let lease = ids.lease().expect("a free word");
        let weak = lease.downgrade();
        assert_eq!(weak.upgrade().as_deref(), Some("river"));
        drop(lease);
        assert!(weak.upgrade().is_none(), "the word went back");
        let again = ids.lease().expect("river is free again");
        assert!(
            weak.upgrade().is_none(),
            "a word leased again names something else: {again}"
        );
    }

    #[test]
    #[should_panic(expected = "is listed twice")]
    fn a_word_listed_twice_is_refused() {
        let _ = WordAllocator::new(["river", "river"]);
    }

    #[test]
    #[should_panic(expected = "which is not leased")]
    fn releasing_a_free_word_is_a_bug() {
        WordAllocator::new(["river"]).release("river");
    }
}
