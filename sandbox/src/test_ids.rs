//! Id allocators for tests.

#![allow(clippy::missing_panics_doc, clippy::must_use_candidate)]

use std::sync::{Arc, Mutex};

use aither_core::{AnyIdAllocator, IdAllocator, IdLease, IdsExhausted, WordAllocator};

/// A lease on `word` alone, for tests that name a job themselves.
pub fn lease(word: &str) -> IdLease {
    AnyIdAllocator::new(WordAllocator::new([word]))
        .lease()
        .expect("a one-word allocator has its word free")
}

/// The words a [`RecordingIds`] hands out, in order.
const WORDS: [&str; 26] = [
    "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
];

/// An allocator recording every word it leases and every release.
///
/// It hands out the NATO alphabet's words in order, each once, so a test
/// knows which word its `n`th lease got.
#[derive(Clone, Debug, Default)]
pub struct RecordingIds(Arc<Mutex<Record>>);

#[derive(Debug, Default)]
struct Record {
    leased: Vec<String>,
    released: Vec<String>,
}

impl RecordingIds {
    /// Shares this allocator for a terminal or store to lease from.
    pub fn allocator(&self) -> AnyIdAllocator {
        AnyIdAllocator::new(self.clone())
    }

    /// Every word leased, in order.
    pub fn leased(&self) -> Vec<String> {
        self.0.lock().unwrap().leased.clone()
    }

    /// Every word released, in order.
    pub fn released(&self) -> Vec<String> {
        self.0.lock().unwrap().released.clone()
    }

    /// The words leased and not yet released.
    pub fn live(&self) -> Vec<String> {
        let record = self.0.lock().unwrap();
        record
            .leased
            .iter()
            .filter(|word| !record.released.contains(word))
            .cloned()
            .collect()
    }
}

impl IdAllocator for RecordingIds {
    fn lease(&self) -> Result<String, IdsExhausted> {
        let mut record = self.0.lock().unwrap();
        let word = (*WORDS.get(record.leased.len()).ok_or(IdsExhausted)?).to_string();
        record.leased.push(word.clone());
        drop(record);
        Ok(word)
    }

    fn release(&self, word: &str) {
        let mut record = self.0.lock().unwrap();
        assert!(
            record.leased.iter().any(|leased| leased == word),
            "released '{word}', which was never leased"
        );
        assert!(
            !record.released.iter().any(|released| released == word),
            "released '{word}' twice"
        );
        record.released.push(word.to_string());
    }
}
