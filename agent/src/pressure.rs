//! The level machinery memory pressure and disk pressure share: a level is
//! read each tick, and a change of it is a [`Shift`] the agent hears of once.

/// A change of a pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift<L> {
    /// The level rose to `L`.
    Rose(L),
    /// The level fell to `L`, possibly to normal.
    Fell(L),
}

/// Remembers the last level and reports each change of it once.
#[derive(Debug, Clone)]
pub struct LevelTracker<L> {
    level: L,
}

impl<L: Copy + Ord> LevelTracker<L> {
    /// A tracker at `level`, the level of normal.
    pub(crate) const fn new(level: L) -> Self {
        Self { level }
    }

    /// The level last observed.
    pub(crate) const fn level(&self) -> L {
        self.level
    }

    /// Takes the level now; returns the shift it makes, if any.
    pub(crate) fn observe(&mut self, level: L) -> Option<Shift<L>> {
        let shift = match level.cmp(&self.level) {
            std::cmp::Ordering::Greater => Some(Shift::Rose(level)),
            std::cmp::Ordering::Less => Some(Shift::Fell(level)),
            std::cmp::Ordering::Equal => None,
        };
        self.level = level;
        shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_change_is_reported_once() {
        let mut tracker = LevelTracker::new(0);
        assert_eq!(tracker.observe(0), None);
        assert_eq!(tracker.observe(2), Some(Shift::Rose(2)));
        assert_eq!(tracker.observe(2), None);
        assert_eq!(tracker.observe(1), Some(Shift::Fell(1)));
        assert_eq!(tracker.observe(0), Some(Shift::Fell(0)));
    }
}
