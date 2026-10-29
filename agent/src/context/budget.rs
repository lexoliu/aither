//! Context window accounting calibrated against the provider's own counts.
//!
//! Character heuristics are the only thing available before a request is
//! sent, and they are wrong by a different factor for prose, code, JSON tool
//! arguments and base64 attachments. Every response, however, comes back with
//! the number of tokens the provider actually charged for the prompt it just
//! read. [`ContextBudget`] keeps that pairing — the characters we assembled
//! and the tokens they turned into — and uses the resulting ratio to size
//! everything appended since. The estimate therefore converges on the
//! provider's own tokenizer within a single turn instead of drifting with the
//! content mix.

use aither_core::llm::Usage;
use num_traits::ToPrimitive as _;

/// Characters per token assumed before the provider has reported anything.
const DEFAULT_CHARS_PER_TOKEN: f64 = 4.0;

/// Bounds on the calibrated ratio.
///
/// A ratio outside this range means the pairing was corrupted (a cached
/// request that reported no prompt tokens, an attachment counted as pixels
/// rather than characters) and the measurement is not trustworthy.
const MIN_CHARS_PER_TOKEN: f64 = 1.0;
const MAX_CHARS_PER_TOKEN: f64 = 16.0;

/// Window assumed until the provider reports the model's real context length.
const DEFAULT_WINDOW: u32 = 100_000;

/// A characters-to-tokens observation taken from one completed request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Calibration {
    /// Characters in the prompt that was sent.
    chars: usize,
    /// Prompt tokens the provider charged for it, cache reads included.
    tokens: u32,
}

/// Tracks how much of the model's context window the assembled prompt uses.
///
/// # Example
///
/// ```
/// use aither_agent::ContextBudget;
///
/// let mut budget = ContextBudget::new(1000);
/// budget.observe_assembled(4000);
/// // Without a provider measurement the default ratio applies.
/// assert_eq!(budget.estimated_tokens(), 1000);
/// assert!((budget.usage() - 1.0).abs() < f32::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    /// Usable context window in tokens.
    window: u32,
    /// Characters in the most recently assembled prompt.
    assembled_chars: usize,
    /// Characters appended after that prompt was assembled.
    appended_chars: usize,
    /// Latest trustworthy provider measurement.
    calibration: Option<Calibration>,
}

impl Default for ContextBudget {
    /// A budget for a conservatively small window.
    ///
    /// The real window arrives with the model profile; until then the agent
    /// assumes the smallest window a modern model is likely to have, so a
    /// threshold fires early rather than never.
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW)
    }
}

impl ContextBudget {
    /// Creates a budget for a model with `window` usable prompt tokens.
    #[must_use]
    pub const fn new(window: u32) -> Self {
        Self {
            window,
            assembled_chars: 0,
            appended_chars: 0,
            calibration: None,
        }
    }

    /// The usable context window in tokens.
    #[must_use]
    pub const fn window(&self) -> u32 {
        self.window
    }

    /// Replaces the context window, keeping the calibration.
    ///
    /// Used when the model profile arrives after the agent was constructed.
    pub const fn set_window(&mut self, window: u32) {
        self.window = window;
    }

    /// Records the size of the prompt that is about to be sent.
    ///
    /// Anything counted by [`append`](Self::append) since the previous call is
    /// now part of the assembled prompt, so the appended tally resets.
    pub const fn observe_assembled(&mut self, chars: usize) {
        self.assembled_chars = chars;
        self.appended_chars = 0;
    }

    /// Counts content added to the window after the last assembly.
    ///
    /// Tool results and alerts land between requests; charging them against
    /// the window immediately is what lets the compaction threshold fire at a
    /// request boundary rather than one turn late.
    pub const fn append(&mut self, chars: usize) {
        self.appended_chars = self.appended_chars.saturating_add(chars);
    }

    /// Calibrates the estimator against a provider usage report.
    ///
    /// The prompt cost is `prompt_tokens` plus whatever the provider served
    /// from its cache: a cached request reports a tiny `prompt_tokens` for a
    /// window that is in fact nearly full, and calibrating on that number
    /// alone would make the agent believe it had unlimited room.
    pub fn calibrate(&mut self, usage: &Usage) {
        let Some(tokens) = prompt_cost(usage) else {
            return;
        };
        if tokens == 0 || self.assembled_chars == 0 {
            return;
        }
        let ratio = to_f64(self.assembled_chars) / f64::from(tokens);
        if !(MIN_CHARS_PER_TOKEN..=MAX_CHARS_PER_TOKEN).contains(&ratio) {
            tracing::debug!(
                chars = self.assembled_chars,
                tokens,
                ratio,
                "discarding implausible context budget calibration"
            );
            return;
        }
        self.calibration = Some(Calibration {
            chars: self.assembled_chars,
            tokens,
        });
    }

    /// Characters per token, calibrated if a measurement is available.
    #[must_use]
    pub fn chars_per_token(&self) -> f64 {
        self.calibration
            .map_or(DEFAULT_CHARS_PER_TOKEN, |measured| {
                to_f64(measured.chars) / f64::from(measured.tokens)
            })
    }

    /// Converts a character count into tokens at the calibrated ratio.
    #[must_use]
    pub fn tokens_for_chars(&self, chars: usize) -> usize {
        to_usize((to_f64(chars) / self.chars_per_token()).ceil())
    }

    /// Converts a token count into characters at the calibrated ratio.
    #[must_use]
    pub fn chars_for_tokens(&self, tokens: usize) -> usize {
        to_usize(to_f64(tokens) * self.chars_per_token())
    }

    /// The estimated prompt tokens currently held in the window.
    ///
    /// The measured part of the window is taken from the provider verbatim;
    /// only the characters added since are converted at the calibrated ratio.
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        match self.calibration {
            Some(measured) if measured.chars == self.assembled_chars => {
                to_usize_u32(measured.tokens) + self.tokens_for_chars(self.appended_chars)
            }
            _ => self.tokens_for_chars(self.assembled_chars + self.appended_chars),
        }
    }

    /// The fraction of the window in use, clamped to `0.0..=1.0`.
    ///
    /// A zero window (no profile yet) reports no usage: thresholds must not
    /// fire on a window the agent has not learned the size of.
    #[must_use]
    pub fn usage(&self) -> f32 {
        if self.window == 0 {
            return 0.0;
        }
        let fraction = to_f64(self.estimated_tokens()) / f64::from(self.window);
        fraction.clamp(0.0, 1.0).to_f32().unwrap_or(1.0)
    }

    /// Tokens still available before the window is full.
    #[must_use]
    pub fn remaining_tokens(&self) -> usize {
        to_usize_u32(self.window).saturating_sub(self.estimated_tokens())
    }

    /// The token allowance for a fraction of the window.
    ///
    /// Used for [`ContextPolicy::restore_budget`](super::ContextPolicy::restore_budget).
    #[must_use]
    pub fn fraction_tokens(&self, fraction: f32) -> usize {
        to_usize(f64::from(self.window) * f64::from(fraction.clamp(0.0, 1.0)))
    }
}

/// Widens a count for ratio arithmetic, saturating rather than wrapping.
fn to_f64(value: usize) -> f64 {
    value.to_f64().unwrap_or(f64::MAX)
}

/// Widens a token count reported by a provider.
fn to_usize_u32(value: u32) -> usize {
    value.to_usize().unwrap_or(usize::MAX)
}

/// Narrows a non-negative ratio result back to a count.
fn to_usize(value: f64) -> usize {
    value.floor().to_usize().unwrap_or(usize::MAX)
}

/// The tokens a provider charged for reading a prompt.
///
/// Two usage shapes exist in the wild: providers that report the whole
/// prompt in `prompt_tokens` with cache hits as an informational subset
/// (`OpenAI`, Gemini, the Claude adapter's normalized total), and providers
/// that report only the uncached remainder alongside the cached parts
/// (Claude's raw API fields). A cached count larger than the reported
/// prompt can only come from the second shape — a subset can never exceed
/// its total — so that case sums; otherwise `prompt_tokens` already
/// includes the cached tokens and summing would double-count them.
fn prompt_cost(usage: &Usage) -> Option<u32> {
    let prompt = usage.prompt_tokens?;
    let cached = usage
        .cache_read_tokens
        .unwrap_or(0)
        .saturating_add(usage.cache_write_tokens.unwrap_or(0));
    Some(if cached > prompt {
        prompt.saturating_add(cached)
    } else {
        prompt
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u32, cache_read: u32) -> Usage {
        Usage {
            prompt_tokens: Some(prompt),
            cache_read_tokens: Some(cache_read),
            ..Usage::default()
        }
    }

    #[test]
    fn uncalibrated_budget_uses_the_default_ratio() {
        let mut budget = ContextBudget::new(1_000);
        budget.observe_assembled(2_000);
        assert_eq!(budget.estimated_tokens(), 500);
        assert!((budget.usage() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn calibration_replaces_the_heuristic_with_provider_counts() {
        let mut budget = ContextBudget::new(10_000);
        budget.observe_assembled(6_000);
        // Dense code: 2 characters per token, not 4.
        budget.calibrate(&usage(3_000, 0));
        assert_eq!(budget.estimated_tokens(), 3_000);
        assert!((budget.chars_per_token() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn cache_reads_count_towards_the_prompt_cost() {
        let mut budget = ContextBudget::new(10_000);
        budget.observe_assembled(8_000);
        // A cached request reports 100 fresh tokens over 3_900 cached ones.
        budget.calibrate(&usage(100, 3_900));
        assert_eq!(budget.estimated_tokens(), 4_000);
    }

    #[test]
    fn cached_subsets_are_not_double_counted() {
        // OpenAI and Gemini report the whole prompt in `prompt_tokens` with
        // the cached share as an informational subset; summing the two
        // double-counts the cache reads.
        let mut budget = ContextBudget::new(10_000);
        budget.observe_assembled(8_000);
        budget.calibrate(&usage(4_000, 3_900));
        assert_eq!(budget.estimated_tokens(), 4_000);
    }

    #[test]
    fn appended_content_is_charged_on_top_of_the_measurement() {
        let mut budget = ContextBudget::new(10_000);
        budget.observe_assembled(4_000);
        budget.calibrate(&usage(1_000, 0));
        budget.append(400);
        assert_eq!(budget.estimated_tokens(), 1_100);
    }

    #[test]
    fn implausible_measurements_are_discarded() {
        let mut budget = ContextBudget::new(10_000);
        budget.observe_assembled(4_000);
        budget.calibrate(&usage(1, 0));
        assert!((budget.chars_per_token() - DEFAULT_CHARS_PER_TOKEN).abs() < 1e-9);
    }

    #[test]
    fn an_unknown_window_never_reports_pressure() {
        let mut budget = ContextBudget::new(0);
        budget.observe_assembled(1_000_000);
        assert!((budget.usage() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn fraction_tokens_scales_the_window() {
        let budget = ContextBudget::new(200_000);
        assert_eq!(budget.fraction_tokens(0.15), 30_000);
        assert_eq!(budget.fraction_tokens(2.0), 200_000);
    }
}
