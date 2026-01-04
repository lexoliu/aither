//! Skill matching based on trigger phrases.

// Precision loss from usize->f32 is acceptable for confidence scoring
#![allow(clippy::cast_precision_loss)]

use crate::Skill;

/// Matches user prompts against skill triggers.
///
/// Uses simple substring matching for now. Can be extended
/// to support more sophisticated matching (embeddings, fuzzy, etc.)
#[derive(Debug, Default)]
pub struct SkillMatcher {
    /// Minimum confidence threshold (0.0 - 1.0).
    threshold: f32,
}

/// Result of matching a skill against a prompt.
#[derive(Debug, Clone)]
pub struct MatchResult<'a> {
    /// The matched skill.
    pub skill: &'a Skill,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    /// The trigger that matched.
    pub matched_trigger: Option<String>,
}

impl SkillMatcher {
    /// Create a new matcher with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self { threshold: 0.5 }
    }

    /// Set the minimum confidence threshold for matches.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // clamp is not const
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Find skills that match the given prompt.
    ///
    /// Returns matches sorted by confidence (highest first).
    #[must_use]
    pub fn match_prompt<'a>(&self, prompt: &str, skills: &'a [Skill]) -> Vec<MatchResult<'a>> {
        let prompt_lower = prompt.to_lowercase();

        let mut matches: Vec<MatchResult<'a>> = skills
            .iter()
            .filter_map(|skill| Self::score_skill(skill, &prompt_lower))
            .filter(|m| m.confidence >= self.threshold)
            .collect();

        // Sort by confidence descending
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Score how well a skill matches the prompt.
    fn score_skill<'a>(skill: &'a Skill, prompt_lower: &str) -> Option<MatchResult<'a>> {
        // Check triggers
        for trigger in &skill.triggers {
            let trigger_lower = trigger.to_lowercase();
            if prompt_lower.contains(&trigger_lower) {
                // Exact trigger match - high confidence
                let confidence = Self::calculate_confidence(&trigger_lower, prompt_lower);
                return Some(MatchResult {
                    skill,
                    confidence,
                    matched_trigger: Some(trigger.clone()),
                });
            }
        }

        // Check skill name
        let name_lower = skill.name.to_lowercase();
        if prompt_lower.contains(&name_lower) {
            let confidence = Self::calculate_confidence(&name_lower, prompt_lower) * 0.8;
            return Some(MatchResult {
                skill,
                confidence,
                matched_trigger: None,
            });
        }

        // Check description keywords
        let desc_lower = skill.description.to_lowercase();
        let desc_words: Vec<&str> = desc_lower.split_whitespace().collect();
        let prompt_words: Vec<&str> = prompt_lower.split_whitespace().collect();

        let matching_words = desc_words
            .iter()
            .filter(|w| w.len() > 3) // Skip short words
            .filter(|w| prompt_words.contains(w))
            .count();

        if matching_words >= 2 {
            let confidence = (matching_words as f32 / desc_words.len() as f32).min(0.6);
            return Some(MatchResult {
                skill,
                confidence,
                matched_trigger: None,
            });
        }

        None
    }

    /// Calculate confidence based on how much of the prompt the trigger covers.
    fn calculate_confidence(trigger: &str, prompt: &str) -> f32 {
        let trigger_len = trigger.len() as f32;
        let prompt_len = prompt.len() as f32;

        // Base confidence from coverage
        let coverage = (trigger_len / prompt_len).min(1.0);

        // Boost for longer triggers (more specific)
        let length_bonus = (trigger_len / 20.0).min(0.3);

        (0.5 + coverage * 0.3 + length_bonus).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skill(name: &str, triggers: &[&str]) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{name} description"),
            triggers: triggers.iter().map(|s| s.to_string()).collect(),
            instructions: String::new(),
            allowed_tools: None,
            resources: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_match_by_trigger() {
        let skills = vec![
            make_skill("code-review", &["review", "security audit"]),
            make_skill("refactor", &["refactor", "clean up"]),
        ];

        let matcher = SkillMatcher::new();
        let matches = matcher.match_prompt("please review this code", &skills);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].skill.name, "code-review");
        assert_eq!(matches[0].matched_trigger, Some("review".to_string()));
    }

    #[test]
    fn test_match_by_name() {
        let skills = vec![make_skill("code-review", &[])];

        let matcher = SkillMatcher::new();
        let matches = matcher.match_prompt("run code-review", &skills);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].skill.name, "code-review");
        assert!(matches[0].matched_trigger.is_none());
    }

    #[test]
    fn test_case_insensitive() {
        let skills = vec![make_skill("code-review", &["Review"])];

        let matcher = SkillMatcher::new();
        let matches = matcher.match_prompt("REVIEW this code", &skills);

        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_no_match() {
        let skills = vec![make_skill("code-review", &["review"])];

        let matcher = SkillMatcher::new();
        let matches = matcher.match_prompt("write some tests", &skills);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_threshold() {
        let skills = vec![make_skill("test", &["test"])];

        // High threshold - short trigger won't match long prompt
        let matcher = SkillMatcher::new().with_threshold(0.9);
        let matches = matcher.match_prompt(
            "this is a very long prompt that mentions test somewhere",
            &skills,
        );

        assert!(matches.is_empty());

        // Low threshold - will match
        let matcher = SkillMatcher::new().with_threshold(0.3);
        let matches = matcher.match_prompt(
            "this is a very long prompt that mentions test somewhere",
            &skills,
        );

        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_multiple_matches_sorted() {
        let skills = vec![
            make_skill("general", &["code"]),
            make_skill("specific", &["security code review"]),
        ];

        let matcher = SkillMatcher::new().with_threshold(0.0);
        let matches = matcher.match_prompt("do a security code review", &skills);

        // More specific match should be first
        assert!(matches.len() >= 1);
        assert_eq!(matches[0].skill.name, "specific");
    }
}
