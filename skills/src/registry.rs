//! Skill registry for storing and querying skills.

use std::collections::HashMap;

use crate::{MatchResult, Skill, SkillError, SkillLoader, SkillMatcher};

/// Registry of available skills.
///
/// Stores skills by name and provides methods to find relevant skills
/// based on user prompts.
///
/// # Example
///
/// ```rust,ignore
/// let loader = SkillLoader::new().add_path("~/.aither/skills");
/// let mut registry = SkillRegistry::new();
/// registry.load_from(&loader)?;
///
/// // Find skills relevant to a prompt
/// let matches = registry.match_prompt("review this code");
/// for m in matches {
///     println!("Skill: {} (confidence: {:.2})", m.skill.name, m.confidence);
/// }
/// ```
#[derive(Debug, Default)]
pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
    matcher: SkillMatcher,
}

impl SkillRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with a custom matcher.
    #[must_use]
    pub fn with_matcher(matcher: SkillMatcher) -> Self {
        Self {
            skills: HashMap::new(),
            matcher,
        }
    }

    /// Load skills from a loader.
    ///
    /// Skills are added to the registry, potentially overwriting
    /// existing skills with the same name.
    ///
    /// # Errors
    ///
    /// Returns an error if the loader fails to load skills.
    pub fn load_from(&mut self, loader: &SkillLoader) -> Result<usize, SkillError> {
        let skills = loader.load_all()?;
        let count = skills.len();

        for skill in skills {
            self.skills.insert(skill.name.clone(), skill);
        }

        Ok(count)
    }

    /// Register a skill directly.
    pub fn register(&mut self, skill: Skill) {
        self.skills.insert(skill.name.clone(), skill);
    }

    /// Get a skill by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// Check if a skill exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    /// Get all registered skill names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.skills.keys().map(String::as_str).collect()
    }

    /// Get all registered skills.
    #[must_use]
    pub fn all(&self) -> Vec<&Skill> {
        self.skills.values().collect()
    }

    /// Number of registered skills.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Find skills that match a user prompt.
    ///
    /// Returns matches sorted by confidence (highest first).
    #[must_use]
    pub fn match_prompt(&self, prompt: &str) -> Vec<MatchResult<'_>> {
        // Clone skills for matching (matcher takes owned slice)
        let skills_owned: Vec<Skill> = self.skills.values().cloned().collect();
        let matches = self.matcher.match_prompt(prompt, &skills_owned);

        // Re-map to registry references for stable lifetimes
        matches
            .into_iter()
            .filter_map(|m| {
                self.skills.get(&m.skill.name).map(|skill| MatchResult {
                    skill,
                    confidence: m.confidence,
                    matched_trigger: m.matched_trigger,
                })
            })
            .collect()
    }

    /// Remove a skill from the registry.
    pub fn remove(&mut self, name: &str) -> Option<Skill> {
        self.skills.remove(name)
    }

    /// Clear all skills from the registry.
    pub fn clear(&mut self) {
        self.skills.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skill(name: &str, triggers: &[&str]) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{name} description"),
            triggers: triggers
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            instructions: format!("Instructions for {name}"),
            allowed_tools: None,
            resources: HashMap::new(),
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = SkillRegistry::new();
        registry.register(make_skill("test", &[]));

        assert!(registry.contains("test"));
        assert_eq!(
            registry.get("test").map(|s| &s.name),
            Some(&"test".to_string())
        );
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_names_and_all() {
        let mut registry = SkillRegistry::new();
        registry.register(make_skill("alpha", &[]));
        registry.register(make_skill("beta", &[]));

        let names = registry.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));

        assert_eq!(registry.all().len(), 2);
    }

    #[test]
    fn test_match_prompt() {
        let mut registry = SkillRegistry::new();
        registry.register(make_skill("code-review", &["review", "audit"]));
        registry.register(make_skill("refactor", &["refactor"]));

        let matches = registry.match_prompt("please review this code");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].skill.name, "code-review");
    }

    #[test]
    fn test_remove_and_clear() {
        let mut registry = SkillRegistry::new();
        registry.register(make_skill("a", &[]));
        registry.register(make_skill("b", &[]));

        assert_eq!(registry.len(), 2);

        registry.remove("a");
        assert_eq!(registry.len(), 1);
        assert!(!registry.contains("a"));

        registry.clear();
        assert!(registry.is_empty());
    }
}
