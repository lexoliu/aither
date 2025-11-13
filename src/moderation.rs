use alloc::vec::Vec;
use core::future::Future;

/// Trait for content moderation services.
pub trait Moderation {
    /// The error type returned by moderation operations.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Moderates the provided content and returns a result asynchronously.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to be moderated.
    fn moderate(
        &self,
        content: &str,
    ) -> impl Future<Output = Result<ModerationResult, Self::Error>> + Send;
}

/// The result of a moderation operation.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModerationResult {
    /// Indicates whether the content was flagged.
    flagged: bool,
    /// The categories of violations that were detected in the content.
    /// All categories in this list represent detected violations, with their respective confidence scores.
    categories: Vec<ModerationCategory>,
}

impl ModerationResult {
    /// Creates a new moderation result.
    ///
    /// # Arguments
    ///
    /// * `flagged` - Whether the content was flagged as violating policies
    /// * `categories` - List of detected violation categories with confidence scores
    #[must_use]
    pub const fn new(flagged: bool, categories: Vec<ModerationCategory>) -> Self {
        Self {
            flagged,
            categories,
        }
    }

    /// Returns whether the content was flagged.
    #[must_use]
    pub const fn is_flagged(&self) -> bool {
        self.flagged
    }

    /// Returns the detected violation categories.
    #[must_use]
    pub fn categories(&self) -> &[ModerationCategory] {
        &self.categories
    }

    /// Returns the number of detected violations.
    #[must_use]
    pub const fn violation_count(&self) -> usize {
        self.categories.len()
    }

    /// Returns whether any violations were detected.
    #[must_use]
    pub const fn has_violations(&self) -> bool {
        !self.categories.is_empty()
    }
}

/// Categories of content moderation.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ModerationCategory {
    /// Hate category with a confidence score.
    Hate {
        /// Confidence score indicating the severity/certainty of hate content detection (0.0-1.0).
        score: f32,
    },
    /// Harassment category with a confidence score.
    Harassment {
        /// Confidence score indicating the severity/certainty of harassment content detection (0.0-1.0).
        score: f32,
    },
    /// Sexual category with a confidence score.
    Sexual {
        /// Confidence score indicating the severity/certainty of sexual content detection (0.0-1.0).
        score: f32,
    },
    /// Violence category with a confidence score.
    Violence {
        /// Confidence score indicating the severity/certainty of violence content detection (0.0-1.0).
        score: f32,
    },
    /// Self-harm category with a confidence score.
    SelfHarm {
        /// Confidence score indicating the severity/certainty of self-harm content detection (0.0-1.0).
        score: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};
    use core::convert::Infallible;

    struct MockModeration;

    impl Moderation for MockModeration {
        type Error = Infallible;

        async fn moderate(&self, content: &str) -> Result<ModerationResult, Self::Error> {
            // Mock moderation logic based on content
            let flagged = content.contains("bad") || content.contains("harmful");
            let mut categories = Vec::new();

            if content.contains("hate") {
                categories.push(ModerationCategory::Hate { score: 0.9 });
            }
            if content.contains("violence") {
                categories.push(ModerationCategory::Violence { score: 0.8 });
            }
            if content.contains("sexual") {
                categories.push(ModerationCategory::Sexual { score: 0.7 });
            }
            if content.contains("harassment") {
                categories.push(ModerationCategory::Harassment { score: 0.85 });
            }
            if content.contains("self-harm") {
                categories.push(ModerationCategory::SelfHarm { score: 0.95 });
            }

            Ok(ModerationResult::new(flagged, categories))
        }
    }

    #[tokio::test]
    async fn moderation_clean_content() {
        let moderation = MockModeration;
        let result = moderation
            .moderate("This is a nice and friendly message")
            .await
            .unwrap();

        assert!(!result.is_flagged());
        assert!(!result.has_violations());
    }

    #[tokio::test]
    async fn moderation_flagged_content() {
        let moderation = MockModeration;
        let result = moderation
            .moderate("This contains bad content")
            .await
            .unwrap();

        assert!(result.is_flagged());
        assert!(!result.has_violations()); // No specific categories, just flagged
    }

    #[tokio::test]
    async fn moderation_hate_content() {
        let moderation = MockModeration;
        let result = moderation
            .moderate("This message contains hate speech")
            .await
            .unwrap();

        assert!(!result.is_flagged()); // Not flagged by "bad" keyword
        assert_eq!(result.violation_count(), 1);

        match &result.categories()[0] {
            ModerationCategory::Hate { score } => {
                assert!((score - 0.9).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Hate category"),
        }
    }

    #[tokio::test]
    async fn moderation_violence_content() {
        let moderation = MockModeration;
        let result = moderation
            .moderate("This message promotes violence")
            .await
            .unwrap();

        assert!(!result.is_flagged());
        assert_eq!(result.violation_count(), 1);

        match &result.categories()[0] {
            ModerationCategory::Violence { score } => {
                assert!((score - 0.8).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Violence category"),
        }
    }

    #[tokio::test]
    async fn moderation_multiple_categories() {
        let moderation = MockModeration;
        let result = moderation
            .moderate("This bad message contains hate and violence")
            .await
            .unwrap();

        assert!(result.is_flagged());
        assert_eq!(result.violation_count(), 2);

        // Check that both categories are present
        let has_hate = result
            .categories
            .iter()
            .any(|cat| matches!(cat, ModerationCategory::Hate { .. }));
        let has_violence = result
            .categories
            .iter()
            .any(|cat| matches!(cat, ModerationCategory::Violence { .. }));

        assert!(has_hate);
        assert!(has_violence);
    }

    #[tokio::test]
    async fn moderation_all_categories() {
        let moderation = MockModeration;
        let result = moderation
            .moderate("harmful content with hate, violence, sexual, harassment, and self-harm")
            .await
            .unwrap();

        assert!(result.is_flagged());
        assert_eq!(result.violation_count(), 5);

        // Verify all category types are present
        let mut found_categories = [false; 5]; // hate, violence, sexual, harassment, self-harm

        for category in result.categories() {
            match category {
                ModerationCategory::Hate { score } => {
                    found_categories[0] = true;
                    assert!((score - 0.9).abs() < f32::EPSILON);
                }
                ModerationCategory::Violence { score } => {
                    found_categories[1] = true;
                    assert!((score - 0.8).abs() < f32::EPSILON);
                }
                ModerationCategory::Sexual { score } => {
                    found_categories[2] = true;
                    assert!((score - 0.7).abs() < f32::EPSILON);
                }
                ModerationCategory::Harassment { score } => {
                    found_categories[3] = true;
                    assert!((score - 0.85).abs() < f32::EPSILON);
                }
                ModerationCategory::SelfHarm { score } => {
                    found_categories[4] = true;
                    assert!((score - 0.95).abs() < f32::EPSILON);
                }
            }
        }

        assert!(
            found_categories.iter().all(|&found| found),
            "Not all categories were found"
        );
    }

    #[test]
    fn moderation_result_creation() {
        let result = ModerationResult::new(
            true,
            vec![
                ModerationCategory::Hate { score: 0.8 },
                ModerationCategory::Violence { score: 0.9 },
            ],
        );

        assert!(result.is_flagged());
        assert_eq!(result.categories().len(), 2);
        assert_eq!(result.violation_count(), 2);
        assert!(result.has_violations());
    }

    #[test]
    fn moderation_category_equality() {
        let cat1 = ModerationCategory::Hate { score: 0.8 };
        let cat2 = ModerationCategory::Hate { score: 0.8 };
        let cat3 = ModerationCategory::Hate { score: 0.9 };
        let cat4 = ModerationCategory::Violence { score: 0.8 };

        assert_eq!(cat1, cat2);
        assert_ne!(cat1, cat3);
        assert_ne!(cat1, cat4);
    }

    #[test]
    fn moderation_category_clone() {
        let original = ModerationCategory::Sexual { score: 0.7 };
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    #[test]
    fn moderation_category_debug() {
        let category = ModerationCategory::Harassment { score: 0.85 };
        let debug_string = format!("{category:?}");

        assert!(debug_string.contains("Harassment"));
        assert!(debug_string.contains("0.85"));
    }

    #[tokio::test]
    async fn moderation_empty_content() {
        let moderation = MockModeration;
        let result = moderation.moderate("").await.unwrap();

        assert!(!result.is_flagged());
        assert!(!result.has_violations());
    }

    #[tokio::test]
    async fn moderation_whitespace_content() {
        let moderation = MockModeration;
        let result = moderation.moderate("   \n\t  ").await.unwrap();

        assert!(!result.is_flagged());
        assert!(!result.has_violations());
    }
}
