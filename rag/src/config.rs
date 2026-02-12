//! Configuration for RAG.

use std::path::PathBuf;

/// Configuration for a RAG instance.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Path to the persistence file.
    pub index_path: PathBuf,
    /// Minimum similarity score for search results.
    pub similarity_threshold: f32,
    /// Default number of results to return.
    pub default_top_k: usize,
    /// Whether to enable content deduplication.
    pub deduplication: bool,
    /// Whether to automatically save after indexing operations.
    pub auto_save: bool,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            index_path: PathBuf::from("./rag_index.redb"),
            similarity_threshold: 0.0,
            default_top_k: 5,
            deduplication: true,
            auto_save: true,
        }
    }
}

impl RagConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a builder for custom configuration.
    #[must_use]
    pub fn builder() -> RagConfigBuilder {
        RagConfigBuilder::new()
    }
}

/// Builder for RAG configuration.
#[derive(Debug, Default)]
pub struct RagConfigBuilder {
    config: RagConfig,
}

impl RagConfigBuilder {
    /// Creates a new configuration builder with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RagConfig::default(),
        }
    }

    /// Sets the index persistence path.
    #[must_use]
    pub fn index_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.index_path = path.into();
        self
    }

    /// Sets the minimum similarity threshold for search results.
    #[must_use]
    pub const fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.config.similarity_threshold = threshold;
        self
    }

    /// Sets the default number of results to return.
    #[must_use]
    pub const fn default_top_k(mut self, k: usize) -> Self {
        self.config.default_top_k = k;
        self
    }

    /// Enables or disables content deduplication.
    #[must_use]
    pub const fn deduplication(mut self, enabled: bool) -> Self {
        self.config.deduplication = enabled;
        self
    }

    /// Enables or disables automatic saving after indexing.
    #[must_use]
    pub const fn auto_save(mut self, enabled: bool) -> Self {
        self.config.auto_save = enabled;
        self
    }

    /// Builds the configuration.
    #[must_use]
    pub fn build(self) -> RagConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = RagConfig::default();
        assert_eq!(config.index_path, PathBuf::from("./rag_index.redb"));
        assert_eq!(config.similarity_threshold, 0.0);
        assert_eq!(config.default_top_k, 5);
        assert!(config.deduplication);
        assert!(config.auto_save);
    }

    #[test]
    fn builder_config() {
        let config = RagConfig::builder()
            .index_path("/custom/path.redb")
            .similarity_threshold(0.5)
            .default_top_k(10)
            .deduplication(false)
            .auto_save(false)
            .build();

        assert_eq!(config.index_path, PathBuf::from("/custom/path.redb"));
        assert_eq!(config.similarity_threshold, 0.5);
        assert_eq!(config.default_top_k, 10);
        assert!(!config.deduplication);
        assert!(!config.auto_save);
    }
}
