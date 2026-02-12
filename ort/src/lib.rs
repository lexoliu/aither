//! Local ONNX Runtime embedding models for aither.
//!
//! This crate provides [`OrtEmbedding`], an implementation of [`aither_core::EmbeddingModel`]
//! that runs ONNX embedding models locally using ONNX Runtime.
//!
//! # Features
//!
//! - **No auto-download**: You provide the model and tokenizer paths
//! - **Auto-detect dimension**: Embedding dimension is detected from model outputs
//! - **Multiple pooling strategies**: `LastToken`, `Mean`, `Cls`
//! - **GPU acceleration**: CUDA and `CoreML` enabled by default
//!
//! # Example
//!
//! ```rust,no_run
//! use aither_ort::{OrtEmbedding, PoolingStrategy};
//! use aither_core::EmbeddingModel;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load from model directory (auto-finds model.onnx and tokenizer.json)
//! let embedder = OrtEmbedding::from_directory("./models/Qwen3-Embedding-0.6B-ONNX")?;
//! println!("Embedding dimension: {}", embedder.dim());
//!
//! // Generate embeddings
//! let embedding = embedder.embed("Hello, world!").await?;
//! # Ok(())
//! # }
//! ```

mod error;
mod pooling;

pub use error::OrtError;
pub use pooling::PoolingStrategy;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use aither_core::EmbeddingModel;
use ndarray::Ix3;
use ort::session::{Session, builder::GraphOptimizationLevel};
use tokenizers::Tokenizer;

/// An embedding model backed by ONNX Runtime.
///
/// This struct wraps an ONNX model session and tokenizer to provide
/// text embedding functionality. It implements [`EmbeddingModel`] for
/// seamless integration with other aither components.
///
/// # Example
///
/// ```rust,no_run
/// use aither_ort::OrtEmbedding;
///
/// // Simple: load from directory
/// let embedder = OrtEmbedding::from_directory("./model")?;
///
/// // Custom: use builder
/// let embedder = OrtEmbedding::builder()
///     .model_path("./model/model.onnx")
///     .tokenizer_path("./model/tokenizer.json")
///     .build()?;
/// # Ok::<(), aither_ort::OrtError>(())
/// ```
pub struct OrtEmbedding {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimension: usize,
    pooling: PoolingStrategy,
    normalize: bool,
}

impl std::fmt::Debug for OrtEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtEmbedding")
            .field("dimension", &self.dimension)
            .field("pooling", &self.pooling)
            .field("normalize", &self.normalize)
            .finish_non_exhaustive()
    }
}

impl OrtEmbedding {
    /// Load an embedding model from a directory.
    ///
    /// This method automatically locates `model.onnx` (or files in `onnx/` subdirectory)
    /// and `tokenizer.json` within the specified directory.
    ///
    /// # Arguments
    /// * `path` - Path to the model directory
    ///
    /// # Errors
    /// Returns an error if the model or tokenizer cannot be found or loaded.
    pub fn from_directory(path: impl AsRef<Path>) -> Result<Self, OrtError> {
        let dir = path.as_ref();

        // Find model file
        let model_path = find_model_file(dir)?;

        // Find tokenizer file
        let tokenizer_path = find_tokenizer_file(dir)?;

        Self::builder()
            .model_path(model_path)
            .tokenizer_path(tokenizer_path)
            .build()
    }

    /// Create a builder for custom configuration.
    #[must_use]
    pub fn builder() -> OrtEmbeddingBuilder {
        OrtEmbeddingBuilder::default()
    }

    /// Returns the embedding dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the pooling strategy.
    #[must_use]
    pub const fn pooling(&self) -> PoolingStrategy {
        self.pooling
    }

    /// Returns whether L2 normalization is enabled.
    #[must_use]
    pub const fn normalize(&self) -> bool {
        self.normalize
    }
}

impl EmbeddingModel for OrtEmbedding {
    fn dim(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, text: &str) -> aither_core::Result<Vec<f32>> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| OrtError::Tokenization(e.to_string()))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();
        let seq_len = input_ids.len();

        // Create tensors [batch=1, seq_len]
        let input_ids_tensor =
            ort::value::Tensor::from_array(([1, seq_len], input_ids.into_boxed_slice()))
                .map_err(OrtError::from)?;
        let attention_mask_tensor =
            ort::value::Tensor::from_array(([1, seq_len], attention_mask.into_boxed_slice()))
                .map_err(OrtError::from)?;

        // Run inference and extract to owned array (before releasing session lock)
        let hidden_states_owned = {
            let mut session = self.session.lock().expect("session lock poisoned");
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                ])
                .map_err(OrtError::from)?;

            // Extract hidden states - try common output names
            let hidden_states = outputs
                .get("last_hidden_state")
                .or_else(|| outputs.get("hidden_states"))
                .or_else(|| outputs.get("output"))
                .ok_or(OrtError::InvalidOutputShape(0))?;

            let view = hidden_states
                .try_extract_array::<f32>()
                .map_err(OrtError::from)?;

            // Convert to owned array before releasing lock
            view.to_owned()
        };

        // Ensure 3D shape [batch, seq_len, hidden_dim]
        let shape = hidden_states_owned.shape();
        if shape.len() != 3 {
            return Err(OrtError::InvalidOutputShape(shape.len()).into());
        }

        let view_3d = hidden_states_owned
            .into_dimensionality::<Ix3>()
            .map_err(|e| OrtError::Shape(e.to_string()))?;

        // Apply pooling
        let attention_mask_u32: Vec<u32> = encoding.get_attention_mask().to_vec();
        let mut embedding = self.pooling.apply(&view_3d.view(), &attention_mask_u32);

        // Normalize if enabled
        if self.normalize {
            l2_normalize(&mut embedding);
        }

        Ok(embedding)
    }
}

/// Builder for [`OrtEmbedding`].
#[derive(Debug, Default)]
pub struct OrtEmbeddingBuilder {
    model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    pooling: PoolingStrategy,
    normalize: bool,
}

impl OrtEmbeddingBuilder {
    /// Set the path to the ONNX model file.
    #[must_use]
    pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the path to the tokenizer.json file.
    #[must_use]
    pub fn tokenizer_path(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the pooling strategy.
    ///
    /// Default: [`PoolingStrategy::LastToken`] (for Qwen3-style models)
    #[must_use]
    pub const fn pooling(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling = strategy;
        self
    }

    /// Enable or disable L2 normalization.
    ///
    /// Default: `true`
    #[must_use]
    pub const fn normalize(mut self, enabled: bool) -> Self {
        self.normalize = enabled;
        self
    }

    /// Build the [`OrtEmbedding`] instance.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Model path is not specified
    /// - Model file cannot be loaded
    /// - Tokenizer file cannot be loaded
    /// - Model output dimension cannot be determined
    pub fn build(self) -> Result<OrtEmbedding, OrtError> {
        let model_path = self.model_path.ok_or(OrtError::MissingModelPath)?;

        if !model_path.exists() {
            return Err(OrtError::ModelNotFound(model_path));
        }

        // Load tokenizer
        let tokenizer_path = self.tokenizer_path.ok_or_else(|| {
            // Try to find tokenizer relative to model
            OrtError::TokenizerNotFound(model_path.parent().unwrap_or(&model_path).to_path_buf())
        })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| OrtError::tokenizer(&tokenizer_path, e))?;

        // Load ONNX session with optimizations
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus())?
            .commit_from_file(&model_path)?;

        // Auto-detect dimension from model outputs
        let dimension = detect_embedding_dimension(&session)?;

        Ok(OrtEmbedding {
            session: Mutex::new(session),
            tokenizer,
            dimension,
            pooling: self.pooling,
            normalize: self.normalize,
        })
    }
}

impl Default for OrtEmbedding {
    fn default() -> Self {
        panic!(
            "OrtEmbedding requires a model path; use OrtEmbedding::from_directory() or OrtEmbedding::builder()"
        )
    }
}

/// L2 normalize a vector in place.
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// Find the ONNX model file in a directory.
fn find_model_file(dir: &Path) -> Result<PathBuf, OrtError> {
    // Check common locations
    let candidates = [
        dir.join("model.onnx"),
        dir.join("onnx/model.onnx"),
        dir.join("onnx/model_fp32.onnx"),
        dir.join("onnx/model_fp16.onnx"),
        dir.join("onnx/model_q8.onnx"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    // Look for any .onnx file
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "onnx") {
                return Ok(path);
            }
        }
    }

    // Check onnx subdirectory
    let onnx_dir = dir.join("onnx");
    if onnx_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&onnx_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "onnx") {
                    return Ok(path);
                }
            }
        }
    }

    Err(OrtError::ModelNotFound(dir.to_path_buf()))
}

/// Find the tokenizer.json file in a directory.
fn find_tokenizer_file(dir: &Path) -> Result<PathBuf, OrtError> {
    let candidates = [dir.join("tokenizer.json"), dir.join("onnx/tokenizer.json")];

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    Err(OrtError::TokenizerNotFound(dir.to_path_buf()))
}

/// Detect the embedding dimension from model output metadata.
fn detect_embedding_dimension(session: &Session) -> Result<usize, OrtError> {
    // Look for the output that contains hidden states
    for output in session.outputs() {
        // Get tensor type info if available
        if let ort::value::ValueType::Tensor { shape, .. } = output.dtype() {
            // Expect shape [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            if shape.len() >= 2 {
                // Last dimension is typically the hidden dimension
                if let Some(&dim) = shape.last() {
                    if dim > 0 {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        return Ok(dim as usize);
                    }
                }
            }
        }
    }

    // Fallback: common dimensions
    // This shouldn't happen with well-formed models
    Err(OrtError::InvalidOutputShape(0))
}

/// Get number of CPU cores for parallelism.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_model_path() {
        let result = OrtEmbeddingBuilder::default().build();
        assert!(matches!(result, Err(OrtError::MissingModelPath)));
    }

    #[test]
    fn builder_validates_model_exists() {
        let result = OrtEmbeddingBuilder::default()
            .model_path("/nonexistent/model.onnx")
            .tokenizer_path("/nonexistent/tokenizer.json")
            .build();
        assert!(matches!(result, Err(OrtError::ModelNotFound(_))));
    }

    #[test]
    fn l2_normalize_works() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize(&mut vec);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0];
        l2_normalize(&mut vec);
        assert_eq!(vec, vec![0.0, 0.0]);
    }
}
