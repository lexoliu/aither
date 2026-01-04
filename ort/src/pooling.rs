//! Pooling strategies for extracting embeddings from transformer outputs.

use ndarray::ArrayView3;

/// Strategy for pooling hidden states into a single embedding vector.
///
/// Different embedding models require different pooling strategies:
/// - Decoder models (Qwen3-Embedding) use [`LastToken`](PoolingStrategy::LastToken)
/// - Encoder models (BERT, MiniLM) typically use [`Cls`](PoolingStrategy::Cls) or [`Mean`](PoolingStrategy::Mean)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the last non-padding token's hidden state.
    ///
    /// This is the default for decoder-style embedding models like Qwen3-Embedding.
    #[default]
    LastToken,

    /// Average all non-padding token hidden states.
    ///
    /// A common choice for encoder models that produces robust embeddings.
    Mean,

    /// Use the first token's hidden state (typically [CLS] token).
    ///
    /// Traditional approach for BERT-style encoder models.
    Cls,
}

impl PoolingStrategy {
    /// Apply the pooling strategy to hidden states.
    ///
    /// # Arguments
    /// * `hidden_states` - Tensor of shape `[batch=1, seq_len, hidden_dim]`
    /// * `attention_mask` - Mask indicating valid tokens (1) vs padding (0)
    ///
    /// # Returns
    /// A vector of `hidden_dim` floats representing the pooled embedding.
    #[must_use]
    pub fn apply(&self, hidden_states: &ArrayView3<f32>, attention_mask: &[u32]) -> Vec<f32> {
        let seq_len = hidden_states.shape()[1];
        let hidden_dim = hidden_states.shape()[2];

        match self {
            Self::LastToken => {
                // Find the last non-padding token
                let last_idx = attention_mask
                    .iter()
                    .rposition(|&m| m != 0)
                    .unwrap_or(seq_len - 1);

                // Extract the hidden state at that position
                (0..hidden_dim)
                    .map(|d| hidden_states[[0, last_idx, d]])
                    .collect()
            }
            Self::Mean => {
                // Count valid tokens
                let valid_count: f32 = attention_mask.iter().map(|&m| m as f32).sum();
                if valid_count == 0.0 {
                    return vec![0.0; hidden_dim];
                }

                // Sum hidden states weighted by attention mask
                let mut result = vec![0.0; hidden_dim];
                for (seq_idx, &mask) in attention_mask.iter().enumerate() {
                    if mask != 0 {
                        for (d, value) in result.iter_mut().enumerate() {
                            *value += hidden_states[[0, seq_idx, d]];
                        }
                    }
                }

                // Divide by count
                for value in &mut result {
                    *value /= valid_count;
                }

                result
            }
            Self::Cls => {
                // First token (index 0)
                (0..hidden_dim).map(|d| hidden_states[[0, 0, d]]).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn last_token_pooling() {
        // Shape: [1, 3, 4] - batch=1, seq_len=3, hidden_dim=4
        let hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // token 0
                5.0, 6.0, 7.0, 8.0, // token 1
                9.0, 10.0, 11.0, 12.0, // token 2
            ],
        )
        .unwrap();
        let mask = vec![1, 1, 1]; // all valid

        let result = PoolingStrategy::LastToken.apply(&hidden.view(), &mask);
        assert_eq!(result, vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn last_token_with_padding() {
        let hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // token 0
                5.0, 6.0, 7.0, 8.0, // token 1 (last valid)
                0.0, 0.0, 0.0, 0.0, // token 2 (padding)
            ],
        )
        .unwrap();
        let mask = vec![1, 1, 0]; // token 2 is padding

        let result = PoolingStrategy::LastToken.apply(&hidden.view(), &mask);
        assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn mean_pooling() {
        let hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // token 0
                5.0, 6.0, 7.0, 8.0, // token 1
                9.0, 10.0, 11.0, 12.0, // token 2 (padding)
            ],
        )
        .unwrap();
        let mask = vec![1, 1, 0]; // only first 2 valid

        let result = PoolingStrategy::Mean.apply(&hidden.view(), &mask);
        // Mean of tokens 0 and 1
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn cls_pooling() {
        let hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // token 0 (CLS)
                5.0, 6.0, 7.0, 8.0, // token 1
                9.0, 10.0, 11.0, 12.0, // token 2
            ],
        )
        .unwrap();
        let mask = vec![1, 1, 1];

        let result = PoolingStrategy::Cls.apply(&hidden.view(), &mask);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
