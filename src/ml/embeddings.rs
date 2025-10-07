//! # Time Series Embeddings and Similarity Detection
//!
//! This module provides neural embeddings for time series data and sophisticated
//! similarity detection methods for clustering, anomaly detection, and pattern matching.
//!
//! ## Features
//! - Multiple embedding architectures (Autoencoder, VAE, Contrastive Learning)
//! - Advanced similarity metrics (DTW, embedding-based, statistical)
//! - Clustering and pattern discovery
//! - Efficient similarity search with indexing
//!
//! ## Example
//! ```rust,no_run
//! use chronos::ml::embeddings::{EmbeddingConfig, EmbeddingType, create_time_series_embeddings};
//! use chronos::TimeSeries;
//!
//! let config = EmbeddingConfig {
//!     embedding_type: EmbeddingType::Autoencoder {
//!         hidden_layers: vec![64, 32],
//!         latent_dim: 16,
//!     },
//!     dimension: 16,
//!     window_size: Some(10),
//!     ..Default::default()
//! };
//!
//! let result = create_time_series_embeddings(&time_series_data, &config)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::ml::{MLResult, Device, TrainingConfig, TrainingHistory};
use crate::timeseries::TimeSeries;

/// Type of embedding architecture
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmbeddingType {
    /// Autoencoder with specified hidden layers and latent dimension
    Autoencoder {
        hidden_layers: Vec<usize>,
        latent_dim: usize,
    },
    /// Variational Autoencoder for probabilistic embeddings
    VariationalAutoencoder {
        latent_dim: usize,
        beta: f64, // β-VAE parameter for disentanglement
    },
    /// Contrastive learning with SimCLR-style architecture
    ContrastiveLearning {
        temperature: f64,
        negative_samples: usize,
    },
    /// Transformer-based embedding
    Transformer {
        model_dim: usize,
        num_heads: usize,
        num_layers: usize,
    },
    /// LSTM-based embedding
    LSTM {
        hidden_size: usize,
        num_layers: usize,
    },
    /// WaveNet-style convolutional architecture
    WaveNet {
        dilation_channels: usize,
        residual_channels: usize,
    },
    /// Time series specific VAE
    TimeSeriesVAE {
        encoder_layers: Vec<usize>,
        decoder_layers: Vec<usize>,
    },
}

impl Default for EmbeddingType {
    fn default() -> Self {
        Self::Autoencoder {
            hidden_layers: vec![64, 32],
            latent_dim: 16,
        }
    }
}

/// Normalization method for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Robust scaling using median and IQR
    Robust,
    /// L2 normalization
    L2,
    /// No normalization
    None,
}

impl Default for NormalizationType {
    fn default() -> Self {
        Self::ZScore
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    /// Include statistical features
    pub statistical: bool,
    /// Include frequency domain features
    pub frequency: bool,
    /// Include temporal features
    pub temporal: bool,
    /// Include shape-based features
    pub shape: bool,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            statistical: true,
            frequency: true,
            temporal: true,
            shape: true,
        }
    }
}

/// Configuration for time series embeddings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Type of embedding architecture
    pub embedding_type: EmbeddingType,
    /// Dimension of the embedding space
    pub dimension: usize,
    /// Window size for sliding window approach
    pub window_size: Option<usize>,
    /// Overlap ratio for windows (0.0 to 1.0)
    pub overlap: f64,
    /// Normalization method
    pub normalization: NormalizationType,
    /// Feature extraction configuration
    pub feature_extraction: FeatureExtractionConfig,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Device for computation
    pub device: Device,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_type: EmbeddingType::default(),
            dimension: 16,
            window_size: Some(10),
            overlap: 0.5,
            normalization: NormalizationType::default(),
            feature_extraction: FeatureExtractionConfig::default(),
            training_config: TrainingConfig::default(),
            device: Device::Auto,
        }
    }
}

/// Training configuration for embedding models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingTrainingConfig {
    /// Base training configuration
    pub base_config: TrainingConfig,
    /// Reconstruction weight for autoencoders
    pub reconstruction_weight: f64,
    /// Regularization weight
    pub regularization_weight: f64,
    /// KL divergence weight for VAEs
    pub kl_weight: f64,
    /// Contrastive loss weight
    pub contrastive_weight: f64,
}

impl Default for EmbeddingTrainingConfig {
    fn default() -> Self {
        Self {
            base_config: TrainingConfig::default(),
            reconstruction_weight: 1.0,
            regularization_weight: 0.01,
            kl_weight: 0.1,
            contrastive_weight: 1.0,
        }
    }
}

/// Metadata for embedding results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Number of time series embedded
    pub num_series: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Model type used
    pub model_type: String,
    /// Training time in seconds
    pub training_time_secs: f64,
    /// Average reconstruction error (if applicable)
    pub avg_reconstruction_error: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl EmbeddingMetadata {
    /// Create new embedding metadata
    pub fn new(num_series: usize, embedding_dim: usize, model_type: String) -> Self {
        Self {
            num_series,
            embedding_dim,
            model_type,
            training_time_secs: 0.0,
            avg_reconstruction_error: None,
            metadata: HashMap::new(),
        }
    }
}

/// Result of embedding operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// Embeddings for each time series
    pub embeddings: Vec<Vec<f64>>,
    /// IDs of the time series
    pub time_series_ids: Vec<String>,
    /// Metadata about the embedding process
    pub metadata: EmbeddingMetadata,
    /// Training history if available
    pub training_history: Option<TrainingHistory>,
}

impl EmbeddingResult {
    /// Create new embedding result
    pub fn new(
        embeddings: Vec<Vec<f64>>,
        time_series_ids: Vec<String>,
        metadata: EmbeddingMetadata,
    ) -> Self {
        Self {
            embeddings,
            time_series_ids,
            metadata,
            training_history: None,
        }
    }

    /// Get embedding for specific time series by ID
    pub fn get_embedding(&self, id: &str) -> Option<&Vec<f64>> {
        self.time_series_ids
            .iter()
            .position(|x| x == id)
            .and_then(|idx| self.embeddings.get(idx))
    }
}

/// Similarity computation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimilarityMethod {
    // Distance-based methods
    Euclidean,
    Manhattan,
    Cosine,
    Mahalanobis {
        covariance_matrix: Vec<Vec<f64>>,
    },

    // Time series specific
    DynamicTimeWarping {
        window: Option<usize>,
    },
    SoftDTW {
        gamma: f64,
    },
    LCSS {
        epsilon: f64,
        delta: usize,
    }, // Longest Common Subsequence
    EDR {
        epsilon: f64,
    }, // Edit Distance on Real sequences

    // Statistical
    CrossCorrelation {
        max_lag: usize,
    },
    MutualInformation,
    TransferEntropy,

    // Frequency domain
    SpectralSimilarity {
        method: SpectralMethod,
    },
    WaveletSimilarity {
        wavelet_type: WaveletType,
    },
}

impl Default for SimilarityMethod {
    fn default() -> Self {
        Self::Euclidean
    }
}

/// Spectral similarity methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpectralMethod {
    /// Power spectral density comparison
    PowerSpectrum,
    /// Cross-spectral density
    CrossSpectrum,
    /// Coherence function
    Coherence,
}

/// Wavelet types for similarity
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WaveletType {
    Haar,
    Daubechies4,
    Symlet8,
    Coiflet5,
}

/// Alignment information for similarity results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentInfo {
    /// Optimal alignment path
    pub alignment_path: Vec<(usize, usize)>,
    /// Alignment cost
    pub cost: f64,
    /// Warping distance
    pub warping_distance: f64,
}

/// Result of similarity computation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// The time series
    pub time_series: TimeSeries,
    /// Similarity score (higher = more similar, 0-1)
    pub similarity_score: f64,
    /// Distance measure (lower = more similar)
    pub distance: f64,
    /// Alignment information if applicable
    pub alignment_info: Option<AlignmentInfo>,
}

impl SimilarityResult {
    /// Create new similarity result
    pub fn new(time_series: TimeSeries, similarity_score: f64, distance: f64) -> Self {
        Self {
            time_series,
            similarity_score,
            distance,
            alignment_info: None,
        }
    }

    /// Add alignment information
    pub fn with_alignment(mut self, alignment_info: AlignmentInfo) -> Self {
        self.alignment_info = Some(alignment_info);
        self
    }
}

/// Trait for time series encoder
pub trait TimeSeriesEncoder: Send + Sync {
    /// Encode a time series into an embedding
    fn encode(&self, time_series: &TimeSeries) -> MLResult<Vec<f64>>;

    /// Decode an embedding back to a time series (if applicable)
    fn decode(&self, embedding: &[f64]) -> MLResult<TimeSeries>;

    /// Get the embedding dimension
    fn embedding_dim(&self) -> usize;
}

/// Embedding model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingModel {
    /// Type of embedding
    pub model_type: EmbeddingType,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Normalization type
    pub normalization: NormalizationType,
    /// Training configuration
    pub training_config: EmbeddingTrainingConfig,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl EmbeddingModel {
    /// Create new embedding model
    pub fn new(model_type: EmbeddingType, embedding_dimension: usize) -> Self {
        Self {
            model_type,
            embedding_dimension,
            normalization: NormalizationType::default(),
            training_config: EmbeddingTrainingConfig::default(),
            metadata: HashMap::new(),
        }
    }

    /// Set normalization type
    pub fn with_normalization(mut self, normalization: NormalizationType) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set training configuration
    pub fn with_training_config(mut self, training_config: EmbeddingTrainingConfig) -> Self {
        self.training_config = training_config;
        self
    }
}

// Core embedding functions

/// Create time series embeddings
pub fn create_time_series_embeddings(
    _data: &[TimeSeries],
    _config: &EmbeddingConfig,
) -> MLResult<EmbeddingResult> {
    // TODO: Implement embedding creation
    unimplemented!("Embedding creation not yet implemented")
}

/// Compute similarity between two time series
pub fn compute_time_series_similarity(
    _ts1: &TimeSeries,
    _ts2: &TimeSeries,
    _method: &SimilarityMethod,
) -> MLResult<f64> {
    // TODO: Implement similarity computation
    unimplemented!("Similarity computation not yet implemented")
}

/// Find similar time series using embeddings
pub fn find_similar_time_series(
    _query: &TimeSeries,
    _database: &[TimeSeries],
    _embedding_model: &EmbeddingModel,
    _k: usize,
) -> MLResult<Vec<SimilarityResult>> {
    // TODO: Implement similarity search
    unimplemented!("Similarity search not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.dimension, 16);
        assert!(config.window_size.is_some());
        assert_eq!(config.overlap, 0.5);
    }

    #[test]
    fn test_embedding_type_default() {
        let embedding_type = EmbeddingType::default();
        match embedding_type {
            EmbeddingType::Autoencoder { latent_dim, .. } => {
                assert_eq!(latent_dim, 16);
            }
            _ => panic!("Expected Autoencoder as default"),
        }
    }

    #[test]
    fn test_normalization_type() {
        let norm = NormalizationType::default();
        assert_eq!(norm, NormalizationType::ZScore);
    }

    #[test]
    fn test_embedding_metadata_creation() {
        let metadata = EmbeddingMetadata::new(10, 16, "Autoencoder".to_string());
        assert_eq!(metadata.num_series, 10);
        assert_eq!(metadata.embedding_dim, 16);
        assert_eq!(metadata.model_type, "Autoencoder");
    }

    #[test]
    fn test_similarity_result_creation() {
        let ts = TimeSeries::empty("test".to_string());
        let result = SimilarityResult::new(ts, 0.95, 0.05);
        assert_eq!(result.similarity_score, 0.95);
        assert_eq!(result.distance, 0.05);
        assert!(result.alignment_info.is_none());
    }
}
