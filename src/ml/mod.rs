//! # Machine Learning Module
//!
//! Comprehensive machine learning integration for time series analysis, providing:
//! - Neural network architectures (LSTM, GRU, Transformers)
//! - Tensor operations with GPU acceleration
//! - Model persistence and serialization
//! - Integration with existing time series analysis modules

pub mod types;
pub mod tensor;
pub mod activations;
pub mod persistence;
pub mod recurrent;
pub mod transformer;
pub mod ensemble;
pub mod automl;
pub mod interfaces;
pub mod embeddings;

// Re-export commonly used types
pub use types::{
    NeuralNetwork, Layer, TrainingConfig, Device, DevicePreference,
    OptimizerType, LossFunction, EarlyStoppingConfig, ModelMetadata,
    TrainingHistory, EpochMetrics,
};

pub use tensor::{
    Tensor, TensorOps, Shape, DataType, GpuBackend,
};

pub use activations::{
    ActivationFunction, apply_activation,
    relu, sigmoid, tanh, softmax, swish, gelu, leaky_relu, elu,
};

pub use persistence::{
    ModelFormat, SerializedModel, save_model, load_model,
    ModelVersion, ModelCheckpoint,
};

pub use recurrent::{
    // LSTM types
    LSTMConfig, LSTMForecaster, create_lstm_forecaster, forecast_with_lstm,

    // GRU types
    GRUConfig, GRUForecaster, create_gru_forecaster, forecast_with_gru,

    // Dataset and utilities
    TimeSeriesDataset, DataSplit, RecurrentFeatureConfig,
    NormalizationParams, NormalizationMethod, TrainingStats,
};

pub use transformer::{
    // Transformer types
    TransformerConfig, TransformerForecaster, create_transformer_forecaster, forecast_with_transformer,

    // Attention and encoding
    PositionalEncoding, PositionalEncodingType, MultiHeadAttention,
    TransformerEncoder, TransformerEncoderLayer, AttentionAnalysis,
};

pub use ensemble::{
    // Core ensemble types
    EnsembleForecaster, NeuralModelWrapper, TraditionalModelWrapper,
    EnsembleConfig, CombinationStrategy, WeightOptimizationMethod,
    UncertaintyMethod,

    // Stacking ensemble
    StackingEnsemble, MetaFeatureConfig,

    // Dynamic weighting
    DynamicWeightManager, AdaptationMethod,

    // Diversity metrics
    DiversityMetrics,

    // Performance tracking
    PerformanceHistory, ModelPerformance,

    // Confidence weighting
    ConfidenceWeightedEnsemble, ConfidenceCalibration,

    // Results
    EnsembleForecastResult, ModelPrediction, EnsembleTrainingResult,

    // Core functions
    create_ensemble_forecaster, optimize_ensemble_weights,
};

pub use automl::{
    // Core types
    AutoMLConfig, OptimizationBudget, ComputeResources, SearchSpace,
    ParameterConfiguration, ParameterValue, OptimizedModel,
    PerformanceMetrics, OptimizationHistory,

    // Optimization methods
    OptimizationMethod, AcquisitionFunction,

    // NAS types
    ArchitectureSearchSpace, ArchitectureConstraints, OptimalArchitecture,
    NASStrategy, ArchitectureDescription,

    // Model selection
    ModelCandidate, SelectionCriteria, ModelRecommendation,

    // Multi-objective
    MultiObjectiveConfig, Objective, MOOMethod, ParetoFront, ParetoSolution,

    // Meta-learning
    DatasetFeatures, StatisticalFeatures, TemporalFeatures, ComplexityFeatures,

    // Early stopping
    EarlyStoppingConfig, EarlyStoppingMode,

    // Core functions
    auto_optimize_model, optimize_hyperparameters, search_neural_architecture,
    auto_configure_ensemble,
};

pub use interfaces::{ForecastingModel, ModelCapabilities};

pub use embeddings::{
    // Core types
    EmbeddingType, EmbeddingConfig, EmbeddingModel, EmbeddingResult,
    EmbeddingMetadata, EmbeddingTrainingConfig,

    // Normalization and features
    NormalizationType, FeatureExtractionConfig,

    // Similarity types
    SimilarityMethod, SimilarityResult, AlignmentInfo,
    SpectralMethod, WaveletType,

    // Encoder trait
    TimeSeriesEncoder,

    // Core functions
    create_time_series_embeddings, compute_time_series_similarity,
    find_similar_time_series,
};

/// Result type for ML operations
pub type MLResult<T> = Result<T, MLError>;

/// Error types for machine learning operations
#[derive(Debug, thiserror::Error)]
pub enum MLError {
    #[error("Tensor operation failed: {0}")]
    TensorOperation(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("GPU not available: {0}")]
    GpuNotAvailable(String),
}

impl MLError {
    pub fn tensor_operation(msg: impl Into<String>) -> Self {
        Self::TensorOperation(msg.into())
    }

    pub fn device(msg: impl Into<String>) -> Self {
        Self::Device(msg.into())
    }

    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }

    pub fn training(msg: impl Into<String>) -> Self {
        Self::Training(msg.into())
    }

    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    pub fn gpu_not_available(msg: impl Into<String>) -> Self {
        Self::GpuNotAvailable(msg.into())
    }
}
