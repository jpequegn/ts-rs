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
