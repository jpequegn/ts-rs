//! Core ML types and neural network structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Device type for computation (CPU or GPU)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Device {
    /// CPU computation
    CPU,
    /// CUDA GPU with device index
    CUDA(usize),
    /// Apple Silicon GPU (Metal)
    Metal,
    /// Automatic device selection (prefers GPU if available)
    Auto,
}

impl Default for Device {
    fn default() -> Self {
        Self::Auto
    }
}

impl Device {
    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::CUDA(_) | Device::Metal)
    }

    /// Get device name for display
    pub fn name(&self) -> String {
        match self {
            Device::CPU => "CPU".to_string(),
            Device::CUDA(idx) => format!("CUDA:{}", idx),
            Device::Metal => "Metal".to_string(),
            Device::Auto => "Auto".to_string(),
        }
    }
}

/// Device preference for automatic selection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DevicePreference {
    /// Prefer CPU even if GPU available
    PreferCPU,
    /// Prefer GPU, fallback to CPU
    PreferGPU,
    /// Require GPU, error if not available
    RequireGPU,
}

impl Default for DevicePreference {
    fn default() -> Self {
        Self::PreferGPU
    }
}

/// Neural network layer types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Layer {
    /// Dense/fully connected layer
    Dense {
        input_size: usize,
        output_size: usize,
        activation: ActivationType,
    },
    /// Long Short-Term Memory layer
    LSTM {
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    },
    /// Gated Recurrent Unit layer
    GRU {
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    },
    /// Transformer layer
    Transformer {
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
    },
    /// Dropout regularization
    Dropout {
        rate: f32,
    },
    /// Batch normalization
    BatchNorm {
        features: usize,
    },
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    GELU,
    LeakyReLU { alpha: f32 },
    ELU { alpha: f32 },
    Linear, // No activation
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::ReLU
    }
}

/// Optimizer types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD {
        learning_rate: f64,
        momentum: f64,
    },
    /// Adam optimizer
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// AdamW optimizer
    AdamW {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    },
    /// RMSprop optimizer
    RMSprop {
        learning_rate: f64,
        alpha: f64,
        epsilon: f64,
    },
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Loss function types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Huber loss
    Huber { delta: f64 },
    /// Cross Entropy
    CrossEntropy,
    /// Binary Cross Entropy
    BinaryCrossEntropy,
}

impl Default for LossFunction {
    fn default() -> Self {
        Self::MSE
    }
}

/// Early stopping configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Number of epochs with no improvement before stopping
    pub patience: usize,
    /// Minimum change to qualify as improvement
    pub min_delta: f64,
    /// Metric to monitor ('loss' or 'val_loss')
    pub monitor: String,
    /// Whether to restore best weights
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0001,
            monitor: "val_loss".to_string(),
            restore_best_weights: true,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f64,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Device preference
    pub device_preference: DevicePreference,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
            early_stopping: Some(EarlyStoppingConfig::default()),
            device_preference: DevicePreference::default(),
            shuffle: true,
            seed: None,
        }
    }
}

/// Metrics for a single epoch
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss (if validation set exists)
    pub val_loss: Option<f64>,
    /// Training time in seconds
    pub duration_secs: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Training history
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Metrics for each epoch
    pub epochs: Vec<EpochMetrics>,
    /// Best epoch number
    pub best_epoch: Option<usize>,
    /// Best validation loss
    pub best_val_loss: Option<f64>,
    /// Total training time in seconds
    pub total_duration_secs: f64,
}

impl TrainingHistory {
    /// Create new empty training history
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            best_epoch: None,
            best_val_loss: None,
            total_duration_secs: 0.0,
        }
    }

    /// Add epoch metrics
    pub fn add_epoch(&mut self, metrics: EpochMetrics) {
        // Update best metrics if applicable
        if let Some(val_loss) = metrics.val_loss {
            if self.best_val_loss.is_none() || val_loss < self.best_val_loss.unwrap() {
                self.best_val_loss = Some(val_loss);
                self.best_epoch = Some(metrics.epoch);
            }
        }

        self.total_duration_secs += metrics.duration_secs;
        self.epochs.push(metrics);
    }

    /// Get final training loss
    pub fn final_train_loss(&self) -> Option<f64> {
        self.epochs.last().map(|e| e.train_loss)
    }

    /// Get final validation loss
    pub fn final_val_loss(&self) -> Option<f64> {
        self.epochs.last().and_then(|e| e.val_loss)
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Model metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Creation timestamp
    pub created_at: String,
    /// Training configuration used
    pub training_config: TrainingConfig,
    /// Training history
    pub training_history: Option<TrainingHistory>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new model metadata
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            created_at: chrono::Utc::now().to_rfc3339(),
            training_config: TrainingConfig::default(),
            training_history: None,
            metadata: HashMap::new(),
        }
    }

    /// Add custom metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Neural network structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork {
    /// Network layers
    pub layers: Vec<Layer>,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Loss function
    pub loss_function: LossFunction,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Device for computation
    pub device: Device,
    /// Model metadata
    pub metadata: ModelMetadata,
}

impl NeuralNetwork {
    /// Create a new neural network
    pub fn new(
        layers: Vec<Layer>,
        optimizer: OptimizerType,
        loss_function: LossFunction,
    ) -> Self {
        Self {
            layers,
            optimizer,
            loss_function,
            training_config: TrainingConfig::default(),
            device: Device::default(),
            metadata: ModelMetadata::new("neural_network", "1.0.0"),
        }
    }

    /// Set training configuration
    pub fn with_training_config(mut self, config: TrainingConfig) -> Self {
        self.training_config = config;
        self
    }

    /// Set device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get total number of parameters (placeholder - actual implementation depends on layer weights)
    pub fn num_parameters(&self) -> usize {
        // This is a placeholder - actual implementation would calculate based on layer weights
        self.layers.iter().map(|layer| {
            match layer {
                Layer::Dense { input_size, output_size, .. } => input_size * output_size + output_size,
                Layer::LSTM { input_size, hidden_size, num_layers } => {
                    // Simplified LSTM parameter count
                    num_layers * (4 * hidden_size * (input_size + hidden_size + 1))
                },
                Layer::GRU { input_size, hidden_size, num_layers } => {
                    // Simplified GRU parameter count
                    num_layers * (3 * hidden_size * (input_size + hidden_size + 1))
                },
                Layer::Transformer { d_model, n_heads, d_ff, n_layers } => {
                    // Simplified transformer parameter count
                    n_layers * (
                        4 * d_model * d_model +  // Attention
                        2 * d_model * d_ff       // Feed-forward
                    )
                },
                Layer::BatchNorm { features } => features * 2,
                Layer::Dropout { .. } => 0,
            }
        }).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let cpu = Device::CPU;
        assert!(!cpu.is_gpu());
        assert_eq!(cpu.name(), "CPU");

        let cuda = Device::CUDA(0);
        assert!(cuda.is_gpu());
        assert_eq!(cuda.name(), "CUDA:0");

        let metal = Device::Metal;
        assert!(metal.is_gpu());
        assert_eq!(metal.name(), "Metal");
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.validation_split, 0.2);
        assert!(config.early_stopping.is_some());
    }

    #[test]
    fn test_neural_network_creation() {
        let layers = vec![
            Layer::Dense {
                input_size: 10,
                output_size: 20,
                activation: ActivationType::ReLU,
            },
            Layer::Dense {
                input_size: 20,
                output_size: 1,
                activation: ActivationType::Linear,
            },
        ];

        let nn = NeuralNetwork::new(
            layers,
            OptimizerType::default(),
            LossFunction::MSE,
        );

        assert_eq!(nn.num_layers(), 2);
        assert!(nn.num_parameters() > 0);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();

        history.add_epoch(EpochMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_loss: Some(1.2),
            duration_secs: 1.5,
            metrics: HashMap::new(),
        });

        history.add_epoch(EpochMetrics {
            epoch: 1,
            train_loss: 0.8,
            val_loss: Some(0.9),
            duration_secs: 1.4,
            metrics: HashMap::new(),
        });

        assert_eq!(history.epochs.len(), 2);
        assert_eq!(history.best_epoch, Some(1));
        assert_eq!(history.best_val_loss, Some(0.9));
        assert_eq!(history.final_train_loss(), Some(0.8));
    }

    #[test]
    fn test_layer_parameter_count() {
        let layer = Layer::Dense {
            input_size: 10,
            output_size: 5,
            activation: ActivationType::ReLU,
        };

        let nn = NeuralNetwork::new(
            vec![layer],
            OptimizerType::default(),
            LossFunction::MSE,
        );

        // Dense layer: 10 * 5 + 5 = 55 parameters
        assert_eq!(nn.num_parameters(), 55);
    }

    #[test]
    fn test_model_metadata() {
        let mut metadata = ModelMetadata::new("test_model", "1.0.0");
        metadata.add_metadata("author", "test");

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.metadata.get("author"), Some(&"test".to_string()));
    }
}
