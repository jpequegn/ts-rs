//! Model persistence and serialization

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use crate::ml::{MLError, MLResult};
use crate::ml::types::{NeuralNetwork, ModelMetadata, TrainingHistory};

/// Model serialization format
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// JSON format (human-readable)
    JSON,
    /// Binary format (compact)
    Binary,
    /// MessagePack format (compact, fast)
    MessagePack,
}

impl Default for ModelFormat {
    fn default() -> Self {
        Self::Binary
    }
}

/// Model version for compatibility tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Get version as string
    pub fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    /// Check if compatible with another version
    pub fn is_compatible(&self, other: &ModelVersion) -> bool {
        // Compatible if major version matches
        self.major == other.major
    }
}

impl Default for ModelVersion {
    fn default() -> Self {
        Self::new(1, 0, 0)
    }
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Model checkpoint for saving training progress
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    /// Model architecture
    pub model: NeuralNetwork,
    /// Model version
    pub version: ModelVersion,
    /// Checkpoint epoch
    pub epoch: usize,
    /// Checkpoint timestamp
    pub timestamp: String,
    /// Validation loss at checkpoint
    pub val_loss: Option<f64>,
    /// Training history up to this checkpoint
    pub training_history: Option<TrainingHistory>,
}

impl ModelCheckpoint {
    /// Create a new checkpoint
    pub fn new(model: NeuralNetwork, epoch: usize) -> Self {
        Self {
            model,
            version: ModelVersion::default(),
            epoch,
            timestamp: chrono::Utc::now().to_rfc3339(),
            val_loss: None,
            training_history: None,
        }
    }

    /// Set validation loss
    pub fn with_val_loss(mut self, val_loss: f64) -> Self {
        self.val_loss = Some(val_loss);
        self
    }

    /// Set training history
    pub fn with_history(mut self, history: TrainingHistory) -> Self {
        self.training_history = Some(history);
        self
    }
}

/// Serialized model container
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model architecture
    pub model: NeuralNetwork,
    /// Model version
    pub version: ModelVersion,
    /// Serialization format
    pub format: ModelFormat,
    /// Serialization timestamp
    pub timestamp: String,
    /// Model weights (placeholder - actual implementation would include tensor data)
    pub weights: Vec<u8>,
}

impl SerializedModel {
    /// Create a new serialized model
    pub fn new(model: NeuralNetwork, format: ModelFormat) -> Self {
        Self {
            model,
            version: ModelVersion::default(),
            format,
            timestamp: chrono::Utc::now().to_rfc3339(),
            weights: Vec::new(), // Placeholder
        }
    }
}

/// Save model to file
pub fn save_model<P: AsRef<Path>>(
    model: &NeuralNetwork,
    path: P,
    format: ModelFormat,
) -> MLResult<()> {
    let serialized = SerializedModel::new(model.clone(), format);

    match format {
        ModelFormat::JSON => {
            let json = serde_json::to_string_pretty(&serialized)
                .map_err(|e| MLError::serialization(format!("JSON serialization failed: {}", e)))?;

            let mut file = File::create(path.as_ref())
                .map_err(|e| MLError::serialization(format!("Failed to create file: {}", e)))?;

            file.write_all(json.as_bytes())
                .map_err(|e| MLError::serialization(format!("Failed to write file: {}", e)))?;
        }
        ModelFormat::Binary => {
            let binary = bincode::serialize(&serialized)
                .map_err(|e| MLError::serialization(format!("Binary serialization failed: {}", e)))?;

            let mut file = File::create(path.as_ref())
                .map_err(|e| MLError::serialization(format!("Failed to create file: {}", e)))?;

            file.write_all(&binary)
                .map_err(|e| MLError::serialization(format!("Failed to write file: {}", e)))?;
        }
        ModelFormat::MessagePack => {
            // Placeholder for MessagePack support
            return Err(MLError::serialization("MessagePack format not yet implemented"));
        }
    }

    Ok(())
}

/// Load model from file
pub fn load_model<P: AsRef<Path>>(path: P, format: ModelFormat) -> MLResult<NeuralNetwork> {
    let mut file = File::open(path.as_ref())
        .map_err(|e| MLError::serialization(format!("Failed to open file: {}", e)))?;

    let serialized = match format {
        ModelFormat::JSON => {
            let mut json = String::new();
            file.read_to_string(&mut json)
                .map_err(|e| MLError::serialization(format!("Failed to read file: {}", e)))?;

            serde_json::from_str::<SerializedModel>(&json)
                .map_err(|e| MLError::serialization(format!("JSON deserialization failed: {}", e)))?
        }
        ModelFormat::Binary => {
            let mut binary = Vec::new();
            file.read_to_end(&mut binary)
                .map_err(|e| MLError::serialization(format!("Failed to read file: {}", e)))?;

            bincode::deserialize::<SerializedModel>(&binary)
                .map_err(|e| MLError::serialization(format!("Binary deserialization failed: {}", e)))?
        }
        ModelFormat::MessagePack => {
            return Err(MLError::serialization("MessagePack format not yet implemented"));
        }
    };

    // Verify version compatibility
    let current_version = ModelVersion::default();
    if !current_version.is_compatible(&serialized.version) {
        return Err(MLError::serialization(format!(
            "Incompatible model version: {} (expected {})",
            serialized.version, current_version
        )));
    }

    Ok(serialized.model)
}

/// Save model checkpoint
pub fn save_checkpoint<P: AsRef<Path>>(
    checkpoint: &ModelCheckpoint,
    path: P,
    format: ModelFormat,
) -> MLResult<()> {
    match format {
        ModelFormat::JSON => {
            let json = serde_json::to_string_pretty(&checkpoint)
                .map_err(|e| MLError::serialization(format!("JSON serialization failed: {}", e)))?;

            let mut file = File::create(path.as_ref())
                .map_err(|e| MLError::serialization(format!("Failed to create file: {}", e)))?;

            file.write_all(json.as_bytes())
                .map_err(|e| MLError::serialization(format!("Failed to write file: {}", e)))?;
        }
        ModelFormat::Binary => {
            let binary = bincode::serialize(&checkpoint)
                .map_err(|e| MLError::serialization(format!("Binary serialization failed: {}", e)))?;

            let mut file = File::create(path.as_ref())
                .map_err(|e| MLError::serialization(format!("Failed to create file: {}", e)))?;

            file.write_all(&binary)
                .map_err(|e| MLError::serialization(format!("Failed to write file: {}", e)))?;
        }
        ModelFormat::MessagePack => {
            return Err(MLError::serialization("MessagePack format not yet implemented"));
        }
    }

    Ok(())
}

/// Load model checkpoint
pub fn load_checkpoint<P: AsRef<Path>>(
    path: P,
    format: ModelFormat,
) -> MLResult<ModelCheckpoint> {
    let mut file = File::open(path.as_ref())
        .map_err(|e| MLError::serialization(format!("Failed to open file: {}", e)))?;

    let checkpoint = match format {
        ModelFormat::JSON => {
            let mut json = String::new();
            file.read_to_string(&mut json)
                .map_err(|e| MLError::serialization(format!("Failed to read file: {}", e)))?;

            serde_json::from_str::<ModelCheckpoint>(&json)
                .map_err(|e| MLError::serialization(format!("JSON deserialization failed: {}", e)))?
        }
        ModelFormat::Binary => {
            let mut binary = Vec::new();
            file.read_to_end(&mut binary)
                .map_err(|e| MLError::serialization(format!("Failed to read file: {}", e)))?;

            bincode::deserialize::<ModelCheckpoint>(&binary)
                .map_err(|e| MLError::serialization(format!("Binary deserialization failed: {}", e)))?
        }
        ModelFormat::MessagePack => {
            return Err(MLError::serialization("MessagePack format not yet implemented"));
        }
    };

    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::types::{Layer, OptimizerType, LossFunction, ActivationType};
    use tempfile::NamedTempFile;

    fn create_test_model() -> NeuralNetwork {
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

        NeuralNetwork::new(layers, OptimizerType::default(), LossFunction::MSE)
    }

    #[test]
    fn test_model_version() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);
        let v3 = ModelVersion::new(2, 0, 0);

        assert!(v1.is_compatible(&v2));
        assert!(!v1.is_compatible(&v3));
        assert_eq!(v1.to_string(), "1.0.0");
    }

    #[test]
    fn test_save_load_json() {
        let model = create_test_model();
        let temp_file = NamedTempFile::new().unwrap();

        // Save model
        save_model(&model, temp_file.path(), ModelFormat::JSON).unwrap();

        // Load model
        let loaded = load_model(temp_file.path(), ModelFormat::JSON).unwrap();

        assert_eq!(loaded.layers.len(), model.layers.len());
        assert_eq!(loaded.loss_function, model.loss_function);
    }

    #[test]
    fn test_save_load_binary() {
        let model = create_test_model();
        let temp_file = NamedTempFile::new().unwrap();

        // Save model
        save_model(&model, temp_file.path(), ModelFormat::Binary).unwrap();

        // Load model
        let loaded = load_model(temp_file.path(), ModelFormat::Binary).unwrap();

        assert_eq!(loaded.layers.len(), model.layers.len());
        assert_eq!(loaded.loss_function, model.loss_function);
    }

    #[test]
    fn test_checkpoint() {
        let model = create_test_model();
        let checkpoint = ModelCheckpoint::new(model.clone(), 10)
            .with_val_loss(0.5);

        assert_eq!(checkpoint.epoch, 10);
        assert_eq!(checkpoint.val_loss, Some(0.5));
    }

    #[test]
    fn test_save_load_checkpoint() {
        let model = create_test_model();
        let checkpoint = ModelCheckpoint::new(model, 5)
            .with_val_loss(0.3);

        let temp_file = NamedTempFile::new().unwrap();

        // Save checkpoint
        save_checkpoint(&checkpoint, temp_file.path(), ModelFormat::Binary).unwrap();

        // Load checkpoint
        let loaded = load_checkpoint(temp_file.path(), ModelFormat::Binary).unwrap();

        assert_eq!(loaded.epoch, checkpoint.epoch);
        assert_eq!(loaded.val_loss, checkpoint.val_loss);
    }
}
