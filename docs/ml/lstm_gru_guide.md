# LSTM and GRU Networks Guide

Comprehensive guide to using LSTM and GRU recurrent neural networks for time series forecasting in Chronos.

## Overview

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are recurrent neural network architectures designed to capture temporal dependencies in sequential data. They excel at learning patterns in time series data for forecasting tasks.

### When to Use Each

**LSTM (Long Short-Term Memory)**:
- Best for standard time series with moderate sequence lengths
- Better at capturing long-term dependencies
- More parameters = better capacity but slower training
- Ideal for complex patterns with significant historical context

**GRU (Gated Recurrent Unit)**:
- Faster training due to fewer parameters
- Good for real-time applications requiring quick inference
- Better for simpler patterns or shorter sequences
- More memory-efficient than LSTM

## Quick Start

### LSTM Forecasting

```rust
use chronos::ml::{LSTMConfig, create_lstm_forecaster, forecast_with_lstm};
use chronos::TimeSeries;

// Prepare your time series data
let time_series = TimeSeries::new(
    "sensor_data".to_string(),
    timestamps,
    values
)?;

// Configure LSTM
let config = LSTMConfig {
    hidden_size: 64,
    num_layers: 2,
    sequence_length: 10,
    dropout_rate: 0.2,
    output_size: 1,
    ..Default::default()
};

// Train forecaster
let forecaster = create_lstm_forecaster(&config, &time_series)?;

// Make predictions
let forecast = forecast_with_lstm(&forecaster, &input_sequence, 5)?;
```

### GRU Forecasting

```rust
use chronos::ml::{GRUConfig, create_gru_forecaster, forecast_with_gru};

// Configure GRU
let config = GRUConfig {
    hidden_size: 64,
    num_layers: 2,
    sequence_length: 10,
    output_size: 1,
    ..Default::default()
};

// Train and predict
let forecaster = create_gru_forecaster(&config, &time_series)?;
let forecast = forecast_with_gru(&forecaster, &input_sequence, 5)?;
```

## Configuration Guide

### LSTM Configuration

```rust
pub struct LSTMConfig {
    // Network architecture
    pub hidden_size: usize,      // Size of hidden state (32-256)
    pub num_layers: usize,        // Number of LSTM layers (1-4)
    pub output_size: usize,       // Output dimension (usually 1)

    // Training parameters
    pub sequence_length: usize,   // Input sequence length (5-100)
    pub dropout_rate: f64,        // Dropout for regularization (0.0-0.5)
    pub learning_rate: f64,       // Learning rate (0.0001-0.01)
    pub batch_size: usize,        // Batch size (16-128)
    pub epochs: usize,            // Training epochs (50-500)

    // Data handling
    pub train_ratio: f64,         // Training data split (0.6-0.8)
    pub val_ratio: f64,           // Validation data split (0.1-0.2)

    // Device
    pub device: Device,           // CPU, CUDA, Metal, Auto
}
```

### GRU Configuration

```rust
pub struct GRUConfig {
    // Same parameters as LSTM
    // GRU has fewer parameters internally but same configuration interface
}
```

### Parameter Tuning Guidelines

**Hidden Size** (`hidden_size`):
- Start with 32-64 for simple patterns
- Use 64-128 for moderate complexity
- Use 128-256 for complex patterns
- Larger = more capacity but slower training

**Number of Layers** (`num_layers`):
- Start with 1-2 layers
- Add more layers for complex hierarchical patterns
- Diminishing returns after 3-4 layers
- More layers = risk of overfitting

**Sequence Length** (`sequence_length`):
- Match to your data's periodicity (e.g., 24 for hourly data with daily patterns)
- Longer sequences capture more context but require more memory
- Typical range: 10-50 for most applications

**Dropout Rate** (`dropout_rate`):
- Start with 0.1-0.2
- Increase to 0.3-0.5 if overfitting
- Set to 0.0 for small datasets
- Applied between LSTM/GRU layers

**Learning Rate** (`learning_rate`):
- Default: 0.001 works well for most cases
- Decrease to 0.0001 if training is unstable
- Increase to 0.01 for faster convergence (risk of instability)
- Use learning rate scheduling for best results

**Batch Size** (`batch_size`):
- Smaller (16-32): Better generalization, slower training
- Larger (64-128): Faster training, requires more memory
- Adjust based on dataset size and available memory

**Epochs** (`epochs`):
- Start with 100 epochs
- Monitor validation loss to prevent overfitting
- Use early stopping for automatic epoch selection
- More epochs needed for complex patterns

## Dataset Preparation

### Creating a Time Series Dataset

```rust
use chronos::ml::TimeSeriesDataset;

let dataset = TimeSeriesDataset::from_timeseries(
    &time_series,
    sequence_length: 10,    // Look back 10 time steps
    horizon: 1,             // Predict 1 step ahead
    train_ratio: 0.7,       // 70% training
    val_ratio: 0.15,        // 15% validation, 15% test
)?;

// Access splits
let train_data = dataset.train_sequences();
let val_data = dataset.val_sequences();
let test_data = dataset.test_sequences();
```

### Normalization

Normalization is applied automatically during training:

```rust
// Z-score normalization (default)
let forecaster = create_lstm_forecaster(&config, &time_series)?;

// The forecaster stores normalization parameters
// Predictions are automatically denormalized
let forecast = forecaster.forecast(&input)?; // Already in original scale
```

Manual normalization control:

```rust
use chronos::ml::{NormalizationParams, NormalizationMethod};

// Create custom normalization
let norm_params = NormalizationParams::from_data(
    &data,
    NormalizationMethod::ZScore
)?;

// Apply normalization
let normalized = norm_params.normalize(&data);

// Denormalize predictions
let original_scale = norm_params.denormalize(&predictions);
```

## Training Process

### Basic Training

```rust
use chronos::ml::LSTMForecaster;

// Create forecaster
let mut forecaster = LSTMForecaster::new(&config, Device::Auto)?;

// Train on dataset
forecaster.train(&dataset)?;

// Training automatically:
// 1. Normalizes data
// 2. Creates batches
// 3. Trains for specified epochs
// 4. Monitors validation loss
// 5. Stores best model
```

### Advanced Training with Monitoring

```rust
use chronos::ml::TrainingConfig;

let training_config = TrainingConfig {
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100,
    validation_split: 0.2,
    early_stopping: Some(EarlyStoppingConfig {
        patience: 10,           // Stop if no improvement for 10 epochs
        min_delta: 0.001,       // Minimum improvement threshold
    }),
    ..Default::default()
};

let forecaster = LSTMForecaster::with_training_config(
    &config,
    &training_config,
    Device::Auto
)?;

forecaster.train(&dataset)?;

// Access training history
let history = forecaster.training_history();
println!("Best epoch: {}", history.best_epoch);
println!("Final loss: {}", history.final_loss);
```

## Making Predictions

### Single-Step Prediction

```rust
// Predict next value
let input_sequence = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let prediction = forecaster.forecast(&input_sequence)?;

println!("Next value: {}", prediction[0]);
```

### Multi-Step Prediction

```rust
// Predict 5 steps ahead
let forecast = forecast_with_lstm(&forecaster, &input_sequence, 5)?;

for (i, value) in forecast.iter().enumerate() {
    println!("Step {}: {}", i + 1, value);
}
```

### Rolling Forecast

```rust
// Predict with rolling window
fn rolling_forecast(
    forecaster: &LSTMForecaster,
    initial_sequence: &[f64],
    steps: usize,
) -> Vec<f64> {
    let mut predictions = Vec::new();
    let mut current_sequence = initial_sequence.to_vec();

    for _ in 0..steps {
        let pred = forecaster.forecast(&current_sequence).unwrap();
        predictions.push(pred[0]);

        // Update sequence for next prediction
        current_sequence.remove(0);
        current_sequence.push(pred[0]);
    }

    predictions
}

let forecast = rolling_forecast(&forecaster, &input_sequence, 10);
```

## Model Persistence

### Saving Models

```rust
use chronos::ml::{save_model, ModelFormat};

// Save in binary format (fastest)
save_model(&forecaster.network, "lstm_model.bin", ModelFormat::Binary)?;

// Save in JSON format (human-readable)
save_model(&forecaster.network, "lstm_model.json", ModelFormat::JSON)?;
```

### Loading Models

```rust
use chronos::ml::load_model;

// Load saved model
let loaded_network = load_model("lstm_model.bin", ModelFormat::Binary)?;

// Create forecaster with loaded network
let mut forecaster = LSTMForecaster::new(&config, Device::Auto)?;
forecaster.network = loaded_network;
```

## GPU Acceleration

### Device Selection

```rust
use chronos::ml::Device;

// Automatic device selection (prefers GPU)
let forecaster = LSTMForecaster::new(&config, Device::Auto)?;

// Explicit CUDA GPU
let forecaster = LSTMForecaster::new(&config, Device::CUDA(0))?;

// Explicit Metal (macOS)
let forecaster = LSTMForecaster::new(&config, Device::Metal)?;

// CPU only
let forecaster = LSTMForecaster::new(&config, Device::CPU)?;
```

### Performance Tips

- GPU provides 5-10x speedup for training
- Larger batch sizes benefit more from GPU
- CPU is sufficient for small datasets (<1000 samples)
- Use GPU for sequence_length > 20 or hidden_size > 64

## Best Practices

### 1. Data Preparation

```rust
// Ensure sufficient data
assert!(time_series.len() > sequence_length * 20);

// Handle missing values before training
let cleaned = time_series.fill_missing(InterpolationMethod::Linear)?;

// Remove outliers if necessary
let outliers = detect_outliers(&cleaned)?;
let filtered = remove_outliers(&cleaned, &outliers)?;
```

### 2. Hyperparameter Selection

```rust
// Start with sensible defaults
let base_config = LSTMConfig {
    hidden_size: 64,
    num_layers: 2,
    sequence_length: 20,
    dropout_rate: 0.2,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100,
    ..Default::default()
};

// Adjust based on validation performance
```

### 3. Preventing Overfitting

```rust
// Use dropout
config.dropout_rate = 0.2;

// Use early stopping
config.early_stopping = Some(EarlyStoppingConfig {
    patience: 10,
    min_delta: 0.001,
});

// Increase validation split
config.val_ratio = 0.2;

// Reduce model complexity
config.hidden_size = 32;
config.num_layers = 1;
```

### 4. Handling Long Sequences

```rust
// For sequences > 100 time steps
config.hidden_size = 128;  // Increase capacity
config.num_layers = 3;     // Add depth
config.batch_size = 16;    // Reduce for memory

// Consider using Transformer for very long sequences
```

## Common Patterns

### Stock Price Forecasting

```rust
let config = LSTMConfig {
    hidden_size: 128,
    num_layers: 2,
    sequence_length: 30,  // 30 days of history
    dropout_rate: 0.3,    // Higher dropout for noisy data
    learning_rate: 0.0001,
    batch_size: 64,
    epochs: 200,
    ..Default::default()
};
```

### IoT Sensor Data

```rust
let config = GRUConfig {
    hidden_size: 32,      // Simpler patterns
    num_layers: 1,
    sequence_length: 24,  // 24 hours for daily patterns
    dropout_rate: 0.1,
    learning_rate: 0.001,
    batch_size: 128,      // Larger batches for efficiency
    epochs: 100,
    ..Default::default()
};
```

### Energy Consumption

```rust
let config = LSTMConfig {
    hidden_size: 64,
    num_layers: 2,
    sequence_length: 168, // 1 week of hourly data
    dropout_rate: 0.2,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 150,
    ..Default::default()
};
```

## Troubleshooting

### Training Loss Not Decreasing

**Solutions**:
- Increase learning rate to 0.01
- Reduce model complexity (smaller hidden_size)
- Check data normalization
- Ensure sufficient training data

### Overfitting (Train loss << Validation loss)

**Solutions**:
- Increase dropout_rate to 0.3-0.5
- Use early stopping
- Reduce hidden_size or num_layers
- Get more training data

### Slow Training

**Solutions**:
- Use GPU (Device::Auto or Device::CUDA(0))
- Increase batch_size to 64-128
- Use GRU instead of LSTM
- Reduce sequence_length

### Out of Memory

**Solutions**:
- Reduce batch_size to 16 or 8
- Reduce sequence_length
- Use GRU instead of LSTM
- Use CPU instead of GPU

### Poor Forecast Accuracy

**Solutions**:
- Increase sequence_length to capture more context
- Add more layers (num_layers = 3-4)
- Increase hidden_size for more capacity
- Train for more epochs
- Check data quality and preprocessing

## API Reference

### Core Types

```rust
pub struct LSTMConfig { /* ... */ }
pub struct GRUConfig { /* ... */ }
pub struct LSTMForecaster { /* ... */ }
pub struct GRUForecaster { /* ... */ }
pub struct TimeSeriesDataset { /* ... */ }
pub struct NormalizationParams { /* ... */ }
```

### Functions

```rust
// Training and creation
pub fn create_lstm_forecaster(config: &LSTMConfig, data: &TimeSeries) -> MLResult<LSTMForecaster>
pub fn create_gru_forecaster(config: &GRUConfig, data: &TimeSeries) -> MLResult<GRUForecaster>

// Forecasting
pub fn forecast_with_lstm(forecaster: &LSTMForecaster, input: &[f64], steps: usize) -> MLResult<Vec<f64>>
pub fn forecast_with_gru(forecaster: &GRUForecaster, input: &[f64], steps: usize) -> MLResult<Vec<f64>>
```

## Examples

See the `examples/ml/` directory for complete examples:
- `examples/ml/stock_forecasting.rs` - Stock price prediction with LSTM
- `examples/ml/sensor_forecasting.rs` - IoT sensor prediction with GRU
- `examples/ml/energy_forecasting.rs` - Energy consumption forecasting

## Further Reading

- [Transformer Guide](transformer_guide.md) - For longer sequences and interpretability
- [GPU Acceleration Guide](gpu_acceleration.md) - Detailed GPU setup
- [Model Persistence Guide](persistence.md) - Advanced model management
